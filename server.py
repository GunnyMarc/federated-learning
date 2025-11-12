"""
Federated Learning Server implementation.

This module implements the central server that coordinates federated learning training.
"""

from typing import List, Optional, Dict, Any, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm
import numpy as np

from algorithms import FederatedAlgorithm, FedAvg


class FederatedServer:
    """
    Central server for federated learning.
    
    The server orchestrates the federated learning process:
    1. Maintains the global model
    2. Selects clients for each round
    3. Distributes model to clients
    4. Aggregates client updates
    5. Evaluates global model
    
    Args:
        model (nn.Module): Initial global model
        algorithm (FederatedAlgorithm): Federated learning algorithm to use
        num_clients (int): Total number of clients
        device (str): Device to use ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        model: nn.Module,
        algorithm: FederatedAlgorithm,
        num_clients: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.global_model = model
        self.algorithm = algorithm
        self.num_clients = num_clients
        self.device = torch.device(device)
        self.global_model.to(self.device)
        
        self.common_loader: Optional[DataLoader] = None
        self.history: Dict[str, List] = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': [],
            'rounds': []
        }
    
    def set_common_dataset(self, common_loader: DataLoader):
        """
        Set a common dataset for fine-tuning (helps with non-IID data).
        
        Args:
            common_loader (DataLoader): DataLoader for common dataset
        """
        self.common_loader = common_loader
    
    def select_clients(
        self,
        num_selected: Optional[int] = None,
        selection_strategy: str = 'random'
    ) -> List[int]:
        """
        Select clients for the current round.
        
        Args:
            num_selected (int): Number of clients to select (None = all clients)
            selection_strategy (str): Strategy for selection ('random', 'all')
            
        Returns:
            List[int]: Indices of selected clients
        """
        if selection_strategy == 'all' or num_selected is None:
            return list(range(self.num_clients))
        
        elif selection_strategy == 'random':
            num_selected = min(num_selected, self.num_clients)
            return np.random.choice(
                self.num_clients, 
                size=num_selected, 
                replace=False
            ).tolist()
        
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    def train_round(
        self,
        train_loaders: List[DataLoader],
        criterion: nn.Module,
        learning_rate: float,
        selected_clients: Optional[List[int]] = None,
        local_epochs: int = 1
    ) -> float:
        """
        Execute one round of federated training.
        
        Args:
            train_loaders (List[DataLoader]): List of client data loaders
            criterion (nn.Module): Loss function
            learning_rate (float): Learning rate for client optimizers
            selected_clients (List[int]): Indices of selected clients
            local_epochs (int): Number of local training epochs
            
        Returns:
            float: Average training loss across clients
        """
        if selected_clients is None:
            selected_clients = list(range(self.num_clients))
        
        client_models = []
        client_weights = []
        total_loss = 0.0
        
        for client_id in selected_clients:
            # Create a copy of global model for this client
            client_model = copy.deepcopy(self.global_model)
            client_model.to(self.device)
            
            # Create optimizer for client
            optimizer = torch.optim.SGD(
                client_model.parameters(), 
                lr=learning_rate
            )
            
            # Client update
            if isinstance(self.algorithm, FedAvg) or hasattr(self.algorithm, 'client_update'):
                updated_model = self.algorithm.client_update(
                    model=client_model,
                    train_loader=train_loaders[client_id],
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    epochs=local_epochs
                )
                client_models.append(updated_model)
                client_weights.append(len(train_loaders[client_id].dataset))
            
            # Calculate client loss for monitoring
            client_model.eval()
            with torch.no_grad():
                batch_loss = 0.0
                num_batches = 0
                for data, target in train_loaders[client_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    output = client_model(data)
                    loss = criterion(output, target)
                    batch_loss += loss.item()
                    num_batches += 1
                total_loss += batch_loss / num_batches
        
        # Aggregate client updates
        self.global_model = self.algorithm.aggregate_updates(
            global_model=self.global_model,
            client_models=client_models,
            client_weights=client_weights
        )
        
        # Fine-tune on common dataset if available
        if self.common_loader is not None:
            self._finetune_on_common_data(criterion, learning_rate)
        
        avg_loss = total_loss / len(selected_clients)
        return avg_loss
    
    def _finetune_on_common_data(
        self,
        criterion: nn.Module,
        learning_rate: float,
        epochs: int = 1
    ):
        """
        Fine-tune global model on common dataset.
        
        This helps mitigate non-IID data challenges.
        """
        self.global_model.train()
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=learning_rate)
        
        for _ in range(epochs):
            for data, target in self.common_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.global_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    def evaluate(self, test_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        """
        Evaluate global model on test set.
        
        Args:
            test_loader (DataLoader): Test data loader
            criterion (nn.Module): Loss function
            
        Returns:
            tuple[float, float]: (test_loss, test_accuracy)
        """
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = correct / total
        
        return test_loss, accuracy
    
    def train(
        self,
        train_loaders: List[DataLoader],
        test_loader: DataLoader,
        criterion: nn.Module,
        num_rounds: int,
        learning_rate: float = 0.01,
        clients_per_round: Optional[int] = None,
        local_epochs: int = 1,
        eval_every: int = 1,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Train the federated model for multiple rounds.
        
        Args:
            train_loaders (List[DataLoader]): List of client data loaders
            test_loader (DataLoader): Test data loader
            criterion (nn.Module): Loss function
            num_rounds (int): Number of communication rounds
            learning_rate (float): Learning rate
            clients_per_round (int): Number of clients per round (None = all)
            local_epochs (int): Local training epochs per round
            eval_every (int): Evaluate every N rounds
            verbose (bool): Print progress
            
        Returns:
            Dict[str, List]: Training history
        """
        progress_bar = tqdm(range(num_rounds), desc="Federated Training") if verbose else range(num_rounds)
        
        for round_num in progress_bar:
            # Select clients for this round
            selected_clients = self.select_clients(
                num_selected=clients_per_round,
                selection_strategy='random' if clients_per_round else 'all'
            )
            
            # Train one round
            train_loss = self.train_round(
                train_loaders=train_loaders,
                criterion=criterion,
                learning_rate=learning_rate,
                selected_clients=selected_clients,
                local_epochs=local_epochs
            )
            
            # Evaluate
            if (round_num + 1) % eval_every == 0:
                test_loss, test_accuracy = self.evaluate(test_loader, criterion)
                
                self.history['rounds'].append(round_num + 1)
                self.history['train_loss'].append(train_loss)
                self.history['test_loss'].append(test_loss)
                self.history['test_accuracy'].append(test_accuracy)
                
                if verbose:
                    progress_bar.set_postfix({
                        'Round': round_num + 1,
                        'Loss': f'{train_loss:.4f}',
                        'Test Acc': f'{test_accuracy:.4f}'
                    })
        
        return self.history
    
    def save_model(self, path: str):
        """Save the global model."""
        torch.save(self.global_model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load a saved model."""
        self.global_model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    print("Testing FederatedServer...")
    
    from models import SimpleCNN
    from algorithms import FedAvg
    from data_utils import create_federated_dataloaders
    
    # Create model and algorithm
    model = SimpleCNN(num_classes=10)
    algorithm = FedAvg(local_epochs=1)
    
    # Create server
    server = FederatedServer(
        model=model,
        algorithm=algorithm,
        num_clients=8,
        device='cpu'
    )
    
    print(f"Server initialized with {server.num_clients} clients")
    print(f"Using algorithm: {server.algorithm.name}")
    print(f"Device: {server.device}")
    
    # Test client selection
    selected = server.select_clients(num_selected=3, selection_strategy='random')
    print(f"Selected clients: {selected}")
    
    print("\nServer tests passed!")
