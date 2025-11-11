"""
Federated Averaging (FedAvg) Algorithm

Implementation of the FedAvg algorithm as described in:
"Communication-Efficient Learning of Deep Networks from Decentralized Data"
by McMahan et al.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple
import copy
from tqdm import tqdm


class FedAvgClient:
    """
    Client for Federated Averaging algorithm.

    Each client trains a local model on its private data for E epochs,
    then returns the updated weights to the server.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            learning_rate: float = 0.01,
            device: str = 'cpu'
    ):
        """
        Initialize FedAvg client.

        Args:
            model (nn.Module): Neural network model
            train_loader (DataLoader): Client's local training data
            learning_rate (float): Learning rate for local training
            device (str): Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_dataset_size(self) -> int:
        """Get the number of samples in client's dataset."""
        return len(self.train_loader.dataset)

    def set_weights(self, weights: List[torch.Tensor]):
        """
        Set model weights from global model.

        Args:
            weights (list): List of weight tensors
        """
        with torch.no_grad():
            for param, weight in zip(self.model.parameters(), weights):
                param.data.copy_(weight.to(self.device))

    def get_weights(self) -> List[torch.Tensor]:
        """
        Get current model weights.

        Returns:
            list: List of weight tensors
        """
        return [param.data.clone().cpu() for param in self.model.parameters()]

    def train(self, epochs: int = 1) -> Tuple[List[torch.Tensor], float]:
        """
        Train the local model for specified number of epochs.

        This is Step 1 and Step 2 of FedAvg:
        1. Compute gradients on client's data
        2. Update local model on client

        Args:
            epochs (int): Number of local training epochs

        Returns:
            tuple: (updated weights, average loss)
        """
        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss / len(self.train_loader)

        avg_loss = total_loss / epochs

        # Return updated weights
        return self.get_weights(), avg_loss


class FedAvgServer:
    """
    Server for Federated Averaging algorithm.

    The server maintains the global model and orchestrates the training process.
    """

    def __init__(
            self,
            model: nn.Module,
            device: str = 'cpu'
    ):
        """
        Initialize FedAvg server.

        Args:
            model (nn.Module): Global model
            device (str): Device to run on
        """
        self.global_model = model.to(device)
        self.device = device

    def get_weights(self) -> List[torch.Tensor]:
        """Get global model weights."""
        return [param.data.clone().cpu() for param in self.global_model.parameters()]

    def set_weights(self, weights: List[torch.Tensor]):
        """
        Set global model weights.

        Args:
            weights (list): List of weight tensors
        """
        with torch.no_grad():
            for param, weight in zip(self.global_model.parameters(), weights):
                param.data.copy_(weight.to(self.device))

    def aggregate_weights(
            self,
            client_weights: List[List[torch.Tensor]],
            client_sizes: List[int]
    ) -> List[torch.Tensor]:
        """
        Aggregate client weights using weighted averaging.

        This is Step 3 of FedAvg: Aggregate in global model.

        Args:
            client_weights (list): List of weight lists from each client
            client_sizes (list): Number of samples each client used

        Returns:
            list: Aggregated weights
        """
        total_samples = sum(client_sizes)

        # Initialize aggregated weights
        aggregated_weights = []

        # Aggregate each layer
        num_layers = len(client_weights[0])
        for layer_idx in range(num_layers):
            # Weighted sum
            layer_sum = torch.zeros_like(client_weights[0][layer_idx])

            for client_idx, weights in enumerate(client_weights):
                weight = client_sizes[client_idx] / total_samples
                layer_sum += weights[layer_idx] * weight

            aggregated_weights.append(layer_sum)

        return aggregated_weights

    def evaluate(
            self,
            test_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate global model on test data.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            tuple: (accuracy, average loss)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.global_model(data)
                loss = criterion(output, target)

                total_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        return accuracy, avg_loss


class FedAvgTrainer:
    """
    Orchestrates the complete FedAvg training process.
    """

    def __init__(
            self,
            model: nn.Module,
            client_loaders: List[DataLoader],
            test_loader: DataLoader,
            learning_rate: float = 0.01,
            device: str = 'cpu'
    ):
        """
        Initialize FedAvg trainer.

        Args:
            model (nn.Module): Model architecture
            client_loaders (list): List of DataLoaders for each client
            test_loader (DataLoader): Test data loader
            learning_rate (float): Learning rate for local training
            device (str): Device to use
        """
        self.server = FedAvgServer(copy.deepcopy(model), device)
        self.clients = [
            FedAvgClient(
                copy.deepcopy(model),
                loader,
                learning_rate,
                device
            )
            for loader in client_loaders
        ]
        self.test_loader = test_loader
        self.device = device

        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': []
        }

    def train_round(
            self,
            local_epochs: int = 1,
            client_fraction: float = 1.0
    ) -> dict:
        """
        Execute one round of federated training.

        Args:
            local_epochs (int): Number of epochs each client trains
            client_fraction (float): Fraction of clients to use (for client sampling)

        Returns:
            dict: Round statistics
        """
        # Select clients for this round
        num_selected = max(1, int(len(self.clients) * client_fraction))
        selected_indices = torch.randperm(len(self.clients))[:num_selected].tolist()

        # Get global weights
        global_weights = self.server.get_weights()

        # Collect updates from selected clients
        client_weights = []
        client_sizes = []
        client_losses = []

        for idx in selected_indices:
            client = self.clients[idx]

            # Set global weights
            client.set_weights(global_weights)

            # Train locally
            weights, loss = client.train(epochs=local_epochs)

            client_weights.append(weights)
            client_sizes.append(client.get_dataset_size())
            client_losses.append(loss)

        # Aggregate weights on server
        aggregated_weights = self.server.aggregate_weights(
            client_weights,
            client_sizes
        )

        # Update global model
        self.server.set_weights(aggregated_weights)

        # Evaluate
        test_acc, test_loss = self.server.evaluate(self.test_loader)

        # Statistics
        stats = {
            'num_clients': num_selected,
            'avg_train_loss': sum(client_losses) / len(client_losses),
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }

        return stats

    def train(
            self,
            num_rounds: int,
            local_epochs: int = 1,
            client_fraction: float = 1.0,
            verbose: bool = True
    ) -> dict:
        """
        Train the federated model for specified number of rounds.

        Args:
            num_rounds (int): Number of communication rounds
            local_epochs (int): Number of local training epochs per round
            client_fraction (float): Fraction of clients to use per round
            verbose (bool): Whether to print progress

        Returns:
            dict: Training history
        """
        iterator = tqdm(range(num_rounds)) if verbose else range(num_rounds)

        for round_num in iterator:
            stats = self.train_round(local_epochs, client_fraction)

            # Update history
            self.history['train_loss'].append(stats['avg_train_loss'])
            self.history['test_loss'].append(stats['test_loss'])
            self.history['test_accuracy'].append(stats['test_accuracy'])

            if verbose:
                iterator.set_description(
                    f"Round {round_num + 1}/{num_rounds} | "
                    f"Loss: {stats['test_loss']:.4f} | "
                    f"Acc: {stats['test_accuracy']:.4f}"
                )

        return self.history

    def get_global_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.server.global_model


if __name__ == "__main__":
    print("Testing FedAvg implementation...")

    # This is a minimal test - see examples/ for full usage
    from torch.utils.data import TensorDataset

    # Create dummy data
    num_clients = 3
    samples_per_client = 100

    client_loaders = []
    for _ in range(num_clients):
        X = torch.randn(samples_per_client, 3, 32, 32)
        y = torch.randint(0, 10, (samples_per_client,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)
        client_loaders.append(loader)

    # Test data
    X_test = torch.randn(50, 3, 32, 32)
    y_test = torch.randint(0, 10, (50,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Create model
    from src.models.simple_nn import SimpleNN

    model = SimpleNN()

    # Train
    trainer = FedAvgTrainer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader
    )

    print("Training for 3 rounds...")
    history = trainer.train(num_rounds=3, verbose=True)

    print("\nTraining complete!")
    print(f"Final accuracy: {history['test_accuracy'][-1]:.4f}")