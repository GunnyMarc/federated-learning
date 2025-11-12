"""
Federated learning algorithms implementation.

This module implements the core federated learning algorithms:
- FedSGD (Federated Stochastic Gradient Descent)
- FedAvg (Federated Averaging)
"""

from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy


class FederatedAlgorithm:
    """Base class for federated learning algorithms."""
    
    def __init__(self):
        self.name = "BaseAlgorithm"
    
    def client_update(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        **kwargs
    ) -> nn.Module:
        """
        Perform client-side update.
        
        Args:
            model: Global model
            train_loader: Client's training data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Updated model
        """
        raise NotImplementedError
    
    def aggregate_updates(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        client_weights: List[float]
    ) -> nn.Module:
        """
        Aggregate client updates into global model.
        
        Args:
            global_model: Current global model
            client_models: List of updated client models
            client_weights: Weights for aggregation (typically proportional to data size)
            
        Returns:
            Updated global model
        """
        raise NotImplementedError


class FedSGD(FederatedAlgorithm):
    """
    Federated Stochastic Gradient Descent (FedSGD).
    
    In FedSGD:
    1. Clients compute gradients on their local data
    2. Server aggregates gradients (weighted by dataset size)
    3. Server updates global model using aggregated gradients
    
    This is equivalent to mini-batch SGD where each client is a mini-batch.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "FedSGD"
    
    def client_update(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradients on client data.
        
        Returns:
            Dict containing gradients for each parameter
        """
        model.train()
        model.to(device)
        
        # Store gradients
        gradients = {}
        
        total_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in gradients:
                        gradients[name] = param.grad.clone()
                    else:
                        gradients[name] += param.grad.clone()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Average gradients
        for name in gradients:
            gradients[name] /= num_batches
        
        return gradients
    
    def aggregate_updates(
        self,
        global_model: nn.Module,
        client_gradients: List[Dict[str, torch.Tensor]],
        client_weights: List[float],
        learning_rate: float = 0.01
    ) -> nn.Module:
        """
        Aggregate gradients and update global model.
        
        Args:
            global_model: Current global model
            client_gradients: List of gradient dictionaries from clients
            client_weights: Weights for aggregation
            learning_rate: Learning rate for server-side update
            
        Returns:
            Updated global model
        """
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Aggregate gradients
        aggregated_gradients = {}
        for name, param in global_model.named_parameters():
            aggregated_gradients[name] = torch.zeros_like(param.data)
            
            for client_grad, weight in zip(client_gradients, normalized_weights):
                if name in client_grad:
                    aggregated_gradients[name] += weight * client_grad[name]
        
        # Update global model
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= learning_rate * aggregated_gradients[name]
        
        return global_model


class FedAvg(FederatedAlgorithm):
    """
    Federated Averaging (FedAvg).
    
    In FedAvg:
    1. Clients train model for multiple local epochs
    2. Clients send updated model weights to server
    3. Server aggregates weights (weighted average by dataset size)
    
    This is more communication-efficient than FedSGD.
    
    Args:
        local_epochs (int): Number of local training epochs per round
        local_batch_size (int): Batch size for local training
    """
    
    def __init__(self, local_epochs: int = 1, local_batch_size: int = 32):
        super().__init__()
        self.name = "FedAvg"
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
    
    def client_update(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = None,
        **kwargs
    ) -> nn.Module:
        """
        Train model on client data for multiple epochs.
        
        Args:
            model: Global model
            train_loader: Client's training data
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            epochs: Number of local epochs (overrides default)
            
        Returns:
            Updated model
        """
        epochs = epochs or self.local_epochs
        
        model.train()
        model.to(device)
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model
    
    def aggregate_updates(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        client_weights: List[float]
    ) -> nn.Module:
        """
        Aggregate client model weights into global model.
        
        Uses weighted averaging where weights are proportional to client dataset sizes.
        
        Args:
            global_model: Current global model
            client_models: List of updated client models
            client_weights: Weights for aggregation (dataset sizes)
            
        Returns:
            Updated global model
        """
        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated state dict
        aggregated_state_dict = copy.deepcopy(global_model.state_dict())
        
        # Zero out all parameters
        for key in aggregated_state_dict:
            aggregated_state_dict[key] = torch.zeros_like(aggregated_state_dict[key])
        
        # Weighted sum of client models
        for client_model, weight in zip(client_models, normalized_weights):
            client_state_dict = client_model.state_dict()
            for key in aggregated_state_dict:
                aggregated_state_dict[key] += weight * client_state_dict[key]
        
        # Update global model
        global_model.load_state_dict(aggregated_state_dict)
        
        return global_model


class FedProx(FederatedAlgorithm):
    """
    Federated Proximal (FedProx).
    
    FedProx adds a proximal term to the local objective, which helps with
    heterogeneous data and system settings.
    
    Args:
        local_epochs (int): Number of local training epochs
        mu (float): Proximal term coefficient
    """
    
    def __init__(self, local_epochs: int = 1, mu: float = 0.01):
        super().__init__()
        self.name = "FedProx"
        self.local_epochs = local_epochs
        self.mu = mu
    
    def client_update(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = None,
        **kwargs
    ) -> nn.Module:
        """
        Train model with proximal term.
        """
        epochs = epochs or self.local_epochs
        
        # Store global model parameters
        global_params = {name: param.clone().detach() 
                        for name, param in model.named_parameters()}
        
        model.train()
        model.to(device)
        
        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Add proximal term
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    proximal_term += ((param - global_params[name]) ** 2).sum()
                loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
        
        return model
    
    def aggregate_updates(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        client_weights: List[float]
    ) -> nn.Module:
        """Same aggregation as FedAvg."""
        return FedAvg().aggregate_updates(global_model, client_models, client_weights)


def get_algorithm(algorithm_name: str, **kwargs) -> FederatedAlgorithm:
    """
    Factory function to get federated learning algorithm.
    
    Args:
        algorithm_name (str): Name of algorithm ('fedsgd', 'fedavg', 'fedprox')
        **kwargs: Algorithm-specific parameters
        
    Returns:
        FederatedAlgorithm: Initialized algorithm
        
    Raises:
        ValueError: If algorithm_name is not recognized
    """
    algorithms = {
        'fedsgd': FedSGD,
        'fedavg': FedAvg,
        'fedprox': FedProx,
    }
    
    algorithm_name = algorithm_name.lower()
    if algorithm_name not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm_name](**kwargs)


if __name__ == "__main__":
    print("Testing federated learning algorithms...")
    
    # Create dummy model and data
    from models import SimpleCNN
    model = SimpleCNN(num_classes=10)
    
    print("\n1. Testing FedSGD...")
    fedsgd = FedSGD()
    print(f"Algorithm: {fedsgd.name}")
    
    print("\n2. Testing FedAvg...")
    fedavg = FedAvg(local_epochs=5)
    print(f"Algorithm: {fedavg.name}")
    print(f"Local epochs: {fedavg.local_epochs}")
    
    print("\n3. Testing FedProx...")
    fedprox = FedProx(local_epochs=5, mu=0.01)
    print(f"Algorithm: {fedprox.name}")
    print(f"Mu parameter: {fedprox.mu}")
    
    print("\n4. Testing algorithm factory...")
    alg = get_algorithm('fedavg', local_epochs=3)
    print(f"Created: {alg.name}")
    
    print("\nAll tests passed!")
