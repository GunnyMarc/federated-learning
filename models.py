"""
Neural network models for federated learning.

This module provides various model architectures optimized for federated learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    Simple fully connected neural network for CIFAR-10.
    
    This is the basic architecture used in the federated learning paper demonstrations.
    It flattens images and uses fully connected layers.
    
    Args:
        input_size (int): Size of flattened input (default: 3*32*32 for CIFAR-10)
        hidden_size (int): Size of hidden layer
        num_classes (int): Number of output classes
    """
    
    def __init__(self, input_size: int = 3 * 32 * 32, hidden_size: int = 128, num_classes: int = 10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for image classification.
    
    A basic CNN with convolutional and fully connected layers, suitable for
    federated learning experiments with image data.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
    """
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN with batch normalization and more layers.
    
    This model includes batch normalization for better convergence in
    federated learning scenarios with non-IID data.
    
    Args:
        num_classes (int): Number of output classes
    """
    
    def __init__(self, num_classes: int = 10):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MNISTNet(nn.Module):
    """
    Simple CNN optimized for MNIST dataset.
    
    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST)
    """
    
    def __init__(self, num_classes: int = 10):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name (str): Name of the model ('simple_nn', 'simple_cnn', 'improved_cnn', 'mnist_net')
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        nn.Module: Initialized model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    models = {
        'simple_nn': SimpleNN,
        'simple_cnn': SimpleCNN,
        'improved_cnn': ImprovedCNN,
        'mnist_net': MNISTNet,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
    
    return models[model_name](num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Test models
    print("Testing SimpleNN...")
    model = SimpleNN()
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("\nTesting SimpleCNN...")
    model = SimpleCNN()
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("\nTesting ImprovedCNN...")
    model = ImprovedCNN()
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    print("\nTesting MNISTNet...")
    model = MNISTNet()
    x_mnist = torch.randn(4, 1, 28, 28)
    output = model(x_mnist)
    print(f"Input shape: {x_mnist.shape}, Output shape: {output.shape}")
    
    print("\nAll models working correctly!")
