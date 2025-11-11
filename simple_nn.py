"""
Simple Neural Network for CIFAR-10 Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """
    A simple fully connected neural network for image classification.

    Architecture:
        - Input: Flattened images (32*32*3 = 3072)
        - Hidden layers: 128 -> 64 -> 32
        - Output: 10 classes
    """

    def __init__(self, input_size=3072, hidden_sizes=None, num_classes=10):
        """
        Initialize the neural network.

        Args:
            input_size (int): Size of input features (default: 3072 for CIFAR-10)
            hidden_sizes (list): List of hidden layer sizes
            num_classes (int): Number of output classes (default: 10)
        """
        super(SimpleNN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Build layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Pass through network
        x = self.network(x)

        return x

    def get_weights(self):
        """
        Get all model weights as a list of tensors.

        Returns:
            list: List of weight tensors
        """
        return [param.data.clone() for param in self.parameters()]

    def set_weights(self, weights):
        """
        Set model weights from a list of tensors.

        Args:
            weights (list): List of weight tensors
        """
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.data.copy_(weight)

    def get_gradients(self):
        """
        Get all gradients as a list of tensors.

        Returns:
            list: List of gradient tensors
        """
        return [param.grad.clone() if param.grad is not None else None
                for param in self.parameters()]


class CNNModel(nn.Module):
    """
    Convolutional Neural Network for image classification.

    A more sophisticated model for better performance on image tasks.
    """

    def __init__(self, num_classes=10):
        """
        Initialize the CNN.

        Args:
            num_classes (int): Number of output classes
        """
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        """Forward pass through the CNN."""
        # Convolutional layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_weights(self):
        """Get all model weights."""
        return [param.data.clone() for param in self.parameters()]

    def set_weights(self, weights):
        """Set model weights."""
        with torch.no_grad():
            for param, weight in zip(self.parameters(), weights):
                param.data.copy_(weight)


def create_model(model_type='simple', num_classes=10, **kwargs):
    """
    Factory function to create models.

    Args:
        model_type (str): Type of model ('simple' or 'cnn')
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for model initialization

    Returns:
        nn.Module: Initialized model
    """
    if model_type == 'simple':
        return SimpleNN(num_classes=num_classes, **kwargs)
    elif model_type == 'cnn':
        return CNNModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    print("Testing SimpleNN...")
    model = SimpleNN()
    test_input = torch.randn(4, 3, 32, 32)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    print("\nTesting CNNModel...")
    model = CNNModel()
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")