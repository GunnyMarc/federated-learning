# Quick Start Guide

This guide will help you get started with federated learning in just a few minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-learning-pytorch.git
cd federated-learning-pytorch

# Install dependencies
pip install -r requirements.txt
```

## Your First Federated Learning Experiment

### Step 1: Basic Example

Create a file called `my_first_fl.py`:

```python
import torch
import torch.nn as nn
from federated_learning import (
    SimpleCNN,
    FedAvg,
    create_federated_dataloaders,
    FederatedServer
)

# 1. Create federated data loaders
train_loaders, test_loader = create_federated_dataloaders(
    num_clients=5,
    dataset_name='CIFAR10',
    batch_size=32,
    iid=True
)

# 2. Create model
model = SimpleCNN(num_classes=10)

# 3. Create federated server
server = FederatedServer(
    model=model,
    algorithm=FedAvg(local_epochs=1),
    num_clients=5
)

# 4. Train!
criterion = nn.CrossEntropyLoss()
history = server.train(
    train_loaders=train_loaders,
    test_loader=test_loader,
    criterion=criterion,
    num_rounds=10,
    learning_rate=0.01
)

# 5. Check results
print(f"Final accuracy: {history['test_accuracy'][-1]:.4f}")
```

Run it:
```bash
python my_first_fl.py
```

### Step 2: Non-IID Data (More Realistic)

```python
# Create non-IID data distribution
train_loaders, test_loader = create_federated_dataloaders(
    num_clients=8,
    dataset_name='CIFAR10',
    batch_size=32,
    iid=False,  # Non-IID!
    classes_per_client=3  # Each client sees only 3 classes
)

# Rest is the same...
```

### Step 3: Using Pre-built Examples

```bash
# IID training
python examples/train_federated_iid.py --num_clients 10 --num_rounds 20

# Non-IID training
python examples/train_federated_non_iid.py --classes_per_client 2 --use_common_dataset

# Compare all approaches
python examples/compare_approaches.py
```

## Understanding the Output

During training, you'll see:
```
Round number: 1 | Loss: 2.1234 | Accuracy: 0.4567
Round number: 2 | Loss: 1.9876 | Accuracy: 0.5123
...
```

- **Round**: One communication round (all clients train and update global model)
- **Loss**: Average loss across clients
- **Accuracy**: Test accuracy on global test set

## Key Concepts

### IID vs Non-IID

**IID (Independent and Identically Distributed)**
- Each client has similar data distribution
- All classes are represented equally
- Easier to train, better convergence

**Non-IID**
- Each client has different data distribution
- Some classes may be missing from some clients
- More realistic, represents real-world scenarios
- Harder to train, may need mitigation strategies

### Communication Rounds

In federated learning, we use "communication rounds" instead of "epochs":
- **1 communication round** = all selected clients train once and send updates
- Typically uses fewer rounds than traditional training epochs
- Each round involves network communication

### Client Selection

Not all clients participate in each round:
```python
server.train(
    ...
    clients_per_round=5,  # Only 5 clients per round
    ...
)
```

This simulates real-world scenarios where:
- Devices may be offline
- Battery/bandwidth constraints
- Privacy preferences

## Common Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_clients` | Total number of clients | 10-100 |
| `num_rounds` | Communication rounds | 10-100 |
| `local_epochs` | Training epochs per client per round | 1-5 |
| `batch_size` | Batch size for local training | 32-128 |
| `learning_rate` | Learning rate | 0.001-0.1 |
| `clients_per_round` | Clients participating per round | 10-100% |

## Troubleshooting

### Low Accuracy with Non-IID Data?

Try these solutions:

1. **Use a common dataset**:
```python
from federated_learning import create_common_dataset

common_loader = create_common_dataset('CIFAR10', num_samples=1000)
server.set_common_dataset(common_loader)
```

2. **Increase local epochs**:
```python
algorithm = FedAvg(local_epochs=5)  # Train more locally
```

3. **Use FedProx algorithm**:
```python
from federated_learning import FedProx
algorithm = FedProx(mu=0.01)  # Proximal term helps with heterogeneity
```

### Out of Memory?

1. Reduce batch size
2. Use fewer clients per round
3. Use a smaller model
4. Use CPU instead of GPU (slower but less memory)

### Slow Training?

1. Increase `num_workers` in data loaders
2. Use GPU if available
3. Reduce number of clients
4. Increase batch size

## Next Steps

1. **Read the examples**: Check out `examples/` for complete scripts
2. **Experiment with algorithms**: Try FedSGD, FedAvg, and FedProx
3. **Try different datasets**: MNIST, FashionMNIST, CIFAR100
4. **Customize models**: Create your own model architectures
5. **Analyze results**: Use the plotting functions to visualize training

## Advanced Usage

### Custom Model

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
        
    def forward(self, x):
        # Your forward pass
        return x

model = MyModel()
server = FederatedServer(model=model, ...)
```

### Custom Data Distribution

```python
from federated_learning import create_non_iid_split
from torch.utils.data import DataLoader

# Create your own split
train_dataset, _ = load_dataset('CIFAR10')
client_datasets = create_non_iid_split(
    train_dataset,
    num_clients=10,
    classes_per_client=2
)

# Create loaders
train_loaders = [
    DataLoader(ds, batch_size=32, shuffle=True)
    for ds in client_datasets
]
```

## Getting Help

- 📖 Check the [full documentation](docs/)
- 💬 Open an issue on GitHub
- 📧 Email: marc@techgeezer.io
