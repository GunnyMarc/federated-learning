# Federated Learning: Privacy-Preserving Machine Learning

A comprehensive PyTorch implementation of Federated Learning algorithms, demonstrating privacy-preserving machine learning techniques where training data remains on client devices.

## Overview

This repository implements federated learning from scratch, showcasing how machine learning models can be trained collaboratively across multiple decentralized devices without centralizing sensitive data.

### Key Features

- **Privacy-First Design**: Data never leaves client devices
- **Multiple Algorithms**: Implementation of FedSGD and FedAvg
- **IID & Non-IID Support**: Handle both uniform and skewed data distributions
- **Client Selection**: Random and strategic client sampling
- **Comprehensive Examples**: Ready-to-run examples with CIFAR-10

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/GunnyMarc/federated-learning.git
cd federated-learning

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Basic Usage

```python
from federated_learning import FederatedServer
from federated_learning.algorithms import FedAvg
from federated_learning.data_utils import create_federated_dataloaders
from federated_learning.models import SimpleCNN
import torch.nn as nn
import torch.optim as optim

# Create federated data loaders
train_loaders, test_loader = create_federated_dataloaders(
    num_clients=10,
    dataset_name='CIFAR10',
    batch_size=32,
    iid=True
)

# Initialize global model
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()

# Create federated server
server = FederatedServer(
    model=model,
    algorithm=FedAvg(),
    num_clients=10
)

# Train
results = server.train(
    train_loaders=train_loaders,
    test_loader=test_loader,
    criterion=criterion,
    num_rounds=10,
    learning_rate=0.01
)
```

## Federated Learning Concepts

### What is Federated Learning?

Federated Learning is a machine learning paradigm where:
1. **Models go to data** instead of data going to models
2. **Training happens locally** on client devices
3. **Only model updates** are shared with the central server
4. **Privacy is preserved** as raw data never leaves devices

### When to Use Federated Learning

Federated learning is ideal when:

- **Proxy data is unsuitable**: Device data differs from public datasets
- **Data is privacy-sensitive**: Photos, messages, health records
- **Data is naturally labeled**: User interactions provide labels
- **Bandwidth is limited**: Sending model updates < sending raw data

### Key Algorithms

#### FedSGD (Federated Stochastic Gradient Descent)
- Clients compute gradients on local data
- Server aggregates gradients proportionally
- Single training pass per communication round

#### FedAvg (Federated Averaging)
- Clients train model for multiple epochs locally
- Server averages model weights (not gradients)
- More efficient communication
- Better convergence properties

## Examples

### Example 1: IID Data Distribution

```bash
python examples/train_federated_iid.py \
    --num_clients 8 \
    --num_rounds 20 \
    --batch_size 32 \
    --algorithm fedavg
```

### Example 2: Non-IID Data Distribution

```bash
python examples/train_federated_non_iid.py \
    --num_clients 8 \
    --num_rounds 20 \
    --batch_size 32 \
    --classes_per_client 3
```

### Example 3: Client Selection

```bash
python examples/train_with_client_selection.py \
    --num_clients 100 \
    --clients_per_round 10 \
    --num_rounds 50
```

### Example 4: Compare All Approaches

```bash
python examples/compare_approaches.py
```

## Results

### Performance Comparison (CIFAR-10, 20 rounds)

| Approach | Final Accuracy | Communication Cost |
|----------|---------------|-------------------|
| Traditional Training | 52.64% | N/A |
| FedAvg (IID, all clients) | 51.56% | Low |
| FedAvg (Non-IID) | 31.43% | Low |
| FedAvg (Non-IID + fine-tuning) | 41.22% | Low |

*Note: These are baseline results with a simple neural network for demonstration purposes.*

## Architecture

```
┌─────────────────────────────────────────┐
│         Federated Server                │
│  - Maintains global model               │
│  - Orchestrates training rounds         │
│  - Aggregates client updates            │
└───────────┬─────────────────────────────┘
            │
            ├──────┬──────┬──────┬─────────
            │      │      │      │
         ┌──▼──┐ ┌─▼──┐ ┌─▼──┐ ┌─▼──┐
         │C1   │ │C2  │ │C3  │ │CN  │
         │Data │ │Data│ │Data│ │Data│
         └─────┘ └────┘ └────┘ └────┘
         Clients with local data
```

## License

This project is licensed under the MIT License.

## References

- McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks"
