# Project Structure

This document explains the organization of the federated learning repository and what each file does.

## Directory Structure

```
federated-learning-pytorch/
├── README.md                      # Main project documentation
├── QUICKSTART.md                  # Quick start guide
├── PROJECT_STRUCTURE.md           # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
│
├── federated_learning/            # Main package directory
│   ├── __init__.py               # Package initialization
│   ├── models.py                 # Neural network architectures
│   ├── algorithms.py             # Federated learning algorithms
│   ├── data_utils.py             # Data loading and preparation
│   └── server.py                 # Federated server implementation
│
├── examples/                      # Example scripts
│   ├── train_federated_iid.py   # Train with IID data
│   ├── train_federated_non_iid.py # Train with non-IID data
│   ├── compare_approaches.py     # Compare all approaches
│   └── complete_example.py       # Comprehensive demo
│
└── tests/                         # Unit tests (optional)
    └── ...
```

## Core Package Files

### `__init__.py`
Package initialization file that exports all public APIs. Import federated learning components from here:

```python
from federated_learning import (
    SimpleCNN,           # Models
    FedAvg,              # Algorithms
    FederatedServer,     # Server
    create_federated_dataloaders  # Data utilities
)
```

### `models.py`
Neural network model architectures optimized for federated learning.

**Classes:**
- `SimpleNN` - Basic fully connected network
- `SimpleCNN` - Simple convolutional neural network
- `ImprovedCNN` - CNN with batch normalization
- `MNISTNet` - Optimized for MNIST dataset

**Functions:**
- `get_model(name)` - Factory function to create models

**Usage:**
```python
from federated_learning.models import SimpleCNN, get_model

model = SimpleCNN(num_classes=10)
# or
model = get_model('simple_cnn', num_classes=10)
```

### `algorithms.py`
Implementation of federated learning algorithms.

**Classes:**
- `FederatedAlgorithm` - Base class for all algorithms
- `FedSGD` - Federated Stochastic Gradient Descent
- `FedAvg` - Federated Averaging
- `FedProx` - Federated Proximal

**Key Methods:**
- `client_update()` - Perform local training on client
- `aggregate_updates()` - Aggregate updates from all clients

**Usage:**
```python
from federated_learning.algorithms import FedAvg

algorithm = FedAvg(local_epochs=5)
```

### `data_utils.py`
Utilities for loading and preparing federated datasets.

**Functions:**
- `load_dataset(name)` - Load standard datasets (CIFAR-10, MNIST, etc.)
- `create_iid_split(dataset, num_clients)` - Create uniform data split
- `create_non_iid_split(dataset, num_clients, ...)` - Create skewed data split
- `create_federated_dataloaders(...)` - One-stop function to create all loaders
- `create_common_dataset(...)` - Create small uniform dataset for fine-tuning
- `analyze_data_distribution(...)` - Analyze class distribution across clients

**Usage:**
```python
from federated_learning.data_utils import create_federated_dataloaders

# IID split
train_loaders, test_loader = create_federated_dataloaders(
    num_clients=10,
    dataset_name='CIFAR10',
    iid=True
)

# Non-IID split
train_loaders, test_loader = create_federated_dataloaders(
    num_clients=10,
    dataset_name='CIFAR10',
    iid=False,
    classes_per_client=3
)
```

### `server.py`
Central server that coordinates federated learning.

**Class: `FederatedServer`**
- Maintains global model
- Selects clients for each round
- Distributes model to clients
- Aggregates client updates
- Evaluates global model

**Key Methods:**
- `train(...)` - Main training loop
- `select_clients(...)` - Select clients for a round
- `train_round(...)` - Execute one communication round
- `evaluate(...)` - Evaluate on test set
- `save_model(path)` - Save trained model
- `set_common_dataset(loader)` - Set common dataset for fine-tuning

**Usage:**
```python
from federated_learning import FederatedServer, FedAvg
from models import SimpleCNN

model = SimpleCNN()
server = FederatedServer(
    model=model,
    algorithm=FedAvg(),
    num_clients=10
)

history = server.train(
    train_loaders=train_loaders,
    test_loader=test_loader,
    criterion=criterion,
    num_rounds=20,
    learning_rate=0.01
)
```

## Example Scripts

### `train_federated_iid.py`
Complete example of federated learning with IID (uniform) data distribution.

**Features:**
- Command-line arguments for configuration
- Plotting training curves
- Model saving
- Progress tracking

**Run:**
```bash
python examples/train_federated_iid.py \
    --num_clients 8 \
    --num_rounds 20 \
    --algorithm fedavg \
    --plot_results
```

### `train_federated_non_iid.py`
Example with non-IID (skewed) data distribution, closer to real-world scenarios.

**Features:**
- Non-IID data creation
- Data distribution visualization
- Common dataset fine-tuning option
- Performance comparison

**Run:**
```bash
python examples/train_federated_non_iid.py \
    --classes_per_client 3 \
    --use_common_dataset \
    --plot_results
```

### `compare_approaches.py`
Comprehensive comparison of all approaches:
1. Traditional centralized training
2. Federated learning with IID data
3. Federated learning with non-IID data
4. Federated learning with non-IID data + common dataset

**Run:**
```bash
python examples/compare_approaches.py
```

### `complete_example.py`
End-to-end demonstration with extensive visualization and analysis.

**Features:**
- Multiple experiments in one script
- Data distribution visualization
- Training curve plotting
- Final comparison plot
- Key insights and recommendations

**Run:**
```bash
python examples/complete_example.py
```

## Configuration Files

### `requirements.txt`
Lists all Python package dependencies:
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision datasets and transforms
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `scikit-learn` - Machine learning utilities
- And more...

**Install:**
```bash
pip install -r requirements.txt
```

### `setup.py`
Package installation configuration. Allows installing as a package:

```bash
pip install -e .
```

After installation, you can import from anywhere:
```python
from federated_learning import FederatedServer
```

## Documentation Files

### `README.md`
Main project documentation with:
- Overview and features
- Installation instructions
- Quick start guide
- Architecture diagrams
- Examples and usage
- API reference

### `QUICKSTART.md`
Beginner-friendly guide to get started quickly:
- Simple examples
- Step-by-step tutorials
- Common parameters
- Troubleshooting
- Next steps

### `PROJECT_STRUCTURE.md` (this file)
Detailed explanation of every file and directory in the repository.

### `LICENSE`
MIT License - allows free use, modification, and distribution with attribution.

## Typical Workflow

Here's how to use this repository:

### 1. Installation
```bash
git clone https://github.com/yourusername/federated-learning-pytorch.git
cd federated-learning-pytorch
pip install -r requirements.txt
```

### 2. Run Examples
```bash
# Quick test with IID data
python examples/train_federated_iid.py --num_rounds 10

# More realistic non-IID scenario
python examples/train_federated_non_iid.py --classes_per_client 2

# Full comparison
python examples/compare_approaches.py
```

### 3. Build Your Own
```python
# your_experiment.py
from federated_learning import *

# Your custom federated learning code here
```

### 4. Advanced Usage
- Modify `models.py` to add custom architectures
- Extend `algorithms.py` with new federated algorithms
- Customize `data_utils.py` for your own datasets
- Adapt `server.py` for specific requirements

## Key Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Simplicity**: Clean APIs that are easy to understand
3. **Flexibility**: Easy to customize and extend
4. **Documentation**: Extensive comments and docstrings
5. **Practical**: Ready-to-run examples for common scenarios

## File Dependencies

```
models.py
    ↓
algorithms.py → server.py
    ↓              ↓
data_utils.py → examples/*.py
```

All example scripts depend on the core package files but are independent of each other.

## Adding New Components

### New Model Architecture
1. Add class to `models.py`
2. Inherit from `nn.Module`
3. Add to `get_model()` factory function

### New Algorithm
1. Add class to `algorithms.py`
2. Inherit from `FederatedAlgorithm`
3. Implement `client_update()` and `aggregate_updates()`
4. Add to `get_algorithm()` factory function

### New Dataset
1. Add loading logic to `load_dataset()` in `data_utils.py`
2. Define appropriate transforms
3. Update `create_federated_dataloaders()` to support it

## Getting Help

- **API Documentation**: Check docstrings in each module
- **Examples**: Look at `examples/` directory
- **Issues**: Open an issue on GitHub
- **Questions**: Email or discussion board

## Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Add tests if applicable
5. Submit a pull request

---

This structure is designed to be both beginner-friendly and production-ready. Start with the examples, then dive into the core modules as needed!
