# Getting Started with Federated Learning

This
guide
will
help
you
get
up and running
with the Federated Learning Tutorial repository.

## Quick Start (5 minutes)

### 1. Installation

```bash
# Clone the repository
git
clone
https: // github.com / yourusername / federated - learning - tutorial.git
cd
federated - learning - tutorial

# Create virtual environment (recommended)
python - m
venv
venv
source
venv / bin / activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip
install - r
requirements.txt

# Or install as package
pip
install - e.
```

### 2. Run Your First Example

```bash
# Run the quick start example
python
examples / quick_start.py
```

This
will:
- Download
CIFAR - 10
dataset
automatically
- Create
8
federated
clients
- Train
for 10 rounds
    - Show
    you
    the
    results

Expected
output:
```
Configuration:
- Number
of
clients: 8
- Training
rounds: 10
- Device: cpu / cuda

[Step 1]
Preparing
data...
✓ Created
8
clients
✓ Each
client
has
~6250
samples
...

Final
Results:
- Test
Accuracy: 0.5156(51.56 %)
```

## Understanding the Code

### Basic Usage Pattern

```python
from src.models.simple_nn import SimpleNN
from src.data.data_distribution import DataDistributor
from src.algorithms.fedavg import FedAvgTrainer

# 1. Prepare data
distributor = DataDistributor('cifar10')
client_loaders, test_loader = distributor.create_iid_distribution(
    num_clients=8,
    batch_size=32
)

# 2. Create model
model = SimpleNN(num_classes=10)

# 3. Initialize trainer
trainer = FedAvgTrainer(
    model=model,
    client_loaders=client_loaders,
    test_loader=test_loader,
    learning_rate=0.01
)

# 4. Train
history = trainer.train(num_rounds=10)

# 5. Get results
print(f"Final accuracy: {history['test_accuracy'][-1]:.4f}")
```

## Examples Overview

### 1. Quick Start (`examples/quick_start.py`)
** Best
for **: First - time
users
- Simplest
possible
example
- Shows
basic
workflow
- ~5
minutes
to
run

### 2. Complete Demo (`examples/complete_federated_learning_demo.py`)
** Best
for **: Understanding
all
concepts
- IID
vs
Non - IID
data
- FedAvg
vs
FedSGD
- Comparison
with traditional training
- Multiple
experiments
- ~30
minutes
to
run

### 3. Non-IID Handling (`examples/noniid_handling_demo.py`)
** Best
for **: Learning
about
data
heterogeneity
- Demonstrates
the
Non - IID
challenge
- Shows
fine - tuning
solution
- Includes
visualizations
- ~15
minutes
to
run

## Key Concepts

### 1. Data Distribution

```python
# IID: Each client has balanced, representative data
client_loaders, test_loader = distributor.create_iid_distribution(
    num_clients=8
)

# Non-IID: Clients have heterogeneous data
client_loaders, test_loader = distributor.create_non_iid_distribution(
    num_clients=8,
    alpha=0.5  # Lower = more skewed
)
```

### 2. Algorithms

** FedAvg(Federated
Averaging) **:
- Clients
train
for multiple epochs locally
- Server
aggregates
model
weights
- More
communication - efficient

** FedSGD(Federated
SGD) **:
- Clients
compute
gradients
only
- Server
aggregates
gradients
- Faster
convergence
per
round

### 3. Client Selection

```python
# Use all clients (100%)
history = trainer.train(
    num_rounds=10,
    client_fraction=1.0
)

# Random 50% of clients per round
history = trainer.train(
    num_rounds=10,
    client_fraction=0.5
)
```

## Common Tasks

### Train with Your Own Dataset

```python
from torch.utils.data import DataLoader, TensorDataset

# Create your datasets
client_loaders = []
for client_data in your_client_datasets:
    X, y = client_data
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    client_loaders.append(loader)

# Rest is the same
trainer = FedAvgTrainer(model, client_loaders, test_loader)
history = trainer.train(num_rounds=10)
```

### Use a Different Model

```python
from src.models.simple_nn import CNNModel

# Use CNN instead of simple NN
model = CNNModel(num_classes=10)

# Or create custom model
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here

    def forward(self, x):
        # Your forward pass
        return x

    def get_weights(self):
        return [p.data.clone() for p in self.parameters()]

    def set_weights(self, weights):
        with torch.no_grad():
            for p, w in zip(self.parameters(), weights):
                p.data.copy_(w)


model = CustomModel()
```

### Save and Load Models

```python
# After training
trained_model = trainer.get_global_model()
torch.save(trained_model.state_dict(), 'my_model.pth')

# Load later
model = SimpleNN(num_classes=10)
model.load_state_dict(torch.load('my_model.pth'))
model.eval()
```

### Visualize Results

```python
import matplotlib.pyplot as plt

history = trainer.train(num_rounds=10)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['test_accuracy'])
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history['test_loss'])
plt.xlabel('Round')
plt.ylabel('Loss')
plt.title('Test Loss')

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()
```

## Troubleshooting

### "CUDA out of memory"

```python
# Use CPU instead
DEVICE = 'cpu'

# Or reduce batch size
client_loaders, test_loader = distributor.create_iid_distribution(
    num_clients=8,
    batch_size=16  # Smaller batch size
)
```

### "Dataset not found"

The
first
run
will
download
CIFAR - 10
automatically.If
this
fails:

```python
# Manually specify data directory
distributor = DataDistributor('cifar10', data_dir='/path/to/data')
```

### Slow training

```python
# Reduce number of clients or rounds
trainer.train(
    num_rounds=5,  # Fewer rounds
    client_fraction=0.5  # Fewer clients per round
)
```

## Next Steps

1. ** Experiment
with parameters **: Try
different
numbers
of
clients, rounds, and learning
rates
2. ** Try
Non - IID
data **: See
how
data
heterogeneity
affects
performance
3. ** Compare
algorithms **: Run
FedAvg
vs
FedSGD
4. ** Use
your
own
data **: Apply
federated
learning
to
your
problem
5. ** Read
the
paper **: [Communication - Efficient Learning of Deep Networks
from Decentralized Data](https: // arxiv.org / abs / 1602.05629)

## Getting Help

- 📖 Check
the[README.md](README.md)
for detailed documentation
    - 💻 Look
    at
    examples in the
    `examples / ` directory
- Report
issues
on
GitHub
- 📧 Contact: marc@techgeezer.io

## Resources

- ** Original
FedAvg
Paper **: https: // arxiv.org / abs / 1602.05629
- ** PyTorch
Documentation **: https: // pytorch.org / docs /
- ** TensorFlow
Federated **: https: // www.tensorflow.org / federated
- ** PySyft **: https: // github.com / OpenMined / PySyft

---

Happy Federated Learning!