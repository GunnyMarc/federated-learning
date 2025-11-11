# Federated Learning: Privacy-Preserving Machine Learning

A comprehensive implementation of Federated Learning algorithms in PyTorch, demonstrating how to train machine learning models across decentralized devices while preserving data privacy.

## Overview

This repository implements federated learning from scratch, showcasing how to:
- Train models without centralizing sensitive data
- Handle Non-IID (Non-Independent and Identically Distributed) data
- Implement FedSGD and FedAvg algorithms
- Compare federated learning with traditional centralized training

## Key Features

- **Complete Implementation**: Full federated learning pipeline in PyTorch
- **Multiple Algorithms**: Both FedSGD and FedAvg implementations
- **Real-world Scenarios**: Handles IID and Non-IID data distributions
- **Client Selection**: Random client sampling for realistic scenarios
- **Performance Comparison**: Benchmarks against traditional training
- **Educational**: Extensively commented code for learning

## Architecture

```
Central Server (Global Model)
        ↓
    Distribute model weights
        ↓
Client Devices (Local Training)
        ↓
    Update local models
        ↓
    Send updates to server
        ↓
Server Aggregates Updates
        ↓
    Repeat
```

## Quick Start

### Prerequisites

```bash
python >= 3.10
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
matplotlib >= 3.7.0
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/federated-learning-tutorial.git
cd federated-learning-tutorial

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from federated_learning import FederatedLearning

# Initialize federated learning system
fl_system = FederatedLearning(
    num_clients=8,
    data_type='iid',  # or 'non_iid'
    algorithm='fedavg'  # or 'fedsgd'
)

# Train the model
fl_system.train(
    num_rounds=10,
    client_fraction=1.0,
    local_epochs=1
)

# Evaluate
accuracy = fl_system.evaluate()
print(f"Final Accuracy: {accuracy:.4f}")
```

## Repository Structure

```
federated-learning-tutorial/
├── README.md
├── requirements.txt
├── setup.py
├── examples/
│   ├── basic_federated_learning.py
│   ├── iid_vs_noniid_comparison.py
│   ├── client_selection_demo.py
│   └── traditional_vs_federated.py
├── src/
│   ├── __init__.py
│   ├── federated_learning.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── fedsgd.py
│   │   └── fedavg.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_distribution.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── simple_nn.py
│   └── utils/
│       ├── __init__.py
│       ├── aggregation.py
│       ├── client_selection.py
│       └── visualization.py
├── tests/
│   ├── test_algorithms.py
│   ├── test_data_distribution.py
│   └── test_aggregation.py
└── notebooks/
    └── federated_learning_tutorial.ipynb
```

## Algorithms Implemented

### 1. Federated Stochastic Gradient Descent (FedSGD)

FedSGD treats each client as a mini-batch and aggregates gradients:

```python
# Compute gradients on client data
gradients = compute_gradients(client_data, model_weights)

# Aggregate on server
global_weights = aggregate_gradients(all_gradients)
```

### 2. Federated Averaging (FedAvg)

FedAvg allows multiple local epochs and aggregates model weights:

```python
# Train locally for E epochs
for epoch in range(E):
    local_weights = train_local_model(client_data)

# Aggregate weights on server
global_weights = weighted_average(all_client_weights)
```

## Key Concepts

### When to Use Federated Learning

Federated learning is ideal when:

1. **Proxy data is not suitable**: Real device data differs significantly from public datasets
2. **Data is privacy-sensitive**: Contains personal information (photos, texts, health data)
3. **Data is naturally labeled**: User interactions provide implicit labels

### Challenges Addressed

- **Privacy Concerns**: Data never leaves user devices
- **Bandwidth Efficiency**: Only model updates are transmitted
- **Non-IID Data**: Techniques to handle heterogeneous data distributions
- **Client Selection**: Strategies for selecting active clients per round

## Performance Comparisons

### IID vs Non-IID Data

| Scenario | Round 6 Accuracy |
|----------|------------------|
| Traditional Training | ~52.6% |
| Federated (IID) | ~51.6% |
| Federated (Non-IID) | ~31.4% |
| Federated (Non-IID + Fine-tuning) | ~41.2% |

*Note: These are illustrative results from simplified models*

## Advanced Features

### Client Selection Strategies

```python
# Random client selection
selected_clients = random_client_selection(
    num_clients=total_clients,
    fraction=0.3
)

# Availability-based selection
selected_clients = availability_based_selection(
    clients=all_clients,
    time_window='daytime'
)
```

### Handling Non-IID Data

```python
# Fine-tune on common dataset
def train_with_common_data(client_models, common_dataset):
    for model in client_models:
        fine_tune(model, common_dataset, epochs=1)
    return client_models
```

## Examples

### Example 1: Basic Federated Learning

```python
from src.federated_learning import FederatedLearning

# Create federated learning system
fl = FederatedLearning(num_clients=8)

# Train for 10 communication rounds
results = fl.train(num_rounds=10)

# Plot training progress
fl.plot_results(results)
```

### Example 2: Non-IID Data Handling

```python
# Create non-IID data distribution
fl = FederatedLearning(
    num_clients=8,
    data_type='non_iid',
    use_common_data=True
)

# Train with fine-tuning
results = fl.train(num_rounds=10, fine_tune=True)
```

See the `examples/` directory for more detailed use cases.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_algorithms.py

# Run with coverage
pytest --cov=src tests/
```

## Documentation

Detailed documentation for each component:

- [Algorithms](docs/algorithms.md): Deep dive into FedSGD and FedAvg
- [Data Distribution](docs/data_distribution.md): IID vs Non-IID handling
- [Client Selection](docs/client_selection.md): Selection strategies
- [API Reference](docs/api_reference.md): Complete API documentation

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the foundational papers:
  - McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data"
  - Konečný et al. "Federated Learning: Strategies for Improving Communication Efficiency"
- CIFAR-10 dataset from the Canadian Institute for Advanced Research

## Contact

For questions or feedback:
- Open an issue on GitHub
- Email: your.email@example.com

## Related Resources

- [Federated Learning Paper](https://arxiv.org/abs/1602.05629)
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [PySyft - Federated Learning Library](https://github.com/OpenMined/PySyft)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{federated_learning_tutorial,
  author = {Your Name},
  title = {Federated Learning: Privacy-Preserving Machine Learning},
  year = {2024},
  url = {https://github.com/yourusername/federated-learning-tutorial}
}
```

---

**Note**: This implementation is for educational purposes. For production use, consider using established frameworks like TensorFlow Federated or PySyft.