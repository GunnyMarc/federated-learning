# Repository Structure

This document provides a complete overview of the repository structure and file purposes.

## Directory Tree

```
federated-learning-tutorial/
├── README.md                          # Main documentation
├── GETTING_STARTED.md                 # Quick start guide
├── STRUCTURE.md                       # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation script
│
├── src/                               # Source code
│   ├── __init__.py                    # Package initialization
│   │
│   ├── models/                        # Neural network models
│   │   ├── __init__.py
│   │   └── simple_nn.py               # SimpleNN and CNNModel implementations
│   │
│   ├── algorithms/                    # Federated learning algorithms
│   │   ├── __init__.py
│   │   ├── fedavg.py                  # FedAvg implementation
│   │   └── fedsgd.py                  # FedSGD implementation
│   │
│   ├── data/                          # Data loading and distribution
│   │   ├── __init__.py
│   │   └── data_distribution.py       # IID/Non-IID data distribution
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       └── aggregation.py             # Weight/gradient aggregation
│
├── examples/                          # Example scripts
│   ├── quick_start.py                 # Simplest example (5 min)
│   ├── complete_federated_learning_demo.py  # Comprehensive demo (30 min)
│   └── noniid_handling_demo.py        # Non-IID data handling (15 min)
│
└── tests/                             # Unit tests
    └── test_federated_learning.py     # All tests
```

## File Descriptions

### Core Documentation

#### `README.md`
- Main project documentation
- Installation instructions
- API reference
- Usage examples
- Feature overview

#### `GETTING_STARTED.md`
- Quick start guide for beginners
- Step-by-step tutorials
- Common tasks and recipes
- Troubleshooting tips

#### `STRUCTURE.md` (this file)
- Repository organization
- File purposes and relationships
- Code flow diagrams

### Configuration Files

#### `requirements.txt`
Dependencies:
- `torch>=2.0.0` - PyTorch for deep learning
- `torchvision>=0.15.0` - For CIFAR-10 dataset
- `numpy>=1.24.0` - Numerical operations
- `matplotlib>=3.7.0` - Plotting and visualization
- `tqdm>=4.65.0` - Progress bars

#### `setup.py`
- Package installation configuration
- Makes code installable via pip
- Defines package metadata

#### `.gitignore`
Ignores:
- Python cache files (`__pycache__/`)
- Virtual environments (`venv/`)
- Data directories (`data/`)
- Model checkpoints (`*.pth`)
- IDE files (`.vscode/`, `.idea/`)

### Source Code (`src/`)

#### `src/models/simple_nn.py`
**Purpose**: Neural network architectures

**Classes**:
- `SimpleNN`: Fully connected network for CIFAR-10
  - Input: Flattened 32×32×3 images (3072 features)
  - Hidden: [128, 64, 32]
  - Output: 10 classes
  
- `CNNModel`: Convolutional network (more advanced)
  - 3 conv layers with batch normalization
  - Max pooling and dropout
  - 2 fully connected layers

**Functions**:
- `create_model()`: Factory function for model creation

**Key Methods**:
- `get_weights()`: Extract model parameters
- `set_weights()`: Load model parameters
- `get_gradients()`: Extract gradients

#### `src/algorithms/fedavg.py`
**Purpose**: Federated Averaging algorithm implementation

**Classes**:
- `FedAvgClient`: Client-side operations
  - Local training for E epochs
  - Returns updated weights
  
- `FedAvgServer`: Server-side operations
  - Maintains global model
  - Aggregates client weights
  - Evaluates on test data
  
- `FedAvgTrainer`: Complete training orchestration
  - Manages communication rounds
  - Client selection
  - History tracking

**Algorithm Flow**:
```
1. Server sends global weights to clients
2. Each client trains locally for E epochs
3. Clients return updated weights
4. Server aggregates weights (weighted average)
5. Repeat
```

#### `src/algorithms/fedsgd.py`
**Purpose**: Federated SGD algorithm implementation

**Classes**:
- `FedSGDClient`: Client-side operations
  - Computes gradients on local data
  - Returns gradients (not weights)
  
- `FedSGDServer`: Server-side operations
  - Aggregates gradients
  - Applies gradients to update model
  
- `FedSGDTrainer`: Training orchestration

**Algorithm Flow**:
```
1. Server sends global weights to clients
2. Each client computes gradients
3. Clients return gradients
4. Server aggregates and applies gradients
5. Repeat
```

#### `src/data/data_distribution.py`
**Purpose**: Data loading and distribution

**Class**: `DataDistributor`

**Key Methods**:
- `create_iid_distribution()`: Creates balanced data across clients
  - Each client gets equal share
  - Uniform class distribution
  
- `create_non_iid_distribution()`: Creates heterogeneous data
  - Uses Dirichlet distribution
  - Parameter `alpha` controls skewness
  - Lower alpha = more skewed
  
- `create_pathological_non_iid_distribution()`: Extreme heterogeneity
  - Each client has only K classes
  
- `visualize_distribution()`: Plots data distribution

**Function**: `create_common_dataset()`
- Creates small balanced dataset for fine-tuning
- Used to mitigate Non-IID issues

#### `src/utils/aggregation.py`
**Purpose**: Aggregation strategies for combining client updates

**Functions**:
- `aggregate_weights()`: Weighted average of model weights
- `aggregate_gradients()`: Weighted average of gradients
- `aggregate_with_momentum()`: Aggregation with server momentum
- `fedprox_aggregate()`: FedProx algorithm aggregation
- `median_aggregation()`: Byzantine-robust median
- `trimmed_mean_aggregation()`: Robust to outliers
- `krum_aggregation()`: Byzantine-robust selection

**Class**: `AggregationStrategy`
- Wrapper for different strategies
- Easy switching between methods

### Examples (`examples/`)

#### `quick_start.py`
**Best for**: Beginners, first-time users
**Time**: ~5 minutes
**Covers**:
- Basic workflow
- Minimal code
- Clear output

**Code Structure**:
```python
1. Configure (clients, rounds)
2. Prepare data
3. Create model
4. Initialize trainer
5. Train
6. Show results
```

#### `complete_federated_learning_demo.py`
**Best for**: Understanding all concepts
**Time**: ~30 minutes
**Covers**:
- IID data training
- Non-IID data training
- FedSGD algorithm
- Traditional centralized training
- Random client selection
- Comprehensive comparisons

**Experiments**:
1. FedAvg with IID data
2. FedAvg with Non-IID data
3. FedSGD with IID data
4. Traditional training
5. Random client selection

**Output**:
- Multiple comparison plots
- Performance metrics table
- Key insights

#### `noniid_handling_demo.py`
**Best for**: Learning about data heterogeneity
**Time**: ~15 minutes
**Covers**:
- Non-IID challenge demonstration
- Fine-tuning on common dataset
- Before/after comparison
- Visualization

**Experiments**:
1. Non-IID without fine-tuning
2. Non-IID with fine-tuning
3. IID for reference

### Tests (`tests/`)

#### `test_federated_learning.py`
**Purpose**: Comprehensive test suite

**Test Classes**:
- `TestModels`: Model creation and operations
- `TestAggregation`: Aggregation functions
- `TestFedAvg`: FedAvg algorithm
- `TestFedSGD`: FedSGD algorithm
- `TestDataDistribution`: Data loading

**Run Tests**:
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_federated_learning.py::TestFedAvg

# With coverage
pytest --cov=src tests/
```

## Code Flow Diagrams

### FedAvg Training Flow

```
┌─────────────────────────────────────────┐
│         FedAvgTrainer.train()           │
│    (Orchestrates entire process)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│      For each communication round:       │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  1. Server gets global weights           │
│     server.get_weights()                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  2. For each client:                     │
│     a) Set global weights                │
│        client.set_weights(global_w)      │
│     b) Train locally                     │
│        weights, loss = client.train()    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  3. Server aggregates weights            │
│     aggregated = server.aggregate(...)   │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  4. Update global model                  │
│     server.set_weights(aggregated)       │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  5. Evaluate on test set                 │
│     acc, loss = server.evaluate()        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  6. Record history                       │
│     history['test_accuracy'].append(acc) │
└──────────────────────────────────────────┘
```

### Data Distribution Flow

```
┌─────────────────────────────────────────┐
│      DataDistributor.__init__()         │
│   (Initialize with dataset name)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│       load_dataset(train=True)           │
│   (Download/load CIFAR-10 or MNIST)     │
└──────────────┬───────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
┌──────────┐    ┌──────────────┐
│   IID    │    │   Non-IID    │
└────┬─────┘    └──────┬───────┘
     │                 │
     ▼                 ▼
┌─────────┐    ┌────────────────┐
│ Shuffle │    │ Dirichlet      │
│ & Split │    │ Distribution   │
└────┬────┘    └────┬───────────┘
     │              │
     └──────┬───────┘
            ▼
┌──────────────────────────────┐
│  Create DataLoaders          │
│  (one per client + test)     │
└──────────────────────────────┘
```

## Key Relationships

### Model ↔ Client ↔ Server
```
Model (SimpleNN/CNNModel)
   ↕ weights
Client (FedAvgClient/FedSGDClient)
   ↕ updates
Server (FedAvgServer/FedSGDServer)
   ↕ global model
Trainer (FedAvgTrainer/FedSGDTrainer)
```

### Data Flow
```
Dataset (CIFAR-10)
   ↓
DataDistributor
   ↓
Client DataLoaders (split by client)
   ↓
Client Training
   ↓
Model Updates
   ↓
Server Aggregation
   ↓
Global Model
```

## Extending the Code

### Add a New Aggregation Strategy

1. Add function to `src/utils/aggregation.py`:
```python
def my_custom_aggregation(client_weights, **kwargs):
    # Your aggregation logic
    return aggregated_weights
```

2. Update `AggregationStrategy.aggregate()`:
```python
elif self.strategy == 'my_custom':
    return my_custom_aggregation(client_weights, **self.kwargs)
```

### Add a New Dataset

1. Update `DataDistributor._get_transforms()`:
```python
elif self.dataset_name == 'my_dataset':
    return transforms.Compose([...])
```

2. Update `DataDistributor.load_dataset()`:
```python
elif self.dataset_name == 'my_dataset':
    return datasets.MyDataset(...)
```

### Add a New Algorithm

1. Create `src/algorithms/my_algorithm.py`
2. Implement `MyAlgorithmClient`, `MyAlgorithmServer`, `MyAlgorithmTrainer`
3. Follow the pattern from FedAvg/FedSGD
4. Add to `src/algorithms/__init__.py`

## Performance Tips

### Memory Optimization
- Use smaller batch sizes
- Reduce number of clients
- Use gradient checkpointing
- Clear GPU cache: `torch.cuda.empty_cache()`

### Speed Optimization
- Use GPU if available
- Increase batch size (if memory allows)
- Reduce local epochs
- Use DataLoader with `num_workers>0`

### Quality Optimization
- More communication rounds
- More local epochs
- Better hyperparameter tuning
- Use learning rate scheduling

## Common Patterns

### Pattern 1: Basic Training
```python
distributor = DataDistributor('cifar10')
client_loaders, test_loader = distributor.create_iid_distribution(8)
model = SimpleNN()
trainer = FedAvgTrainer(model, client_loaders, test_loader)
history = trainer.train(num_rounds=10)
```

### Pattern 2: Non-IID with Fine-tuning
```python
client_loaders, test_loader = distributor.create_non_iid_distribution(8)
common_loader = create_common_dataset(num_samples=500)
# Use custom trainer with fine-tuning (see noniid_handling_demo.py)
```

### Pattern 3: Comparison Study
```python
histories = {}
for method in ['fedavg', 'fedsgd']:
    trainer = create_trainer(method)
    histories[method] = trainer.train(10)
plot_comparison(histories)
```

---

This structure provides a solid foundation for federated learning research and development. Each component is modular and can be extended independently.