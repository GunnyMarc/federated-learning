"""
Tests for Federated Learning Algorithms
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.simple_nn import SimpleNN, CNNModel
from src.algorithms.fedavg import FedAvgTrainer, FedAvgClient, FedAvgServer
from src.algorithms.fedsgd import FedSGDTrainer, FedSGDClient, FedSGDServer
from src.utils.aggregation import (
    aggregate_weights,
    aggregate_gradients,
    AggregationStrategy
)
from src.data.data_distribution import DataDistributor


@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    return SimpleNN(input_size=100, hidden_sizes=[50, 20], num_classes=10)


@pytest.fixture
def dummy_data():
    """Create dummy datasets for testing."""
    # Create training data
    num_clients = 3
    samples_per_client = 50

    client_loaders = []
    for _ in range(num_clients):
        X = torch.randn(samples_per_client, 10, 10)  # Simplified input
        y = torch.randint(0, 10, (samples_per_client,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        client_loaders.append(loader)

    # Create test data
    X_test = torch.randn(30, 10, 10)
    y_test = torch.randint(0, 10, (30,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return client_loaders, test_loader


class TestModels:
    """Test model creation and functionality."""

    def test_simple_nn_creation(self):
        """Test SimpleNN model creation."""
        model = SimpleNN(input_size=100, num_classes=10)
        assert isinstance(model, nn.Module)

        # Test forward pass
        x = torch.randn(4, 10, 10)
        output = model(x)
        assert output.shape == (4, 10)

    def test_cnn_creation(self):
        """Test CNN model creation."""
        model = CNNModel(num_classes=10)
        assert isinstance(model, nn.Module)

        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        output = model(x)
        assert output.shape == (4, 10)

    def test_model_weight_operations(self):
        """Test getting and setting model weights."""
        model = SimpleNN(input_size=100, num_classes=10)

        # Get weights
        weights = model.get_weights()
        assert isinstance(weights, list)
        assert len(weights) > 0

        # Modify weights
        modified_weights = [w * 2 for w in weights]

        # Set weights
        model.set_weights(modified_weights)
        new_weights = model.get_weights()

        # Check if weights were updated
        assert torch.allclose(new_weights[0], modified_weights[0])


class TestAggregation:
    """Test aggregation functions."""

    def test_aggregate_weights_uniform(self):
        """Test weight aggregation with uniform weights."""
        # Create dummy weights
        num_clients = 3
        layer_sizes = [(10, 5), (5, 2)]

        client_weights = []
        for _ in range(num_clients):
            weights = [torch.randn(size) for size in layer_sizes]
            client_weights.append(weights)

        # Aggregate
        aggregated = aggregate_weights(client_weights)

        # Check output
        assert len(aggregated) == len(layer_sizes)
        for i, size in enumerate(layer_sizes):
            assert aggregated[i].shape == size

    def test_aggregate_weights_weighted(self):
        """Test weight aggregation with weighted averaging."""
        num_clients = 3
        layer_sizes = [(10, 5)]

        client_weights = []
        for _ in range(num_clients):
            weights = [torch.ones(layer_sizes[0])]
            client_weights.append(weights)

        client_sizes = [100, 200, 300]  # Different client sizes

        # Aggregate
        aggregated = aggregate_weights(client_weights, client_sizes)

        # Result should still be ones (since all clients have ones)
        assert torch.allclose(aggregated[0], torch.ones(layer_sizes[0]))

    def test_aggregation_strategy(self):
        """Test AggregationStrategy wrapper."""
        num_clients = 3
        layer_sizes = [(10, 5)]

        client_weights = []
        for _ in range(num_clients):
            weights = [torch.randn(layer_sizes[0])]
            client_weights.append(weights)

        # Test different strategies
        strategy = AggregationStrategy('fedavg')
        result = strategy.aggregate(client_weights)
        assert len(result) == 1

        strategy = AggregationStrategy('median')
        result = strategy.aggregate(client_weights)
        assert len(result) == 1


class TestFedAvg:
    """Test FedAvg algorithm."""

    def test_fedavg_client_creation(self, dummy_model, dummy_data):
        """Test FedAvg client creation."""
        client_loaders, _ = dummy_data

        client = FedAvgClient(
            model=dummy_model,
            train_loader=client_loaders[0],
            learning_rate=0.01
        )

        assert client.get_dataset_size() > 0

    def test_fedavg_client_training(self, dummy_model, dummy_data):
        """Test FedAvg client local training."""
        client_loaders, _ = dummy_data

        client = FedAvgClient(
            model=dummy_model,
            train_loader=client_loaders[0],
            learning_rate=0.01
        )

        # Train for 1 epoch
        weights, loss = client.train(epochs=1)

        assert isinstance(weights, list)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_fedavg_server_creation(self, dummy_model):
        """Test FedAvg server creation."""
        server = FedAvgServer(model=dummy_model)

        weights = server.get_weights()
        assert isinstance(weights, list)

    def test_fedavg_server_aggregation(self, dummy_model):
        """Test FedAvg server aggregation."""
        server = FedAvgServer(model=dummy_model)

        # Create dummy client weights
        num_clients = 3
        client_weights = [server.get_weights() for _ in range(num_clients)]
        client_sizes = [100, 150, 200]

        # Aggregate
        aggregated = server.aggregate_weights(client_weights, client_sizes)

        assert isinstance(aggregated, list)
        assert len(aggregated) == len(client_weights[0])

    def test_fedavg_trainer(self, dummy_model, dummy_data):
        """Test complete FedAvg training."""
        client_loaders, test_loader = dummy_data

        trainer = FedAvgTrainer(
            model=dummy_model,
            client_loaders=client_loaders,
            test_loader=test_loader,
            learning_rate=0.01
        )

        # Train for 2 rounds
        history = trainer.train(num_rounds=2, verbose=False)

        assert 'test_accuracy' in history
        assert 'test_loss' in history
        assert len(history['test_accuracy']) == 2


class TestFedSGD:
    """Test FedSGD algorithm."""

    def test_fedsgd_client_creation(self, dummy_model, dummy_data):
        """Test FedSGD client creation."""
        client_loaders, _ = dummy_data

        client = FedSGDClient(
            model=dummy_model,
            train_loader=client_loaders[0]
        )

        assert client.get_dataset_size() > 0

    def test_fedsgd_client_gradient_computation(self, dummy_model, dummy_data):
        """Test FedSGD client gradient computation."""
        client_loaders, _ = dummy_data

        client = FedSGDClient(
            model=dummy_model,
            train_loader=client_loaders[0]
        )

        # Compute gradients
        gradients, loss = client.compute_gradients()

        assert isinstance(gradients, list)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_fedsgd_server_creation(self, dummy_model):
        """Test FedSGD server creation."""
        server = FedSGDServer(model=dummy_model, learning_rate=0.01)

        weights = server.get_weights()
        assert isinstance(weights, list)

    def test_fedsgd_trainer(self, dummy_model, dummy_data):
        """Test complete FedSGD training."""
        client_loaders, test_loader = dummy_data

        trainer = FedSGDTrainer(
            model=dummy_model,
            client_loaders=client_loaders,
            test_loader=test_loader,
            learning_rate=0.01
        )

        # Train for 2 rounds
        history = trainer.train(num_rounds=2, verbose=False)

        assert 'test_accuracy' in history
        assert 'test_loss' in history
        assert len(history['test_accuracy']) == 2


class TestDataDistribution:
    """Test data distribution functionality."""

    def test_data_distributor_creation(self):
        """Test DataDistributor creation."""
        distributor = DataDistributor('cifar10')
        assert distributor.dataset_name == 'cifar10'

    def test_iid_distribution_creation(self):
        """Test IID data distribution creation."""
        distributor = DataDistributor('cifar10')

        # This will download CIFAR-10 if not already present
        client_loaders, test_loader = distributor.create_iid_distribution(
            num_clients=4,
            batch_size=32
        )

        assert len(client_loaders) == 4
        assert isinstance(test_loader, DataLoader)

    def test_non_iid_distribution_creation(self):
        """Test Non-IID data distribution creation."""
        distributor = DataDistributor('cifar10')

        client_loaders, test_loader = distributor.create_non_iid_distribution(
            num_clients=4,
            batch_size=32,
            alpha=0.5
        )

        assert len(client_loaders) == 4
        assert isinstance(test_loader, DataLoader)


def test_end_to_end_fedavg():
    """End-to-end test of FedAvg training."""
    # Create simple dummy data
    num_clients = 2
    client_loaders = []

    for _ in range(num_clients):
        X = torch.randn(20, 10, 10)
        y = torch.randint(0, 5, (20,))
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        client_loaders.append(loader)

    X_test = torch.randn(10, 10, 10)
    y_test = torch.randint(0, 5, (10,))
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=10)

    # Create model and trainer
    model = SimpleNN(input_size=100, num_classes=5)
    trainer = FedAvgTrainer(
        model=model,
        client_loaders=client_loaders,
        test_loader=test_loader
    )

    # Train
    history = trainer.train(num_rounds=2, verbose=False)

    # Verify training completed
    assert len(history['test_accuracy']) == 2
    final_model = trainer.get_global_model()
    assert isinstance(final_model, nn.Module)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])