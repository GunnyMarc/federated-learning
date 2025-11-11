"""
Aggregation Utilities for Federated Learning

Implements various aggregation strategies for combining client updates.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


def aggregate_weights(
        client_weights: List[List[torch.Tensor]],
        client_sizes: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Aggregate client model weights using weighted averaging.

    This implements the core aggregation step in FedAvg algorithm.

    Args:
        client_weights (list): List of weight lists from each client
        client_sizes (list, optional): Number of samples each client used.
                                       If None, use simple averaging.

    Returns:
        list: Aggregated weights
    """
    num_clients = len(client_weights)

    # If no client sizes provided, use uniform weights
    if client_sizes is None:
        client_sizes = [1] * num_clients

    # Calculate total samples
    total_samples = sum(client_sizes)

    # Initialize aggregated weights
    aggregated_weights = []

    # Aggregate each layer
    num_layers = len(client_weights[0])
    for layer_idx in range(num_layers):
        # Weighted sum of this layer across all clients
        layer_sum = torch.zeros_like(client_weights[0][layer_idx])

        for client_idx, weights in enumerate(client_weights):
            weight = client_sizes[client_idx] / total_samples
            layer_sum += weights[layer_idx] * weight

        aggregated_weights.append(layer_sum)

    return aggregated_weights


def aggregate_gradients(
        client_gradients: List[List[torch.Tensor]],
        client_sizes: Optional[List[int]] = None
) -> List[torch.Tensor]:
    """
    Aggregate client gradients using weighted averaging.

    This implements the core aggregation step in FedSGD algorithm.

    Args:
        client_gradients (list): List of gradient lists from each client
        client_sizes (list, optional): Number of samples each client used

    Returns:
        list: Aggregated gradients
    """
    # Gradient aggregation is the same as weight aggregation
    return aggregate_weights(client_gradients, client_sizes)


def aggregate_with_momentum(
        client_weights: List[List[torch.Tensor]],
        global_weights: List[torch.Tensor],
        client_sizes: Optional[List[int]] = None,
        momentum: float = 0.9
) -> List[torch.Tensor]:
    """
    Aggregate client weights with server-side momentum.

    This can help accelerate convergence and smooth updates.

    Args:
        client_weights (list): List of weight lists from each client
        global_weights (list): Current global model weights
        client_sizes (list, optional): Number of samples each client used
        momentum (float): Momentum coefficient

    Returns:
        list: Aggregated weights with momentum
    """
    # Get standard aggregation
    aggregated = aggregate_weights(client_weights, client_sizes)

    # Apply momentum
    result = []
    for global_w, agg_w in zip(global_weights, aggregated):
        new_w = momentum * global_w + (1 - momentum) * agg_w
        result.append(new_w)

    return result


def fedprox_aggregate(
        client_weights: List[List[torch.Tensor]],
        global_weights: List[torch.Tensor],
        client_sizes: Optional[List[int]] = None,
        mu: float = 0.01
) -> List[torch.Tensor]:
    """
    Aggregate using FedProx algorithm.

    FedProx adds a proximal term to keep local models close to global model,
    which helps with heterogeneous data.

    Args:
        client_weights (list): List of weight lists from each client
        global_weights (list): Current global model weights
        client_sizes (list, optional): Number of samples each client used
        mu (float): Proximal term coefficient

    Returns:
        list: Aggregated weights
    """
    # Standard aggregation
    aggregated = aggregate_weights(client_weights, client_sizes)

    # Add proximal term
    result = []
    for global_w, agg_w in zip(global_weights, aggregated):
        new_w = agg_w + mu * (global_w - agg_w)
        result.append(new_w)

    return result


def median_aggregation(
        client_weights: List[List[torch.Tensor]]
) -> List[torch.Tensor]:
    """
    Aggregate using coordinate-wise median.

    More robust to Byzantine (malicious) clients than mean aggregation.

    Args:
        client_weights (list): List of weight lists from each client

    Returns:
        list: Aggregated weights using median
    """
    num_layers = len(client_weights[0])
    aggregated_weights = []

    for layer_idx in range(num_layers):
        # Stack all client weights for this layer
        layer_weights = torch.stack([
            client_weights[i][layer_idx] for i in range(len(client_weights))
        ])

        # Compute median along client dimension
        median_weights = torch.median(layer_weights, dim=0)[0]
        aggregated_weights.append(median_weights)

    return aggregated_weights


def trimmed_mean_aggregation(
        client_weights: List[List[torch.Tensor]],
        trim_ratio: float = 0.1
) -> List[torch.Tensor]:
    """
    Aggregate using trimmed mean.

    Removes extreme values before averaging, providing robustness to outliers.

    Args:
        client_weights (list): List of weight lists from each client
        trim_ratio (float): Fraction of values to trim from each end

    Returns:
        list: Aggregated weights using trimmed mean
    """
    num_clients = len(client_weights)
    num_layers = len(client_weights[0])

    # Calculate number of clients to trim
    num_trim = int(num_clients * trim_ratio)

    aggregated_weights = []

    for layer_idx in range(num_layers):
        # Stack all client weights for this layer
        layer_weights = torch.stack([
            client_weights[i][layer_idx] for i in range(num_clients)
        ])

        # Sort along client dimension
        sorted_weights = torch.sort(layer_weights, dim=0)[0]

        # Trim and average
        if num_trim > 0:
            trimmed_weights = sorted_weights[num_trim:-num_trim]
        else:
            trimmed_weights = sorted_weights

        mean_weights = torch.mean(trimmed_weights, dim=0)
        aggregated_weights.append(mean_weights)

    return aggregated_weights


def krum_aggregation(
        client_weights: List[List[torch.Tensor]],
        num_byzantine: int = 0
) -> List[torch.Tensor]:
    """
    Aggregate using Krum algorithm.

    Selects the most representative client update based on distances to other clients.
    Robust to Byzantine attacks.

    Args:
        client_weights (list): List of weight lists from each client
        num_byzantine (int): Expected number of Byzantine clients

    Returns:
        list: Selected client weights (most representative)
    """
    num_clients = len(client_weights)

    # Compute pairwise distances
    distances = torch.zeros(num_clients, num_clients)

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 0
            for layer_i, layer_j in zip(client_weights[i], client_weights[j]):
                dist += torch.sum((layer_i - layer_j) ** 2)

            distances[i, j] = dist
            distances[j, i] = dist

    # Compute scores (sum of distances to closest neighbors)
    num_closest = num_clients - num_byzantine - 2
    scores = []

    for i in range(num_clients):
        closest_distances = torch.topk(
            distances[i],
            num_closest,
            largest=False
        )[0]
        scores.append(torch.sum(closest_distances).item())

    # Select client with minimum score
    selected_idx = np.argmin(scores)

    return client_weights[selected_idx]


class AggregationStrategy:
    """
    Wrapper class for different aggregation strategies.
    """

    def __init__(self, strategy='fedavg', **kwargs):
        """
        Initialize aggregation strategy.

        Args:
            strategy (str): Aggregation strategy name
            **kwargs: Additional parameters for specific strategies
        """
        self.strategy = strategy
        self.kwargs = kwargs

    def aggregate(
            self,
            client_weights: List[List[torch.Tensor]],
            global_weights: Optional[List[torch.Tensor]] = None,
            client_sizes: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Perform aggregation based on selected strategy.

        Args:
            client_weights (list): Client model weights
            global_weights (list, optional): Current global weights
            client_sizes (list, optional): Number of samples per client

        Returns:
            list: Aggregated weights
        """
        if self.strategy == 'fedavg':
            return aggregate_weights(client_weights, client_sizes)

        elif self.strategy == 'fedsgd':
            return aggregate_gradients(client_weights, client_sizes)

        elif self.strategy == 'momentum':
            if global_weights is None:
                raise ValueError("Global weights required for momentum aggregation")
            return aggregate_with_momentum(
                client_weights,
                global_weights,
                client_sizes,
                momentum=self.kwargs.get('momentum', 0.9)
            )

        elif self.strategy == 'fedprox':
            if global_weights is None:
                raise ValueError("Global weights required for FedProx aggregation")
            return fedprox_aggregate(
                client_weights,
                global_weights,
                client_sizes,
                mu=self.kwargs.get('mu', 0.01)
            )

        elif self.strategy == 'median':
            return median_aggregation(client_weights)

        elif self.strategy == 'trimmed_mean':
            return trimmed_mean_aggregation(
                client_weights,
                trim_ratio=self.kwargs.get('trim_ratio', 0.1)
            )

        elif self.strategy == 'krum':
            return krum_aggregation(
                client_weights,
                num_byzantine=self.kwargs.get('num_byzantine', 0)
            )

        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")


if __name__ == "__main__":
    # Test aggregation functions
    print("Testing aggregation functions...")

    # Create dummy client weights
    num_clients = 3
    layer_sizes = [(10, 5), (5, 2)]

    client_weights = []
    for _ in range(num_clients):
        weights = [torch.randn(size) for size in layer_sizes]
        client_weights.append(weights)

    # Test simple averaging
    print("\n1. Testing weighted averaging...")
    client_sizes = [100, 150, 200]
    aggregated = aggregate_weights(client_weights, client_sizes)
    print(f"Aggregated weight shapes: {[w.shape for w in aggregated]}")

    # Test median aggregation
    print("\n2. Testing median aggregation...")
    aggregated = median_aggregation(client_weights)
    print(f"Aggregated weight shapes: {[w.shape for w in aggregated]}")

    # Test aggregation strategy wrapper
    print("\n3. Testing AggregationStrategy wrapper...")
    strategy = AggregationStrategy('fedavg')
    aggregated = strategy.aggregate(client_weights, client_sizes=client_sizes)
    print(f"Aggregated weight shapes: {[w.shape for w in aggregated]}")

    print("\nAll aggregation tests passed!")