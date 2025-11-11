"""Utility functions for federated learning."""

from src.utils.aggregation import (
    aggregate_weights,
    aggregate_gradients,
    AggregationStrategy
)

__all__ = [
    'aggregate_weights',
    'aggregate_gradients',
    'AggregationStrategy'
]