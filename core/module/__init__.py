"""
Federated Learning Module

This module provides modular components for Federated Learning:
- Strategies: FedAvg, FedProx implementations
- Aggregators: Various aggregation methods
- Evaluators: Model evaluation with detailed metrics

Example usage:
    >>> from module import create_strategy, evaluate_model
    >>> 
    >>> # Create strategy
    >>> strategy = create_strategy('FedAvg')
    >>> 
    >>> # Evaluate model
    >>> metrics = evaluate_model(model, test_loader, device)
"""

from .strategies import (
    FederatedStrategy,
    FedAvgStrategy,
    FedProxStrategy,
    create_strategy
)

from .aggregators import (
    calculate_sample_weights,
    weighted_average,
    median_aggregation,
    trimmed_mean_aggregation,
    cosine_similarity
)

from .evaluators import (
    evaluate_model,
    evaluate_per_class,
    get_confusion_matrix,
    compare_metrics,
    format_metrics
)

__all__ = [
    # Strategies
    'FederatedStrategy',
    'FedAvgStrategy',
    'FedProxStrategy',
    'create_strategy',
    
    # Aggregators
    'calculate_sample_weights',
    'weighted_average',
    'median_aggregation',
    'trimmed_mean_aggregation',
    'cosine_similarity',
    
    # Evaluators
    'evaluate_model',
    'evaluate_per_class',
    'get_confusion_matrix',
    'compare_metrics',
    'format_metrics',
]

__version__ = '1.0.0'
