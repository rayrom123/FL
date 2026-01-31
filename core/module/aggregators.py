"""
Federated Learning Aggregators Module

Utility functions for aggregating client updates.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def calculate_sample_weights(client_samples: List[int]) -> List[float]:
    """
    Calculate normalized weights based on client sample counts.
    
    Args:
        client_samples: List of sample counts from each client
        
    Returns:
        List of normalized weights (sum to 1.0)
        
    Example:
        >>> calculate_sample_weights([100, 200, 300])
        [0.1667, 0.3333, 0.5000]
    """
    total_samples = sum(client_samples)
    return [n / total_samples for n in client_samples]


def weighted_average(client_weights: List[Dict], weights: List[float]) -> Dict:
    """
    Compute weighted average of client weights.
    
    Args:
        client_weights: List of state_dicts from clients
        weights: List of weights for each client (should sum to 1.0)
        
    Returns:
        Averaged weights (state_dict)
    """
    global_weights = {}
    
    for key in client_weights[0].keys():
        global_weights[key] = torch.zeros_like(client_weights[0][key])
    
    for i, client_weight in enumerate(client_weights):
        for key in global_weights.keys():
            if torch.is_floating_point(client_weight[key]):
                global_weights[key] += client_weight[key] * weights[i]
            else:
                # Keep first non-floating value
                if i == 0:
                    global_weights[key] = client_weight[key]
    
    return global_weights


def median_aggregation(client_weights: List[Dict]) -> Dict:
    """
    Robust aggregation using element-wise median.
    Useful for defending against Byzantine attacks.
    
    Args:
        client_weights: List of state_dicts from clients
        
    Returns:
        Median aggregated weights (state_dict)
    """
    global_weights = {}
    
    for key in client_weights[0].keys():
        if torch.is_floating_point(client_weights[0][key]):
            # Stack all client weights for this parameter
            stacked = torch.stack([w[key] for w in client_weights])
            # Compute element-wise median
            global_weights[key] = torch.median(stacked, dim=0)[0]
        else:
            # Keep first non-floating value
            global_weights[key] = client_weights[0][key]
    
    return global_weights


def trimmed_mean_aggregation(client_weights: List[Dict], trim_ratio: float = 0.2) -> Dict:
    """
    Robust aggregation using trimmed mean.
    Removes extreme values before averaging.
    
    Args:
        client_weights: List of state_dicts from clients
        trim_ratio: Fraction of values to trim from each end (0.0 to 0.5)
        
    Returns:
        Trimmed mean aggregated weights (state_dict)
    """
    assert 0 <= trim_ratio < 0.5, "trim_ratio must be in [0, 0.5)"
    
    global_weights = {}
    num_clients = len(client_weights)
    trim_count = int(num_clients * trim_ratio)
    
    for key in client_weights[0].keys():
        if torch.is_floating_point(client_weights[0][key]):
            # Stack all client weights
            stacked = torch.stack([w[key] for w in client_weights])
            # Sort along client dimension
            sorted_weights = torch.sort(stacked, dim=0)[0]
            # Trim and average
            if trim_count > 0:
                trimmed = sorted_weights[trim_count:-trim_count]
            else:
                trimmed = sorted_weights
            global_weights[key] = torch.mean(trimmed, dim=0)
        else:
            global_weights[key] = client_weights[0][key]
    
    return global_weights


def cosine_similarity(weights1: Dict, weights2: Dict) -> float:
    """
    Calculate cosine similarity between two weight dictionaries.
    Useful for analyzing client drift.
    
    Args:
        weights1, weights2: State dicts to compare
        
    Returns:
        Cosine similarity [-1, 1]
    """
    vec1 = []
    vec2 = []
    
    for key in weights1.keys():
        if torch.is_floating_point(weights1[key]):
            vec1.append(weights1[key].flatten())
            vec2.append(weights2[key].flatten())
    
    vec1 = torch.cat(vec1)
    vec2 = torch.cat(vec2)
    
    cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return cos_sim.item()
