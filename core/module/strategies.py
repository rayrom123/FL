"""
Federated Learning Strategies Module

Implements various FL aggregation strategies:
- FedAvg (Federated Averaging)
- FedProx (Federated Proximal)
"""

import torch
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class FederatedStrategy(ABC):
    """Base class for federated learning strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """
        Aggregate client weights into global weights.
        
        Args:
            client_weights: List of state_dicts from clients
            client_samples: List of sample counts from each client
            
        Returns:
            Aggregated global weights (state_dict)
        """
        pass


class FedAvgStrategy(FederatedStrategy):
    """
    Federated Averaging (FedAvg) Strategy
    
    Reference: McMahan et al., "Communication-Efficient Learning 
    of Deep Networks from Decentralized Data", AISTATS 2017
    
    Aggregation formula:
        w_{t+1} = Σ (n_k / n) * w_{t+1}^k
    
    where:
        - n_k: number of samples at client k
        - n: total samples across all clients
        - w_{t+1}^k: updated weights from client k
    """
    
    def __init__(self):
        super().__init__("FedAvg")
    
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """
        Weighted average aggregation based on client sample counts.
        """
        total_samples = sum(client_samples)
        global_weights = copy.deepcopy(client_weights[0])
        
        # Initialize global weights to zero
        for key in global_weights.keys():
            global_weights[key] = torch.zeros_like(global_weights[key])
        
        # Weighted aggregation
        for i in range(len(client_weights)):
            weight = client_samples[i] / total_samples
            for key in global_weights.keys():
                if torch.is_floating_point(client_weights[i][key]):
                    global_weights[key] += client_weights[i][key] * weight
                else:
                    # For non-floating point parameters (e.g., batch norm running stats)
                    if i == 0:
                        global_weights[key] = client_weights[i][key]
                    else:
                        global_weights[key] = torch.max(global_weights[key], client_weights[i][key])
        
        return global_weights


class FedProxStrategy(FederatedStrategy):
    """
    Federated Proximal (FedProx) Strategy
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks", 
    MLSys 2020
    
    Note: FedProx uses the same server-side aggregation as FedAvg.
    The difference is in the client-side loss function (proximal term).
    The proximal term is handled in client_app.py during training.
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Args:
            mu: Proximal term coefficient (typically 0.001 - 0.1)
        """
        super().__init__("FedProx")
        self.mu = mu
    
    def aggregate(self, client_weights: List[Dict], client_samples: List[int]) -> Dict:
        """
        FedProx uses FedAvg aggregation on server side.
        The proximal term (mu) is only used during client training.
        """
        # Delegate to FedAvg
        fedavg = FedAvgStrategy()
        return fedavg.aggregate(client_weights, client_samples)


# Factory function for easy strategy creation
def create_strategy(algorithm: str, **kwargs) -> FederatedStrategy:
    """
    Factory function to create FL strategies.
    
    Args:
        algorithm: Strategy name ('FedAvg', 'FedProx')
        **kwargs: Additional parameters for the strategy
        
    Returns:
        FederatedStrategy instance
        
    Example:
        >>> strategy = create_strategy('FedAvg')
        >>> strategy = create_strategy('FedProx', mu=0.01)
    """
    strategies = {
        'FedAvg': FedAvgStrategy,
        'FedProx': FedProxStrategy,
    }
    
    if algorithm not in strategies:
        raise ValueError(f"Unknown strategy: {algorithm}. Available: {list(strategies.keys())}")
    
    return strategies[algorithm](**kwargs)
