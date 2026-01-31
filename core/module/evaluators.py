"""
Federated Learning Evaluators Module

Functions for evaluating global and local models with detailed metrics.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix
)
from typing import Dict, Tuple, List, Optional
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate model with comprehensive metrics for IDS.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        verbose: If True, print detailed classification report
        
    Returns:
        Dictionary with metrics:
            - loss: Average cross-entropy loss
            - accuracy: Overall accuracy (%)
            - precision: Weighted precision (%)
            - recall: Weighted recall (%)
            - f1: Weighted F1-score (%)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) * 100
    
    if verbose:
        print("\n--- DETAILED CLASSIFICATION REPORT ---")
        print(classification_report(all_labels, all_predictions, zero_division=0))
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_per_class(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate model with per-class metrics.
    Useful for analyzing performance on specific attack types.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: Optional list of class names for labeling
        
    Returns:
        Dictionary mapping class_id to metrics dict
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate per-class metrics
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    unique_classes = np.unique(all_labels)
    per_class_metrics = {}
    
    for i, class_id in enumerate(unique_classes):
        class_name = class_names[class_id] if class_names else f"Class {class_id}"
        per_class_metrics[class_id] = {
            'name': class_name,
            'precision': precision_per_class[i] * 100,
            'recall': recall_per_class[i] * 100,
            'f1': f1_per_class[i] * 100,
            'support': np.sum(np.array(all_labels) == class_id)
        }
    
    return per_class_metrics


def get_confusion_matrix(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    normalize: bool = False
) -> np.ndarray:
    """
    Generate confusion matrix for model predictions.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        normalize: If True, normalize confusion matrix by true labels
        
    Returns:
        Confusion matrix (numpy array)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def compare_metrics(metrics1: Dict[str, float], metrics2: Dict[str, float]) -> Dict[str, float]:
    """
    Compare two sets of metrics (e.g., before/after aggregation).
    
    Args:
        metrics1: First metrics dict
        metrics2: Second metrics dict
        
    Returns:
        Dictionary with differences
    """
    diff = {}
    for key in metrics1.keys():
        if key in metrics2:
            diff[key] = metrics2[key] - metrics1[key]
    return diff


def format_metrics(metrics: Dict[str, float], round_digits: int = 2) -> str:
    """
    Format metrics dictionary as a human-readable string.
    
    Args:
        metrics: Metrics dictionary
        round_digits: Number of decimal places
        
    Returns:
        Formatted string
        
    Example:
        >>> metrics = {'accuracy': 85.23, 'f1': 82.15}
        >>> print(format_metrics(metrics))
        "Acc: 85.23% | F1: 82.15%"
    """
    parts = []
    
    # Order metrics for display
    order = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    
    for key in order:
        if key in metrics:
            if key == 'loss':
                parts.append(f"Loss: {metrics[key]:.{round_digits}f}")
            else:
                short_name = {
                    'accuracy': 'Acc',
                    'precision': 'Prec',
                    'recall': 'Rec',
                    'f1': 'F1'
                }.get(key, key)
                parts.append(f"{short_name}: {metrics[key]:.{round_digits}f}%")
    
    return " | ".join(parts)
