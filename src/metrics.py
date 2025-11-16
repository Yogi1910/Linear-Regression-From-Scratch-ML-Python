"""
Evaluation metrics for regression models.
"""

import numpy as np
from typing import Union


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        MSE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        MAE value
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        R² score
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_features: int) -> float:
    """
    Calculate Adjusted R-squared.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        n_features: Number of features in the model
    
    Returns:
        Adjusted R² score
    """
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)
    
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
    
    Returns:
        MAPE value as percentage
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = None) -> dict:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        n_features: Number of features (for adjusted R²)
    
    Returns:
        Dictionary containing all regression metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }
    
    if n_features is not None:
        metrics['adjusted_r2'] = adjusted_r2_score(y_true, y_pred, n_features)
    
    return metrics