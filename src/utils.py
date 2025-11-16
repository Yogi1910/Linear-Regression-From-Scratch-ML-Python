"""
Utility functions for linear regression implementation.
"""

import numpy as np
from typing import Tuple, Optional


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Add intercept column (column of ones) to feature matrix.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
    
    Returns:
        Feature matrix with intercept column of shape (n_samples, n_features + 1)
    """
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate([intercept, X], axis=1)


def normalize_features(X: np.ndarray, method: str = 'standardize') -> Tuple[np.ndarray, dict]:
    """
    Normalize features using standardization or min-max scaling.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        method: Normalization method ('standardize' or 'minmax')
    
    Returns:
        Tuple of (normalized_X, normalization_params)
    """
    X = np.asarray(X)
    
    if method == 'standardize':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        X_normalized = (X - mean) / std
        params = {'method': 'standardize', 'mean': mean, 'std': std}
        
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        X_normalized = (X - min_val) / range_val
        params = {'method': 'minmax', 'min': min_val, 'max': max_val}
    
    else:
        raise ValueError("Method must be 'standardize' or 'minmax'")
    
    return X_normalized, params


def denormalize_features(X_normalized: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize features using stored normalization parameters.
    
    Args:
        X_normalized: Normalized feature matrix
        params: Normalization parameters from normalize_features
    
    Returns:
        Original scale feature matrix
    """
    if params['method'] == 'standardize':
        return X_normalized * params['std'] + params['mean']
    elif params['method'] == 'minmax':
        return X_normalized * (params['max'] - params['min']) + params['min']
    else:
        raise ValueError("Unknown normalization method")


def train_test_split(X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, 
                    random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of dataset to include in test split
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features up to specified degree.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        degree: Maximum degree of polynomial features
    
    Returns:
        Polynomial feature matrix
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n_samples, n_features = X.shape
    
    # Start with original features
    poly_features = [X]
    
    # Add polynomial terms
    for d in range(2, degree + 1):
        for i in range(n_features):
            poly_features.append((X[:, i] ** d).reshape(-1, 1))
    
    return np.concatenate(poly_features, axis=1)


def compute_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix for features.
    
    Args:
        X: Feature matrix
    
    Returns:
        Correlation matrix
    """
    return np.corrcoef(X.T)


def detect_multicollinearity(X: np.ndarray, threshold: float = 0.8) -> dict:
    """
    Detect multicollinearity in features using correlation matrix.
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold for multicollinearity detection
    
    Returns:
        Dictionary with correlation matrix and highly correlated pairs
    """
    corr_matrix = compute_correlation_matrix(X)
    n_features = X.shape[1]
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'threshold': threshold
    }