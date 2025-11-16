"""
Regularized Linear Regression implementations (Ridge and Lasso).
"""

import numpy as np
from typing import Optional
from .utils import add_intercept


class RidgeRegression:
    """
    Ridge Regression (L2 regularization) implementation.
    
    Adds L2 penalty term to the cost function to prevent overfitting.
    """
    
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        """
        Initialize Ridge Regression model.
        
        Args:
            alpha: Regularization strength (higher values = more regularization)
            fit_intercept: Whether to calculate the intercept for this model
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
        """
        Fit Ridge regression model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Returns the instance itself
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Add intercept column if needed
        if self.fit_intercept:
            X_with_intercept = add_intercept(X)
        else:
            X_with_intercept = X
        
        # Create regularization matrix (don't regularize intercept)
        n_features = X_with_intercept.shape[1]
        reg_matrix = self.alpha * np.eye(n_features)
        if self.fit_intercept:
            reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Ridge regression solution: theta = (X^T X + alpha*I)^(-1) X^T y
        try:
            theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept + reg_matrix,
                                  X_with_intercept.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            theta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept + reg_matrix) @ X_with_intercept.T @ y
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Ridge regression model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.coef_ + self.intercept_


class LassoRegression:
    """
    Lasso Regression (L1 regularization) implementation using coordinate descent.
    
    Adds L1 penalty term to the cost function for feature selection.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-4,
                 fit_intercept: bool = True):
        """
        Initialize Lasso Regression model.
        
        Args:
            alpha: Regularization strength (higher values = more regularization)
            max_iterations: Maximum number of iterations for coordinate descent
            tolerance: Convergence tolerance
            fit_intercept: Whether to calculate the intercept for this model
        """
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_iterations_ = 0
        self._is_fitted = False
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """
        Soft thresholding operator for L1 regularization.
        
        Args:
            x: Input value
            threshold: Threshold value
        
        Returns:
            Soft-thresholded value
        """
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegression':
        """
        Fit Lasso regression model using coordinate descent.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
        
        Returns:
            self: Returns the instance itself
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # Center the data if fitting intercept
        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features)
            y_mean = 0.0
        
        # Initialize coefficients
        coef = np.zeros(n_features)
        
        # Coordinate descent algorithm
        for iteration in range(self.max_iterations):
            coef_old = coef.copy()
            
            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y_centered - X_centered @ coef + coef[j] * X_centered[:, j]
                
                # Compute coordinate update
                rho = X_centered[:, j] @ residual
                
                # Soft thresholding
                coef[j] = self._soft_threshold(rho / n_samples, self.alpha / n_samples)
            
            # Check convergence
            if np.sum(np.abs(coef - coef_old)) < self.tolerance:
                break
        
        self.n_iterations_ = iteration + 1
        self.coef_ = coef
        
        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Lasso regression model.
        
        Args:
            X: Samples of shape (n_samples, n_features)
        
        Returns:
            Predicted values of shape (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.coef_ + self.intercept_