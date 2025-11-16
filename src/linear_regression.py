"""
Linear Regression implementation from scratch using NumPy.
"""

import numpy as np
from typing import Optional, Tuple
from .utils import add_intercept


class LinearRegression:
    """
    Linear Regression using Ordinary Least Squares (OLS).
    
    This implementation uses the normal equation to find the optimal parameters
    that minimize the sum of squared residuals.
    """
    
    def __init__(self, fit_intercept: bool = True):
        """
        Initialize Linear Regression model.
        
        Args:
            fit_intercept: Whether to calculate the intercept for this model
        """
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear model using the normal equation.
        
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
        
        # Normal equation: theta = (X^T X)^(-1) X^T y
        try:
            theta = np.linalg.solve(X_with_intercept.T @ X_with_intercept, 
                                  X_with_intercept.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            theta = np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        
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
        Predict using the linear model.
        
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
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Args:
            X: Test samples of shape (n_samples, n_features)
            y: True values of shape (n_samples,)
        
        Returns:
            R^2 score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get parameters for this estimator."""
        return {'fit_intercept': self.fit_intercept}
    
    def set_params(self, **params) -> 'LinearRegression':
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self