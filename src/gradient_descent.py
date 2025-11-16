"""
Gradient Descent optimization for Linear Regression.
"""

import numpy as np
from typing import Optional, List, Tuple
from .utils import add_intercept


class GradientDescent:
    """
    Linear Regression using Gradient Descent optimization.
    
    Implements batch gradient descent to minimize the mean squared error cost function.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 fit_intercept: bool = True):
        """
        Initialize Gradient Descent optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance for cost function
            fit_intercept: Whether to fit an intercept term
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        
        self.coef_ = None
        self.intercept_ = None
        self.cost_history_ = []
        self.n_iterations_ = 0
        self._is_fitted = False
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute the mean squared error cost function.
        
        Args:
            X: Feature matrix with intercept column
            y: Target values
            theta: Parameter vector
        
        Returns:
            Cost value
        """
        m = len(y)
        predictions = X @ theta
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute gradients of the cost function.
        
        Args:
            X: Feature matrix with intercept column
            y: Target values
            theta: Parameter vector
        
        Returns:
            Gradient vector
        """
        m = len(y)
        predictions = X @ theta
        gradients = (1 / m) * X.T @ (predictions - y)
        return gradients
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescent':
        """
        Fit linear model using gradient descent.
        
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
        
        # Initialize parameters
        n_features = X_with_intercept.shape[1]
        theta = np.random.normal(0, 0.01, n_features)
        
        self.cost_history_ = []
        prev_cost = float('inf')
        
        # Gradient descent loop
        for i in range(self.max_iterations):
            # Compute cost and gradients
            cost = self._compute_cost(X_with_intercept, y, theta)
            gradients = self._compute_gradients(X_with_intercept, y, theta)
            
            # Update parameters
            theta -= self.learning_rate * gradients
            
            # Store cost history
            self.cost_history_.append(cost)
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                break
            
            prev_cost = cost
        
        self.n_iterations_ = i + 1
        
        # Extract coefficients and intercept
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