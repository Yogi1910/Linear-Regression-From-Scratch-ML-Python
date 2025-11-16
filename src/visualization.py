"""
Visualization utilities for linear regression analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from .metrics import regression_metrics


def plot_regression_line(X: np.ndarray, y: np.ndarray, model, 
                        title: str = "Linear Regression Fit",
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot regression line for single feature data.
    
    Args:
        X: Feature data (single feature)
        y: Target values
        model: Fitted regression model
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of data points
    ax.scatter(X.flatten(), y, alpha=0.6, color='blue', label='Data points')
    
    # Plot regression line
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_range)
    ax.plot(X_range, y_pred, color='red', linewidth=2, label='Regression line')
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str = "Residual Plot",
                  figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot residual analysis plots.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs Fitted values
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot of residuals
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cost_history(cost_history: List[float],
                     title: str = "Cost Function History",
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot cost function history during training.
    
    Args:
        cost_history: List of cost values during training
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(cost_history, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_feature_importance(coef: np.ndarray, feature_names: Optional[List[str]] = None,
                           title: str = "Feature Importance",
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot feature importance based on coefficient magnitudes.
    
    Args:
        coef: Model coefficients
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(coef))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute coefficient values
    abs_coef = np.abs(coef)
    sorted_idx = np.argsort(abs_coef)
    
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, abs_coef[sorted_idx])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('|Coefficient|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Predictions vs Actual",
                             figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score to plot
    r2 = regression_metrics(y_true, y_pred)['r2']
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig


def plot_regularization_path(alphas: np.ndarray, coefs: np.ndarray, 
                            feature_names: Optional[List[str]] = None,
                            title: str = "Regularization Path",
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot regularization path showing how coefficients change with regularization strength.
    
    Args:
        alphas: Array of regularization parameters
        coefs: Coefficient matrix of shape (n_alphas, n_features)
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(coefs.shape[1])]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, name in enumerate(feature_names):
        ax.plot(alphas, coefs[:, i], label=name, linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Regularization parameter (α)')
    ax.set_ylabel('Coefficient value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig