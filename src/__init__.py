"""
Linear Regression from Scratch - Core Implementation Package
"""

from .linear_regression import LinearRegression
from .gradient_descent import GradientDescent
from .regularization import RidgeRegression, LassoRegression
from .metrics import r2_score, mean_squared_error, mean_absolute_error
from .visualization import plot_regression_line, plot_residuals, plot_cost_history
from .utils import add_intercept, normalize_features, train_test_split

__version__ = "1.0.0"
__author__ = "Linear Regression Team"

__all__ = [
    "LinearRegression",
    "GradientDescent", 
    "RidgeRegression",
    "LassoRegression",
    "r2_score",
    "mean_squared_error",
    "mean_absolute_error",
    "plot_regression_line",
    "plot_residuals", 
    "plot_cost_history",
    "add_intercept",
    "normalize_features",
    "train_test_split"
]