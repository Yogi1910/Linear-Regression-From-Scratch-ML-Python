"""
Unit tests for metrics module.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics import (
    mean_squared_error, mean_absolute_error, root_mean_squared_error,
    r2_score, adjusted_r2_score, mean_absolute_percentage_error,
    regression_metrics
)


class TestMetrics:
    
    def setup_method(self):
        """Set up test data."""
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        self.y_perfect = np.array([1, 2, 3, 4, 5])
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        mse = mean_squared_error(self.y_true, self.y_pred)
        expected_mse = np.mean((self.y_true - self.y_pred) ** 2)
        assert abs(mse - expected_mse) < 1e-10
        
        # Perfect predictions should have MSE = 0
        mse_perfect = mean_squared_error(self.y_true, self.y_perfect)
        assert mse_perfect == 0.0
    
    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        mae = mean_absolute_error(self.y_true, self.y_pred)
        expected_mae = np.mean(np.abs(self.y_true - self.y_pred))
        assert abs(mae - expected_mae) < 1e-10
        
        # Perfect predictions should have MAE = 0
        mae_perfect = mean_absolute_error(self.y_true, self.y_perfect)
        assert mae_perfect == 0.0
    
    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        expected_rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        assert abs(rmse - expected_rmse) < 1e-10
    
    def test_r2_score(self):
        """Test R² score calculation."""
        r2 = r2_score(self.y_true, self.y_pred)
        
        # R² should be between 0 and 1 for reasonable predictions
        assert 0 <= r2 <= 1
        
        # Perfect predictions should have R² = 1
        r2_perfect = r2_score(self.y_true, self.y_perfect)
        assert abs(r2_perfect - 1.0) < 1e-10
    
    def test_adjusted_r2_score(self):
        """Test adjusted R² score calculation."""
        r2 = r2_score(self.y_true, self.y_pred)
        adj_r2 = adjusted_r2_score(self.y_true, self.y_pred, n_features=2)
        
        # Adjusted R² should be less than or equal to R²
        assert adj_r2 <= r2
    
    def test_mean_absolute_percentage_error(self):
        """Test MAPE calculation."""
        mape = mean_absolute_percentage_error(self.y_true, self.y_pred)
        
        # MAPE should be positive
        assert mape >= 0
        
        # Perfect predictions should have MAPE = 0
        mape_perfect = mean_absolute_percentage_error(self.y_true, self.y_perfect)
        assert mape_perfect == 0.0
    
    def test_mape_zero_division(self):
        """Test MAPE with zero values in y_true."""
        y_true_with_zero = np.array([0, 1, 2, 3, 4])
        y_pred_with_zero = np.array([0.1, 1.1, 1.9, 3.1, 3.9])
        
        # Should handle zero values by excluding them
        mape = mean_absolute_percentage_error(y_true_with_zero, y_pred_with_zero)
        assert mape >= 0
    
    def test_regression_metrics(self):
        """Test comprehensive regression metrics function."""
        metrics = regression_metrics(self.y_true, self.y_pred, n_features=2)
        
        # Should contain all expected metrics
        expected_keys = ['mse', 'rmse', 'mae', 'r2', 'mape', 'adjusted_r2']
        for key in expected_keys:
            assert key in metrics
        
        # All metrics should be reasonable values
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['mape'] >= 0
        assert metrics['r2'] <= 1
        assert metrics['adjusted_r2'] <= metrics['r2']
    
    def test_regression_metrics_without_features(self):
        """Test regression metrics without n_features parameter."""
        metrics = regression_metrics(self.y_true, self.y_pred)
        
        # Should not contain adjusted_r2
        assert 'adjusted_r2' not in metrics
        assert 'r2' in metrics
    
    def test_array_conversion(self):
        """Test that functions handle list inputs."""
        y_true_list = [1, 2, 3, 4, 5]
        y_pred_list = [1.1, 1.9, 3.1, 3.9, 5.1]
        
        mse_array = mean_squared_error(self.y_true, self.y_pred)
        mse_list = mean_squared_error(y_true_list, y_pred_list)
        
        assert abs(mse_array - mse_list) < 1e-10