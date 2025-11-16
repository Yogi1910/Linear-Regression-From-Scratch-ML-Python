"""
Unit tests for LinearRegression class.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from linear_regression import LinearRegression


class TestLinearRegression:
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_simple = np.array([[1], [2], [3], [4], [5]])
        self.y_simple = np.array([2, 4, 6, 8, 10])
        
        # More complex data
        self.X_multi = np.random.randn(100, 3)
        self.true_coef = np.array([1.5, -2.0, 0.5])
        self.true_intercept = 1.0
        self.y_multi = self.X_multi @ self.true_coef + self.true_intercept + 0.1 * np.random.randn(100)
    
    def test_simple_linear_regression(self):
        """Test simple linear regression with perfect linear data."""
        model = LinearRegression()
        model.fit(self.X_simple, self.y_simple)
        
        # Should fit perfectly
        assert abs(model.coef_[0] - 2.0) < 1e-10
        assert abs(model.intercept_ - 0.0) < 1e-10
        
        # Test predictions
        predictions = model.predict(self.X_simple)
        np.testing.assert_array_almost_equal(predictions, self.y_simple)
    
    def test_multiple_linear_regression(self):
        """Test multiple linear regression."""
        model = LinearRegression()
        model.fit(self.X_multi, self.y_multi)
        
        # Coefficients should be close to true values
        np.testing.assert_array_almost_equal(model.coef_, self.true_coef, decimal=1)
        assert abs(model.intercept_ - self.true_intercept) < 0.2
    
    def test_no_intercept(self):
        """Test linear regression without intercept."""
        model = LinearRegression(fit_intercept=False)
        model.fit(self.X_simple, self.y_simple)
        
        assert model.intercept_ == 0.0
        assert model.coef_[0] == 2.0
    
    def test_r2_score(self):
        """Test R² score calculation."""
        model = LinearRegression()
        model.fit(self.X_simple, self.y_simple)
        
        r2 = model.score(self.X_simple, self.y_simple)
        assert r2 == 1.0  # Perfect fit
    
    def test_1d_input(self):
        """Test that 1D input is handled correctly."""
        X_1d = np.array([1, 2, 3, 4, 5])
        y_1d = np.array([2, 4, 6, 8, 10])
        
        model = LinearRegression()
        model.fit(X_1d, y_1d)
        
        predictions = model.predict(X_1d)
        np.testing.assert_array_almost_equal(predictions, y_1d)
    
    def test_prediction_before_fit(self):
        """Test that prediction raises error before fitting."""
        model = LinearRegression()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(self.X_simple)
    
    def test_get_set_params(self):
        """Test parameter getting and setting."""
        model = LinearRegression(fit_intercept=False)
        params = model.get_params()
        
        assert params['fit_intercept'] == False
        
        model.set_params(fit_intercept=True)
        assert model.fit_intercept == True
    
    def test_singular_matrix(self):
        """Test handling of singular matrices."""
        # Create singular matrix (linearly dependent columns)
        X_singular = np.array([[1, 2], [2, 4], [3, 6]])
        y_singular = np.array([1, 2, 3])
        
        model = LinearRegression()
        # Should not raise error, should use pseudo-inverse
        model.fit(X_singular, y_singular)
        
        assert model.coef_ is not None
        assert model.intercept_ is not None