"""
Unit tests for regularization module.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from regularization import RidgeRegression, LassoRegression
from linear_regression import LinearRegression


class TestRidgeRegression:
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Simple data
        self.X_simple = np.array([[1], [2], [3], [4], [5]])
        self.y_simple = np.array([2, 4, 6, 8, 10])
        
        # Data with multicollinearity
        self.X_multi = np.random.randn(100, 3)
        self.X_multi = np.column_stack([self.X_multi, self.X_multi[:, 0] + 0.1 * np.random.randn(100)])
        self.true_coef = np.array([1.5, -2.0, 0.5, 0.0])
        self.y_multi = self.X_multi @ self.true_coef + 0.1 * np.random.randn(100)
    
    def test_ridge_vs_ols_simple(self):
        """Test Ridge regression on simple data."""
        ridge = RidgeRegression(alpha=0.0)  # No regularization
        ols = LinearRegression()
        
        ridge.fit(self.X_simple, self.y_simple)
        ols.fit(self.X_simple, self.y_simple)
        
        # With alpha=0, should be similar to OLS
        np.testing.assert_array_almost_equal(ridge.coef_, ols.coef_, decimal=5)
        assert abs(ridge.intercept_ - ols.intercept_) < 1e-5
    
    def test_ridge_regularization_effect(self):
        """Test that regularization shrinks coefficients."""
        ridge_low = RidgeRegression(alpha=0.1)
        ridge_high = RidgeRegression(alpha=10.0)
        
        ridge_low.fit(self.X_multi, self.y_multi)
        ridge_high.fit(self.X_multi, self.y_multi)
        
        # Higher regularization should lead to smaller coefficients
        coef_norm_low = np.linalg.norm(ridge_low.coef_)
        coef_norm_high = np.linalg.norm(ridge_high.coef_)
        
        assert coef_norm_high < coef_norm_low
    
    def test_ridge_no_intercept(self):
        """Test Ridge regression without intercept."""
        ridge = RidgeRegression(alpha=1.0, fit_intercept=False)
        ridge.fit(self.X_simple, self.y_simple)
        
        assert ridge.intercept_ == 0.0
    
    def test_ridge_prediction(self):
        """Test Ridge regression predictions."""
        ridge = RidgeRegression(alpha=1.0)
        ridge.fit(self.X_multi, self.y_multi)
        
        predictions = ridge.predict(self.X_multi)
        assert len(predictions) == len(self.y_multi)
    
    def test_ridge_prediction_before_fit(self):
        """Test prediction before fitting raises error."""
        ridge = RidgeRegression()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            ridge.predict(self.X_simple)


class TestLassoRegression:
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Sparse data (some coefficients should be zero)
        self.X_sparse = np.random.randn(100, 5)
        self.true_coef = np.array([2.0, 0.0, -1.5, 0.0, 1.0])  # Some zeros
        self.y_sparse = self.X_sparse @ self.true_coef + 0.1 * np.random.randn(100)
    
    def test_lasso_sparsity(self):
        """Test that Lasso produces sparse solutions."""
        lasso = LassoRegression(alpha=0.1)
        lasso.fit(self.X_sparse, self.y_sparse)
        
        # Should have some coefficients close to zero
        n_zero_coef = np.sum(np.abs(lasso.coef_) < 1e-3)
        assert n_zero_coef > 0
    
    def test_lasso_alpha_effect(self):
        """Test effect of regularization parameter."""
        lasso_low = LassoRegression(alpha=0.01)
        lasso_high = LassoRegression(alpha=1.0)
        
        lasso_low.fit(self.X_sparse, self.y_sparse)
        lasso_high.fit(self.X_sparse, self.y_sparse)
        
        # Higher alpha should lead to more sparsity
        n_zero_low = np.sum(np.abs(lasso_low.coef_) < 1e-3)
        n_zero_high = np.sum(np.abs(lasso_high.coef_) < 1e-3)
        
        assert n_zero_high >= n_zero_low
    
    def test_lasso_convergence(self):
        """Test that Lasso converges."""
        lasso = LassoRegression(alpha=0.1, max_iterations=1000)
        lasso.fit(self.X_sparse, self.y_sparse)
        
        # Should converge before max iterations
        assert lasso.n_iterations_ <= 1000
    
    def test_lasso_no_intercept(self):
        """Test Lasso without intercept."""
        lasso = LassoRegression(alpha=0.1, fit_intercept=False)
        lasso.fit(self.X_sparse, self.y_sparse)
        
        assert lasso.intercept_ == 0.0
    
    def test_lasso_prediction(self):
        """Test Lasso predictions."""
        lasso = LassoRegression(alpha=0.1)
        lasso.fit(self.X_sparse, self.y_sparse)
        
        predictions = lasso.predict(self.X_sparse)
        assert len(predictions) == len(self.y_sparse)
    
    def test_lasso_prediction_before_fit(self):
        """Test prediction before fitting raises error."""
        lasso = LassoRegression()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            lasso.predict(self.X_sparse)
    
    def test_soft_threshold(self):
        """Test soft thresholding function."""
        lasso = LassoRegression()
        
        # Test soft thresholding
        assert lasso._soft_threshold(2.0, 1.0) == 1.0
        assert lasso._soft_threshold(-2.0, 1.0) == -1.0
        assert lasso._soft_threshold(0.5, 1.0) == 0.0
        assert lasso._soft_threshold(-0.5, 1.0) == 0.0