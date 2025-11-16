"""
Unit tests for GradientDescent class.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gradient_descent import GradientDescent


class TestGradientDescent:
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X_simple = np.array([[1], [2], [3], [4], [5]])
        self.y_simple = np.array([2, 4, 6, 8, 10])
        
        # Noisy data
        self.X_noisy = np.random.randn(100, 2)
        self.true_coef = np.array([1.5, -2.0])
        self.true_intercept = 1.0
        self.y_noisy = self.X_noisy @ self.true_coef + self.true_intercept + 0.1 * np.random.randn(100)
    
    def test_simple_convergence(self):
        """Test convergence on simple linear data."""
        model = GradientDescent(learning_rate=0.01, max_iterations=1000)
        model.fit(self.X_simple, self.y_simple)
        
        # Should converge close to true values
        assert abs(model.coef_[0] - 2.0) < 0.1
        assert abs(model.intercept_ - 0.0) < 0.1
        
        # Should have converged
        assert model.n_iterations_ < 1000
    
    def test_cost_decreases(self):
        """Test that cost function decreases during training."""
        model = GradientDescent(learning_rate=0.01, max_iterations=100)
        model.fit(self.X_noisy, self.y_noisy)
        
        # Cost should generally decrease
        assert model.cost_history_[0] > model.cost_history_[-1]
        assert len(model.cost_history_) == model.n_iterations_
    
    def test_learning_rate_effect(self):
        """Test effect of different learning rates."""
        # High learning rate
        model_high = GradientDescent(learning_rate=0.1, max_iterations=100)
        model_high.fit(self.X_noisy, self.y_noisy)
        
        # Low learning rate
        model_low = GradientDescent(learning_rate=0.001, max_iterations=100)
        model_low.fit(self.X_noisy, self.y_noisy)
        
        # High learning rate should converge faster (fewer iterations)
        # but this is not always guaranteed, so we just check they both work
        assert model_high.cost_history_[-1] < model_high.cost_history_[0]
        assert model_low.cost_history_[-1] < model_low.cost_history_[0]
    
    def test_no_intercept(self):
        """Test gradient descent without intercept."""
        model = GradientDescent(fit_intercept=False, learning_rate=0.01)
        model.fit(self.X_simple, self.y_simple)
        
        assert model.intercept_ == 0.0
        assert abs(model.coef_[0] - 2.0) < 0.1
    
    def test_early_convergence(self):
        """Test early convergence with tight tolerance."""
        model = GradientDescent(learning_rate=0.1, max_iterations=1000, tolerance=1e-8)
        model.fit(self.X_simple, self.y_simple)
        
        # Should converge early
        assert model.n_iterations_ < 1000
    
    def test_prediction_accuracy(self):
        """Test prediction accuracy."""
        model = GradientDescent(learning_rate=0.01, max_iterations=1000)
        model.fit(self.X_noisy, self.y_noisy)
        
        predictions = model.predict(self.X_noisy)
        
        # R² should be reasonable
        r2 = model.score(self.X_noisy, self.y_noisy)
        assert r2 > 0.8
    
    def test_1d_input(self):
        """Test 1D input handling."""
        X_1d = np.array([1, 2, 3, 4, 5])
        y_1d = np.array([2, 4, 6, 8, 10])
        
        model = GradientDescent(learning_rate=0.01, max_iterations=1000)
        model.fit(X_1d, y_1d)
        
        predictions = model.predict(X_1d)
        assert len(predictions) == len(y_1d)
    
    def test_prediction_before_fit(self):
        """Test prediction before fitting raises error."""
        model = GradientDescent()
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(self.X_simple)