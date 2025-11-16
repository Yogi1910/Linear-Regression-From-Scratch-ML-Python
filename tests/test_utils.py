"""
Unit tests for utils module.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    add_intercept, normalize_features, denormalize_features,
    train_test_split, polynomial_features, compute_correlation_matrix,
    detect_multicollinearity
)


class TestUtils:
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.random.randn(100, 3)
        self.y = np.random.randn(100)
    
    def test_add_intercept(self):
        """Test adding intercept column."""
        X_with_intercept = add_intercept(self.X)
        
        # Should add one column
        assert X_with_intercept.shape == (self.X.shape[0], self.X.shape[1] + 1)
        
        # First column should be all ones
        np.testing.assert_array_equal(X_with_intercept[:, 0], np.ones(self.X.shape[0]))
        
        # Remaining columns should be original X
        np.testing.assert_array_equal(X_with_intercept[:, 1:], self.X)
    
    def test_normalize_features_standardize(self):
        """Test feature standardization."""
        X_norm, params = normalize_features(self.X, method='standardize')
        
        # Should have zero mean and unit variance (approximately)
        np.testing.assert_array_almost_equal(np.mean(X_norm, axis=0), 0, decimal=10)
        np.testing.assert_array_almost_equal(np.std(X_norm, axis=0), 1, decimal=10)
        
        # Check parameters
        assert params['method'] == 'standardize'
        assert 'mean' in params
        assert 'std' in params
    
    def test_normalize_features_minmax(self):
        """Test min-max normalization."""
        X_norm, params = normalize_features(self.X, method='minmax')
        
        # Should be in range [0, 1]
        assert np.all(X_norm >= 0)
        assert np.all(X_norm <= 1)
        
        # Check parameters
        assert params['method'] == 'minmax'
        assert 'min' in params
        assert 'max' in params
    
    def test_denormalize_features(self):
        """Test feature denormalization."""
        # Test standardization
        X_norm, params = normalize_features(self.X, method='standardize')
        X_denorm = denormalize_features(X_norm, params)
        np.testing.assert_array_almost_equal(X_denorm, self.X)
        
        # Test min-max
        X_norm, params = normalize_features(self.X, method='minmax')
        X_denorm = denormalize_features(X_norm, params)
        np.testing.assert_array_almost_equal(X_denorm, self.X)
    
    def test_normalize_invalid_method(self):
        """Test invalid normalization method."""
        with pytest.raises(ValueError, match="Method must be"):
            normalize_features(self.X, method='invalid')
    
    def test_train_test_split(self):
        """Test train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check that all data is preserved
        assert len(X_train) + len(X_test) == len(self.X)
        assert len(y_train) + len(y_test) == len(self.y)
    
    def test_train_test_split_reproducible(self):
        """Test that train-test split is reproducible."""
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
    
    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        X_simple = np.array([[1], [2], [3]])
        X_poly = polynomial_features(X_simple, degree=2)
        
        # Should have original feature + squared feature
        expected = np.array([[1, 1], [2, 4], [3, 9]])
        np.testing.assert_array_equal(X_poly, expected)
    
    def test_polynomial_features_multiple(self):
        """Test polynomial features with multiple input features."""
        X_multi = np.array([[1, 2], [2, 3]])
        X_poly = polynomial_features(X_multi, degree=2)
        
        # Should have original features + squared features
        assert X_poly.shape[1] == 4  # 2 original + 2 squared
        
        # Check that original features are preserved
        np.testing.assert_array_equal(X_poly[:, :2], X_multi)
    
    def test_polynomial_features_1d_input(self):
        """Test polynomial features with 1D input."""
        X_1d = np.array([1, 2, 3])
        X_poly = polynomial_features(X_1d, degree=2)
        
        expected = np.array([[1, 1], [2, 4], [3, 9]])
        np.testing.assert_array_equal(X_poly, expected)
    
    def test_compute_correlation_matrix(self):
        """Test correlation matrix computation."""
        corr_matrix = compute_correlation_matrix(self.X)
        
        # Should be square matrix
        assert corr_matrix.shape == (self.X.shape[1], self.X.shape[1])
        
        # Diagonal should be 1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)
    
    def test_detect_multicollinearity(self):
        """Test multicollinearity detection."""
        # Create data with multicollinearity
        X_multi = np.random.randn(100, 3)
        X_multi[:, 2] = X_multi[:, 0] + 0.01 * np.random.randn(100)  # Highly correlated
        
        result = detect_multicollinearity(X_multi, threshold=0.8)
        
        # Should detect high correlation
        assert 'correlation_matrix' in result
        assert 'high_correlation_pairs' in result
        assert 'threshold' in result
        
        # Should find at least one highly correlated pair
        assert len(result['high_correlation_pairs']) > 0
    
    def test_detect_multicollinearity_no_correlation(self):
        """Test multicollinearity detection with uncorrelated features."""
        result = detect_multicollinearity(self.X, threshold=0.8)
        
        # Should not find high correlations in random data
        assert len(result['high_correlation_pairs']) == 0