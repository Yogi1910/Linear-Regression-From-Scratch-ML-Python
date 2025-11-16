"""
Data generation utilities for linear regression examples.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_synthetic_linear_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate random true coefficients
    true_coef = np.random.randn(n_features)
    true_intercept = np.random.randn()
    
    # Generate target with linear relationship + noise
    y = X @ true_coef + true_intercept + noise * np.random.randn(n_samples)
    
    return X, y


def generate_real_estate_data(n_samples: int = 500, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic real estate price data.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with features and target price
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    size_sqft = np.random.normal(2000, 800, n_samples)
    size_sqft = np.clip(size_sqft, 500, 5000)  # Reasonable range
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.25, 0.05])
    
    age_years = np.random.exponential(15, n_samples)
    age_years = np.clip(age_years, 0, 100)
    
    # Generate price with realistic relationships
    price = (
        150 * size_sqft +  # $150 per sqft
        10000 * bedrooms +  # $10k per bedroom
        -500 * age_years +  # Depreciation
        50000 +  # Base price
        np.random.normal(0, 20000, n_samples)  # Noise
    )
    
    # Ensure positive prices
    price = np.maximum(price, 50000)
    
    return pd.DataFrame({
        'size_sqft': size_sqft,
        'bedrooms': bedrooms,
        'age_years': age_years,
        'price': price
    })


if __name__ == "__main__":
    # Generate and save synthetic linear data
    X, y = generate_synthetic_linear_data(n_samples=200, random_state=42)
    synthetic_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    synthetic_df['target'] = y
    synthetic_df.to_csv('synthetic_linear.csv', index=False)
    print("Generated synthetic_linear.csv")
    
    # Generate and save real estate data
    real_estate_df = generate_real_estate_data(n_samples=500, random_state=42)
    real_estate_df.to_csv('real_estate_prices.csv', index=False)
    print("Generated real_estate_prices.csv")