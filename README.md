# Linear Regression from Scratch - ML Python

A comprehensive implementation of linear regression algorithms from scratch in Python, including theoretical foundations, multiple optimization methods, and practical applications.

## Features

- **Core Implementation**: Pure Python linear regression with NumPy
- **Multiple Optimization Methods**: Gradient descent, normal equation
- **Regularization**: Ridge and Lasso regression implementations
- **Comprehensive Metrics**: R², MSE, MAE, and more
- **Interactive Notebooks**: Step-by-step theory and implementation
- **Visualization Tools**: Plots for understanding model behavior
- **Real Data Examples**: Synthetic and real-world datasets

## Project Structure

```
linear-regression-from-scratch-ml-python/
├── src/                    # Core implementation modules
├── notebooks/              # Jupyter notebooks with theory and examples
├── data/                   # Datasets and data generation scripts
├── tests/                  # Unit tests
├── docs/                   # Documentation and theory explanations
└── assets/                 # Visualizations and plots
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the basic example:
   ```python
   from src.linear_regression import LinearRegression
   
   # Create and train model
   model = LinearRegression()
   model.fit(X_train, y_train)
   predictions = model.predict(X_test)
   ```

3. Explore the notebooks for detailed explanations and examples.

## Theory Coverage

- Ordinary Least Squares (OLS) derivation
- Geometric interpretation of linear regression
- Gradient descent optimization
- Regularization techniques (Ridge/Lasso)
- Model assumptions and diagnostics
- Bias-variance tradeoff

## License

MIT License - see LICENSE file for details.