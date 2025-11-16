# Visualization Guide for Linear Regression

This guide covers essential visualizations for understanding, diagnosing, and presenting linear regression models.

## 1. Data Exploration Visualizations

### 1.1 Scatter Plots

**Purpose**: Examine relationships between features and target variable.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_feature_relationships(X, y, feature_names=None):
    """Plot scatter plots for each feature vs target."""
    n_features = X.shape[1]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_features):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        ax.scatter(X[:, i], y, alpha=0.6, s=30)
        ax.set_xlabel(f'Feature {i}' if feature_names is None else feature_names[i])
        ax.set_ylabel('Target')
        ax.set_title(f'Feature {i} vs Target')
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    return fig
```

### 1.2 Correlation Heatmap

**Purpose**: Visualize correlations between features.

```python
def plot_correlation_heatmap(X, feature_names=None, figsize=(10, 8)):
    """Plot correlation heatmap of features."""
    corr_matrix = np.corrcoef(X.T)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    
    # Add correlation values
    for i in range(len(feature_names)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    return fig
```

### 1.3 Distribution Plots

**Purpose**: Understand the distribution of features and target.

```python
def plot_distributions(X, y, feature_names=None):
    """Plot histograms of features and target."""
    n_features = X.shape[1]
    n_plots = n_features + 1  # +1 for target
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Plot feature distributions
    for i in range(n_features):
        axes[i].hist(X[:, i], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(f'Feature {i}' if feature_names is None else feature_names[i])
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution of Feature {i}')
        axes[i].grid(True, alpha=0.3)
    
    # Plot target distribution
    axes[n_features].hist(y, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[n_features].set_xlabel('Target')
    axes[n_features].set_ylabel('Frequency')
    axes[n_features].set_title('Distribution of Target')
    axes[n_features].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig
```

## 2. Model Fitting Visualizations

### 2.1 Regression Line (1D)

**Purpose**: Show fitted line for single-feature regression.

```python
def plot_regression_line_1d(X, y, model, title="Linear Regression Fit"):
    """Plot regression line for 1D data."""
    if X.shape[1] != 1:
        raise ValueError("This function is for 1D data only")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot of data
    ax.scatter(X.flatten(), y, alpha=0.6, color='blue', s=50, label='Data points')
    
    # Create smooth line for regression
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred_range = model.predict(X_range)
    
    ax.plot(X_range, y_pred_range, color='red', linewidth=2, label='Regression line')
    
    # Add equation to plot
    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f'y = {coef:.3f}x + {intercept:.3f}'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
```

### 2.2 Predictions vs Actual

**Purpose**: Evaluate model performance visually.

```python
def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
    """Plot predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Calculate and display R²
    from src.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    return fig
```

## 3. Diagnostic Visualizations

### 3.1 Residual Plots

**Purpose**: Check model assumptions and identify patterns.

```python
def plot_residual_analysis(y_true, y_pred, figsize=(15, 5)):
    """Comprehensive residual analysis plots."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Residuals vs Fitted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Scale-Location Plot
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    axes[2].scatter(y_pred, sqrt_abs_residuals, alpha=0.6)
    axes[2].set_xlabel('Fitted Values')
    axes[2].set_ylabel('√|Residuals|')
    axes[2].set_title('Scale-Location Plot')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### 3.2 Residual Distribution

**Purpose**: Check normality assumption.

```python
def plot_residual_distribution(y_true, y_pred):
    """Plot residual distribution with normal overlay."""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with normal overlay
    ax1.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black', label='Residuals')
    
    # Overlay normal distribution
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_curve = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax1.plot(x, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
    
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Density')
    ax1.set_title('Residual Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(residuals, vert=True)
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## 4. Feature Importance Visualization

### 4.1 Coefficient Plot

**Purpose**: Show relative importance of features.

```python
def plot_feature_importance(model, feature_names=None, title="Feature Importance"):
    """Plot feature coefficients as importance measure."""
    coef = model.coef_
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(coef))]
    
    # Sort by absolute coefficient value
    abs_coef = np.abs(coef)
    sorted_idx = np.argsort(abs_coef)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(coef) * 0.4)))
    
    # Horizontal bar plot
    colors = ['red' if c < 0 else 'blue' for c in coef[sorted_idx]]
    bars = ax.barh(range(len(coef)), coef[sorted_idx], color=colors, alpha=0.7)
    
    # Customize plot
    ax.set_yticks(range(len(coef)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_xlabel('Coefficient Value')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, coef[sorted_idx])):
        ax.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                va='center', ha='left' if val >= 0 else 'right')
    
    plt.tight_layout()
    return fig
```

## 5. Regularization Visualizations

### 5.1 Regularization Path

**Purpose**: Show how coefficients change with regularization strength.

```python
def plot_regularization_path(alphas, coefs, feature_names=None, title="Regularization Path"):
    """Plot coefficient paths for different regularization strengths."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(coefs.shape[1])]
    
    # Plot each coefficient path
    for i, name in enumerate(feature_names):
        ax.plot(alphas, coefs[:, i], marker='o', linewidth=2, label=name)
    
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Parameter (α)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### 5.2 Cross-Validation Curve

**Purpose**: Select optimal regularization parameter.

```python
def plot_cv_curve(alphas, train_scores, val_scores, title="Cross-Validation Curve"):
    """Plot training and validation scores vs regularization parameter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot curves with error bands
    ax.plot(alphas, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(alphas, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    
    ax.plot(alphas, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(alphas, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    # Mark best alpha
    best_idx = np.argmax(val_mean)
    best_alpha = alphas[best_idx]
    ax.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2, 
               label=f'Best α = {best_alpha:.4f}')
    
    ax.set_xscale('log')
    ax.set_xlabel('Regularization Parameter (α)')
    ax.set_ylabel('Score (R²)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## 6. Gradient Descent Visualization

### 6.1 Cost Function History

**Purpose**: Monitor convergence during training.

```python
def plot_cost_history(cost_history, title="Cost Function During Training"):
    """Plot cost function history."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(cost_history, linewidth=2, color='blue')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add convergence info
    final_cost = cost_history[-1]
    n_iterations = len(cost_history)
    ax.text(0.7, 0.95, f'Final Cost: {final_cost:.6f}\nIterations: {n_iterations}', 
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig
```

### 6.2 Cost Surface (2D)

**Purpose**: Visualize optimization landscape for 2 parameters.

```python
def plot_cost_surface_2d(X, y, cost_history_params=None, title="Cost Surface"):
    """Plot 2D cost surface (works for 1 feature + intercept)."""
    if X.shape[1] != 1:
        raise ValueError("This visualization works for 1 feature only")
    
    # Create parameter grid
    theta0_range = np.linspace(-2, 2, 50)  # intercept
    theta1_range = np.linspace(-2, 2, 50)  # slope
    
    # Calculate cost for each parameter combination
    costs = np.zeros((len(theta0_range), len(theta1_range)))
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    for i, theta0 in enumerate(theta0_range):
        for j, theta1 in enumerate(theta1_range):
            theta = np.array([theta0, theta1])
            pred = X_with_intercept @ theta
            cost = 0.5 * np.mean((pred - y) ** 2)
            costs[i, j] = cost
    
    # Create contour plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contour(theta1_range, theta0_range, costs, levels=20)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot parameter history if provided
    if cost_history_params is not None:
        params_array = np.array(cost_history_params)
        ax.plot(params_array[:, 1], params_array[:, 0], 'ro-', markersize=3, 
                linewidth=1, label='Gradient Descent Path')
        ax.plot(params_array[0, 1], params_array[0, 0], 'go', markersize=8, label='Start')
        ax.plot(params_array[-1, 1], params_array[-1, 0], 'ro', markersize=8, label='End')
    
    ax.set_xlabel('θ₁ (slope)')
    ax.set_ylabel('θ₀ (intercept)')
    ax.set_title(title)
    if cost_history_params is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## 7. Advanced Visualizations

### 7.1 Learning Curves

**Purpose**: Diagnose bias vs variance.

```python
def plot_learning_curves(train_sizes, train_scores, val_scores, title="Learning Curves"):
    """Plot learning curves to diagnose bias/variance."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score (R²)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

### 7.2 Partial Dependence Plots

**Purpose**: Show effect of individual features.

```python
def plot_partial_dependence(model, X, feature_idx, feature_name=None, n_points=50):
    """Plot partial dependence for a single feature."""
    if feature_name is None:
        feature_name = f'Feature_{feature_idx}'
    
    # Create range for the feature
    feature_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_points)
    
    # Calculate partial dependence
    partial_deps = []
    for val in feature_range:
        X_temp = X.copy()
        X_temp[:, feature_idx] = val
        pred = model.predict(X_temp)
        partial_deps.append(np.mean(pred))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(feature_range, partial_deps, linewidth=2, color='blue')
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Partial Dependence')
    ax.set_title(f'Partial Dependence Plot: {feature_name}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## 8. Complete Visualization Dashboard

```python
def create_regression_dashboard(X, y, model, feature_names=None):
    """Create comprehensive visualization dashboard."""
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Create subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Feature relationships (top row)
    for i in range(min(3, X.shape[1])):
        ax = plt.subplot(3, 4, i + 1)
        plt.scatter(X[:, i], y, alpha=0.6)
        plt.xlabel(feature_names[i])
        plt.ylabel('Target')
        plt.title(f'{feature_names[i]} vs Target')
        plt.grid(True, alpha=0.3)
    
    # 2. Predictions vs Actual
    ax = plt.subplot(3, 4, 4)
    plt.scatter(y, y_pred, alpha=0.6)
    min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # 3. Residual plots (middle row)
    residuals = y - y_pred
    
    # Residuals vs Fitted
    ax = plt.subplot(3, 4, 5)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = plt.subplot(3, 4, 6)
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=ax)
    plt.title('Normal Q-Q Plot')
    plt.grid(True, alpha=0.3)
    
    # Residual histogram
    ax = plt.subplot(3, 4, 7)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    # Feature importance
    ax = plt.subplot(3, 4, 8)
    coef = model.coef_
    colors = ['red' if c < 0 else 'blue' for c in coef]
    plt.barh(range(len(coef)), coef, color=colors, alpha=0.7)
    plt.yticks(range(len(coef)), feature_names)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    # 4. Correlation matrix (bottom)
    ax = plt.subplot(3, 2, 5)
    corr_matrix = np.corrcoef(X.T)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names)
    plt.title('Feature Correlations')
    plt.colorbar(im, ax=ax)
    
    # 5. Model metrics
    ax = plt.subplot(3, 2, 6)
    ax.axis('off')
    
    from src.metrics import regression_metrics
    metrics = regression_metrics(y, y_pred, n_features=X.shape[1])
    
    metrics_text = "Model Performance:\n\n"
    for metric, value in metrics.items():
        metrics_text += f"{metric.upper()}: {value:.4f}\n"
    
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Usage example:
# fig = create_regression_dashboard(X, y, model, feature_names=['size', 'bedrooms', 'age'])
# plt.show()
```

This comprehensive visualization guide provides all the essential plots needed for linear regression analysis, from data exploration to model diagnostics and performance evaluation.