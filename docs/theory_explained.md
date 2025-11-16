# Linear Regression Theory Explained

## Introduction

Linear regression is one of the fundamental algorithms in machine learning and statistics. It models the relationship between a dependent variable (target) and one or more independent variables (features) using a linear equation.

## Mathematical Foundation

### Simple Linear Regression

For a single feature, the linear regression model is:

```
y = β₀ + β₁x + ε
```

Where:
- `y` is the dependent variable (target)
- `x` is the independent variable (feature)
- `β₀` is the intercept (bias term)
- `β₁` is the slope (coefficient)
- `ε` is the error term

### Multiple Linear Regression

For multiple features, the model extends to:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

In matrix form:
```
y = Xβ + ε
```

Where:
- `y` is the target vector (n × 1)
- `X` is the feature matrix (n × p)
- `β` is the coefficient vector (p × 1)
- `ε` is the error vector (n × 1)

## Cost Function

Linear regression uses the **Mean Squared Error (MSE)** as the cost function:

```
J(β) = (1/2m) Σᵢ₌₁ᵐ (hβ(xᵢ) - yᵢ)²
```

In matrix form:
```
J(β) = (1/2m)(Xβ - y)ᵀ(Xβ - y)
```

The goal is to find the parameters β that minimize this cost function.

## Optimization Methods

### 1. Normal Equation (Analytical Solution)

The optimal parameters can be found analytically using the normal equation:

```
β = (XᵀX)⁻¹Xᵀy
```

**Advantages:**
- Exact solution (no approximation)
- No hyperparameters to tune
- Works well for small to medium datasets

**Disadvantages:**
- Computationally expensive for large datasets (O(n³))
- Requires matrix inversion
- May be unstable if XᵀX is singular

### 2. Gradient Descent (Iterative Solution)

Gradient descent iteratively updates parameters in the direction of steepest descent:

```
β := β - α∇J(β)
```

Where the gradient is:
```
∇J(β) = (1/m)Xᵀ(Xβ - y)
```

**Advantages:**
- Scales well to large datasets
- Can handle singular matrices
- Extensible to other optimization algorithms

**Disadvantages:**
- Requires tuning learning rate α
- May converge slowly
- Can get stuck in local minima (though MSE is convex)

## Assumptions of Linear Regression

1. **Linearity**: The relationship between features and target is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: Features are not highly correlated

## Model Evaluation

### R-squared (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- `SS_res = Σ(yᵢ - ŷᵢ)²` (sum of squared residuals)
- `SS_tot = Σ(yᵢ - ȳ)²` (total sum of squares)

R² represents the proportion of variance in the target variable explained by the model.

### Adjusted R-squared

```
R²_adj = 1 - (1 - R²) × (n - 1)/(n - p - 1)
```

Adjusts R² for the number of features, penalizing models with many features.

### Other Metrics

- **Mean Squared Error (MSE)**: Average of squared residuals
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average of absolute residuals

## Regularization

### Ridge Regression (L2 Regularization)

Adds L2 penalty to prevent overfitting:

```
J(β) = MSE + α||β||₂²
```

Solution:
```
β = (XᵀX + αI)⁻¹Xᵀy
```

### Lasso Regression (L1 Regularization)

Adds L1 penalty for feature selection:

```
J(β) = MSE + α||β||₁
```

No closed-form solution; requires iterative methods like coordinate descent.

## Geometric Interpretation

Linear regression can be viewed as finding the projection of the target vector y onto the column space of the feature matrix X. The projection matrix is:

```
P = X(XᵀX)⁻¹Xᵀ
```

The predicted values are:
```
ŷ = Py
```

And the residuals are:
```
e = y - ŷ = (I - P)y
```

## Practical Considerations

### Feature Scaling

- **Standardization**: (x - μ)/σ
- **Min-Max Scaling**: (x - min)/(max - min)

Feature scaling is important for:
- Gradient descent convergence
- Regularization effectiveness
- Numerical stability

### Handling Categorical Variables

- **One-hot encoding**: Create binary features for each category
- **Label encoding**: Assign numerical values (for ordinal variables)

### Dealing with Multicollinearity

- Remove highly correlated features
- Use regularization (Ridge/Lasso)
- Principal Component Analysis (PCA)

## Extensions and Variations

1. **Polynomial Regression**: Add polynomial features
2. **Interaction Terms**: Include feature interactions
3. **Robust Regression**: Handle outliers better
4. **Bayesian Linear Regression**: Incorporate prior beliefs
5. **Generalized Linear Models**: Extend to non-Gaussian distributions