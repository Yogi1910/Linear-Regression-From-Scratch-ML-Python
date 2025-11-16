# Linear Regression Assumptions Checklist

This document provides a comprehensive checklist for validating linear regression assumptions and diagnostic procedures.

## 1. Linearity Assumption

**Assumption**: The relationship between independent variables and the dependent variable is linear.

### How to Check:
- [ ] **Scatter plots**: Plot each feature against the target variable
- [ ] **Residual plots**: Plot residuals vs. fitted values
- [ ] **Partial regression plots**: For multiple regression

### Diagnostic Code:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot for each feature
for i, feature in enumerate(feature_names):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, i], y, alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel('Target')
    plt.title(f'Linearity Check: {feature} vs Target')
    plt.show()

# Residual vs fitted plot
residuals = y - model.predict(X)
fitted = model.predict(X)
plt.figure(figsize=(8, 6))
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()
```

### What to Look For:
- ✅ **Good**: Random scatter around zero line
- ❌ **Bad**: Curved patterns, funnel shapes

### Solutions if Violated:
- Transform variables (log, square root, polynomial)
- Add interaction terms
- Use non-linear models

## 2. Independence Assumption

**Assumption**: Observations are independent of each other.

### How to Check:
- [ ] **Durbin-Watson test**: For time series data
- [ ] **Residual autocorrelation**: Plot residuals vs. time/order
- [ ] **Domain knowledge**: Consider data collection process

### Diagnostic Code:
```python
# For time series data
from scipy.stats import durbin_watson

# Calculate Durbin-Watson statistic
dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson statistic: {dw_stat:.3f}")

# Plot residuals vs. observation order
plt.figure(figsize=(10, 6))
plt.plot(residuals, marker='o', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Observation Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Observation Order')
plt.show()
```

### What to Look For:
- ✅ **Good**: DW statistic around 2.0, no patterns in residual plot
- ❌ **Bad**: DW statistic far from 2.0, systematic patterns

### Solutions if Violated:
- Use time series models (ARIMA, etc.)
- Include lagged variables
- Use robust standard errors

## 3. Homoscedasticity Assumption

**Assumption**: Constant variance of residuals across all levels of independent variables.

### How to Check:
- [ ] **Residual plots**: Plot residuals vs. fitted values
- [ ] **Breusch-Pagan test**: Statistical test for heteroscedasticity
- [ ] **Scale-location plot**: Plot √|residuals| vs. fitted values

### Diagnostic Code:
```python
from scipy.stats import levene

# Residuals vs fitted plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Homoscedasticity Check')
plt.show()

# Scale-location plot
plt.figure(figsize=(8, 6))
plt.scatter(fitted, np.sqrt(np.abs(residuals)), alpha=0.6)
plt.xlabel('Fitted Values')
plt.ylabel('√|Residuals|')
plt.title('Scale-Location Plot')
plt.show()

# Breusch-Pagan test (simplified)
from scipy.stats import pearsonr
correlation, p_value = pearsonr(fitted, residuals**2)
print(f"Correlation between fitted values and squared residuals: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
```

### What to Look For:
- ✅ **Good**: Constant spread of residuals, no correlation with fitted values
- ❌ **Bad**: Funnel shape, increasing/decreasing variance

### Solutions if Violated:
- Transform target variable (log, Box-Cox)
- Use weighted least squares
- Use robust regression methods

## 4. Normality of Residuals

**Assumption**: Residuals are normally distributed.

### How to Check:
- [ ] **Q-Q plot**: Quantile-quantile plot of residuals
- [ ] **Histogram**: Distribution of residuals
- [ ] **Shapiro-Wilk test**: Statistical test for normality
- [ ] **Kolmogorov-Smirnov test**: Another normality test

### Diagnostic Code:
```python
from scipy.stats import shapiro, kstest, probplot

# Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

probplot(residuals, dist="norm", plot=ax1)
ax1.set_title('Q-Q Plot')

# Histogram
ax2.hist(residuals, bins=20, density=True, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Density')
ax2.set_title('Histogram of Residuals')

# Add normal curve
x = np.linspace(residuals.min(), residuals.max(), 100)
ax2.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2)

plt.tight_layout()
plt.show()

# Statistical tests
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"Shapiro-Wilk test: statistic={shapiro_stat:.3f}, p-value={shapiro_p:.3f}")

ks_stat, ks_p = kstest(residuals, 'norm')
print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")
```

### What to Look For:
- ✅ **Good**: Points follow diagonal line in Q-Q plot, bell-shaped histogram
- ❌ **Bad**: Systematic deviations from diagonal, skewed distribution

### Solutions if Violated:
- Transform variables
- Use robust regression
- Bootstrap methods for inference

## 5. No Multicollinearity

**Assumption**: Independent variables are not highly correlated with each other.

### How to Check:
- [ ] **Correlation matrix**: Pairwise correlations between features
- [ ] **Variance Inflation Factor (VIF)**: Measure of multicollinearity
- [ ] **Condition number**: Matrix condition number

### Diagnostic Code:
```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Correlation matrix
corr_matrix = np.corrcoef(X.T)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            xticklabels=feature_names, yticklabels=feature_names)
plt.title('Feature Correlation Matrix')
plt.show()

# VIF calculation
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = range(X.shape[1])
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return vif_data

# Add intercept for VIF calculation
X_with_intercept = add_intercept(X)
vif_df = calculate_vif(X_with_intercept[:, 1:])  # Exclude intercept
print("Variance Inflation Factors:")
print(vif_df)

# Condition number
cond_number = np.linalg.cond(X.T @ X)
print(f"Condition number: {cond_number:.2f}")
```

### What to Look For:
- ✅ **Good**: Correlations < 0.8, VIF < 5, condition number < 30
- ❌ **Bad**: High correlations, VIF > 10, condition number > 30

### Solutions if Violated:
- Remove highly correlated features
- Use regularization (Ridge/Lasso)
- Principal Component Analysis (PCA)
- Combine correlated features

## Complete Diagnostic Function

```python
def diagnose_linear_regression(X, y, model, feature_names=None):
    """
    Comprehensive diagnostic function for linear regression assumptions.
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Get predictions and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    print("=== LINEAR REGRESSION DIAGNOSTICS ===\n")
    
    # 1. Model Performance
    from src.metrics import regression_metrics
    metrics = regression_metrics(y, y_pred, n_features=X.shape[1])
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    print()
    
    # 2. Linearity Check
    print("1. LINEARITY CHECK:")
    print("   - Examine scatter plots and residual plots")
    print("   - Look for random scatter around zero line")
    print()
    
    # 3. Independence Check
    print("2. INDEPENDENCE CHECK:")
    dw_stat = durbin_watson(residuals)
    print(f"   - Durbin-Watson statistic: {dw_stat:.3f}")
    if 1.5 <= dw_stat <= 2.5:
        print("   ✅ No strong evidence of autocorrelation")
    else:
        print("   ❌ Possible autocorrelation detected")
    print()
    
    # 4. Homoscedasticity Check
    print("3. HOMOSCEDASTICITY CHECK:")
    correlation, p_value = pearsonr(y_pred, residuals**2)
    print(f"   - Correlation (fitted vs squared residuals): {correlation:.3f}")
    print(f"   - P-value: {p_value:.3f}")
    if p_value > 0.05:
        print("   ✅ No strong evidence of heteroscedasticity")
    else:
        print("   ❌ Possible heteroscedasticity detected")
    print()
    
    # 5. Normality Check
    print("4. NORMALITY CHECK:")
    shapiro_stat, shapiro_p = shapiro(residuals)
    print(f"   - Shapiro-Wilk test: p-value = {shapiro_p:.3f}")
    if shapiro_p > 0.05:
        print("   ✅ Residuals appear normally distributed")
    else:
        print("   ❌ Residuals may not be normally distributed")
    print()
    
    # 6. Multicollinearity Check
    print("5. MULTICOLLINEARITY CHECK:")
    # Condition number
    cond_number = np.linalg.cond(X.T @ X)
    print(f"   - Condition number: {cond_number:.2f}")
    if cond_number < 30:
        print("   ✅ No strong multicollinearity")
    else:
        print("   ❌ Possible multicollinearity detected")
    
    # High correlations
    corr_matrix = np.corrcoef(X.T)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    if high_corr_pairs:
        print("   - High correlations detected:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"     {feat1} - {feat2}: {corr:.3f}")
    else:
        print("   - No high correlations detected")
    
    print("\n=== DIAGNOSTIC COMPLETE ===")

# Usage example:
# diagnose_linear_regression(X, y, model, feature_names=['size', 'bedrooms', 'age'])
```

## Summary Checklist

- [ ] **Linearity**: Scatter plots and residual plots show linear relationships
- [ ] **Independence**: Durbin-Watson ≈ 2.0, no patterns in residual sequence
- [ ] **Homoscedasticity**: Constant residual variance, no funnel shapes
- [ ] **Normality**: Q-Q plot follows diagonal, Shapiro-Wilk p > 0.05
- [ ] **No Multicollinearity**: VIF < 5, correlations < 0.8, condition number < 30

If assumptions are violated, consider:
- Data transformations
- Regularization techniques
- Robust regression methods
- Alternative modeling approaches