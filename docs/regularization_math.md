# Regularization Mathematics

This document provides detailed mathematical explanations of regularization techniques in linear regression.

## 1. Introduction to Regularization

### The Overfitting Problem

In standard linear regression, we minimize:
```
J(β) = ||Xβ - y||²
```

This can lead to overfitting when:
- Number of features p is large relative to samples n
- Features are highly correlated (multicollinearity)
- Model complexity is too high

### Regularization Solution

Add a penalty term to the cost function:
```
J_reg(β) = ||Xβ - y||² + λR(β)
```

Where:
- λ ≥ 0 is the regularization parameter
- R(β) is the regularization function

## 2. Ridge Regression (L2 Regularization)

### Mathematical Formulation

Ridge regression uses L2 penalty:
```
J_ridge(β) = ||Xβ - y||² + λ||β||₂²
```

Where ||β||₂² = Σⱼβⱼ² is the squared L2 norm.

### Expanded Form

```
J_ridge(β) = (Xβ - y)ᵀ(Xβ - y) + λβᵀβ
           = βᵀXᵀXβ - 2yᵀXβ + yᵀy + λβᵀβ
           = βᵀ(XᵀX + λI)β - 2yᵀXβ + yᵀy
```

### Analytical Solution

Taking the gradient and setting to zero:
```
∇J_ridge(β) = 2(XᵀX + λI)β - 2Xᵀy = 0
```

Solving for β:
```
β_ridge = (XᵀX + λI)⁻¹Xᵀy
```

### Properties of Ridge Solution

#### 1. Always Invertible
For λ > 0, the matrix (XᵀX + λI) is always positive definite, ensuring invertibility even when XᵀX is singular.

#### 2. Shrinkage Effect
Ridge regression shrinks coefficients towards zero. To see this, consider the Singular Value Decomposition (SVD) of X:

```
X = UDVᵀ
```

Where U and V are orthogonal matrices, and D is diagonal with singular values dⱼ.

The ridge solution becomes:
```
β_ridge = VD(D² + λI)⁻¹DUᵀy
```

The shrinkage factors are:
```
fⱼ = dⱼ²/(dⱼ² + λ)
```

As λ increases, fⱼ decreases, shrinking coefficients more.

#### 3. Bias-Variance Decomposition

**Bias**:
```
Bias(β_ridge) = E[β_ridge] - β = -(XᵀX + λI)⁻¹λβ
```

**Variance**:
```
Var(β_ridge) = σ²(XᵀX + λI)⁻¹XᵀX(XᵀX + λI)⁻¹
```

Ridge regression introduces bias but reduces variance.

### Geometric Interpretation

Ridge regression can be viewed as constrained optimization:
```
minimize ||Xβ - y||²
subject to ||β||₂² ≤ t
```

The constraint region is a hypersphere. The solution occurs where the constraint boundary touches the smallest error ellipse.

## 3. Lasso Regression (L1 Regularization)

### Mathematical Formulation

Lasso uses L1 penalty:
```
J_lasso(β) = ||Xβ - y||² + λ||β||₁
```

Where ||β||₁ = Σⱼ|βⱼ| is the L1 norm.

### Non-Differentiability

The L1 norm is not differentiable at βⱼ = 0. We use subgradients:

```
∂|βⱼ|/∂βⱼ = {
  +1      if βⱼ > 0
  -1      if βⱼ < 0
  [-1,+1] if βⱼ = 0
}
```

### Coordinate Descent Algorithm

Since there's no closed-form solution, we use coordinate descent:

#### Algorithm Steps:
1. Initialize β⁽⁰⁾
2. For each iteration t:
   - For each coordinate j:
     - Compute partial residual: rⱼ = y - Σₖ≠ⱼ Xₖβₖ⁽ᵗ⁾
     - Update: βⱼ⁽ᵗ⁺¹⁾ = S(Xⱼᵀrⱼ/||Xⱼ||², λ/||Xⱼ||²)

#### Soft-Thresholding Operator:
```
S(z, γ) = sign(z) · max(|z| - γ, 0) = {
  z - γ  if z > γ
  0      if |z| ≤ γ  
  z + γ  if z < -γ
}
```

### Derivation of Soft-Thresholding

For coordinate j, we minimize:
```
f(βⱼ) = ½(rⱼ - Xⱼβⱼ)ᵀ(rⱼ - Xⱼβⱼ) + λ|βⱼ|
```

Taking the derivative (subgradient):
```
∂f/∂βⱼ = -Xⱼᵀrⱼ + ||Xⱼ||²βⱼ + λ · sign(βⱼ)
```

Setting to zero and solving gives the soft-thresholding update.

### Sparsity Property

Lasso produces sparse solutions (some βⱼ = 0) due to the L1 penalty. This occurs when:
```
|Xⱼᵀrⱼ|/||Xⱼ||² ≤ λ/||Xⱼ||²
```

### Geometric Interpretation

Lasso constraint region is:
```
||β||₁ ≤ t
```

This forms a diamond (in 2D) or hypercube (higher dimensions) with sharp corners at the axes, promoting sparsity.

## 4. Elastic Net Regularization

### Formulation

Combines L1 and L2 penalties:
```
J_elastic(β) = ||Xβ - y||² + λ₁||β||₁ + λ₂||β||₂²
```

Often parameterized as:
```
J_elastic(β) = ||Xβ - y||² + λ[(1-α)||β||₂² + α||β||₁]
```

Where α ∈ [0,1] controls the mixing ratio.

### Properties

- **α = 0**: Pure Ridge regression
- **α = 1**: Pure Lasso regression
- **0 < α < 1**: Combines benefits of both

### Coordinate Descent for Elastic Net

The soft-thresholding becomes:
```
βⱼ = S(Xⱼᵀrⱼ/(||Xⱼ||² + λ(1-α)), λα/(||Xⱼ||² + λ(1-α)))
```

## 5. Regularization Path

### Definition

The regularization path shows how coefficients change as λ varies:
```
β(λ) = argmin_β [||Xβ - y||² + λR(β)]
```

### Ridge Path

For Ridge regression, coefficients shrink smoothly:
```
β_ridge(λ) = (XᵀX + λI)⁻¹Xᵀy
```

As λ → ∞, β_ridge(λ) → 0.

### Lasso Path

Lasso path is piecewise linear. Coefficients can become exactly zero at finite λ values.

### Degrees of Freedom

For Ridge regression:
```
df(λ) = tr[X(XᵀX + λI)⁻¹Xᵀ] = Σⱼ dⱼ²/(dⱼ² + λ)
```

For Lasso:
```
df(λ) = |{j : βⱼ(λ) ≠ 0}|
```

## 6. Cross-Validation for λ Selection

### k-Fold Cross-Validation

1. Split data into k folds
2. For each λ:
   - Train on k-1 folds
   - Validate on remaining fold
   - Compute validation error
3. Select λ with minimum average validation error

### Generalized Cross-Validation (GCV)

For Ridge regression, GCV provides an efficient approximation:
```
GCV(λ) = n||y - Xβ_ridge(λ)||²/[n - df(λ)]²
```

### Information Criteria

**AIC (Akaike Information Criterion)**:
```
AIC(λ) = n log(RSS(λ)/n) + 2df(λ)
```

**BIC (Bayesian Information Criterion)**:
```
BIC(λ) = n log(RSS(λ)/n) + log(n)df(λ)
```

## 7. Bayesian Interpretation

### Ridge as MAP Estimation

Ridge regression corresponds to MAP estimation with Gaussian prior:
```
β ~ N(0, σ²/λ · I)
```

The posterior mode is:
```
β_MAP = (XᵀX + λI)⁻¹Xᵀy
```

### Lasso as MAP Estimation

Lasso corresponds to MAP estimation with Laplace prior:
```
p(βⱼ) ∝ exp(-λ|βⱼ|)
```

### Hierarchical Bayesian Models

More flexible priors can be used:
```
βⱼ|τⱼ ~ N(0, τⱼ)
τⱼ ~ Gamma(a, b)
```

This leads to adaptive regularization.

## 8. Computational Considerations

### Ridge Regression Complexity

- Direct solution: O(p³) for matrix inversion
- SVD approach: O(np²) + O(p³)
- Iterative methods: O(np) per iteration

### Lasso Complexity

- Coordinate descent: O(np) per iteration
- Typically converges in few iterations
- Path algorithms: Compute entire path efficiently

### Warm Starts

Use solution at λₖ as initialization for λₖ₊₁ to speed up convergence.

## 9. Extensions and Variations

### Group Lasso

Penalizes groups of coefficients:
```
J_group(β) = ||Xβ - y||² + λ Σₘ ||βₘ||₂
```

Where βₘ represents coefficients in group m.

### Fused Lasso

Penalizes differences between adjacent coefficients:
```
J_fused(β) = ||Xβ - y||² + λ₁||β||₁ + λ₂ Σⱼ |βⱼ₊₁ - βⱼ|
```

### Adaptive Lasso

Uses weighted L1 penalty:
```
J_adaptive(β) = ||Xβ - y||² + λ Σⱼ wⱼ|βⱼ|
```

Where weights wⱼ = 1/|β̂ⱼ|^γ from initial estimate.

## 10. Theoretical Properties

### Oracle Properties

An estimator has oracle properties if:
1. **Consistency in variable selection**: Correctly identifies non-zero coefficients
2. **Asymptotic normality**: Non-zero coefficients are asymptotically normal

### Conditions for Lasso

Lasso satisfies oracle properties under:
- **Irrepresentable condition**: Design matrix satisfies certain correlation conditions
- **β-min condition**: Non-zero coefficients are sufficiently large

### Minimax Optimality

Under sparsity assumptions, Lasso achieves minimax optimal rates:
```
||β̂ - β||₂² = O(s log p / n)
```

Where s is the number of non-zero coefficients.

## Summary Table

| Method | Penalty | Solution | Key Property |
|--------|---------|----------|--------------|
| Ridge | λ||β||₂² | (XᵀX + λI)⁻¹Xᵀy | Shrinkage, always invertible |
| Lasso | λ||β||₁ | Coordinate descent | Sparsity, feature selection |
| Elastic Net | λ₁||β||₁ + λ₂||β||₂² | Modified coordinate descent | Combines both benefits |
| Group Lasso | λΣₘ||βₘ||₂ | Block coordinate descent | Group sparsity |