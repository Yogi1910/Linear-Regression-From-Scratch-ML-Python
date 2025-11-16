# Mathematical Derivation of Linear Regression

This document provides step-by-step mathematical derivations for linear regression algorithms.

## 1. Ordinary Least Squares (OLS) Derivation

### Problem Setup

Given:
- Feature matrix: **X** ∈ ℝⁿˣᵖ (n samples, p features)
- Target vector: **y** ∈ ℝⁿ
- Parameter vector: **β** ∈ ℝᵖ

We want to find **β** that minimizes the sum of squared residuals.

### Step 1: Define the Cost Function

The cost function (sum of squared errors) is:

```
J(β) = ½||Xβ - y||²
```

Expanding this:
```
J(β) = ½(Xβ - y)ᵀ(Xβ - y)
```

### Step 2: Expand the Cost Function

```
J(β) = ½[(Xβ)ᵀ(Xβ) - (Xβ)ᵀy - yᵀ(Xβ) + yᵀy]
```

Since (Xβ)ᵀy = yᵀ(Xβ) (both are scalars):
```
J(β) = ½[βᵀXᵀXβ - 2yᵀXβ + yᵀy]
```

### Step 3: Take the Derivative

To minimize J(β), we take the derivative with respect to β:

```
∂J(β)/∂β = ½[2XᵀXβ - 2Xᵀy]
         = XᵀXβ - Xᵀy
```

### Step 4: Set Derivative to Zero

For the minimum:
```
∂J(β)/∂β = 0
XᵀXβ - Xᵀy = 0
XᵀXβ = Xᵀy
```

### Step 5: Solve for β (Normal Equation)

If XᵀX is invertible:
```
β = (XᵀX)⁻¹Xᵀy
```

This is the **Normal Equation**.

### Step 6: Verify Second Derivative (Convexity)

The second derivative (Hessian) is:
```
∂²J(β)/∂β² = XᵀX
```

Since XᵀX is positive semi-definite, J(β) is convex, confirming our solution is a global minimum.

## 2. Gradient Descent Derivation

### Algorithm Setup

Instead of solving analytically, we can use iterative optimization:

```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - α∇J(β⁽ᵗ⁾)
```

Where α is the learning rate.

### Gradient Calculation

From our previous derivation:
```
∇J(β) = XᵀXβ - Xᵀy = Xᵀ(Xβ - y)
```

### Update Rule

```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - α·Xᵀ(Xβ⁽ᵗ⁾ - y)
```

For computational efficiency, we often use:
```
β⁽ᵗ⁺¹⁾ = β⁽ᵗ⁾ - (α/m)·Xᵀ(Xβ⁽ᵗ⁾ - y)
```

Where m is the number of samples.

## 3. Ridge Regression Derivation

### Modified Cost Function

Ridge regression adds an L2 penalty term:
```
J(β) = ½||Xβ - y||² + ½λ||β||²
```

Where λ > 0 is the regularization parameter.

### Expanded Cost Function

```
J(β) = ½(Xβ - y)ᵀ(Xβ - y) + ½λβᵀβ
     = ½[βᵀXᵀXβ - 2yᵀXβ + yᵀy + λβᵀβ]
```

### Gradient Calculation

```
∂J(β)/∂β = XᵀXβ - Xᵀy + λβ
         = (XᵀX + λI)β - Xᵀy
```

### Ridge Solution

Setting the gradient to zero:
```
(XᵀX + λI)β = Xᵀy
β = (XᵀX + λI)⁻¹Xᵀy
```

This is the **Ridge Regression Solution**.

### Properties of Ridge Solution

1. **Always invertible**: XᵀX + λI is always positive definite for λ > 0
2. **Shrinkage**: Coefficients are shrunk towards zero
3. **Bias-variance tradeoff**: Introduces bias but reduces variance

## 4. Lasso Regression Derivation

### Cost Function

Lasso uses L1 regularization:
```
J(β) = ½||Xβ - y||² + λ||β||₁
```

Where ||β||₁ = Σᵢ|βᵢ| is the L1 norm.

### Subgradient

The L1 norm is not differentiable at zero, so we use subgradients:

```
∂|βⱼ|/∂βⱼ = {
  +1    if βⱼ > 0
  -1    if βⱼ < 0
  [-1,1] if βⱼ = 0
}
```

### Coordinate Descent Solution

Lasso is typically solved using coordinate descent. For each coordinate j:

1. **Compute partial residual**:
   ```
   rⱼ = y - Σₖ≠ⱼ Xₖβₖ
   ```

2. **Update coordinate**:
   ```
   βⱼ = S(Xⱼᵀrⱼ/||Xⱼ||², λ/||Xⱼ||²)
   ```

Where S(z, γ) is the soft-thresholding operator:
```
S(z, γ) = {
  z - γ  if z > γ
  0      if |z| ≤ γ
  z + γ  if z < -γ
}
```

## 5. Geometric Interpretation

### Projection Matrix

The OLS solution can be viewed as projecting y onto the column space of X:

```
ŷ = Xβ = X(XᵀX)⁻¹Xᵀy = Py
```

Where P = X(XᵀX)⁻¹Xᵀ is the projection matrix.

### Properties of Projection Matrix

1. **Idempotent**: P² = P
2. **Symmetric**: P = Pᵀ
3. **Residual orthogonality**: Xᵀ(y - ŷ) = 0

### Residual Vector

The residual vector is:
```
e = y - ŷ = y - Py = (I - P)y
```

This is orthogonal to the column space of X.

## 6. Statistical Properties

### Unbiasedness

Under the assumption E[ε] = 0:
```
E[β̂] = E[(XᵀX)⁻¹Xᵀy]
     = E[(XᵀX)⁻¹Xᵀ(Xβ + ε)]
     = β + (XᵀX)⁻¹XᵀE[ε]
     = β
```

So the OLS estimator is unbiased.

### Variance

Under the assumption Var(ε) = σ²I:
```
Var(β̂) = Var[(XᵀX)⁻¹Xᵀy]
        = (XᵀX)⁻¹XᵀVar(y)X(XᵀX)⁻¹
        = (XᵀX)⁻¹Xᵀσ²IX(XᵀX)⁻¹
        = σ²(XᵀX)⁻¹
```

### Gauss-Markov Theorem

Under the classical linear regression assumptions, the OLS estimator is the **Best Linear Unbiased Estimator (BLUE)**.

## 7. Maximum Likelihood Derivation

### Assumptions

Assume errors are normally distributed:
```
εᵢ ~ N(0, σ²)
```

Therefore:
```
yᵢ ~ N(Xᵢβ, σ²)
```

### Likelihood Function

```
L(β, σ²) = ∏ᵢ₌₁ⁿ (1/√(2πσ²)) exp(-(yᵢ - Xᵢβ)²/(2σ²))
```

### Log-Likelihood

```
ℓ(β, σ²) = -n/2 log(2πσ²) - 1/(2σ²) Σᵢ₌₁ⁿ(yᵢ - Xᵢβ)²
```

### Maximizing Log-Likelihood

Taking the derivative with respect to β and setting to zero:
```
∂ℓ/∂β = 1/σ² Σᵢ₌₁ⁿ Xᵢ(yᵢ - Xᵢβ) = 0
```

This gives us the same normal equation:
```
β̂ = (XᵀX)⁻¹Xᵀy
```

### MLE for σ²

```
∂ℓ/∂σ² = -n/(2σ²) + 1/(2σ⁴) Σᵢ₌₁ⁿ(yᵢ - Xᵢβ̂)² = 0
```

Solving:
```
σ̂² = 1/n Σᵢ₌₁ⁿ(yᵢ - Xᵢβ̂)²
```

## 8. Bayesian Linear Regression

### Prior Distribution

Assume a normal prior for β:
```
β ~ N(μ₀, Σ₀)
```

### Posterior Distribution

Given the likelihood and prior, the posterior is:
```
β|y ~ N(μₙ, Σₙ)
```

Where:
```
Σₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹
μₙ = Σₙ(Σ₀⁻¹μ₀ + σ⁻²Xᵀy)
```

### MAP Estimate

The Maximum A Posteriori (MAP) estimate is:
```
β̂ₘₐₚ = μₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹(Σ₀⁻¹μ₀ + σ⁻²Xᵀy)
```

With μ₀ = 0 and Σ₀ = λ⁻¹I, this reduces to Ridge regression!

## Summary of Key Results

| Method | Solution | Key Property |
|--------|----------|--------------|
| OLS | β̂ = (XᵀX)⁻¹Xᵀy | Unbiased, minimum variance |
| Ridge | β̂ = (XᵀX + λI)⁻¹Xᵀy | Biased, reduced variance |
| Lasso | Coordinate descent | Sparse solutions |
| Bayesian | β̂ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹(Σ₀⁻¹μ₀ + σ⁻²Xᵀy) | Incorporates prior knowledge |