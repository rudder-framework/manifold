# GARCH(1,1) Volatility Derivation: Complete Mathematical Proof

## Problem Statement

**Given:** A time series of battery capacity measurements from NASA Li-ion Battery B0047

**Find:** The GARCH(1,1) model parameters that characterize variance dynamics

**Model:**
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where:
- $\omega$ = baseline variance (long-run)
- $\alpha$ = shock impact coefficient
- $\beta$ = variance persistence
- $\epsilon_t$ = innovation (shock) at time t

---

## Step 0: Load the Data


```python
import numpy as np
import polars as pl
np.set_printoptions(precision=6, suppress=True)

# Load battery data
obs = pl.read_parquet('../data/battery_45_48/observations.parquet')
b47 = obs.filter(
    (pl.col('entity_id') == 'B0047') & 
    (pl.col('signal_id') == 'capacity')
).sort('timestamp')

X = b47['value'].to_numpy()
n = len(X)

print(f"GARCH(1,1) Model Estimation")
print(f"="*60)
print(f"\nTotal observations: n = {n}")
print(f"\nFirst 15 capacity values (Ah):")
print(f"\n  {'t':<4} {'xₜ (Ah)':<12}")
print(f"  {'-'*4} {'-'*12}")
for i in range(15):
    print(f"  {i:<4} {X[i]:.6f}")
```

    GARCH(1,1) Model Estimation
    ============================================================
    
    Total observations: n = 69
    
    First 15 capacity values (Ah):
    
      t    xₜ (Ah)     
      ---- ------------
      0    1.674305
      1    1.524366
      2    1.508076
      3    1.483558
      4    1.467139
      5    1.448858
      6    1.445853
      7    1.431118
      8    1.419275
      9    1.399997
      10   1.388516
      11   1.365223
      12   1.406044
      13   1.405754
      14   1.386766


---

## Step 1: Compute Returns (First Differences)

### Definition:
GARCH models variance of returns/changes, not levels:
$$r_t = x_t - x_{t-1}$$

### Solution:


```python
print("Step 1: Compute Returns (First Differences)")
print("="*60)
print(f"\nFormula: rₜ = xₜ - xₜ₋₁")

# Compute returns
r = np.diff(X)  # r[0] = X[1] - X[0], etc.
T = len(r)  # Number of returns

print(f"\nNumber of returns: T = n - 1 = {n} - 1 = {T}")
print(f"\nComputing first 10 returns:")
print(f"\n  {'t':<4} {'xₜ':<12} {'-':<3} {'xₜ₋₁':<12} {'=':<3} {'rₜ':<12}")
print(f"  {'-'*4} {'-'*12} {'-'*3} {'-'*12} {'-'*3} {'-'*12}")
for t in range(10):
    print(f"  {t+1:<4} {X[t+1]:<12.6f} {'-':<3} {X[t]:<12.6f} {'=':<3} {r[t]:+.6f}")

print(f"\nReturn statistics:")
print(f"  Mean(r): μ = {np.mean(r):.6f}")
print(f"  Std(r):  σ = {np.std(r):.6f}")
print(f"  Min(r):     {np.min(r):.6f}")
print(f"  Max(r):     {np.max(r):.6f}")
```

    Step 1: Compute Returns (First Differences)
    ============================================================
    
    Formula: rₜ = xₜ - xₜ₋₁
    
    Number of returns: T = n - 1 = 69 - 1 = 68
    
    Computing first 10 returns:
    
      t    xₜ           -   xₜ₋₁         =   rₜ          
      ---- ------------ --- ------------ --- ------------
      1    1.524366     -   1.674305     =   -0.149939
      2    1.508076     -   1.524366     =   -0.016290
      3    1.483558     -   1.508076     =   -0.024519
      4    1.467139     -   1.483558     =   -0.016419
      5    1.448858     -   1.467139     =   -0.018281
      6    1.445853     -   1.448858     =   -0.003005
      7    1.431118     -   1.445853     =   -0.014735
      8    1.419275     -   1.431118     =   -0.011844
      9    1.399997     -   1.419275     =   -0.019277
      10   1.388516     -   1.399997     =   -0.011482
    
    Return statistics:
      Mean(r): μ = -0.007612
      Std(r):  σ = 0.028486
      Min(r):     -0.149939
      Max(r):     0.084813


---

## Step 2: De-mean the Returns

### Definition:
The GARCH model assumes mean-zero innovations:
$$\epsilon_t = r_t - \mu$$

where $\mu = \frac{1}{T}\sum_{t=1}^{T} r_t$

### Solution:


```python
print("Step 2: De-mean the Returns (Compute Innovations)")
print("="*60)
print(f"\nFormula: εₜ = rₜ - μ")

# Compute mean
mu = np.mean(r)
print(f"\nMean of returns:")
print(f"  μ = (1/T) × Σrₜ")
print(f"    = (1/{T}) × {np.sum(r):.6f}")
print(f"    = {mu:.6f}")

# De-mean
epsilon = r - mu

print(f"\nComputing innovations (first 10):")
print(f"\n  {'t':<4} {'rₜ':<12} {'-':<3} {'μ':<12} {'=':<3} {'εₜ':<12}")
print(f"  {'-'*4} {'-'*12} {'-'*3} {'-'*12} {'-'*3} {'-'*12}")
for t in range(10):
    print(f"  {t+1:<4} {r[t]:+.6f}    {'-':<3} {mu:.6f}     {'=':<3} {epsilon[t]:+.6f}")

print(f"\nVerification: Mean(ε) = {np.mean(epsilon):.10f} ≈ 0 ✓")
```

    Step 2: De-mean the Returns (Compute Innovations)
    ============================================================
    
    Formula: εₜ = rₜ - μ
    
    Mean of returns:
      μ = (1/T) × Σrₜ
        = (1/68) × -0.517596
        = -0.007612
    
    Computing innovations (first 10):
    
      t    rₜ           -   μ            =   εₜ          
      ---- ------------ --- ------------ --- ------------
      1    -0.149939    -   -0.007612     =   -0.142327
      2    -0.016290    -   -0.007612     =   -0.008678
      3    -0.024519    -   -0.007612     =   -0.016907
      4    -0.016419    -   -0.007612     =   -0.008807
      5    -0.018281    -   -0.007612     =   -0.010669
      6    -0.003005    -   -0.007612     =   +0.004607
      7    -0.014735    -   -0.007612     =   -0.007123
      8    -0.011844    -   -0.007612     =   -0.004232
      9    -0.019277    -   -0.007612     =   -0.011665
      10   -0.011482    -   -0.007612     =   -0.003870
    
    Verification: Mean(ε) = 0.0000000000 ≈ 0 ✓


---

## Step 3: Compute Squared Innovations

### Definition:
GARCH models the dynamics of squared innovations (variance proxy):
$$\epsilon_t^2$$

### Solution:


```python
print("Step 3: Compute Squared Innovations")
print("="*60)
print(f"\nFormula: εₜ² (proxy for instantaneous variance)")

# Squared innovations
epsilon_sq = epsilon ** 2

print(f"\nComputing squared innovations (first 15):")
print(f"\n  {'t':<4} {'εₜ':<14} {'εₜ²':<14}")
print(f"  {'-'*4} {'-'*14} {'-'*14}")
for t in range(15):
    print(f"  {t+1:<4} {epsilon[t]:+.6f}      {epsilon_sq[t]:.8f}")

print(f"\nSquared innovation statistics:")
print(f"  Mean(ε²):    {np.mean(epsilon_sq):.8f}")
print(f"  Variance(ε): {np.var(epsilon):.8f}")
print(f"  Max(ε²):     {np.max(epsilon_sq):.8f}")
```

    Step 3: Compute Squared Innovations
    ============================================================
    
    Formula: εₜ² (proxy for instantaneous variance)
    
    Computing squared innovations (first 15):
    
      t    εₜ             εₜ²           
      ---- -------------- --------------
      1    -0.142327      0.02025693
      2    -0.008678      0.00007531
      3    -0.016907      0.00028584
      4    -0.008807      0.00007756
      5    -0.010669      0.00011383
      6    +0.004607      0.00002122
      7    -0.007123      0.00005074
      8    -0.004232      0.00001791
      9    -0.011665      0.00013608
      10   -0.003870      0.00001498
      11   -0.015680      0.00024588
      12   +0.048432      0.00234570
      13   +0.007322      0.00005361
      14   -0.011376      0.00012942
      15   -0.008649      0.00007481
    
    Squared innovation statistics:
      Mean(ε²):    0.00081148
      Variance(ε): 0.00081148
      Max(ε²):     0.02025693


---

## Step 4: GARCH(1,1) Model Specification

### Model:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### Constraints:
- $\omega > 0$ (positive baseline variance)
- $\alpha \geq 0$ (non-negative shock impact)
- $\beta \geq 0$ (non-negative persistence)
- $\alpha + \beta < 1$ (stationarity condition)

### Unconditional Variance:
$$\bar{\sigma}^2 = \frac{\omega}{1 - \alpha - \beta}$$


```python
print("Step 4: GARCH(1,1) Model Specification")
print("="*60)
print(f"""
Model: σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²

Parameters to estimate:
  ω (omega): Baseline variance component
  α (alpha): Shock impact coefficient  
  β (beta):  Variance persistence coefficient

Constraints:
  ω > 0
  α ≥ 0
  β ≥ 0  
  α + β < 1 (stationarity)

Key quantities:
  Persistence = α + β
  Unconditional variance = ω / (1 - α - β)
""")
```

    Step 4: GARCH(1,1) Model Specification
    ============================================================
    
    Model: σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²
    
    Parameters to estimate:
      ω (omega): Baseline variance component
      α (alpha): Shock impact coefficient  
      β (beta):  Variance persistence coefficient
    
    Constraints:
      ω > 0
      α ≥ 0
      β ≥ 0  
      α + β < 1 (stationarity)
    
    Key quantities:
      Persistence = α + β
      Unconditional variance = ω / (1 - α - β)
    


---

## Step 5: Maximum Likelihood Estimation Setup

### Log-Likelihood Function:
Assuming Gaussian innovations:
$$\mathcal{L}(\omega, \alpha, \beta) = -\frac{T}{2}\ln(2\pi) - \frac{1}{2}\sum_{t=1}^{T}\left[\ln(\sigma_t^2) + \frac{\epsilon_t^2}{\sigma_t^2}\right]$$

### Solution:


```python
print("Step 5: Maximum Likelihood Estimation")
print("="*60)
print(f"""
Log-Likelihood (Gaussian):

  L(ω,α,β) = -(T/2)·ln(2π) - (1/2)·Σ[ln(σₜ²) + εₜ²/σₜ²]

We maximize this by iterating over possible parameter values.
""")

def garch_variance(epsilon_sq, omega, alpha, beta):
    """Compute GARCH(1,1) conditional variances."""
    T = len(epsilon_sq)
    sigma2 = np.zeros(T)
    
    # Initialize with unconditional variance
    sigma2[0] = np.var(epsilon_sq[:10]) if omega / (1 - alpha - beta) <= 0 else omega / (1 - alpha - beta)
    
    # Recursion
    for t in range(1, T):
        sigma2[t] = omega + alpha * epsilon_sq[t-1] + beta * sigma2[t-1]
    
    return sigma2

def garch_log_likelihood(params, epsilon_sq):
    """Compute negative log-likelihood for GARCH(1,1)."""
    omega, alpha, beta = params
    
    # Constraints
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    
    T = len(epsilon_sq)
    sigma2 = garch_variance(epsilon_sq, omega, alpha, beta)
    
    # Avoid numerical issues
    sigma2 = np.maximum(sigma2, 1e-10)
    
    # Log-likelihood
    ll = -0.5 * np.sum(np.log(sigma2) + epsilon_sq / sigma2)
    
    return -ll  # Return negative for minimization

print("Log-likelihood function defined.")
```

    Step 5: Maximum Likelihood Estimation
    ============================================================
    
    Log-Likelihood (Gaussian):
    
      L(ω,α,β) = -(T/2)·ln(2π) - (1/2)·Σ[ln(σₜ²) + εₜ²/σₜ²]
    
    We maximize this by iterating over possible parameter values.
    
    Log-likelihood function defined.


---

## Step 6: Grid Search for Initial Parameters


```python
print("Step 6: Grid Search for Parameter Estimation")
print("="*60)

# Sample variance as starting point
sample_var = np.var(epsilon)
print(f"\nSample variance of innovations: σ² = {sample_var:.8f}")

# Grid search
best_ll = -np.inf
best_params = None

print(f"\nSearching parameter space:")
print(f"  ω ∈ [0.00001, 0.001]")
print(f"  α ∈ [0.01, 0.3]")
print(f"  β ∈ [0.5, 0.95]")

omega_grid = np.linspace(0.00001, 0.001, 20)
alpha_grid = np.linspace(0.01, 0.3, 15)
beta_grid = np.linspace(0.5, 0.95, 15)

results = []
for omega in omega_grid:
    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha + beta < 1:
                neg_ll = garch_log_likelihood([omega, alpha, beta], epsilon_sq)
                ll = -neg_ll
                results.append((omega, alpha, beta, ll))
                if ll > best_ll:
                    best_ll = ll
                    best_params = (omega, alpha, beta)

omega_opt, alpha_opt, beta_opt = best_params
persistence = alpha_opt + beta_opt
unconditional_var = omega_opt / (1 - persistence)

print(f"\nGrid search complete ({len(results)} parameter combinations tested)")
print(f"\n" + "-"*60)
print(f"Best parameters found:")
print(f"\n  ω (omega) = {omega_opt:.8f}")
print(f"  α (alpha) = {alpha_opt:.6f}")
print(f"  β (beta)  = {beta_opt:.6f}")
print(f"\n  Persistence = α + β = {alpha_opt:.6f} + {beta_opt:.6f} = {persistence:.6f}")
print(f"  Unconditional variance = ω/(1-α-β) = {unconditional_var:.8f}")
print(f"\n  Log-likelihood = {best_ll:.4f}")
```

    Step 6: Grid Search for Parameter Estimation
    ============================================================
    
    Sample variance of innovations: σ² = 0.00081148
    
    Searching parameter space:
      ω ∈ [0.00001, 0.001]
      α ∈ [0.01, 0.3]
      β ∈ [0.5, 0.95]
    
    Grid search complete (3360 parameter combinations tested)
    
    ------------------------------------------------------------
    Best parameters found:
    
      ω (omega) = 0.00037474
      α (alpha) = 0.010000
      β (beta)  = 0.532143
    
      Persistence = α + β = 0.010000 + 0.532143 = 0.542143
      Unconditional variance = ω/(1-α-β) = 0.00081846
    
      Log-likelihood = 207.6700


---

## Step 7: Detailed Variance Recursion

### Formula:
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

### Solution with actual values:


```python
print("Step 7: Detailed Variance Recursion")
print("="*60)
print(f"\nUsing estimated parameters:")
print(f"  ω = {omega_opt:.8f}")
print(f"  α = {alpha_opt:.6f}")
print(f"  β = {beta_opt:.6f}")

print(f"\nFormula: σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²")
print(f"\nStep-by-step variance computation:")

# Compute variances
sigma2 = garch_variance(epsilon_sq, omega_opt, alpha_opt, beta_opt)

print(f"\nInitialization:")
print(f"  σ₀² = unconditional variance = ω/(1-α-β) = {sigma2[0]:.8f}")

print(f"\n" + "-"*60)
print(f"Recursion (first 10 steps):")
print(f"\n  {'t':<4} {'Formula':<60} {'σₜ²':<14}")
print(f"  {'-'*4} {'-'*60} {'-'*14}")

for t in range(1, min(11, T)):
    term1 = omega_opt
    term2 = alpha_opt * epsilon_sq[t-1]
    term3 = beta_opt * sigma2[t-1]
    
    formula = f"{omega_opt:.6f} + {alpha_opt:.4f}×{epsilon_sq[t-1]:.6f} + {beta_opt:.4f}×{sigma2[t-1]:.6f}"
    print(f"  {t:<4} {formula:<60} {sigma2[t]:.8f}")

print(f"\n" + "-"*60)
print(f"\nExpanded calculation for t=2:")
t = 2
print(f"\n  σ₂² = ω + α·ε₁² + β·σ₁²")
print(f"\n      = {omega_opt:.8f}")
print(f"        + {alpha_opt:.6f} × {epsilon_sq[t-1]:.8f}")
print(f"        + {beta_opt:.6f} × {sigma2[t-1]:.8f}")
print(f"\n      = {omega_opt:.8f}")
print(f"        + {alpha_opt * epsilon_sq[t-1]:.8f}")
print(f"        + {beta_opt * sigma2[t-1]:.8f}")
print(f"\n      = {sigma2[t]:.8f}")
```

    Step 7: Detailed Variance Recursion
    ============================================================
    
    Using estimated parameters:
      ω = 0.00037474
      α = 0.010000
      β = 0.532143
    
    Formula: σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²
    
    Step-by-step variance computation:
    
    Initialization:
      σ₀² = unconditional variance = ω/(1-α-β) = 0.00081846
    
    ------------------------------------------------------------
    Recursion (first 10 steps):
    
      t    Formula                                                      σₜ²           
      ---- ------------------------------------------------------------ --------------
      1    0.000375 + 0.0100×0.020257 + 0.5321×0.000818                 0.00101284
      2    0.000375 + 0.0100×0.000075 + 0.5321×0.001013                 0.00091447
      3    0.000375 + 0.0100×0.000286 + 0.5321×0.000914                 0.00086422
      4    0.000375 + 0.0100×0.000078 + 0.5321×0.000864                 0.00083540
      5    0.000375 + 0.0100×0.000114 + 0.5321×0.000835                 0.00082043
      6    0.000375 + 0.0100×0.000021 + 0.5321×0.000820                 0.00081153
      7    0.000375 + 0.0100×0.000051 + 0.5321×0.000812                 0.00080710
      8    0.000375 + 0.0100×0.000018 + 0.5321×0.000807                 0.00080441
      9    0.000375 + 0.0100×0.000136 + 0.5321×0.000804                 0.00080416
      10   0.000375 + 0.0100×0.000015 + 0.5321×0.000804                 0.00080281
    
    ------------------------------------------------------------
    
    Expanded calculation for t=2:
    
      σ₂² = ω + α·ε₁² + β·σ₁²
    
          = 0.00037474
            + 0.010000 × 0.00007531
            + 0.532143 × 0.00101284
    
          = 0.00037474
            + 0.00000075
            + 0.00053898
    
          = 0.00091447


---

## Step 8: Compute Conditional Volatility

### Definition:
$$\sigma_t = \sqrt{\sigma_t^2}$$


```python
print("Step 8: Conditional Volatility (Standard Deviation)")
print("="*60)
print(f"\nFormula: σₜ = √σₜ²")

sigma = np.sqrt(sigma2)

print(f"\nConditional volatility (first 15 periods):")
print(f"\n  {'t':<4} {'σₜ²':<14} {'σₜ':<14}")
print(f"  {'-'*4} {'-'*14} {'-'*14}")
for t in range(15):
    print(f"  {t+1:<4} {sigma2[t]:.8f}     {sigma[t]:.8f}")

print(f"\nVolatility statistics:")
print(f"  Mean(σ):  {np.mean(sigma):.6f}")
print(f"  Min(σ):   {np.min(sigma):.6f}")
print(f"  Max(σ):   {np.max(sigma):.6f}")
```

    Step 8: Conditional Volatility (Standard Deviation)
    ============================================================
    
    Formula: σₜ = √σₜ²
    
    Conditional volatility (first 15 periods):
    
      t    σₜ²            σₜ            
      ---- -------------- --------------
      1    0.00081846     0.02860870
      2    0.00101284     0.03182519
      3    0.00091447     0.03024015
      4    0.00086422     0.02939766
      5    0.00083540     0.02890332
      6    0.00082043     0.02864312
      7    0.00081153     0.02848744
      8    0.00080710     0.02840944
      9    0.00080441     0.02836206
      10   0.00080416     0.02835766
      11   0.00080281     0.02833395
      12   0.00080441     0.02836207
      13   0.00082625     0.02874462
      14   0.00081496     0.02854746
      15   0.00080970     0.02845531
    
    Volatility statistics:
      Mean(σ):  0.028608
      Min(σ):   0.028307
      Max(σ):   0.031825


---

## Step 9: Interpretation of Parameters

### Key Metrics:

| Parameter | Meaning |
|-----------|--------|
| ω (omega) | Baseline variance contribution |
| α (alpha) | Impact of recent shock on variance |
| β (beta) | Persistence of past variance |
| α + β | Total persistence (half-life of shocks) |


```python
print("Step 9: Physical Interpretation")
print("="*60)

print(f"\nEstimated GARCH(1,1) Parameters:")
print(f"\n  ω (omega) = {omega_opt:.8f}")
print(f"  α (alpha) = {alpha_opt:.6f}")
print(f"  β (beta)  = {beta_opt:.6f}")

print(f"\n" + "-"*60)
print(f"Interpretation:")

# Persistence
print(f"\n1. PERSISTENCE = α + β = {persistence:.6f}")
if persistence > 0.95:
    print(f"   → Very high persistence (>0.95)")
    print(f"   → Variance shocks decay very slowly")
    print(f"   → Near-integrated GARCH (IGARCH)")
elif persistence > 0.85:
    print(f"   → High persistence (0.85-0.95)")
    print(f"   → Variance shocks persist for many periods")
elif persistence > 0.5:
    print(f"   → Moderate persistence (0.5-0.85)")
    print(f"   → Variance shocks decay at moderate rate")
else:
    print(f"   → Low persistence (<0.5)")
    print(f"   → Variance shocks decay quickly")

# Half-life
if persistence > 0 and persistence < 1:
    half_life = np.log(0.5) / np.log(persistence)
    print(f"\n   Half-life of variance shocks: {half_life:.1f} periods")

# Alpha interpretation
print(f"\n2. α (ARCH effect) = {alpha_opt:.6f}")
print(f"   → {alpha_opt*100:.1f}% of variance comes from most recent shock")

# Beta interpretation  
print(f"\n3. β (GARCH effect) = {beta_opt:.6f}")
print(f"   → {beta_opt*100:.1f}% of variance inherited from previous period")

# Unconditional variance
print(f"\n4. Unconditional variance = {unconditional_var:.8f}")
print(f"   Unconditional volatility = {np.sqrt(unconditional_var):.6f}")

print(f"\n" + "-"*60)
print(f"For Battery B0047 capacity:")
print(f"  • {'High' if persistence > 0.85 else 'Moderate'} variance persistence")
print(f"  • Capacity changes show {'stable' if persistence > 0.85 else 'moderate'} volatility dynamics")
print(f"  • Shocks to variance decay over ~{half_life:.0f} cycles" if persistence < 1 else "")
```

    Step 9: Physical Interpretation
    ============================================================
    
    Estimated GARCH(1,1) Parameters:
    
      ω (omega) = 0.00037474
      α (alpha) = 0.010000
      β (beta)  = 0.532143
    
    ------------------------------------------------------------
    Interpretation:
    
    1. PERSISTENCE = α + β = 0.542143
       → Moderate persistence (0.5-0.85)
       → Variance shocks decay at moderate rate
    
       Half-life of variance shocks: 1.1 periods
    
    2. α (ARCH effect) = 0.010000
       → 1.0% of variance comes from most recent shock
    
    3. β (GARCH effect) = 0.532143
       → 53.2% of variance inherited from previous period
    
    4. Unconditional variance = 0.00081846
       Unconditional volatility = 0.028609
    
    ------------------------------------------------------------
    For Battery B0047 capacity:
      • Moderate variance persistence
      • Capacity changes show moderate volatility dynamics
      • Shocks to variance decay over ~1 cycles


---

## Summary

### Complete Solution


```python
print("="*70)
print("COMPLETE SOLUTION SUMMARY")
print("="*70)

half_life_str = f"{half_life:.1f}" if persistence < 1 else "∞"

print(f"""
GIVEN:
  • Battery B0047 capacity time series
  • n = {n} observations, T = {T} returns
  • Range: [{X.min():.4f}, {X.max():.4f}] Ah

MODEL: GARCH(1,1)
  σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²

STEPS:
  1. Compute returns: rₜ = xₜ - xₜ₋₁
     Mean return: μ = {mu:.6f}
  
  2. Compute innovations: εₜ = rₜ - μ
     Sample variance: {sample_var:.8f}
  
  3. Maximum Likelihood Estimation:
     Maximize: L = -(T/2)·ln(2π) - (1/2)·Σ[ln(σₜ²) + εₜ²/σₜ²]
  
  4. Variance recursion with optimal parameters:
     σₜ² = {omega_opt:.8f} + {alpha_opt:.6f}·εₜ₋₁² + {beta_opt:.6f}·σₜ₋₁²

DETAILED CALCULATION (t=2):
  σ₂² = ω + α·ε₁² + β·σ₁²
      = {omega_opt:.8f} + {alpha_opt:.6f}×{epsilon_sq[1]:.8f} + {beta_opt:.6f}×{sigma2[1]:.8f}
      = {sigma2[2]:.8f}

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   GARCH(1,1) PARAMETERS:                                       ║
║                                                                ║
║     ω (omega) = {omega_opt:.8f}                               ║
║     α (alpha) = {alpha_opt:.6f}                                 ║
║     β (beta)  = {beta_opt:.6f}                                 ║
║                                                                ║
║   PERSISTENCE = α + β = {persistence:.6f}                       ║
║   HALF-LIFE = {half_life_str} periods                                   ║
║                                                                ║
║   INTERPRETATION: {'High' if persistence > 0.85 else 'Moderate'} variance persistence             ║
║   Battery capacity volatility is {'stable' if persistence > 0.85 else 'moderately dynamic'}              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
```

    ======================================================================
    COMPLETE SOLUTION SUMMARY
    ======================================================================
    
    GIVEN:
      • Battery B0047 capacity time series
      • n = 69 observations, T = 68 returns
      • Range: [1.1060, 1.6743] Ah
    
    MODEL: GARCH(1,1)
      σₜ² = ω + α·εₜ₋₁² + β·σₜ₋₁²
    
    STEPS:
      1. Compute returns: rₜ = xₜ - xₜ₋₁
         Mean return: μ = -0.007612
    
      2. Compute innovations: εₜ = rₜ - μ
         Sample variance: 0.00081148
    
      3. Maximum Likelihood Estimation:
         Maximize: L = -(T/2)·ln(2π) - (1/2)·Σ[ln(σₜ²) + εₜ²/σₜ²]
    
      4. Variance recursion with optimal parameters:
         σₜ² = 0.00037474 + 0.010000·εₜ₋₁² + 0.532143·σₜ₋₁²
    
    DETAILED CALCULATION (t=2):
      σ₂² = ω + α·ε₁² + β·σ₁²
          = 0.00037474 + 0.010000×0.00007531 + 0.532143×0.00101284
          = 0.00091447
    
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   GARCH(1,1) PARAMETERS:                                       ║
    ║                                                                ║
    ║     ω (omega) = 0.00037474                               ║
    ║     α (alpha) = 0.010000                                 ║
    ║     β (beta)  = 0.532143                                 ║
    ║                                                                ║
    ║   PERSISTENCE = α + β = 0.542143                       ║
    ║   HALF-LIFE = 1.1 periods                                   ║
    ║                                                                ║
    ║   INTERPRETATION: Moderate variance persistence             ║
    ║   Battery capacity volatility is moderately dynamic              ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    


---

*PRISM Behavioral Geometry Engine - Mathematical Derivation Proof*

*Battery: NASA B0047 | Signal: Capacity (Ah) | Method: GARCH(1,1) Volatility Model*
