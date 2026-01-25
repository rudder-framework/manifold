# Hurst Exponent Derivation: Complete Mathematical Proof

## Problem Statement

**Given:** A time series of battery capacity measurements from NASA Li-ion Battery B0047

**Find:** The Hurst exponent H using Rescaled Range (R/S) Analysis

**Method:** We will compute H by:
1. Computing R/S statistics for multiple window sizes
2. Fitting the relationship: $\log(R/S) = H \cdot \log(n) + c$
3. The slope H indicates long-range dependence

---

## Step 0: Load Data and Configure Window

**⚙️ CONFIGURABLE PARAMETERS:**
- `WINDOW_START`: Starting index for detailed calculation
- `WINDOW_SIZE`: Size of window for R/S demonstration
- `ENTITY_ID`: Which battery to analyze


```python
import numpy as np
import polars as pl
np.set_printoptions(precision=6, suppress=True)

# ============================================================
# ⚙️ USER CONFIGURABLE PARAMETERS - CHANGE THESE!
# ============================================================
ENTITY_ID = 'B0047'      # Battery: B0045, B0046, B0047, or B0048
WINDOW_START = 19        # Starting cycle index (0-based)
WINDOW_SIZE = 8          # Window size for detailed calculation
# ============================================================

# Load battery data
obs = pl.read_parquet('../data/battery_45_48/observations.parquet')
battery = obs.filter(
    (pl.col('entity_id') == ENTITY_ID) & 
    (pl.col('signal_id') == 'capacity')
).sort('timestamp')

X = battery['value'].to_numpy()
n = len(X)

print(f"╔════════════════════════════════════════════════════════════════╗")
print(f"║  HURST EXPONENT DERIVATION                                     ║")
print(f"╠════════════════════════════════════════════════════════════════╣")
print(f"║  Battery:      {ENTITY_ID:<48} ║")
print(f"║  Total cycles: {n:<48} ║")
print(f"║  Window start: cycle {WINDOW_START+1} (index {WINDOW_START}){' '*30} ║")
print(f"║  Window size:  {WINDOW_SIZE} cycles{' '*40} ║")
print(f"║  Window range: cycles {WINDOW_START+1} to {WINDOW_START+WINDOW_SIZE}{' '*32} ║")
print(f"╚════════════════════════════════════════════════════════════════╝")

print(f"\nComplete dataset X (capacity in Ah):")
for i in range(0, n, 10):
    chunk = X[i:min(i+10, n)]
    vals = ', '.join([f'{v:.4f}' for v in chunk])
    marker = " ← WINDOW" if WINDOW_START >= i and WINDOW_START < i+10 else ""
    print(f"  x[{i:2d}:{min(i+10,n):2d}] = [{vals}]{marker}")
```

    ╔════════════════════════════════════════════════════════════════╗
    ║  HURST EXPONENT DERIVATION                                     ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Battery:      B0047                                            ║
    ║  Total cycles: 69                                               ║
    ║  Window start: cycle 20 (index 19)                               ║
    ║  Window size:  8 cycles                                         ║
    ║  Window range: cycles 20 to 27                                 ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Complete dataset X (capacity in Ah):
      x[ 0:10] = [1.6743, 1.5244, 1.5081, 1.4836, 1.4671, 1.4489, 1.4459, 1.4311, 1.4193, 1.4000]
      x[10:20] = [1.3885, 1.3652, 1.4060, 1.4058, 1.3868, 1.3705, 1.3497, 1.3252, 1.3112, 1.3394] ← WINDOW
      x[20:30] = [1.2849, 1.2817, 1.2600, 1.2661, 1.2414, 1.2299, 1.2281, 1.2140, 1.2173, 1.2073]
      x[30:40] = [1.1880, 1.1863, 1.2614, 1.2648, 1.2466, 1.2334, 1.2129, 1.1999, 1.1903, 1.1893]
      x[40:50] = [1.1671, 1.1615, 1.1556, 1.1489, 1.1436, 1.1418, 1.1361, 1.1236, 1.1320, 1.1170]
      x[50:60] = [1.1200, 1.1060, 1.1908, 1.2012, 1.1907, 1.1764, 1.1628, 1.1516, 1.1584, 1.1692]
      x[60:69] = [1.1588, 1.1440, 1.1384, 1.2213, 1.2031, 1.1897, 1.1774, 1.1584, 1.1567]


---

## Step 1: Extract Window Data

We extract the specific window of data for our detailed R/S calculation demonstration.


```python
# Extract window
s = WINDOW_SIZE
start_idx = WINDOW_START
end_idx = start_idx + s

window = X[start_idx:end_idx]

print(f"Step 1: Extract Window Data")
print(f"="*60)
print(f"\nWindow Configuration:")
print(f"  Start index:  {start_idx} (cycle {start_idx+1})")
print(f"  End index:    {end_idx-1} (cycle {end_idx})")
print(f"  Window size:  s = {s}")

print(f"\n{'='*60}")
print("GIVEN DATA FOR THIS WINDOW")
print('='*60)
print(f"\n  Notation: x = [x₀, x₁, ..., x{s-1}]")
print(f"\n  Actual values:")
print(f"  x = [{', '.join([f'{v:.6f}' for v in window])}]")

print(f"\nTable form:")
print(f"\n  {'i':<4} {'Cycle':<8} {'xᵢ (Ah)':<14} {'Description':<20}")
print(f"  {'-'*4} {'-'*8} {'-'*14} {'-'*20}")
for i, v in enumerate(window):
    cycle = start_idx + i + 1
    if i == 0:
        desc = "← Window start"
    elif i == s-1:
        desc = "← Window end"
    else:
        desc = ""
    print(f"  {i:<4} {cycle:<8} {v:<14.6f} {desc:<20}")
```

    Step 1: Extract Window Data
    ============================================================
    
    Window Configuration:
      Start index:  19 (cycle 20)
      End index:    26 (cycle 27)
      Window size:  s = 8
    
    ============================================================
    GIVEN DATA FOR THIS WINDOW
    ============================================================
    
      Notation: x = [x₀, x₁, ..., x7]
    
      Actual values:
      x = [1.339423, 1.284916, 1.281719, 1.260005, 1.266070, 1.241359, 1.229887, 1.228089]
    
    Table form:
    
      i    Cycle    xᵢ (Ah)        Description         
      ---- -------- -------------- --------------------
      0    20       1.339423       ← Window start      
      1    21       1.284916                           
      2    22       1.281719                           
      3    23       1.260005                           
      4    24       1.266070                           
      5    25       1.241359                           
      6    26       1.229887                           
      7    27       1.228089       ← Window end        


---

## Step 2: Compute the Mean

### Formula:
$$\bar{x} = \frac{1}{s} \sum_{i=0}^{s-1} x_i$$

### Solution:


```python
print("Step 2: Compute the Mean")
print("="*60)
print(f"\nFormula: x̄ = (1/s) × Σxᵢ")
print(f"\nSubstituting values from window [cycle {start_idx+1} to {start_idx+s}]:")
print(f"\n  x̄ = (1/{s}) × (x₀ + x₁ + ... + x{s-1})")
print(f"\n  x̄ = (1/{s}) × ({window[0]:.6f}")
for i in range(1, s-1):
    print(f"              + {window[i]:.6f}")
print(f"              + {window[s-1]:.6f})")

total = np.sum(window)
print(f"\n  x̄ = (1/{s}) × {total:.6f}")

x_bar = total / s
print(f"\n  x̄ = {total:.6f} / {s}")
print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  x̄ = {x_bar:.6f} Ah                      │")
print(f"  │  (mean capacity over cycles {start_idx+1}-{start_idx+s})     │")
print(f"  └─────────────────────────────────────────┘")
```

    Step 2: Compute the Mean
    ============================================================
    
    Formula: x̄ = (1/s) × Σxᵢ
    
    Substituting values from window [cycle 20 to 27]:
    
      x̄ = (1/8) × (x₀ + x₁ + ... + x7)
    
      x̄ = (1/8) × (1.339423
                  + 1.284916
                  + 1.281719
                  + 1.260005
                  + 1.266070
                  + 1.241359
                  + 1.229887
                  + 1.228089)
    
      x̄ = (1/8) × 10.131469
    
      x̄ = 10.131469 / 8
    
      ┌─────────────────────────────────────────┐
      │  x̄ = 1.266434 Ah                      │
      │  (mean capacity over cycles 20-27)     │
      └─────────────────────────────────────────┘


---

## Step 3: Compute Deviations from Mean

### Formula:
$$y_i = x_i - \bar{x}$$

### Solution:


```python
print("Step 3: Compute Deviations from Mean")
print("="*60)
print(f"\nFormula: yᵢ = xᵢ - x̄")
print(f"\nWhere x̄ = {x_bar:.6f} (from Step 2)")
print(f"\nComputing each deviation:")
print(f"\n  {'i':<4} {'Cycle':<6} {'xᵢ':<12} {'-':<3} {'x̄':<12} {'=':<3} {'yᵢ':<12}")
print(f"  {'-'*4} {'-'*6} {'-'*12} {'-'*3} {'-'*12} {'-'*3} {'-'*12}")

y = window - x_bar
for i in range(s):
    cycle = start_idx + i + 1
    print(f"  {i:<4} {cycle:<6} {window[i]:<12.6f} {'-':<3} {x_bar:<12.6f} {'=':<3} {y[i]:+12.6f}")

print(f"\nDeviation vector:")
print(f"\n  y = [{', '.join([f'{v:+.6f}' for v in y])}]")

print(f"\n✓ Verification: Σyᵢ = {np.sum(y):.10f} ≈ 0")
```

    Step 3: Compute Deviations from Mean
    ============================================================
    
    Formula: yᵢ = xᵢ - x̄
    
    Where x̄ = 1.266434 (from Step 2)
    
    Computing each deviation:
    
      i    Cycle  xᵢ           -   x̄           =   yᵢ          
      ---- ------ ------------ --- ------------ --- ------------
      0    20     1.339423     -   1.266434     =      +0.072990
      1    21     1.284916     -   1.266434     =      +0.018482
      2    22     1.281719     -   1.266434     =      +0.015286
      3    23     1.260005     -   1.266434     =      -0.006428
      4    24     1.266070     -   1.266434     =      -0.000363
      5    25     1.241359     -   1.266434     =      -0.025075
      6    26     1.229887     -   1.266434     =      -0.036546
      7    27     1.228089     -   1.266434     =      -0.038345
    
    Deviation vector:
    
      y = [+0.072990, +0.018482, +0.015286, -0.006428, -0.000363, -0.025075, -0.036546, -0.038345]
    
    ✓ Verification: Σyᵢ = -0.0000000000 ≈ 0


---

## Step 4: Compute Cumulative Deviation Series

### Formula:
$$Z_k = \sum_{i=0}^{k} y_i \quad \text{for } k = 0, 1, ..., s-1$$

### Solution:


```python
print("Step 4: Compute Cumulative Deviation Series")
print("="*60)
print(f"\nFormula: Zₖ = Σᵢ₌₀ᵏ yᵢ (running sum of deviations)")
print(f"\nComputing step by step:")

Z = np.cumsum(y)

print(f"\n  Z₀ = y₀")
print(f"     = {y[0]:+.6f}")

for k in range(1, s):
    print(f"\n  Z{k} = Z{k-1} + y{k}")
    print(f"     = {Z[k-1]:+.6f} + ({y[k]:+.6f})")
    print(f"     = {Z[k]:+.6f}")

print(f"\n" + "-"*60)
print(f"Cumulative deviation series:")
print(f"\n  Z = [{', '.join([f'{v:+.6f}' for v in Z])}]")
print(f"\n  {'k':<4} {'Zₖ':<14} {'Visual':<40}")
print(f"  {'-'*4} {'-'*14} {'-'*40}")
z_range = max(abs(Z.min()), abs(Z.max()))
for k, z in enumerate(Z):
    bar_len = int(20 * z / z_range) if z_range > 0 else 0
    if bar_len >= 0:
        bar = ' ' * 20 + '|' + '█' * bar_len
    else:
        bar = ' ' * (20 + bar_len) + '█' * (-bar_len) + '|'
    print(f"  {k:<4} {z:+.6f}      {bar}")
```

    Step 4: Compute Cumulative Deviation Series
    ============================================================
    
    Formula: Zₖ = Σᵢ₌₀ᵏ yᵢ (running sum of deviations)
    
    Computing step by step:
    
      Z₀ = y₀
         = +0.072990
    
      Z1 = Z0 + y1
         = +0.072990 + (+0.018482)
         = +0.091472
    
      Z2 = Z1 + y2
         = +0.091472 + (+0.015286)
         = +0.106757
    
      Z3 = Z2 + y3
         = +0.106757 + (-0.006428)
         = +0.100329
    
      Z4 = Z3 + y4
         = +0.100329 + (-0.000363)
         = +0.099966
    
      Z5 = Z4 + y5
         = +0.099966 + (-0.025075)
         = +0.074891
    
      Z6 = Z5 + y6
         = +0.074891 + (-0.036546)
         = +0.038345
    
      Z7 = Z6 + y7
         = +0.038345 + (-0.038345)
         = -0.000000
    
    ------------------------------------------------------------
    Cumulative deviation series:
    
      Z = [+0.072990, +0.091472, +0.106757, +0.100329, +0.099966, +0.074891, +0.038345, -0.000000]
    
      k    Zₖ             Visual                                  
      ---- -------------- ----------------------------------------
      0    +0.072990                          |█████████████
      1    +0.091472                          |█████████████████
      2    +0.106757                          |████████████████████
      3    +0.100329                          |██████████████████
      4    +0.099966                          |██████████████████
      5    +0.074891                          |██████████████
      6    +0.038345                          |███████
      7    -0.000000                          |


---

## Step 5: Compute the Range R

### Formula:
$$R = \max(Z) - \min(Z)$$

### Solution:


```python
print("Step 5: Compute the Range R")
print("="*60)
print(f"\nFormula: R = max(Z) - min(Z)")
print(f"\nFrom the cumulative series Z:")
print(f"\n  Z = [{', '.join([f'{v:+.6f}' for v in Z])}]")

Z_max = np.max(Z)
Z_min = np.min(Z)
Z_max_idx = np.argmax(Z)
Z_min_idx = np.argmin(Z)

print(f"\nFinding extrema:")
print(f"\n  max(Z) = Z{Z_max_idx} = {Z_max:+.6f}  (at cycle {start_idx + Z_max_idx + 1})")
print(f"  min(Z) = Z{Z_min_idx} = {Z_min:+.6f}  (at cycle {start_idx + Z_min_idx + 1})")

R = Z_max - Z_min
print(f"\nComputing R:")
print(f"\n  R = max(Z) - min(Z)")
print(f"    = {Z_max:+.6f} - ({Z_min:+.6f})")
print(f"    = {Z_max:+.6f} + {-Z_min:+.6f}")
print(f"\n  ┌─────────────────────────┐")
print(f"  │  R = {R:.6f}           │")
print(f"  └─────────────────────────┘")
```

    Step 5: Compute the Range R
    ============================================================
    
    Formula: R = max(Z) - min(Z)
    
    From the cumulative series Z:
    
      Z = [+0.072990, +0.091472, +0.106757, +0.100329, +0.099966, +0.074891, +0.038345, -0.000000]
    
    Finding extrema:
    
      max(Z) = Z2 = +0.106757  (at cycle 22)
      min(Z) = Z7 = -0.000000  (at cycle 27)
    
    Computing R:
    
      R = max(Z) - min(Z)
        = +0.106757 - (-0.000000)
        = +0.106757 + +0.000000
    
      ┌─────────────────────────┐
      │  R = 0.106757           │
      └─────────────────────────┘


---

## Step 6: Compute the Standard Deviation S

### Formula:
$$S = \sqrt{\frac{1}{s-1} \sum_{i=0}^{s-1} (x_i - \bar{x})^2} = \sqrt{\frac{1}{s-1} \sum_{i=0}^{s-1} y_i^2}$$

### Solution:


```python
print("Step 6: Compute the Standard Deviation S")
print("="*60)
print(f"\nFormula: S = √[ (1/(s-1)) × Σyᵢ² ]")
print(f"\nWhere s = {s}, so s-1 = {s-1}")

y_squared = y ** 2
print(f"\nFirst, compute yᵢ²:")
print(f"\n  {'i':<4} {'yᵢ':<14} {'yᵢ²':<14}")
print(f"  {'-'*4} {'-'*14} {'-'*14}")
for i in range(s):
    print(f"  {i:<4} {y[i]:+.6f}      {y_squared[i]:.8f}")

sum_y_sq = np.sum(y_squared)
print(f"\n  Σyᵢ² = {sum_y_sq:.8f}")

variance = sum_y_sq / (s - 1)
print(f"\nCompute variance:")
print(f"\n  σ² = Σyᵢ² / (s-1)")
print(f"     = {sum_y_sq:.8f} / {s-1}")
print(f"     = {variance:.8f}")

S = np.sqrt(variance)
print(f"\nCompute standard deviation:")
print(f"\n  S = √σ²")
print(f"    = √{variance:.8f}")
print(f"\n  ┌─────────────────────────┐")
print(f"  │  S = {S:.6f}           │")
print(f"  └─────────────────────────┘")
```

    Step 6: Compute the Standard Deviation S
    ============================================================
    
    Formula: S = √[ (1/(s-1)) × Σyᵢ² ]
    
    Where s = 8, so s-1 = 7
    
    First, compute yᵢ²:
    
      i    yᵢ             yᵢ²           
      ---- -------------- --------------
      0    +0.072990      0.00532752
      1    +0.018482      0.00034159
      2    +0.015286      0.00023365
      3    -0.006428      0.00004132
      4    -0.000363      0.00000013
      5    -0.025075      0.00062873
      6    -0.036546      0.00133563
      7    -0.038345      0.00147034
    
      Σyᵢ² = 0.00937891
    
    Compute variance:
    
      σ² = Σyᵢ² / (s-1)
         = 0.00937891 / 7
         = 0.00133984
    
    Compute standard deviation:
    
      S = √σ²
        = √0.00133984
    
      ┌─────────────────────────┐
      │  S = 0.036604           │
      └─────────────────────────┘


---

## Step 7: Compute the Rescaled Range (R/S)

### Formula:
$$\frac{R}{S} = \frac{\max(Z) - \min(Z)}{S}$$

### Solution:


```python
print("Step 7: Compute the Rescaled Range (R/S)")
print("="*60)
print(f"\nFormula: (R/S) = R / S")
print(f"\nSubstituting computed values:")
print(f"\n  R = {R:.6f}  (from Step 5)")
print(f"  S = {S:.6f}  (from Step 6)")

RS = R / S
print(f"\n  (R/S) = {R:.6f} / {S:.6f}")
print(f"\n  ╔═══════════════════════════════════════════════════════╗")
print(f"  ║  (R/S) for window size {s} = {RS:.6f}                 ║")
print(f"  ║  Window: cycles {start_idx+1} to {start_idx+s} (Battery {ENTITY_ID})        ║")
print(f"  ╚═══════════════════════════════════════════════════════╝")
```

    Step 7: Compute the Rescaled Range (R/S)
    ============================================================
    
    Formula: (R/S) = R / S
    
    Substituting computed values:
    
      R = 0.106757  (from Step 5)
      S = 0.036604  (from Step 6)
    
      (R/S) = 0.106757 / 0.036604
    
      ╔═══════════════════════════════════════════════════════╗
      ║  (R/S) for window size 8 = 2.916562                 ║
      ║  Window: cycles 20 to 27 (Battery B0047)        ║
      ╚═══════════════════════════════════════════════════════╝


---

## Step 8: Repeat for Multiple Window Sizes

To estimate H, we need R/S values for multiple window sizes and fit a line in log-log space.


```python
print("Step 8: Compute R/S for Multiple Window Sizes")
print("="*60)

def compute_rs_for_window(series, window_size):
    """Compute mean R/S for a given window size."""
    n = len(series)
    n_windows = n // window_size
    rs_values = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        w = series[start:end]
        
        # Mean, deviations, cumulative, range, std
        w_mean = np.mean(w)
        w_dev = w - w_mean
        w_cumsum = np.cumsum(w_dev)
        w_R = np.max(w_cumsum) - np.min(w_cumsum)
        w_S = np.std(w, ddof=1)
        
        if w_S > 0:
            rs_values.append(w_R / w_S)
    
    return np.mean(rs_values) if rs_values else 0, len(rs_values)

# Window sizes (logarithmically spaced)
window_sizes = [8, 11, 14, 17, 22]

print(f"\nUsing full series: n = {n} cycles")
print(f"Window sizes: {window_sizes}")
print(f"\nComputing mean R/S for each window size:")
print(f"\n  {'s':<6} {'n_windows':<12} {'mean(R/S)':<14} {'log(s)':<12} {'log(R/S)':<12}")
print(f"  {'-'*6} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")

results = []
for ws in window_sizes:
    rs_mean, n_win = compute_rs_for_window(X, ws)
    log_s = np.log(ws)
    log_rs = np.log(rs_mean)
    results.append((ws, n_win, rs_mean, log_s, log_rs))
    print(f"  {ws:<6} {n_win:<12} {rs_mean:<14.6f} {log_s:<12.6f} {log_rs:<12.6f}")

print(f"\nData points for linear regression:")
print(f"\n  x = log(s)   = [{', '.join([f'{r[3]:.4f}' for r in results])}]")
print(f"  y = log(R/S) = [{', '.join([f'{r[4]:.4f}' for r in results])}]")
```

    Step 8: Compute R/S for Multiple Window Sizes
    ============================================================
    
    Using full series: n = 69 cycles
    Window sizes: [8, 11, 14, 17, 22]
    
    Computing mean R/S for each window size:
    
      s      n_windows    mean(R/S)      log(s)       log(R/S)    
      ------ ------------ -------------- ------------ ------------
      8      8            2.941717       2.079442     1.078993    
      11     6            4.300713       2.397895     1.458781    
      14     4            5.453725       2.639057     1.696299    
      17     4            5.971595       2.833213     1.787014    
      22     3            7.761640       3.091042     2.049194    
    
    Data points for linear regression:
    
      x = log(s)   = [2.0794, 2.3979, 2.6391, 2.8332, 3.0910]
      y = log(R/S) = [1.0790, 1.4588, 1.6963, 1.7870, 2.0492]


---

## Step 9: Linear Regression in Log-Log Space

### Model:
$$\log(R/S) = H \cdot \log(s) + c$$

The slope H is the Hurst exponent.

### Using Ordinary Least Squares:
$$H = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$


```python
print("Step 9: Linear Regression for Hurst Exponent")
print("="*60)

# Extract log values
log_s = np.array([r[3] for r in results])
log_rs = np.array([r[4] for r in results])
m = len(log_s)

print(f"\nGiven {m} data points:")
print(f"  x = log(s)   = [{', '.join([f'{v:.6f}' for v in log_s])}]")
print(f"  y = log(R/S) = [{', '.join([f'{v:.6f}' for v in log_rs])}]")

# Compute means
x_mean = np.mean(log_s)
y_mean = np.mean(log_rs)

print(f"\nStep 9a: Compute means")
print(f"  x̄ = {x_mean:.6f}")
print(f"  ȳ = {y_mean:.6f}")

# Compute deviations
x_dev = log_s - x_mean
y_dev = log_rs - y_mean
xy_products = x_dev * y_dev
x_sq = x_dev ** 2

print(f"\nStep 9b: Compute deviations and products")
print(f"\n  {'i':<4} {'xᵢ-x̄':<14} {'yᵢ-ȳ':<14} {'(xᵢ-x̄)(yᵢ-ȳ)':<14} {'(xᵢ-x̄)²':<14}")
print(f"  {'-'*4} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
for i in range(m):
    print(f"  {i:<4} {x_dev[i]:+.6f}      {y_dev[i]:+.6f}      {xy_products[i]:+.6f}      {x_sq[i]:.6f}")
print(f"  {'-'*4} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
print(f"  Σ                             {np.sum(xy_products):+.6f}      {np.sum(x_sq):.6f}")

# Compute slope (Hurst exponent)
numerator = np.sum(xy_products)
denominator = np.sum(x_sq)

print(f"\nStep 9c: Compute slope (Hurst exponent)")
print(f"\n  H = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²")
print(f"    = {numerator:.6f} / {denominator:.6f}")

H = numerator / denominator
c = y_mean - H * x_mean

print(f"\n  ╔════════════════════════════════════════════════════════╗")
print(f"  ║                                                        ║")
print(f"  ║   HURST EXPONENT:  H = {H:.6f}                        ║")
print(f"  ║   Battery: {ENTITY_ID}                                       ║")
print(f"  ║                                                        ║")
print(f"  ╚════════════════════════════════════════════════════════╝")

print(f"\n  Regression line: log(R/S) = {H:.6f}·log(s) + ({c:.6f})")
```

    Step 9: Linear Regression for Hurst Exponent
    ============================================================
    
    Given 5 data points:
      x = log(s)   = [2.079442, 2.397895, 2.639057, 2.833213, 3.091042]
      y = log(R/S) = [1.078993, 1.458781, 1.696299, 1.787014, 2.049194]
    
    Step 9a: Compute means
      x̄ = 2.608130
      ȳ = 1.614056
    
    Step 9b: Compute deviations and products
    
      i    xᵢ-x̄          yᵢ-ȳ           (xᵢ-x̄)(yᵢ-ȳ)  (xᵢ-x̄)²      
      ---- -------------- -------------- -------------- --------------
      0    -0.528688      -0.535063      +0.282882      0.279511
      1    -0.210235      -0.155275      +0.032644      0.044199
      2    +0.030927      +0.082243      +0.002544      0.000957
      3    +0.225083      +0.172958      +0.038930      0.050663
      4    +0.482912      +0.435137      +0.210133      0.233204
      ---- -------------- -------------- -------------- --------------
      Σ                             +0.567133      0.608534
    
    Step 9c: Compute slope (Hurst exponent)
    
      H = Σ(xᵢ-x̄)(yᵢ-ȳ) / Σ(xᵢ-x̄)²
        = 0.567133 / 0.608534
    
      ╔════════════════════════════════════════════════════════╗
      ║                                                        ║
      ║   HURST EXPONENT:  H = 0.931966                        ║
      ║   Battery: B0047                                       ║
      ║                                                        ║
      ╚════════════════════════════════════════════════════════╝
    
      Regression line: log(R/S) = 0.931966·log(s) + (-0.816632)


---

## Step 10: Interpretation


```python
print("Step 10: Physical Interpretation")
print("="*60)
print(f"\nComputed Hurst exponent: H = {H:.6f}")
print(f"\nHurst Exponent Scale:")
print(f"\n  | H Value | Behavior |")
print(f"  |---------|----------|")
print(f"  | H < 0.5 | Anti-persistent (mean-reverting) |")
print(f"  | H = 0.5 | Random walk (no memory) |")
print(f"  | H > 0.5 | Persistent (trending) |")
print(f"  | H ≈ 1.0 | Strong persistence |")

if H > 0.9:
    behavior = "STRONGLY PERSISTENT"
    meaning = "Near-deterministic trend"
elif H > 0.5:
    behavior = "PERSISTENT"
    meaning = "Trending - past predicts future direction"
elif H < 0.5:
    behavior = "ANTI-PERSISTENT"
    meaning = "Mean-reverting"
else:
    behavior = "RANDOM WALK"
    meaning = "No long-range memory"

print(f"\n  ┌────────────────────────────────────────────────────────┐")
print(f"  │  Behavior: {behavior:<45} │")
print(f"  │  Meaning:  {meaning:<45} │")
print(f"  └────────────────────────────────────────────────────────┘")

print(f"\n  For Battery {ENTITY_ID} capacity degradation:")
print(f"  • The capacity loss follows a predictable downward trend")
print(f"  • Past degradation behavior predicts future degradation")
print(f"  • The signal is amenable to prognostic modeling")
```

    Step 10: Physical Interpretation
    ============================================================
    
    Computed Hurst exponent: H = 0.931966
    
    Hurst Exponent Scale:
    
      | H Value | Behavior |
      |---------|----------|
      | H < 0.5 | Anti-persistent (mean-reverting) |
      | H = 0.5 | Random walk (no memory) |
      | H > 0.5 | Persistent (trending) |
      | H ≈ 1.0 | Strong persistence |
    
      ┌────────────────────────────────────────────────────────┐
      │  Behavior: STRONGLY PERSISTENT                           │
      │  Meaning:  Near-deterministic trend                      │
      └────────────────────────────────────────────────────────┘
    
      For Battery B0047 capacity degradation:
      • The capacity loss follows a predictable downward trend
      • Past degradation behavior predicts future degradation
      • The signal is amenable to prognostic modeling


---

## Summary


```python
print("="*70)
print("COMPLETE SOLUTION SUMMARY")
print("="*70)
print(f"""
GIVEN:
  • Battery {ENTITY_ID} capacity time series
  • n = {n} observations
  • Range: [{X.min():.4f}, {X.max():.4f}] Ah

DETAILED WINDOW:
  • Cycles {start_idx+1} to {start_idx+s} (indices {start_idx} to {end_idx-1})
  • Window size: s = {s}

METHOD: Rescaled Range (R/S) Analysis

DETAILED CALCULATION (cycles {start_idx+1}-{start_idx+s}):
  • Data: [{', '.join([f'{v:.4f}' for v in window])}]
  • Mean: x̄ = {x_bar:.6f} Ah
  • Range of cumulative deviations: R = {R:.6f}
  • Standard deviation: S = {S:.6f}
  • Rescaled range: (R/S) = {RS:.6f}

MULTI-SCALE ANALYSIS:
  • Window sizes: {window_sizes}
  • Linear regression: log(R/S) = H·log(s) + c
  • Slope = H = {numerator:.6f} / {denominator:.6f}

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   FINAL ANSWER:  H = {H:.6f}                                  ║
║                                                                ║
║   INTERPRETATION: {behavior:<41} ║
║   Battery capacity degradation follows a predictable trend    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

To explore different windows, change WINDOW_START and WINDOW_SIZE 
in Step 0 and re-run the notebook.
""")
```

    ======================================================================
    COMPLETE SOLUTION SUMMARY
    ======================================================================
    
    GIVEN:
      • Battery B0047 capacity time series
      • n = 69 observations
      • Range: [1.1060, 1.6743] Ah
    
    DETAILED WINDOW:
      • Cycles 20 to 27 (indices 19 to 26)
      • Window size: s = 8
    
    METHOD: Rescaled Range (R/S) Analysis
    
    DETAILED CALCULATION (cycles 20-27):
      • Data: [1.3394, 1.2849, 1.2817, 1.2600, 1.2661, 1.2414, 1.2299, 1.2281]
      • Mean: x̄ = 1.266434 Ah
      • Range of cumulative deviations: R = 0.106757
      • Standard deviation: S = 0.036604
      • Rescaled range: (R/S) = 2.916562
    
    MULTI-SCALE ANALYSIS:
      • Window sizes: [8, 11, 14, 17, 22]
      • Linear regression: log(R/S) = H·log(s) + c
      • Slope = H = 0.567133 / 0.608534
    
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   FINAL ANSWER:  H = 0.931966                                  ║
    ║                                                                ║
    ║   INTERPRETATION: STRONGLY PERSISTENT                       ║
    ║   Battery capacity degradation follows a predictable trend    ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    
    To explore different windows, change WINDOW_START and WINDOW_SIZE 
    in Step 0 and re-run the notebook.
    


---

## 🔬 Try Different Windows!

Change the parameters in **Step 0** to see how the calculations look at different points in the battery's life:

| Phase | Suggested WINDOW_START | Description |
|-------|------------------------|-------------|
| Early life | 0 | Fresh battery, minimal degradation |
| Mid life | 30 | Active degradation period |
| Late life | 55 | Approaching end-of-life |

You can also try different batteries: `B0045`, `B0046`, `B0047`, `B0048`

---

*PRISM Behavioral Geometry Engine - Mathematical Derivation Proof*

*Battery: NASA B0047 | Signal: Capacity (Ah) | Method: Hurst R/S Analysis*
