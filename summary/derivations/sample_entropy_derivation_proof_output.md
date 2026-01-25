# Sample Entropy Derivation: Complete Mathematical Proof

## Problem Statement

**Given:** A time series of battery capacity measurements from NASA Li-ion Battery B0047

**Find:** The Sample Entropy (SampEn) which measures signal complexity/regularity

**Method:** We will compute SampEn by:
1. Creating template vectors of length m (embedding dimension)
2. Counting similar patterns using tolerance r
3. Computing the probability ratio: $SampEn = -\ln(A/B)$

---

## Step 0: Load the Data and Set Parameters


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

X_full = b47['value'].to_numpy()

# Use a subset for detailed demonstration (first 20 points)
X = X_full[:20]
N = len(X)

# Standard parameters
m = 2  # embedding dimension
r = 0.2 * np.std(X)  # tolerance = 0.2 × standard deviation

print(f"Sample Entropy Parameters")
print(f"="*60)
print(f"\nEmbedding dimension: m = {m}")
print(f"Tolerance: r = 0.2 × σ")
print(f"         = 0.2 × {np.std(X):.6f}")
print(f"         = {r:.6f}")
print(f"\nSeries length: N = {N}")
print(f"\nData X (first 20 cycles, capacity in Ah):")
print(f"\n  {'i':<4} {'xᵢ (Ah)':<12}")
print(f"  {'-'*4} {'-'*12}")
for i, v in enumerate(X):
    print(f"  {i:<4} {v:.6f}")
```

    Sample Entropy Parameters
    ============================================================
    
    Embedding dimension: m = 2
    Tolerance: r = 0.2 × σ
             = 0.2 × 0.081143
             = 0.016229
    
    Series length: N = 20
    
    Data X (first 20 cycles, capacity in Ah):
    
      i    xᵢ (Ah)     
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
      15   1.370505
      16   1.349743
      17   1.325153
      18   1.311194
      19   1.339423


---

## Step 1: Create Template Vectors of Length m

### Definition:
A template vector $\mathbf{u}_i^{(m)}$ of length m starting at index i is:
$$\mathbf{u}_i^{(m)} = [x_i, x_{i+1}, ..., x_{i+m-1}]$$

For m = 2, we have N - m = N - 2 template vectors.

### Solution:


```python
print("Step 1: Create Template Vectors of Length m = 2")
print("="*60)
print(f"\nDefinition: uᵢ^(m) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ₋₁]")
print(f"\nFor m = {m}, each template has {m} elements.")
print(f"Number of templates: N - m = {N} - {m} = {N - m}")

# Create template vectors of length m
templates_m = []
for i in range(N - m):
    template = X[i:i+m]
    templates_m.append(template)

print(f"\nAll {N - m} template vectors:")
print(f"\n  {'i':<4} {'uᵢ^(2)':<35} {'Components':<25}")
print(f"  {'-'*4} {'-'*35} {'-'*25}")
for i, t in enumerate(templates_m):
    t_str = f"[{t[0]:.6f}, {t[1]:.6f}]"
    comp = f"[x{i}, x{i+1}]"
    print(f"  {i:<4} {t_str:<35} {comp:<25}")
```

    Step 1: Create Template Vectors of Length m = 2
    ============================================================
    
    Definition: uᵢ^(m) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ₋₁]
    
    For m = 2, each template has 2 elements.
    Number of templates: N - m = 20 - 2 = 18
    
    All 18 template vectors:
    
      i    uᵢ^(2)                              Components               
      ---- ----------------------------------- -------------------------
      0    [1.674305, 1.524366]                [x0, x1]                 
      1    [1.524366, 1.508076]                [x1, x2]                 
      2    [1.508076, 1.483558]                [x2, x3]                 
      3    [1.483558, 1.467139]                [x3, x4]                 
      4    [1.467139, 1.448858]                [x4, x5]                 
      5    [1.448858, 1.445853]                [x5, x6]                 
      6    [1.445853, 1.431118]                [x6, x7]                 
      7    [1.431118, 1.419275]                [x7, x8]                 
      8    [1.419275, 1.399997]                [x8, x9]                 
      9    [1.399997, 1.388516]                [x9, x10]                
      10   [1.388516, 1.365223]                [x10, x11]               
      11   [1.365223, 1.406044]                [x11, x12]               
      12   [1.406044, 1.405754]                [x12, x13]               
      13   [1.405754, 1.386766]                [x13, x14]               
      14   [1.386766, 1.370505]                [x14, x15]               
      15   [1.370505, 1.349743]                [x15, x16]               
      16   [1.349743, 1.325153]                [x16, x17]               
      17   [1.325153, 1.311194]                [x17, x18]               


---

## Step 2: Define the Distance Function

### Chebyshev Distance (Maximum Norm):
$$d(\mathbf{u}_i, \mathbf{u}_j) = \max_k |u_i[k] - u_j[k]|$$

Two templates are "similar" if $d(\mathbf{u}_i, \mathbf{u}_j) < r$

### Solution:


```python
print("Step 2: Distance Function (Chebyshev / Max Norm)")
print("="*60)
print(f"\nDefinition: d(uᵢ, uⱼ) = max|uᵢ[k] - uⱼ[k]| over all k")
print(f"\nTwo templates match if d(uᵢ, uⱼ) < r = {r:.6f}")

def chebyshev_distance(u1, u2):
    """Compute Chebyshev (max) distance between two vectors."""
    return np.max(np.abs(u1 - u2))

# Example calculation
print(f"\n" + "-"*60)
print(f"Example: Distance between u₀ and u₁")
print(f"\n  u₀ = [{templates_m[0][0]:.6f}, {templates_m[0][1]:.6f}]")
print(f"  u₁ = [{templates_m[1][0]:.6f}, {templates_m[1][1]:.6f}]")

diff0 = abs(templates_m[0][0] - templates_m[1][0])
diff1 = abs(templates_m[0][1] - templates_m[1][1])

print(f"\n  |u₀[0] - u₁[0]| = |{templates_m[0][0]:.6f} - {templates_m[1][0]:.6f}|")
print(f"                  = {diff0:.6f}")
print(f"\n  |u₀[1] - u₁[1]| = |{templates_m[0][1]:.6f} - {templates_m[1][1]:.6f}|")
print(f"                  = {diff1:.6f}")

d_01 = chebyshev_distance(templates_m[0], templates_m[1])
print(f"\n  d(u₀, u₁) = max({diff0:.6f}, {diff1:.6f})")
print(f"            = {d_01:.6f}")

match_01 = d_01 < r
print(f"\n  Is d(u₀, u₁) < r?")
print(f"  {d_01:.6f} < {r:.6f}?")
print(f"  Answer: {'YES - Match!' if match_01 else 'NO - Not a match'}")
```

    Step 2: Distance Function (Chebyshev / Max Norm)
    ============================================================
    
    Definition: d(uᵢ, uⱼ) = max|uᵢ[k] - uⱼ[k]| over all k
    
    Two templates match if d(uᵢ, uⱼ) < r = 0.016229
    
    ------------------------------------------------------------
    Example: Distance between u₀ and u₁
    
      u₀ = [1.674305, 1.524366]
      u₁ = [1.524366, 1.508076]
    
      |u₀[0] - u₁[0]| = |1.674305 - 1.524366|
                      = 0.149939
    
      |u₀[1] - u₁[1]| = |1.524366 - 1.508076|
                      = 0.016290
    
      d(u₀, u₁) = max(0.149939, 0.016290)
                = 0.149939
    
      Is d(u₀, u₁) < r?
      0.149939 < 0.016229?
      Answer: NO - Not a match


---

## Step 3: Count Matching Pairs B^m (Length m Templates)

### Definition:
$B^m$ = Number of template pairs $(i, j)$ where $i \neq j$ and $d(\mathbf{u}_i^{(m)}, \mathbf{u}_j^{(m)}) < r$

We count all pairs where $i < j$ to avoid double-counting.

### Solution:


```python
print("Step 3: Count Matching Pairs B^m (Length m = 2)")
print("="*60)
print(f"\nCompute distance for all pairs (i, j) where i < j")
print(f"A pair matches if d(uᵢ, uⱼ) < r = {r:.6f}")
print(f"\nTotal possible pairs: C({N-m}, 2) = {(N-m)*(N-m-1)//2}")

# Compute all pairwise distances and find matches
B_m_matches = []  # List of matching pairs
distance_matrix = np.zeros((N-m, N-m))

print(f"\nDistance matrix (showing d(uᵢ, uⱼ) for i < j):")
print(f"\n  Pairs checked: (showing first 10 matches and non-matches)")
print(f"\n  {'(i,j)':<8} {'d(uᵢ,uⱼ)':<12} {'< r?':<8} {'Match?':<8}")
print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8}")

shown = 0
for i in range(N - m):
    for j in range(i + 1, N - m):
        d = chebyshev_distance(templates_m[i], templates_m[j])
        distance_matrix[i, j] = d
        is_match = d < r
        if is_match:
            B_m_matches.append((i, j, d))
        if shown < 15:
            match_str = "YES ✓" if is_match else "no"
            print(f"  ({i},{j}){' '*(5-len(str(i))-len(str(j)))} {d:<12.6f} {r:.4f}   {match_str}")
            shown += 1

if shown < (N-m)*(N-m-1)//2:
    print(f"  ... ({(N-m)*(N-m-1)//2 - shown} more pairs checked)")

B_m = len(B_m_matches)
print(f"\n" + "-"*60)
print(f"Matching pairs found (d < {r:.6f}):")
print(f"\n  {'(i,j)':<8} {'d(uᵢ,uⱼ)':<12}")
print(f"  {'-'*8} {'-'*12}")
for i, j, d in B_m_matches[:10]:
    print(f"  ({i},{j}){' '*(5-len(str(i))-len(str(j)))} {d:.6f}")
if len(B_m_matches) > 10:
    print(f"  ... ({len(B_m_matches) - 10} more matches)")

print(f"\n  ┌─────────────────────────────────────┐")
print(f"  │  B^m = B^{m} = {B_m:<4} matching pairs   │")
print(f"  └─────────────────────────────────────┘")
```

    Step 3: Count Matching Pairs B^m (Length m = 2)
    ============================================================
    
    Compute distance for all pairs (i, j) where i < j
    A pair matches if d(uᵢ, uⱼ) < r = 0.016229
    
    Total possible pairs: C(18, 2) = 153
    
    Distance matrix (showing d(uᵢ, uⱼ) for i < j):
    
      Pairs checked: (showing first 10 matches and non-matches)
    
      (i,j)    d(uᵢ,uⱼ)     < r?     Match?  
      -------- ------------ -------- --------
      (0,1)    0.149939     0.0162   no
      (0,2)    0.166228     0.0162   no
      (0,3)    0.190747     0.0162   no
      (0,4)    0.207166     0.0162   no
      (0,5)    0.225447     0.0162   no
      (0,6)    0.228451     0.0162   no
      (0,7)    0.243186     0.0162   no
      (0,8)    0.255030     0.0162   no
      (0,9)    0.274307     0.0162   no
      (0,10)   0.285789     0.0162   no
      (0,11)   0.309081     0.0162   no
      (0,12)   0.268261     0.0162   no
      (0,13)   0.268551     0.0162   no
      (0,14)   0.287538     0.0162   no
      (0,15)   0.303799     0.0162   no
      ... (138 more pairs checked)
    
    ------------------------------------------------------------
    Matching pairs found (d < 0.016229):
    
      (i,j)    d(uᵢ,uⱼ)    
      -------- ------------
      (5,6)    0.014735
      (6,7)    0.014735
      (8,12)   0.013230
      (8,13)   0.013520
      (9,13)   0.005757
      (10,14)  0.005282
    
      ┌─────────────────────────────────────┐
      │  B^m = B^2 = 6    matching pairs   │
      └─────────────────────────────────────┘


---

## Step 4: Create Template Vectors of Length m+1

### Definition:
$$\mathbf{u}_i^{(m+1)} = [x_i, x_{i+1}, ..., x_{i+m}]$$

For m+1 = 3, we have N - (m+1) = N - 3 template vectors.

### Solution:


```python
print(f"Step 4: Create Template Vectors of Length m+1 = {m+1}")
print("="*60)
print(f"\nDefinition: uᵢ^(m+1) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ]")
print(f"\nFor m+1 = {m+1}, each template has {m+1} elements.")
print(f"Number of templates: N - (m+1) = {N} - {m+1} = {N - m - 1}")

# Create template vectors of length m+1
templates_m1 = []
for i in range(N - m - 1):
    template = X[i:i+m+1]
    templates_m1.append(template)

print(f"\nAll {N - m - 1} template vectors:")
print(f"\n  {'i':<4} {'uᵢ^(3)':<50}")
print(f"  {'-'*4} {'-'*50}")
for i, t in enumerate(templates_m1):
    t_str = f"[{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]"
    print(f"  {i:<4} {t_str}")
```

    Step 4: Create Template Vectors of Length m+1 = 3
    ============================================================
    
    Definition: uᵢ^(m+1) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ]
    
    For m+1 = 3, each template has 3 elements.
    Number of templates: N - (m+1) = 20 - 3 = 17
    
    All 17 template vectors:
    
      i    uᵢ^(3)                                            
      ---- --------------------------------------------------
      0    [1.674305, 1.524366, 1.508076]
      1    [1.524366, 1.508076, 1.483558]
      2    [1.508076, 1.483558, 1.467139]
      3    [1.483558, 1.467139, 1.448858]
      4    [1.467139, 1.448858, 1.445853]
      5    [1.448858, 1.445853, 1.431118]
      6    [1.445853, 1.431118, 1.419275]
      7    [1.431118, 1.419275, 1.399997]
      8    [1.419275, 1.399997, 1.388516]
      9    [1.399997, 1.388516, 1.365223]
      10   [1.388516, 1.365223, 1.406044]
      11   [1.365223, 1.406044, 1.405754]
      12   [1.406044, 1.405754, 1.386766]
      13   [1.405754, 1.386766, 1.370505]
      14   [1.386766, 1.370505, 1.349743]
      15   [1.370505, 1.349743, 1.325153]
      16   [1.349743, 1.325153, 1.311194]


---

## Step 5: Count Matching Pairs A^(m+1) (Length m+1 Templates)

### Definition:
$A^{m+1}$ = Number of template pairs $(i, j)$ where $i \neq j$ and $d(\mathbf{u}_i^{(m+1)}, \mathbf{u}_j^{(m+1)}) < r$

### Key Insight:
If two length-3 templates match, their length-2 prefixes also match. So $A \leq B$.

### Solution:


```python
print(f"Step 5: Count Matching Pairs A^(m+1) = A^{m+1} (Length {m+1})")
print("="*60)
print(f"\nCompute distance for all pairs (i, j) where i < j")
print(f"A pair matches if d(uᵢ, uⱼ) < r = {r:.6f}")
print(f"\nTotal possible pairs: C({N-m-1}, 2) = {(N-m-1)*(N-m-2)//2}")

# Compute all pairwise distances and find matches
A_m1_matches = []  # List of matching pairs

print(f"\n  {'(i,j)':<8} {'d(uᵢ,uⱼ)':<12} {'< r?':<8} {'Match?':<8}")
print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8}")

shown = 0
for i in range(N - m - 1):
    for j in range(i + 1, N - m - 1):
        d = chebyshev_distance(templates_m1[i], templates_m1[j])
        is_match = d < r
        if is_match:
            A_m1_matches.append((i, j, d))
        if shown < 15:
            match_str = "YES ✓" if is_match else "no"
            print(f"  ({i},{j}){' '*(5-len(str(i))-len(str(j)))} {d:<12.6f} {r:.4f}   {match_str}")
            shown += 1

if shown < (N-m-1)*(N-m-2)//2:
    print(f"  ... ({(N-m-1)*(N-m-2)//2 - shown} more pairs checked)")

A_m1 = len(A_m1_matches)
print(f"\n" + "-"*60)
print(f"Matching pairs found (d < {r:.6f}):")
if A_m1 > 0:
    print(f"\n  {'(i,j)':<8} {'d(uᵢ,uⱼ)':<12}")
    print(f"  {'-'*8} {'-'*12}")
    for i, j, d in A_m1_matches[:10]:
        print(f"  ({i},{j}){' '*(5-len(str(i))-len(str(j)))} {d:.6f}")
    if len(A_m1_matches) > 10:
        print(f"  ... ({len(A_m1_matches) - 10} more matches)")
else:
    print(f"\n  (No matches found)")

print(f"\n  ┌─────────────────────────────────────┐")
print(f"  │  A^(m+1) = A^{m+1} = {A_m1:<4} matching pairs │")
print(f"  └─────────────────────────────────────┘")
```

    Step 5: Count Matching Pairs A^(m+1) = A^3 (Length 3)
    ============================================================
    
    Compute distance for all pairs (i, j) where i < j
    A pair matches if d(uᵢ, uⱼ) < r = 0.016229
    
    Total possible pairs: C(17, 2) = 136
    
      (i,j)    d(uᵢ,uⱼ)     < r?     Match?  
      -------- ------------ -------- --------
      (0,1)    0.149939     0.0162   no
      (0,2)    0.166228     0.0162   no
      (0,3)    0.190747     0.0162   no
      (0,4)    0.207166     0.0162   no
      (0,5)    0.225447     0.0162   no
      (0,6)    0.228451     0.0162   no
      (0,7)    0.243186     0.0162   no
      (0,8)    0.255030     0.0162   no
      (0,9)    0.274307     0.0162   no
      (0,10)   0.285789     0.0162   no
      (0,11)   0.309081     0.0162   no
      (0,12)   0.268261     0.0162   no
      (0,13)   0.268551     0.0162   no
      (0,14)   0.287538     0.0162   no
      (0,15)   0.303799     0.0162   no
      ... (121 more pairs checked)
    
    ------------------------------------------------------------
    Matching pairs found (d < 0.016229):
    
      (i,j)    d(uᵢ,uⱼ)    
      -------- ------------
      (5,6)    0.014735
      (8,12)   0.013230
      (9,13)   0.005757
    
      ┌─────────────────────────────────────┐
      │  A^(m+1) = A^3 = 3    matching pairs │
      └─────────────────────────────────────┘


---

## Step 6: Compute Conditional Probabilities

### Definitions:
$$B = \frac{B^m}{\binom{N-m}{2}} = \frac{B^m}{(N-m)(N-m-1)/2}$$

$$A = \frac{A^{m+1}}{\binom{N-m-1}{2}} = \frac{A^{m+1}}{(N-m-1)(N-m-2)/2}$$

### Solution:


```python
print("Step 6: Compute Conditional Probabilities")
print("="*60)

# Total possible pairs
n_pairs_m = (N - m) * (N - m - 1) // 2
n_pairs_m1 = (N - m - 1) * (N - m - 2) // 2

print(f"\nFor length-m templates:")
print(f"  Total pairs: C(N-m, 2) = C({N-m}, 2)")
print(f"             = ({N-m}) × ({N-m-1}) / 2")
print(f"             = {(N-m) * (N-m-1)} / 2")
print(f"             = {n_pairs_m}")

print(f"\nFor length-(m+1) templates:")
print(f"  Total pairs: C(N-m-1, 2) = C({N-m-1}, 2)")
print(f"             = ({N-m-1}) × ({N-m-2}) / 2")
print(f"             = {(N-m-1) * (N-m-2)} / 2")
print(f"             = {n_pairs_m1}")

# Compute probabilities
B_prob = B_m / n_pairs_m if n_pairs_m > 0 else 0
A_prob = A_m1 / n_pairs_m1 if n_pairs_m1 > 0 else 0

print(f"\n" + "-"*60)
print(f"Computing B (probability of m-length match):")
print(f"\n  B = B^m / C(N-m, 2)")
print(f"    = {B_m} / {n_pairs_m}")
print(f"\n  ┌─────────────────────────┐")
print(f"  │  B = {B_prob:.6f}         │")
print(f"  └─────────────────────────┘")

print(f"\n" + "-"*60)
print(f"Computing A (probability of (m+1)-length match):")
print(f"\n  A = A^(m+1) / C(N-m-1, 2)")
print(f"    = {A_m1} / {n_pairs_m1}")
print(f"\n  ┌─────────────────────────┐")
print(f"  │  A = {A_prob:.6f}         │")
print(f"  └─────────────────────────┘")
```

    Step 6: Compute Conditional Probabilities
    ============================================================
    
    For length-m templates:
      Total pairs: C(N-m, 2) = C(18, 2)
                 = (18) × (17) / 2
                 = 306 / 2
                 = 153
    
    For length-(m+1) templates:
      Total pairs: C(N-m-1, 2) = C(17, 2)
                 = (17) × (16) / 2
                 = 272 / 2
                 = 136
    
    ------------------------------------------------------------
    Computing B (probability of m-length match):
    
      B = B^m / C(N-m, 2)
        = 6 / 153
    
      ┌─────────────────────────┐
      │  B = 0.039216         │
      └─────────────────────────┘
    
    ------------------------------------------------------------
    Computing A (probability of (m+1)-length match):
    
      A = A^(m+1) / C(N-m-1, 2)
        = 3 / 136
    
      ┌─────────────────────────┐
      │  A = 0.022059         │
      └─────────────────────────┘


---

## Step 7: Compute Sample Entropy

### Formula:
$$SampEn(m, r, N) = -\ln\left(\frac{A}{B}\right) = \ln(B) - \ln(A)$$

If A = 0 (no matches at m+1), SampEn is undefined (infinity).

### Solution:


```python
print("Step 7: Compute Sample Entropy")
print("="*60)
print(f"\nFormula: SampEn = -ln(A/B) = ln(B) - ln(A)")

print(f"\nSubstituting values:")
print(f"  A = {A_prob:.6f}")
print(f"  B = {B_prob:.6f}")

if A_prob > 0 and B_prob > 0:
    ratio = A_prob / B_prob
    print(f"\n  A / B = {A_prob:.6f} / {B_prob:.6f}")
    print(f"        = {ratio:.6f}")
    
    ln_ratio = np.log(ratio)
    print(f"\n  ln(A/B) = ln({ratio:.6f})")
    print(f"          = {ln_ratio:.6f}")
    
    SampEn = -ln_ratio
    print(f"\n  SampEn = -ln(A/B)")
    print(f"         = -({ln_ratio:.6f})")
    print(f"\n  ╔════════════════════════════════════════════╗")
    print(f"  ║                                            ║")
    print(f"  ║   SAMPLE ENTROPY:  SampEn = {SampEn:.6f}     ║")
    print(f"  ║                                            ║")
    print(f"  ╚════════════════════════════════════════════╝")
elif A_prob == 0:
    print(f"\n  A = 0, so A/B = 0")
    print(f"  ln(0) = -∞")
    print(f"  SampEn = -(-∞) = ∞")
    print(f"\n  ╔════════════════════════════════════════════╗")
    print(f"  ║  SampEn = ∞ (no (m+1)-length matches)      ║")
    print(f"  ╚════════════════════════════════════════════╝")
    SampEn = np.inf
else:
    print(f"\n  B = 0, entropy undefined (no m-length matches)")
    SampEn = np.nan
```

    Step 7: Compute Sample Entropy
    ============================================================
    
    Formula: SampEn = -ln(A/B) = ln(B) - ln(A)
    
    Substituting values:
      A = 0.022059
      B = 0.039216
    
      A / B = 0.022059 / 0.039216
            = 0.562500
    
      ln(A/B) = ln(0.562500)
              = -0.575364
    
      SampEn = -ln(A/B)
             = -(-0.575364)
    
      ╔════════════════════════════════════════════╗
      ║                                            ║
      ║   SAMPLE ENTROPY:  SampEn = 0.575364     ║
      ║                                            ║
      ╚════════════════════════════════════════════╝


---

## Step 8: Alternative Computation Using Full Series


```python
print("Step 8: Sample Entropy for Full Series (N = 69)")
print("="*60)

# Use full series
X_full_calc = X_full
N_full = len(X_full_calc)
r_full = 0.2 * np.std(X_full_calc)

print(f"\nParameters:")
print(f"  N = {N_full}")
print(f"  m = {m}")
print(f"  r = 0.2 × σ = 0.2 × {np.std(X_full_calc):.6f} = {r_full:.6f}")

# Count B^m matches
templates_m_full = [X_full_calc[i:i+m] for i in range(N_full - m)]
B_m_full = 0
for i in range(len(templates_m_full)):
    for j in range(i + 1, len(templates_m_full)):
        if chebyshev_distance(templates_m_full[i], templates_m_full[j]) < r_full:
            B_m_full += 1

# Count A^(m+1) matches
templates_m1_full = [X_full_calc[i:i+m+1] for i in range(N_full - m - 1)]
A_m1_full = 0
for i in range(len(templates_m1_full)):
    for j in range(i + 1, len(templates_m1_full)):
        if chebyshev_distance(templates_m1_full[i], templates_m1_full[j]) < r_full:
            A_m1_full += 1

n_pairs_m_full = (N_full - m) * (N_full - m - 1) // 2
n_pairs_m1_full = (N_full - m - 1) * (N_full - m - 2) // 2

B_prob_full = B_m_full / n_pairs_m_full if n_pairs_m_full > 0 else 0
A_prob_full = A_m1_full / n_pairs_m1_full if n_pairs_m1_full > 0 else 0

print(f"\nResults:")
print(f"  B^m = {B_m_full} matches out of {n_pairs_m_full} pairs")
print(f"  A^(m+1) = {A_m1_full} matches out of {n_pairs_m1_full} pairs")
print(f"\n  B = {B_prob_full:.6f}")
print(f"  A = {A_prob_full:.6f}")

if A_prob_full > 0 and B_prob_full > 0:
    SampEn_full = -np.log(A_prob_full / B_prob_full)
    print(f"\n  A/B = {A_prob_full / B_prob_full:.6f}")
    print(f"\n  ╔════════════════════════════════════════════════════╗")
    print(f"  ║                                                    ║")
    print(f"  ║   SAMPLE ENTROPY (full series): {SampEn_full:.6f}       ║")
    print(f"  ║                                                    ║")
    print(f"  ╚════════════════════════════════════════════════════╝")
else:
    SampEn_full = np.inf
    print(f"\n  SampEn = ∞ (no matches)")
```

    Step 8: Sample Entropy for Full Series (N = 69)
    ============================================================
    
    Parameters:
      N = 69
      m = 2
      r = 0.2 × σ = 0.2 × 0.121299 = 0.024260
    
    Results:
      B^m = 224 matches out of 2211 pairs
      A^(m+1) = 156 matches out of 2145 pairs
    
      B = 0.101312
      A = 0.072727
    
      A/B = 0.717857
    
      ╔════════════════════════════════════════════════════╗
      ║                                                    ║
      ║   SAMPLE ENTROPY (full series): 0.331485       ║
      ║                                                    ║
      ╚════════════════════════════════════════════════════╝


---

## Step 9: Interpretation

### Sample Entropy Scale:

| SampEn Value | Interpretation |
|--------------|----------------|
| SampEn ≈ 0 | Highly regular/predictable (many matches) |
| SampEn ≈ 0.5 | Moderate complexity |
| SampEn > 1.0 | High complexity/irregularity |
| SampEn = ∞ | Maximum irregularity (no pattern repeats) |


```python
print("Step 9: Physical Interpretation")
print("="*60)

se = SampEn_full if not np.isinf(SampEn_full) else SampEn

print(f"\nComputed Sample Entropy: SampEn = {se:.6f}")
print(f"\nInterpretation:")

if np.isinf(se):
    complexity = "MAXIMUM IRREGULARITY"
    meaning = "No repeating patterns detected"
elif se < 0.3:
    complexity = "LOW COMPLEXITY"
    meaning = "Highly regular/predictable signal"
elif se < 0.7:
    complexity = "MODERATE COMPLEXITY"
    meaning = "Some regularity with variation"
else:
    complexity = "HIGH COMPLEXITY"
    meaning = "Irregular, unpredictable signal"

print(f"\n  ┌────────────────────────────────────────────────────────┐")
print(f"  │  Complexity: {complexity:<43} │")
print(f"  │  Meaning:    {meaning:<43} │")
print(f"  └────────────────────────────────────────────────────────┘")

print(f"\n  For Battery B0047 capacity degradation:")
print(f"  • The capacity signal shows {complexity.lower()}")
print(f"  • {'High regularity suggests predictable degradation pattern' if se < 0.5 else 'Moderate complexity suggests some unpredictable variation'}")
```

    Step 9: Physical Interpretation
    ============================================================
    
    Computed Sample Entropy: SampEn = 0.331485
    
    Interpretation:
    
      ┌────────────────────────────────────────────────────────┐
      │  Complexity: MODERATE COMPLEXITY                         │
      │  Meaning:    Some regularity with variation              │
      └────────────────────────────────────────────────────────┘
    
      For Battery B0047 capacity degradation:
      • The capacity signal shows moderate complexity
      • High regularity suggests predictable degradation pattern


---

## Summary

### Complete Solution


```python
print("="*70)
print("COMPLETE SOLUTION SUMMARY")
print("="*70)

final_se = SampEn_full if not np.isinf(SampEn_full) else SampEn

print(f"""
GIVEN:
  • Battery B0047 capacity time series
  • N = {N_full} observations (full series)
  • Range: [{X_full.min():.4f}, {X_full.max():.4f}] Ah

PARAMETERS:
  • Embedding dimension: m = {m}
  • Tolerance: r = 0.2 × σ = {r_full:.6f}

METHOD: Sample Entropy (SampEn)

STEPS:
  1. Create template vectors of length m:
     uᵢ^(m) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ₋₁]
     
  2. Count B^m = matching pairs where d(uᵢ, uⱼ) < r
     B^{m} = {B_m_full} matches out of {n_pairs_m_full} pairs
     
  3. Create template vectors of length m+1:
     uᵢ^(m+1) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ]
     
  4. Count A^(m+1) = matching pairs for length m+1
     A^{m+1} = {A_m1_full} matches out of {n_pairs_m1_full} pairs
     
  5. Compute probabilities:
     B = B^m / C(N-m, 2) = {B_m_full} / {n_pairs_m_full} = {B_prob_full:.6f}
     A = A^(m+1) / C(N-m-1, 2) = {A_m1_full} / {n_pairs_m1_full} = {A_prob_full:.6f}
     
  6. Compute Sample Entropy:
     SampEn = -ln(A/B) = -ln({A_prob_full:.6f} / {B_prob_full:.6f})

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   FINAL ANSWER:  SampEn = {final_se:.6f}                        ║
║                                                                ║
║   INTERPRETATION: {'Low complexity (regular signal)' if final_se < 0.5 else 'Moderate complexity'}           ║
║   Battery degradation follows a relatively regular pattern    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
```

    ======================================================================
    COMPLETE SOLUTION SUMMARY
    ======================================================================
    
    GIVEN:
      • Battery B0047 capacity time series
      • N = 69 observations (full series)
      • Range: [1.1060, 1.6743] Ah
    
    PARAMETERS:
      • Embedding dimension: m = 2
      • Tolerance: r = 0.2 × σ = 0.024260
    
    METHOD: Sample Entropy (SampEn)
    
    STEPS:
      1. Create template vectors of length m:
         uᵢ^(m) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ₋₁]
    
      2. Count B^m = matching pairs where d(uᵢ, uⱼ) < r
         B^2 = 224 matches out of 2211 pairs
    
      3. Create template vectors of length m+1:
         uᵢ^(m+1) = [xᵢ, xᵢ₊₁, ..., xᵢ₊ₘ]
    
      4. Count A^(m+1) = matching pairs for length m+1
         A^3 = 156 matches out of 2145 pairs
    
      5. Compute probabilities:
         B = B^m / C(N-m, 2) = 224 / 2211 = 0.101312
         A = A^(m+1) / C(N-m-1, 2) = 156 / 2145 = 0.072727
    
      6. Compute Sample Entropy:
         SampEn = -ln(A/B) = -ln(0.072727 / 0.101312)
    
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║   FINAL ANSWER:  SampEn = 0.331485                        ║
    ║                                                                ║
    ║   INTERPRETATION: Low complexity (regular signal)           ║
    ║   Battery degradation follows a relatively regular pattern    ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    


---

*PRISM Behavioral Geometry Engine - Mathematical Derivation Proof*

*Battery: NASA B0047 | Signal: Capacity (Ah) | Method: Sample Entropy (SampEn)*
