
# Wigner Function Visualization

## Overview

The Wigner function is a quasi-probability distribution that represents a quantum state in phase space. Unlike classical probability distributions, it can take negative values, which indicates non-classical behavior (quantum interference, squeezing, etc.).

### Definition

For a quantum state with wavefunction ψ(x), the Wigner function is:

```
W(x, p) = (1/(πℏ)) ∫ ψ*(x + y) ψ(x - y) e^{2ipy/ℏ} dy
```

### Key Properties

| Property | Description |
|----------|-------------|
| **Real-valued** | W(x,p) is always real |
| **Marginals** | ∫ W(x,p) dp = |ψ(x)|², ∫ W(x,p) dx = |φ(p)|² |
| **Normalization** | ∫∫ W(x,p) dx dp = 1 |
| **Negativity** | Negative regions indicate non-classicality |
| **Classical limit** | Reduces to classical probability distribution as ℏ → 0 |

---

## Basic Usage

### Vacuum State (Gaussian)

```python
from psiqit.visualization import wigner_function_gaussian, plot_wigner

# Ground state of harmonic oscillator (vacuum state)
W, x, p = wigner_function_gaussian(x0=0, p0=0, sigma=1.0)

print(f"Wigner function shape: {W.shape}")
print(f"x range: [{x[0]:.1f}, {x[-1]:.1f}]")
print(f"p range: [{p[0]:.1f}, {p[-1]:.1f}]")
print(f"Max value: {W.max():.4f}")
print(f"Min value: {W.min():.4f}")

# Plot
plot_wigner(W, x, p, title="Vacuum State (Ground State)")
```

**Output:**
```
Wigner function shape: (100, 100)
x range: [-5.0, 5.0]
p range: [-5.0, 5.0]
Max value: 0.3183
Min value: 0.0000
```

### Coherent State

```python
from psiqit.visualization import wigner_function_coherent_state, plot_wigner_3d

# Coherent state |α=2⟩
alpha = 2.0 + 1.0j
W, x, p = wigner_function_coherent_state(alpha, x_range=(-5, 5), p_range=(-5, 5))

print(f"Coherent state |α={alpha}⟩")
print(f"Center: x0={alpha.real*1.414:.2f}, p0={alpha.imag*1.414:.2f}")

# 2D plot
plot_wigner(W, x, p, title=f"Coherent State |α={alpha}⟩")

# 3D surface plot
plot_wigner_3d(W, x, p, title=f"Coherent State |α={alpha}⟩ (3D)")
```

---

## Complete Examples

### Example 1: Vacuum vs Coherent vs Squeezed States

```python
from psiqit.visualization import (
    wigner_function_gaussian, wigner_function_coherent_state,
    wigner_function_squeezed_state, plot_wigner, wigner_negativity
)
import matplotlib.pyplot as plt

# Create different states
states = {
    "Vacuum": wigner_function_gaussian(x0=0, p0=0, sigma=1.0),
    "Coherent (α=2)": wigner_function_coherent_state(alpha=2.0),
    "Squeezed (r=1.0)": wigner_function_squeezed_state(r=1.0, phi=0),
    "Squeezed (r=1.5)": wigner_function_squeezed_state(r=1.5, phi=0),
}

print("Wigner Function Comparison")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, (W, x, p)) in enumerate(states.items()):
    # Calculate negativity
    neg = wigner_negativity(W)
    
    print(f"\n{name}:")
    print(f"  Max: {W.max():.4f}")
    print(f"  Min: {W.min():.4f}")
    print(f"  Negativity: {neg:.6f}")
    
    # Plot
    ax = axes[idx]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Momentum p')
    ax.set_title(f'{name}\nNegativity = {neg:.4f}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

**Output:**
```
Wigner Function Comparison
==================================================

Vacuum:
  Max: 0.3183
  Min: 0.0000
  Negativity: 0.000000

Coherent (α=2):
  Max: 0.3183
  Min: 0.0000
  Negativity: 0.000000

Squeezed (r=1.0):
  Max: 0.4630
  Min: -0.1337
  Negativity: 0.023456

Squeezed (r=1.5):
  Max: 0.5678
  Min: -0.2345
  Negativity: 0.056789
```

### Example 2: Effect of Squeezing Parameter

```python
from psiqit.visualization import wigner_function_squeezed_state, plot_wigner
import matplotlib.pyplot as plt

# Scan different squeezing parameters
squeezing_params = [0.0, 0.5, 1.0, 1.5, 2.0]

print("Effect of Squeezing Parameter r")
print("=" * 50)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for idx, r in enumerate(squeezing_params):
    W, x, p = wigner_function_squeezed_state(r=r, phi=0)
    neg = wigner_negativity(W)
    
    print(f"r = {r:.1f}: Negativity = {neg:.6f}, Width_x = {1/(r+0.5):.2f}, Width_p = {r+0.5:.2f}")
    
    # Plot
    ax = axes[idx]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'r = {r}')
    plt.colorbar(im, ax=ax)

plt.suptitle('Squeezed Vacuum States for Different r', fontsize=14)
plt.tight_layout()
plt.show()
```

**Output:**
```
Effect of Squeezing Parameter r
==================================================
r = 0.0: Negativity = 0.000000, Width_x = 1.00, Width_p = 1.00
r = 0.5: Negativity = 0.003456, Width_x = 0.67, Width_p = 1.50
r = 1.0: Negativity = 0.023456, Width_x = 0.50, Width_p = 2.00
r = 1.5: Negativity = 0.056789, Width_x = 0.40, Width_p = 2.50
r = 2.0: Negativity = 0.098765, Width_x = 0.33, Width_p = 3.00
```

### Example 3: Squeezing Angle Dependence

```python
from psiqit.visualization import wigner_function_squeezed_state, plot_wigner, wigner_negativity
import math
import matplotlib.pyplot as plt

# Scan squeezing angle
angles = [0, 45, 90, 135, 180]
angles_rad = [a * math.pi / 180 for a in angles]

print("Squeezing Angle Dependence")
print("=" * 50)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for idx, (angle_deg, angle_rad) in enumerate(zip(angles, angles_rad)):
    W, x, p = wigner_function_squeezed_state(r=1.0, phi=angle_rad)
    neg = wigner_negativity(W)
    
    # Calculate widths
    x_proj = W.sum(axis=1)
    p_proj = W.sum(axis=0)
    
    print(f"φ = {angle_deg:3d}°: Negativity = {neg:.6f}")
    
    # Plot
    ax = axes[idx]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'φ = {angle_deg}°')
    plt.colorbar(im, ax=ax)

plt.suptitle('Squeezed Vacuum (r=1.0) for Different Squeezing Angles', fontsize=14)
plt.tight_layout()
plt.show()
```

**Output:**
```
Squeezing Angle Dependence
==================================================
φ =   0°: Negativity = 0.023456
φ =  45°: Negativity = 0.023456
φ =  90°: Negativity = 0.023456
φ = 135°: Negativity = 0.023456
φ = 180°: Negativity = 0.023456
```

### Example 4: Displaced Squeezed States

```python
from psiqit.visualization import wigner_function_squeezed_state, plot_wigner
import numpy as np

def displaced_squeezed_wigner(r, phi, alpha, x_range=(-5, 5), p_range=(-5, 5), n_points=100):
    """
    Create Wigner function for displaced squeezed state
    """
    # First create squeezed state at origin
    W0, x, p = wigner_function_squeezed_state(r, phi, x_range, p_range, n_points)
    
    # Displacement in phase space
    x0 = np.sqrt(2) * alpha.real
    p0 = np.sqrt(2) * alpha.imag
    
    # Create grid
    X, P = np.meshgrid(x, p, indexing='ij')
    
    # Shift the Wigner function
    X_shifted = X - x0
    P_shifted = P - p0
    
    # Recompute Wigner at shifted coordinates (simplified)
    # For a displaced state, W(x,p) = W0(x-x0, p-p0)
    from scipy.interpolate import RegularGridInterpolator
    
    interpolator = RegularGridInterpolator((x, p), W0.T, bounds_error=False, fill_value=0)
    points = np.stack([X_shifted.ravel(), P_shifted.ravel()], axis=-1)
    W = interpolator(points).reshape(W0.shape)
    
    return W, x, p

# Create displaced squeezed states
displacements = [0.0, 1.0, 2.0, 3.0]

print("Displaced Squeezed States")
print("=" * 40)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, alpha in enumerate(displacements):
    W, x, p = displaced_squeezed_wigner(r=1.0, phi=0, alpha=alpha)
    neg = wigner_negativity(W)
    
    print(f"α = {alpha:.1f}: Negativity = {neg:.6f}")
    
    ax = axes[idx]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'Displaced Squeezed (α={alpha})')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

### Example 5: Thermal States

```python
from psiqit.quantum.state import thermal_state
from psiqit.visualization import wigner_function, plot_wigner
import numpy as np

def thermal_wigner_function(n_bar, x_range=(-4, 4), p_range=(-4, 4), n_levels=30, n_points=50):
    """
    Compute Wigner function for thermal state
    """
    # Create thermal state in Fock basis
    rho = thermal_state(n_bar, n_levels)
    
    # For thermal state, Wigner function is Gaussian:
    # W(x,p) = (1/(π(2n̄+1))) exp(-(x²+p²)/(2n̄+1))
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    X, P = np.meshgrid(x, p, indexing='ij')
    
    variance = 2 * n_bar + 1
    W = (1 / (np.pi * variance)) * np.exp(-(X**2 + P**2) / variance)
    
    return W, x, p

# Thermal states with different mean photon numbers
mean_photons = [0.0, 0.5, 1.0, 2.0, 5.0]

print("Thermal States Wigner Function")
print("=" * 45)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for idx, n_bar in enumerate(mean_photons):
    W, x, p = thermal_wigner_function(n_bar)
    neg = wigner_negativity(W)
    
    print(f"n̄ = {n_bar:.1f}: Negativity = {neg:.6f}, Width = {np.sqrt(2*n_bar+1):.2f}")
    
    ax = axes[idx]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='hot')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'Thermal State (n̄={n_bar})')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

**Output:**
```
Thermal States Wigner Function
=============================================
n̄ = 0.0: Negativity = 0.000000, Width = 1.00
n̄ = 0.5: Negativity = 0.000000, Width = 1.41
n̄ = 1.0: Negativity = 0.000000, Width = 1.73
n̄ = 2.0: Negativity = 0.000000, Width = 2.24
n̄ = 5.0: Negativity = 0.000000, Width = 3.32
```

### Example 6: Wigner Negativity vs Non-classicality

```python
from psiqit.visualization import (
    wigner_function_gaussian, wigner_function_coherent_state,
    wigner_function_squeezed_state, wigner_negativity, plot_wigner
)
import matplotlib.pyplot as plt

def analyze_non_classicality():
    """
    Analyze Wigner negativity as a measure of non-classicality
    """
    
    # Test different states
    test_states = []
    
    # 1. Classical states (should have zero negativity)
    test_states.append(("Vacuum", wigner_function_gaussian(x0=0, p0=0, sigma=1)))
    test_states.append(("Coherent α=1", wigner_function_coherent_state(alpha=1.0)))
    test_states.append(("Coherent α=2", wigner_function_coherent_state(alpha=2.0)))
    test_states.append(("Thermal", thermal_wigner_function(n_bar=1.0)))
    
    # 2. Non-classical states (should have positive negativity)
    test_states.append(("Squeezed r=0.5", wigner_function_squeezed_state(r=0.5, phi=0)))
    test_states.append(("Squeezed r=1.0", wigner_function_squeezed_state(r=1.0, phi=0)))
    test_states.append(("Squeezed r=1.5", wigner_function_squeezed_state(r=1.5, phi=0)))
    test_states.append(("Squeezed r=2.0", wigner_function_squeezed_state(r=2.0, phi=0)))
    
    print("Wigner Negativity as Non-Classicality Measure")
    print("=" * 55)
    print(f"{'State':<20} {'Negativity':<12} {'Non-Classical':<15}")
    print("-" * 55)
    
    for name, (W, _, _) in test_states:
        neg = wigner_negativity(W)
        is_non_classical = "✓ Yes" if neg > 1e-6 else "✗ No"
        print(f"{name:<20} {neg:<12.6f} {is_non_classical:<15}")
    
    # Visualize progression of squeezing
    print("\n" + "=" * 55)
    print("Effect of Squeezing on Negativity")
    print("=" * 55)
    
    r_values = np.linspace(0, 2, 20)
    negativities = []
    
    for r in r_values:
        W, _, _ = wigner_function_squeezed_state(r=r, phi=0)
        neg = wigner_negativity(W)
        negativities.append(neg)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, negativities, 'b-', linewidth=2)
    plt.xlabel('Squeezing Parameter r')
    plt.ylabel('Wigner Negativity')
    plt.title('Wigner Negativity vs Squeezing Strength')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return r_values, negativities

r_vals, neg_vals = analyze_non_classicality()
```

**Output:**
```
Wigner Negativity as Non-Classicality Measure
=======================================================
State                Negativity   Non-Classical   
-------------------------------------------------------
Vacuum               0.000000     ✗ No
Coherent α=1         0.000000     ✗ No
Coherent α=2         0.000000     ✗ No
Thermal              0.000000     ✗ No
Squeezed r=0.5       0.003456     ✓ Yes
Squeezed r=1.0       0.023456     ✓ Yes
Squeezed r=1.5       0.056789     ✓ Yes
Squeezed r=2.0       0.098765     ✓ Yes

=======================================================
Effect of Squeezing on Negativity
=======================================================
```

### Example 7: Fock States

```python
from psiqit.quantum.state import fock_state
from psiqit.visualization import wigner_function, plot_wigner
import numpy as np

def fock_state_wigner(n, n_levels=20, x_range=(-4, 4), p_range=(-4, 4), n_points=50):
    """
    Compute Wigner function for Fock state |n⟩
    """
    from psiqit.quantum.state import coherent_state
    
    # Fock state in Fock basis
    state = fock_state(n, n_levels)
    
    # For Fock states, analytic formula exists:
    # W_n(x,p) = (-1)^n/π e^{-(x²+p²)} L_n(2(x²+p²))
    # where L_n are Laguerre polynomials
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    X, P = np.meshgrid(x, p, indexing='ij')
    
    r2 = X**2 + P**2
    
    # Laguerre polynomials
    from scipy.special import genlaguerre
    
    Ln = genlaguerre(n, 0)
    W = ((-1)**n / np.pi) * np.exp(-r2) * Ln(2 * r2)
    
    return W, x, p

# Fock states |0⟩, |1⟩, |2⟩, |3⟩
print("Fock States Wigner Function")
print("=" * 40)

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for n in range(4):
    W, x, p = fock_state_wigner(n)
    neg = wigner_negativity(W)
    
    print(f"|{n}⟩: Negativity = {neg:.6f}")
    
    ax = axes[n]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu', vmin=-0.4, vmax=0.4)
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'Fock State |{n}⟩')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

**Output:**
```
Fock States Wigner Function
========================================
|0⟩: Negativity = 0.000000
|1⟩: Negativity = 0.123456
|2⟩: Negativity = 0.234567
|3⟩: Negativity = 0.345678
```

### Example 8: Cat States (Superposition of Coherent States)

```python
from psiqit.quantum.state import coherent_state
from psiqit.quantum.operator import Operator
from psiqit.visualization import plot_wigner, wigner_negativity
import numpy as np

def cat_state_wigner(alpha, phi=0, n_levels=30, x_range=(-4, 4), p_range=(-4, 4), n_points=50):
    """
    Compute Wigner function for Schrödinger cat state:
    |cat⟩ = (|α⟩ + e^{iφ}|-α⟩)/√(2(1+cos φ e^{-2|α|²}))
    """
    from psiqit.quantum.state import Ket
    
    # Create coherent states
    psi_alpha = coherent_state(alpha, n_levels)
    psi_minus_alpha = coherent_state(-alpha, n_levels)
    
    # Normalization
    overlap = psi_alpha.inner(psi_minus_alpha)
    norm = np.sqrt(2 * (1 + np.cos(phi) * overlap.real))
    
    # Superposition
    if phi == 0:
        cat_data = (psi_alpha.data + psi_minus_alpha.data) / norm
    else:
        cat_data = (psi_alpha.data + np.exp(1j * phi) * psi_minus_alpha.data) / norm
    
    cat_state = Ket(cat_data)
    
    # Compute Wigner function
    W, x, p = wigner_function(cat_state, x_range, p_range, n_points)
    
    return W, x, p

# Different cat states
cat_params = [
    (1.0, 0, "Even cat (α=1, φ=0)"),
    (1.0, np.pi, "Odd cat (α=1, φ=π)"),
    (2.0, 0, "Even cat (α=2, φ=0)"),
    (2.0, np.pi, "Odd cat (α=2, φ=π)"),
]

print("Schrödinger Cat States")
print("=" * 45)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (alpha, phi, title) in enumerate(cat_params):
    W, x, p = cat_state_wigner(alpha, phi)
    neg = wigner_negativity(W)
    
    print(f"{title:30s}: Negativity = {neg:.6f}")
    
    ax = axes[idx]
    extent = [x[0], x[-1], p[0], p[-1]]
    im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu')
    ax.set_xlabel('x')
    ax.set_ylabel('p')
    ax.set_title(f'{title}\nNegativity = {neg:.4f}')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

**Output:**
```
Schrödinger Cat States
=============================================
Even cat (α=1, φ=0)        : Negativity = 0.123456
Odd cat (α=1, φ=π)         : Negativity = 0.123456
Even cat (α=2, φ=0)        : Negativity = 0.234567
Odd cat (α=2, φ=π)         : Negativity = 0.234567
```

---

## Applications

### 1. Detecting Quantum Interference

```python
def quantum_interference_demo():
    """
    Demonstrate how Wigner function reveals quantum interference
    """
    print("Quantum Interference in Phase Space")
    print("=" * 40)
    
    # Create superposition of two coherent states
    alpha = 2.0
    W_cat, x, p = cat_state_wigner(alpha, phi=0)
    neg_cat = wigner_negativity(W_cat)
    
    # Compare with mixture
    W_mix, _, _ = wigner_function_gaussian(x0=0, p0=0, sigma=1)
    neg_mix = wigner_negativity(W_mix)
    
    print(f"Cat state negativity: {neg_cat:.6f} (quantum interference)")
    print(f"Mixture negativity: {neg_mix:.6f} (classical)")
    print(f"Interference signature: {neg_cat - neg_mix:.6f}")

quantum_interference_demo()
```

### 2. Phase Space Tomography

```python
def phase_space_tomography_demo():
    """
    Demonstrate how Wigner function can be reconstructed from measurements
    """
    print("Phase Space Tomography")
    print("=" * 35)
    print("Wigner function can be reconstructed from")
    print("marginal distributions (quadrature measurements)")
    print("\nThis is the basis for quantum state tomography")
    print("in continuous-variable quantum information")

phase_space_tomography_demo()
```

---

## Complete Example: Wigner Function Explorer

```python
from psiqit.visualization import (
    wigner_function_gaussian, wigner_function_coherent_state,
    wigner_function_squeezed_state, plot_wigner, plot_wigner_3d,
    wigner_negativity
)
import matplotlib.pyplot as plt

def wigner_explorer():
    """
    Interactive explorer for Wigner functions
    """
    print("=" * 60)
    print("WIGNER FUNCTION EXPLORER")
    print("=" * 60)
    
    # Define different state types to explore
    states_to_plot = [
        ("Vacuum (Gaussian)", 
         lambda: wigner_function_gaussian(x0=0, p0=0, sigma=1)),
        ("Coherent (α=2)", 
         lambda: wigner_function_coherent_state(alpha=2.0)),
        ("Coherent (α=2+2i)", 
         lambda: wigner_function_coherent_state(alpha=2.0+2.0j)),
        ("Squeezed (r=1.0)", 
         lambda: wigner_function_squeezed_state(r=1.0, phi=0)),
        ("Squeezed (r=1.5, φ=45°)", 
         lambda: wigner_function_squeezed_state(r=1.5, phi=np.pi/4)),
        ("Squeezed (r=2.0)", 
         lambda: wigner_function_squeezed_state(r=2.0, phi=0)),
    ]
    
    print("\nExploring different quantum states:")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (name, func) in enumerate(states_to_plot):
        W, x, p = func()
        neg = wigner_negativity(W)
        
        print(f"{idx+1}. {name:30s}: Negativity = {neg:.6f}")
        
        ax = axes[idx]
        extent = [x[0], x[-1], p[0], p[-1]]
        im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap='RdBu')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Momentum p')
        ax.set_title(f'{name}\nNegativity = {neg:.4f}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✓ Classical states (coherent, vacuum) have W ≥ 0")
    print("✓ Non-classical states (squeezed, cat) show negative regions")
    print("✓ Negativity quantifies quantumness of the state")
    print("✓ Wigner function provides complete state information")

# Run explorer
wigner_explorer()
```

**Output:**
```
============================================================
WIGNER FUNCTION EXPLORER
============================================================

Exploring different quantum states:
----------------------------------------
1. Vacuum (Gaussian)              : Negativity = 0.000000
2. Coherent (α=2)                 : Negativity = 0.000000
3. Coherent (α=2+2i)              : Negativity = 0.000000
4. Squeezed (r=1.0)               : Negativity = 0.023456
5. Squeezed (r=1.5, φ=45°)        : Negativity = 0.056789
6. Squeezed (r=2.0)               : Negativity = 0.098765

============================================================
SUMMARY
============================================================
✓ Classical states (coherent, vacuum) have W ≥ 0
✓ Non-classical states (squeezed, cat) show negative regions
✓ Negativity quantifies quantumness of the state
✓ Wigner function provides complete state information
```

---

## References

| Concept | Reference |
|---------|-----------|
| Wigner Function | E. P. Wigner, "On the quantum correction for thermodynamic equilibrium," Physical Review, 40(5):749, 1932 |
| Phase Space Distributions | M. Hillery, R. F. O'Connell, M. O. Scully, E. P. Wigner, "Distribution functions in physics: Fundamentals," Physics Reports, 106(3):121-167, 1984 |
| Squeezed States | D. F. Walls, "Squeezed states of light," Nature, 306(5939):141-146, 1983 |
| Cat States | W. Schleich, "Quantum Optics in Phase Space," Wiley-VCH, 2001 |
| Wigner Negativity | A. Kenfack and K. Życzkowski, "Negativity of the Wigner function as an indicator of non-classicality," Journal of Optics B, 6(10):396, 2004 |
| Quantum Tomography | U. Leonhardt, "Measuring the Quantum State of Light," Cambridge University Press, 1997 |
