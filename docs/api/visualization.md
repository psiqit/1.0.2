
# Visualization API

## Module: `psiqit.visualization`

This package provides comprehensive visualization tools for quantum computing, including Bloch sphere visualization, quantum circuit drawing, Wigner and Husimi phase-space functions, quantum animations, and symbolic equation rendering.

**Available tools:**
- **Bloch Sphere** - Visualize single-qubit states on the Bloch sphere
- **Circuit Drawing** - ASCII/Unicode quantum circuit diagrams
- **Wigner Function** - Phase-space quasi-probability distribution
- **Husimi Q-Function** - Non-negative phase-space distribution
- **Animation** - Time-dependent quantum evolution animations
- **Symbolic Plotting** - LaTeX equation rendering

---

## Bloch Sphere Visualization (`bloch.py`)

**Geometric representation of single-qubit states** - Any pure state of a qubit can be represented as a point on the unit sphere: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩, where θ ∈ [0, π] is the polar angle and φ ∈ [0, 2π) is the azimuthal angle.

### Functions

| Function | Description |
|----------|-------------|
| `state_to_bloch(state)` | Convert single-qubit state to (x, y, z) coordinates |
| `bloch_to_state(x, y, z)` | Convert Bloch coordinates to quantum state |
| `spherical_to_bloch(theta, phi)` | Convert spherical to Cartesian coordinates |
| `bloch_to_spherical(x, y, z)` | Convert Cartesian to spherical coordinates |
| `bloch_sphere(state, points, title)` | Plot Bloch sphere with optional state vector |
| `plot_multiple_states(states, colors)` | Plot multiple states on same Bloch sphere |
| `animate_bloch(states, interval)` | Animate sequence of states on Bloch sphere |

### Example 1: Converting Between Representations

```python
from psiqit.visualization import state_to_bloch, bloch_to_state, spherical_to_bloch
from psiqit.quantum.state import zero, one, plus, ip

# Get Bloch coordinates for common states
print(f"|0⟩: {state_to_bloch(zero())}")    # (0, 0, 1)
print(f"|1⟩: {state_to_bloch(one())}")     # (0, 0, -1)
print(f"|+⟩: {state_to_bloch(plus())}")    # (1, 0, 0)
print(f"|i+⟩: {state_to_bloch(ip())}")     # (0, 1, 0)

# Convert back from coordinates
psi = bloch_to_state(1, 0, 0)  # Should be |+⟩
print(psi)  # 0.707|0⟩ + 0.707|1⟩

# Spherical to Cartesian
x, y, z = spherical_to_bloch(theta=0, phi=0)  # North pole
print(f"North pole: ({x:.1f}, {y:.1f}, {z:.1f})")  # (0, 0, 1)
```

### Example 2: Basic Bloch Sphere Plot

```python
from psiqit.visualization import bloch_sphere
from psiqit.quantum.state import plus

# Plot a single state
bloch_sphere(plus(), title="|+⟩ State on Bloch Sphere")

# Plot with custom points
bloch_sphere(points=[(1,0,0), (0,1,0), (0,0,1)], 
             title="Cardinal Points")

# Save to file
bloch_sphere(plus(), filename="bloch_plus.png")
```

### Example 3: Multiple States on One Sphere

```python
from psiqit.visualization import plot_multiple_states
from psiqit.quantum.state import zero, one, plus, ip, im

states = [zero(), one(), plus(), ip(), im()]
colors = ['blue', 'red', 'green', 'orange', 'purple']

plot_multiple_states(states, colors=colors, 
                     title="Five Cardinal States")

# Without custom colors (auto-assigned)
plot_multiple_states(states)
```

### Example 4: Animating States on Bloch Sphere

```python
from psiqit.visualization import animate_bloch
from psiqit.quantum.state import zero, plus, one, minus, zero

# Create a rotation sequence
states = [zero(), plus(), one(), minus(), zero()]

# Animate (shows state vector moving)
animate_bloch(states, interval=500)

# Save animation
animate_bloch(states, filename="bloch_rotation.gif")
```

---

## Circuit Drawing (`circuit_drawer.py`)

**ASCII and Unicode quantum circuit diagrams** - Visualize quantum circuits using text-based diagrams with box-drawing characters.

### Functions

| Function | Description |
|----------|-------------|
| `draw_circuit(circuit, style)` | Draw circuit with ASCII or Unicode |
| `circuit_to_text(circuit)` | Simple text list of gates |
| `circuit_statistics(circuit)` | Get gate counts and statistics |

### Example 1: Basic Circuit Drawing

```python
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.visualization import draw_circuit, circuit_to_text

# Create Bell state circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Unicode style (more beautiful)
print(draw_circuit(circ, style='unicode'))
# q0: ─[H]─●─
# q1: ─────⊕─

# ASCII style (compatible everywhere)
print(draw_circuit(circ, style='ascii'))
# q0: ---[H]---o---
# q1: ---------X---

# Text representation with gate list
print(circuit_to_text(circ))
# Quantum Circuit: 2 qubits, depth 2
# --------------------------------------------------
#   1. H        on q0
#   2. CNOT     on q0, q1
```

### Example 2: Drawing Multi-Qubit Circuits

```python
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.visualization import draw_circuit

# GHZ state circuit
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)
circ.x(1)
circ.rz(2, 3.14159/2)

print(draw_circuit(circ, style='unicode'))
# q0: ─[H]─●──●──────
# q1: ─────⊕─────[X]─
# q2: ────────⊕──[Rz]─
```

### Example 3: Circuit Statistics

```python
from psiqit.visualization import circuit_statistics

stats = circuit_statistics(circ)
print(f"Qubits: {stats['n_qubits']}")
print(f"Depth: {stats['depth']}")
print(f"Total gates: {stats['n_gates']}")
print(f"Gate types: {stats['gate_types']}")
# Gate types: {'H': 1, 'CNOT': 2, 'X': 1, 'Rz': 1}
```

---

## Wigner Function (`wigner.py`)

**Wigner quasi-probability distribution** - Phase-space representation of quantum states. The Wigner function can be negative, indicating non-classical behavior such as squeezing or Schrödinger cat states.

### Functions

| Function | Description |
|----------|-------------|
| `wigner_function(state, x_range, p_range)` | Compute Wigner function numerically |
| `wigner_function_analytic(psi_func, ...)` | Compute from analytic wavefunction |
| `wigner_function_gaussian(x0, p0, sigma)` | Analytic Gaussian Wigner function |
| `wigner_function_coherent_state(alpha)` | Wigner function for coherent state |
| `wigner_function_squeezed_state(r, phi)` | Wigner function for squeezed vacuum |
| `plot_wigner(W, x, p, title)` | 2D color plot of Wigner function |
| `plot_wigner_3d(W, x, p, title)` | 3D surface plot |
| `wigner_negativity(W)` | Measure of non-classicality |

### Example 1: Gaussian and Coherent States

```python
from psiqit.visualization import (
    wigner_function_gaussian, wigner_function_coherent_state,
    plot_wigner, wigner_negativity
)

# Vacuum state (ground state of harmonic oscillator)
W, x, p = wigner_function_gaussian(x0=0, p0=0, sigma=1)
plot_wigner(W, x, p, title="Vacuum State")

# Coherent state |α=1⟩
W, x, p = wigner_function_coherent_state(alpha=1.0)
plot_wigner(W, x, p, title="Coherent State |α=1⟩")

# Check negativity (should be zero for coherent states)
neg = wigner_negativity(W)
print(f"Wigner negativity: {neg:.6f}")  # 0.000000
```

### Example 2: Squeezed States

```python
from psiqit.visualization import wigner_function_squeezed_state, plot_wigner_3d
import math

# Squeezed vacuum along x-direction
W, x, p = wigner_function_squeezed_state(r=1.0, phi=0)
plot_wigner(W, x, p, title="Squeezed Vacuum (r=1.0, φ=0)")

# Squeezed at 45 degrees
W, x, p = wigner_function_squeezed_state(r=1.0, phi=math.pi/4)
plot_wigner_3d(W, x, p, title="Squeezed Vacuum (3D View)")

# Squeezing increases negativity
for r in [0.0, 0.5, 1.0, 1.5]:
    W, _, _ = wigner_function_squeezed_state(r=r, phi=0)
    neg = wigner_negativity(W)
    print(f"r={r:.1f}, negativity={neg:.6f}")
```

### Example 3: Custom Wavefunction

```python
from psiqit.visualization import wigner_function_analytic, plot_wigner
import numpy as np

# Define a cat state (superposition of two coherent states)
def cat_state(x, alpha=2.0):
    # Simplified cat state: |α⟩ + |-α⟩
    psi1 = (1/np.pi**0.25) * np.exp(-(x - np.sqrt(2)*alpha)**2 / 2)
    psi2 = (1/np.pi**0.25) * np.exp(-(x + np.sqrt(2)*alpha)**2 / 2)
    return (psi1 + psi2) / np.sqrt(2 * (1 + np.exp(-2*alpha**2)))

W, x, p = wigner_function_analytic(cat_state, x_range=(-4, 4), p_range=(-4, 4))
plot_wigner(W, x, p, title="Schrödinger Cat State")
```

---

## Husimi Q-Function (`husimi.py`)

**Husimi Q-function (non-negative phase-space distribution)** - Always non-negative and can be interpreted as a probability distribution for simultaneous position and momentum measurements.

### Functions

| Function | Description |
|----------|-------------|
| `husimi_function_gaussian(x0, p0, sigma)` | Q-function for Gaussian state |
| `husimi_function_coherent_state(alpha)` | Q-function for coherent state |
| `plot_husimi(Q, x, p, title)` | 2D plot of Q-function |

### Example

```python
from psiqit.visualization import (
    husimi_function_gaussian, husimi_function_coherent_state,
    plot_husimi
)

# Vacuum state (Gaussian)
Q, x, p = husimi_function_gaussian(x0=0, p0=0, sigma=1)
plot_husimi(Q, x, p, title="Vacuum State (Husimi Q)")

# Coherent state |α=2⟩
Q, x, p = husimi_function_coherent_state(alpha=2.0 + 1.0j)
plot_husimi(Q, x, p, title="Coherent State (Husimi Q)")
```

---

## Quantum Animation (`animation.py`)

**Animate time-dependent quantum phenomena** - Create animations of wavefunction evolution and Bloch sphere dynamics.

### Functions

| Function | Description |
|----------|-------------|
| `animate_wavefunction(psi_func, x_range, t_range)` | Animate wavefunction evolution |
| `animate_bloch_sphere(states, interval)` | Animate sequence on Bloch sphere |

### Example 1: Wavefunction Animation

```python
from psiqit.visualization import animate_wavefunction
import numpy as np

# Gaussian wavepacket moving to the right
def moving_gaussian(x, t):
    return np.exp(-(x - t)**2 / 4) * np.exp(1j * 2 * x)

animate_wavefunction(moving_gaussian, 
                     x_range=(-5, 5), 
                     t_range=(0, 5),
                     n_frames=100,
                     interval=50)

# Stationary state of harmonic oscillator
def harmonic_oscillator(x, t):
    return (1/np.pi**0.25) * np.exp(-x**2/2) * np.exp(-0.5j * t)

animate_wavefunction(harmonic_oscillator,
                     x_range=(-5, 5),
                     t_range=(0, 10),
                     save_path="harmonic_oscillator.gif")
```

### Example 2: Bloch Sphere Animation

```python
from psiqit.visualization import animate_bloch_sphere
from psiqit.quantum.state import zero, plus, one, minus, ip, im

# Rotation around equator
states = [zero(), plus(), one(), minus(), zero()]
animate_bloch_sphere(states, interval=500)

# More complex trajectory
states = [zero(), ip(), one(), im(), zero()]
animate_bloch_sphere(states, interval=400, save_path="bloch_trajectory.gif")
```

---

## Symbolic Equation Plotting (`symbolic_plot.py`)

**Render mathematical equations as images** - Generate LaTeX representations of common quantum mechanics equations.

### Functions

| Function | Description |
|----------|-------------|
| `render_equation(latex, output_file)` | Render LaTeX equation to image |
| `schrodinger_equation(time_dependent)` | Schrödinger equation |
| `dirac_notation()` | Dirac notation (ket, bra, inner/outer product) |
| `pauli_matrices()` | Pauli matrix definitions |
| `bloch_sphere_equation(theta, phi)` | Bloch sphere representation |
| `heisenberg_uncertainty()` | Δx Δp ≥ ℏ/2 |
| `commutation_relation()` | [x, p] = iℏ |
| `maxwell_equations()` | Maxwell's equations |

### Example 1: Basic Equation Rendering

```python
from psiqit.visualization import render_equation, schrodinger_equation

# Get Schrödinger equation
eq = schrodinger_equation(time_dependent=True)
print(eq)
# i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)

# Render to image
render_equation(eq, output_file="schrodinger.png")

# Time-independent version
eq_ti = schrodinger_equation(time_dependent=False)
print(eq_ti)  # \hat{H}\psi(\mathbf{r}) = E\psi(\mathbf{r})
```

### Example 2: Dirac Notation

```python
from psiqit.visualization import dirac_notation

dirac = dirac_notation()
print(f"Ket: {dirac['ket']}")           # |\psi\rangle
print(f"Bra: {dirac['bra']}")           # \langle\psi|
print(f"Inner: {dirac['inner']}")       # \langle\phi|\psi\rangle
print(f"Outer: {dirac['outer']}")       # |\psi\rangle\langle\phi|
print(f"Expectation: {dirac['expectation']}")  # \langle\psi|\hat{A}|\psi\rangle
```

### Example 3: Pauli Matrices and Bloch Sphere

```python
from psiqit.visualization import pauli_matrices, bloch_sphere_equation

pauli = pauli_matrices()
print(pauli['sigma_x'])  # \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
print(pauli['sigma_y'])
print(pauli['sigma_z'])

# Bloch sphere with specific angles
eq = bloch_sphere_equation(theta=1.57, phi=0)
print(eq)  # |ψ⟩ = cos(0.78)|0⟩ + e^{0.00i} sin(0.78)|1⟩
```

### Example 4: Complete Set of Equations

```python
from psiqit.visualization import (
    render_equation, heisenberg_uncertainty, commutation_relation,
    maxwell_equations
)

# Uncertainty principle
render_equation(heisenberg_uncertainty(), output_file="uncertainty.png")

# Commutation relation
render_equation(commutation_relation(), output_file="commutation.png")

# Maxwell's equations
for i, eq in enumerate(maxwell_equations()):
    render_equation(eq, output_file=f"maxwell_{i+1}.png")
```

---

## Complete Examples

### Example 1: Visualization Pipeline for a Quantum State

```python
from psiqit.quantum.state import plus, ip
from psiqit.visualization import (
    state_to_bloch, bloch_sphere, plot_multiple_states,
    wigner_function_coherent_state, plot_wigner
)
import math

# Create a state |ψ⟩ = (|0⟩ + i|1⟩)/√2
psi = ip()

# 1. Get Bloch coordinates
x, y, z = state_to_bloch(psi)
print(f"Bloch coordinates: ({x:.2f}, {y:.2f}, {z:.2f})")

# 2. Plot on Bloch sphere
bloch_sphere(psi, title="State on Bloch Sphere")

# 3. Compare multiple states
plot_multiple_states([plus(), psi], 
                     colors=['green', 'orange'],
                     title="|+⟩ vs |i+⟩")

# 4. Phase-space representation (coherent state example)
alpha = complex(1, 1)
W, x, p = wigner_function_coherent_state(alpha)
plot_wigner(W, x, p, title=f"Coherent State α={alpha}")
```

### Example 2: Complete Circuit Visualization

```python
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.visualization import draw_circuit, circuit_to_text, circuit_statistics

# Build a quantum teleportation circuit
circ = QuantumCircuit(3)

# Create Bell pair between q1 and q2
circ.h(1)
circ.cx(1, 2)

# Prepare state to teleport on q0
circ.h(0)

# Bell measurement
circ.cx(0, 1)
circ.h(0)

# Display
print("=== Circuit Diagram ===")
print(draw_circuit(circ, style='unicode'))

print("\n=== Gate List ===")
print(circuit_to_text(circ))

print("\n=== Statistics ===")
stats = circuit_statistics(circ)
for key, value in stats.items():
    print(f"  {key}: {value}")
```

### Example 3: Animation of Rabi Oscillations

```python
from psiqit.visualization import animate_bloch_sphere
from psiqit.quantum.state import Ket
import math

# Create a sequence of states for Rabi oscillation
# |ψ(t)⟩ = cos(Ωt/2)|0⟩ + sin(Ωt/2)|1⟩
states = []
for t in range(100):
    theta = math.pi * t / 50  # Ωt = π * t/50
    cos_t = math.cos(theta/2)
    sin_t = math.sin(theta/2)
    state = Ket([cos_t, sin_t])
    states.append(state)

# Animate the oscillation
animate_bloch_sphere(states, interval=50, save_path="rabi_oscillation.gif")
```

---

## Module Contents

```python
__all__ = [
    # Bloch
    'state_to_bloch', 'bloch_to_state', 'spherical_to_bloch', 'bloch_to_spherical',
    'bloch_sphere', 'plot_multiple_states', 'animate_bloch',
    # Circuit drawing
    'draw_circuit', 'circuit_to_text', 'circuit_statistics',
    # Wigner
    'wigner_function', 'wigner_function_analytic', 'wigner_function_gaussian',
    'wigner_function_coherent_state', 'wigner_function_squeezed_state',
    'plot_wigner', 'plot_wigner_3d', 'wigner_negativity',
    # Husimi
    'husimi_function_gaussian', 'husimi_function_coherent_state', 'plot_husimi',
    # Animation
    'animate_wavefunction', 'animate_bloch_sphere',
    # Symbolic
    'render_equation', 'schrodinger_equation', 'dirac_notation', 'pauli_matrices',
    'bloch_sphere_equation', 'heisenberg_uncertainty', 'commutation_relation',
    'maxwell_equations',
]
```

---

## Summary Table: Visualization Methods

| Method | Purpose | Dimension | Output Type |
|--------|---------|-----------|-------------|
| Bloch Sphere | Single-qubit state | 3D | Interactive/static plot |
| Circuit Drawing | Quantum circuit structure | 2D text | ASCII/Unicode string |
| Wigner Function | Phase-space distribution | 2D/3D | Colormap plot |
| Husimi Q-Function | Non-negative phase-space | 2D | Colormap plot |
| Wavefunction Animation | Time evolution | 1D+time | Animated plot |
| Symbolic Equations | Mathematical notation | Text/LaTeX | Image or string |

---

## References

| Concept | Reference |
|---------|-----------|
| Bloch Sphere | F. Bloch, "Nuclear induction," Physical Review, 70(7-8):460, 1946 |
| Wigner Function | E. P. Wigner, "On the quantum correction for thermodynamic equilibrium," Physical Review, 40(5):749, 1932 |
| Husimi Q-Function | K. Husimi, "Some formal properties of the density matrix," Proc. Phys.-Math. Soc. Japan, 22(4):264-314, 1940 |
| Quantum Circuits | M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information," 2010 |
| Squeezed States | D. F. Walls, "Squeezed states of light," Nature, 306(5939):141-146, 1983 |
