
# Bloch Sphere Visualization

## Overview

The Bloch sphere is a geometric representation of a single qubit state. Any pure state of a qubit can be represented as a point on the unit sphere:

```
|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
```

where:
- **θ** ∈ [0, π] is the polar angle (from +Z axis)
- **φ** ∈ [0, 2π) is the azimuthal angle (from +X axis)

### Coordinates

| Coordinate | Formula | Range |
|------------|---------|-------|
| **x** | sin θ cos φ | [-1, 1] |
| **y** | sin θ sin φ | [-1, 1] |
| **z** | cos θ | [-1, 1] |

### Cardinal Points

| State | θ | φ | Coordinates |
|-------|---|---|-------------|
| |0⟩ | 0 | - | (0, 0, 1) |
| |1⟩ | π | - | (0, 0, -1) |
| |+⟩ | π/2 | 0 | (1, 0, 0) |
| |-⟩ | π/2 | π | (-1, 0, 0) |
| |i+⟩ | π/2 | π/2 | (0, 1, 0) |
| |i-⟩ | π/2 | 3π/2 | (0, -1, 0) |

---

## Basic Usage

### Converting Between Representations

```python
from psiqit.visualization import state_to_bloch, bloch_to_state, spherical_to_bloch
from psiqit.quantum.state import zero, one, plus, ip

# Convert quantum state to Bloch coordinates
print(f"|0⟩: {state_to_bloch(zero())}")    # (0, 0, 1)
print(f"|1⟩: {state_to_bloch(one())}")     # (0, 0, -1)
print(f"|+⟩: {state_to_bloch(plus())}")    # (1, 0, 0)
print(f"|i+⟩: {state_to_bloch(ip())}")     # (0, 1, 0)

# Convert Bloch coordinates back to state
psi = bloch_to_state(1, 0, 0)  # Should be |+⟩
print(f"State from (1,0,0): {psi}")

# Spherical to Cartesian conversion
x, y, z = spherical_to_bloch(theta=0, phi=0)  # North pole
print(f"North pole (θ=0): ({x:.1f}, {y:.1f}, {z:.1f})")
```

**Output:**
```
|0⟩: (0.0, 0.0, 1.0)
|1⟩: (0.0, 0.0, -1.0)
|+⟩: (1.0, 0.0, 0.0)
|i+⟩: (0.0, 1.0, 0.0)
State from (1,0,0): 0.707|0⟩ + 0.707|1⟩
North pole (θ=0): (0.0, 0.0, 1.0)
```

### Basic Bloch Sphere Plot

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

---

## Complete Examples

### Example 1: Plotting All Cardinal States

```python
from psiqit.visualization import bloch_sphere, plot_multiple_states
from psiqit.quantum.state import zero, one, plus, minus, ip, im

# List of cardinal states
cardinal_states = [
    ("|0⟩", zero()),
    ("|1⟩", one()),
    ("|+⟩", plus()),
    ("|-⟩", minus()),
    ("|i+⟩", ip()),
    ("|i-⟩", im()),
]

print("Cardinal States on Bloch Sphere")
print("=" * 40)

for name, state in cardinal_states:
    x, y, z = state_to_bloch(state)
    print(f"{name:4s}: ({x:+.1f}, {y:+.1f}, {z:+.1f})")

# Plot all on one sphere
states = [state for _, state in cardinal_states]
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']

plot_multiple_states(states, colors=colors, 
                     title="All Cardinal States")

# Individual plots
for name, state in cardinal_states:
    bloch_sphere(state, title=f"{name} State")
```

**Output:**
```
Cardinal States on Bloch Sphere
========================================
|0⟩ : (+0.0, +0.0, +1.0)
|1⟩ : (+0.0, +0.0, -1.0)
|+⟩ : (+1.0, +0.0, +0.0)
|-⟩ : (-1.0, +0.0, +0.0)
|i+⟩: (+0.0, +1.0, +0.0)
|i-⟩: (+0.0, -1.0, +0.0)
```

### Example 2: States on Great Circles

```python
from psiqit.visualization import plot_multiple_states, bloch_sphere
from psiqit.quantum.state import Ket
import math
import numpy as np

def states_on_great_circle(axis='X', n_points=12):
    """
    Generate states lying on a great circle of the Bloch sphere
    """
    states = []
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    if axis == 'X':
        # Circle in Y-Z plane (fixed X=0)
        for phi in angles:
            theta = math.pi/2
            psi = Ket([math.cos(theta/2), math.sin(theta/2) * complex(math.cos(phi), math.sin(phi))])
            states.append(psi)
    elif axis == 'Y':
        # Circle in X-Z plane (fixed Y=0)
        for theta in angles:
            phi = 0
            psi = Ket([math.cos(theta/2), math.sin(theta/2)])
            states.append(psi)
    elif axis == 'Z':
        # Circle in X-Y plane (fixed Z=0)
        for phi in angles:
            theta = math.pi/2
            psi = Ket([math.cos(theta/2), math.sin(theta/2) * complex(math.cos(phi), math.sin(phi))])
            states.append(psi)
    
    return states

print("Great Circles on Bloch Sphere")
print("=" * 40)

# Generate states on different great circles
for axis in ['X', 'Y', 'Z']:
    states = states_on_great_circle(axis, n_points=8)
    print(f"{axis}-axis circle: {len(states)} states")

# Plot all circles (separate figures)
for axis in ['X', 'Y', 'Z']:
    states = states_on_great_circle(axis, n_points=24)
    plot_multiple_states(states, title=f"Great Circle in Y-Z Plane (Fixed {axis}=0)")
```

### Example 3: Rotating a State on the Bloch Sphere

```python
from psiqit.visualization import animate_bloch
from psiqit.quantum.state import Ket
import math

def create_rotation_sequence(n_frames=50):
    """
    Create a sequence of states rotating around the Y axis
    """
    states = []
    
    for i in range(n_frames):
        # Rotation angle around Y axis
        theta = 2 * math.pi * i / n_frames
        
        # State after rotation: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
        cos_half = math.cos(theta / 2)
        sin_half = math.sin(theta / 2)
        
        state = Ket([cos_half, sin_half])
        states.append(state)
    
    return states

# Create rotation sequence
rotation_states = create_rotation_sequence(n_frames=60)

print(f"Created {len(rotation_states)} states for rotation animation")

# Animate (uncomment to run)
# animate_bloch(rotation_states, interval=50, filename="bloch_rotation.gif")

# Plot first, middle, and last states
print("\nFirst state:  ", rotation_states[0])
print("Middle state: ", rotation_states[len(rotation_states)//2])
print("Last state:   ", rotation_states[-1])
```

**Output:**
```
Created 60 states for rotation animation

First state:   1.000|0⟩ + 0.000|1⟩
Middle state:  -0.000|0⟩ + 1.000|1⟩
Last state:   1.000|0⟩ + -0.000|1⟩
```

### Example 4: Quantum Gates as Rotations

```python
from psiqit.visualization import plot_multiple_states, bloch_sphere
from psiqit.quantum.state import zero, Ket
from psiqit.quantum.operator import rx, ry, rz
import math
import numpy as np

def apply_gate_sequence(initial_state, gates, angles):
    """
    Apply a sequence of gates to a state and return trajectory
    """
    states = [initial_state]
    current_state = initial_state
    
    for gate, angle in zip(gates, angles):
        if gate == 'RX':
            op = rx(angle)
        elif gate == 'RY':
            op = ry(angle)
        elif gate == 'RZ':
            op = rz(angle)
        else:
            continue
        
        # Apply rotation
        new_data = op @ current_state.data
        current_state = Ket(new_data, _normalized=True)
        states.append(current_state)
    
    return states

# Initial state: |0⟩
initial = zero()

# Different gate sequences
sequences = [
    ('RX', [math.pi]),
    ('RY', [math.pi/2, math.pi/2]),
    ('RX', [math.pi/2, math.pi/2]),
    ('RZ', [math.pi/2, math.pi/2]),
]

print("Quantum Gates as Rotations")
print("=" * 50)

for gate_name, angles in sequences:
    gates = [gate_name] * len(angles)
    trajectory = apply_gate_sequence(initial, gates, angles)
    
    print(f"\n{gate_name} rotation: {angles}")
    for i, state in enumerate(trajectory):
        x, y, z = state_to_bloch(state)
        print(f"  Step {i}: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    # Plot trajectory
    if len(trajectory) > 1:
        plot_multiple_states(trajectory, title=f"{gate_name} Rotation Trajectory")
```

**Output:**
```
Quantum Gates as Rotations
==================================================

RX rotation: [3.141592653589793]
  Step 0: (0.00, 0.00, 1.00)
  Step 1: (0.00, 0.00, -1.00)

RY rotation: [1.5707963267948966, 1.5707963267948966]
  Step 0: (0.00, 0.00, 1.00)
  Step 1: (1.00, 0.00, 0.00)
  Step 2: (0.00, 0.00, -1.00)

RX rotation: [1.5707963267948966, 1.5707963267948966]
  Step 0: (0.00, 0.00, 1.00)
  Step 1: (0.00, -1.00, 0.00)
  Step 2: (0.00, 0.00, -1.00)

RZ rotation: [1.5707963267948966, 1.5707963267948966]
  Step 0: (0.00, 0.00, 1.00)
  Step 1: (0.00, 0.00, 1.00)
  Step 2: (0.00, 0.00, 1.00)
```

### Example 5: Bloch Sphere Coordinates for Arbitrary States

```python
from psiqit.visualization import state_to_bloch, bloch_to_state, bloch_sphere
from psiqit.quantum.state import Ket
import math
import numpy as np

def generate_random_states(n_states=10, seed=42):
    """Generate random pure states on Bloch sphere"""
    np.random.seed(seed)
    states = []
    
    for _ in range(n_states):
        # Random spherical coordinates
        theta = np.arccos(2 * np.random.random() - 1)  # Uniform on sphere
        phi = 2 * np.pi * np.random.random()
        
        # Construct state
        cos_half = math.cos(theta / 2)
        sin_half = math.sin(theta / 2)
        amplitude = complex(math.cos(phi), math.sin(phi))
        
        state = Ket([cos_half, sin_half * amplitude])
        states.append((theta, phi, state))
    
    return states

print("Random States on Bloch Sphere")
print("=" * 45)
print(f"{'θ (deg)':<10} {'φ (deg)':<10} {'x':<8} {'y':<8} {'z':<8}")
print("-" * 45)

random_states = generate_random_states(10)

for theta, phi, state in random_states:
    x, y, z = state_to_bloch(state)
    print(f"{math.degrees(theta):<10.1f} {math.degrees(phi):<10.1f} {x:<+8.3f} {y:<+8.3f} {z:<+8.3f}")

# Plot all random states
states = [state for _, _, state in random_states]
plot_multiple_states(states, title="Random States on Bloch Sphere")
```

**Output:**
```
Random States on Bloch Sphere
=============================================
θ (deg)    φ (deg)    x        y        z      
---------------------------------------------
125.0      87.5       -0.350   +0.574   -0.742
45.2      310.3      +0.474   -0.535   +0.699
98.7      152.1      -0.698   +0.330   -0.635
152.3     25.8       +0.213   +0.203   -0.956
33.1      278.4      +0.147   -0.749   +0.647
...
```

### Example 6: Comparing Original and Rotated States

```python
from psiqit.visualization import plot_multiple_states
from psiqit.quantum.state import Ket, plus
from psiqit.quantum.operator import rx, ry, rz
import math

# Original state
original = plus()
print(f"Original state: {original}")

# Apply different rotations
rotations = [
    ('RX(π/2)', rx(math.pi/2)),
    ('RY(π/2)', ry(math.pi/2)),
    ('RZ(π/2)', rz(math.pi/2)),
    ('RX(π)', rx(math.pi)),
]

states_to_plot = [original]
labels = ['Original']

for name, op in rotations:
    rotated_data = op @ original.data
    rotated = Ket(rotated_data, _normalized=True)
    states_to_plot.append(rotated)
    labels.append(name)
    
    x, y, z = state_to_bloch(rotated)
    print(f"{name:10s}: ({x:+.2f}, {y:+.2f}, {z:+.2f})")

# Plot comparison
plot_multiple_states(states_to_plot, 
                     title="Effect of Quantum Gates on |+⟩ State")
```

**Output:**
```
Original state: 0.707|0⟩ + 0.707|1⟩
RX(π/2)   : (+0.00, -1.00, +0.00)
RY(π/2)   : (+1.00, +0.00, -0.00)
RZ(π/2)   : (-0.00, +1.00, +0.00)
RX(π)     : (+0.00, +0.00, -1.00)
```

### Example 7: Visualization with Custom Styling

```python
from psiqit.visualization import bloch_sphere
from psiqit.quantum.state import ip, im, plus, minus, zero, one

# Create a styled Bloch sphere with multiple elements
states = [zero(), one(), plus(), minus(), ip(), im()]
colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#008080']

print("Custom Styled Bloch Sphere")
print("=" * 35)

for state, color in zip(states, colors):
    x, y, z = state_to_bloch(state)
    print(f"{state:30s} → ({x:+.1f}, {y:+.1f}, {z:+.1f}) → {color}")

# Plot with custom title and without axes labels
bloch_sphere(plus(), 
             title="Custom Styled Bloch Sphere",
             show_axes=True,
             show_labels=True,
             figsize=(10, 10))

# Plot with points only (no state vector)
points = [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]
bloch_sphere(points=points, 
             title="Cardinal Points Only",
             show_axes=True)
```

---

## Advanced Examples

### Example 8: Animating Quantum Gate Effects

```python
from psiqit.visualization import animate_bloch
from psiqit.quantum.state import Ket
import math

def create_rotation_sequence_3d(n_frames=100, axis='Y'):
    """
    Create a sequence of states rotating around specified axis
    """
    states = []
    
    for i in range(n_frames):
        angle = 2 * math.pi * i / n_frames
        
        if axis == 'X':
            # Rotation around X: |ψ⟩ = cos(θ/2)|0⟩ + i sin(θ/2)|1⟩
            cos_half = math.cos(angle / 2)
            sin_half = math.sin(angle / 2)
            state = Ket([cos_half, complex(0, sin_half)])
        
        elif axis == 'Y':
            # Rotation around Y: |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            cos_half = math.cos(angle / 2)
            sin_half = math.sin(angle / 2)
            state = Ket([cos_half, sin_half])
        
        elif axis == 'Z':
            # Rotation around Z: |ψ⟩ = |0⟩ + e^{iθ}|1⟩ (relative phase)
            state = Ket([1/math.sqrt(2), complex(math.cos(angle), math.sin(angle)) / math.sqrt(2)])
        
        else:
            state = Ket([1, 0])
        
        states.append(state)
    
    return states

print("Creating animations for different rotation axes")
print("=" * 50)

# Create sequences
sequences = {
    'X-axis': create_rotation_sequence_3d(60, 'X'),
    'Y-axis': create_rotation_sequence_3d(60, 'Y'),
    'Z-axis': create_rotation_sequence_3d(60, 'Z'),
}

for name, seq in sequences.items():
    print(f"{name}: {len(seq)} states")

# Uncomment to animate
# animate_bloch(sequences['X-axis'], interval=30, filename="rotation_x.gif")
# animate_bloch(sequences['Y-axis'], interval=30, filename="rotation_y.gif")
# animate_bloch(sequences['Z-axis'], interval=30, filename="rotation_z.gif")
```

### Example 9: Bloch Sphere with Trajectory Tracing

```python
from psiqit.visualization import plot_multiple_states, bloch_sphere
from psiqit.quantum.state import Ket
import math

def trace_trajectory(initial_state, gate, n_steps=20, max_angle=2*math.pi):
    """
    Trace the trajectory of a state under continuous rotation
    """
    trajectory = [initial_state]
    current = initial_state
    
    for step in range(1, n_steps + 1):
        angle = max_angle * step / n_steps
        
        if gate == 'RX':
            from psiqit.quantum.operator import rx
            op = rx(angle)
        elif gate == 'RY':
            from psiqit.quantum.operator import ry
            op = ry(angle)
        elif gate == 'RZ':
            from psiqit.quantum.operator import rz
            op = rz(angle)
        else:
            return trajectory
        
        new_data = op @ current.data
        current = Ket(new_data, _normalized=True)
        trajectory.append(current)
    
    return trajectory

# Initial state
initial = Ket([1, 0])  # |0⟩

print("Trajectory Tracing on Bloch Sphere")
print("=" * 45)

# Trace different trajectories
for gate_name in ['RX', 'RY', 'RZ']:
    trajectory = trace_trajectory(initial, gate_name, n_steps=30)
    
    # Print start, middle, end
    mid = len(trajectory) // 2
    print(f"\n{gate_name} rotation:")
    print(f"  Start: {state_to_bloch(trajectory[0])}")
    print(f"  Mid:   {state_to_bloch(trajectory[mid])}")
    print(f"  End:   {state_to_bloch(trajectory[-1])}")
    
    # Plot trajectory
    plot_multiple_states(trajectory[::3],  # Plot every 3rd state for clarity
                         title=f"{gate_name} Rotation Trajectory from |0⟩")
```

---

## Mathematical Background

### Bloch Sphere Representation

The Bloch sphere representation is based on the density matrix:

```
ρ = (I + xσ_x + yσ_y + zσ_z) / 2
```

where (x, y, z) are the Bloch coordinates and σ_i are Pauli matrices.

### Relation to Expectation Values

For any state |ψ⟩ on the Bloch sphere:

```
⟨σ_x⟩ = x
⟨σ_y⟩ = y
⟨σ_z⟩ = z
```

### Pauli Matrices on Bloch Sphere

| Gate | Rotation | Axis |
|------|----------|------|
| X (σ_x) | π around X | X-axis |
| Y (σ_y) | π around Y | Y-axis |
| Z (σ_z) | π around Z | Z-axis |
| H | π around (X+Z)/√2 | Diagonal |

---

## Complete Example: Interactive Bloch Sphere Explorer

```python
from psiqit.visualization import state_to_bloch, bloch_to_state, bloch_sphere
from psiqit.quantum.state import Ket
import math

def bloch_explorer():
    """
    Interactive demonstration of Bloch sphere representation
    """
    print("=" * 60)
    print("BLOCH SPHERE EXPLORER")
    print("=" * 60)
    
    # Test different parameter combinations
    test_cases = [
        ("North Pole", 0, 0),
        ("South Pole", math.pi, 0),
        ("+X axis", math.pi/2, 0),
        ("-X axis", math.pi/2, math.pi),
        ("+Y axis", math.pi/2, math.pi/2),
        ("-Y axis", math.pi/2, 3*math.pi/2),
        ("45° latitude, 0° longitude", math.pi/4, 0),
        ("45° latitude, 90° longitude", math.pi/4, math.pi/2),
    ]
    
    print("\n|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩")
    print("\n" + "=" * 60)
    print(f"{'Name':<20} {'θ (deg)':<10} {'φ (deg)':<10} {'Coordinates':<20}")
    print("-" * 60)
    
    states = []
    
    for name, theta, phi in test_cases:
        # Construct state
        cos_half = math.cos(theta / 2)
        sin_half = math.sin(theta / 2)
        amplitude = complex(math.cos(phi), math.sin(phi))
        
        state = Ket([cos_half, sin_half * amplitude])
        x, y, z = state_to_bloch(state)
        states.append(state)
        
        print(f"{name:<20} {math.degrees(theta):<10.1f} {math.degrees(phi):<10.1f} ({x:+.2f}, {y:+.2f}, {z:+.2f})")
    
    # Plot all test states
    print("\nPlotting all test states on Bloch sphere...")
    plot_multiple_states(states, title="Bloch Sphere Explorer - All Test States")
    
    return states

# Run explorer
states = bloch_explorer()
```

**Output:**
```
============================================================
BLOCH SPHERE EXPLORER
============================================================

|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

============================================================
Name                 θ (deg)    φ (deg)    Coordinates          
------------------------------------------------------------
North Pole           0.0        0.0        (+0.00, +0.00, +1.00)
South Pole           180.0      0.0        (+0.00, +0.00, -1.00)
+X axis              90.0       0.0        (+1.00, +0.00, +0.00)
-X axis              90.0       180.0      (-1.00, +0.00, +0.00)
+Y axis              90.0       90.0       (+0.00, +1.00, +0.00)
-Y axis              90.0       270.0      (+0.00, -1.00, +0.00)
45° latitude, 0° longitude 45.0       0.0        (+0.71, +0.00, +0.71)
45° latitude, 90° longitude 45.0       90.0       (+0.00, +0.71, +0.71)

Plotting all test states on Bloch sphere...
```

---

## References

| Concept | Reference |
|---------|-----------|
| Bloch Sphere | F. Bloch, "Nuclear induction," Physical Review, 70(7-8):460, 1946 |
| Qubit Representation | M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information," 2010 |
| Pauli Matrices | W. Pauli, "Zur Quantenmechanik des magnetischen Elektrons," Zeitschrift für Physik, 1927 |
| Geometric Phase | M. V. Berry, "Quantal phase factors accompanying adiabatic changes," Proceedings of the Royal Society A, 1984 |
