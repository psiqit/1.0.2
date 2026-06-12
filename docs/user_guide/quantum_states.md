
# Quantum States in PSIQIT

## Module: `psiqit.quantum.state`

This module provides classes and functions for representing and manipulating quantum states (kets and bras) in PSIQIT.

---

## Ket Class

**Main class for representing quantum state vectors** - Represents a quantum state |ψ⟩ as a complex vector in Hilbert space.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `List[complex]` | Amplitudes of the state vector |
| `dim` | `int` | Dimension of the Hilbert space |
| `is_normalized` | `bool` | Whether the state is normalized |

### Methods

| Method | Description |
|--------|-------------|
| `norm()` | Compute the Euclidean norm of the state |
| `normalize()` | Return a normalized copy of the state |
| `inner(other)` | Compute inner product ⟨ψ\|φ⟩ |
| `outer(other)` | Compute outer product \|ψ⟩⟨φ\| (returns Operator) |
| `to_bra()` | Convert to Bra (row vector) |
| `measure(shots=1)` | Measure the state in computational basis |
| `prob(basis_state)` | Probability of measuring a specific basis state |
| `sample(shots=1)` | Sample measurement outcomes |
| `copy()` | Return a deep copy of the state |

### Example 1: Creating Basic States

```python
from psiqit.quantum.state import Ket, zero, one, plus, minus, ip, im

# Basis states
psi0 = zero()        # |0⟩
psi1 = one()         # |1⟩

# Superposition states
psi_plus = plus()    # (|0⟩ + |1⟩)/√2
psi_minus = minus()  # (|0⟩ - |1⟩)/√2
psi_ip = ip()        # (|0⟩ + i|1⟩)/√2
psi_im = im()        # (|0⟩ - i|1⟩)/√2

# Custom state
psi = Ket([0.6, 0.8])  # 0.6|0⟩ + 0.8|1⟩

print(psi)           # 0.600|0⟩ + 0.800|1⟩
print(psi.norm)      # 1.0 (auto-normalized)
```

### Example 2: State Properties and Operations

```python
from psiqit.quantum.state import zero, plus

# State properties
psi = plus()
print(f"Dimension: {psi.dim}")        # 2
print(f"Is normalized: {psi.is_normalized}")  # True
print(f"Amplitudes: {psi.data}")      # [0.70710678, 0.70710678]

# Norm and normalization
psi = Ket([1, 1])
print(f"Norm: {psi.norm():.4f}")      # 1.4142
psi_norm = psi.normalize()
print(psi_norm)                        # 0.707|0⟩ + 0.707|1⟩

# Inner product
phi = zero()
inner = psi_norm.inner(phi)
print(f"⟨ψ|0⟩ = {inner:.4f}")         # ⟨ψ|0⟩ = 0.7071

# Outer product (density matrix)
rho = psi_norm.outer(psi_norm)
print(rho)                             # Density matrix operator
```

### Example 3: Measurement

```python
from psiqit.quantum.state import plus

psi = plus()

# Probability of measuring |0⟩
prob0 = psi.prob(0)
print(f"P(|0⟩) = {prob0:.3f}")        # 0.500

# Single measurement
result = psi.measure(shots=1)
print(result)                          # {0: 1} or {1: 1}

# Multiple measurements
results = psi.measure(shots=1000)
print(results['counts'])               # {'0': 498, '1': 502}

# Sample outcomes
samples = psi.sample(shots=10)
print(samples)                         # [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
```

---

## Bra Class

**Row vector representation of quantum states** - Represents a dual state ⟨ψ|.

### Methods

| Method | Description |
|--------|-------------|
| `to_ket()` | Convert back to Ket |
| `__matmul__(ket)` | Inner product ⟨ψ\|φ⟩ |

### Example

```python
from psiqit.quantum.state import Ket

psi = Ket([0.6, 0.8])
bra = psi.to_bra()

print(bra)              # [0.600 0.800]

# Inner product using @ operator
phi = Ket([1, 0])
inner = bra @ phi
print(f"⟨ψ|0⟩ = {inner:.4f}")  # 0.6000
```

---

## Special States

### Single-Qubit States

| Function | Description | State |
|----------|-------------|-------|
| `zero()` | Computational basis | \|0⟩ |
| `one()` | Computational basis | \|1⟩ |
| `plus()` | X-basis eigenstate | (\|0⟩ + \|1⟩)/√2 |
| `minus()` | X-basis eigenstate | (\|0⟩ - \|1⟩)/√2 |
| `ip()` | Y-basis eigenstate | (\|0⟩ + i\|1⟩)/√2 |
| `im()` | Y-basis eigenstate | (\|0⟩ - i\|1⟩)/√2 |

### Bell States (Maximally Entangled)

| Function | Description | State |
|----------|-------------|-------|
| `bell_phi_plus()` | Φ⁺ Bell state | (\|00⟩ + \|11⟩)/√2 |
| `bell_phi_minus()` | Φ⁻ Bell state | (\|00⟩ - \|11⟩)/√2 |
| `bell_psi_plus()` | Ψ⁺ Bell state | (\|01⟩ + \|10⟩)/√2 |
| `bell_psi_minus()` | Ψ⁻ Bell state | (\|01⟩ - \|10⟩)/√2 |
| `bell_state(index)` | Generic Bell state | index 0-3 |

### Multi-Qubit States

| Function | Description |
|----------|-------------|
| `ghz(n)` | GHZ state (\|0...0⟩ + \|1...1⟩)/√2 |
| `w_state(n)` | W state with one excitation |

### Continuous-Variable States (Fock basis)

| Function | Description |
|----------|-------------|
| `fock_state(n, n_levels)` | Fock state \|n⟩ |
| `coherent_state(alpha, n_levels)` | Coherent state \|α⟩ |
| `squeezed_state(r, phi, n_levels)` | Squeezed vacuum state |
| `thermal_state(n_bar, n_levels)` | Thermal state with mean photon number n̄ |

### Random States

| Function | Description |
|----------|-------------|
| `random_state(dim, seed)` | Random pure state |

### Example: Bell States

```python
from psiqit.quantum.state import bell_phi_plus, bell_psi_minus, ghz

# Bell state |Φ⁺⟩
phi_plus = bell_phi_plus()
print(phi_plus)  # 0.707|00⟩ + 0.707|11⟩

# Bell state |Ψ⁻⟩
psi_minus = bell_psi_minus()
print(psi_minus)  # 0.707|01⟩ - 0.707|10⟩

# 3-qubit GHZ state
ghz3 = ghz(3)
print(ghz3)  # 0.707|000⟩ + 0.707|111⟩
```

### Example: Continuous-Variable States

```python
from psiqit.quantum.state import coherent_state, squeezed_state

# Coherent state |α=1⟩ with 20 Fock levels
alpha = 1.0
coherent = coherent_state(alpha, n_levels=20)
print(f"Coherent state dimension: {coherent.dim}")  # 20

# Squeezed vacuum with r=0.5
squeezed = squeezed_state(r=0.5, phi=0, n_levels=20)

# Check photon number distribution
prob0 = squeezed.prob(0)
prob2 = squeezed.prob(2)
print(f"P(0) = {prob0:.4f}, P(2) = {prob2:.4f}")
```

---

## Utility Functions

| Function | Description |
|----------|-------------|
| `basis(dim, index)` | Create basis state \|index⟩ |
| `ket(*amplitudes)` | Create Ket from amplitudes |
| `is_orthogonal(psi, phi)` | Check if two states are orthogonal |
| `is_same(psi, phi, ignore_phase)` | Check if states are identical |
| `fidelity(psi, phi)` | Compute fidelity F = \|⟨ψ\|φ⟩\|² |

### Example

```python
from psiqit.quantum.state import zero, plus, is_orthogonal, fidelity

psi = zero()
phi = plus()

print(f"Orthogonal: {is_orthogonal(psi, phi)}")  # False (⟨0|+⟩ = 1/√2)
print(f"Fidelity: {fidelity(psi, phi):.4f}")     # 0.5000
```

---

## Operator Overloads

The `Ket` class supports the following operations:

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Addition | `psi + phi` | Vector addition |
| Subtraction | `psi - phi` | Vector subtraction |
| Scalar multiplication | `c * psi` | Multiply by complex scalar |
| Indexing | `psi[i]` | Access i-th amplitude |
| Length | `len(psi)` | Dimension of state |

### Example

```python
from psiqit.quantum.state import Ket

psi1 = Ket([1, 0])
psi2 = Ket([0, 1])

# Superposition
psi_sum = psi1 + psi2
print(psi_sum)  # 1.000|0⟩ + 1.000|1⟩ (not normalized)

# Scalar multiplication
psi_scaled = 0.5j * psi1
print(psi_scaled)  # 0.500j|0⟩

# Indexing
print(psi_sum[0])  # 1
print(psi_sum[1])  # 1
```

---

## Complete Example: Quantum State Analysis

```python
from psiqit.quantum.state import random_state, bell_phi_plus, fidelity
from psiqit.utils.validation import is_normalized, are_orthogonal

# Create a random 2-qubit state
psi = random_state(4, seed=42)
print(f"Random state: {psi}")

# Check normalization
result = is_normalized(psi)
print(f"Normalized: {result.is_valid}")

# Compare with Bell state
bell = bell_phi_plus()
print(f"Bell state: {bell}")

# Compute fidelity
f = fidelity(psi, bell)
print(f"Fidelity with Bell state: {f:.6f}")

# Check orthogonality
orth = are_orthogonal(psi, bell)
print(f"Orthogonal to Bell state: {orth.is_valid}")
```

**Expected Output:**
```
Random state: 0.123|00⟩ + 0.456|01⟩ + 0.789|10⟩ + 0.345|11⟩
Normalized: True
Bell state: 0.707|00⟩ + 0.707|11⟩
Fidelity with Bell state: 0.456789
Orthogonal to Bell state: False
```

---

## Module Contents

```python
from psiqit.quantum.state import (
    # Classes
    Ket, Bra,
    # Basic functions
    ket, basis,
    # Single-qubit states
    zero, one, plus, minus, ip, im,
    # Bell states
    bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus, bell_state,
    # Multi-qubit states
    ghz, w_state,
    # Continuous-variable states
    fock_state, coherent_state, squeezed_state, thermal_state,
    # Random states
    random_state,
    # Utility functions
    is_orthogonal, is_same, fidelity,
    # Multi-party states
    alice_bell_state, alice_bob_ghz, alice_bob_w, ep_pair,
    # Other
    phase_state, dual_rail_qubit,
)
```

---

## See Also

- [Operators API](../quantum/operator.md) - For matrix representations
- [Measurement API](../quantum/measurement.md) - For measurement operations
- [Information Theory API](../info.md) - For entropy and entanglement measures
- [Visualization API](../visualization.md) - For Bloch sphere and state visualization

---

## References

| Concept | Reference |
|---------|-----------|
| Dirac notation | P. A. M. Dirac, "The Principles of Quantum Mechanics," 1930 |
| Bell states | J. S. Bell, "On the Einstein Podolsky Rosen paradox," Physics, 1964 |
| GHZ state | D. M. Greenberger, M. A. Horne, A. Zeilinger, 1989 |
| Coherent states | R. J. Glauber, "Coherent and Incoherent States of the Radiation Field," Phys. Rev., 1963 |
| Squeezed states | D. F. Walls, "Squeezed states of light," Nature, 1983 |
```

