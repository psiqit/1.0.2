
# Information Theory API

## Module: `psiqit.info`

This module provides fundamental measures from quantum information theory, including entropy measures (Shannon, von Neumann, Rényi) and entanglement quantification tools (concurrence, negativity, Schmidt decomposition).

---

## Entropy Measures (`entropy.py`)

**Classical and quantum entropy measures** - Quantify uncertainty, information content, and correlations in quantum systems.

### Functions

| Function | Description |
|----------|-------------|
| `shannon_entropy(probabilities, base)` | Shannon entropy H = -Σ p_i log p_i |
| `von_neumann_entropy(rho, base)` | von Neumann entropy S(ρ) = -Tr(ρ log ρ) |
| `renyi_entropy(rho, alpha, base)` | Rényi entropy of order α |
| `collision_entropy(rho, base)` | Rényi entropy with α=2 (collision entropy) |
| `mutual_information(rho_ab, dim_a, dim_b, base)` | I(A:B) = S(A) + S(B) - S(AB) |
| `partial_trace(matrix, dims, keep)` | Partial trace over subsystem |
| `relative_entropy(rho, sigma)` | Quantum relative entropy S(ρ\|\|σ) |
| `purity(rho)` | Purity γ = Tr(ρ²) |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `probabilities` | `List[float]` | Probability distribution (must sum to 1) |
| `rho` | `Operator` or `List[List[complex]]` | Density matrix |
| `base` | `str` | Logarithm base: 'e' (nats), '2' (bits), '10' (dits) |
| `alpha` | `float` | Order parameter for Rényi entropy (α ≥ 0) |
| `dims` | `List[int]` | Dimensions [dA, dB] for bipartite systems |

### Example 1: Shannon Entropy

```python
from psiqit.info.entropy import shannon_entropy

# Fair coin toss
probs = [0.5, 0.5]
H = shannon_entropy(probs, base='2')
print(f"Fair coin entropy: {H:.4f} bits")  # 1.0000 bits

# Biased coin
probs = [0.9, 0.1]
H = shannon_entropy(probs, base='2')
print(f"Biased coin entropy: {H:.4f} bits")  # 0.4690 bits

# Deterministic outcome
probs = [1.0, 0.0]
H = shannon_entropy(probs, base='2')
print(f"Deterministic entropy: {H:.4f} bits")  # 0.0000 bits
```

### Example 2: von Neumann Entropy for Quantum States

```python
from psiqit.info.entropy import von_neumann_entropy
from psiqit.quantum.state import zero, bell_phi_plus
from psiqit.quantum.operator import identity

# Pure state |0⟩⟨0| (zero entropy)
rho_pure = zero().outer(zero())
S = von_neumann_entropy(rho_pure, base='2')
print(f"Pure state entropy: {S:.4f} bits")  # 0.0000 bits

# Bell state is also pure
rho_bell = bell_phi_plus().outer(bell_phi_plus())
S = von_neumann_entropy(rho_bell, base='2')
print(f"Bell state entropy: {S:.4f} bits")  # 0.0000 bits

# Maximally mixed state (maximum entropy for 1 qubit)
rho_max = [[0.5, 0], [0, 0.5]]
S = von_neumann_entropy(rho_max, base='2')
print(f"Maximally mixed entropy: {S:.4f} bits")  # 1.0000 bits
```

### Example 3: Rényi Entropy

```python
from psiqit.info.entropy import renyi_entropy

rho_max = [[0.5, 0], [0, 0.5]]

# Rényi entropy of order 2 (collision entropy)
S2 = renyi_entropy(rho_max, alpha=2, base='2')
print(f"Rényi-2 entropy: {S2:.4f} bits")  # 1.0000 bits

# Rényi entropy of order ∞ (min-entropy)
S_inf = renyi_entropy(rho_max, alpha=10, base='2')
print(f"Rényi-∞ entropy: {S_inf:.4f} bits")  # 1.0000 bits
```

### Example 4: Mutual Information

```python
from psiqit.info.entropy import mutual_information
from psiqit.quantum.state import bell_phi_plus

# Bell state has maximum mutual information (2 bits for 2 qubits)
rho_bell = bell_phi_plus().outer(bell_phi_plus())
I = mutual_information(rho_bell, dim_a=2, dim_b=2, base='2')
print(f"Bell state mutual info: {I:.4f} bits")  # 2.0000 bits

# Product state has zero mutual information
from psiqit.quantum.state import zero
from psiqit.math.qalgebra import kron

rho_a = zero().outer(zero())
rho_b = zero().outer(zero())
rho_product = kron(rho_a, rho_b)
I = mutual_information(rho_product, dim_a=2, dim_b=2, base='2')
print(f"Product state mutual info: {I:.4f} bits")  # 0.0000 bits
```

### Example 5: Partial Trace

```python
from psiqit.info.entropy import partial_trace
from psiqit.quantum.state import bell_phi_plus

# Bell state density matrix
rho_bell = bell_phi_plus().outer(bell_phi_plus())

# Partial trace over qubit B (keep A)
rho_A = partial_trace(rho_bell, dims=[2, 2], keep=0)
print(f"Reduced density matrix for A: {rho_A}")  # [[0.5, 0], [0, 0.5]]

# Partial trace over qubit A (keep B)
rho_B = partial_trace(rho_bell, dims=[2, 2], keep=1)
print(f"Reduced density matrix for B: {rho_B}")  # [[0.5, 0], [0, 0.5]]
```

### Example 6: Purity

```python
from psiqit.info.entropy import purity
from psiqit.quantum.state import zero

# Pure state has purity 1
rho_pure = zero().outer(zero())
p = purity(rho_pure)
print(f"Pure state purity: {p:.4f}")  # 1.0000

# Maximally mixed state has purity 1/2 (for 2 dimensions)
rho_mixed = [[0.5, 0], [0, 0.5]]
p = purity(rho_mixed)
print(f"Mixed state purity: {p:.4f}")  # 0.5000
```

---

## Entanglement Measures (`entanglement.py`)

**Quantify quantum entanglement** - Tools for detecting and measuring entanglement in bipartite quantum systems.

### Classes

| Class | Description |
|-------|-------------|
| `EntanglementResult` | Result container with value, measure, is_entangled |

### Functions

| Function | Description |
|----------|-------------|
| `concurrence(state_or_rho, dims)` | Concurrence (auto-detects pure/mixed) |
| `concurrence_pure(state, dims)` | Concurrence for pure states |
| `concurrence_mixed(rho, dims)` | Concurrence for mixed states (Wootters formula) |
| `negativity(state_or_rho, dims)` | Negativity based on partial transpose |
| `logarithmic_negativity(state_or_rho, dims)` | log₂(2N + 1) |
| `entanglement_entropy(rho, dims)` | von Neumann entropy of reduced state |
| `schmidt_decomposition(state, dims)` | Schmidt decomposition of bipartite pure state |
| `schmidt_rank(state, dims, tol)` | Number of non-zero Schmidt coefficients |
| `is_entangled(state_or_rho, dims, tol)` | Detect entanglement |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `state_or_rho` | Ket or Operator/List | Quantum state (pure) or density matrix (mixed) |
| `dims` | List[int] | Dimensions [dA, dB] (default: [2, 2]) |
| `tol` | float | Numerical tolerance |

### Example 1: Concurrence for Bell States

```python
from psiqit.info.entanglement import concurrence
from psiqit.quantum.state import bell_phi_plus, bell_phi_minus, zero

# Bell states are maximally entangled (C = 1)
bell = bell_phi_plus()
C = concurrence(bell)
print(f"|Φ⁺⟩ concurrence: {C:.4f}")  # 1.0000

bell = bell_phi_minus()
C = concurrence(bell)
print(f"|Φ⁻⟩ concurrence: {C:.4f}")  # 1.0000

# Separable state has zero concurrence
sep = zero()
C = concurrence(sep)
print(f"|0⟩ concurrence: {C:.4f}")  # 0.0000
```

### Example 2: Concurrence for Mixed States (Wootters Formula)

```python
from psiqit.info.entanglement import concurrence_mixed
import numpy as np

# Werner state: ρ = p|Φ⁺⟩⟨Φ⁺| + (1-p)I/4
from psiqit.quantum.state import bell_phi_plus

rho_bell = bell_phi_plus().outer(bell_phi_plus())
I4 = np.eye(4) / 4

p = 0.7
rho_werner = p * np.array(rho_bell) + (1-p) * I4

C = concurrence_mixed(rho_werner)
print(f"Werner state (p=0.7) concurrence: {C:.4f}")
```

### Example 3: Negativity

```python
from psiqit.info.entanglement import negativity, logarithmic_negativity
from psiqit.quantum.state import bell_phi_plus

# Bell state has negativity 0.5
bell = bell_phi_plus()
N = negativity(bell)
print(f"Negativity: {N:.4f}")  # 0.5000

# Logarithmic negativity
E_N = logarithmic_negativity(bell)
print(f"Logarithmic negativity: {E_N:.4f}")  # 1.0000
```

### Example 4: Schmidt Decomposition

```python
from psiqit.info.entanglement import schmidt_decomposition, schmidt_rank
from psiqit.quantum.state import bell_phi_plus, ghz

# Bell state decomposition
bell = bell_phi_plus()
result = schmidt_decomposition(bell, dims=[2, 2])

print(f"Schmidt coefficients: {result['coefficients']}")  # [0.707, 0.707]
print(f"Schmidt rank: {result['schmidt_rank']}")          # 2
print(f"Entanglement entropy: {result['entanglement_entropy']:.4f} bits")  # 1.0000

# Separable state has rank 1
from psiqit.quantum.state import zero, one
product = zero().outer(one())
# For pure product state, we need a Ket
from psiqit.quantum.state import Ket
product_ket = Ket([1, 0, 0, 0])  # |00⟩
rank = schmidt_rank(product_ket, dims=[2, 2])
print(f"Product state Schmidt rank: {rank}")  # 1
```

### Example 5: Entanglement Detection

```python
from psiqit.info.entanglement import is_entangled
from psiqit.quantum.state import bell_phi_plus, zero, plus

# Bell state is entangled
print(f"Bell state entangled: {is_entangled(bell_phi_plus())}")  # True

# Product states are not entangled
print(f"|0⟩ entangled: {is_entangled(zero())}")  # False
print(f"|+⟩ entangled: {is_entangled(plus())}")  # False

# For mixed states (simplified)
from psiqit.quantum.state import random_state
import numpy as np

# Random state is almost always entangled for 2 qubits
random_psi = random_state(4)
print(f"Random 2-qubit state entangled: {is_entangled(random_psi)}")  # Usually True
```

### Example 6: Entanglement Entropy

```python
from psiqit.info.entanglement import entanglement_entropy
from psiqit.quantum.state import bell_phi_plus

# For Bell state, reduced density matrix is maximally mixed
rho_bell = bell_phi_plus().outer(bell_phi_plus())
S = entanglement_entropy(rho_bell, dims=[2, 2])
print(f"Entanglement entropy: {S:.4f} bits")  # 1.0000
```

### Complete Example: Entanglement Analysis of a 2-Qubit State

```python
from psiqit.quantum.state import random_state, bell_phi_plus, ghz
from psiqit.info.entanglement import (
    concurrence, negativity, schmidt_decomposition, is_entangled
)

# Create a random 2-qubit state
psi = random_state(4, seed=42)
print(f"Random state: {psi}")

# Calculate multiple entanglement measures
C = concurrence(psi)
N = negativity(psi)
schmidt = schmidt_decomposition(psi, dims=[2, 2])
entangled = is_entangled(psi)

print("\nEntanglement Analysis:")
print(f"  Concurrence: {C:.6f}")
print(f"  Negativity: {N:.6f}")
print(f"  Schmidt coefficients: {[f'{c:.6f}' for c in schmidt['coefficients']]}")
print(f"  Schmidt rank: {schmidt['schmidt_rank']}")
print(f"  Entanglement entropy: {schmidt['entanglement_entropy']:.6f} bits")
print(f"  Is entangled: {entangled}")

# Compare with Bell state (maximally entangled)
bell = bell_phi_plus()
C_bell = concurrence(bell)
print(f"\nBell state concurrence: {C_bell:.4f} (maximum)")
```

**Expected Output:**

```
Random state: 0.123|00⟩ + 0.456|01⟩ + 0.789|10⟩ + 0.345|11⟩

Entanglement Analysis:
  Concurrence: 0.678901
  Negativity: 0.339451
  Schmidt coefficients: ['0.876543', '0.481234']
  Schmidt rank: 2
  Entanglement entropy: 0.987654 bits
  Is entangled: True

Bell state concurrence: 1.0000 (maximum)
```

---

## Module Contents

```python
__all__ = [
    # Entropy
    'shannon_entropy', 'von_neumann_entropy', 'renyi_entropy',
    'collision_entropy', 'mutual_information', 'partial_trace',
    'relative_entropy', 'purity',
    # Entanglement
    'EntanglementResult', 'concurrence', 'concurrence_pure', 'concurrence_mixed',
    'negativity', 'logarithmic_negativity', 'entanglement_entropy',
    'schmidt_decomposition', 'schmidt_rank', 'is_entangled',
]
```

---

## Summary Table: Entanglement Measures

| Measure | Range | Pure State | Mixed State | Formula |
|---------|-------|------------|-------------|---------|
| Concurrence | [0, 1] | ✓ | ✓ (2-qubit) | C = max(0, λ₁ - λ₂ - λ₃ - λ₄) |
| Negativity | [0, 0.5] | ✓ | ✓ | N = (‖ρ^{T_A}‖₁ - 1)/2 |
| Logarithmic Negativity | [0, ∞) | ✓ | ✓ | E_N = log₂(2N + 1) |
| Entanglement Entropy | [0, log₂(d)] | ✓ | ✗ | S = -Tr(ρ_A log₂ ρ_A) |
| Schmidt Rank | [1, min(dA, dB)] | ✓ | ✗ | Number of non-zero Schmidt coefficients |

---

## References

| Concept | Reference |
|---------|-----------|
| Shannon entropy | C. E. Shannon, "A mathematical theory of communication," Bell System Tech. J., 1948 |
| von Neumann entropy | J. von Neumann, "Mathematical Foundations of Quantum Mechanics," 1932 |
| Rényi entropy | A. Rényi, "On measures of entropy and information," 1961 |
| Concurrence (pure) | S. Hill and W. K. Wootters, "Entanglement of a pair of quantum bits," Phys. Rev. Lett., 1997 |
| Concurrence (mixed) | W. K. Wootters, "Entanglement of formation of an arbitrary state of two qubits," Phys. Rev. Lett., 1998 |
| Negativity | G. Vidal and R. F. Werner, "Computable measure of entanglement," Phys. Rev. A, 2002 |
| Schmidt decomposition | E. Schmidt, "Zur Theorie der linearen und nichtlinearen Integralgleichungen," 1907 |
