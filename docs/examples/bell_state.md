
# Bell State Example

## Overview

Bell states are maximally entangled quantum states of two qubits. They form a basis for the two-qubit Hilbert space and are fundamental to quantum communication protocols like quantum teleportation and superdense coding.

The four Bell states are:

| State | Formula |
|-------|---------|
| \|Φ⁺⟩ | (\|00⟩ + \|11⟩)/√2 |
| \|Φ⁻⟩ | (\|00⟩ - \|11⟩)/√2 |
| \|Ψ⁺⟩ | (\|01⟩ + \|10⟩)/√2 |
| \|Ψ⁻⟩ | (\|01⟩ - \|10⟩)/√2 |

---

## Creating Bell States

### Method 1: Using Built-in Functions

```python
from psiqit.quantum.state import (
    bell_phi_plus, bell_phi_minus,
    bell_psi_plus, bell_psi_minus
)

# Create all four Bell states
phi_plus = bell_phi_plus()   # (|00⟩ + |11⟩)/√2
phi_minus = bell_phi_minus() # (|00⟩ - |11⟩)/√2
psi_plus = bell_psi_plus()   # (|01⟩ + |10⟩)/√2
psi_minus = bell_psi_minus() # (|01⟩ - |10⟩)/√2

print(f"|Φ⁺⟩ = {phi_plus}")
print(f"|Φ⁻⟩ = {phi_minus}")
print(f"|Ψ⁺⟩ = {psi_plus}")
print(f"|Ψ⁻⟩ = {psi_minus}")
```

**Output:**
```
|Φ⁺⟩ = 0.707|00⟩ + 0.707|11⟩
|Φ⁻⟩ = 0.707|00⟩ - 0.707|11⟩
|Ψ⁺⟩ = 0.707|01⟩ + 0.707|10⟩
|Ψ⁻⟩ = 0.707|01⟩ - 0.707|10⟩
```

### Method 2: Using Generic Bell State Function

```python
from psiqit.quantum.state import bell_state

# Index 0 = |Φ⁺⟩, 1 = |Φ⁻⟩, 2 = |Ψ⁺⟩, 3 = |Ψ⁻⟩
for i in range(4):
    bell = bell_state(i)
    print(f"Bell state {i}: {bell}")
```

### Method 3: Building from Quantum Circuit

```python
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.visualization import draw_circuit

# Create Bell state using quantum circuit
circ = QuantumCircuit(2)
circ.h(0)      # Hadamard on first qubit
circ.cx(0, 1)  # CNOT with control=0, target=1

print("Circuit to create |Φ⁺⟩:")
print(draw_circuit(circ, style='unicode'))

# Run circuit to get state
state = circ.run()
print(f"\nResulting state: {state}")
```

**Output:**
```
Circuit to create |Φ⁺⟩:
q0: ─[H]─●─
q1: ─────⊕─

Resulting state: 0.707|00⟩ + 0.707|11⟩
```

---

## Measuring Bell States

### Measurement Statistics

```python
from psiqit.quantum.state import bell_phi_plus

# Create Bell state
bell = bell_phi_plus()

# Measure 1000 times
results = bell.measure(shots=1000)

print(f"Measurement results: {results['counts']}")
# Expected: {'00': ~500, '11': ~500}
```

**Output:**
```
Measurement results: {'00': 503, '11': 497}
```

### Probability Distribution

```python
from psiqit.quantum.state import bell_phi_plus

bell = bell_phi_plus()

# Calculate probabilities for each basis state
for basis_state in range(4):
    prob = bell.prob(basis_state)
    print(f"P(|{basis_state:02b}⟩) = {prob:.4f}")
```

**Output:**
```
P(|00⟩) = 0.5000
P(|01⟩) = 0.0000
P(|10⟩) = 0.0000
P(|11⟩) = 0.5000
```

---

## Entanglement Verification

### Concurrence

```python
from psiqit.quantum.state import bell_phi_plus, bell_psi_minus
from psiqit.info.entanglement import concurrence

# Bell states are maximally entangled (concurrence = 1)
phi_plus = bell_phi_plus()
psi_minus = bell_psi_minus()

print(f"Concurrence of |Φ⁺⟩: {concurrence(phi_plus):.4f}")
print(f"Concurrence of |Ψ⁻⟩: {concurrence(psi_minus):.4f}")

# Compare with product state (concurrence = 0)
from psiqit.quantum.state import zero
product = zero()
print(f"Concurrence of product state: {concurrence(product):.4f}")
```

**Output:**
```
Concurrence of |Φ⁺⟩: 1.0000
Concurrence of |Ψ⁻⟩: 1.0000
Concurrence of product state: 0.0000
```

### Negativity

```python
from psiqit.quantum.state import bell_phi_plus
from psiqit.info.entanglement import negativity, logarithmic_negativity

bell = bell_phi_plus()

N = negativity(bell)
E_N = logarithmic_negativity(bell)

print(f"Negativity: {N:.4f}")
print(f"Logarithmic negativity: {E_N:.4f}")
```

**Output:**
```
Negativity: 0.5000
Logarithmic negativity: 1.0000
```

### Schmidt Decomposition

```python
from psiqit.quantum.state import bell_phi_plus
from psiqit.info.entanglement import schmidt_decomposition

bell = bell_phi_plus()
result = schmidt_decomposition(bell, dims=[2, 2])

print(f"Schmidt coefficients: {result['coefficients']}")
print(f"Schmidt rank: {result['schmidt_rank']}")
print(f"Entanglement entropy: {result['entanglement_entropy']:.4f} bits")
```

**Output:**
```
Schmidt coefficients: [0.70710678, 0.70710678]
Schmidt rank: 2
Entanglement entropy: 1.0000 bits
```

---

## Visualizing Bell States

### On the Bloch Sphere (Reduced States)

```python
from psiqit.quantum.state import bell_phi_plus
from psiqit.utils.conversion import ket_to_density
from psiqit.info.entropy import partial_trace
from psiqit.visualization import bloch_sphere

# Create Bell state
bell = bell_phi_plus()

# Convert to density matrix
rho = ket_to_density(bell)

# Partial trace over qubit B (keep qubit A)
rho_A = partial_trace(rho, dims=[2, 2], keep=0)

print(f"Reduced density matrix for qubit A: {rho_A}")
# [[0.5, 0], [0, 0.5]] - maximally mixed state

# The reduced state is maximally mixed (center of Bloch sphere)
# This indicates maximal entanglement
```

### Circuit Diagram

```python
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.visualization import draw_circuit, circuit_to_text

# Create Bell state circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Different visualization styles
print("Unicode style:")
print(draw_circuit(circ, style='unicode'))

print("\nASCII style:")
print(draw_circuit(circ, style='ascii'))

print("\nText representation:")
print(circuit_to_text(circ))
```

**Output:**
```
Unicode style:
q0: ─[H]─●─
q1: ─────⊕─

ASCII style:
q0: ---[H]---o---
q1: ---------X---

Text representation:
Quantum Circuit: 2 qubits, depth 2
--------------------------------------------------
  1. H        on q0
  2. CNOT     on q0, q1
```

---

## Quantum Teleportation with Bell States

```python
from psiqit.quantum.state import bell_phi_plus, plus
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.info.entanglement import fidelity

# State to teleport (|+⟩)
state_to_teleport = plus()
print(f"State to teleport: {state_to_teleport}")

# Teleportation circuit
circ = QuantumCircuit(3)

# Create Bell pair between qubits 1 and 2
circ.h(1)
circ.cx(1, 2)

# Prepare state on qubit 0 (simplified)
circ.h(0)

# Bell measurement
circ.cx(0, 1)
circ.h(0)

# Run circuit
final_state = circ.run()

# In a full teleportation protocol, we would apply corrections
# based on measurement outcomes
print(f"Final state (before correction): {final_state}")
```

---

## Applications of Bell States

### 1. Superdense Coding

```python
from psiqit.quantum.parties import SuperdenseCoding

# Encode 2 classical bits into 1 qubit
sd = SuperdenseCoding()

# Send bits (0, 1)
result = sd.run(bits=(0, 1))
print(f"Decoded bits: {result}")
```

### 2. Quantum Key Distribution (BB84)

```python
from psiqit.quantum.parties import BB84

# Run BB84 protocol with Bell states
bb84 = BB84(eavesdropping=False)
key = bb84.run(n_bits=100)
print(f"Generated key length: {len(key)} bits")
```

---

## Complete Example: Bell State Analysis

```python
from psiqit.quantum.state import bell_phi_plus, bell_psi_minus
from psiqit.info.entanglement import concurrence, negativity, schmidt_decomposition
from psiqit.info.entropy import von_neumann_entropy, mutual_information
from psiqit.utils.conversion import ket_to_density
from psiqit.visualization import plot_multiple_states

# Analyze two different Bell states
states = [
    ("|Φ⁺⟩", bell_phi_plus()),
    ("|Ψ⁻⟩", bell_psi_minus())
]

print("=" * 60)
print("Bell State Analysis")
print("=" * 60)

for name, state in states:
    print(f"\n{name}: {state}")
    print("-" * 40)
    
    # Concurrence
    C = concurrence(state)
    print(f"  Concurrence: {C:.4f}")
    
    # Negativity
    N = negativity(state)
    print(f"  Negativity: {N:.4f}")
    
    # Schmidt decomposition
    schmidt = schmidt_decomposition(state, dims=[2, 2])
    print(f"  Schmidt coefficients: {[f'{c:.4f}' for c in schmidt['coefficients']]}")
    print(f"  Schmidt rank: {schmidt['schmidt_rank']}")
    
    # Mutual information
    rho = ket_to_density(state)
    I = mutual_information(rho, dim_a=2, dim_b=2, base='2')
    print(f"  Mutual information: {I:.4f} bits")
    
    # Check entanglement
    from psiqit.info.entanglement import is_entangled
    print(f"  Is entangled: {is_entangled(state)}")
```

**Expected Output:**
```
============================================================
Bell State Analysis
============================================================

|Φ⁺⟩: 0.707|00⟩ + 0.707|11⟩
----------------------------------------
  Concurrence: 1.0000
  Negativity: 0.5000
  Schmidt coefficients: ['0.7071', '0.7071']
  Schmidt rank: 2
  Mutual information: 2.0000 bits
  Is entangled: True

|Ψ⁻⟩: 0.707|01⟩ - 0.707|10⟩
----------------------------------------
  Concurrence: 1.0000
  Negativity: 0.5000
  Schmidt coefficients: ['0.7071', '0.7071']
  Schmidt rank: 2
  Mutual information: 2.0000 bits
  Is entangled: True
```

---

## References

| Concept | Reference |
|---------|-----------|
| Bell states | J. S. Bell, "On the Einstein Podolsky Rosen paradox," Physics, 1(3):195-200, 1964 |
| Quantum teleportation | C. H. Bennett et al., "Teleporting an unknown quantum state," Phys. Rev. Lett., 70(13):1895-1899, 1993 |
| Superdense coding | C. H. Bennett and S. J. Wiesner, "Communication via one- and two-particle operators," Phys. Rev. Lett., 69(20):2881-2884, 1992 |
| Concurrence | W. K. Wootters, "Entanglement of formation of an arbitrary state of two qubits," Phys. Rev. Lett., 80(10):2245-2248, 1998 |
