
# Circuits API

## Module: `psiqit.circuits`

This module provides classes for building, simulating, and visualizing quantum circuits. It supports all common quantum gates and allows both state vector simulation and measurement sampling.

---

## QuantumCircuit

**Main class for building and simulating quantum circuits** - Create circuits by adding gates, then run simulation to obtain the final state vector or measurement results.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_qubits` | `int` | Number of qubits in the circuit |
| `depth` | `int` | Number of gates in the circuit |

### Single-Qubit Gates

| Method | Description | Example |
|--------|-------------|---------|
| `x(qubit)` | Pauli X (NOT) gate | `circ.x(0)` |
| `y(qubit)` | Pauli Y gate | `circ.y(0)` |
| `z(qubit)` | Pauli Z gate (phase flip) | `circ.z(0)` |
| `h(qubit)` | Hadamard gate (creates superposition) | `circ.h(0)` |
| `s(qubit)` | S gate (π/2 phase) | `circ.s(0)` |
| `t(qubit)` | T gate (π/4 phase) | `circ.t(0)` |
| `rx(qubit, theta)` | Rotation around X-axis | `circ.rx(0, math.pi/2)` |
| `ry(qubit, theta)` | Rotation around Y-axis | `circ.ry(0, math.pi/2)` |
| `rz(qubit, theta)` | Rotation around Z-axis | `circ.rz(0, math.pi/2)` |

### Multi-Qubit Gates

| Method | Description | Example |
|--------|-------------|---------|
| `cx(control, target)` | CNOT gate | `circ.cx(0, 1)` |
| `cz(control, target)` | Controlled-Z gate | `circ.cz(0, 1)` |
| `swap(qubit1, qubit2)` | SWAP gate | `circ.swap(0, 1)` |
| `toffoli(control1, control2, target)` | Toffoli (CCNOT) gate | `circ.toffoli(0, 1, 2)` |

### Execution Methods

| Method | Description |
|--------|-------------|
| `run()` | Execute circuit and return final state vector (Ket) |
| `measure(qubit=None, shots=1)` | Measure qubit(s) and return outcomes |
| `draw()` | Return ASCII/Unicode circuit diagram |
| `reset()` | Clear all gates and reset to \|0...0⟩ |

### Example 1: Bell State Creation

```python
from psiqit.circuits.circuit import QuantumCircuit

# Create a 2-qubit circuit
circ = QuantumCircuit(2)

# Add gates to create Bell state
circ.h(0)      # Hadamard on qubit 0
circ.cx(0, 1)  # CNOT with control=0, target=1

# Run simulation
state = circ.run()
print(state)  # 0.707|00⟩ + 0.707|11⟩

# Measure
results = circ.measure(shots=1024)
print(results['counts'])  # {'00': 512, '11': 512}
```

### Example 2: Creating a GHZ State

```python
from psiqit.circuits.circuit import QuantumCircuit

# Create a 3-qubit GHZ state (|000⟩ + |111⟩)/√2
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)

state = circ.run()
print(state)  # 0.707|000⟩ + 0.707|111⟩

# Draw the circuit
print(circ.draw())
```

### Example 3: Quantum Teleportation Circuit

```python
from psiqit.circuits.circuit import QuantumCircuit

# Teleportation circuit (simplified)
# Qubit 0: state to teleport (|ψ⟩)
# Qubit 1: Alice's half of Bell pair
# Qubit 2: Bob's half of Bell pair

circ = QuantumCircuit(3)

# Create Bell pair between qubits 1 and 2
circ.h(1)
circ.cx(1, 2)

# Prepare some state on qubit 0
circ.h(0)   # For example, |+⟩ state

# Bell measurement on qubits 0 and 1
circ.cx(0, 1)
circ.h(0)

# Measure qubits 0 and 1
results = circ.measure(qubit=[0, 1], shots=1)
print(f"Bell measurement results: {results}")
```

### Example 4: Grover's Algorithm Oracle

```python
from psiqit.circuits.circuit import QuantumCircuit

# Oracle for marking target state |11⟩ in 2-qubit system
def grover_oracle_11():
    circ = QuantumCircuit(2)
    circ.cz(0, 1)  # Phase flip when both qubits are |1⟩
    return circ

# Create full Grover circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.h(1)          # Superposition
circ.cz(0, 1)      # Oracle
circ.h(0)
circ.h(1)
circ.x(0)
circ.x(1)
circ.cz(0, 1)      # Diffusion operator
circ.x(0)
circ.x(1)
circ.h(0)
circ.h(1)

state = circ.run()
print(f"Final state: {state}")

results = circ.measure(shots=1024)
print(f"Measurement results: {results['counts']}")  # Should be biased toward |11⟩
```

### Example 5: Circuit with Parameterized Rotations

```python
from psiqit.circuits.circuit import QuantumCircuit
import math

# Create a circuit with arbitrary rotations
circ = QuantumCircuit(1)

# Apply rotations around X, Y, and Z axes
theta = math.pi / 3
circ.rx(0, theta)
circ.ry(0, theta)
circ.rz(0, theta)

# Run the circuit
state = circ.run()
print(f"State after rotations: {state}")

# The state will be a specific point on the Bloch sphere
```

### Example 6: Multi-Qubit Gate Operations

```python
from psiqit.circuits.circuit import QuantumCircuit

# Create a 3-qubit circuit to demonstrate various multi-qubit gates
circ = QuantumCircuit(3)

# CNOT: flip target if control is |1⟩
circ.cx(0, 1)

# CZ: apply Z gate to target if control is |1⟩
circ.cz(0, 2)

# SWAP: exchange states of two qubits
circ.swap(1, 2)

# Toffoli (CCNOT): flip target if both controls are |1⟩
circ.toffoli(0, 1, 2)

# Run simulation
state = circ.run()
print(f"Final state: {state}")
```

### Example 7: Measurement and Statistics

```python
from psiqit.circuits.circuit import QuantumCircuit
import numpy as np

# Create a circuit with superposition and measurement
circ = QuantumCircuit(2)
circ.h(0)      # Create superposition on qubit 0
circ.cx(0, 1)  # Entangle with qubit 1

# Run many shots to get statistics
results = circ.measure(shots=1000)

# Print statistics
counts = results['counts']
total = sum(counts.values())

print("Measurement statistics:")
for outcome, count in sorted(counts.items()):
    percentage = (count / total) * 100
    print(f"  |{outcome}⟩: {count} ({percentage:.1f}%)")

# Access the state vector as well
state = circ.run()
print(f"\nState vector: {state}")
```

### Example 8: Resetting and Reusing Circuits

```python
from psiqit.circuits.circuit import QuantumCircuit

circ = QuantumCircuit(2)

# First computation
circ.h(0)
circ.cx(0, 1)
state1 = circ.run()
print(f"First state: {state1}")

# Reset the circuit
circ.reset()

# Second, different computation
circ.x(0)
circ.h(1)
state2 = circ.run()
print(f"Second state: {state2}")
```

### Example 9: Deep Circuit with Many Gates

```python
from psiqit.circuits.circuit import QuantumCircuit

# Create a deeper circuit to explore
circ = QuantumCircuit(3)

# Layer 1: Initial superposition
for i in range(3):
    circ.h(i)

# Layer 2: Entangling operations
circ.cz(0, 1)
circ.cz(1, 2)

# Layer 3: Single-qubit rotations
import math
circ.rx(0, math.pi/4)
circ.ry(1, math.pi/3)
circ.rz(2, math.pi/6)

# Layer 4: More entangling
circ.toffoli(0, 1, 2)
circ.swap(0, 2)

# Execute
state = circ.run()
print(f"Circuit depth: {circ.depth}")
print(f"Final state dimension: {state.dim}")

# Display circuit diagram
print("\nCircuit diagram:")
print(circ.draw())
```

## Circuit Visualization

The `draw()` method provides ASCII/Unicode circuit diagrams for debugging and documentation.

### Example: Visualizing a Circuit

```python
from psiqit.circuits.circuit import QuantumCircuit

circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)
circ.x(1)
circ.rz(2, 3.14159/2)

print(circ.draw())
```

**Output (example diagram):**
```
q0: ──H──●──●──────────
q1: ─────X──────X───────
q2: ───────────X───Rz───
```

## Module Contents

```python
__all__ = [
    'QuantumCircuit',
]
```

## References

| Concept | Reference |
|---------|-----------|
| Quantum circuits | M. Nielsen and I. Chuang, "Quantum Computation and Quantum Information," 2010 |
| Bell state | J. S. Bell, "On the Einstein Podolsky Rosen paradox," Physics, 1964 |
| GHZ state | D. M. Greenberger, M. A. Horne, A. Zeilinger, "Going beyond Bell's theorem," 1989 |
| Quantum teleportation | C. H. Bennett et al., "Teleporting an unknown quantum state," Phys. Rev. Lett., 1993 |
| Grover's algorithm | L. K. Grover, "A fast quantum mechanical algorithm for database search," Phys. Rev. Lett., 1996 |
