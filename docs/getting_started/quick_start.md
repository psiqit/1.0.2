# Quick Start Guide

This guide will help you get started with PSIQIT through practical examples.

## Table of Contents

- [Quantum States](#quantum-states)
- [Quantum Gates](#quantum-gates)
- [Quantum Circuits](#quantum-circuits)
- [Measurements](#measurements)
- [Quantum Algorithms](#quantum-algorithms)
- [Variational Methods](#variational-methods)
- [Entanglement Measures](#entanglement-measures)
- [Visualization](#visualization)
- [Quantum Communication](#quantum-communication)
- [Next Steps](#next-steps)

---

## Quantum States

PSIQIT provides various quantum states through the `psiqit.quantum.state` module.

### Basic States

```python
from psiqit.quantum.state import zero, one, plus, minus, ip, im

print(zero())  # 1.000|0⟩
print(one())   # 1.000|1⟩
print(plus())  # 0.707|0⟩ + 0.707|1⟩
Bell States (Entangled States)
python
from psiqit.quantum.state import bell_phi_plus, bell_phi_minus
from psiqit.quantum.state import bell_psi_plus, bell_psi_minus

# |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
bell = bell_phi_plus()
print(bell)  # 0.707|00⟩ + 0.707|11⟩
Multi-Qubit States
python
from psiqit.quantum.state import ghz, w_state

# GHZ state: (|000⟩ + |111⟩)/√2
ghz_state = ghz(3)
print(ghz_state)

# W state: (|100⟩ + |010⟩ + |001⟩)/√3
w = w_state(3)
print(w)
Quantum Gates
Gates are available in psiqit.quantum.operator.

Single-Qubit Gates
python
from psiqit.quantum.operator import pauli_x, pauli_y, pauli_z, hadamard
from psiqit.quantum.state import zero

# Apply X gate (NOT) to |0⟩ -> |1⟩
X = pauli_x()
result = X @ zero()
print(result)  # 1.000|1⟩

# Apply Hadamard gate to |0⟩ -> |+⟩
H = hadamard()
result = H @ zero()
print(result)  # 0.707|0⟩ + 0.707|1⟩
Rotation Gates
python
from psiqit.quantum.operator import rx, ry, rz
import math

# Rotation around X-axis by π/2
Rx = rx(math.pi/2)
result = Rx @ zero()
print(result)
Two-Qubit Gates
python
from psiqit.quantum.operator import cnot, swap

# CNOT gate
CNOT = cnot()
# SWAP gate
SWAP = swap()
Quantum Circuits
Build and simulate quantum circuits using psiqit.circuits.circuit.

Creating a Circuit
python
from psiqit.circuits.circuit import QuantumCircuit

# Create a 2-qubit circuit
circ = QuantumCircuit(2)

# Add gates
circ.h(0)      # Hadamard on qubit 0
circ.cx(0, 1)  # CNOT with control=0, target=1

# Run the circuit
state = circ.run()
print(state)  # 0.707|00⟩ + 0.707|11⟩ (Bell state)
Circuit Information
python
print(f"Number of qubits: {circ.n_qubits}")
print(f"Circuit depth: {circ.depth}")
print(circ.draw())  # ASCII circuit diagram
Measurements
Projective Measurement
python
from psiqit.quantum.measurement import measure
from psiqit.quantum.state import plus

psi = plus()
result = measure(psi, shots=1000)
print(result['counts'])        # {0: 498, 1: 502}
print(result['probabilities']) # [0.5, 0.5]
Expectation Values
python
from psiqit.quantum.measurement import expectation
from psiqit.quantum.operator import pauli_z
from psiqit.quantum.state import zero

exp_val = expectation(pauli_z(), zero())
print(exp_val)  # 1.0
POVM Measurements
python
from psiqit.quantum.measurement import povm_z_basis
from psiqit.quantum.state import zero

povm = povm_z_basis()
result = povm.measure(zero(), shots=1000)
print(result['counts'])  # {'0': 1000, '1': 0}
Quantum Algorithms
Grover's Search Algorithm
python
from psiqit.algorithms.grover import grover_search

# Search for state 5 in a 3-qubit system
result = grover_search(n_qubits=3, target=5, shots=1024)
print(f"Most likely state: {result.most_likely}")  # 5
print(f"Success: {result.success}")               # True
Deutsch-Jozsa Algorithm
python
from psiqit.algorithms.deutsch_jozsa import deutsch_jozsa_constant, deutsch_jozsa_balanced

# Test a constant function
result = deutsch_jozsa_constant(n_qubits=3, constant_value=0)
print(result)  # Function is CONSTANT

# Test a balanced function
result = deutsch_jozsa_balanced(n_qubits=3)
print(result)  # Function is BALANCED
Shor's Factoring Algorithm
python
from psiqit.algorithms.shor import Shor

shor = Shor()
result = shor.factor(15)
print(result)  # 15 = 3 × 5
Variational Methods
VQE (Variational Quantum Eigensolver)
python
from psiqit.variational.vqe import VQE

# Hamiltonian: H = Z₀ + Z₁ + 0.5·Z₀Z₁
hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}

vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
result = vqe.run(n_iterations=100, learning_rate=0.1)
print(f"Ground state energy: {result.optimal_energy:.6f}")
QAOA (Quantum Approximate Optimization Algorithm)
python
from psiqit.variational.qaoa import QAOA, maxcut_hamiltonian

# Max-Cut on a triangle graph
edges = [(0, 1), (1, 2), (2, 0)]
H = maxcut_hamiltonian(edges)

qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
result = qaoa.run(n_iterations=100)
print(f"Max-Cut energy: {result.optimal_energy:.6f}")
Entanglement Measures
python
from psiqit.quantum.state import bell_phi_plus
from psiqit.info.entanglement import concurrence, negativity, schmidt_decomposition

bell = bell_phi_plus()

# Concurrence (Wootters formula)
C = concurrence(bell)
print(f"Concurrence: {C:.4f}")  # 1.0000

# Negativity (based on partial transpose)
N = negativity(bell)
print(f"Negativity: {N:.4f}")   # 0.5000

# Schmidt decomposition
schmidt = schmidt_decomposition(bell, dims=[2, 2])
print(f"Schmidt coefficients: {schmidt['coefficients']}")
print(f"Schmidt rank: {schmidt['schmidt_rank']}")  # 2
Visualization
Bloch Sphere
python
from psiqit.quantum.state import plus, ip, zero
from psiqit.visualization.bloch import bloch_sphere, plot_multiple_states

# Single state
bloch_sphere(plus(), title="|+⟩ State")

# Multiple states
plot_multiple_states([zero(), plus(), ip()],
                      colors=['blue', 'red', 'green'])
Wigner Function
python
from psiqit.visualization.wigner import wigner_function_coherent_state, plot_wigner

W, x, p = wigner_function_coherent_state(alpha=1.0)
plot_wigner(W, x, p, title="Coherent State |α=1⟩")
Circuit Drawing
python
from psiqit.visualization.circuit_drawer import draw_circuit

circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
print(draw_circuit(circ, style='unicode'))
Quantum Communication
BB84 Quantum Key Distribution
python
from psiqit.quantum.parties import BB84

bb84 = BB84(eavesdropping=False)
key = bb84.run(n_bits=100)
print(f"Shared key length: {len(key)}")
Quantum Teleportation
python
from psiqit.quantum.parties import QuantumTeleportation
from psiqit.quantum.state import plus

teleport = QuantumTeleportation()
state = plus()
teleported = teleport.run(state)
print(f"Original: {state}")
print(f"Teleported: {teleported}")
Next Steps
Explore the User Guide for detailed tutorials

Browse the API Reference for complete documentation

Check out Examples for more code samples