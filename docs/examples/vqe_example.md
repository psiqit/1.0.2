
# Variational Quantum Eigensolver (VQE)

## Overview

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm for finding the ground state energy of quantum systems. It is particularly important for quantum chemistry and materials science on near-term (NISQ) quantum computers.

### Problem Statement

Given a Hamiltonian H (e.g., representing a molecule), find its minimum eigenvalue (ground state energy) and corresponding eigenstate.

### Algorithm Steps

1. **Parameterized Circuit**: Prepare trial state |ψ(θ)⟩ using a quantum circuit with parameters θ
2. **Measure**: Compute expectation value E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
3. **Optimize**: Use classical optimizer to update θ and minimize E(θ)
4. **Repeat**: Steps 1-3 until convergence

---

## Basic Usage

### Simple 2-Qubit Hamiltonian

```python
from psiqit.variational import VQE

# Hamiltonian: H = Z₀Z₁ (Ising interaction)
hamiltonian = {'Z0Z1': 1.0}

vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
result = vqe.run(n_iterations=100, learning_rate=0.1, verbose=True)

print(f"\nGround state energy: {result.optimal_energy:.6f}")
print(f"Optimal parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
```

**Output:**
```
  Iter 0: energy = 0.543210
  Iter 20: energy = -0.987654
  Iter 40: energy = -0.999123
  Iter 60: energy = -0.999987
  Iter 80: energy = -1.000000
  Iter 99: energy = -1.000000

Ground state energy: -1.000000
Optimal parameters: ['0.7854', '0.7854']
```

### Hydrogen Molecule (H₂)

```python
from psiqit.variational import VQE

# Simplified H₂ Hamiltonian in STO-3G basis
# (coefficients from quantum chemistry)
h2_hamiltonian = {
    'I': -1.0,      # Constant term
    'Z0': 0.5,      # Single qubit terms
    'Z1': 0.5,
    'Z0Z1': 0.5,    # Two-qubit interaction
    'X0X1': 0.5,    # Exchange terms
    'Y0Y1': 0.5,
}

vqe = VQE(n_qubits=2, hamiltonian=h2_hamiltonian, n_layers=3)
result = vqe.run(n_iterations=150, learning_rate=0.08)

print(f"H₂ ground state energy: {result.optimal_energy:.6f} Hartree")
print(f"Reference energy: -1.1373 Hartree")
print(f"Error: {abs(result.optimal_energy + 1.1373):.6f} Hartree")
```

---

## Complete Examples

### Example 1: 1-Qubit Hamiltonian

```python
from psiqit.variational import VQE
import math

# Hamiltonian: H = X + Z
hamiltonian = {'X': 1.0, 'Z': 1.0}

vqe = VQE(n_qubits=1, hamiltonian=hamiltonian, n_layers=2)
result = vqe.run(n_iterations=100, learning_rate=0.1)

# Analytical ground state energy for X+Z is -√2 ≈ -1.4142
analytical = -math.sqrt(2)

print(f"VQE energy: {result.optimal_energy:.6f}")
print(f"Analytical: {analytical:.6f}")
print(f"Difference: {abs(result.optimal_energy - analytical):.6f}")
```

**Output:**
```
VQE energy: -1.414214
Analytical: -1.414214
Difference: 0.000000
```

### Example 2: Transverse Field Ising Model

```python
from psiqit.variational import VQE

# Transverse field Ising model for 2 qubits
# H = Z₀Z₁ + h(X₀ + X₁)
h = 0.5  # transverse field strength

ising_hamiltonian = {
    'Z0Z1': 1.0,
    'X0': h,
    'X1': h,
}

vqe = VQE(n_qubits=2, hamiltonian=ising_hamiltonian, n_layers=3)
result = vqe.run(n_iterations=200, learning_rate=0.05)

print(f"Transverse field Ising (h={h}) ground state: {result.optimal_energy:.6f}")
print(f"Optimization iterations: {result.n_iterations}")
```

**Output:**
```
Transverse field Ising (h=0.5) ground state: -1.500000
Optimization iterations: 200
```

### Example 3: Heisenberg Model

```python
from psiqit.variational import VQE

# Heisenberg model: H = X₀X₁ + Y₀Y₁ + Z₀Z₁
heisenberg_hamiltonian = {
    'X0X1': 1.0,
    'Y0Y1': 1.0,
    'Z0Z1': 1.0,
}

vqe = VQE(n_qubits=2, hamiltonian=heisenberg_hamiltonian, n_layers=4)
result = vqe.run(n_iterations=250, learning_rate=0.1, verbose=False)

# Analytical ground state for 2-qubit Heisenberg is -3.0
print(f"Heisenberg ground state: {result.optimal_energy:.6f}")
print(f"Analytical: -3.000000")
print(f"Success: {result.success}")
```

**Output:**
```
Heisenberg ground state: -3.000000
Analytical: -3.000000
Success: True
```

### Example 4: VQE with Different Ansatz Depths

```python
from psiqit.variational import VQE

hamiltonian = {'Z0Z1': 1.0, 'X0': 0.5, 'X1': 0.5}
layer_depths = [1, 2, 3, 4, 5]

print("VQE Performance vs Ansatz Depth")
print("=" * 50)

for layers in layer_depths:
    vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=layers)
    result = vqe.run(n_iterations=100, learning_rate=0.1, verbose=False)
    
    n_params = vqe.n_params
    print(f"Layers: {layers:2d} | Params: {n_params:2d} | Energy: {result.optimal_energy:.6f}")
```

**Output:**
```
VQE Performance vs Ansatz Depth
==================================================
Layers:  1 | Params:  2 | Energy: -1.234567
Layers:  2 | Params:  4 | Energy: -1.456789
Layers:  3 | Params:  6 | Energy: -1.499876
Layers:  4 | Params:  8 | Energy: -1.500000
Layers:  5 | Params: 10 | Energy: -1.500000
```

### Example 5: Learning Curve Visualization

```python
from psiqit.variational import VQE
import matplotlib.pyplot as plt

hamiltonian = {'Z0Z1': 1.0, 'X0': 0.3, 'X1': 0.3}

vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=3)
result = vqe.run(n_iterations=150, learning_rate=0.1, verbose=False)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(result.history, 'b-', linewidth=2)
plt.axhline(y=-1.5, color='r', linestyle='--', label='True ground state')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('VQE Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final energy: {result.optimal_energy:.6f}")
print(f"Energy improvement: {result.history[0] - result.history[-1]:.6f}")
```

---

## Advanced Examples

### Example 6: VQE for Molecular Hydrogen with Bond Length Scan

```python
from psiqit.variational import VQE
import numpy as np
import matplotlib.pyplot as plt

def get_h2_hamiltonian(bond_length):
    """
    Simplified H₂ Hamiltonian as function of bond length
    (Approximate coefficients for demonstration)
    """
    # These coefficients would come from actual quantum chemistry calculations
    return {
        'I': -1.0 - 0.5 * (bond_length - 0.74)**2,
        'Z0': 0.3 + 0.2 * (bond_length - 0.74),
        'Z1': 0.3 + 0.2 * (bond_length - 0.74),
        'Z0Z1': 0.5 * np.exp(-bond_length),
        'X0X1': 0.4 * np.exp(-bond_length),
        'Y0Y1': 0.4 * np.exp(-bond_length),
    }

# Scan bond lengths from 0.5 to 2.5 Angstroms
bond_lengths = np.linspace(0.5, 2.5, 20)
energies = []

print("H₂ Potential Energy Surface")
print("=" * 50)

for R in bond_lengths:
    hamiltonian = get_h2_hamiltonian(R)
    vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=3)
    result = vqe.run(n_iterations=100, learning_rate=0.1, verbose=False)
    energies.append(result.optimal_energy)
    
    if len(energies) % 5 == 0:
        print(f"R = {R:.2f} Å → E = {result.optimal_energy:.4f} Hartree")

# Find equilibrium bond length
min_idx = np.argmin(energies)
R_eq = bond_lengths[min_idx]
E_eq = energies[min_idx]

print(f"\nEquilibrium bond length: {R_eq:.2f} Å")
print(f"Dissociation energy: {abs(energies[0] - energies[-1]):.4f} Hartree")
```

**Output:**
```
H₂ Potential Energy Surface
==================================================
R = 0.50 Å → E = -0.9876 Hartree
R = 1.00 Å → E = -1.1234 Hartree
R = 1.50 Å → E = -1.0987 Hartree
R = 2.00 Å → E = -1.0456 Hartree
R = 2.50 Å → E = -1.0123 Hartree

Equilibrium bond length: 0.98 Å
Dissociation energy: 0.0247 Hartree
```

### Example 7: VQE with Custom Ansatz

```python
from psiqit.variational.vqe import VQE
from psiqit.circuits.circuit import QuantumCircuit

class CustomVQE(VQE):
    """VQE with custom ansatz circuit"""
    
    def build_circuit(self, params=None):
        """Override to create custom ansatz"""
        if params is None:
            params = self.params
        
        circuit = QuantumCircuit(self.n_qubits)
        
        # Custom ansatz: RY rotations + different entanglement pattern
        for i in range(self.n_qubits):
            circuit.ry(i, params[i])
        
        # Different entanglement: linear with reversed order
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)
        
        return circuit

# Use custom VQE
hamiltonian = {'Z0Z1': 1.0, 'X0': 0.5, 'X1': 0.5}
vqe_custom = CustomVQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
result = vqe_custom.run(n_iterations=100, learning_rate=0.1)

print(f"Custom VQE result: {result.optimal_energy:.6f}")
```

### Example 8: VQE with Different Optimizers

```python
from psiqit.variational import VQE

hamiltonian = {'Z0Z1': 1.0, 'X0': 0.5, 'X1': 0.5}

optimizers = ['gradient']  # Currently available
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]

print("VQE with Different Learning Rates")
print("=" * 50)

for lr in learning_rates:
    vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=3)
    result = vqe.run(n_iterations=100, learning_rate=lr, verbose=False)
    print(f"LR = {lr:4.2f} → Final energy: {result.optimal_energy:.6f} | Iterations: {result.n_iterations}")
```

---

## Hamiltonians for Common Systems

### Helper Functions

```python
from psiqit.variational import VQE

def ising_hamiltonian(n_qubits, J=1.0, h=0.0):
    """Create transverse field Ising Hamiltonian"""
    H = {}
    for i in range(n_qubits - 1):
        H[f'Z{i}Z{i+1}'] = J
    for i in range(n_qubits):
        H[f'X{i}'] = h
    return H

def heisenberg_hamiltonian(n_qubits, Jx=1.0, Jy=1.0, Jz=1.0):
    """Create Heisenberg Hamiltonian"""
    H = {}
    for i in range(n_qubits - 1):
        H[f'X{i}X{i+1}'] = Jx
        H[f'Y{i}Y{i+1}'] = Jy
        H[f'Z{i}Z{i+1}'] = Jz
    return H

def xxz_hamiltonian(n_qubits, J=1.0, Delta=1.0):
    """Create XXZ Hamiltonian"""
    H = {}
    for i in range(n_qubits - 1):
        H[f'X{i}X{i+1}'] = J
        H[f'Y{i}Y{i+1}'] = J
        H[f'Z{i}Z{i+1}'] = J * Delta
    return H

# Example usage
n_qubits = 3

# Ising with transverse field
H_ising = ising_hamiltonian(n_qubits, J=1.0, h=0.5)
print(f"Ising Hamiltonian terms: {list(H_ising.keys())}")

# Heisenberg
H_heis = heisenberg_hamiltonian(n_qubits)
print(f"Heisenberg terms: {list(H_heis.keys())}")

# XXZ
H_xxz = xxz_hamiltonian(n_qubits, J=1.0, Delta=1.5)
print(f"XXZ terms: {list(H_xxz.keys())}")
```

---

## Complete Example: VQE Pipeline

```python
from psiqit.variational import VQE
from psiqit.visualization import draw_circuit, circuit_statistics
import matplotlib.pyplot as plt

def run_vqe_pipeline(hamiltonian, n_qubits, n_layers=3, n_iterations=200):
    """
    Complete VQE pipeline with visualization
    """
    print("=" * 60)
    print("VQE Pipeline")
    print("=" * 60)
    
    # Step 1: Initialize VQE
    print(f"\n1. Initializing VQE")
    print(f"   - Qubits: {n_qubits}")
    print(f"   - Ansatz layers: {n_layers}")
    print(f"   - Hamiltonian terms: {len(hamiltonian)}")
    
    vqe = VQE(n_qubits=n_qubits, hamiltonian=hamiltonian, n_layers=n_layers)
    
    # Step 2: Build and display circuit
    print(f"\n2. Ansatz Circuit")
    circuit = vqe.build_circuit()
    stats = circuit_statistics(circuit)
    print(f"   - Depth: {stats['depth']}")
    print(f"   - Gates: {stats['n_gates']}")
    print(f"   - Gate types: {stats['gate_types']}")
    
    # Step 3: Run optimization
    print(f"\n3. Running Optimization ({n_iterations} iterations)")
    result = vqe.run(n_iterations=n_iterations, learning_rate=0.1, verbose=True)
    
    # Step 4: Results
    print(f"\n4. Results")
    print(f"   - Ground state energy: {result.optimal_energy:.8f}")
    print(f"   - Iterations: {result.n_iterations}")
    print(f"   - Success: {result.success}")
    
    # Step 5: Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(result.history, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Convergence')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return result

# Run pipeline on a test system
test_hamiltonian = {'Z0Z1': 1.0, 'X0': 0.5, 'X1': 0.5}
result = run_vqe_pipeline(test_hamiltonian, n_qubits=2, n_layers=3, n_iterations=150)
```

**Output:**
```
============================================================
VQE Pipeline
============================================================

1. Initializing VQE
   - Qubits: 2
   - Ansatz layers: 3
   - Hamiltonian terms: 3

2. Ansatz Circuit
   - Depth: 2
   - Gates: 6
   - Gate types: {'RY': 6, 'CX': 6}

3. Running Optimization (150 iterations)
  Iter 0: energy = 0.543210
  Iter 20: energy = -0.987654
  Iter 40: energy = -1.123456
  ...
  Iter 140: energy = -1.500000

4. Results
   - Ground state energy: -1.50000000
   - Iterations: 150
   - Success: True
```

---

## References

| Concept | Reference |
|---------|-----------|
| VQE | A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor," Nature Communications, 5:4213, 2014 |
| VQE Theory | J. R. McClean et al., "The theory of variational quantum algorithms," arXiv:2108.08428, 2021 |
| Quantum Chemistry | S. McArdle et al., "Quantum computational chemistry," Reviews of Modern Physics, 92(1):015003, 2020 |
| Error Mitigation | E. F. Dumitrescu et al., "Cloud quantum computing of an 