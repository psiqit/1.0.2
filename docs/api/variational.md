
# Variational Methods API

## Module: `psiqit.variational`

This module provides variational quantum algorithms for optimization, simulation, and machine learning. Variational quantum algorithms are hybrid quantum-classical algorithms that use a parameterized quantum circuit (ansatz) and a classical optimizer to minimize a cost function. They are particularly important for near-term (NISQ) quantum computers.

**Available methods:**
- **VQE** - Variational Quantum Eigensolver for ground state energies
- **QAOA** - Quantum Approximate Optimization Algorithm for combinatorial optimization
- **SSVQE** - Subspace-Search VQE for excited states
- **ADAPT-VQE** - Adaptive construction of variational ansatz
- **VQD** - Variational Quantum Deflation for excited states
- **Rayleigh-Ritz** - Classical variational method
- **TDVP** - Time-Dependent Variational Principle
- **Variational Monte Carlo** - Stochastic optimization
- **Hartree-Fock** - Electronic structure method

---

## VQEResult

**Result container for VQE optimization.**

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `optimal_params` | `List[float]` | Optimal parameters found |
| `optimal_energy` | `float` | Minimum energy (ground state energy) achieved |
| `n_iterations` | `int` | Number of optimization iterations |
| `history` | `List[float]` | Energy values at each iteration |
| `success` | `bool` | Whether optimization completed successfully |

### Example

```python
from psiqit.variational import VQEResult

result = VQEResult(optimal_params=[0.1, 0.2], optimal_energy=-1.0,
                   n_iterations=100, history=[-0.5, -0.8, -1.0],
                   success=True)
print(result)  # VQE(energy=-1.000000, iterations=100)
```

---

## Variational Quantum Eigensolver (VQE)

**Variational method for finding ground state energy of Hamiltonians** - Uses a parameterized quantum circuit (ansatz) to prepare trial states and a classical optimizer to minimize the energy expectation value.

### Class

| Class | Description |
|-------|-------------|
| `VQE` | Variational Quantum Eigensolver |
| `VQEResult` | Result container |

### Methods

| Method | Description |
|--------|-------------|
| `__init__(n_qubits, hamiltonian, n_layers, optimizer)` | Initialize VQE |
| `build_circuit(params)` | Build variational quantum circuit |
| `evaluate_energy(params)` | Evaluate energy expectation value |
| `run(n_iterations, learning_rate, verbose)` | Run VQE optimization |

### Example 1: Simple 1-Qubit Hamiltonian

```python
from psiqit.variational import VQE

# Hamiltonian: H = X + Z
hamiltonian = {'X': 1.0, 'Z': 1.0}
vqe = VQE(n_qubits=1, hamiltonian=hamiltonian, n_layers=2)
result = vqe.run(n_iterations=100, learning_rate=0.1)

print(f"Ground state energy: {result.optimal_energy:.6f}")
print(f"Optimal parameters: {result.optimal_params}")
```

### Example 2: Heisenberg Model

```python
from psiqit.variational import VQE

# 2-qubit Heisenberg model: H = X₀X₁ + Y₀Y₁ + Z₀Z₁
hamiltonian = {'X0X1': 1.0, 'Y0Y1': 1.0, 'Z0Z1': 1.0}
vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=3)
result = vqe.run(n_iterations=200, learning_rate=0.05)

print(f"Heisenberg ground state: {result.optimal_energy:.6f}")
```

### Example 3: Custom Ansatz Circuit

```python
from psiqit.variational import VQE

hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}
vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)

# Build and inspect the circuit
circuit = vqe.build_circuit()
print(f"Circuit depth: {circuit.depth}")
print(circuit.draw())
```

---

## Quantum Approximate Optimization Algorithm (QAOA)

**QAOA for solving combinatorial optimization problems** - Uses a parameterized quantum circuit that alternates between applying the cost Hamiltonian (encoding the problem) and a mixer Hamiltonian (exploring the solution space).

### Classes

| Class | Description |
|-------|-------------|
| `QAOA` | Quantum Approximate Optimization Algorithm |
| `QAOResult` | Result container |

### Functions

| Function | Description |
|----------|-------------|
| `maxcut_hamiltonian(edges)` | Create Max-Cut Hamiltonian from graph edges |
| `qubo_to_hamiltonian(qubo, n_qubits)` | Convert QUBO problem to Hamiltonian |

### QAOA Circuit Structure

The QAOA circuit for p layers is:
```
|ψ(γ,β)⟩ = e^{-iβ_p H_M} e^{-iγ_p H_C} ... e^{-iβ_1 H_M} e^{-iγ_1 H_C} |+⟩^{⊗n}
```

- **γ (gamma)** : angle for the cost Hamiltonian evolution
- **β (beta)** : angle for the mixer Hamiltonian evolution

### Example 1: Max-Cut on a Triangle Graph

```python
from psiqit.variational import QAOA, maxcut_hamiltonian

# Triangle graph edges
edges = [(0, 1), (1, 2), (2, 0)]
H = maxcut_hamiltonian(edges)

qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
result = qaoa.run(n_iterations=100, learning_rate=0.1)

print(f"Max-Cut energy: {result.optimal_energy:.4f}")
print(f"Optimal parameters: {result.optimal_params}")
```

### Example 2: QUBO Problem

```python
from psiqit.variational import QAOA, qubo_to_hamiltonian

# Minimize x₀ + x₁ - 2x₀x₁
qubo = {(0, 0): 1, (1, 1): 1, (0, 1): -2}
H = qubo_to_hamiltonian(qubo, n_qubits=2)

qaoa = QAOA(n_qubits=2, hamiltonian=H, p=1)
result = qaoa.run(n_iterations=100)

print(f"Optimal energy: {result.optimal_energy:.6f}")
```

### Example 3: Building QAOA Circuit

```python
from psiqit.variational import QAOA

edges = [(0, 1), (1, 2)]
H = maxcut_hamiltonian(edges)

qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
circuit = qaoa.build_circuit([0.5, 0.3, 0.7, 0.2])

print(f"QAOA circuit depth: {circuit.depth}")
```

---

## Advanced Variational Methods

### SSVQE (Subspace-Search VQE)

**Finds multiple eigenstates (ground and excited) simultaneously** - Uses orthogonal initial states and minimizes a weighted sum of energies.

| Class | Description |
|-------|-------------|
| `SSVQE` | Subspace-Search VQE for excited states |
| `MultiStateResult` | Result container |

```python
from psiqit.variational import SSVQE

# Hamiltonian for finding multiple states
hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}

ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=3)
result = ssvqe.run(n_iterations=200, learning_rate=0.1)

print(f"Energies: {[f'{e:.6f}' for e in result.energies]}")
```

### ADAPT-VQE

**Adaptive construction of variational ansatz** - Builds the ansatz circuit adaptively by adding operators that have the largest gradient contribution.

| Class | Description |
|-------|-------------|
| `ADAPTVQE` | Adaptive VQE with operator pool |

```python
from psiqit.variational import ADAPTVQE

hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}

adapt = ADAPTVQE(n_qubits=2, hamiltonian=hamiltonian)
result = adapt.run(max_iterations=10, grad_threshold=0.01)

print(f"Ground state energy: {result.optimal_energy:.6f}")
print(f"Number of operators: {len(adapt.ansatz_operators)}")
```

### Variational Quantum Deflation (VQD)

**Finds excited states by penalizing overlap with previously found states** - Adds a penalty term to the cost function that penalizes overlap with lower-energy states.

| Class | Description |
|-------|-------------|
| `VariationalQuantumDeflation` | VQD for excited states |

```python
from psiqit.variational import VariationalQuantumDeflation

hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}
vqd = VariationalQuantumDeflation(n_qubits=2, hamiltonian=hamiltonian)

# Find ground state
ground = vqd.find_excited_state(penalty_weight=0)

# Find first excited state
excited = vqd.find_excited_state(penalty_weight=1.0)

print(f"Ground state: {ground.optimal_energy:.6f}")
print(f"First excited: {excited.optimal_energy:.6f}")
```

---

## General Variational Methods

### Rayleigh-Ritz Method

**Classical variational method** - Finds the minimum eigenvalue of a matrix by optimizing over a parameterized trial state.

| Class | Description |
|-------|-------------|
| `RayleighRitz` | Rayleigh-Ritz variational method |

```python
from psiqit.variational import RayleighRitz
import math

# 2x2 Hamiltonian matrix
H = [[1, 0], [0, -1]]

# Trial state as function of parameters
def trial_state(theta):
    return [math.cos(theta/2), math.sin(theta/2)]

rr = RayleighRitz(H, trial_state, n_params=1)
result = rr.optimize(n_iterations=100, learning_rate=0.1)

print(f"Ground state energy: {result.optimal_value:.6f}")
print(f"Optimal theta: {result.optimal_params[0]:.4f}")
```

### Time-Dependent Variational Principle (TDVP)

**Solves time-dependent Schrödinger equation** - Finds optimal parameters such that the variational state best approximates the true time-evolved state.

| Class | Description |
|-------|-------------|
| `TDVP` | Time-Dependent Variational Principle |

```python
from psiqit.variational import TDVP

# Define trial wavefunction
def trial_wavefunction(theta, phi):
    return [math.cos(theta/2), math.sin(theta/2)*math.exp(1j*phi)]

# Define Hamiltonian energy function
def energy(params):
    # Simplified energy calculation
    return -math.cos(params[0])  # Example

tdvp = TDVP(hamiltonian=energy, ansatz=trial_wavefunction, n_params=2)

params0 = [0.5, 0.3]
params_history, energies = tdvp.evolve(params0, t_max=5.0, dt=0.01)

print(f"Initial energy: {energies[0]:.6f}")
print(f"Final energy: {energies[-1]:.6f}")
```

### Variational Monte Carlo (VMC)

**Uses Monte Carlo sampling to compute expectation values** - Particularly useful for large systems where exact diagonalization is infeasible.

| Class | Description |
|-------|-------------|
| `VariationalMonteCarlo` | VMC with Metropolis sampling |

```python
from psiqit.variational import VariationalMonteCarlo
import math

# Trial wavefunction for harmonic oscillator
def psi(x, alpha):
    return math.exp(-alpha * x**2)

# Harmonic oscillator potential
def potential(x):
    return 0.5 * x**2

vmc = VariationalMonteCarlo(
    wavefunction=psi,
    potential=potential,
    n_params=1
)

result = vmc.optimize(n_samples=5000, n_iterations=50, learning_rate=0.01)

print(f"Optimal alpha: {result.optimal_params[0]:.4f}")
print(f"Ground state energy: {result.optimal_value:.6f}")
```

### Hartree-Fock Method

**Mean-field approach for electronic structure** - Approximates the many-electron wavefunction as a single Slater determinant and solves self-consistent field equations.

| Class | Description |
|-------|-------------|
| `HartreeFock` | Hartree-Fock method |

```python
from psiqit.variational import HartreeFock

# 2-electron system with 4 orbitals
hf = HartreeFock(n_electrons=2, n_orbitals=4)
result = hf.solve(max_iter=50, verbose=True)

print(f"HF energy: {result.optimal_value:.6f}")
print(f"Iterations: {result.iterations}")
```

---

## Complete Examples

### Example 1: VQE for Hydrogen Molecule

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
result = vqe.run(n_iterations=150, learning_rate=0.08, verbose=True)

print(f"H₂ ground state energy: {result.optimal_energy:.6f} Hartree")
print(f"Optimization iterations: {result.n_iterations}")
```

### Example 2: QAOA for Max-Cut on a Square Graph

```python
from psiqit.variational import QAOA, maxcut_hamiltonian

# Square graph (4 nodes, 4 edges)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
H = maxcut_hamiltonian(edges)

# 2-layer QAOA
qaoa = QAOA(n_qubits=4, hamiltonian=H, p=2)
result = qaoa.run(n_iterations=150, learning_rate=0.1)

print(f"Max-Cut energy: {result.optimal_energy:.4f}")
print(f"Optimal params (γ,β): {[f'{p:.4f}' for p in result.optimal_params]}")
```

### Example 3: Excited States with SSVQE

```python
from psiqit.variational import SSVQE

# Transverse field Ising model (2 qubits)
ising_hamiltonian = {
    'Z0Z1': 1.0,   # Ising interaction
    'X0': 0.5,     # Transverse field
    'X1': 0.5,
}

ssvqe = SSVQE(n_qubits=2, hamiltonian=ising_hamiltonian, n_states=3)
result = ssvqe.run(n_iterations=200, learning_rate=0.1, verbose=True)

print("Eigenstate energies:")
for i, e in enumerate(result.energies):
    print(f"  State {i}: {e:.6f}")

print(f"\nEnergy gap (E₁ - E₀): {result.energies[1] - result.energies[0]:.6f}")
```

---

## Comparison Table

| Method | Problem | Type | Output |
|--------|---------|------|--------|
| VQE | Ground state energy | Hybrid quantum-classical | Single energy |
| QAOA | Combinatorial optimization | Hybrid quantum-classical | Energy + parameters |
| SSVQE | Multiple eigenstates | Hybrid quantum-classical | Multiple energies |
| ADAPT-VQE | Ground state (shallow circuits) | Adaptive hybrid | Energy + ansatz |
| VQD | Excited states | Hybrid quantum-classical | Energy per state |
| Rayleigh-Ritz | Matrix eigenvalues | Classical | Minimum eigenvalue |
| TDVP | Time evolution | Classical/quantum | Parameter trajectory |
| VMC | Large systems | Classical Monte Carlo | Ground state energy |
| Hartree-Fock | Electronic structure | Classical SCF | HF energy |

---

## Module Contents

```python
__all__ = [
    # VQE
    'VQEResult', 'VQE',
    # QAOA
    'QAOResult', 'QAOA', 'maxcut_hamiltonian', 'qubo_to_hamiltonian',
    # Advanced Variational
    'MultiStateResult', 'SSVQE', 'ADAPTVQE', 'VariationalQuantumDeflation',
    # General Variational
    'VariationalResult', 'RayleighRitz', 'TDVP', 
    'VariationalMonteCarlo', 'HartreeFock',
]
```

---

## References

| Method | Reference |
|--------|-----------|
| VQE | A. Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor," Nature Communications, 5:4213, 2014 |
| QAOA | E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028, 2014 |
| SSVQE | K. M. Nakanishi, K. Mitarai, and K. Fujii, "Subspace-search variational quantum eigensolver for excited states," Physical Review Research, 1(3):033062, 2019 |
| ADAPT-VQE | H. R. Grimsley et al., "An adaptive variational algorithm for exact molecular simulations on a quantum computer," Nature Communications, 10(1):3007, 2019 |
| VQD | O. Higgott, D. Wang, and S. Brierley, "Variational Quantum Computation of Excited States," Quantum, 3:156, 2019 |
| Rayleigh-Ritz | W. Ritz, "Über eine neue Methode zur Lösung gewisser Variationsprobleme der mathematischen Physik," 1909 |
| TDVP | P. Kramer and M. Saraceno, "Geometry of the Time-Dependent Variational Principle," 1981 |
| VMC | W. M. C. Foulkes et al., "Quantum Monte Carlo simulations of solids," Reviews of Modern Physics, 73(1):33-83, 2001 |
| Hartree-Fock | D. R. Hartree, "The wave mechanics of an atom with a non-Coulomb central field," 1928 |
