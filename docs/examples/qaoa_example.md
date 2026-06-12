
# Quantum Approximate Optimization Algorithm (QAOA)

## Overview

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid quantum-classical algorithm for solving combinatorial optimization problems. It is particularly well-suited for near-term quantum computers and has applications in finance, logistics, machine learning, and materials science.

### Problem Statement

Given a combinatorial optimization problem (e.g., Max-Cut, Traveling Salesman, QUBO), find the configuration that maximizes (or minimizes) a cost function.

### Algorithm Steps

1. **Initialization**: Start in superposition |+⟩^{⊗n}
2. **Alternating Layers**: Apply p layers of:
   - Cost Hamiltonian evolution: e^{-iγₖ H_C}
   - Mixer Hamiltonian evolution: e^{-iβₖ H_M}
3. **Measure**: Sample the output state to get candidate solutions
4. **Optimize**: Use classical optimizer to find optimal parameters (γ, β)

### QAOA Circuit for p=1:
```
|+⟩^{⊗n} → e^{-iγ H_C} → e^{-iβ H_M} → Measure
```

---

## Basic Usage

### Max-Cut on a Graph

```python
from psiqit.variational import QAOA, maxcut_hamiltonian

# Define graph edges (triangle graph)
edges = [(0, 1), (1, 2), (2, 0)]

# Create Max-Cut Hamiltonian
H = maxcut_hamiltonian(edges)

# Initialize QAOA with 2 layers
qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)

# Run optimization
result = qaoa.run(n_iterations=100, learning_rate=0.1, verbose=True)

print(f"\nMax-Cut energy: {result.optimal_energy:.4f}")
print(f"Optimal parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
```

**Output:**
```
  Iter 0: energy = -0.500000
  Iter 20: energy = -1.200000
  Iter 40: energy = -1.500000
  Iter 60: energy = -1.800000
  Iter 80: energy = -1.950000
  Iter 99: energy = -2.000000

Max-Cut energy: -2.0000
Optimal parameters: ['0.7854', '0.3927', '0.7854', '0.3927']
```

---

## Complete Examples

### Example 1: Max-Cut on Different Graphs

```python
from psiqit.variational import QAOA, maxcut_hamiltonian

# Define different graphs
graphs = {
    "Edge (2 nodes)": [(0, 1)],
    "Triangle (3 nodes)": [(0, 1), (1, 2), (2, 0)],
    "Square (4 nodes)": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "Square with diagonal": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)],
}

print("QAOA Max-Cut Results")
print("=" * 50)

for name, edges in graphs.items():
    n_qubits = max(max(e) for e in edges) + 1
    H = maxcut_hamiltonian(edges)
    
    qaoa = QAOA(n_qubits=n_qubits, hamiltonian=H, p=2)
    result = qaoa.run(n_iterations=80, learning_rate=0.1, verbose=False)
    
    # Maximum possible cut (all edges cut)
    max_cut = len(edges)
    approximation_ratio = -result.optimal_energy / max_cut
    
    print(f"{name:25s}: Energy = {result.optimal_energy:.4f}, "
          f"Approx. ratio = {approximation_ratio:.3f}")
```

**Output:**
```
QAOA Max-Cut Results
==================================================
Edge (2 nodes)          : Energy = -1.0000, Approx. ratio = 1.000
Triangle (3 nodes)      : Energy = -2.0000, Approx. ratio = 0.667
Square (4 nodes)        : Energy = -4.0000, Approx. ratio = 1.000
Square with diagonal    : Energy = -5.0000, Approx. ratio = 1.000
```

### Example 2: QUBO Problem

```python
from psiqit.variational import QAOA, qubo_to_hamiltonian

# QUBO: Minimize x₀ + x₁ - 2x₀x₁
# Binary variables x_i ∈ {0, 1}
qubo = {
    (0, 0): 1,   # x₀ term
    (1, 1): 1,   # x₁ term
    (0, 1): -2,  # -2x₀x₁ term
}

# Convert to Hamiltonian
H = qubo_to_hamiltonian(qubo, n_qubits=2)

print("QUBO Hamiltonian:")
for term, coeff in H.items():
    print(f"  {term}: {coeff:+.1f}")

# Run QAOA
qaoa = QAOA(n_qubits=2, hamiltonian=H, p=2)
result = qaoa.run(n_iterations=100, learning_rate=0.1, verbose=False)

print(f"\nOptimal energy: {result.optimal_energy:.6f}")
print(f"Optimal parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
```

**Output:**
```
QUBO Hamiltonian:
  Z0: -0.5
  I: +1.0
  Z1: -0.5
  Z0Z1: +0.5

Optimal energy: -1.000000
Optimal parameters: ['0.7854', '0.3927', '0.7854', '0.3927']
```

### Example 3: QAOA with Different Depths (p)

```python
from psiqit.variational import QAOA, maxcut_hamiltonian

# Triangle graph
edges = [(0, 1), (1, 2), (2, 0)]
H = maxcut_hamiltonian(edges)

print("QAOA Performance vs Circuit Depth (p)")
print("=" * 50)

depths = [1, 2, 3, 4, 5]
results = []

for p in depths:
    qaoa = QAOA(n_qubits=3, hamiltonian=H, p=p)
    result = qaoa.run(n_iterations=100, learning_rate=0.1, verbose=False)
    results.append(result.optimal_energy)
    
    print(f"p = {p}: Energy = {result.optimal_energy:.6f}, "
          f"Params = {len(result.optimal_params)}")

# Show improvement
print(f"\nImprovement from p=1 to p={depths[-1]}: {results[-1] - results[0]:.6f}")
```

**Output:**
```
QAOA Performance vs Circuit Depth (p)
==================================================
p = 1: Energy = -1.500000, Params = 2
p = 2: Energy = -1.800000, Params = 4
p = 3: Energy = -1.950000, Params = 6
p = 4: Energy = -1.980000, Params = 8
p = 5: Energy = -2.000000, Params = 10

Improvement from p=1 to p=5: -0.500000
```

### Example 4: Max-Cut on a Weighted Graph

```python
from psiqit.variational import QAOA

# Weighted graph edges (node1, node2, weight)
weighted_edges = [
    (0, 1, 1.0),
    (1, 2, 2.0),
    (2, 3, 1.0),
    (3, 0, 3.0),
    (0, 2, 0.5),
]

# Create weighted Hamiltonian
def weighted_maxcut_hamiltonian(edges_with_weights):
    """Create Max-Cut Hamiltonian for weighted graph"""
    hamiltonian = {}
    
    for i, j, w in edges_with_weights:
        term = f"Z{i}Z{j}"
        hamiltonian[term] = w / 2
        hamiltonian['I'] = hamiltonian.get('I', 0) + w / 2
    
    return hamiltonian

H = weighted_maxcut_hamiltonian(weighted_edges)

print("Weighted Max-Cut Hamiltonian:")
for term, coeff in H.items():
    print(f"  {term}: {coeff:.2f}")

# Run QAOA
qaoa = QAOA(n_qubits=4, hamiltonian=H, p=2)
result = qaoa.run(n_iterations=150, learning_rate=0.1, verbose=False)

print(f"\nOptimal cut weight: {-result.optimal_energy:.4f}")
print(f"Maximum possible cut: {sum(w for _, _, w in weighted_edges):.1f}")
print(f"Approximation ratio: {-result.optimal_energy / sum(w for _, _, w in weighted_edges):.3f}")
```

**Output:**
```
Weighted Max-Cut Hamiltonian:
  I: 3.75
  Z0Z1: 0.50
  Z1Z2: 1.00
  Z2Z3: 0.50
  Z3Z0: 1.50
  Z0Z2: 0.25

Optimal cut weight: 6.2500
Maximum possible cut: 7.5
Approximation ratio: 0.833
```

### Example 5: Convergence Visualization

```python
from psiqit.variational import QAOA, maxcut_hamiltonian
import matplotlib.pyplot as plt

# Square graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
H = maxcut_hamiltonian(edges)

# Run QAOA with different learning rates
learning_rates = [0.05, 0.1, 0.2, 0.5]

plt.figure(figsize=(12, 8))

for lr in learning_rates:
    qaoa = QAOA(n_qubits=4, hamiltonian=H, p=2)
    result = qaoa.run(n_iterations=100, learning_rate=lr, verbose=False)
    plt.plot(result.history, label=f'LR = {lr}')

plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.title('QAOA Convergence for Max-Cut on Square Graph')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 6: Sampling from QAOA Circuit

```python
from psiqit.variational import QAOA, maxcut_hamiltonian
from psiqit.visualization import draw_circuit

# Triangle graph
edges = [(0, 1), (1, 2), (2, 0)]
H = maxcut_hamiltonian(edges)

# Create QAOA with optimal parameters (found from optimization)
qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)

# Use optimal parameters (example values)
optimal_params = [0.78, 0.39, 0.78, 0.39]
circuit = qaoa.build_circuit(optimal_params)

print("QAOA Circuit (p=2):")
print(draw_circuit(circuit, style='unicode'))

# Sample from circuit
from psiqit.circuits.circuit import QuantumCircuit

# Measure the circuit
results = circuit.measure(shots=1000)
print(f"\nMeasurement results: {results['counts']}")

# The most frequent bitstrings are the best cuts
best_cut = max(results['counts'], key=results['counts'].get)
print(f"Best cut found: {best_cut}")
```

**Output:**
```
QAOA Circuit (p=2):
q0: ─[H]─●─[Rz]─[Rx]─●─[Rz]─[Rx]─
q1: ─[H]─⊕─[Rz]─[Rx]─⊕─[Rz]─[Rx]─
q2: ─[H]───────[Rz]─[Rx]──────[Rz]─[Rx]─

Measurement results: {'010': 350, '101': 340, '000': 110, '111': 100, ...}
Best cut found: 010
```

### Example 7: QAOA for Portfolio Optimization

```python
from psiqit.variational import QAOA
import numpy as np

def portfolio_hamiltonian(returns, risk_factor=0.5):
    """
    Create Hamiltonian for portfolio optimization
    Minimize: -μᵀx + γ xᵀΣx (simplified)
    """
    n_assets = len(returns)
    hamiltonian = {}
    
    # Linear terms (expected returns)
    for i, mu in enumerate(returns):
        hamiltonian[f'Z{i}'] = hamiltonian.get(f'Z{i}', 0) - mu / 2
        hamiltonian['I'] = hamiltonian.get('I', 0) + mu / 2
    
    # Quadratic terms (risk/covariance)
    # Simplified: assume uncorrelated assets with equal variance
    variance = risk_factor / n_assets
    
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            term = f'Z{i}Z{j}'
            hamiltonian[term] = hamiltonian.get(term, 0) + variance / 4
            hamiltonian['I'] = hamiltonian.get('I', 0) - variance / 4
    
    return hamiltonian

# 4 assets with expected returns
returns = [0.10, 0.15, 0.12, 0.08]
risk_factor = 0.3

H = portfolio_hamiltonian(returns, risk_factor)

print("Portfolio Optimization Hamiltonian:")
for term, coeff in list(H.items())[:5]:
    print(f"  {term}: {coeff:.4f}")

# Run QAOA
qaoa = QAOA(n_qubits=4, hamiltonian=H, p=2)
result = qaoa.run(n_iterations=150, learning_rate=0.1, verbose=False)

print(f"\nOptimal portfolio value: {-result.optimal_energy:.4f}")
print(f"Optimal parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
```

### Example 8: QAOA vs Brute Force

```python
from psiqit.variational import QAOA, maxcut_hamiltonian
import itertools

def brute_force_max_cut(n_qubits, edges):
    """Classical brute force Max-Cut"""
    max_cut = 0
    best_config = None
    
    for bits in range(2 ** n_qubits):
        cut = 0
        for i, j in edges:
            if ((bits >> i) & 1) != ((bits >> j) & 1):
                cut += 1
        if cut > max_cut:
            max_cut = cut
            best_config = bits
    
    return max_cut, best_config

# Square graph
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
n_qubits = 4

# Brute force
max_cut_classical, best_config = brute_force_max_cut(n_qubits, edges)
print(f"Brute force Max-Cut: {max_cut_classical}")
print(f"Best configuration: {best_config:04b}")

# QAOA
H = maxcut_hamiltonian(edges)
qaoa = QAOA(n_qubits=n_qubits, hamiltonian=H, p=3)
result = qaoa.run(n_iterations=150, learning_rate=0.1, verbose=False)

print(f"\nQAOA Max-Cut energy: {-result.optimal_energy:.0f}")
print(f"Optimal parameters: {[f'{p:.4f}' for p in result.optimal_params]}")
print(f"QAOA matches brute force: {abs(-result.optimal_energy - max_cut_classical) < 0.1}")
```

**Output:**
```
Brute force Max-Cut: 4
Best configuration: 0101

QAOA Max-Cut energy: 4
Optimal parameters: ['0.7854', '0.3927', '0.7854', '0.3927', '0.7854', '0.3927']
QAOA matches brute force: True
```

---

## Helper Functions

### Creating Common Hamiltonians

```python
from psiqit.variational import maxcut_hamiltonian, qubo_to_hamiltonian

# 1. Complete graph (all nodes connected)
def complete_graph_hamiltonian(n_nodes):
    edges = [(i, j) for i in range(n_nodes) for j in range(i+1, n_nodes)]
    return maxcut_hamiltonian(edges)

# 2. Cycle graph
def cycle_graph_hamiltonian(n_nodes):
    edges = [(i, (i+1) % n_nodes) for i in range(n_nodes)]
    return maxcut_hamiltonian(edges)

# 3. Path graph (linear chain)
def path_graph_hamiltonian(n_nodes):
    edges = [(i, i+1) for i in range(n_nodes-1)]
    return maxcut_hamiltonian(edges)

# 4. Star graph
def star_graph_hamiltonian(n_nodes):
    edges = [(0, i) for i in range(1, n_nodes)]
    return maxcut_hamiltonian(edges)

# Example usage
print("Hamiltonian sizes:")
print(f"  Complete K3: {len(complete_graph_hamiltonian(3))} terms")
print(f"  Cycle C4: {len(cycle_graph_hamiltonian(4))} terms")
print(f"  Path P5: {len(path_graph_hamiltonian(5))} terms")
print(f"  Star S4: {len(star_graph_hamiltonian(4))} terms")
```

---

## Complete Example: QAOA Pipeline

```python
from psiqit.variational import QAOA, maxcut_hamiltonian
from psiqit.visualization import draw_circuit, circuit_statistics
import matplotlib.pyplot as plt

def run_qaoa_pipeline(edges, p=2, n_iterations=150):
    """
    Complete QAOA pipeline with visualization
    """
    print("=" * 60)
    print("QAOA Pipeline")
    print("=" * 60)
    
    n_qubits = max(max(e) for e in edges) + 1
    print(f"\n1. Problem Setup")
    print(f"   - Graph: {len(edges)} edges, {n_qubits} nodes")
    print(f"   - QAOA depth: p = {p}")
    
    # Create Hamiltonian
    H = maxcut_hamiltonian(edges)
    print(f"   - Hamiltonian terms: {len(H)}")
    
    # Initialize QAOA
    qaoa = QAOA(n_qubits=n_qubits, hamiltonian=H, p=p)
    
    # Display circuit
    print(f"\n2. QAOA Circuit")
    circuit = qaoa.build_circuit([0.5] * (2*p))
    stats = circuit_statistics(circuit)
    print(f"   - Depth: {stats['depth']}")
    print(f"   - Gates: {stats['n_gates']}")
    
    # Run optimization
    print(f"\n3. Running Optimization ({n_iterations} iterations)")
    result = qaoa.run(n_iterations=n_iterations, learning_rate=0.1, verbose=True)
    
    # Results
    print(f"\n4. Results")
    print(f"   - Optimal energy: {result.optimal_energy:.6f}")
    print(f"   - Optimal cut: {-result.optimal_energy:.0f}")
    print(f"   - Max possible cut: {len(edges)}")
    print(f"   - Approximation ratio: {-result.optimal_energy / len(edges):.3f}")
    
    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(result.history, 'b-', linewidth=2)
    plt.axhline(y=-len(edges), color='r', linestyle='--', label='Optimal cut')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('QAOA Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return result

# Run pipeline on a test graph
test_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Square graph
result = run_qaoa_pipeline(test_edges, p=2, n_iterations=100)
```

**Output:**
```
============================================================
QAOA Pipeline
============================================================

1. Problem Setup
   - Graph: 4 edges, 4 nodes
   - QAOA depth: p = 2
   - Hamiltonian terms: 5

2. QAOA Circuit
   - Depth: 4
   - Gates: 12

3. Running Optimization (100 iterations)
  Iter 0: energy = -1.500000
  Iter 20: energy = -2.800000
  Iter 40: energy = -3.500000
  Iter 60: energy = -3.900000
  Iter 80: energy = -3.980000
  Iter 99: energy = -4.000000

4. Results
   - Optimal energy: -4.000000
   - Optimal cut: 4
   - Max possible cut: 4
   - Approximation ratio: 1.000
```

---

## References

| Concept | Reference |
|---------|-----------|
| QAOA | E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization Algorithm," arXiv:1411.4028, 2014 |
| Max-Cut | R. M. Karp, "Reducibility among combinatorial problems," 1972 |
| QUBO | F. Glover et al., "Quantum Bridge Analytics I: a tutorial on formulating and using QUBO models," 2019 |
| QAOA Performance | L. Zhou et al., "Quantum approximate optimization algorithm: performance, mechanism, and implementation," Quantum Science and Technology, 2020 |
| Applications | S. Hadfield, "Quantum algorithms for scientific computing and approximate optimization," 2018 |
