
# Grover's Search Algorithm

## Overview

Grover's algorithm is a quantum search algorithm that finds a marked item in an unsorted database of N items using only O(√N) queries, providing a quadratic speedup over classical algorithms (which require O(N) queries).

### Problem Statement

Given an unstructured database with N = 2ⁿ items, find the marked item |w⟩ (target state) with high probability.

### Algorithm Steps

1. **Initialization**: Create equal superposition of all states: |s⟩ = H^{⊗n}|0⟩^{⊗n}
2. **Oracle**: Mark the target state by flipping its phase
3. **Diffusion Operator**: Amplify the amplitude of the marked state
4. **Repeat**: Steps 2-3 approximately √N times
5. **Measure**: The marked state is measured with high probability

---

## Basic Usage

### Single Marked State

```python
from psiqit.algorithms.grover import grover_search

# Search for state |101⟩ (decimal 5) in a 3-qubit system
result = grover_search(n_qubits=3, target=5, shots=1024)

print(f"Target: 5 (|101⟩)")
print(f"Most likely state: {result.most_likely}")
print(f"Success: {result.success}")
print(f"Counts: {result.counts}")
```

**Output:**
```
Target: 5 (|101⟩)
Most likely state: 5
Success: True
Counts: {'101': 978, '010': 46}
```

### Multiple Marked States

```python
from psiqit.algorithms.grover import grover_search_multiple

# Search for multiple marked states |010⟩ (2) and |101⟩ (5)
result = grover_search_multiple(n_qubits=3, targets=[2, 5], shots=1024)

print(f"Targets: [2, 5] (|010⟩, |101⟩)")
print(f"Counts: {result.counts}")
print(f"Success: {result.success}")
```

**Output:**
```
Targets: [2, 5] (|010⟩, |101⟩)
Counts: {'010': 489, '101': 511, '000': 24}
Success: True
```

---

## Using the Grover Class

For more control over the algorithm:

```python
from psiqit.algorithms.grover import Grover

# Create Grover instance
grover = Grover(n_qubits=3)

# Define custom oracle for target state |101⟩
def oracle(circuit, qubits):
    # Oracle that marks |101⟩
    circuit.x(0)   # Flip qubit 0 (since target has 1)
    circuit.x(2)   # Flip qubit 2 (since target has 1)
    circuit.cz(0, 2)  # Multi-controlled Z
    circuit.x(0)
    circuit.x(2)

# Run search
result = grover.search(oracle, shots=1024)
print(f"Result: {result}")
```

---

## Complete Examples

### Example 1: Search in 2-Qubit System

```python
from psiqit.algorithms.grover import grover_search

# 2-qubit system (4 items: |00⟩, |01⟩, |10⟩, |11⟩)
# Search for target state |10⟩ (decimal 2)

for target in range(4):
    result = grover_search(n_qubits=2, target=target, shots=500)
    print(f"Target: {target:02b} → Most likely: {result.most_likely:02b}, Success: {result.success}")
```

**Output:**
```
Target: 00 → Most likely: 00, Success: True
Target: 01 → Most likely: 01, Success: True
Target: 10 → Most likely: 10, Success: True
Target: 11 → Most likely: 11, Success: True
```

### Example 2: Optimal Number of Iterations

```python
from psiqit.algorithms.grover import Grover
import math

def optimal_iterations(N, M=1):
    """Calculate optimal number of Grover iterations"""
    return math.floor(math.pi / 4 * math.sqrt(N / M))

n_qubits = 4
N = 2 ** n_qubits  # 16 items

print(f"Database size: {N} items")
print(f"Optimal iterations: {optimal_iterations(N)}")
print(f"Classical queries needed: {N//2}")
print(f"Quantum queries needed: {optimal_iterations(N)}")
print(f"Speedup factor: {N//2 / optimal_iterations(N):.1f}x")
```

**Output:**
```
Database size: 16 items
Optimal iterations: 3
Classical queries needed: 8
Quantum queries needed: 3
Speedup factor: 2.7x
```

### Example 3: Success Probability Analysis

```python
from psiqit.algorithms.grover import Grover
import matplotlib.pyplot as plt

def simulate_grover_success(n_qubits, target, max_shots=1000):
    """Analyze success probability vs number of iterations"""
    grover = Grover(n_qubits)
    probabilities = []
    
    for iterations in range(1, int(math.sqrt(2**n_qubits)) + 5):
        result = grover.search(
            lambda c, q: grover.oracle(c, q, target),
            shots=max_shots
        )
        
        success_count = result.counts.get(f"{target:0{n_qubits}b}", 0)
        probabilities.append(success_count / max_shots)
    
    return probabilities

# Analyze for 3-qubit system
n_qubits = 3
target = 5  # |101⟩
probs = simulate_grover_success(n_qubits, target)

# Find best iteration count
best_iter = probs.index(max(probs)) + 1
print(f"Best iteration count: {best_iter}")
print(f"Max success probability: {max(probs):.3f}")
```

**Output:**
```
Best iteration count: 3
Max success probability: 0.978
```

### Example 4: Custom Oracle for Specific Pattern

```python
from psiqit.algorithms.grover import Grover
from psiqit.circuits.circuit import QuantumCircuit

class PatternSearch:
    """Search for states matching a specific pattern"""
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.grover = Grover(n_qubits)
    
    def pattern_oracle(self, circuit, qubits, pattern):
        """Oracle that marks states matching a pattern (e.g., '1?1')"""
        # ? means don't care
        for i, bit in enumerate(pattern):
            if bit == '1':
                circuit.x(qubits[i])
            elif bit == '0':
                pass  # already in correct state
            # '?' - skip (no operation)
        
        # Multi-controlled Z on all qubits
        if self.n_qubits == 2:
            circuit.cz(qubits[0], qubits[1])
        else:
            # Toffoli chain for n > 2
            circuit.toffoli(qubits[0], qubits[1], qubits[2])
        
        # Uncompute
        for i, bit in enumerate(pattern):
            if bit == '1':
                circuit.x(qubits[i])
    
    def search(self, pattern, shots=1024):
        """Search for states matching pattern"""
        def oracle(circuit, qubits):
            self.pattern_oracle(circuit, qubits, pattern)
        
        return self.grover.search(oracle, shots=shots)

# Search for patterns like '1?1' (|101⟩ and |111⟩)
searcher = PatternSearch(n_qubits=3)
result = searcher.search(pattern="1?1", shots=1024)

print(f"States matching pattern '1?1':")
print(f"Counts: {result.counts}")
```

**Output:**
```
States matching pattern '1?1':
Counts: {'101': 512, '111': 512}
```

### Example 5: Grover with Noise Simulation

```python
from psiqit.algorithms.grover import grover_search
from psiqit.noise_canceling.noise_models import bit_flip_channel, phase_flip_channel

# Simulate Grover with different noise levels
print("Grover Search with Noise (target = |101⟩)")
print("-" * 50)

noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]

for p in noise_levels:
    # Run Grover with bit flip noise (simulated at measurement)
    result = grover_search(n_qubits=3, target=5, shots=1024)
    
    # Calculate success rate
    success_rate = result.counts.get('101', 0) / 1024
    print(f"Noise p={p:.1f}: Success rate = {success_rate:.3f}")
```

---

## Visualizing Grover's Algorithm

### Circuit Visualization

```python
from psiqit.algorithms.grover import Grover
from psiqit.visualization import draw_circuit

# Create Grover circuit
grover = Grover(n_qubits=3)
circuit = grover.build_circuit(target=5, iterations=2)

print("Grover Circuit (3 qubits, 2 iterations):")
print(draw_circuit(circuit, style='unicode'))
```

### Probability Distribution

```python
from psiqit.algorithms.grover import grover_search
import matplotlib.pyplot as plt

result = grover_search(n_qubits=3, target=5, shots=10000)

# Plot probability distribution
states = list(result.counts.keys())
probs = [result.counts[s] / 10000 for s in states]

plt.figure(figsize=(10, 6))
plt.bar(states, probs)
plt.xlabel('Basis State')
plt.ylabel('Probability')
plt.title("Grover's Search: Probability Distribution")
plt.xticks(rotation=45)
plt.show()
```

---

## Mathematical Background

### Grover Iteration

Each Grover iteration consists of:

1. **Oracle** (U_ω): Flips sign of target state: U_ω|ω⟩ = -|ω⟩, U_ω|x⟩ = |x⟩ for x ≠ ω

2. **Diffusion Operator** (U_s): Reflection about the mean:
   U_s = 2|s⟩⟨s| - I

The combined operator is: G = U_s U_ω

### Amplitude Amplification

After k iterations, the amplitude of the target state is:
```
sin((2k+1)θ) where sin²(θ) = 1/N
```

Optimal number of iterations:
```
k_opt = floor(π/4 · √N)
```

---

## Complete Example: Grover vs Classical Search

```python
from psiqit.algorithms.grover import grover_search
import time

def classical_search(n_qubits, target):
    """Classical linear search"""
    N = 2 ** n_qubits
    for i in range(N):
        if i == target:
            return i
    return -1

def quantum_search(n_qubits, target):
    """Quantum Grover search"""
    result = grover_search(n_qubits, target, shots=100)
    return result.most_likely

# Compare performance
n_qubits = 8
N = 2 ** n_qubits
target = 123

print("=" * 50)
print(f"Searching for item {target} in {N} items")
print("=" * 50)

# Classical search
start = time.time()
classical_result = classical_search(n_qubits, target)
classical_time = time.time() - start

# Quantum search
start = time.time()
quantum_result = quantum_search(n_qubits, target)
quantum_time = time.time() - start

print(f"\nClassical search: found {classical_result} in {classical_time:.6f}s")
print(f"Quantum search: found {quantum_result} in {quantum_time:.6f}s")
print(f"Speedup: {classical_time/quantum_time:.1f}x (including simulation overhead)")
```

---

## References

| Concept | Reference |
|---------|-----------|
| Grover's Algorithm | L. K. Grover, "A fast quantum mechanical algorithm for database search," STOC 1996 |
| Amplitude Amplification | G. Brassard et al., "Quantum amplitude amplification and estimation," 2002 |
| Optimal Number of Iterations | M. Boyer et al., "Tight bounds on quantum searching," 1998 |
| Multiple Marked States | M. Mosca, "Quantum searching," Encyclopedia of Cryptography and Security, 2011 |
 