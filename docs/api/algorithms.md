
# Algorithms API

## Module: `psiqit.algorithms`

This module provides implementations of famous quantum algorithms that demonstrate quantum advantage over classical counterparts.

---

## Grover's Search Algorithm

**Quadratic speedup for unstructured search** - Finds a marked item in an unsorted database of N items using O(√N) queries.

### Classes

| Class | Description |
|-------|-------------|
| `GroverResult` | Result container with counts, most likely state, success status |
| `Grover` | Main Grover search implementation |

### Functions

| Function | Description |
|----------|-------------|
| `grover_search(n_qubits, target, shots)` | Search for a single marked state |
| `grover_search_multiple(n_qubits, targets, shots)` | Search for multiple marked states |

### Example

```python
from psiqit.algorithms.grover import grover_search

# Search for state |101⟩ (decimal 5) in a 3-qubit system
result = grover_search(n_qubits=3, target=5, shots=1024)

print(f"Most likely state: {result.most_likely}")  # 5
print(f"Success: {result.success}")               # True
print(f"Counts: {result.counts}")                 # {'101': 978, '010': 46}
```

---

## Deutsch-Jozsa Algorithm

**Exponential speedup** - Determines whether a Boolean function f: {0,1}ⁿ → {0,1} is constant or balanced using a single quantum query.

### Classes

| Class | Description |
|-------|-------------|
| `DeutschJozsaResult` | Result with is_constant, is_balanced, confidence |
| `DeutschJozsa` | n-qubit Deutsch-Jozsa algorithm |
| `DeutschAlgorithm` | 1-qubit Deutsch algorithm (simplified version) |

### Functions

| Function | Description |
|----------|-------------|
| `deutsch_jozsa_constant(n_qubits, constant_value, shots)` | Test a constant function |
| `deutsch_jozsa_balanced(n_qubits, shots)` | Test a balanced function |
| `deutsch_algorithm_constant(constant_value, shots)` | 1-qubit constant test |
| `deutsch_algorithm_balanced(shots)` | 1-qubit balanced test |

### Example

```python
from psiqit.algorithms.deutsch_jozsa import deutsch_jozsa_constant, deutsch_jozsa_balanced

# Test a constant function (always returns 0)
result = deutsch_jozsa_constant(n_qubits=3, constant_value=0)
print(result)  # Function is CONSTANT

# Test a balanced function (returns 0 for half the inputs, 1 for half)
result = deutsch_jozsa_balanced(n_qubits=3)
print(result)  # Function is BALANCED
```

---

## Shor's Factoring Algorithm

**Exponential speedup** - Factors integers exponentially faster than the best known classical algorithms. Has significant implications for cryptography (RSA).

### Classes

| Class | Description |
|-------|-------------|
| `ShorResult` | Result with factors, period, success status |
| `Shor` | Main Shor's algorithm implementation |

### Functions

| Function | Description |
|----------|-------------|
| `factorize(N)` | Convenience function for factoring |

### Example

```python
from psiqit.algorithms.shor import Shor, factorize

# Using the class
shor = Shor()
result = shor.factor(15)
print(result)  # 15 = 3 × 5

# Using convenience function
factors = factorize(21)
print(factors)  # [3, 7]
```

---

## Quantum Fourier Transform (QFT)

**Quantum analogue of classical DFT** - Transforms a quantum state from the computational basis to the Fourier basis. Used in Shor's algorithm and quantum phase estimation.

### Classes

| Class | Description |
|-------|-------------|
| `QFTResult` | Result with input_state, output_state, circuit_depth |
| `QFT` | Quantum Fourier Transform |
| `IQFT` | Inverse Quantum Fourier Transform |

### Functions

| Function | Description |
|----------|-------------|
| `qft_circuit(n_qubits, inverse)` | Create QFT circuit |
| `apply_qft(state, inverse)` | Apply QFT to a state |
| `qft_matrix(n_qubits, inverse)` | Get QFT matrix representation |

### Example

```python
from psiqit.algorithms.qft import QFT, apply_qft
from psiqit.quantum.state import basis

# Create QFT for 3 qubits
qft = QFT(n_qubits=3)

# Apply to basis state |001⟩
state = basis(8, 1)  # |001⟩
transformed = qft.apply(state)
print(transformed)  # Equal superposition with phases

# Using convenience function
result = apply_qft(state)
```

---

## Quantum Phase Estimation (QPE)

**Estimates eigenvalue phases** - Given a unitary operator U and an eigenstate |ψ⟩ with U|ψ⟩ = e^{2πiφ}|ψ⟩, estimates φ with high precision.

### Classes

| Class | Description |
|-------|-------------|
| `QPEResult` | Result with phase, eigenvalue, precision |
| `QPE` | Quantum Phase Estimation (simplified) |

### Example

```python
from psiqit.algorithms.qpe import QPE

# Estimate phase with 4-bit precision
qpe = QPE(n_phase_qubits=4)
result = qpe.estimate()

print(f"Estimated phase: {result.phase:.6f}")
print(f"Eigenvalue: {result.eigenvalue}")
```

---

## Simon's Algorithm

**Exponential speedup** - Finds the hidden period s of a function f: {0,1}ⁿ → {0,1}ⁿ where f(x) = f(y) iff y = x ⊕ s.

### Classes

| Class | Description |
|-------|-------------|
| `SimonResult` | Result with period, equations, success |
| `Simon` | Simon's algorithm |

### Example

```python
from psiqit.algorithms.simon import Simon

# Define oracle for s = 3 (binary 011)
def oracle(circuit, x_qubits, ancilla):
    circuit.cx(0, ancilla)
    circuit.cx(1, ancilla)

simon = Simon(n_qubits=3)
result = simon.run(oracle)

print(f"Found period: {result.period:03b}")  # 011
print(f"Success: {result.success}")          # True
```

---

## Bernstein-Vazirani Algorithm

**Finds hidden bit string** - Determines a hidden string s ∈ {0,1}ⁿ using a single quantum query, compared to n classical queries.

### Classes

| Class | Description |
|-------|-------------|
| `BernsteinVaziraniResult` | Result with hidden_string, n_qubits, success |
| `BernsteinVazirani` | Bernstein-Vazirani algorithm |

### Functions

| Function | Description |
|----------|-------------|
| `bernstein_vazirani(n_qubits)` | Convenience function |

### Example

```python
from psiqit.algorithms.bernstein_vazirani import bernstein_vazirani

# Find hidden 3-bit string
hidden = bernstein_vazirani(n_qubits=3)
print(f"Hidden string: {hidden:03b}")  # e.g., 101
print(f"Decimal: {hidden}")            # 5
```

---

## Summary Table: Quantum Algorithms

| Algorithm | Problem | Speedup | Qubits |
|-----------|---------|---------|--------|
| Grover | Unstructured search | Quadratic | O(log N) |
| Deutsch-Jozsa | Constant vs. balanced | Exponential | n + 1 |
| Shor | Integer factorization | Exponential | O(log N) |
| QFT | Fourier transform | Exponential | n |
| QPE | Phase estimation | Exponential | n + m |
| Simon | Period finding | Exponential | n + 1 |
| Bernstein-Vazirani | Hidden string | Exponential | n + 1 |

---

## Module Contents

```python
__all__ = [
    # Grover
    'Grover', 'GroverResult', 'grover_search', 'grover_search_multiple',
    # Deutsch-Jozsa
    'DeutschJozsa', 'DeutschJozsaResult', 'DeutschAlgorithm',
    'deutsch_jozsa_constant', 'deutsch_jozsa_balanced',
    'deutsch_algorithm_constant', 'deutsch_algorithm_balanced',
    # Shor
    'Shor', 'ShorResult', 'factorize',
    # QFT
    'QFT', 'QFTResult', 'IQFT', 'qft_circuit', 'apply_qft', 'qft_matrix',
    # QPE
    'QPE', 'QPEResult',
    # Simon
    'Simon', 'SimonResult',
    # Bernstein-Vazirani
    'BernsteinVazirani', 'BernsteinVaziraniResult', 'bernstein_vazirani',
]
```

---

## References

| Algorithm | Reference |
|-----------|-----------|
| Grover | L. K. Grover, "A fast quantum mechanical algorithm for database search," STOC 1996 |
| Deutsch-Jozsa | D. Deutsch and R. Jozsa, "Rapid solution of problems by quantum computation," 1992 |
| Shor | P. W. Shor, "Polynomial-Time Algorithms for Prime Factorization," SIAM 1997 |
| QFT | D. Coppersmith, "An approximate Fourier transform useful in quantum factoring," 1994 |
| QPE | A. Y. Kitaev, "Quantum measurements and the Abelian stabilizer problem," 1995 |
| Simon | D. R. Simon, "On the power of quantum computation," FOCS 1994 |
| Bernstein-Vazirani | E. Bernstein and U. Vazirani, "Quantum complexity theory," STOC 1993 |
