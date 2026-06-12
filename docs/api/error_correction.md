
# Error Correction API

## Module: `psiqit.error_correction`

This module provides implementations of quantum error correction codes that protect quantum information against decoherence and operational errors. Quantum error correction is essential for building fault-tolerant quantum computers.

**Available codes:**
- **BitFlipCode**: 3-qubit repetition code for bit flip errors
- **PhaseFlipCode**: 3-qubit code for phase flip errors (using Hadamard transform)
- **ShorCode**: 9-qubit code that corrects any single-qubit error
- **SteaneCode**: 7-qubit CSS code (Calderbank-Shor-Steane)

---

## CorrectionResult

**Result container for error correction operations.**

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `corrected_state` | `Ket` | Corrected quantum state |
| `syndrome` | `List[int]` | Measured syndrome bits (error signature) |
| `errors_detected` | `int` | Number of errors detected |
| `errors_corrected` | `int` | Number of errors successfully corrected |
| `success` | `bool` | Whether correction was successful |

### Example

```python
from psiqit.error_correction import CorrectionResult

result = CorrectionResult(
    corrected_state=corrected_psi,
    syndrome=[0, 0, 0],
    errors_detected=0,
    errors_corrected=0,
    success=True
)
print(f"Correction success: {result.success}")
```

---

## BitFlipCode

**3-qubit repetition code for bit flip errors** - Encodes a logical qubit into 3 physical qubits: |0⟩_L = |000⟩, |1⟩_L = |111⟩. Can detect and correct up to one bit flip error using majority voting.

### Class

| Class | Description |
|-------|-------------|
| `BitFlipCode` | 3-qubit repetition code (or n-qubit generalization) |

### Methods

| Method | Description |
|--------|-------------|
| `encode(state)` | Encode single qubit into repetition code |
| `decode(encoded)` | Decode and correct errors using majority voting |
| `circuit_encode()` | Create circuit that encodes a qubit |

### Example 1: Encode and Decode

```python
from psiqit.error_correction import BitFlipCode
from psiqit.quantum.state import zero, one

code = BitFlipCode(n=3)

# Encode logical |0⟩
logical_zero = code.encode(zero())
print(f"Encoded |0⟩: {logical_zero}")  # 1.000|000⟩

# Encode logical |1⟩
logical_one = code.encode(one())
print(f"Encoded |1⟩: {logical_one}")  # 1.000|111⟩
```

### Example 2: Error Correction

```python
from psiqit.error_correction import BitFlipCode
from psiqit.quantum.state import zero
from psiqit.quantum.operator import pauli_x
from psiqit.circuits.circuit import QuantumCircuit

code = BitFlipCode(n=3)

# Encode logical |0⟩
logical = code.encode(zero())

# Simulate a bit flip error on qubit 1
# Apply X gate to second qubit
circuit = QuantumCircuit(3)
# This is simplified - actual error simulation would need state manipulation

# Decode and correct
result = code.decode(logical)
print(f"Corrected state: {result.corrected_state}")  # Should be |0⟩
print(f"Errors detected: {result.errors_detected}")
print(f"Success: {result.success}")
```

### Example 3: Encoding Circuit

```python
from psiqit.error_correction import BitFlipCode
from psiqit.visualization.circuit_drawer import draw_circuit

code = BitFlipCode(n=3)
circuit = code.circuit_encode()

print("Encoding Circuit:")
print(draw_circuit(circuit, style='unicode'))
# q0: ────●──●──
# q1: ────X─────
# q2: ────────X─
```

---

## PhaseFlipCode

**3-qubit repetition code for phase flip errors** - Uses Hadamard transform to convert phase flips into bit flips, then applies the bit flip code.

### Class

| Class | Description |
|-------|-------------|
| `PhaseFlipCode` | 3-qubit code for phase flip errors |

### Methods

| Method | Description |
|--------|-------------|
| `encode(state)` | Encode single qubit into phase flip code |
| `decode(encoded)` | Decode and correct phase flip errors |

### Example

```python
from psiqit.error_correction import PhaseFlipCode
from psiqit.quantum.state import plus, minus

code = PhaseFlipCode(n=3)

# Encode logical |+⟩
logical_plus = code.encode(plus())
print(f"Encoded |+⟩: {logical_plus}")

# Encode logical |-⟩
logical_minus = code.encode(minus())
print(f"Encoded |-⟩: {logical_minus}")

# Decode (with error correction)
result = code.decode(logical_plus)
print(f"Decoded: {result.corrected_state}")
```

---

## ShorCode

**9-qubit Shor code** - First quantum error-correcting code that can correct arbitrary single-qubit errors (both bit flips and phase flips). Concatenates the bit flip and phase flip codes.

### Class

| Class | Description |
|-------|-------------|
| `ShorCode` | 9-qubit Shor code |

### Methods

| Method | Description |
|--------|-------------|
| `encode(state)` | Encode single qubit into Shor code |
| `decode(encoded)` | Decode Shor code |

### Encoding

| Logical State | Encoding |
|---------------|----------|
| \|0⟩_L | (\|000⟩ + \|111⟩)⊗³ / √8 |
| \|1⟩_L | (\|000⟩ - \|111⟩)⊗³ / √8 |

### Example

```python
from psiqit.error_correction import ShorCode
from psiqit.quantum.state import zero, one

code = ShorCode()

# Encode logical |0⟩
logical_zero = code.encode(zero())
print(f"Encoded |0⟩ dimension: {logical_zero.dim}")  # 512 (2^9)

# Encode logical |1⟩
logical_one = code.encode(one())

# Decode
result = code.decode(logical_zero)
print(f"Decoded state: {result.corrected_state}")
print(f"Success: {result.success}")
```

---

## SteaneCode

**7-qubit Steane code** - A Calderbank-Shor-Steane (CSS) code based on the classical [7,4,3] Hamming code. Corrects both bit flip and phase flip errors using only 7 physical qubits.

### Class

| Class | Description |
|-------|-------------|
| `SteaneCode` | 7-qubit Steane code |

### Methods

| Method | Description |
|--------|-------------|
| `encode(state)` | Encode single qubit into Steane code |
| `decode(encoded)` | Decode Steane code |

### Example

```python
from psiqit.error_correction import SteaneCode
from psiqit.quantum.state import zero

code = SteaneCode()

# Encode
logical_zero = code.encode(zero())
print(f"Encoded |0⟩ dimension: {logical_zero.dim}")  # 128 (2^7)

# Decode
result = code.decode(logical_zero)
print(f"Decoded state: {result.corrected_state}")
```

---

## detect_error

Measure syndrome to detect errors in a quantum circuit.

### Function

| Function | Description |
|----------|-------------|
| `detect_error(circuit, syndrome_qubits)` | Measure syndrome qubits |

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `circuit` | `QuantumCircuit` | Quantum circuit to measure |
| `syndrome_qubits` | `List[int]` | Qubits used for syndrome measurement |

### Returns

| Return | Type | Description |
|--------|------|-------------|
| `syndrome` | `List[int]` | List of syndrome bits |

### Example

```python
from psiqit.error_correction import detect_error
from psiqit.circuits.circuit import QuantumCircuit

circuit = QuantumCircuit(5)
# ... add gates ...

# Measure syndrome on qubits 0, 1, 2
syndrome = detect_error(circuit, syndrome_qubits=[0, 1, 2])
print(f"Syndrome: {syndrome}")
```

---

## Complete Example: Error Correction Comparison

```python
from psiqit.error_correction import BitFlipCode, PhaseFlipCode, ShorCode, SteaneCode
from psiqit.quantum.state import zero, plus
from psiqit.info.entanglement import fidelity

# Create codes
bit_code = BitFlipCode(3)
phase_code = PhaseFlipCode(3)
shor_code = ShorCode()
steane_code = SteaneCode()

# Test states
states = [zero(), plus()]

print("=" * 60)
print("Quantum Error Correction Codes Comparison")
print("=" * 60)

for state in states:
    print(f"\nOriginal state: {state}")
    print("-" * 40)

    # Bit flip code
    encoded = bit_code.encode(state)
    decoded = bit_code.decode(encoded)
    f = fidelity(state, decoded.corrected_state)
    print(f"BitFlipCode:   Fidelity = {f:.4f}, Success = {decoded.success}")

    # Phase flip code
    encoded = phase_code.encode(state)
    decoded = phase_code.decode(encoded)
    f = fidelity(state, decoded.corrected_state)
    print(f"PhaseFlipCode: Fidelity = {f:.4f}, Success = {decoded.success}")

    # Shor code
    encoded = shor_code.encode(state)
    decoded = shor_code.decode(encoded)
    f = fidelity(state, decoded.corrected_state)
    print(f"ShorCode:      Fidelity = {f:.4f}, Success = {decoded.success}")

    # Steane code
    encoded = steane_code.encode(state)
    decoded = steane_code.decode(encoded)
    f = fidelity(state, decoded.corrected_state)
    print(f"SteaneCode:    Fidelity = {f:.4f}, Success = {decoded.success}")
```

### Expected Output:

```
============================================================
Quantum Error Correction Codes Comparison
============================================================

Original state: 1.000|0⟩
----------------------------------------
BitFlipCode:   Fidelity = 1.0000, Success = True
PhaseFlipCode: Fidelity = 1.0000, Success = True
ShorCode:      Fidelity = 1.0000, Success = True
SteaneCode:    Fidelity = 1.0000, Success = True

Original state: 0.707|0⟩ + 0.707|1⟩
----------------------------------------
BitFlipCode:   Fidelity = 1.0000, Success = True
PhaseFlipCode: Fidelity = 1.0000, Success = True
ShorCode:      Fidelity = 1.0000, Success = True
SteaneCode:    Fidelity = 1.0000, Success = True
```

---

## Comparison Table

| Code | # Qubits | Corrects | Type | Distance |
|------|----------|----------|------|----------|
| BitFlipCode | 3 | Bit flips | Repetition | 3 |
| PhaseFlipCode | 3 | Phase flips | Repetition + H | 3 |
| ShorCode | 9 | Any single-qubit error | Concatenated | 3 |
| SteaneCode | 7 | Any single-qubit error | CSS | 3 |

---

## Module Contents

```python
__all__ = [
    'CorrectionResult',
    'BitFlipCode',
    'PhaseFlipCode',
    'ShorCode',
    'SteaneCode',
    'detect_error',
]
```

---

## References

| Code | Reference |
|------|-----------|
| Shor Code | P. W. Shor, "Scheme for reducing decoherence in quantum computer memory," Phys. Rev. A, 1995 |
| Steane Code | A. M. Steane, "Error correcting codes in quantum theory," Phys. Rev. Lett., 1996 |
| CSS Codes | A. R. Calderbank and P. W. Shor, "Good quantum error-correcting codes exist," Phys. Rev. A, 1996 |
| Surface Code | A. Y. Kitaev, "Fault-tolerant quantum computation by anyons," Annals of Physics, 2003 |
