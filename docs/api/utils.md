# Utilities API

## Module: `psiqit.utils`

This module provides utility functions for quantum computing, including state/operator conversions, polarization optics (Jones calculus), random quantum object generation, and validation routines.

**Submodules:**
- **`conversion`** - Ket ↔ density matrix, basis transformation, Bloch coordinates, Pauli decomposition
- **`polarization`** - Jones vectors/matrices, Stokes parameters, Poincaré sphere
- **`random`** - Random states, density matrices, unitaries, Hermitian operators
- **`validation`** - Unitary, Hermitian, positive semidefinite, density matrix checks

---

## Conversion Utilities (`conversion.py`)

**Convert between different quantum representations** - Ket ↔ density matrix, basis transformation, Bloch coordinates, Pauli decomposition.

### Classes

| Class | Description |
|-------|-------------|
| `ConversionResult` | Result container with success, result, message, details |

### Ket ↔ Density Matrix

| Function | Description |
|----------|-------------|
| `ket_to_density(state)` | Convert |ψ⟩ to ρ = |ψ⟩⟨ψ| |
| `density_to_ket(rho, tol)` | Convert ρ to |ψ⟩ (pure states only) |
| `is_pure_state(rho, tol)` | Check if density matrix represents pure state |

### Basis Transformations

| Function | Description |
|----------|-------------|
| `change_basis(state, basis)` | Express state in different basis |
| `to_computational_basis(state)` | Express in computational basis |
| `to_pauli_basis(matrix)` | Decompose 2x2 matrix into Pauli basis |
| `from_pauli_basis(coeffs)` | Construct matrix from Pauli coefficients |

### Bloch Sphere Conversions

| Function | Description |
|----------|-------------|
| `to_bloch_coordinates(state)` | Single-qubit state → (x, y, z) |
| `from_bloch_coordinates(x, y, z)` | (x, y, z) → quantum state |
| `spherical_to_bloch(theta, phi)` | Spherical → Cartesian |
| `bloch_to_spherical(x, y, z)` | Cartesian → spherical |

### Matrix Representations

| Function | Description |
|----------|-------------|
| `to_list(matrix)` | Convert to list of lists |
| `to_operator(matrix, name)` | Convert to Operator object |
| `to_numpy(matrix)` | Convert to numpy array (if available) |
| `to_ket(amplitudes)` | Convert amplitude list to Ket |
| `to_bra(amplitudes)` | Convert amplitude list to Bra |
| `vector_to_matrix(vec)` | Column vector to matrix |
| `matrix_to_vector(mat)` | Matrix to column vector |

### Example 1: Ket to Density Matrix

```python
from psiqit.utils.conversion import ket_to_density, density_to_ket
from psiqit.quantum.state import zero

# Pure state
psi = zero()
rho = ket_to_density(psi)
print(rho)  # [[1, 0], [0, 0]]

# Convert back (pure state only)
result = density_to_ket(rho)
print(result.success)   # True
print(result.result)    # 1.000|0⟩
Example 2: Pauli Decomposition
python
from psiqit.utils.conversion import to_pauli_basis, from_pauli_basis
from psiqit.quantum.operator import hadamard

# Decompose Hadamard into Pauli basis
H = hadamard()
coeffs = to_pauli_basis(H)
print(coeffs)  # {'I': 0.0, 'X': 0.707, 'Y': 0.0, 'Z': 0.707}

# Reconstruct from coefficients
H_reconstructed = from_pauli_basis(coeffs)
Example 3: Bloch Coordinates
python
from psiqit.utils.conversion import to_bloch_coordinates, from_bloch_coordinates
from psiqit.quantum.state import zero, plus

print(to_bloch_coordinates(zero()))  # (0, 0, 1)
print(to_bloch_coordinates(plus()))  # (1, 0, 0)

# Convert back
psi = from_bloch_coordinates(1, 0, 0)
print(psi)  # 0.707|0⟩ + 0.707|1⟩ (|+⟩)
Polarization Optics (polarization.py)
Jones calculus for polarization optics - Jones vectors for polarization states, Jones matrices for optical elements, Stokes parameters, and Poincaré sphere.

Jones Vectors (Polarization States)
Function	Description	Jones Vector
horizontal()	Horizontally polarized	[1; 0]
vertical()	Vertically polarized	[0; 1]
diagonal(angle)	Linear at given angle	[cosθ; sinθ]
anti_diagonal()	Anti-diagonal	[1/√2; -1/√2]
circular_right()	Right-circular	[1; -i]/√2
circular_left()	Left-circular	[1; i]/√2
elliptical(a, b, delta)	Elliptical	[a; b e^{iδ}]/√(a²+b²)
Jones Matrices (Optical Elements)
Function	Description	Parameters
polarizer(angle)	Linear polarizer	Transmission axis angle
waveplate(retardance, angle)	General waveplate	Retardance, fast axis
quarter_waveplate(angle)	Quarter-wave plate (π/2)	Fast axis angle
half_waveplate(angle)	Half-wave plate (π)	Fast axis angle
full_waveplate(angle)	Full-wave plate (2π)	Fast axis angle
rotator(angle)	Optical rotator	Rotation angle
linear_retarder(delta, angle)	Linear retarder	Retardance, fast axis
circular_dichroism(theta)	Circular dichroism	Rotation angle
Polarization Analysis
Function	Description
stokes_parameters(jones)	Compute (S₀, S₁, S₂, S₃)
degree_of_polarization(jones)	Degree of polarization (0 to 1)
polarization_angle(jones)	Orientation angle (degrees)
ellipticity(jones)	Ellipticity angle (degrees)
Poincaré Sphere
Function	Description
to_poincare(jones)	Jones vector → (x, y, z) on Poincaré sphere
from_poincare(x, y, z)	Poincaré coordinates → Jones vector
Operations
Function	Description
apply_jones(matrix, jones)	Apply Jones matrix to Jones vector
cascade_jones(matrices)	Cascade multiple Jones matrices
is_unitary_jones(matrix, tol)	Check if matrix is unitary (lossless)
Example 1: Polarization States
python
from psiqit.utils.polarization import (
    horizontal, circular_right, quarter_waveplate, apply_jones
)

# Create horizontally polarized light
H = horizontal()
print(f"H = {H[0][0]}, {H[1][0]}")  # 1, 0

# Apply quarter-wave plate at 45° to get circular polarization
QWP = quarter_waveplate(angle=45)
output = apply_jones(QWP, H)
print(f"Output: {output[0][0]:.3f}, {output[1][0]:.3f}")  # 0.707, -0.707j
Example 2: Stokes Parameters
python
from psiqit.utils.polarization import stokes_parameters, degree_of_polarization
from psiqit.utils.polarization import circular_right

# Right-circular polarization
R = circular_right()
S0, S1, S2, S3 = stokes_parameters(R)
print(f"S₀={S0:.1f}, S₁={S1:.1f}, S₂={S2:.1f}, S₃={S3:.1f}")

dop = degree_of_polarization(R)
print(f"DOP = {dop:.1f}")  # 1.0 (fully polarized)
Example 3: Poincaré Sphere
python
from psiqit.utils.polarization import to_poincare, from_poincare
from psiqit.utils.polarization import horizontal, circular_right

# Convert to Poincaré coordinates
x, y, z = to_poincare(horizontal())
print(f"Horizontal: ({x:.1f}, {y:.1f}, {z:.1f})")  # (1.0, 0.0, 0.0)

x, y, z = to_poincare(circular_right())
print(f"Right-circular: ({x:.1f}, {y:.1f}, {z:.1f})")  # (0.0, 0.0, 1.0)

# Convert back
jones = from_poincare(1, 0, 0)
Random Generation (random.py)
Generate random quantum objects - Random states (Haar measure), density matrices, unitaries, Hermitian operators, Pauli rotations.

Classes
Class	Description
RandomResult	Result container with result, seed, type, dimension
Random States
Function	Description
random_state(dim, seed)	Random pure state (Haar measure)
random_qubit_state(seed)	Random single-qubit state
random_n_qubit_state(n_qubits, seed)	Random n-qubit state
Random Density Matrices
Function	Description
random_density_matrix(dim, rank, seed)	Random density matrix (specified rank)
random_qubit_density_matrix(seed)	Random 2x2 density matrix
Random Operators
Function	Description
random_unitary(dim, seed)	Random unitary (Haar measure)
random_hermitian(dim, seed)	Random Hermitian matrix
random_positive_operator(dim, seed)	Random positive semidefinite operator
Random Pauli
Function	Description
random_pauli_rotation(seed)	Random (gate, angle) from {X, Y, Z}
random_pauli_string(n_qubits, seed)	Random Pauli string (e.g., 'XIZY')
Utility
Function	Description
set_random_seed(seed)	Set global random seed
Example 1: Random States
python
from psiqit.utils.random import random_state, random_qubit_state, set_random_seed

# Reproducible random state
set_random_seed(42)
psi1 = random_state(2)
psi2 = random_state(2, seed=42)  # Same as psi1

# Random n-qubit state
psi = random_n_qubit_state(3)
print(f"3-qubit state dimension: {psi.dim}")  # 8
Example 2: Random Density Matrices
python
from psiqit.utils.random import random_density_matrix, random_qubit_density_matrix

# Full rank density matrix
rho = random_density_matrix(4)
print(f"4x4 density matrix created")

# Rank-1 (pure state) density matrix
rho_pure = random_density_matrix(4, rank=1)

# Single qubit
rho = random_qubit_density_matrix()
Example 3: Random Unitaries and Hermitian
python
from psiqit.utils.random import random_unitary, random_hermitian
from psiqit.utils.validation import is_unitary, is_hermitian

U = random_unitary(4)
print(f"Unitary: {is_unitary(U).is_valid}")  # True

H = random_hermitian(4)
print(f"Hermitian: {is_hermitian(H).is_valid}")  # True
Validation Utilities (validation.py)
Validate quantum states, operators, and circuits - Check unitarity, Hermiticity, positive semidefinite, density matrix validity, normalization, orthogonality.

Classes
Class	Description
ValidationResult	Result with is_valid, message, details (supports __bool__)
Matrix Validation
Function	Description
is_unitary(matrix, tol)	Check U†U = I
is_hermitian(matrix, tol)	Check H = H†
is_positive_semidefinite(matrix, tol)	Check all eigenvalues ≥ 0
is_density_matrix(matrix, tol)	Check Hermitian + positive + trace = 1
State Validation
Function	Description
is_normalized(state, tol)	Check ⟨ψ	ψ⟩ = 1
are_orthogonal(state1, state2, tol)	Check ⟨ψ	φ⟩ = 0
are_identical(state1, state2, ignore_phase, tol)	Check if states are identical
fidelity(state1, state2)	Compute	⟨ψ	φ⟩	²
Circuit Validation
Function	Description
validate_circuit(circuit)	Check if valid QuantumCircuit instance
Example 1: Unitary and Hermitian Checks
python
from psiqit.utils.validation import is_unitary, is_hermitian
from psiqit.quantum.operator import hadamard, pauli_z

H = hadamard()
result = is_unitary(H)
print(result)  # ✓ H is unitary
print(result.is_valid)  # True

Z = pauli_z()
result = is_hermitian(Z)
print(result)  # ✓ Z is Hermitian
Example 2: Density Matrix Validation
python
from psiqit.utils.validation import is_density_matrix
from psiqit.quantum.state import zero

# Pure state density matrix
rho = zero().outer(zero())
result = is_density_matrix(rho)
print(result)  # ✓ Density matrix is a valid density matrix

# Invalid (trace != 1)
rho_invalid = [[2, 0], [0, 0]]
result = is_density_matrix(rho_invalid)
print(result)  # ✗ Density matrix is NOT a valid density matrix
Example 3: State Validation
python
from psiqit.utils.validation import is_normalized, are_orthogonal, fidelity
from psiqit.quantum.state import zero, one, plus

print(is_normalized(plus()))  # ✓ State is normalized
print(are_orthogonal(zero(), one()))  # ✓ States are orthogonal
print(fidelity(zero(), plus()))  # 0.5
Complete Example: Random State Generation and Validation
python
from psiqit.utils.random import random_state, set_random_seed
from psiqit.utils.validation import is_normalized, is_hermitian, fidelity
from psiqit.utils.conversion import ket_to_density
from psiqit.quantum.state import bell_phi_plus

set_random_seed(42)

print("=" * 50)
print("Random State Generation and Validation")
print("=" * 50)

# Generate random states
psi1 = random_state(4)  # 2-qubit random state
psi2 = random_state(4)

print(f"\nRandom 2-qubit state: {psi1}")
print(f"Normalized: {is_normalized(psi1).is_valid}")

# Compute fidelity
f = fidelity(psi1, psi2)
print(f"Fidelity between random states: {f:.6f}")

# Convert to density matrix
rho = ket_to_density(psi1)
print(f"Density matrix shape: {len(rho)}x{len(rho[0])}")

# Validate density matrix
from psiqit.utils.validation import is_density_matrix
result = is_density_matrix(rho)
print(f"Valid density matrix: {result.is_valid}")

# Compare with Bell state
bell = bell_phi_plus()
f = fidelity(psi1, bell)
print(f"Fidelity with Bell state: {f:.6f}")
Expected Output:

text
==================================================
Random State Generation and Validation
==================================================

Random 2-qubit state: 0.123|00⟩ + 0.456|01⟩ + ...
Normalized: True
Fidelity between random states: 0.123456
Density matrix shape: 4x4
Valid density matrix: True
Fidelity with Bell state: 0.234567
Module Contents
python
__all__ = [
    # Conversion
    'ConversionResult',
    'ket_to_density', 'density_to_ket', 'is_pure_state',
    'change_basis', 'to_computational_basis',
    'to_pauli_basis', 'from_pauli_basis',
    'to_bloch_coordinates', 'from_bloch_coordinates',
    'to_ket', 'to_bra', 'to_list', 'to_operator', 'to_numpy',
    # Polarization
    'horizontal', 'vertical', 'diagonal', 'anti_diagonal',
    'circular_right', 'circular_left', 'elliptical',
    'polarizer', 'waveplate', 'quarter_waveplate', 'half_waveplate',
    'rotator', 'linear_retarder', 'circular_dichroism',
    'stokes_parameters', 'degree_of_polarization',
    'polarization_angle', 'ellipticity',
    'to_poincare', 'from_poincare',
    'apply_jones', 'cascade_jones', 'is_unitary_jones',
    # Random
    'RandomResult',
    'random_state', 'random_qubit_state', 'random_n_qubit_state',
    'random_density_matrix', 'random_qubit_density_matrix',
    'random_unitary', 'random_hermitian', 'random_positive_operator',
    'random_pauli_rotation', 'random_pauli_string',
    'set_random_seed',
    # Validation
    'ValidationResult',
    'is_unitary', 'is_hermitian', 'is_positive_semidefinite',
    'is_density_matrix', 'is_normalized',
    'are_orthogonal', 'are_identical', 'fidelity',
    'validate_circuit',
]

#References
Topic	Reference
Jones Calculus	R. C. Jones, "A new calculus for the treatment of optical systems," J. Opt. Soc. Am., 1941

Stokes Parameters	G. G. Stokes, "On the composition and resolution of streams of polarized light," 1852

Poincaré Sphere	H. Poincaré, "Théorie mathématique de la lumière," 1892

Haar Measure	F. Mezzadri, "How to generate random matrices from the classical compact groups," Notices of the AMS, 2007