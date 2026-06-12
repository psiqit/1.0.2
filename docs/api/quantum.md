# Quantum Foundations API

## Module: `psiqit.quantum`

This module provides the core quantum mechanics foundations for quantum computing, including quantum states (Ket/Bra), operators (gates and observables), measurements, interference phenomena, and quantum communication parties.

**Submodules:**
- **`state`** - Ket/Bra vectors, quantum states (Bell, GHZ, W, coherent, squeezed, Fock)
- **`operator`** - Quantum gates (Pauli, Hadamard, rotations, CNOT, Toffoli) and observables
- **`measurement`** - Projective measurements, POVM, expectation values, tomography
- **`interference`** - Double-slit experiment, Mach-Zehnder interferometer
- **`parties`** - Alice, Bob, Charlie, BB84 QKD, superdense coding, teleportation

---

## Quantum States (`state.py`)

**Ket and Bra vectors representing quantum states** - Provides state vectors, basis states, entangled states, and continuous variable states.

### Classes

| Class | Description |
|-------|-------------|
| `Ket` | Ket vector |ψ⟩ with amplitudes |
| `Bra` | Bra vector ⟨ψ| (dual of Ket) |

### Ket Methods

| Method | Description |
|--------|-------------|
| `norm()` | Calculate Euclidean norm |
| `normalize()` | Return normalized copy |
| `inner(other)` | Compute ⟨ψ|φ⟩ |
| `outer(other)` | Compute |ψ⟩⟨φ| (density matrix) |
| `measure(shots)` | Perform measurement in computational basis |
| `to_bra()` | Convert to Bra vector |
| `prob(basis_state)` | Probability of specific basis state |
| `sample(shots)` | Sample measurement outcomes |

### Factory Functions

| Function | Description |
|----------|-------------|
| `ket(*amplitudes)` | Create Ket from amplitudes |
| `basis(dim, index)` | Create basis state |index⟩ |

### Single-Qubit States

| Function | Description | Bloch Coordinates |
|----------|-------------|-------------------|
| `zero()` | |0⟩ | (0, 0, 1) |
| `one()` | |1⟩ | (0, 0, -1) |
| `plus()` | |+⟩ = (|0⟩ + |1⟩)/√2 | (1, 0, 0) |
| `minus()` | |-⟩ = (|0⟩ - |1⟩)/√2 | (-1, 0, 0) |
| `ip()` | |i+⟩ = (|0⟩ + i|1⟩)/√2 | (0, 1, 0) |
| `im()` | |i-⟩ = (|0⟩ - i|1⟩)/√2 | (0, -1, 0) |

### Bell States (Maximally Entangled)

| Function | State | Description |
|----------|-------|-------------|
| `bell_phi_plus()` | (|00⟩ + |11⟩)/√2 | |Φ⁺⟩ |
| `bell_phi_minus()` | (|00⟩ - |11⟩)/√2 | |Φ⁻⟩ |
| `bell_psi_plus()` | (|01⟩ + |10⟩)/√2 | |Ψ⁺⟩ |
| `bell_psi_minus()` | (|01⟩ - |10⟩)/√2 | |Ψ⁻⟩ |
| `bell_state(index)` | Get by index (0-3) | Convenience function |

### Multi-Qubit States

| Function | Description |
|----------|-------------|
| `ghz(n)` | GHZ state (|00...0⟩ + |11...1⟩)/√2 |
| `w_state(n)` | W state (equal superposition of states with one |1⟩) |
| `random_state(dim, seed)` | Haar-random pure state |

### Continuous Variable States

| Function | Description |
|----------|-------------|
| `coherent_state(alpha, n_levels)` | Coherent state |α⟩ |
| `squeezed_state(r, phi, n_levels)` | Squeezed vacuum state |
| `fock_state(n, n_levels)` | Fock (number) state |n⟩ |
| `thermal_state(n_bar, n_levels)` | Thermal state (pure state approximation) |
| `phase_state(theta, dim)` | Phase state |θ⟩ = (1/√d) Σ e^{ikθ}|k⟩ |

### Party States (Alice, Bob, Charlie)

| Function | Description |
|----------|-------------|
| `alice_bell_state(index)` | Bell state shared between Alice and Bob |
| `alice_bob_ghz(n)` | GHZ state shared among parties |
| `alice_bob_w(n)` | W state shared among parties |
| `ep_pair()` | EPR pair (|Φ⁺⟩) |

### Utility Functions

| Function | Description |
|----------|-------------|
| `is_orthogonal(psi, phi, tol)` | Check if states are orthogonal |
| `is_same(psi, phi, tol)` | Check if states are identical (up to phase) |
| `fidelity(psi, phi)` | Compute fidelity F = |⟨ψ|φ⟩|² |

### Example 1: Basic States

```python
from psiqit.quantum.state import zero, one, plus, minus

print(zero())   # 1.000|0⟩
print(one())    # 1.000|1⟩
print(plus())   # 0.707|0⟩ + 0.707|1⟩
print(minus())  # 0.707|0⟩ - 0.707|1⟩
Example 2: Bell States
python
from psiqit.quantum.state import bell_phi_plus, bell_state

# Create Bell state |Φ⁺⟩
bell = bell_phi_plus()
print(bell)  # 0.707|00⟩ + 0.707|11⟩

# Get by index
bell = bell_state(0)  # |Φ⁺⟩
bell = bell_state(2)  # |Ψ⁺⟩
Example 3: GHZ and W States
python
from psiqit.quantum.state import ghz, w_state

# 3-qubit GHZ state
ghz3 = ghz(3)
print(ghz3)  # 0.707|000⟩ + 0.707|111⟩

# 3-qubit W state
w3 = w_state(3)
print(w3)    # 0.577|100⟩ + 0.577|010⟩ + 0.577|001⟩
Example 4: Coherent and Squeezed States
python
from psiqit.quantum.state import coherent_state, squeezed_state
import math

# Coherent state |α=1⟩
coherent = coherent_state(alpha=1.0, n_levels=20)
print(f"Coherent state dimension: {coherent.dim}")

# Squeezed vacuum
squeezed = squeezed_state(r=1.0, phi=0, n_levels=20)
Example 5: Ket Operations
python
from psiqit.quantum.state import Ket, zero, one

# Create custom state
psi = Ket([0.6, 0.8])
psi = psi.normalize()
print(psi)  # 0.600|0⟩ + 0.800|1⟩

# Inner product
inner = zero().inner(one())
print(f"⟨0|1⟩ = {inner}")  # 0j

# Outer product (density matrix)
P0 = zero().outer(zero())
print(P0.data)  # [[1, 0], [0, 0]]
Quantum Operators (operator.py)
Quantum gates and observables - Provides unitary operators for quantum gates and Hermitian operators for measurements.

Operator Class
Method	Description
@ (matmul)	Apply to state or compose with another operator
+, -	Add/subtract operators
*	Scalar multiplication
**	Operator power
dagger()	Hermitian conjugate
trace()	Trace of operator
commutator(other)	[A, B] = AB - BA
eigenvalues()	Compute eigenvalues
eigenvectors()	Compute eigenvectors
exp()	Matrix exponential e^A
show(precision)	Pretty-print matrix
Single-Qubit Gates
Function	Matrix	Description
pauli_x()	[[0,1],[1,0]]	NOT gate
pauli_y()	[[0,-i],[i,0]]	Pauli Y
pauli_z()	[[1,0],[0,-1]]	Phase flip
identity()	[[1,0],[0,1]]	Identity
hadamard()	[[1,1],[1,-1]]/√2	Creates superposition
phase(theta)	[[1,0],[0,e^{iθ}]]	Phase shift
s_gate()	phase(π/2)	S gate
t_gate()	phase(π/4)	T gate
rx(theta)	Rotation around X	Rx(θ) = e^{-iθX/2}
ry(theta)	Rotation around Y	Ry(θ) = e^{-iθY/2}
rz(theta)	Rotation around Z	Rz(θ) = e^{-iθZ/2}
Multi-Qubit Gates
Function	Description
cnot()	CNOT (controlled-NOT)
cz()	Controlled-Z
swap()	SWAP gate
toffoli()	Toffoli (CCNOT) gate
fredkin()	Fredkin (CSWAP) gate
Utility Functions
Function	Description
tensor_product(A, B)	A ⊗ B
expectation(operator, state)	⟨ψ	A	ψ⟩
Example 1: Basic Gates
python
from psiqit.quantum.operator import pauli_x, hadamard, cnot
from psiqit.quantum.state import zero

# Apply X gate to |0⟩ → |1⟩
X = pauli_x()
result = X @ zero()
print(result)  # 1.000|1⟩

# Apply Hadamard to |0⟩ → |+⟩
H = hadamard()
result = H @ zero()
print(result)  # 0.707|0⟩ + 0.707|1⟩
Example 2: Rotation Gates
python
from psiqit.quantum.operator import rx, ry, rz
import math

# 90° rotation around X
Rx = rx(math.pi/2)
result = Rx @ zero()
print(result)  # 0.707|0⟩ - 0.707j|1⟩
Example 3: Operator Properties
python
from psiqit.quantum.operator import pauli_x, pauli_z, hadamard

X = pauli_x()
Z = pauli_z()
H = hadamard()

print(f"X unitary: {X.is_unitary}")      # True
print(f"X Hermitian: {X.is_hermitian}")  # True

# Commutator [X, Z] = -2iY
comm = X.commutator(Z)
print(f"[X, Z] = {comm.data}")

# Matrix exponential
exp_X = X.exp()
Measurements (measurement.py)
Quantum measurement tools - Projective measurements, POVM, expectation values, and tomography.

Functions
Function	Description
measure(state, shots, basis)	Projective measurement in computational or custom basis
measure_observable(state, observable, shots)	Measure Hermitian operator
expectation(observable, state)	Compute ⟨ψ	A	ψ⟩
variance(observable, state)	Compute Var(A) = ⟨A²⟩ - ⟨A⟩²
standard_deviation(observable, state)	√Var(A)
born_rule(amplitudes)	Apply Born rule: p =	a	²
measurement_statistics(state, shots)	Comprehensive measurement statistics
state_tomography(data, n_qubits)	Reconstruct density matrix (1 qubit)
POVM Class
Method	Description
__init__(effects, names)	Create POVM from effects
measure(state, shots)	Perform POVM measurement
Predefined POVMs
Function	Description	Outcomes
povm_z_basis()	Z-basis measurement	'0', '1'
povm_x_basis()	X-basis measurement	'+', '-'
povm_y_basis()	Y-basis measurement	'i+', 'i-'
ProjectiveMeasurement Class
Method	Description
from_observable(observable)	Create from Hermitian operator
measure(state, shots)	Perform projective measurement
Example 1: Basic Measurement
python
from psiqit.quantum.measurement import measure
from psiqit.quantum.state import plus

psi = plus()
result = measure(psi, shots=1000)
print(result['counts'])        # {0: 498, 1: 502}
print(result['probabilities']) # [0.5, 0.5]
Example 2: Expectation Value
python
from psiqit.quantum.measurement import expectation
from psiqit.quantum.operator import pauli_z
from psiqit.quantum.state import zero, plus

print(expectation(pauli_z(), zero()))  # 1.0
print(expectation(pauli_z(), plus()))  # 0.0
Example 3: POVM Measurement
python
from psiqit.quantum.measurement import povm_z_basis, povm_x_basis
from psiqit.quantum.state import zero, plus

# Z-basis POVM
povm = povm_z_basis()
result = povm.measure(zero(), shots=1000)
print(result['counts'])  # {'0': 1000, '1': 0}

# X-basis POVM
povm = povm_x_basis()
result = povm.measure(plus(), shots=1000)
print(result['counts'])  # {'+': 1000, '-': 0}
Interference (interference.py)
Quantum interference simulations - Double-slit experiment and Mach-Zehnder interferometer.

Classes
Class	Description
InterferenceResult	Result with x, intensity, visibility, fringe_spacing
DoubleSlit	Double-slit experiment simulation
MachZehnderInterferometer	Mach-Zehnder interferometer
DoubleSlit Methods
Method	Description
intensity(theta)	Intensity at angle θ
pattern(x_range, n_points)	Calculate interference pattern
quantum_probability(position, n_qubits)	Quantum circuit simulation
MachZehnderInterferometer Methods
Method	Description
set_phase(phi)	Set phase shift
get_phase()	Get current phase
quantum_circuit()	Build quantum circuit equivalent
output_probability(port)	Probability at output port
interference_fringe(phase_range, n_points)	Fringe pattern
visibility(phase_range, n_points)	Fringe visibility
Example: Double-Slit Pattern
python
from psiqit.quantum.interference import DoubleSlit
import matplotlib.pyplot as plt

ds = DoubleSlit(slit_distance=1.0, slit_width=0.2,
                wavelength=0.5, distance_to_screen=1.0)

x, intensity, visibility, spacing = ds.pattern(x_range=(-0.1, 0.1))

plt.plot(x, intensity)
plt.xlabel('Position on screen')
plt.ylabel('Intensity')
plt.title(f'Double-Slit Pattern (Visibility={visibility:.3f})')
plt.show()
Quantum Communication Parties (parties.py)
Quantum communication protocols - Alice, Bob, Charlie parties with BB84 QKD, superdense coding, and teleportation.

Classes
Class	Description
Alice	Sender - can prepare, encode, send, measure
Bob	Receiver - can receive, measure
Charlie	Third party
BB84	Quantum Key Distribution protocol
SuperdenseCoding	Send 2 bits with 1 qubit
QuantumTeleportation	Teleport unknown quantum state
Alice Methods
Method	Description
prepare(state)	Prepare quantum state
prepare_random(n_qubits)	Prepare random state
send()	Return prepared state
measure(basis)	Measure in Z or X basis
encode_bit(bit)	Encode bit using X gate
Bob Methods
Method	Description
receive(state)	Receive quantum state
measure(basis)	Measure in Z or X basis
get_bit()	Get last measurement outcome
BB84 Protocol
python
from psiqit.quantum.parties import BB84

# Without eavesdropping
bb84 = BB84(eavesdropping=False)
key = bb84.run(n_bits=100)
print(f"Shared key length: {len(key)}")

# With eavesdropping (detectable)
bb84 = BB84(eavesdropping=True)
key = bb84.run(n_bits=100)
print(f"Key with eavesdropping: {len(key)}")
Complete Example: Bell State Analysis
python
from psiqit.quantum.state import bell_phi_plus
from psiqit.quantum.operator import pauli_z, pauli_x, hadamard
from psiqit.quantum.measurement import expectation, measure
from psiqit.quantum.parties import Alice, Bob

# Create Bell state
bell = bell_phi_plus()
print(f"Bell state: {bell}")

# Measure correlations
Z1 = pauli_z()
Z2 = pauli_z()
correlation = expectation(Z1, bell) * expectation(Z2, bell)
print(f"Correlation: {correlation:.4f}")

# Quantum teleportation simulation
alice = Alice()
bob = Bob()

alice.prepare(bell)
state = alice.send()
bob.receive(state)

print(f"Bob's state: {bob._state}")
Module Contents
python
__all__ = [
    # State
    'Ket', 'Bra', 'ket', 'basis',
    'zero', 'one', 'plus', 'minus', 'ip', 'im',
    'bell_phi_plus', 'bell_phi_minus', 'bell_psi_plus', 'bell_psi_minus', 'bell_state',
    'ghz', 'w_state', 'random_state',
    'coherent_state', 'squeezed_state', 'fock_state', 'thermal_state', 'phase_state',
    'is_orthogonal', 'is_same', 'fidelity',
    # Operator
    'Operator',
    'pauli_x', 'pauli_y', 'pauli_z', 'identity',
    'hadamard', 'phase', 's_gate', 't_gate',
    'rx', 'ry', 'rz',
    'cnot', 'cz', 'swap', 'toffoli', 'fredkin',
    'tensor_product', 'expectation',
    # Measurement
    'measure', 'measure_observable', 'variance', 'standard_deviation',
    'POVM', 'povm_z_basis', 'povm_x_basis', 'povm_y_basis',
    'ProjectiveMeasurement',
    'born_rule', 'measurement_statistics', 'state_tomography',
    # Interference
    'InterferenceResult', 'DoubleSlit', 'MachZehnderInterferometer',
    # Parties
    'Alice', 'Bob', 'Charlie', 'BB84', 'SuperdenseCoding', 'QuantumTeleportation',
]
#References
Topic	Reference
Quantum States	M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information," 2010

Bell States	J. S. Bell, "On the Einstein Podolsky Rosen paradox," Physics, 1964

GHZ State	D. M. Greenberger, M. A. Horne, A. Zeilinger, "Going beyond Bell's theorem," 1989

Coherent States	R. J. Glauber, "Coherent and incoherent states of the radiation field," Phys. Rev., 1963

BB84	C. H. Bennett and G. Brassard, "Quantum cryptography: Public key distribution," 1984

Quantum Teleportation	C. H. Bennett et al., "Teleporting an unknown quantum state," Phys. Rev. Lett., 1993