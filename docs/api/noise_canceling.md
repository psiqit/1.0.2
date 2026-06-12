# Noise Canceling API

## Module: `psiqit.noise_canceling`

This module provides quantum noise models and error mitigation techniques for simulating realistic quantum hardware and improving the accuracy of quantum computations on noisy devices.

**Noise Models:**
- Bit flip, phase flip, bit-phase flip channels (Pauli errors)
- Depolarizing channel (uniform Pauli errors)
- Amplitude damping (T1 relaxation)
- Phase damping (T2 dephasing)

**Error Mitigation Techniques:**
- Zero-Noise Extrapolation (ZNE)
- Probabilistic Error Cancellation (PEC)

---

## NoiseResult

**Result container for noise application.**

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `state` | `Operator` or `List[List[complex]]` | Noisy density matrix |
| `fidelity` | `float` | Fidelity with original state (0 to 1) |
| `purity` | `float` | Purity Tr(ρ²) (1/d to 1) |
| `entropy` | `float` | von Neumann entropy |
| `noise_amplitude` | `float` | Noise amplitude parameter (p, γ, etc.) |
| `success_rate` | `float` | Success rate after error cancellation |

### Example

```python
from psiqit.noise_canceling import NoiseResult

result = NoiseResult(
    state=rho_noisy,
    fidelity=0.85,
    purity=0.72,
    entropy=0.43,
    noise_amplitude=0.1,
    success_rate=1.0
)
print(f"Fidelity: {result.fidelity:.4f}")
print(f"Purity: {result.purity:.4f}")
Pauli Noise Channels (Single Qubit)
bit_flip_channel
Bit flip (X) noise channel - Flips the state |0⟩ ↔ |1⟩ with probability p.

Transformation: ρ → (1-p)ρ + p·XρX

phase_flip_channel
Phase flip (Z) noise channel - Applies a phase flip (Z gate) with probability p.

Transformation: ρ → (1-p)ρ + p·ZρZ

bit_phase_flip_channel
Bit-phase flip (Y) noise channel - Applies a Y gate with probability p.

Transformation: ρ → (1-p)ρ + p·YρY

depolarizing_channel
Depolarizing noise channel - Replaces the state with the maximally mixed state I/2 with probability p.

Transformation: ρ → (1-p)ρ + p·I/2

Parameters
Parameter	Type	Description
state	Ket, Operator, or List	Input quantum state
p	float	Error probability (0 to 1)
Returns
Return	Type	Description
NoiseResult	NoiseResult	Result with noisy state and metrics
Example 1: Bit Flip Channel
python
from psiqit.noise_canceling import bit_flip_channel
from psiqit.quantum.state import zero

# Create pure state |0⟩⟨0|
psi = zero()
rho = psi.outer(psi)

# Apply 10% bit flip noise
result = bit_flip_channel(rho, p=0.1)

print(f"Fidelity: {result.fidelity:.4f}")  # 0.9000
print(f"Purity: {result.purity:.4f}")      # 1.0000 (still pure)
print(f"Entropy: {result.entropy:.4f}")    # 0.0000
Example 2: Phase Flip Channel
python
from psiqit.noise_canceling import phase_flip_channel
from psiqit.quantum.state import plus

# Create |+⟩ state
psi = plus()
rho = psi.outer(psi)

# Apply phase flip noise
result = phase_flip_channel(rho, p=0.2)

print(f"Fidelity: {result.fidelity:.4f}")  # 0.8000
Example 3: Depolarizing Channel
python
from psiqit.noise_canceling import depolarizing_channel
from psiqit.quantum.state import zero

psi = zero()
rho = psi.outer(psi)

# Apply depolarizing noise
result = depolarizing_channel(rho, p=0.3)

print(f"Fidelity: {result.fidelity:.4f}")  # 0.7000
print(f"Purity: {result.purity:.4f}")      # 0.7000 (mixed)
print(f"Entropy: {result.entropy:.4f}")    # ~0.5 bits
Amplitude Damping (T1 Relaxation)
amplitude_damping_channel
Amplitude damping channel - Models energy relaxation from |1⟩ to |0⟩ (T1 process). Common in superconducting qubits.

Kraus operators:

K₀ = [[1, 0], [0, √(1-γ)]]

K₁ = [[0, √γ], [0, 0]]

Parameters
Parameter	Type	Description
state	Ket, Operator, or List	Input quantum state
gamma	float	Damping probability (0 to 1)
Example
python
from psiqit.noise_canceling import amplitude_damping_channel
from psiqit.quantum.state import one

# Start in |1⟩ state
psi = one()
rho = psi.outer(psi)

# Apply amplitude damping (T1 relaxation)
result = amplitude_damping_channel(rho, gamma=0.5)

print(f"Fidelity: {result.fidelity:.4f}")  # 0.5000
# State decays to |0⟩ with probability γ
Phase Damping (T2 Dephasing)
phase_damping_channel
Phase damping channel - Models loss of coherence without energy relaxation (T2 process). Reduces off-diagonal elements of the density matrix.

Kraus operators:

K₀ = [[1, 0], [0, √(1-γ)]]

K₁ = [[0, 0], [0, √γ]]

Example
python
from psiqit.noise_canceling import phase_damping_channel
from psiqit.quantum.state import plus

# Start in |+⟩ state (coherent superposition)
psi = plus()
rho = psi.outer(psi)

# Apply phase damping
result = phase_damping_channel(rho, gamma=0.5)

print(f"Fidelity: {result.fidelity:.4f}")  # 0.7500
print(f"Purity: {result.purity:.4f}")      # 0.6250 (mixed)
Noise Canceling Techniques
zero_noise_extrapolation
Zero-Noise Extrapolation (ZNE) - Estimates noise-free fidelity by running the same circuit at different noise levels and extrapolating to zero noise.

Method: Fits fidelity measurements to an exponential decay model: F(p) = F₀·exp(-αp) and extrapolates to p = 0.

Parameters
Parameter	Type	Description
results	List[NoiseResult]	Results at different noise amplitudes
Returns
Return	Type	Description
fidelity	float	Extrapolated fidelity at zero noise
Example
python
from psiqit.noise_canceling import (
    bit_flip_channel, zero_noise_extrapolation
)
from psiqit.quantum.state import zero

psi = zero()
rho = psi.outer(psi)

# Run at different noise levels
results = []
for p in [0.05, 0.10, 0.15]:
    result = bit_flip_channel(rho, p=p)
    results.append(result)

# Extrapolate to zero noise
f0 = zero_noise_extrapolation(results)
print(f"Zero-noise fidelity: {f0:.4f}")  # ~1.0000
probabilistic_error_cancellation
Probabilistic Error Cancellation (PEC) - Mitigates errors by applying inverse operations with certain probabilities.

Parameters
Parameter	Type	Description
noise_result	NoiseResult	Result from a noise channel
error_rate	float	Estimated error rate
mitigation_strength	float	Strength of mitigation (0 to 1)
Returns
Return	Type	Description
NoiseResult	NoiseResult	Mitigated result
Example
python
from psiqit.noise_canceling import (
    bit_flip_channel, probabilistic_error_cancellation
)
from psiqit.quantum.state import zero

psi = zero()
rho = psi.outer(psi)

# Apply noise
noisy_result = bit_flip_channel(rho, p=0.1)

# Apply PEC
mitigated = probabilistic_error_cancellation(
    noisy_result, error_rate=0.1, mitigation_strength=0.8
)

print(f"Original fidelity: {noisy_result.fidelity:.4f}")
print(f"Mitigated fidelity: {mitigated.fidelity:.4f}")
print(f"Success rate: {mitigated.success_rate:.2%}")
Complete Example: Noise Analysis Comparison
python
import matplotlib.pyplot as plt
import numpy as np
from psiqit.noise_canceling import (
    bit_flip_channel, phase_flip_channel, depolarizing_channel,
    amplitude_damping_channel, phase_damping_channel
)
from psiqit.quantum.state import zero, plus

# States to test
states = {
    "|0⟩": zero(),
    "|+⟩": plus(),
}

# Noise channels to test
channels = {
    "Bit Flip": bit_flip_channel,
    "Phase Flip": phase_flip_channel,
    "Depolarizing": depolarizing_channel,
    "Amplitude Damping": amplitude_damping_channel,
    "Phase Damping": phase_damping_channel,
}

# Noise levels
p_values = np.linspace(0, 0.5, 11)

print("=" * 70)
print("Noise Channel Analysis")
print("=" * 70)

for state_name, state in states.items():
    print(f"\nInitial state: {state_name}")
    print("-" * 40)
    
    for channel_name, channel in channels.items():
        fidelities = []
        rho = state.outer(state)
        
        for p in p_values:
            if channel_name in ["Amplitude Damping", "Phase Damping"]:
                result = channel(rho, gamma=p)
            else:
                result = channel(rho, p=p)
            fidelities.append(result.fidelity)
        
        print(f"  {channel_name:18s}: Fidelity at p=0.3 = {fidelities[6]:.4f}")

# Plot comparison
plt.figure(figsize=(12, 6))

for channel_name, channel in channels.items():
    fidelities = []
    rho = zero().outer(zero())
    
    for p in p_values:
        if channel_name in ["Amplitude Damping", "Phase Damping"]:
            result = channel(rho, gamma=p)
        else:
            result = channel(rho, p=p)
        fidelities.append(result.fidelity)
    
    plt.plot(p_values, fidelities, 'o-', linewidth=2, markersize=4, label=channel_name)

plt.xlabel('Noise Parameter (p or γ)')
plt.ylabel('Fidelity')
plt.title('Quantum Noise Channel Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.show()
Expected Output:

text
======================================================================
Noise Channel Analysis
======================================================================

Initial state: |0⟩
----------------------------------------
  Bit Flip         : Fidelity at p=0.3 = 0.7000
  Phase Flip       : Fidelity at p=0.3 = 1.0000
  Depolarizing     : Fidelity at p=0.3 = 0.7000
  Amplitude Damping: Fidelity at p=0.3 = 0.7000
  Phase Damping    : Fidelity at p=0.3 = 1.0000

Initial state: |+⟩
----------------------------------------
  Bit Flip         : Fidelity at p=0.3 = 1.0000
  Phase Flip       : Fidelity at p=0.3 = 0.7000
  Depolarizing     : Fidelity at p=0.3 = 0.7000
  Amplitude Damping: Fidelity at p=0.3 = 0.8500
  Phase Damping    : Fidelity at p=0.3 = 0.8500
Error Mitigation Example: ZNE in Practice
python
from psiqit.noise_canceling import (
    depolarizing_channel, zero_noise_extrapolation
)
from psiqit.quantum.state import bell_phi_plus

# Create Bell state
bell = bell_phi_plus()
rho = bell.outer(bell)

# Measure fidelity at different noise levels
results = []
noise_levels = [0.05, 0.10, 0.15, 0.20]

print("ZNE Mitigation Demo")
print("-" * 40)

for p in noise_levels:
    result = depolarizing_channel(rho, p=p)
    results.append(result)
    print(f"p = {p:.2f}: Fidelity = {result.fidelity:.6f}")

# Extrapolate to zero noise
f0 = zero_noise_extrapolation(results)
print(f"\nExtrapolated zero-noise fidelity: {f0:.6f}")
print(f"Expected ideal fidelity: 1.000000")
print(f"Error: {abs(1 - f0):.6f}")
## Channel Comparison Table

## Channel Comparison Table

| Channel | Kraus Operators | Best For | Effect on &#124;1⟩ | Effect on &#124;+⟩ |
|---------|-----------------|----------|---------------------|---------------------|
| Bit Flip | X | Bit errors | Flips to &#124;0⟩ | No effect |
| Phase Flip | Z | Phase errors | No effect | Flips to &#124;-⟩ |
| Depolarizing | I, X, Y, Z | General noise | Mixed state | Mixed state |
| Amplitude Damping | K₀, K₁ | T1 relaxation | Decays to &#124;0⟩ | Partial decay |
| Phase Damping | K₀, K₁ | T2 dephasing | Coherence loss | Coherence loss |

##Module Contents

python
__all__ = [
    'NoiseResult',
    'bit_flip_channel',
    'phase_flip_channel',
    'bit_phase_flip_channel',
    'depolarizing_channel',
    'amplitude_damping_channel',
    'phase_damping_channel',
    'zero_noise_extrapolation',
    'probabilistic_error_cancellation',
]
## References

| Concept | Reference |
|---------|-----------|
| Quantum noise channels | M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information," Cambridge University Press, 2010 |
| Amplitude/Phase damping | J. Preskill, "Lecture Notes for Physics 229: Quantum Information and Computation," California Institute of Technology, 1998 |
| Zero-Noise Extrapolation | Y. Li and S. C. Benjamin, "Efficient variational quantum simulator incorporating active error minimization," Physical Review X, vol. 7, no. 2, p. 021050, 2017 |
| Probabilistic Error Cancellation | E. van den Berg, Z. K. Minev, A. Kandala, and K. Temme, "Probabilistic error cancellation with sparse Pauli-Lindblad models on noisy quantum processors," arXiv preprint arXiv:2301.05026, 2023 |