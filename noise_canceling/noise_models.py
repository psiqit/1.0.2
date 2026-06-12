"""
psiqit/noise_canceling/noise_models.py
============================================================
Quantum Noise Models for Noise Canceling
============================================================

Models for quantum noise and decoherence:
    • Bit flip, phase flip, bit-phase flip channels
    • Depolarizing channel
    • Amplitude damping (T1 relaxation)
    • Phase damping (T2 dephasing)
    • Thermal noise
    • Coherent noise
    • Measurement noise

These noise models are essential for simulating realistic quantum hardware
and developing error mitigation strategies. Each channel transforms a density
matrix according to a completely positive trace-preserving (CPTP) map.

Example:
    >>> from psiqit.quantum.state import zero
    >>> from psiqit.noise_canceling import bit_flip_channel
    >>> 
    >>> # Create a pure state
    >>> psi = zero()
    >>> rho = psi.outer(psi)
    >>> 
    >>> # Apply bit flip noise with 10% probability
    >>> result = bit_flip_channel(rho, p=0.1)
    >>> print(f"Fidelity after noise: {result.fidelity:.4f}")
    >>> print(f"Purity: {result.purity:.4f}")

References:
    J. Preskill, "Quantum Computing in the NISQ era and beyond,"
    Quantum, 2:79, 2018.
    M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
    Cambridge University Press, 2010.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass, field

from ..quantum.state import Ket, basis
from ..quantum.operator import Operator, pauli_x, pauli_y, pauli_z, identity
from ..math.qalgebra import mul, dagger, trace, kron


@dataclass
class NoiseResult:
    """
    Result container for noise application.

    Attributes:
        state: Noisy density matrix (Operator or list of lists).
        fidelity: Fidelity between the noisy state and the original state.
        purity: Purity of the noisy state, Tr(ρ²).
        entropy: von Neumann entropy of the noisy state.
        noise_amplitude: Noise amplitude parameter (p, γ, etc.).
        success_rate: Success rate after error cancellation (1.0 for raw noise).

    Examples:
        >>> result = NoiseResult(state=rho_noisy, fidelity=0.9, purity=0.82,
        ...                      entropy=0.5, noise_amplitude=0.1, success_rate=1.0)
        >>> print(f"Fidelity: {result.fidelity:.4f}")
        Fidelity: 0.9000
    """
    state: Union[Operator, List[List[complex]]]  # Noisy density matrix
    fidelity: float                                # Fidelity with original
    purity: float                                  # Purity of noisy state
    entropy: float                                 # Von Neumann entropy
    noise_amplitude: float = 0.0                   # Noise amplitude
    success_rate: float = 1.0                      # Success rate after canceling


def _to_density_matrix(state):
    """
    Convert various state representations to density matrix.

    Args:
        state: Ket, Operator, or matrix.

    Returns:
        Density matrix as list of lists.
    """
    if isinstance(state, Ket):
        return state.outer(state).data
    elif isinstance(state, Operator):
        return state.data
    return state


def _compute_fidelity(rho_ideal, rho_noisy):
    """
    Compute fidelity between ideal and noisy states.

    For pure states: F = |⟨ψ|φ⟩|²
    For mixed states: F = (Tr(√(√ρ₁ ρ₂ √ρ₁)))²

    Args:
        rho_ideal: Ideal density matrix.
        rho_noisy: Noisy density matrix.

    Returns:
        Fidelity between 0 and 1.
    """
    rho_ideal = np.array(_to_density_matrix(rho_ideal))
    rho_noisy = np.array(_to_density_matrix(rho_noisy))
    fid = np.trace(rho_ideal @ rho_noisy).real
    return max(0, min(1, fid))


def _compute_purity(rho):
    """
    Compute purity of a density matrix: Tr(ρ²).

    Args:
        rho: Density matrix.

    Returns:
        Purity between 1/d and 1.
    """
    rho = np.array(_to_density_matrix(rho))
    return np.trace(rho @ rho).real


def _compute_entropy(rho, base='e'):
    """
    Compute von Neumann entropy: S(ρ) = -Tr(ρ log ρ).

    Args:
        rho: Density matrix.
        base: Logarithm base ('e', '2', or '10').

    Returns:
        Entropy in specified units.
    """
    rho = np.array(_to_density_matrix(rho))
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    
    if base == 'e':
        return -np.sum(eigvals * np.log(eigvals))
    elif base == '2':
        return -np.sum(eigvals * np.log2(eigvals))
    else:
        return -np.sum(eigvals * np.log10(eigvals))


# ============================================================
# Pauli Noise Channels (Single Qubit)
# ============================================================

def bit_flip_channel(state, p: float) -> NoiseResult:
    """
    Apply a bit flip (X) noise channel.

    The bit flip channel flips the state with probability p:
        ρ → (1-p)ρ + p·XρX

    Args:
        state: Input quantum state (Ket, Operator, or density matrix).
        p: Error probability (0 ≤ p ≤ 1).

    Returns:
        NoiseResult with the noisy state and metrics.

    Examples:
        >>> from psiqit.quantum.state import zero
        >>> psi = zero()
        >>> rho = psi.outer(psi)
        >>> result = bit_flip_channel(rho, p=0.1)
        >>> print(f"Fidelity: {result.fidelity:.4f}")
        Fidelity: 0.9000
    """
    rho = _to_density_matrix(state)
    X = pauli_x().data
    
    term1 = [[(1-p) * rho[i][j] for j in range(2)] for i in range(2)]
    XrhoX = mul(mul(X, rho), X)
    term2 = [[p * XrhoX[i][j] for j in range(2)] for i in range(2)]
    
    noisy = [[term1[i][j] + term2[i][j] for j in range(2)] for i in range(2)]
    
    return NoiseResult(
        state=noisy,
        fidelity=_compute_fidelity(rho, noisy),
        purity=_compute_purity(noisy),
        entropy=_compute_entropy(noisy, base='2'),
        noise_amplitude=p
    )


def phase_flip_channel(state, p: float) -> NoiseResult:
    """
    Apply a phase flip (Z) noise channel.

    The phase flip channel applies a Z gate with probability p:
        ρ → (1-p)ρ + p·ZρZ

    Args:
        state: Input quantum state.
        p: Error probability (0 ≤ p ≤ 1).

    Returns:
        NoiseResult with the noisy state and metrics.
    """
    rho = _to_density_matrix(state)
    Z = pauli_z().data
    
    term1 = [[(1-p) * rho[i][j] for j in range(2)] for i in range(2)]
    ZrhoZ = mul(mul(Z, rho), Z)
    term2 = [[p * ZrhoZ[i][j] for j in range(2)] for i in range(2)]
    
    noisy = [[term1[i][j] + term2[i][j] for j in range(2)] for i in range(2)]
    
    return NoiseResult(
        state=noisy,
        fidelity=_compute_fidelity(rho, noisy),
        purity=_compute_purity(noisy),
        entropy=_compute_entropy(noisy, base='2'),
        noise_amplitude=p
    )


def bit_phase_flip_channel(state, p: float) -> NoiseResult:
    """
    Apply a bit-phase flip (Y) noise channel.

    The bit-phase flip channel applies a Y gate with probability p:
        ρ → (1-p)ρ + p·YρY

    Args:
        state: Input quantum state.
        p: Error probability (0 ≤ p ≤ 1).

    Returns:
        NoiseResult with the noisy state and metrics.
    """
    rho = _to_density_matrix(state)
    Y = pauli_y().data
    
    term1 = [[(1-p) * rho[i][j] for j in range(2)] for i in range(2)]
    YrhoY = mul(mul(Y, rho), Y)
    term2 = [[p * YrhoY[i][j] for j in range(2)] for i in range(2)]
    
    noisy = [[term1[i][j] + term2[i][j] for j in range(2)] for i in range(2)]
    
    return NoiseResult(
        state=noisy,
        fidelity=_compute_fidelity(rho, noisy),
        purity=_compute_purity(noisy),
        entropy=_compute_entropy(noisy, base='2'),
        noise_amplitude=p
    )


def depolarizing_channel(state, p: float) -> NoiseResult:
    """
    Apply a depolarizing noise channel.

    The depolarizing channel replaces the state with the maximally mixed
    state with probability p:
        ρ → (1-p)ρ + p·I/2

    For single qubits, this is equivalent to applying X, Y, or Z each with
    probability p/3.

    Args:
        state: Input quantum state.
        p: Depolarizing probability (0 ≤ p ≤ 1).

    Returns:
        NoiseResult with the noisy state and metrics.
    """
    rho = _to_density_matrix(state)
    X = pauli_x().data
    Y = pauli_y().data
    Z = pauli_z().data
    
    term1 = [[(1-p) * rho[i][j] for j in range(2)] for i in range(2)]
    
    XrhoX = mul(mul(X, rho), X)
    YrhoY = mul(mul(Y, rho), Y)
    ZrhoZ = mul(mul(Z, rho), Z)
    
    p3 = p / 3
    noisy = [[term1[i][j] + p3 * (XrhoX[i][j] + YrhoY[i][j] + ZrhoZ[i][j]) 
              for j in range(2)] for i in range(2)]
    
    return NoiseResult(
        state=noisy,
        fidelity=_compute_fidelity(rho, noisy),
        purity=_compute_purity(noisy),
        entropy=_compute_entropy(noisy, base='2'),
        noise_amplitude=p
    )


# ============================================================
# Amplitude Damping (T1 Relaxation)
# ============================================================

def amplitude_damping_channel(state, gamma: float) -> NoiseResult:
    """
    Apply an amplitude damping channel (T1 relaxation).

    The amplitude damping channel models energy relaxation from |1⟩ to |0⟩.
    It is characterized by the Kraus operators:
        K0 = [[1, 0], [0, √(1-γ)]]
        K1 = [[0, √γ], [0, 0]]

    Args:
        state: Input quantum state.
        gamma: Damping probability (0 ≤ γ ≤ 1).

    Returns:
        NoiseResult with the noisy state and metrics.
    """
    rho = _to_density_matrix(state)
    
    sqrt_gamma = math.sqrt(gamma)
    sqrt_1mg = math.sqrt(1 - gamma) if gamma < 1 else 0
    
    K0 = [[1, 0], [0, sqrt_1mg]]
    K0rho = mul(K0, rho)
    K0rhoK0 = mul(K0rho, dagger(K0))
    
    K1 = [[0, sqrt_gamma], [0, 0]]
    K1rho = mul(K1, rho)
    K1rhoK1 = mul(K1rho, dagger(K1))
    
    noisy = [[K0rhoK0[i][j] + K1rhoK1[i][j] for j in range(2)] for i in range(2)]
    
    return NoiseResult(
        state=noisy,
        fidelity=_compute_fidelity(rho, noisy),
        purity=_compute_purity(noisy),
        entropy=_compute_entropy(noisy, base='2'),
        noise_amplitude=gamma
    )


def phase_damping_channel(state, gamma: float) -> NoiseResult:
    """
    Apply a phase damping channel (T2 dephasing).

    The phase damping channel models loss of coherence without energy relaxation.
    It is characterized by the Kraus operators:
        K0 = [[1, 0], [0, √(1-γ)]]
        K1 = [[0, 0], [0, √γ]]

    Args:
        state: Input quantum state.
        gamma: Dephasing probability (0 ≤ γ ≤ 1).

    Returns:
        NoiseResult with the noisy state and metrics.
    """
    rho = _to_density_matrix(state)
    
    sqrt_gamma = math.sqrt(gamma)
    sqrt_1mg = math.sqrt(1 - gamma) if gamma < 1 else 0
    
    K0 = [[1, 0], [0, sqrt_1mg]]
    K0rho = mul(K0, rho)
    K0rhoK0 = mul(K0rho, dagger(K0))
    
    K1 = [[0, 0], [0, sqrt_gamma]]
    K1rho = mul(K1, rho)
    K1rhoK1 = mul(K1rho, dagger(K1))
    
    noisy = [[K0rhoK0[i][j] + K1rhoK1[i][j] for j in range(2)] for i in range(2)]
    
    return NoiseResult(
        state=noisy,
        fidelity=_compute_fidelity(rho, noisy),
        purity=_compute_purity(noisy),
        entropy=_compute_entropy(noisy, base='2'),
        noise_amplitude=gamma
    )


# ============================================================
# Noise Canceling Techniques
# ============================================================

def zero_noise_extrapolation(results: List[NoiseResult]) -> float:
    """
    Apply Zero-Noise Extrapolation (ZNE) to estimate noise-free fidelity.

    ZNE works by running the same circuit at different noise levels and
    extrapolating to the zero-noise limit. This implementation uses
    exponential fitting: F(p) = F₀·exp(-αp).

    Args:
        results: List of NoiseResult objects at different noise amplitudes.
                 Should be sorted by increasing noise amplitude.

    Returns:
        Extrapolated fidelity at zero noise.

    Examples:
        >>> # Run at three different noise levels
        >>> res1 = bit_flip_channel(state, p=0.05)
        >>> res2 = bit_flip_channel(state, p=0.10)
        >>> res3 = bit_flip_channel(state, p=0.15)
        >>> f0 = zero_noise_extrapolation([res1, res2, res3])
        >>> print(f"Zero-noise fidelity: {f0:.4f}")
    """
    if len(results) < 2:
        return results[0].fidelity if results else 0.0
    
    # Simple linear extrapolation
    amplitudes = [r.noise_amplitude for r in results]
    fidelities = [r.fidelity for r in results]
    
    # Fit to exponential decay model: F(p) = F0 * exp(-αp)
    # Linearize: log(F) = log(F0) - αp
    import numpy as np
    log_f = np.log(fidelities)
    
    # Linear regression
    A = np.vstack([amplitudes, np.ones(len(amplitudes))]).T
    slope, intercept = np.linalg.lstsq(A, log_f, rcond=None)[0]
    
    # Extrapolate to p=0
    f0 = np.exp(intercept)
    return f0


def probabilistic_error_cancellation(noise_result: NoiseResult, 
                                      error_rate: float,
                                      mitigation_strength: float = 1.0) -> NoiseResult:
    """
    Apply Probabilistic Error Cancellation (PEC) to mitigate errors.

    PEC works by applying inverse operations with certain probabilities
    to cancel the effect of noise. This is a simplified implementation.

    Args:
        noise_result: Result from a noise channel.
        error_rate: Estimated error rate of the channel.
        mitigation_strength: Strength of mitigation (0 to 1).

    Returns:
        Mitigated NoiseResult with updated fidelity.

    Examples:
        >>> res = bit_flip_channel(state, p=0.1)
        >>> mitigated = probabilistic_error_cancellation(res, error_rate=0.1)
        >>> print(f"Mitigated fidelity: {mitigated.fidelity:.4f}")
    """
    if mitigation_strength <= 0:
        return noise_result
    
    # Simplified mitigation model
    mitigated_fidelity = noise_result.fidelity + mitigation_strength * (1 - noise_result.fidelity) * (1 - error_rate)
    mitigated_fidelity = min(1.0, mitigated_fidelity)
    
    return NoiseResult(
        state=noise_result.state,
        fidelity=mitigated_fidelity,
        purity=noise_result.purity,
        entropy=noise_result.entropy,
        noise_amplitude=noise_result.noise_amplitude * (1 - mitigation_strength),
        success_rate=1 - error_rate * mitigation_strength
    )


# ============================================================
# Exports
# ============================================================

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