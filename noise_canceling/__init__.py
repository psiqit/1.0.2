"""
psiqit/noise_canceling/__init__.py
Quantum noise models and canceling techniques

This module provides tools for modeling quantum noise and mitigating its
effects in quantum circuits. It includes:

    • Noise models:
        - Bit flip channel (X error)
        - Phase flip channel (Z error)
        - Bit-phase flip channel (Y error)
        - Depolarizing channel (uniform Pauli errors)
        - Amplitude damping channel (T1 relaxation)
        - Phase damping channel (T2 dephasing)

    • Noise canceling techniques:
        - Zero-Noise Extrapolation (ZNE)
        - Probabilistic Error Cancellation (PEC)

These tools are essential for simulating realistic quantum hardware and
developing error mitigation strategies for NISQ devices.

Example:
    >>> from psiqit.noise_canceling import bit_flip_channel, depolarizing_channel
    >>> from psiqit.quantum.state import zero
    >>> 
    >>> # Create a pure state
    >>> psi = zero()
    >>> rho = psi.outer(psi)
    >>> 
    >>> # Apply bit flip noise with 10% probability
    >>> result = bit_flip_channel(rho, p=0.1)
    >>> print(f"Fidelity after noise: {result.fidelity:.4f}")

References:
    J. Preskill, "Quantum Computing in the NISQ era and beyond,"
    Quantum, 2:79, 2018.
    E. van den Berg et al., "Probabilistic error cancellation with sparse Pauli-Lindblad
    models on noisy quantum processors," arXiv:2301.05026, 2023.
"""

from .noise_models import (
    NoiseResult,
    bit_flip_channel,
    phase_flip_channel,
    bit_phase_flip_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    zero_noise_extrapolation,
    probabilistic_error_cancellation,
)

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