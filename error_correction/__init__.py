"""
psiqit/error_correction/__init__.py
Quantum error correction codes

This module provides implementations of quantum error correction codes,
which are essential for protecting quantum information against decoherence
and operational errors in quantum computers.

Quantum error correction works by encoding logical qubits into multiple
physical qubits, allowing detection and correction of errors without
measuring the quantum information directly.

Available codes:
    - BitFlipCode: 3-qubit repetition code for bit flip errors
    - PhaseFlipCode: 3-qubit code for phase flip errors (using Hadamard transform)
    - ShorCode: 9-qubit code that corrects any single-qubit error
    - SteaneCode: 7-qubit CSS code (Calderbank-Shor-Steane)

Example:
    >>> from psiqit.error_correction import BitFlipCode
    >>> from psiqit.quantum.state import zero
    >>> 
    >>> # Encode a logical |0⟩ state
    >>> code = BitFlipCode(n=3)
    >>> encoded = code.encode(zero())
    >>> 
    >>> # Decode and correct errors
    >>> result = code.decode(encoded)
    >>> print(f"Corrected state: {result.corrected_state}")

References:
    P. W. Shor, "Scheme for reducing decoherence in quantum computer memory,"
    Physical Review A, 52(4):R2493, 1995.
    A. M. Steane, "Error correcting codes in quantum theory,"
    Physical Review Letters, 77(5):793, 1996.
"""

from .codes import (
    CorrectionResult,
    BitFlipCode,
    PhaseFlipCode,
    ShorCode,
    SteaneCode,
    detect_error,
)

__all__ = [
    'CorrectionResult',
    'BitFlipCode',
    'PhaseFlipCode',
    'ShorCode',
    'SteaneCode',
    'detect_error',
]