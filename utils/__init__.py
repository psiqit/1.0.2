"""
psiqit/utils/__init__.py
============================================================
Utilities Package - Quantum State and Operator Utilities
============================================================

This package provides utility functions for quantum computing:
    • State and operator conversions
    • Polarization optics (Jones calculus)
    • Random quantum object generation
    • Validation and checking routines

The utils module is organized into four submodules:

    - conversion: Convert between different representations (Ket ↔ density matrix,
                  basis transformation, Bloch coordinates, Pauli decomposition)
    - polarization: Jones calculus for polarization optics (Jones vectors,
                    Jones matrices, Stokes parameters, Poincaré sphere)
    - random: Generate random quantum objects (states, density matrices,
              unitaries, Hermitian operators, Pauli rotations)
    - validation: Validate quantum objects (unitary, Hermitian, positive
                  semidefinite, density matrix, normalization, orthogonality)

Example:
    >>> from psiqit.utils import is_unitary, random_state, ket_to_density, fidelity
    >>> from psiqit.quantum.state import zero, plus
    >>> from psiqit.quantum.operator import hadamard
    >>> 
    >>> # Validation
    >>> result = is_unitary(hadamard())
    >>> print(result.is_valid)  # True
    >>> 
    >>> # Random generation
    >>> psi = random_state(2)
    >>> 
    >>> # Conversions
    >>> rho = ket_to_density(zero())
    >>> 
    >>> # Fidelity between states
    >>> f = fidelity(zero(), plus())
    >>> print(f"{f:.4f}")  # 0.5000
"""

# ============================================================
# Conversion Utilities
# ============================================================

from .conversion import (
    ConversionResult,
    ket_to_density,
    density_to_ket,
    is_pure_state,
    change_basis,
    to_computational_basis,
    to_pauli_basis,
    from_pauli_basis,
    to_list,
    to_operator,
    to_numpy,
    to_bloch_coordinates,
    from_bloch_coordinates,
    to_ket,
    to_bra,
    to_matrix,
    vector_to_matrix,
    matrix_to_vector,
)

# ============================================================
# Polarization Optics (Jones Calculus)
# ============================================================

from .polarization import (
    # Jones vectors
    horizontal,
    vertical,
    diagonal,
    anti_diagonal,
    circular_right,
    circular_left,
    elliptical,
    # Jones matrices
    polarizer,
    waveplate,
    quarter_waveplate,
    half_waveplate,
    full_waveplate,
    rotator,
    linear_retarder,
    circular_dichroism,
    # Analysis
    stokes_parameters,
    degree_of_polarization,
    polarization_angle,
    ellipticity,
    # Poincaré sphere
    to_poincare,
    from_poincare,
    # Operations
    apply_jones,
    cascade_jones,
    is_unitary_jones,
)

# ============================================================
# Random Generation Utilities
# ============================================================

from .random import (
    RandomResult,
    random_state,
    random_qubit_state,
    random_n_qubit_state,
    random_density_matrix,
    random_qubit_density_matrix,
    random_unitary,
    random_hermitian,
    random_positive_operator,
    random_pauli_rotation,
    random_pauli_string,
    set_random_seed,
)

# ============================================================
# Validation Utilities
# ============================================================

from .validation import (
    ValidationResult,
    is_unitary,
    is_hermitian,
    is_positive_semidefinite,
    is_density_matrix,
    is_normalized,
    are_orthogonal,
    are_identical,
    fidelity,
    validate_circuit,
)

# ============================================================
# Package Metadata
# ============================================================

__version__ = "1.0.0"
__author__ = "PSIQIT Contributors"

__all__ = [
    # Conversion
    'ConversionResult',
    'ket_to_density',
    'density_to_ket',
    'is_pure_state',
    'change_basis',
    'to_computational_basis',
    'to_pauli_basis',
    'from_pauli_basis',
    'to_list',
    'to_operator',
    'to_numpy',
    'to_bloch_coordinates',
    'from_bloch_coordinates',
    'to_ket',
    'to_bra',
    'to_matrix',
    'vector_to_matrix',
    'matrix_to_vector',
    # Polarization
    'horizontal',
    'vertical',
    'diagonal',
    'anti_diagonal',
    'circular_right',
    'circular_left',
    'elliptical',
    'polarizer',
    'waveplate',
    'quarter_waveplate',
    'half_waveplate',
    'full_waveplate',
    'rotator',
    'linear_retarder',
    'circular_dichroism',
    'stokes_parameters',
    'degree_of_polarization',
    'polarization_angle',
    'ellipticity',
    'to_poincare',
    'from_poincare',
    'apply_jones',
    'cascade_jones',
    'is_unitary_jones',
    # Random
    'RandomResult',
    'random_state',
    'random_qubit_state',
    'random_n_qubit_state',
    'random_density_matrix',
    'random_qubit_density_matrix',
    'random_unitary',
    'random_hermitian',
    'random_positive_operator',
    'random_pauli_rotation',
    'random_pauli_string',
    'set_random_seed',
    # Validation
    'ValidationResult',
    'is_unitary',
    'is_hermitian',
    'is_positive_semidefinite',
    'is_density_matrix',
    'is_normalized',
    'are_orthogonal',
    'are_identical',
    'fidelity',
    'validate_circuit',
]