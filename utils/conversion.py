"""
psiqit/utils/conversion.py
============================================================
Quantum State and Operator Conversion Utilities
============================================================

Tools for converting between different representations:
    • Ket ↔ Density matrix
    • Basis transformation
    • Matrix to different representations
    • State vector to Bloch coordinates
    • Pauli decomposition

These conversion utilities are essential for working with different
representations of quantum states and operators in quantum information
processing.

Example:
    >>> from psiqit.utils.conversion import ket_to_density, density_to_ket, to_pauli_basis
    >>> from psiqit.quantum.state import zero, one, plus
    >>> 
    >>> # Convert ket to density matrix
    >>> psi = zero()
    >>> rho = ket_to_density(psi)
    >>> 
    >>> # Convert back (if pure)
    >>> result = density_to_ket(rho)
    >>> psi_recovered = result.result
    >>> 
    >>> # Pauli decomposition
    >>> from psiqit.quantum.operator import hadamard
    >>> coeffs = to_pauli_basis(hadamard())
    >>> print(coeffs)
    >>> 
    >>> # Bloch sphere coordinates
    >>> x, y, z = to_bloch_coordinates(plus())
    >>> print(f"({x:.2f}, {y:.2f}, {z:.2f})")  # (1.00, 0.00, 0.00)

References:
    M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
    Cambridge University Press, 2010.
"""

import math
from typing import List, Union, Tuple, Optional, Dict
from dataclasses import dataclass

from ..math.qalgebra import (
    mul, dagger, trace, eye, conj, transpose, kron,
    outer, inner, norm, normalize, eigenvalues, eigenvectors
)
from ..quantum.state import Ket, Bra
from ..quantum.operator import Operator, pauli_x, pauli_y, pauli_z, identity


@dataclass
class ConversionResult:
    """
    Result container for conversion operations.

    Attributes:
        success: Whether the conversion was successful.
        result: The converted object (if successful).
        message: Human-readable message about the conversion.
        details: Optional dictionary with additional information.

    Examples:
        >>> result = ConversionResult(success=True, result=ket, 
        ...                           message="Conversion successful")
        >>> print(result)
        ✓ Conversion successful
        >>> 
        >>> result = ConversionResult(success=False, result=None,
        ...                           message="Density matrix is mixed")
        >>> print(result)
        ✗ Density matrix is mixed
    """
    success: bool
    result: any
    message: str
    details: Optional[dict] = None
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns:
            String with status indicator and message.
        """
        status = "✓" if self.success else "✗"
        return f"{status} {self.message}"


# ============================================================
# Ket - Density Matrix Conversions
# ============================================================

def ket_to_density(state: Union[Ket, List[complex]]) -> List[List[complex]]:
    """
    Convert a ket state to a density matrix: ρ = |ψ⟩⟨ψ|.

    For a pure state |ψ⟩, the density matrix is the outer product of
    the state with itself.

    Args:
        state: Ket state or list of complex amplitudes.

    Returns:
        Density matrix as a 2D list.

    Examples:
        >>> from psiqit.quantum.state import zero, plus
        >>> rho = ket_to_density(zero())
        >>> print(rho[0][0])  # 1.0
        1.0
        >>> 
        >>> rho = ket_to_density(plus())
        >>> print(f"{rho[0][0]:.2f}")  # 0.50
        0.50
    """
    if isinstance(state, Ket):
        amplitudes = state.data
    else:
        amplitudes = state
    
    return outer(amplitudes, amplitudes)


def density_to_ket(rho: Union[Operator, List[List[complex]]], 
                   tol: float = 1e-10) -> ConversionResult:
    """
    Convert a density matrix to a ket state (only works for pure states).

    For a pure density matrix (rank 1), the corresponding ket is the
    eigenvector with eigenvalue 1.

    Args:
        rho: Density matrix (Operator or list of lists).
        tol: Numerical tolerance.

    Returns:
        ConversionResult containing the ket state if pure, or error message.

    Examples:
        >>> rho = [[1, 0], [0, 0]]
        >>> result = density_to_ket(rho)
        >>> print(result.success)
        True
        >>> print(result.result)  # |0⟩
        1.000|0⟩
        >>> 
        >>> # Mixed state
        >>> rho_mixed = [[0.5, 0], [0, 0.5]]
        >>> result = density_to_ket(rho_mixed)
        >>> print(result.success)
        False
    """
    if isinstance(rho, Operator):
        matrix = rho.data
    else:
        matrix = rho
    
    # Check if pure (rank 1)
    eig_result = eigenvectors(matrix)
    eigenvalues_list = eig_result['values']
    eigenvectors_list = eig_result['vectors']
    
    # Find largest eigenvalue
    max_idx = max(range(len(eigenvalues_list)), 
                  key=lambda i: abs(eigenvalues_list[i]))
    
    purity = abs(eigenvalues_list[max_idx])
    
    if purity < 1 - tol:
        return ConversionResult(
            success=False,
            result=None,
            message="Density matrix is not pure (mixed state)",
            details={'purity': purity}
        )
    
    # Extract eigenvector
    vec = eigenvectors_list[max_idx]
    ket = Ket(vec)
    
    # Normalize
    if not ket.is_normalized:
        ket = ket.normalize()
    
    return ConversionResult(
        success=True,
        result=ket,
        message="Successfully converted to pure state",
        details={'purity': purity}
    )


def is_pure_state(rho: Union[Operator, List[List[complex]]], 
                  tol: float = 1e-10) -> bool:
    """
    Check if a density matrix represents a pure state.

    A state is pure if Tr(ρ²) = 1.

    Args:
        rho: Density matrix (Operator or list of lists).
        tol: Numerical tolerance.

    Returns:
        True if the state is pure, False otherwise.

    Examples:
        >>> psi = zero()
        >>> rho = ket_to_density(psi)
        >>> print(is_pure_state(rho))
        True
        >>> 
        >>> rho_mixed = [[0.5, 0], [0, 0.5]]
        >>> print(is_pure_state(rho_mixed))
        False
    """
    if isinstance(rho, Operator):
        matrix = rho.data
    else:
        matrix = rho
    
    # Compute purity Tr(ρ²)
    rho_sq = mul(matrix, matrix)
    purity = trace(rho_sq).real
    
    return abs(purity - 1.0) < tol


# ============================================================
# Basis Transformations
# ============================================================

def change_basis(state: Union[Ket, List[complex]], 
                 basis: List[Ket]) -> List[complex]:
    """
    Express a quantum state in a different basis.

    Args:
        state: State to transform (Ket or amplitude list).
        basis: List of orthonormal basis states (Kets).

    Returns:
        Coefficients in the new basis.

    Examples:
        >>> from psiqit.quantum.state import zero, plus, minus
        >>> psi = zero()
        >>> x_basis = [plus(), minus()]
        >>> coeffs = change_basis(psi, x_basis)
        >>> print(f"{coeffs[0]:.3f}, {coeffs[1]:.3f}")
        0.707+0.000j, 0.707+0.000j
    """
    if isinstance(state, Ket):
        amplitudes = state.data
    else:
        amplitudes = state
    
    # Convert state to Ket
    ket_state = Ket(amplitudes) if not isinstance(state, Ket) else state
    
    # Project onto each basis state
    coeffs = []
    for b in basis:
        coeff = b.inner(ket_state)
        coeffs.append(coeff)
    
    return coeffs


def to_computational_basis(state: Union[Ket, List[complex]]) -> List[complex]:
    """
    Express a state in the computational basis.

    Since the state is already in the computational basis, this simply
    returns the amplitudes.

    Args:
        state: State (Ket or amplitude list).

    Returns:
        Coefficients in the computational basis.

    Examples:
        >>> from psiqit.quantum.state import plus
        >>> coeffs = to_computational_basis(plus())
        >>> print(f"{coeffs[0]:.3f}, {coeffs[1]:.3f}")
        0.707+0.000j, 0.707+0.000j
    """
    if isinstance(state, Ket):
        return state.data
    return state


def to_pauli_basis(matrix: Union[Operator, List[List[complex]]]) -> Dict[str, float]:
    """
    Decompose a 2x2 matrix into the Pauli basis.

    Any 2x2 matrix A can be written as:
        A = a_I I + a_X X + a_Y Y + a_Z Z
    where a_P = Tr(A·P)/2.

    Args:
        matrix: 2x2 matrix (Operator or list of lists).

    Returns:
        Dictionary with coefficients for 'I', 'X', 'Y', 'Z'.

    Raises:
        ValueError: If the matrix is not 2x2.

    Examples:
        >>> from psiqit.quantum.operator import hadamard
        >>> H = hadamard()
        >>> coeffs = to_pauli_basis(H)
        >>> print(f"I: {coeffs['I']:.3f}, X: {coeffs['X']:.3f}, Z: {coeffs['Z']:.3f}")
        I: 0.000, X: 0.707, Z: 0.707
    """
    if isinstance(matrix, Operator):
        mat = matrix.data
    else:
        mat = matrix
    
    if len(mat) != 2 or len(mat[0]) != 2:
        raise ValueError("Pauli decomposition only for 2x2 matrices")
    
    I = identity().data
    X = pauli_x().data
    Y = pauli_y().data
    Z = pauli_z().data
    
    # Coefficients: a_P = Tr(A·P)/2
    coeff_I = trace(mul(mat, I)).real / 2
    coeff_X = trace(mul(mat, X)).real / 2
    coeff_Y = trace(mul(mat, Y)).real / 2
    coeff_Z = trace(mul(mat, Z)).real / 2
    
    return {
        'I': coeff_I,
        'X': coeff_X,
        'Y': coeff_Y,
        'Z': coeff_Z
    }


def from_pauli_basis(coeffs: Dict[str, float]) -> List[List[complex]]:
    """
    Construct a 2x2 matrix from Pauli basis coefficients.

    Args:
        coeffs: Dictionary with keys 'I', 'X', 'Y', 'Z' and coefficients.

    Returns:
        2x2 matrix as a list of lists.

    Examples:
        >>> coeffs = {'I': 0.5, 'X': 0.5, 'Y': 0, 'Z': 0}
        >>> H = from_pauli_basis(coeffs)
        >>> print(f"{H[0][0]:.3f}, {H[0][1]:.3f}")
        0.500, 0.500
    """
    I = identity().data
    X = pauli_x().data
    Y = pauli_y().data
    Z = pauli_z().data
    
    result = [[0, 0], [0, 0]]
    
    # Add contributions
    for i in range(2):
        for j in range(2):
            result[i][j] = (coeffs.get('I', 0) * I[i][j] +
                           coeffs.get('X', 0) * X[i][j] +
                           coeffs.get('Y', 0) * Y[i][j] +
                           coeffs.get('Z', 0) * Z[i][j])
    
    return result


# ============================================================
# Matrix Representations
# ============================================================

def to_list(matrix: Union[Operator, List[List[complex]]]) -> List[List[complex]]:
    """
    Convert a matrix to a list of lists representation.

    Args:
        matrix: Operator or list of lists.

    Returns:
        Matrix as a list of lists.

    Examples:
        >>> from psiqit.quantum.operator import pauli_x
        >>> mat = to_list(pauli_x())
        >>> print(mat[0][1])  # 1.0
        1.0
    """
    if isinstance(matrix, Operator):
        return matrix.data
    return matrix


def to_operator(matrix: List[List[complex]], name: str = "") -> Operator:
    """
    Convert a matrix to an Operator object.

    Args:
        matrix: List of lists representing the matrix.
        name: Optional name for the operator.

    Returns:
        Operator object.

    Examples:
        >>> X = [[0, 1], [1, 0]]
        >>> op = to_operator(X, name="X")
        >>> print(op.name)
        X
    """
    return Operator(matrix, name)


def to_numpy(matrix: Union[Operator, List[List[complex]]]):
    """
    Convert a matrix to a numpy array (if numpy is available).

    Args:
        matrix: Operator or list of lists.

    Returns:
        Numpy array or None if numpy is not installed.

    Examples:
        >>> arr = to_numpy(pauli_x())
        >>> if arr is not None:
        ...     print(arr.shape)
        (2, 2)
    """
    try:
        import numpy as np
        if isinstance(matrix, Operator):
            return np.array(matrix.data, dtype=complex)
        return np.array(matrix, dtype=complex)
    except ImportError:
        return None


# ============================================================
# Bloch Sphere Conversions
# ============================================================

def to_bloch_coordinates(state: Union[Ket, List[complex]]) -> Tuple[float, float, float]:
    """
    Convert a single-qubit state to Bloch sphere coordinates.

    The Bloch sphere represents a qubit state as:
        |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    with coordinates:
        x = sin θ cos φ, y = sin θ sin φ, z = cos θ

    Args:
        state: Single-qubit state (Ket or amplitude list).

    Returns:
        Tuple (x, y, z) coordinates on the Bloch sphere.

    Raises:
        ValueError: If the state is not a single qubit.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus, ip
        >>> print(to_bloch_coordinates(zero()))  # (0, 0, 1)
        (0.0, 0.0, 1.0)
        >>> print(to_bloch_coordinates(plus()))  # (1, 0, 0)
        (1.0, 0.0, 0.0)
        >>> print(to_bloch_coordinates(ip()))    # (0, 1, 0)
        (0.0, 1.0, 0.0)
    """
    if isinstance(state, Ket):
        if state.dim != 2:
            raise ValueError("Bloch coordinates only for single qubit")
        a, b = state.data[0], state.data[1]
    else:
        if len(state) != 2:
            raise ValueError("Bloch coordinates only for single qubit")
        a, b = state[0], state[1]
    
    # Normalize
    norm_factor = math.sqrt(abs(a)**2 + abs(b)**2)
    if norm_factor > 0:
        a /= norm_factor
        b /= norm_factor
    
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = (abs(a)**2 - abs(b)**2).real
    
    return (x, y, z)


def from_bloch_coordinates(x: float, y: float, z: float) -> Ket:
    """
    Convert Bloch sphere coordinates to a quantum state.

    Args:
        x, y, z: Coordinates on the Bloch sphere (must satisfy x²+y²+z² = 1).

    Returns:
        Ket state.

    Examples:
        >>> psi = from_bloch_coordinates(1, 0, 0)  # |+⟩
        >>> psi = from_bloch_coordinates(0, 1, 0)  # |i+⟩
        >>> psi = from_bloch_coordinates(0, 0, 1)  # |0⟩
    """
    # Check normalization
    r = math.sqrt(x**2 + y**2 + z**2)
    if abs(r - 1.0) > 1e-6:
        # Normalize
        x, y, z = x/r, y/r, z/r
    
    # Spherical coordinates
    theta = math.acos(max(-1, min(1, z)))
    phi = math.atan2(y, x)
    
    # Construct state
    a = math.cos(theta / 2)
    b = math.sin(theta / 2) * complex(math.cos(phi), math.sin(phi))
    
    return Ket([a, b], _normalized=True)


# ============================================================
# Utility Conversions
# ============================================================

def to_ket(amplitudes: List[complex]) -> Ket:
    """
    Convert an amplitude list to a Ket.

    Args:
        amplitudes: List of complex amplitudes.

    Returns:
        Ket state.

    Examples:
        >>> psi = to_ket([1, 0])
        >>> print(psi)
        1.000|0⟩
    """
    return Ket(amplitudes)


def to_bra(amplitudes: List[complex]) -> Bra:
    """
    Convert an amplitude list to a Bra.

    Args:
        amplitudes: List of complex amplitudes.

    Returns:
        Bra vector.

    Examples:
        >>> bra = to_bra([1, 0])
        >>> print(bra)
        ⟨ψ| = [(1+0j), 0j]
    """
    return Bra(amplitudes)


def to_matrix(data: List[List[complex]]) -> List[List[complex]]:
    """
    Ensure data is in matrix format.

    Args:
        data: Input data.

    Returns:
        Matrix as a list of lists.

    Examples:
        >>> mat = to_matrix([[1, 2], [3, 4]])
        >>> print(mat[0][0])
        1
    """
    return data


def vector_to_matrix(vec: List[complex]) -> List[List[complex]]:
    """
    Convert a column vector to matrix form.

    Args:
        vec: Column vector as a list.

    Returns:
        Matrix with one column.

    Examples:
        >>> mat = vector_to_matrix([1, 2])
        >>> print(mat)
        [[1], [2]]
    """
    return [[x] for x in vec]


def matrix_to_vector(mat: List[List[complex]]) -> List[complex]:
    """
    Convert a matrix to a column vector.

    Args:
        mat: Matrix (should have one column or be square).

    Returns:
        Column vector as a list.

    Examples:
        >>> vec = matrix_to_vector([[1], [2]])
        >>> print(vec)
        [1, 2]
    """
    if len(mat[0]) == 1:
        return [row[0] for row in mat]
    return mat


# ============================================================
# Exports
# ============================================================

__all__ = [
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
]