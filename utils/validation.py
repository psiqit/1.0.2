"""
psiqit/utils/validation.py
============================================================
Quantum Validation Utilities
============================================================

Tools for validating quantum states, operators, and circuits:
    • Unitary checking
    • Hermitian checking
    • Positive semidefinite checking
    • Density matrix validation
    • State normalization
    • Orthogonality checking
    • State identity (up to global phase)

These validation tools are essential for debugging quantum circuits,
verifying theoretical predictions, and ensuring that quantum objects
satisfy required properties.

Example:
    >>> from psiqit.utils.validation import is_unitary, is_hermitian, is_density_matrix
    >>> from psiqit.quantum.operator import hadamard, pauli_z
    >>> from psiqit.quantum.state import zero, one
    >>> 
    >>> # Check if Hadamard is unitary
    >>> H = hadamard()
    >>> print(is_unitary(H))  # ✓ Unitary matrix
    >>> 
    >>> # Check if Pauli-Z is Hermitian
    >>> Z = pauli_z()
    >>> print(is_hermitian(Z))  # ✓ Hermitian matrix
    >>> 
    >>> # Check if |0⟩⟨0| is a valid density matrix
    >>> rho = zero().outer(zero())
    >>> print(is_density_matrix(rho))  # ✓ Valid density matrix
    >>> 
    >>> # Check orthogonality
    >>> print(are_orthogonal(zero(), one()))  # ✓ States are orthogonal

References:
    M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
    Cambridge University Press, 2010.
"""

import math
from typing import List, Union, Tuple, Optional
from dataclasses import dataclass

from ..math.qalgebra import (
    mul, dagger, trace, eye, conj, transpose,
    is_unitary as _is_unitary,
    is_hermitian as _is_hermitian,
    is_positive as _is_positive,
    eigenvalues
)
from ..quantum.state import Ket, Bra
from ..quantum.operator import Operator


@dataclass
class ValidationResult:
    """
    Result container for validation checks.

    Attributes:
        is_valid: Whether the validation passed.
        message: Human-readable message describing the result.
        details: Optional dictionary with additional information
                (e.g., eigenvalues, error values).

    Examples:
        >>> result = ValidationResult(is_valid=True, message="Matrix is unitary")
        >>> print(result)
        ✓ Matrix is unitary
        >>> 
        >>> result = ValidationResult(is_valid=False, message="Not Hermitian",
        ...                           details={'error': 'Entry (0,1) mismatch'})
        >>> print(result)
        ✗ Not Hermitian
    """
    is_valid: bool
    message: str
    details: Optional[dict] = None
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns:
            String with status indicator and message.
        """
        status = "✓" if self.is_valid else "✗"
        return f"{status} {self.message}"
    
    def __bool__(self) -> bool:
        """
        Allow using ValidationResult directly in boolean contexts.

        Returns:
            True if validation passed, False otherwise.

        Examples:
            >>> result = is_unitary(hadamard())
            >>> if result:
            ...     print("Unitary!")
            Unitary!
        """
        return self.is_valid


# ============================================================
# Matrix Validation
# ============================================================

def is_unitary(matrix: Union[Operator, List[List[complex]]], 
               tol: float = 1e-10) -> ValidationResult:
    """
    Check if a matrix is unitary: U†U = I.

    A unitary matrix preserves the inner product and represents
    reversible quantum operations (gates).

    Args:
        matrix: Operator or matrix to check.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.operator import hadamard, pauli_x, cnot
        >>> 
        >>> # Hadamard is unitary
        >>> result = is_unitary(hadamard())
        >>> print(result)
        ✓ H is unitary
        >>> 
        >>> # CNOT is unitary
        >>> result = is_unitary(cnot())
        >>> print(result.is_valid)
        True
    """
    # Convert to matrix
    if isinstance(matrix, Operator):
        mat = matrix.data
        name = matrix.name
    else:
        mat = matrix
        name = "Matrix"
    
    n = len(mat)
    I = eye(n)
    
    # Check U†U = I
    U_dag = dagger(mat)
    product = mul(U_dag, mat)
    
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(product[i][j] - expected) > tol:
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} is NOT unitary",
                    details={
                        'error': f"Entry ({i},{j}) = {product[i][j]}, expected {expected}",
                        'norm': trace(product).real
                    }
                )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name} is unitary",
        details={'trace': trace(product).real, 'dimension': n}
    )


def is_hermitian(matrix: Union[Operator, List[List[complex]]], 
                 tol: float = 1e-10) -> ValidationResult:
    """
    Check if a matrix is Hermitian: H = H†.

    Hermitian matrices represent physical observables (measurable quantities)
    and have real eigenvalues.

    Args:
        matrix: Operator or matrix to check.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.operator import pauli_z, hadamard
        >>> 
        # Pauli-Z is Hermitian
        >>> result = is_hermitian(pauli_z())
        >>> print(result)
        ✓ Z is Hermitian
        >>> 
        # Hadamard is Hermitian
        >>> result = is_hermitian(hadamard())
        >>> print(result)
        ✓ H is Hermitian
    """
    if isinstance(matrix, Operator):
        mat = matrix.data
        name = matrix.name
    else:
        mat = matrix
        name = "Matrix"
    
    H_dag = dagger(mat)
    
    for i in range(len(mat)):
        for j in range(len(mat)):
            if abs(mat[i][j] - H_dag[i][j]) > tol:
                return ValidationResult(
                    is_valid=False,
                    message=f"{name} is NOT Hermitian",
                    details={'error': f"Entry ({i},{j}) differs: {mat[i][j]} ≠ {H_dag[i][j]}"}
                )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name} is Hermitian",
        details={'dimension': len(mat)}
    )


def is_positive_semidefinite(matrix: Union[Operator, List[List[complex]]], 
                              tol: float = 1e-10) -> ValidationResult:
    """
    Check if a matrix is positive semidefinite (all eigenvalues ≥ 0).

    Positive semidefinite matrices are used to represent quantum states
    (density matrices) and POVM elements.

    Args:
        matrix: Operator or matrix to check.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.state import zero
        >>> rho = zero().outer(zero())  # |0⟩⟨0|
        >>> result = is_positive_semidefinite(rho)
        >>> print(result.is_valid)
        True
    """
    if isinstance(matrix, Operator):
        mat = matrix.data
        name = matrix.name
    else:
        mat = matrix
        name = "Matrix"
    
    # First check Hermitian
    hermitian_check = is_hermitian(mat, tol)
    if not hermitian_check.is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"{name} is NOT positive semidefinite (not Hermitian)",
            details=hermitian_check.details
        )
    
    # Check eigenvalues
    eigvals = eigenvalues(mat)
    min_eig = min(v.real for v in eigvals)
    
    if min_eig < -tol:
        negative_eigs = [v for v in eigvals if v.real < -tol]
        return ValidationResult(
            is_valid=False,
            message=f"{name} is NOT positive semidefinite",
            details={'min_eigenvalue': min_eig, 'negative_count': len(negative_eigs)}
        )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name} is positive semidefinite",
        details={'min_eigenvalue': min_eig, 'max_eigenvalue': max(v.real for v in eigvals)}
    )


def is_density_matrix(matrix: Union[Operator, List[List[complex]]], 
                      tol: float = 1e-10) -> ValidationResult:
    """
    Check if a matrix is a valid density matrix.

    A valid density matrix must satisfy:
        1. Hermitian: ρ† = ρ
        2. Positive semidefinite: all eigenvalues ≥ 0
        3. Trace = 1: Tr(ρ) = 1

    Args:
        matrix: Operator or matrix to check.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.state import zero, one
        >>> 
        >>> # Pure state density matrix
        >>> rho = zero().outer(zero())
        >>> result = is_density_matrix(rho)
        >>> print(result)
        ✓ Density matrix is a valid density matrix
        >>> 
        >>> # Maximally mixed state
        >>> from psiqit.quantum.operator import identity
        >>> rho = identity(2).data
        >>> for i in range(2):
        ...     for j in range(2):
        ...         rho[i][j] /= 2
        >>> result = is_density_matrix(rho)
        >>> print(result.is_valid)
        True
    """
    if isinstance(matrix, Operator):
        mat = matrix.data
        name = matrix.name
    else:
        mat = matrix
        name = "Density matrix"
    
    # Check Hermitian
    hermitian_check = is_hermitian(mat, tol)
    if not hermitian_check.is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"{name} is NOT a valid density matrix",
            details={'reason': 'Not Hermitian'}
        )
    
    # Check positive semidefinite
    positive_check = is_positive_semidefinite(mat, tol)
    if not positive_check.is_valid:
        return ValidationResult(
            is_valid=False,
            message=f"{name} is NOT a valid density matrix",
            details={'reason': 'Not positive semidefinite'}
        )
    
    # Check trace = 1
    tr = trace(mat)
    if abs(tr - 1.0) > tol:
        return ValidationResult(
            is_valid=False,
            message=f"{name} is NOT a valid density matrix",
            details={'reason': f'Trace = {tr}, expected 1'}
        )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name} is a valid density matrix",
        details={'trace': tr, 'dimension': len(mat)}
    )


# ============================================================
# State Validation
# ============================================================

def is_normalized(state: Union[Ket, List[complex]], tol: float = 1e-10) -> ValidationResult:
    """
    Check if a quantum state is normalized (⟨ψ|ψ⟩ = 1).

    A normalized state has unit norm, which is required for
    proper probability interpretation.

    Args:
        state: Ket state or amplitude list.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.state import zero, plus
        >>> 
        >>> print(is_normalized(zero()))
        ✓ 1.000|0⟩ is normalized
        >>> 
        >>> print(is_normalized(plus()))
        ✓ 0.707|0⟩ + 0.707|1⟩ is normalized
        >>> 
        >>> # Unnormalized state
        >>> from psiqit.quantum.state import Ket
        >>> psi = Ket([1, 1])
        >>> print(is_normalized(psi))
        ✗ Ket([(1+0j), (1+0j)]) is NOT normalized
    """
    if isinstance(state, Ket):
        amplitudes = state.data
        name = str(state)
    else:
        amplitudes = state
        name = "State"
    
    norm_sq = sum(abs(a)**2 for a in amplitudes)
    norm = math.sqrt(norm_sq)
    
    if abs(norm - 1.0) > tol:
        return ValidationResult(
            is_valid=False,
            message=f"{name} is NOT normalized",
            details={'norm': norm, 'expected': 1.0}
        )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name} is normalized",
        details={'norm': norm}
    )


def are_orthogonal(state1: Union[Ket, List[complex]], 
                   state2: Union[Ket, List[complex]],
                   tol: float = 1e-10) -> ValidationResult:
    """
    Check if two quantum states are orthogonal (⟨ψ|φ⟩ = 0).

    Orthogonal states are perfectly distinguishable.

    Args:
        state1: First state.
        state2: Second state.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus
        >>> 
        >>> print(are_orthogonal(zero(), one()))
        ✓ 1.000|0⟩ and 1.000|1⟩ are orthogonal
        >>> 
        >>> print(are_orthogonal(plus(), minus()))
        ✓ 0.707|0⟩ + 0.707|1⟩ and 0.707|0⟩ - 0.707|1⟩ are orthogonal
        >>> 
        >>> print(are_orthogonal(zero(), plus()))
        ✗ 1.000|0⟩ and 0.707|0⟩ + 0.707|1⟩ are NOT orthogonal
    """
    # Convert to amplitudes
    if isinstance(state1, Ket):
        amp1 = state1.data
        name1 = str(state1)
    else:
        amp1 = state1
        name1 = "State 1"
    
    if isinstance(state2, Ket):
        amp2 = state2.data
        name2 = str(state2)
    else:
        amp2 = state2
        name2 = "State 2"
    
    if len(amp1) != len(amp2):
        return ValidationResult(
            is_valid=False,
            message=f"Cannot check orthogonality: dimensions differ",
            details={'dim1': len(amp1), 'dim2': len(amp2)}
        )
    
    overlap = sum(amp1[i].conjugate() * amp2[i] for i in range(len(amp1)))
    
    if abs(overlap) > tol:
        return ValidationResult(
            is_valid=False,
            message=f"{name1} and {name2} are NOT orthogonal",
            details={'overlap': abs(overlap)}
        )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name1} and {name2} are orthogonal",
        details={'overlap': abs(overlap)}
    )


def are_identical(state1: Union[Ket, List[complex]], 
                  state2: Union[Ket, List[complex]],
                  ignore_phase: bool = True,
                  tol: float = 1e-10) -> ValidationResult:
    """
    Check if two quantum states are identical (up to global phase).

    Args:
        state1: First state.
        state2: Second state.
        ignore_phase: If True, ignore global phase differences.
        tol: Numerical tolerance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.quantum.state import zero, Ket
        >>> 
        >>> # Identical states
        >>> psi1 = zero()
        >>> psi2 = Ket([1, 0])
        >>> print(are_identical(psi1, psi2))
        ✓ 1.000|0⟩ and Ket([(1+0j), 0j]) are identical
        >>> 
        >>> # Different by global phase
        >>> psi2 = Ket([-1, 0])
        >>> print(are_identical(psi1, psi2))
        ✓ 1.000|0⟩ and Ket([(-1+0j), 0j]) are identical
        >>> 
        >>> # Different states
        >>> psi2 = one()
        >>> print(are_identical(psi1, psi2))
        ✗ 1.000|0⟩ and 1.000|1⟩ are NOT identical
    """
    # Convert to amplitudes and normalize
    if isinstance(state1, Ket):
        amp1 = state1.normalize().data
        name1 = str(state1)
    else:
        amp1 = state1
        name1 = "State 1"
    
    if isinstance(state2, Ket):
        amp2 = state2.normalize().data
        name2 = str(state2)
    else:
        amp2 = state2
        name2 = "State 2"
    
    if len(amp1) != len(amp2):
        return ValidationResult(
            is_valid=False,
            message=f"Cannot compare: dimensions differ",
            details={'dim1': len(amp1), 'dim2': len(amp2)}
        )
    
    # Check if both are zero vectors
    is_zero1 = all(abs(a) < tol for a in amp1)
    is_zero2 = all(abs(a) < tol for a in amp2)
    
    if is_zero1 and is_zero2:
        return ValidationResult(
            is_valid=True,
            message=f"{name1} and {name2} are identical (both zero)",
            details={'fidelity': 1.0}
        )
    
    if is_zero1 or is_zero2:
        return ValidationResult(
            is_valid=False,
            message=f"{name1} and {name2} are NOT identical (one is zero)",
            details={'fidelity': 0.0}
        )
    
    if ignore_phase:
        # Find phase factor from first non-zero element
        phase = None
        for a1, a2 in zip(amp1, amp2):
            if abs(a1) > tol and abs(a2) > tol:
                phase = a1 / a2
                break
        
        if phase is None:
            return ValidationResult(
                is_valid=False,
                message=f"{name1} and {name2} are NOT identical",
                details={'reason': 'No overlapping non-zero amplitudes'}
            )
        
        # Check all amplitudes
        for a1, a2 in zip(amp1, amp2):
            if abs(a1 - phase * a2) > tol:
                return ValidationResult(
                    is_valid=False,
                    message=f"{name1} and {name2} are NOT identical",
                    details={'max_difference': max(abs(a1 - phase * a2) for a1, a2 in zip(amp1, amp2))}
                )
    else:
        # Direct comparison
        for a1, a2 in zip(amp1, amp2):
            if abs(a1 - a2) > tol:
                return ValidationResult(
                    is_valid=False,
                    message=f"{name1} and {name2} are NOT identical",
                    details={'max_difference': max(abs(a1 - a2) for a1, a2 in zip(amp1, amp2))}
                )
    
    return ValidationResult(
        is_valid=True,
        message=f"{name1} and {name2} are identical",
        details={'fidelity': 1.0}
    )


def fidelity(state1: Union[Ket, List[complex]], 
             state2: Union[Ket, List[complex]]) -> float:
    """
    Calculate the fidelity between two quantum states.

    For pure states, fidelity = |⟨ψ|φ⟩|² (0 to 1).
    Fidelity = 1 means identical states, fidelity = 0 means orthogonal.

    Args:
        state1: First state.
        state2: Second state.

    Returns:
        Fidelity between 0 and 1.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus
        >>> 
        >>> print(fidelity(zero(), zero()))
        1.0
        >>> print(fidelity(zero(), one()))
        0.0
        >>> print(fidelity(zero(), plus()))
        0.5
    """
    if isinstance(state1, Ket):
        amp1 = state1.normalize().data
    else:
        amp1 = state1
    
    if isinstance(state2, Ket):
        amp2 = state2.normalize().data
    else:
        amp2 = state2
    
    overlap = abs(sum(amp1[i].conjugate() * amp2[i] for i in range(len(amp1))))**2
    return min(1.0, overlap)


# ============================================================
# Circuit Validation
# ============================================================

def validate_circuit(circuit) -> ValidationResult:
    """
    Validate a quantum circuit instance.

    Checks that the circuit is a valid QuantumCircuit object with
    a positive number of qubits.

    Args:
        circuit: QuantumCircuit instance.

    Returns:
        ValidationResult with details.

    Examples:
        >>> from psiqit.circuits.circuit import QuantumCircuit
        >>> 
        >>> circ = QuantumCircuit(2)
        >>> result = validate_circuit(circ)
        >>> print(result)
        ✓ Valid circuit: 2 qubits, depth 0
        >>> 
        >>> # Invalid input
        >>> result = validate_circuit("not a circuit")
        >>> print(result)
        ✗ Not a QuantumCircuit instance
    """
    from ..circuits.circuit import QuantumCircuit
    
    if not isinstance(circuit, QuantumCircuit):
        return ValidationResult(
            is_valid=False,
            message="Not a QuantumCircuit instance"
        )
    
    if circuit.n_qubits <= 0:
        return ValidationResult(
            is_valid=False,
            message=f"Invalid number of qubits: {circuit.n_qubits}"
        )
    
    return ValidationResult(
        is_valid=True,
        message=f"Valid circuit: {circuit.n_qubits} qubits, depth {circuit.depth}",
        details={'n_qubits': circuit.n_qubits, 'depth': circuit.depth}
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
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