"""
psiqit/quantum/operator.py
============================================================
Quantum Operators - Gates and Observables
============================================================

Operators in quantum mechanics:
    • Unitary operators (gates) for evolution
    • Hermitian operators (observables) for measurements
    • Compositions and tensor products

This module provides the Operator class and a comprehensive set of
predefined quantum gates (Pauli, Hadamard, rotations, CNOT, Toffoli, etc.).

Operators can be:
    - Applied to quantum states: U|ψ⟩
    - Composed with other operators: U·V
    - Added or subtracted: A + B
    - Multiplied by scalars: cA
    - Raised to powers: Aⁿ

Example:
    >>> from psiqit.quantum.operator import Operator, hadamard, cnot
    >>> from psiqit.quantum.state import zero
    >>> 
    >>> # Create Pauli X gate
    >>> X = Operator([[0, 1], [1, 0]])
    >>> psi = X @ zero()  # Apply X to |0⟩ = |1⟩
    >>> 
    >>> # Use predefined gates
    >>> H = hadamard()
    >>> circuit = H @ H  # H² = I

References:
    M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
    Cambridge University Press, 2010.
"""

import math
from typing import List, Optional, Union

from ..math.qalgebra import (
    eye, zeros, mul, dagger, trace, kron,
    is_unitary, is_hermitian, commutator, anticommutator,
    eigenvalues, eigenvectors, expm, display
)
from .state import Ket, Bra


class Operator:
    """
    Quantum operator Â (matrix representation).

    The Operator class represents linear operators on Hilbert spaces.
    It supports:
        - Unitary operators: quantum gates (U†U = I)
        - Hermitian operators: observables (A† = A)
        - General operators: arbitrary matrices

    Operators can be applied to states (Ket/Bra), composed with other
    operators, added, subtracted, multiplied by scalars, and raised to powers.

    Examples:
        >>> # Pauli X gate
        >>> X = Operator([[0, 1], [1, 0]], "X")
        >>> 
        >>> # Hadamard gate
        >>> v = 1 / math.sqrt(2)
        >>> H = Operator([[v, v], [v, -v]], "H")
        >>> 
        >>> # Apply to state
        >>> from psiqit.quantum.state import zero
        >>> psi = H @ zero()  # |+⟩ state
    """
    
    def __init__(self, data: List[List[complex]], name: str = ""):
        """
        Create an operator from matrix data.

        Args:
            data: 2D list representing the matrix.
            name: Optional name for the operator (e.g., "X", "H", "CNOT").

        Examples:
            >>> X = Operator([[0, 1], [1, 0]], "X")
            >>> I = Operator([[1, 0], [0, 1]], "I")
        """
        self._data = [[complex(x) for x in row] for row in data]
        self._dim = len(data)
        self._name = name
    
    @property
    def data(self) -> List[List[complex]]:
        """
        Return a copy of the matrix data.

        Returns:
            Copy of the operator matrix.

        Examples:
            >>> X = pauli_x()
            >>> X.data
            [[0, 1], [1, 0]]
        """
        return [row[:] for row in self._data]
    
    @property
    def dim(self) -> int:
        """
        Return the dimension of the operator.

        Returns:
            Size of the matrix (number of rows/columns).

        Examples:
            >>> H = hadamard()
            >>> H.dim
            2
        """
        return self._dim
    
    @property
    def name(self) -> str:
        """
        Return the name of the operator.

        Returns:
            Operator name (e.g., "X", "H", "CNOT").
        """
        return self._name
    
    @property
    def is_unitary(self) -> bool:
        """
        Check if the operator is unitary (U†U = I).

        Returns:
            True if the operator is unitary, False otherwise.

        Examples:
            >>> H = hadamard()
            >>> H.is_unitary
            True
            >>> X = pauli_x()
            >>> X.is_unitary
            True
        """
        return is_unitary(self._data)
    
    @property
    def is_hermitian(self) -> bool:
        """
        Check if the operator is Hermitian (A† = A).

        Returns:
            True if the operator is Hermitian, False otherwise.

        Examples:
            >>> Z = pauli_z()
            >>> Z.is_hermitian
            True
            >>> H = hadamard()
            >>> H.is_hermitian
            True
        """
        return is_hermitian(self._data)
    
    # ============================================================
    # Magic Methods
    # ============================================================
    
    def __repr__(self) -> str:
        """
        Return a string representation of the operator.

        Returns:
            String showing name and dimensions.
        """
        if self._name:
            return f"Operator('{self._name}', {self._dim}×{self._dim})"
        return f"Operator({self._dim}×{self._dim})"
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns:
            String showing name and dimensions.
        """
        if self._name:
            return f"{self._name} operator ({self._dim}×{self._dim})"
        return f"Operator ({self._dim}×{self._dim})"
    
    def __matmul__(self, other):
        """
        Apply operator to a state or compose with another operator.

        Args:
            other: Ket (state), Bra, or another Operator.

        Returns:
            - If other is Ket: Ket |ψ'⟩ = A|ψ⟩
            - If other is Bra: Bra ⟨ψ'| = ⟨ψ|A
            - If other is Operator: Operator C = A @ B

        Examples:
            >>> X = pauli_x()
            >>> zero_state = Ket([1, 0])
            >>> one_state = X @ zero_state  # |1⟩
            >>> 
            >>> H = hadamard()
            >>> I = H @ H  # H² = I
        """
        if isinstance(other, Ket):
            # Apply operator to state: |ψ'⟩ = A|ψ⟩
            result_data = []
            for i in range(self._dim):
                val = sum(self._data[i][j] * other.data[j] for j in range(self._dim))
                result_data.append(val)
            return Ket(result_data)
        
        elif isinstance(other, Operator):
            # Compose operators: C = A @ B
            new_data = mul(self._data, other.data)
            return Operator(new_data, f"{self._name}{other._name}")
        
        elif isinstance(other, Bra):
            # Apply operator to bra: ⟨ψ'| = ⟨ψ|A
            result_data = []
            for i in range(self._dim):
                val = sum(self._data[j][i].conjugate() * other.data[j] for j in range(self._dim))
                result_data.append(val)
            return Bra(result_data)
        
        raise TypeError(f"Cannot apply Operator to {type(other)}")
    
    def __add__(self, other: 'Operator') -> 'Operator':
        """
        Add two operators: A + B.

        Args:
            other: Another operator with the same dimension.

        Returns:
            Operator representing the sum.

        Examples:
            >>> X = pauli_x()
            >>> Z = pauli_z()
            >>> H = X + Z  # Hadamard up to scaling
        """
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")
        new_data = [[self._data[i][j] + other._data[i][j] 
                    for j in range(self._dim)] 
                   for i in range(self._dim)]
        return Operator(new_data)
    
    def __sub__(self, other: 'Operator') -> 'Operator':
        """
        Subtract two operators: A - B.

        Args:
            other: Another operator with the same dimension.

        Returns:
            Operator representing the difference.
        """
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch")
        new_data = [[self._data[i][j] - other._data[i][j] 
                    for j in range(self._dim)] 
                   for i in range(self._dim)]
        return Operator(new_data)
    
    def __mul__(self, scalar: complex) -> 'Operator':
        """
        Multiply operator by a scalar: cA.

        Args:
            scalar: Complex scalar multiplier.

        Returns:
            Operator representing the scaled matrix.

        Examples:
            >>> H = hadamard()
            >>> scaled = 0.5 * H
        """
        new_data = [[x * scalar for x in row] for row in self._data]
        return Operator(new_data)
    
    __rmul__ = __mul__
    
    def __pow__(self, n: int) -> 'Operator':
        """
        Raise operator to a power: Aⁿ.

        Args:
            n: Non-negative integer exponent.

        Returns:
            Operator representing Aⁿ.

        Examples:
            >>> X = pauli_x()
            >>> X2 = X @ X  # X² = I
        """
        if n == 0:
            return Operator(eye(self._dim))
        if n == 1:
            return self
        result = self
        for _ in range(n - 1):
            result = result @ self
        return result
    
    # ============================================================
    # Operator Properties
    # ============================================================
    
    def dagger(self) -> 'Operator':
        """
        Return the Hermitian conjugate (adjoint) of the operator.

        Returns:
            Operator representing A†.

        Examples:
            >>> X = pauli_x()
            >>> X_dag = X.dagger()  # X is Hermitian, so X† = X
        """
        return Operator(dagger(self._data), f"{self._name}†")
    
    def trace(self) -> complex:
        """
        Compute the trace of the operator.

        Returns:
            Sum of diagonal elements: Tr(A) = Σ A_{ii}.

        Examples:
            >>> X = pauli_x()
            >>> X.trace()
            0j
            >>> I = identity()
            >>> I.trace()
            2.0
        """
        return trace(self._data)
    
    def commutator(self, other: 'Operator') -> 'Operator':
        """
        Compute the commutator [A, B] = AB - BA.

        Args:
            other: Another operator.

        Returns:
            Operator representing the commutator.

        Examples:
            >>> X = pauli_x()
            >>> Z = pauli_z()
            >>> comm = X.commutator(Z)  # [X, Z] = -2iY
        """
        return Operator(commutator(self._data, other._data))
    
    def anticommutator(self, other: 'Operator') -> 'Operator':
        """
        Compute the anticommutator {A, B} = AB + BA.

        Args:
            other: Another operator.

        Returns:
            Operator representing the anticommutator.

        Examples:
            >>> X = pauli_x()
            >>> X.anticommutator(X)  # {X, X} = 2I
        """
        return Operator(anticommutator(self._data, other._data))
    
    def eigenvalues(self) -> List[complex]:
        """
        Compute the eigenvalues of the operator.

        Returns:
            List of eigenvalues.

        Examples:
            >>> Z = pauli_z()
            >>> Z.eigenvalues()
            [1.0, -1.0]
        """
        return eigenvalues(self._data)
    
    def eigenvectors(self) -> dict:
        """
        Compute the eigenvalues and eigenvectors of the operator.

        Returns:
            Dictionary with keys 'values' and 'vectors'.

        Examples:
            >>> Z = pauli_z()
            >>> result = Z.eigenvectors()
            >>> print(result['values'])
            [1.0, -1.0]
        """
        return eigenvectors(self._data)
    
    def exp(self) -> 'Operator':
        """
        Compute the matrix exponential e^A.

        Returns:
            Operator representing exp(A).

        Examples:
            >>> from psiqit.quantum.operator import pauli_y
            >>> Ry = lambda theta: (-1j * theta/2 * pauli_y()).exp()
        """
        return Operator(expm(self._data), f"exp({self._name})")
    
    # ============================================================
    # Display
    # ============================================================
    
    def show(self, precision: int = 3) -> None:
        """
        Pretty print the operator matrix.

        Args:
            precision: Number of decimal places to display.

        Examples:
            >>> H = hadamard()
            >>> H.show()
            
            H operator:
            [ 0.707  0.707 ]
            [ 0.707 -0.707 ]
        """
        print(f"\n{self._name} operator:" if self._name else "\nOperator:")
        display(self._data, precision)


# ============================================================
# Pauli Gates (Single-Qubit)
# ============================================================

def pauli_x() -> Operator:
    """
    Pauli X gate (NOT gate).

    The X gate flips the state: X|0⟩ = |1⟩, X|1⟩ = |0⟩.

    Returns:
        Operator: X = [[0, 1], [1, 0]]

    Examples:
        >>> X = pauli_x()
        >>> from psiqit.quantum.state import zero
        >>> one = X @ zero()
    """
    return Operator([[0, 1], [1, 0]], "X")


def pauli_y() -> Operator:
    """
    Pauli Y gate.

    The Y gate applies a bit flip and a phase flip: Y = i|1⟩⟨0| - i|0⟩⟨1|.

    Returns:
        Operator: Y = [[0, -i], [i, 0]]

    Examples:
        >>> Y = pauli_y()
    """
    return Operator([[0, -1j], [1j, 0]], "Y")


def pauli_z() -> Operator:
    """
    Pauli Z gate (phase flip).

    The Z gate leaves |0⟩ unchanged and flips the sign of |1⟩: Z|1⟩ = -|1⟩.

    Returns:
        Operator: Z = [[1, 0], [0, -1]]

    Examples:
        >>> Z = pauli_z()
        >>> from psiqit.quantum.state import one
        >>> minus_one = Z @ one()  # -|1⟩
    """
    return Operator([[1, 0], [0, -1]], "Z")


def identity() -> Operator:
    """
    Identity gate.

    Returns:
        Operator: I = [[1, 0], [0, 1]]

    Examples:
        >>> I = identity()
    """
    return Operator([[1, 0], [0, 1]], "I")


# ============================================================
# Single-Qubit Gates
# ============================================================

def hadamard() -> Operator:
    """
    Hadamard gate.

    The Hadamard gate creates superposition:
        H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2
        H|1⟩ = |-⟩ = (|0⟩ - |1⟩)/√2

    Returns:
        Operator: H = (X+Z)/√2 = [[1/√2, 1/√2], [1/√2, -1/√2]]

    Examples:
        >>> H = hadamard()
        >>> from psiqit.quantum.state import zero
        >>> plus = H @ zero()
    """
    v = 1 / math.sqrt(2)
    return Operator([[v, v], [v, -v]], "H")


def phase(theta: float = math.pi/2) -> Operator:
    """
    Phase gate P(θ) = diag(1, e^{iθ}).

    Args:
        theta: Phase angle in radians (default: π/2).

    Returns:
        Operator: P(θ) = [[1, 0], [0, e^{iθ}]]

    Examples:
        >>> S = phase(math.pi/2)  # S gate
        >>> T = phase(math.pi/4)  # T gate
    """
    return Operator([[1, 0], [0, complex(math.cos(theta), math.sin(theta))]], 
                    f"P({theta:.2f})")


def s_gate() -> Operator:
    """
    S gate (phase gate with θ = π/2).

    Returns:
        Operator: S = [[1, 0], [0, i]]

    Examples:
        >>> S = s_gate()
    """
    return phase(math.pi/2, "S")


def t_gate() -> Operator:
    """
    T gate (phase gate with θ = π/4).

    Returns:
        Operator: T = [[1, 0], [0, e^{iπ/4}]]

    Examples:
        >>> T = t_gate()
    """
    return phase(math.pi/4, "T")


# ============================================================
# Rotation Gates
# ============================================================

def rx(theta: float) -> Operator:
    """
    Rotation around the X-axis: Rx(θ) = exp(-iθX/2).

    Args:
        theta: Rotation angle in radians.

    Returns:
        Operator: Rx(θ) = [[cos(θ/2), -i sin(θ/2)], [-i sin(θ/2), cos(θ/2)]]

    Examples:
        >>> Rx_pi = rx(math.pi)  # X gate up to phase
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return Operator([[c, -1j * s], [-1j * s, c]], f"Rx({theta:.2f})")


def ry(theta: float) -> Operator:
    """
    Rotation around the Y-axis: Ry(θ) = exp(-iθY/2).

    Args:
        theta: Rotation angle in radians.

    Returns:
        Operator: Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]

    Examples:
        >>> Ry_pi = ry(math.pi)  # Y gate up to phase
    """
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return Operator([[c, -s], [s, c]], f"Ry({theta:.2f})")


def rz(theta: float) -> Operator:
    """
    Rotation around the Z-axis: Rz(θ) = exp(-iθZ/2).

    Args:
        theta: Rotation angle in radians.

    Returns:
        Operator: Rz(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]

    Examples:
        >>> Rz_pi = rz(math.pi)  # Z gate up to phase
    """
    p = complex(math.cos(theta / 2), -math.sin(theta / 2))
    m = complex(math.cos(theta / 2), math.sin(theta / 2))
    return Operator([[p, 0], [0, m]], f"Rz({theta:.2f})")


# ============================================================
# Two-Qubit Gates
# ============================================================

def cnot() -> Operator:
    """
    CNOT (controlled-NOT) gate.

    The CNOT gate flips the target qubit if the control qubit is |1⟩.

    Returns:
        Operator: 4×4 matrix representing CNOT.

    Examples:
        >>> CNOT = cnot()
    """
    return Operator([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], "CNOT")


def cz() -> Operator:
    """
    CZ (controlled-Z) gate.

    The CZ gate applies a Z gate to the target if the control is |1⟩,
    adding a -1 phase when both qubits are |1⟩.

    Returns:
        Operator: 4×4 matrix representing CZ.

    Examples:
        >>> CZ = cz()
    """
    return Operator([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], "CZ")


def swap() -> Operator:
    """
    SWAP gate.

    The SWAP gate exchanges the states of two qubits.

    Returns:
        Operator: 4×4 matrix representing SWAP.

    Examples:
        >>> SWAP = swap()
    """
    return Operator([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], "SWAP")


# ============================================================
# Three-Qubit Gates
# ============================================================

def toffoli() -> Operator:
    """
    Toffoli (CCNOT) gate.

    The Toffoli gate flips the target qubit if both control qubits are |1⟩.
    It is a universal gate for classical reversible computing.

    Returns:
        Operator: 8×8 matrix representing Toffoli.

    Examples:
        >>> TOFFOLI = toffoli()
    """
    return Operator([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,0]
    ], "Toffoli")


def fredkin() -> Operator:
    """
    Fredkin (CSWAP) gate.

    The Fredkin gate swaps the target qubits if the control qubit is |1⟩.

    Returns:
        Operator: 8×8 matrix representing Fredkin.

    Examples:
        >>> FREDKIN = fredkin()
    """
    return Operator([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1]
    ], "Fredkin")


# ============================================================
# Utility Functions
# ============================================================

def tensor_product(A: Operator, B: Operator) -> Operator:
    """
    Compute the tensor product of two operators: A ⊗ B.

    Args:
        A: First operator.
        B: Second operator.

    Returns:
        Operator representing the tensor product.

    Examples:
        >>> X = pauli_x()
        >>> I = identity()
        >>> X_I = tensor_product(X, I)  # X ⊗ I
    """
    return Operator(kron(A.data, B.data), f"{A.name}⊗{B.name}")


def expectation(operator: Operator, state: Ket) -> complex:
    """
    Compute the expectation value ⟨ψ|A|ψ⟩.

    Args:
        operator: Hermitian operator.
        state: Quantum state (must be normalized).

    Returns:
        Expectation value as a real number.

    Examples:
        >>> Z = pauli_z()
        >>> from psiqit.quantum.state import zero
        >>> expectation(Z, zero())  # ⟨0|Z|0⟩ = 1
        1.0
    """
    if not state.is_normalized:
        raise ValueError("State must be normalized")
    A_psi = operator @ state
    return state.inner(A_psi)


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Core class
    'Operator',
    # Pauli gates
    'pauli_x', 'pauli_y', 'pauli_z', 'identity',
    # Single-qubit gates
    'hadamard', 'phase', 's_gate', 't_gate',
    'rx', 'ry', 'rz',
    # Multi-qubit gates
    'cnot', 'cz', 'swap',
    'toffoli', 'fredkin',
    # Utilities
    'tensor_product', 'expectation',
]