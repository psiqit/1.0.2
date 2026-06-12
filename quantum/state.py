"""
qit/quantum/state.py
Quantum States - Ket and Bra Vectors
"""

import math
import random
from typing import List, Optional

from ..math.qalgebra import (
    inner, norm, normalize, outer as outer_product, SQRT2_INV
)


class Ket:
    """
    Ket vector |ψ⟩ representing a quantum state in a finite-dimensional Hilbert space.

    The Ket class represents a quantum state vector and supports basic operations
    like addition, scalar multiplication, inner product, and outer product.

    Attributes:
        data (List[complex]): Complex amplitudes of the state vector (copy)
        dim (int): Dimension of the Hilbert space
        is_normalized (bool): Whether the state is normalized (⟨ψ|ψ⟩ = 1)

    Examples:
        >>> from psiqit.quantum.state import Ket, zero
        >>> psi = Ket([1, 0])  # |0⟩ state
        >>> phi = zero()       # same as above
        >>> print(psi.inner(phi))  # 1.0
        >>> print(psi.is_normalized)  # True
    """
    
    def __init__(self, data: List[complex], _normalized: bool = False):
        """
        Initialize a Ket state vector.

        Args:
            data: List of complex amplitudes representing the state
            _normalized: Internal flag indicating if the state is already normalized.
                        If False, normalization is not automatically performed.

        Examples:
            >>> psi = Ket([1, 0])           # |0⟩ (normalized)
            >>> psi = Ket([0.707, 0.707])   # |+⟩ (not automatically normalized)
        """
        self._data = [complex(x) for x in data]
        self._dim = len(data)
        self._normalized = _normalized
    
    @property
    def data(self) -> List[complex]:
        """
        Return a copy of the state amplitudes.

        Returns:
            A copy of the complex amplitude list.
        """
        return self._data.copy()
    
    @property
    def dim(self) -> int:
        """
        Return the dimension of the Hilbert space.

        Returns:
            The dimension (number of basis states).
        """
        return self._dim
    
    @property
    def is_normalized(self) -> bool:
        """
        Check if the state is normalized (⟨ψ|ψ⟩ = 1).

        Returns:
            True if the state is normalized within numerical tolerance (1e-10).
        """
        if not self._normalized:
            n = self.norm()
            self._normalized = abs(n - 1.0) < 1e-10
        return self._normalized
    
    def norm(self) -> float:
        """
        Calculate the Euclidean norm of the state.

        Returns:
            The norm ||ψ|| = sqrt(Σ|a_i|²)

        Examples:
            >>> psi = Ket([1, 1])
            >>> psi.norm()
            1.4142135623730951
        """
        return math.sqrt(sum(abs(x)**2 for x in self._data))
    
    def normalize(self) -> 'Ket':
        """
        Return a normalized copy of the state.

        Returns:
            A new Ket object with norm = 1.

        Raises:
            ValueError: If the state is the zero vector.

        Examples:
            >>> psi = Ket([1, 1])
            >>> psi_norm = psi.normalize()
            >>> print(psi_norm.is_normalized)
            True
            >>> print(psi_norm)
            0.707|0⟩ + 0.707|1⟩
        """
        if self.is_normalized:
            return Ket(self._data, _normalized=True)
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize zero vector")
        return Ket([x / n for x in self._data], _normalized=True)
    
    def __repr__(self) -> str:
        """Return a string representation of the Ket."""
        return f"Ket({self._data})"
    
    def __str__(self) -> str:
        """
        Return a readable string representation in Dirac notation.

        Returns:
            A string like "0.707|0⟩ + 0.707|1⟩" or "1.000|0⟩"

        Examples:
            >>> print(zero())
            1.000|0⟩
            >>> print(plus())
            0.707|0⟩ + 0.707|1⟩
        """
        terms = []
        for i, a in enumerate(self._data):
            if abs(a) > 1e-10:
                if abs(a.imag) < 1e-10:
                    amp = f"{a.real:.3f}"
                elif abs(a.real) < 1e-10:
                    amp = f"{a.imag:.3f}j"
                else:
                    sign = "+" if a.imag >= 0 else ""
                    amp = f"{a.real:.3f}{sign}{a.imag:.3f}j"
                terms.append(f"{amp}|{i}⟩")
        if not terms:
            return "|0⟩"
        return " + ".join(terms).replace("+ -", "- ")
    
    def __add__(self, other: 'Ket') -> 'Ket':
        """
        Add two Ket states.

        Args:
            other: Another Ket state with the same dimension.

        Returns:
            A new Ket representing |ψ⟩ + |φ⟩.

        Raises:
            ValueError: If dimensions don't match.

        Examples:
            >>> psi = Ket([1, 0]) + Ket([0, 1])
            >>> print(psi)
            1.000|0⟩ + 1.000|1⟩
        """
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")
        return Ket([self._data[i] + other._data[i] for i in range(self._dim)])
    
    def __sub__(self, other: 'Ket') -> 'Ket':
        """
        Subtract two Ket states.

        Args:
            other: Another Ket state with the same dimension.

        Returns:
            A new Ket representing |ψ⟩ - |φ⟩.

        Raises:
            ValueError: If dimensions don't match.
        """
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")
        return Ket([self._data[i] - other._data[i] for i in range(self._dim)])
    
    def __mul__(self, scalar: complex) -> 'Ket':
        """
        Multiply Ket by a scalar.

        Args:
            scalar: Complex scalar to multiply.

        Returns:
            A new Ket representing c|ψ⟩.

        Examples:
            >>> psi = Ket([1, 0]) * 2
            >>> print(psi)
            2.000|0⟩
        """
        return Ket([x * scalar for x in self._data])
    
    __rmul__ = __mul__
    
    def __getitem__(self, index: int) -> complex:
        """
        Get amplitude at specific index.

        Args:
            index: Index of the basis state.

        Returns:
            Complex amplitude at that index.

        Examples:
            >>> psi = Ket([1, 2])
            >>> psi[0]
            1.0
            >>> psi[1]
            2.0
        """
        return self._data[index]
    
    def __len__(self) -> int:
        """
        Return the dimension of the state.

        Returns:
            The dimension (number of basis states).
        """
        return self._dim
    
    def inner(self, other: 'Ket') -> complex:
        """
        Compute inner product ⟨self|other⟩.

        Args:
            other: Another Ket state with same dimension.

        Returns:
            The complex inner product.

        Raises:
            ValueError: If dimensions don't match.

        Examples:
            >>> zero = Ket([1, 0])
            >>> one = Ket([0, 1])
            >>> zero.inner(one)
            0j
            >>> plus().inner(plus())
            (1+0j)
        """
        if self._dim != other._dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {other._dim}")
        return inner(self._data, other._data)
    
    def outer(self, other: 'Ket') -> 'Operator':
        """
        Compute outer product |self⟩⟨other|.

        Args:
            other: Another Ket state.

        Returns:
            An Operator representing the outer product.

        Examples:
            >>> from psiqit.quantum.state import zero
            >>> P0 = zero().outer(zero())  # Projector onto |0⟩
        """
        from .operator import Operator
        return Operator(outer_product(self._data, other._data))
    
    def to_bra(self) -> 'Bra':
        """
        Convert Ket to Bra (Hermitian conjugate).

        Returns:
            A Bra vector ⟨ψ|.

        Examples:
            >>> psi = Ket([1, 0])
            >>> bra = psi.to_bra()
            >>> print(bra)
            ⟨ψ| = [(1+0j), 0j]
        """
        from .state import Bra
        return Bra([x.conjugate() for x in self._data])
    
    def measure(self, shots: int = 1) -> dict:
        """
        Measure the state in the computational basis.

        Args:
            shots: Number of measurements to perform.

        Returns:
            A dictionary with keys:
                - 'outcomes': List of measurement outcomes (integers)
                - 'counts': Dict mapping outcome to count
                - 'probabilities': List of probabilities for each outcome
                - 'shots': Number of measurements

        Raises:
            ValueError: If the state is not normalized.

        Examples:
            >>> from psiqit.quantum.state import plus
            >>> result = plus().measure(shots=1000)
            >>> print(result['counts'])  # {0: 498, 1: 502}
            >>> print(result['probabilities'])  # [0.5, 0.5]
        """
        if not self.is_normalized:
            raise ValueError("Cannot measure unnormalized state")
        
        probs = [abs(a)**2 for a in self._data]
        total = sum(probs)
        probs = [p / total for p in probs]
        
        outcomes = []
        for _ in range(shots):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    outcomes.append(i)
                    break
        
        counts = {i: outcomes.count(i) for i in set(outcomes)}
        return {
            'outcomes': outcomes,
            'counts': counts,
            'probabilities': probs,
            'shots': shots
        }
    
    def prob(self, basis_state: int) -> float:
        """
        Get probability of a specific basis state.

        Args:
            basis_state: Index of the basis state (0 to dim-1).

        Returns:
            Probability |⟨basis_state|ψ⟩|².

        Raises:
            ValueError: If basis_state is out of range.

        Examples:
            >>> psi = Ket([0.6, 0.8])
            >>> psi.prob(0)
            0.36
            >>> psi.prob(1)
            0.64
        """
        if basis_state >= self._dim:
            raise ValueError(f"Basis state {basis_state} out of range (max {self._dim-1})")
        return abs(self._data[basis_state]) ** 2
    
    def sample(self, shots: int = 1) -> List[int]:
        """
        Sample measurement outcomes.

        Args:
            shots: Number of samples.

        Returns:
            List of measurement outcomes.

        Examples:
            >>> psi = plus()
            >>> samples = psi.sample(10)
            >>> print(samples)  # e.g., [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
        """
        return self.measure(shots=shots)['outcomes']
    
    def copy(self) -> 'Ket':
        """
        Create a deep copy of the Ket.

        Returns:
            A new Ket object with the same amplitudes.
        """
        return Ket(self._data, _normalized=self._normalized)


class Bra:
    """
    Bra vector ⟨ψ| representing the dual of a Ket state.

    The Bra class represents a row vector that is the Hermitian conjugate
    of a Ket state. It is used for inner products ⟨φ|ψ⟩.

    Examples:
        >>> from psiqit.quantum.state import Ket, Bra
        >>> psi = Ket([1, 0])
        >>> bra = psi.to_bra()
        >>> print(bra)
        ⟨ψ| = [(1+0j), 0j]
    """
    
    def __init__(self, data: List[complex]):
        """
        Initialize a Bra vector.

        Args:
            data: List of complex amplitudes (coefficients of basis bras).
        """
        self._data = [complex(x) for x in data]
        self._dim = len(data)
    
    @property
    def data(self) -> List[complex]:
        """
        Return a copy of the bra amplitudes.

        Returns:
            A copy of the complex amplitude list.
        """
        return self._data.copy()
    
    @property
    def dim(self) -> int:
        """
        Return the dimension of the dual space.

        Returns:
            The dimension (number of basis states).
        """
        return self._dim
    
    def __repr__(self) -> str:
        """Return a string representation of the Bra."""
        return f"Bra({self._data})"
    
    def __str__(self) -> str:
        """Return a string representation of the Bra."""
        return f"⟨ψ| = {self._data}"
    
    def __matmul__(self, ket: Ket) -> complex:
        """
        Compute inner product ⟨self|ket⟩.

        Args:
            ket: A Ket state.

        Returns:
            The complex inner product.

        Raises:
            ValueError: If dimensions don't match.

        Examples:
            >>> psi = Ket([1, 0])
            >>> bra = psi.to_bra()
            >>> bra @ psi
            (1+0j)
        """
        if self._dim != ket.dim:
            raise ValueError(f"Dimension mismatch: {self._dim} vs {ket.dim}")
        return inner(self._data, ket.data)
    
    def __getitem__(self, index: int) -> complex:
        """
        Get amplitude at specific index.

        Args:
            index: Index of the basis state.

        Returns:
            Complex amplitude at that index.
        """
        return self._data[index]
    
    def __len__(self) -> int:
        """
        Return the dimension of the bra.

        Returns:
            The dimension (number of basis states).
        """
        return self._dim
    
    def to_ket(self) -> Ket:
        """
        Convert Bra to Ket (Hermitian conjugate).

        Returns:
            A Ket vector |ψ⟩.

        Examples:
            >>> bra = Bra([1, 0])
            >>> psi = bra.to_ket()
            >>> print(psi)
            1.000|0⟩
        """
        return Ket([x.conjugate() for x in self._data])
    
    def copy(self) -> 'Bra':
        """
        Create a deep copy of the Bra.

        Returns:
            A new Bra object with the same amplitudes.
        """
        return Bra(self._data)


# ============================================================
# Factory Functions
# ============================================================

def ket(*amplitudes: complex) -> Ket:
    """
    Create a Ket state from amplitude arguments.

    Args:
        *amplitudes: Variable number of complex amplitudes.

    Returns:
        A Ket state.

    Examples:
        >>> psi = ket(1, 0)  # |0⟩
        >>> psi = ket(0.707, 0.707)  # |+⟩
    """
    return Ket(list(amplitudes))


def basis(dim: int, index: int) -> Ket:
    """
    Create a computational basis state |index⟩.

    Args:
        dim: Dimension of the Hilbert space.
        index: Basis state index (0 to dim-1).

    Returns:
        A basis state |index⟩.

    Raises:
        ValueError: If index is out of range.

    Examples:
        >>> zero = basis(2, 0)  # |0⟩
        >>> one = basis(2, 1)   # |1⟩
        >>> state = basis(4, 2) # |2⟩ in 4-dim space
    """
    if index >= dim or index < 0:
        raise ValueError(f"Index {index} out of range (0 to {dim-1})")
    data = [0.0] * dim
    data[index] = 1.0
    return Ket(data, _normalized=True)


# ============================================================
# Single-Qubit States
# ============================================================

def zero() -> Ket:
    """
    Single-qubit |0⟩ state.

    Returns:
        |0⟩ = [1, 0]ᵀ

    Examples:
        >>> print(zero())
        1.000|0⟩
    """
    return Ket([1.0, 0.0], _normalized=True)


def one() -> Ket:
    """
    Single-qubit |1⟩ state.

    Returns:
        |1⟩ = [0, 1]ᵀ

    Examples:
        >>> print(one())
        1.000|1⟩
    """
    return Ket([0.0, 1.0], _normalized=True)


def plus() -> Ket:
    """
    Single-qubit |+⟩ state (X-basis eigenstate with eigenvalue +1).

    Returns:
        |+⟩ = (|0⟩ + |1⟩)/√2

    Examples:
        >>> print(plus())
        0.707|0⟩ + 0.707|1⟩
    """
    return Ket([SQRT2_INV, SQRT2_INV], _normalized=True)


def minus() -> Ket:
    """
    Single-qubit |-⟩ state (X-basis eigenstate with eigenvalue -1).

    Returns:
        |-⟩ = (|0⟩ - |1⟩)/√2

    Examples:
        >>> print(minus())
        0.707|0⟩ - 0.707|1⟩
    """
    return Ket([SQRT2_INV, -SQRT2_INV], _normalized=True)


def ip() -> Ket:
    """
    Single-qubit |i+⟩ state (Y-basis eigenstate with eigenvalue +1).

    Returns:
        |i+⟩ = (|0⟩ + i|1⟩)/√2

    Examples:
        >>> print(ip())
        0.707|0⟩ + 0.707j|1⟩
    """
    return Ket([SQRT2_INV, complex(0, SQRT2_INV)], _normalized=True)


def im() -> Ket:
    """
    Single-qubit |i-⟩ state (Y-basis eigenstate with eigenvalue -1).

    Returns:
        |i-⟩ = (|0⟩ - i|1⟩)/√2

    Examples:
        >>> print(im())
        0.707|0⟩ - 0.707j|1⟩
    """
    return Ket([SQRT2_INV, complex(0, -SQRT2_INV)], _normalized=True)


# ============================================================
# Bell States
# ============================================================

def bell_phi_plus() -> Ket:
    """
    Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2.

    Returns:
        Maximally entangled Bell state.

    Examples:
        >>> psi = bell_phi_plus()
        >>> print(psi)
        0.707|00⟩ + 0.707|11⟩
    """
    return Ket([SQRT2_INV, 0, 0, SQRT2_INV], _normalized=True)


def bell_phi_minus() -> Ket:
    """
    Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2.

    Returns:
        Maximally entangled Bell state with negative phase.

    Examples:
        >>> psi = bell_phi_minus()
        >>> print(psi)
        0.707|00⟩ - 0.707|11⟩
    """
    return Ket([SQRT2_INV, 0, 0, -SQRT2_INV], _normalized=True)


def bell_psi_plus() -> Ket:
    """
    Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2.

    Returns:
        Maximally entangled Bell state.

    Examples:
        >>> psi = bell_psi_plus()
        >>> print(psi)
        0.707|01⟩ + 0.707|10⟩
    """
    return Ket([0, SQRT2_INV, SQRT2_INV, 0], _normalized=True)


def bell_psi_minus() -> Ket:
    """
    Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2.

    Returns:
        Maximally entangled Bell state with negative phase.

    Examples:
        >>> psi = bell_psi_minus()
        >>> print(psi)
        0.707|01⟩ - 0.707|10⟩
    """
    return Ket([0, SQRT2_INV, -SQRT2_INV, 0], _normalized=True)


def bell_state(index: int) -> Ket:
    """
    Get Bell state by index.

    Args:
        index: 0 for |Φ⁺⟩, 1 for |Φ⁻⟩, 2 for |Ψ⁺⟩, 3 for |Ψ⁻⟩

    Returns:
        The specified Bell state.

    Raises:
        ValueError: If index is not between 0 and 3.

    Examples:
        >>> psi = bell_state(0)  # |Φ⁺⟩
        >>> psi = bell_state(2)  # |Ψ⁺⟩
    """
    states = [bell_phi_plus, bell_phi_minus, bell_psi_plus, bell_psi_minus]
    if index < 0 or index > 3:
        raise ValueError(f"Bell state index must be 0-3, got {index}")
    return states[index]()


# ============================================================
# Multi-Qubit States
# ============================================================

def ghz(n: int) -> Ket:
    """
    Create GHZ state (|00...0⟩ + |11...1⟩)/√2.

    Args:
        n: Number of qubits (≥ 2).

    Returns:
        GHZ state.

    Raises:
        ValueError: If n < 2.

    Examples:
        >>> psi = ghz(3)  # (|000⟩ + |111⟩)/√2
        >>> print(psi)
        0.707|000⟩ + 0.707|111⟩
    """
    if n < 2:
        raise ValueError("GHZ requires at least 2 qubits")
    dim = 2 ** n
    data = [0.0] * dim
    data[0] = SQRT2_INV
    data[-1] = SQRT2_INV
    return Ket(data, _normalized=True)


def w_state(n: int) -> Ket:
    """
    Create W state: superposition of states with exactly one |1⟩.

    Args:
        n: Number of qubits (≥ 2).

    Returns:
        W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n

    Raises:
        ValueError: If n < 2.

    Examples:
        >>> psi = w_state(3)  # (|100⟩ + |010⟩ + |001⟩)/√3
    """
    if n < 2:
        raise ValueError("W state requires at least 2 qubits")
    dim = 2 ** n
    v = 1 / math.sqrt(n)
    data = [0.0] * dim
    for i in range(n):
        idx = 1 << (n - 1 - i)
        data[idx] = v
    return Ket(data, _normalized=True)


# ============================================================
# Random States
# ============================================================

def random_state(dim: int, seed: Optional[int] = None) -> Ket:
    """
    Generate a random pure state (Haar measure).

    Args:
        dim: Dimension of Hilbert space.
        seed: Random seed for reproducibility.

    Returns:
        A random normalized pure state.

    Examples:
        >>> psi = random_state(2)
        >>> print(psi.is_normalized)
        True
        >>> psi = random_state(2, seed=42)  # Reproducible
    """
    if seed is not None:
        random.seed(seed)
    data = [complex(random.gauss(0, 1), random.gauss(0, 1)) for _ in range(dim)]
    n = math.sqrt(sum(abs(x)**2 for x in data))
    return Ket([x / n for x in data], _normalized=True)


# ============================================================
# Utility Functions
# ============================================================

def is_orthogonal(psi: Ket, phi: Ket, tol: float = 1e-10) -> bool:
    """
    Check if two states are orthogonal.

    Args:
        psi: First quantum state.
        phi: Second quantum state.
        tol: Numerical tolerance.

    Returns:
        True if |⟨ψ|φ⟩| < tol.

    Examples:
        >>> zero_state = zero()
        >>> one_state = one()
        >>> is_orthogonal(zero_state, one_state)
        True
    """
    return abs(psi.inner(phi)) < tol


def is_same(psi: Ket, phi: Ket, tol: float = 1e-10) -> bool:
    """
    Check if two states are identical (up to global phase).

    Args:
        psi: First quantum state.
        phi: Second quantum state.
        tol: Numerical tolerance.

    Returns:
        True if ||ψ⟩ and |φ⟩ represent the same physical state.

    Examples:
        >>> psi = plus()
        >>> phi = Ket([0.707, 0.707])
        >>> is_same(psi, phi)
        True
    """
    psi_norm = psi.normalize()
    phi_norm = phi.normalize()
    overlap = psi_norm.inner(phi_norm)
    return abs(abs(overlap) - 1.0) < tol


def fidelity(psi: Ket, phi: Ket) -> float:
    """
    Calculate fidelity between two quantum states.

    Fidelity = |⟨ψ|φ⟩|², where 0 ≤ F ≤ 1.
    F = 1 means identical states, F = 0 means orthogonal.

    Args:
        psi: First quantum state.
        phi: Second quantum state.

    Returns:
        Fidelity value between 0 and 1.

    Examples:
        >>> f = fidelity(zero(), zero())
        >>> print(f)
        1.0
        >>> f = fidelity(zero(), one())
        >>> print(f)
        0.0
        >>> f = fidelity(zero(), plus())
        >>> print(f)
        0.5
    """
    return abs(psi.inner(phi)) ** 2


# ============================================================
# Extended States (Coherent, Squeezed, Fock)
# ============================================================

def fock_state(n: int, n_levels: int) -> Ket:
    """
    Fock (number) state |n⟩ for harmonic oscillator.

    Args:
        n: Photon number (0 ≤ n < n_levels)
        n_levels: Truncation dimension

    Returns:
        Fock state |n⟩

    Raises:
        ValueError: If n is out of range.

    Examples:
        >>> vacuum = fock_state(0, 5)   # |0⟩
        >>> one_photon = fock_state(1, 5)  # |1⟩
    """
    if n >= n_levels or n < 0:
        raise ValueError(f"n={n} must be between 0 and {n_levels-1}")
    
    data = [0.0] * n_levels
    data[n] = 1.0
    return Ket(data, _normalized=True)


def coherent_state(alpha: complex, n_levels: int = 20) -> Ket:
    """
    Coherent state |α⟩.

    |α⟩ = e^{-|α|²/2} Σ_{n=0}^{∞} (αⁿ/√n!) |n⟩

    Args:
        alpha: Complex amplitude α
        n_levels: Truncation dimension

    Returns:
        Truncated coherent state |α⟩

    Examples:
        >>> psi = coherent_state(1.0, 10)      # |α=1⟩
        >>> psi = coherent_state(1+0.5j, 15)   # Complex alpha
    """
    norm_sq = abs(alpha)**2
    prefactor = math.exp(-norm_sq / 2)
    
    data = []
    factorial = 1
    for n in range(n_levels):
        if n > 0:
            factorial *= n
        coeff = prefactor * (alpha**n) / math.sqrt(factorial)
        data.append(coeff)
    
    # Normalize
    norm = math.sqrt(sum(abs(a)**2 for a in data))
    if norm > 0:
        data = [a / norm for a in data]
    
    return Ket(data, _normalized=True)


def squeezed_state(r: float, phi: float = 0.0, n_levels: int = 20) -> Ket:
    """
    Squeezed vacuum state |r, φ⟩.

    Args:
        r: Squeezing parameter (r ≥ 0)
        phi: Squeezing angle (radians)
        n_levels: Truncation dimension

    Returns:
        Truncated squeezed vacuum state

    Examples:
        >>> psi = squeezed_state(r=1.0, phi=0, n_levels=15)
    """
    data = []
    for n in range(n_levels):
        if n % 2 == 0:  # Only even terms
            m = n // 2
            coeff = math.sqrt(math.factorial(2*m)) / (2**m * math.factorial(m))
            coeff *= math.exp(1j * m * phi) * ((-math.tanh(r))**m)
            coeff /= math.sqrt(math.cosh(r))
        else:
            coeff = 0.0
        data.append(coeff)
    
    # Normalize
    norm = math.sqrt(sum(abs(a)**2 for a in data))
    if norm > 0:
        data = [a / norm for a in data]
    
    return Ket(data, _normalized=True)


def thermal_state(n_bar: float, n_levels: int = 20) -> Ket:
    """
    Thermal (mixed) state represented as pure state approximation.

    Args:
        n_bar: Mean photon number
        n_levels: Truncation dimension

    Returns:
        Approximation of thermal state as pure state

    Examples:
        >>> psi = thermal_state(n_bar=1.0, n_levels=10)
    """
    data = []
    for n in range(n_levels):
        prob = (n_bar**n) / ((1 + n_bar)**(n + 1))
        data.append(math.sqrt(prob))
    
    # Normalize
    norm = math.sqrt(sum(a**2 for a in data))
    if norm > 0:
        data = [a / norm for a in data]
    
    return Ket(data, _normalized=True)


# ============================================================
# Multi-Party States (Alice, Bob, Charlie)
# ============================================================

def alice_bell_state(index: int = 0) -> Ket:
    """
    Bell state shared between Alice and Bob.

    Args:
        index: 0 for |Φ⁺⟩, 1 for |Φ⁻⟩, 2 for |Ψ⁺⟩, 3 for |Ψ⁻⟩

    Returns:
        Bell state (2-qubit)

    Examples:
        >>> psi = alice_bell_state(0)  # |Φ⁺⟩ shared by Alice and Bob
    """
    return bell_state(index)


def alice_bob_ghz(n: int = 3) -> Ket:
    """
    GHZ state shared among Alice, Bob, Charlie, etc.

    Args:
        n: Number of parties (qubits)

    Returns:
        GHZ state (|00...0⟩ + |11...1⟩)/√2

    Examples:
        >>> psi = alice_bob_ghz(3)  # Shared by Alice, Bob, Charlie
    """
    return ghz(n)


def alice_bob_w(n: int = 3) -> Ket:
    """
    W state shared among Alice, Bob, Charlie, etc.

    Args:
        n: Number of parties (qubits)

    Returns:
        W state

    Examples:
        >>> psi = alice_bob_w(3)  # Shared by Alice, Bob, Charlie
    """
    return w_state(n)


def ep_pair() -> Ket:
    """
    Einstein-Podolsky-Rosen (EPR) pair (Bell state |Φ⁺⟩).

    Returns:
        EPR pair shared between Alice and Bob

    Examples:
        >>> psi = ep_pair()  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    """
    return bell_phi_plus()


# ============================================================
# Phase States
# ============================================================

def phase_state(theta: float, dim: int = 2) -> Ket:
    """
    Phase state |θ⟩ = (1/√d) Σ e^{ikθ} |k⟩.

    Args:
        theta: Phase angle (radians)
        dim: Dimension of Hilbert space

    Returns:
        Phase state

    Examples:
        >>> psi = phase_state(math.pi/4, dim=4)
    """
    data = []
    norm = 1 / math.sqrt(dim)
    for k in range(dim):
        data.append(norm * math.cos(k * theta) + 1j * norm * math.sin(k * theta))
    
    return Ket(data, _normalized=True)


def dual_rail_qubit() -> Ket:
    """
    Dual-rail qubit representation (photon in two modes).

    |0⟩_L = |01⟩ (photon in mode 0)
    |1⟩_L = |10⟩ (photon in mode 1)

    Returns:
        Dual-rail encoded qubit (|01⟩ + |10⟩)/√2

    Examples:
        >>> psi = dual_rail_qubit()
        >>> print(psi)
        0.707|01⟩ + 0.707|10⟩
    """
    return Ket([0, 1, 1, 0], _normalized=False)


__all__ = [
    'Ket', 'Bra',
    'ket', 'basis',
    'zero', 'one', 'plus', 'minus', 'ip', 'im',
    'bell_phi_plus', 'bell_phi_minus',
    'bell_psi_plus', 'bell_psi_minus', 'bell_state',
    'ghz', 'w_state', 'random_state',
    'is_orthogonal', 'is_same', 'fidelity',
]