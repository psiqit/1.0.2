"""
psiqit/quantum/measurement.py
============================================================
Quantum Measurements - Standardized Outputs
============================================================

This module provides tools for performing measurements on quantum states,
including:

    • Projective measurements in computational or custom bases
    • Observable measurements (Hermitian operators)
    • POVM (Positive Operator-Valued Measure) measurements
    • State tomography (simplified)

All measurement functions return dictionaries with consistent structures,
making it easy to work with results regardless of the measurement type.

Example:
    >>> from psiqit.quantum.state import plus
    >>> from psiqit.quantum.measurement import measure, expectation, povm_z_basis
    >>> 
    >>> # Projective measurement
    >>> result = measure(plus(), shots=1000)
    >>> counts = result['counts']      # Dict[int, int]
    >>> probs = result['probabilities'] # List[float]
    >>> 
    >>> # Expectation value
    >>> from psiqit.quantum.operator import pauli_z
    >>> exp_val = expectation(pauli_z(), plus())
    >>> print(f"⟨Z⟩ = {exp_val:.4f}")
    ⟨Z⟩ = 0.0000
    >>> 
    >>> # POVM measurement
    >>> povm = povm_z_basis()
    >>> result = povm.measure(plus(), shots=1000)
    >>> print(result['counts'])

References:
    M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
    Cambridge University Press, 2010.
"""

import math
import random
from typing import List, Dict, Optional, Tuple, Union, Any

from .state import Ket, Bra
from .operator import Operator, pauli_x, pauli_y, pauli_z, identity
from ..math.qalgebra import eye, dagger, trace, kron, eigenvalues, is_hermitian, is_positive


# ============================================================
# Basic Measurement (Standardized)
# ============================================================

def measure(state: Ket, shots: int = 1, basis: Optional[List[Ket]] = None) -> Dict[str, Any]:
    """
    Perform a projective measurement in the computational basis (or custom basis).

    This function measures the quantum state and returns a dictionary with
    standardized keys: 'outcomes', 'counts', 'probabilities', and 'shots'.

    Args:
        state: Quantum state |ψ⟩ (must be normalized).
        shots: Number of measurements to perform.
        basis: Custom measurement basis (list of orthonormal Kets).
               If None, uses the computational basis.

    Returns:
        Dictionary with keys:
            - 'outcomes': List of measurement outcomes (indices).
            - 'counts': Dictionary mapping outcomes to counts.
            - 'probabilities': List of probabilities for each outcome.
            - 'shots': Number of measurements performed.

    Raises:
        ValueError: If the state is not normalized.

    Examples:
        >>> from psiqit.quantum.state import plus
        >>> psi = plus()
        >>> result = measure(psi, shots=1000)
        >>> print(result['counts'])  # {0: 498, 1: 502}
        >>> print(result['probabilities'])  # [0.5, 0.5]
        >>> 
        >>> # Custom basis (X-basis)
        >>> from psiqit.quantum.state import plus, minus
        >>> basis = [plus(), minus()]
        >>> result = measure(psi, basis=basis, shots=1000)
    """
    if not state.is_normalized:
        raise ValueError("Cannot measure unnormalized state. Call .normalize() first")
    
    if basis is None:
        probs = [abs(a)**2 for a in state.data]
    else:
        probs = [abs(b.inner(state))**2 for b in basis]
    
    # Normalize
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    
    # Sample outcomes
    outcomes = []
    for _ in range(shots):
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                outcomes.append(i)
                break
    
    # Count outcomes (always int keys)
    counts = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    
    return {
        'outcomes': outcomes,
        'counts': counts,
        'probabilities': probs,
        'shots': shots
    }


def measure_observable(state: Ket, observable: Operator, shots: int = 1) -> Dict[str, Any]:
    """
    Measure an observable (Hermitian operator) on a quantum state.

    The possible outcomes are the eigenvalues of the observable, and the
    probabilities are given by the Born rule.

    Args:
        state: Quantum state (must be normalized).
        observable: Hermitian operator to measure.
        shots: Number of measurements.

    Returns:
        Dictionary with keys:
            - 'outcomes': List of measurement outcomes (eigenvalues).
            - 'counts': Dictionary mapping eigenvalues to counts.
            - 'probabilities': List of probabilities for each outcome.
            - 'eigenvalues': List of possible eigenvalues.
            - 'shots': Number of measurements.

    Raises:
        ValueError: If the state is not normalized or observable is not Hermitian.

    Examples:
        >>> from psiqit.quantum.operator import pauli_z
        >>> from psiqit.quantum.state import plus
        >>> Z = pauli_z()
        >>> psi = plus()
        >>> result = measure_observable(psi, Z, shots=1000)
        >>> print(result['counts'])  # {1.0: 498, -1.0: 502}
    """
    if not state.is_normalized:
        raise ValueError("Cannot measure unnormalized state")
    
    if not observable.is_hermitian:
        raise ValueError("Observable must be Hermitian")
    
    # Get eigenvalues and eigenvectors
    eig_result = observable.eigenvectors()
    eigenvalues_list = eig_result['values']
    eigenvectors_list = eig_result['vectors']
    
    # Convert to Ket and normalize
    eigenstates = []
    for vec in eigenvectors_list:
        ket = Ket(vec)
        if not ket.is_normalized:
            ket = ket.normalize()
        eigenstates.append(ket)
    
    # Calculate probabilities
    probs = []
    for eigvec in eigenstates:
        overlap = eigvec.inner(state)
        probs.append(abs(overlap)**2)
    
    # Normalize
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    
    # Round eigenvalues for consistency
    rounded_eigenvalues = []
    for v in eigenvalues_list:
        if isinstance(v, complex):
            v = v.real
        rounded_eigenvalues.append(round(v, 10))
    
    # Sample outcomes
    outcomes = []
    for _ in range(shots):
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                outcomes.append(rounded_eigenvalues[i])
                break
    
    # Count outcomes
    counts = {}
    for o in outcomes:
        counts[o] = counts.get(o, 0) + 1
    
    return {
        'outcomes': outcomes,
        'counts': counts,
        'probabilities': probs,
        'eigenvalues': rounded_eigenvalues,
        'shots': shots
    }


def expectation(observable: Operator, state: Ket) -> float:
    """
    Compute the expectation value ⟨ψ|A|ψ⟩.

    Args:
        observable: Hermitian operator.
        state: Quantum state (must be normalized).

    Returns:
        Expectation value (real number).

    Examples:
        >>> from psiqit.quantum.operator import pauli_z
        >>> from psiqit.quantum.state import zero, plus
        >>> Z = pauli_z()
        >>> print(expectation(Z, zero()))  # ⟨0|Z|0⟩ = 1.0
        1.0
        >>> print(expectation(Z, plus()))  # ⟨+|Z|+⟩ = 0.0
        0.0
    """
    if not state.is_normalized:
        raise ValueError("State must be normalized")
    
    A_psi = observable @ state
    return state.inner(A_psi).real


def variance(observable: Operator, state: Ket) -> float:
    """
    Compute the variance Var(A) = ⟨A²⟩ - ⟨A⟩².

    Args:
        observable: Hermitian operator.
        state: Quantum state.

    Returns:
        Variance (non-negative real number).

    Examples:
        >>> from psiqit.quantum.operator import pauli_z
        >>> from psiqit.quantum.state import plus
        >>> Z = pauli_z()
        >>> psi = plus()
        >>> print(variance(Z, psi))  # For |+⟩, Var(Z) = 1
        1.0
    """
    exp_val = expectation(observable, state)
    exp_val_sq = expectation(observable @ observable, state)
    return exp_val_sq - exp_val**2


def standard_deviation(observable: Operator, state: Ket) -> float:
    """
    Compute the standard deviation √Var(A).

    Args:
        observable: Hermitian operator.
        state: Quantum state.

    Returns:
        Standard deviation (non-negative real number).

    Examples:
        >>> from psiqit.quantum.operator import pauli_z
        >>> from psiqit.quantum.state import plus
        >>> std = standard_deviation(pauli_z(), plus())
        >>> print(std)  # For |+⟩, ΔZ = 1.0
        1.0
    """
    return math.sqrt(variance(observable, state))


# ============================================================
# POVM (Standardized)
# ============================================================

class POVM:
    """
    Positive Operator-Valued Measure (POVM) for generalized measurements.

    A POVM is a set of positive operators {E_i} that sum to the identity.
    The probability of outcome i is p(i) = ⟨ψ|E_i|ψ⟩.

    This class provides a standardized interface for POVM measurements.

    Examples:
        >>> # Z-basis POVM
        >>> povm = POVM([P0, P1], ["0", "1"])
        >>> result = povm.measure(state, shots=1000)
        >>> print(result['counts'])

    References:
        E. B. Davies, "Quantum Theory of Open Systems," Academic Press, 1976.
    """
    
    def __init__(self, effects: List[Operator], names: Optional[List[str]] = None):
        """
        Initialize a POVM.

        Args:
            effects: List of positive operators E_i that sum to identity.
            names: Optional list of names for each outcome.

        Raises:
            ValueError: If effects do not sum to identity.
        """
        self._effects = effects
        self._names = names or [f"outcome_{i}" for i in range(len(effects))]
        self._n_outcomes = len(effects)
        self._validate()
    
    def _validate(self, tol: float = 1e-10):
        """
        Validate that the effects sum to identity.

        Args:
            tol: Numerical tolerance.

        Raises:
            ValueError: If Σ E_i ≠ I.
        """
        n = len(self._effects[0].data)
        total = self._effects[0].data
        for M in self._effects[1:]:
            total = [[total[i][j] + M.data[i][j] for j in range(n)] for i in range(n)]
        I = eye(n)
        for i in range(n):
            for j in range(n):
                if abs(total[i][j] - I[i][j]) > tol:
                    raise ValueError("Effects do not sum to identity")
    
    @property
    def effects(self) -> List[Operator]:
        """Return the list of POVM effects."""
        return self._effects
    
    @property
    def names(self) -> List[str]:
        """Return the list of outcome names."""
        return self._names
    
    @property
    def n_outcomes(self) -> int:
        """Return the number of possible outcomes."""
        return self._n_outcomes
    
    def measure(self, state: Ket, shots: int = 1) -> Dict[str, Any]:
        """
        Perform a POVM measurement.

        Args:
            state: Quantum state (must be normalized).
            shots: Number of measurements.

        Returns:
            Dictionary with keys:
                - 'outcomes': List of outcome names.
                - 'counts': Dictionary mapping outcome names to counts.
                - 'probabilities': Dictionary mapping outcome names to probabilities.
                - 'shots': Number of measurements.

        Examples:
            >>> from psiqit.quantum.state import zero
            >>> povm = povm_z_basis()
            >>> result = povm.measure(zero(), shots=1000)
            >>> print(result['counts'])  # {'0': 1000, '1': 0}
        """
        if not state.is_normalized:
            raise ValueError("Cannot measure unnormalized state")
        
        # Calculate probabilities
        probs = []
        for M in self._effects:
            prob = expectation(M, state)
            probs.append(max(0, prob))
        
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        
        # Sample outcomes
        outcome_indices = []
        for _ in range(shots):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    outcome_indices.append(i)
                    break
        
        # Convert to names
        outcomes = [self._names[i] for i in outcome_indices]
        
        # Count
        counts = {}
        for o in outcomes:
            counts[o] = counts.get(o, 0) + 1
        
        probs_dict = {self._names[i]: probs[i] for i in range(self._n_outcomes)}
        
        return {
            'outcomes': outcomes,
            'counts': counts,
            'probabilities': probs_dict,
            'shots': shots
        }


# ============================================================
# Common POVMs
# ============================================================

def povm_z_basis() -> POVM:
    """
    Create a POVM for Z-basis measurement.

    Returns:
        POVM with outcomes '0' and '1'.

    Examples:
        >>> povm = povm_z_basis()
        >>> from psiqit.quantum.state import zero
        >>> result = povm.measure(zero(), shots=1000)
        >>> print(result['counts']['0'])  # 1000
    """
    P0 = Operator([[1, 0], [0, 0]], "P0")
    P1 = Operator([[0, 0], [0, 1]], "P1")
    return POVM([P0, P1], ["0", "1"])


def povm_x_basis() -> POVM:
    """
    Create a POVM for X-basis measurement.

    Returns:
        POVM with outcomes '+' and '-'.

    Examples:
        >>> povm = povm_x_basis()
        >>> from psiqit.quantum.state import plus
        >>> result = povm.measure(plus(), shots=1000)
        >>> print(result['counts']['+'])  # Approximately 1000
    """
    v = 1 / math.sqrt(2)
    plus_state = Ket([v, v])
    minus_state = Ket([v, -v])
    P_plus = plus_state.outer(plus_state)
    P_minus = minus_state.outer(minus_state)
    return POVM([P_plus, P_minus], ["+", "-"])


def povm_y_basis() -> POVM:
    """
    Create a POVM for Y-basis measurement.

    Returns:
        POVM with outcomes 'i+' and 'i-'.

    Examples:
        >>> povm = povm_y_basis()
        >>> from psiqit.quantum.state import ip
        >>> result = povm.measure(ip(), shots=1000)
        >>> print(result['counts']['i+'])  # Approximately 1000
    """
    v = 1 / math.sqrt(2)
    ip_state = Ket([v, complex(0, v)])
    im_state = Ket([v, complex(0, -v)])
    P_ip = ip_state.outer(ip_state)
    P_im = im_state.outer(im_state)
    return POVM([P_ip, P_im], ["i+", "i-"])


# ============================================================
# Projective Measurement (Standardized)
# ============================================================

class ProjectiveMeasurement:
    """
    Projective measurement (PVM) - a special case of POVM with orthogonal projectors.

    Projective measurements correspond to observables and have outcomes that
    are the eigenvalues of the observable.

    Examples:
        >>> from psiqit.quantum.operator import pauli_z
        >>> proj_meas = ProjectiveMeasurement.from_observable(pauli_z())
        >>> result = proj_meas.measure(state, shots=1000)
        >>> print(result['counts'])  # {1.0: 500, -1.0: 500}
    """
    
    def __init__(self, projectors: List[Operator], eigenvalues: List[float], names: List[str]):
        """
        Initialize a projective measurement.

        Args:
            projectors: List of orthogonal projectors.
            eigenvalues: List of eigenvalues corresponding to each projector.
            names: List of names for each outcome.
        """
        self._projectors = projectors
        self._eigenvalues = eigenvalues
        self._names = names
        self._n_outcomes = len(projectors)
    
    @classmethod
    def from_observable(cls, observable: Operator) -> 'ProjectiveMeasurement':
        """
        Create a projective measurement from an observable.

        Args:
            observable: Hermitian operator.

        Returns:
            ProjectiveMeasurement instance.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z
            >>> meas = ProjectiveMeasurement.from_observable(pauli_z())
        """
        if not observable.is_hermitian:
            raise ValueError("Observable must be Hermitian")
        
        eig_result = observable.eigenvectors()
        eigenvalues_list = eig_result['values']
        eigenvectors_list = eig_result['vectors']
        
        projectors = []
        names = []
        
        for i, vec in enumerate(eigenvectors_list):
            ket = Ket(vec)
            if not ket.is_normalized:
                ket = ket.normalize()
            proj = ket.outer(ket)
            projectors.append(proj)
            val = eigenvalues_list[i]
            if isinstance(val, complex):
                val = val.real
            rounded_val = round(val, 10)
            names.append(f"λ={rounded_val}")
        
        return cls(projectors, eigenvalues_list, names)
    
    @property
    def projectors(self) -> List[Operator]:
        """Return the list of projectors."""
        return self._projectors
    
    @property
    def eigenvalues(self) -> List[float]:
        """Return the list of eigenvalues."""
        return self._eigenvalues
    
    def measure(self, state: Ket, shots: int = 1) -> Dict[str, Any]:
        """
        Perform a projective measurement.

        Args:
            state: Quantum state (must be normalized).
            shots: Number of measurements.

        Returns:
            Dictionary with keys:
                - 'outcomes': List of measurement outcomes (eigenvalues).
                - 'counts': Dictionary mapping eigenvalues to counts.
                - 'probabilities': List of probabilities for each outcome.
                - 'eigenvalues': List of possible eigenvalues.
                - 'shots': Number of measurements.
        """
        if not state.is_normalized:
            raise ValueError("Cannot measure unnormalized state")
        
        # Calculate probabilities
        probs = []
        for P in self._projectors:
            prob = expectation(P, state)
            probs.append(max(0, prob))
        
        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        
        # Round eigenvalues for consistency
        rounded_eigenvalues = []
        for v in self._eigenvalues:
            if isinstance(v, complex):
                v = v.real
            rounded_eigenvalues.append(round(v, 10))
        
        # Sample outcomes
        outcomes = []
        for _ in range(shots):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    outcomes.append(rounded_eigenvalues[i])
                    break
        
        # Count outcomes
        counts = {}
        for o in outcomes:
            counts[o] = counts.get(o, 0) + 1
        
        return {
            'outcomes': outcomes,
            'counts': counts,
            'probabilities': probs,
            'eigenvalues': rounded_eigenvalues,
            'shots': shots
        }


# ============================================================
# Utility Functions (Standardized)
# ============================================================

def born_rule(amplitudes: List[complex]) -> List[float]:
    """
    Apply the Born rule: probability = |amplitude|².

    Args:
        amplitudes: List of complex amplitudes.

    Returns:
        Normalized probabilities (sum to 1).

    Examples:
        >>> born_rule([0.707, 0.707])
        [0.5, 0.5]
        >>> born_rule([1, 0])
        [1.0, 0.0]
    """
    probs = [abs(a)**2 for a in amplitudes]
    total = sum(probs)
    if total > 0:
        return [p / total for p in probs]
    return probs


def measurement_statistics(state: Ket, shots: int = 1000) -> Dict[str, Any]:
    """
    Generate comprehensive measurement statistics.

    Args:
        state: Quantum state.
        shots: Number of measurements.

    Returns:
        Dictionary with keys:
            - 'probabilities': List of probabilities.
            - 'samples': List of measurement outcomes.
            - 'counts': Dictionary mapping outcomes to counts.
            - 'shots': Number of measurements.

    Examples:
        >>> from psiqit.quantum.state import plus
        >>> stats = measurement_statistics(plus(), shots=10000)
        >>> print(stats['probabilities'])  # [0.5, 0.5]
    """
    result = measure(state, shots=shots)
    return {
        'probabilities': result['probabilities'],
        'samples': result['outcomes'],
        'counts': result['counts'],
        'shots': shots
    }


# ============================================================
# State Tomography (simplified)
# ============================================================

def state_tomography(measurement_data: Dict[str, float], n_qubits: int = 1) -> List[List[complex]]:
    """
    Reconstruct a density matrix from measurement data (simplified for 1 qubit).

    For a single qubit, the density matrix can be reconstructed from the
    expectation values of the Pauli matrices:
        ρ = (I + ⟨X⟩X + ⟨Y⟩Y + ⟨Z⟩Z) / 2

    Args:
        measurement_data: Dictionary with keys 'X', 'Y', 'Z' and expectation values.
        n_qubits: Number of qubits (only 1 is currently supported).

    Returns:
        2×2 density matrix.

    Raises:
        NotImplementedError: If n_qubits > 1.

    Examples:
        >>> data = {'X': 0.0, 'Y': 0.0, 'Z': 1.0}
        >>> rho = state_tomography(data)  # |0⟩⟨0|
        >>> print(rho[0][0])  # Should be 1.0
        1.0
    """
    if n_qubits != 1:
        raise NotImplementedError("Tomography for n_qubits > 1 not yet implemented")
    
    rx = measurement_data.get('X', 0.0)
    ry = measurement_data.get('Y', 0.0)
    rz = measurement_data.get('Z', 0.0)
    
    return [
        [0.5 * (1 + rz), 0.5 * (rx - 1j * ry)],
        [0.5 * (rx + 1j * ry), 0.5 * (1 - rz)]
    ]


# ============================================================
# Exports
# ============================================================

__all__ = [
    'measure', 'measure_observable', 'expectation', 'variance', 'standard_deviation',
    'POVM', 'povm_z_basis', 'povm_x_basis', 'povm_y_basis',
    'ProjectiveMeasurement',
    'born_rule', 'measurement_statistics', 'state_tomography',
]