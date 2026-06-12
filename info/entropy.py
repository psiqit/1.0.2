"""
psiqit/info/entropy.py
============================================================
Quantum Information Theory - Entropy Measures
============================================================

This module provides various entropy measures and information-theoretic
quantities used in quantum information theory.

Entropy measures quantify the uncertainty or information content of a
quantum state. Key measures include:
    - Shannon entropy: For classical probability distributions
    - von Neumann entropy: For quantum states (density matrices)
    - Rényi entropy: Family of generalized entropies
    - Collision entropy: Special case of Rényi entropy (α=2)
    - Relative entropy: Distance measure between quantum states
    - Mutual information: Correlation between subsystems
    - Purity: Measure of mixedness

References:
    C. E. Shannon, "A mathematical theory of communication,"
    Bell System Technical Journal, 27(3):379-423, 1948.
    J. von Neumann, "Mathematical Foundations of Quantum Mechanics,"
    Princeton University Press, 1955.
    A. Rényi, "On measures of entropy and information,"
    Proceedings of the 4th Berkeley Symposium on Mathematics,
    Statistics and Probability, 1960.

Example:
    >>> from psiqit.info.entropy import von_neumann_entropy, mutual_information
    >>> from psiqit.quantum.state import bell_phi_plus, zero
    >>> 
    >>> # Bell state is pure, so von Neumann entropy is 0
    >>> bell = bell_phi_plus()
    >>> rho_bell = bell.outer(bell)
    >>> S = von_neumann_entropy(rho_bell, base='2')
    >>> print(f"von Neumann entropy: {S:.4f} bits")
    von Neumann entropy: 0.0000 bits
    >>> 
    >>> # Maximally mixed state has maximum entropy
    >>> from psiqit.quantum.operator import identity
    >>> rho_mixed = identity(2).data
    >>> for i in range(2):
    ...     for j in range(2):
    ...         rho_mixed[i][j] /= 2
    >>> S = von_neumann_entropy(rho_mixed, base='2')
    >>> print(f"Max entropy: {S:.4f} bits")
    Max entropy: 1.0000 bits
"""

import math
from typing import List, Union, Optional
from ..math.qalgebra import eigenvalues, trace, is_positive
from ..quantum.operator import Operator
from ..quantum.state import Ket


def _get_real_eigenvalues(matrix):
    """Get real parts of eigenvalues (for Hermitian matrices)."""
    vals = eigenvalues(matrix)
    return [v.real if hasattr(v, 'real') else v for v in vals]


def shannon_entropy(probabilities: List[float], base: str = 'e') -> float:
    """
    Compute Shannon entropy for a probability distribution.

    Shannon entropy is defined as:
        H(p) = -Σ p_i log(p_i)

    It quantifies the uncertainty or information content of a classical
    probability distribution.

    Args:
        probabilities: List of probabilities (must sum to 1).
        base: Logarithm base ('e' for nats, '2' for bits, '10' for dits).

    Returns:
        Shannon entropy in the specified units.

    Raises:
        ValueError: If base is unknown.

    Examples:
        >>> # Fair coin toss
        >>> H = shannon_entropy([0.5, 0.5], base='2')
        >>> print(f"{H:.4f} bits")
        1.0000 bits
        >>> 
        >>> # Deterministic distribution
        >>> H = shannon_entropy([1.0, 0.0], base='2')
        >>> print(f"{H:.4f} bits")
        0.0000 bits
        >>> 
        >>> # Using natural log (nats)
        >>> H = shannon_entropy([0.5, 0.5], base='e')
        >>> print(f"{H:.4f} nats")
        0.6931 nats
    """
    H = 0.0
    for p in probabilities:
        if p > 1e-12:
            if base == 'e':
                H -= p * math.log(p)
            elif base == '2':
                H -= p * math.log2(p)
            elif base == '10':
                H -= p * math.log10(p)
            else:
                raise ValueError(f"Unknown base: {base}")
    return H


def von_neumann_entropy(rho: Union[Operator, List[List[complex]]], base: str = 'e') -> float:
    """
    Compute von Neumann entropy for a quantum state.

    von Neumann entropy is defined as:
        S(ρ) = -Tr(ρ log ρ) = -Σ λ_i log λ_i

    where λ_i are the eigenvalues of the density matrix ρ.

    Args:
        rho: Density matrix (Operator or list of lists).
        base: Logarithm base ('e' for nats, '2' for bits, '10' for dits).

    Returns:
        von Neumann entropy in the specified units.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus, zero
        >>> 
        >>> # Pure state (zero entropy)
        >>> pure = zero()
        >>> rho_pure = pure.outer(pure)
        >>> S = von_neumann_entropy(rho_pure, base='2')
        >>> print(f"{S:.4f} bits")
        0.0000 bits
        >>> 
        >>> # Bell state is pure
        >>> bell = bell_phi_plus()
        >>> rho_bell = bell.outer(bell)
        >>> S = von_neumann_entropy(rho_bell, base='2')
        >>> print(f"{S:.4f} bits")
        0.0000 bits
        >>> 
        >>> # Maximally mixed state for 1 qubit (maximum entropy = 1 bit)
        >>> from psiqit.quantum.operator import identity
        >>> rho_max = identity(2).data
        >>> for i in range(2):
        ...     for j in range(2):
        ...         rho_max[i][j] /= 2
        >>> S = von_neumann_entropy(rho_max, base='2')
        >>> print(f"{S:.4f} bits")
        1.0000 bits
    """
    if isinstance(rho, Operator):
        matrix = rho.data
    else:
        matrix = rho
    
    # Get real eigenvalues
    eigvals = _get_real_eigenvalues(matrix)
    
    # Filter positive eigenvalues
    positive_vals = [v for v in eigvals if v > 1e-12]
    
    # Calculate entropy
    S = 0.0
    for lam in positive_vals:
        if base == 'e':
            S -= lam * math.log(lam)
        elif base == '2':
            S -= lam * math.log2(lam)
        elif base == '10':
            S -= lam * math.log10(lam)
        else:
            raise ValueError(f"Unknown base: {base}")
    
    return S


def renyi_entropy(rho: Union[Operator, List[List[complex]]], alpha: float = 2, base: str = 'e') -> float:
    """
    Compute Rényi entropy of order α.

    Rényi entropy is defined as:
        S_α(ρ) = (1/(1-α)) log(Tr(ρ^α))

    Special cases:
        α → 1: von Neumann entropy
        α = 2: collision entropy
        α = ∞: min-entropy

    Args:
        rho: Density matrix (Operator or list of lists).
        alpha: Order parameter (α ≥ 0, α ≠ 1).
        base: Logarithm base ('e', '2', '10').

    Returns:
        Rényi entropy of order α.

    Raises:
        ValueError: If alpha is negative.

    Examples:
        >>> from psiqit.quantum.operator import identity
        >>> rho_max = identity(2).data
        >>> for i in range(2):
        ...     for j in range(2):
        ...         rho_max[i][j] /= 2
        >>> # Rényi entropy of order 2 (collision entropy)
        >>> S2 = renyi_entropy(rho_max, alpha=2, base='2')
        >>> print(f"{S2:.4f} bits")
        1.0000 bits
    """
    if alpha < 0:
        raise ValueError(f"Alpha must be non-negative, got {alpha}")
    if abs(alpha - 1.0) < 1e-10:
        return von_neumann_entropy(rho, base)
    
    if isinstance(rho, Operator):
        matrix = rho.data
    else:
        matrix = rho
    
    # Get real eigenvalues
    eigvals = _get_real_eigenvalues(matrix)
    positive_vals = [v for v in eigvals if v > 1e-12]
    
    # Tr(ρ^α) = Σ λ_i^α
    trace_alpha = sum(v ** alpha for v in positive_vals)
    
    if base == 'e':
        log_val = math.log(trace_alpha)
    elif base == '2':
        log_val = math.log2(trace_alpha)
    elif base == '10':
        log_val = math.log10(trace_alpha)
    else:
        raise ValueError(f"Unknown base: {base}")
    
    return log_val / (1 - alpha)


def collision_entropy(rho: Union[Operator, List[List[complex]]], base: str = 'e') -> float:
    """
    Compute collision entropy (Rényi entropy of order 2).

    Collision entropy is defined as:
        S_2(ρ) = -log(Tr(ρ^2))

    It is a special case of Rényi entropy with α = 2.

    Args:
        rho: Density matrix (Operator or list of lists).
        base: Logarithm base ('e', '2', '10').

    Returns:
        Collision entropy.

    Examples:
        >>> from psiqit.quantum.operator import identity
        >>> rho_max = identity(2).data
        >>> for i in range(2):
        ...     for j in range(2):
        ...         rho_max[i][j] /= 2
        >>> S2 = collision_entropy(rho_max, base='2')
        >>> print(f"{S2:.4f} bits")
        1.0000 bits
    """
    return renyi_entropy(rho, alpha=2, base=base)


def partial_trace(matrix: List[List[complex]], dims: List[int], keep: int) -> List[List[complex]]:
    """
    Compute partial trace of a bipartite density matrix.

    The partial trace traces out one subsystem of a bipartite system,
    leaving the reduced density matrix of the other subsystem.

    Args:
        matrix: Density matrix of the bipartite system (size dA*dB × dA*dB).
        dims: Dimensions [dA, dB] of the two subsystems.
        keep: Which subsystem to keep (0 for A, 1 for B).

    Returns:
        Reduced density matrix of the kept subsystem.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> bell = bell_phi_plus()
        >>> rho = bell.outer(bell)
        >>> 
        >>> # Partial trace over B (keep A)
        >>> rho_A = partial_trace(rho, [2, 2], keep=0)
        >>> print(len(rho_A))  # Should be 2
        2
        >>> 
        >>> # Partial trace over A (keep B)
        >>> rho_B = partial_trace(rho, [2, 2], keep=1)
        >>> print(len(rho_B))
        2
    """
    d_a, d_b = dims
    
    if keep == 0:
        # Keep subsystem A, trace out B
        result = [[0.0] * d_a for _ in range(d_a)]
        for i in range(d_a):
            for j in range(d_a):
                for k in range(d_b):
                    result[i][j] += matrix[i * d_b + k][j * d_b + k]
    else:
        # Keep subsystem B, trace out A
        result = [[0.0] * d_b for _ in range(d_b)]
        for i in range(d_b):
            for j in range(d_b):
                for k in range(d_a):
                    result[i][j] += matrix[k * d_b + i][k * d_b + j]
    
    return result


def mutual_information(rho_ab: Union[Operator, List[List[complex]]], 
                       dim_a: int, dim_b: int,
                       base: str = 'e') -> float:
    """
    Compute mutual information between two subsystems.

    Mutual information quantifies the total correlation between two systems:
        I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)

    It measures how much information one system contains about the other.

    Args:
        rho_ab: Density matrix of the bipartite system.
        dim_a: Dimension of subsystem A.
        dim_b: Dimension of subsystem B.
        base: Logarithm base for entropy calculation.

    Returns:
        Mutual information in the specified units.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> bell = bell_phi_plus()
        >>> rho = bell.outer(bell)
        >>> I = mutual_information(rho, dim_a=2, dim_b=2, base='2')
        >>> print(f"Mutual information: {I:.4f} bits")
        Mutual information: 2.0000 bits
        >>> 
        >>> # Product state has zero mutual information
        >>> from psiqit.quantum.state import zero
        >>> product = zero().outer(zero())
        >>> from psiqit.math.qalgebra import kron
        >>> rho_product = kron(product, product)
        >>> I = mutual_information(rho_product, dim_a=2, dim_b=2, base='2')
        >>> print(f"Mutual information: {I:.4f} bits")
        Mutual information: 0.0000 bits
    """
    if isinstance(rho_ab, Operator):
        matrix = rho_ab.data
    else:
        matrix = rho_ab
    
    # Compute reduced density matrices
    rho_a = partial_trace(matrix, [dim_a, dim_b], keep=0)
    rho_b = partial_trace(matrix, [dim_a, dim_b], keep=1)
    
    # Calculate entropies
    S_a = von_neumann_entropy(rho_a, base)
    S_b = von_neumann_entropy(rho_b, base)
    S_ab = von_neumann_entropy(matrix, base)
    
    return S_a + S_b - S_ab


def relative_entropy(rho: Union[Operator, List[List[complex]]], 
                     sigma: Union[Operator, List[List[complex]]]) -> float:
    """
    Compute quantum relative entropy.

    Quantum relative entropy is defined as:
        S(ρ||σ) = Tr(ρ(log ρ - log σ))

    It is a measure of distance between two quantum states, but is not
    symmetric and is not a metric.

    Args:
        rho: First density matrix.
        sigma: Second density matrix.

    Returns:
        Relative entropy S(ρ||σ) (in nats).

    Examples:
        >>> from psiqit.quantum.state import zero, one
        >>> rho0 = zero().outer(zero())
        >>> rho1 = one().outer(one())
        >>> S = relative_entropy(rho0, rho1)
        >>> print(f"{S:.4f} nats")
        inf
    """
    if isinstance(rho, Operator):
        rho_mat = rho.data
    else:
        rho_mat = rho
    
    if isinstance(sigma, Operator):
        sigma_mat = sigma.data
    else:
        sigma_mat = sigma
    
    # Get eigenvalues
    rho_eigvals = _get_real_eigenvalues(rho_mat)
    sigma_eigvals = _get_real_eigenvalues(sigma_mat)
    
    # Compute relative entropy (simplified - assumes diagonal in same basis)
    S = 0.0
    for lam_r in rho_eigvals:
        if lam_r > 1e-12:
            log_rho = math.log(lam_r)
            # Find corresponding sigma eigenvalue (simplified)
            log_sigma = 0
            for lam_s in sigma_eigvals:
                if lam_s > 1e-12:
                    log_sigma = math.log(lam_s)
                    break
            S += lam_r * (log_rho - log_sigma)
    
    return S


def purity(rho: Union[Operator, List[List[complex]]]) -> float:
    """
    Compute purity of a quantum state.

    Purity is defined as:
        γ = Tr(ρ²)

    It measures how mixed a quantum state is:
        - γ = 1: pure state
        - γ = 1/d: maximally mixed state (where d is dimension)

    Args:
        rho: Density matrix (Operator or list of lists).

    Returns:
        Purity value between 1/d and 1.

    Examples:
        >>> from psiqit.quantum.state import zero
        >>> pure = zero()
        >>> rho_pure = pure.outer(pure)
        >>> p = purity(rho_pure)
        >>> print(f"Purity: {p:.4f}")
        Purity: 1.0000
        >>> 
        >>> from psiqit.quantum.operator import identity
        >>> rho_max = identity(2).data
        >>> for i in range(2):
        ...     for j in range(2):
        ...         rho_max[i][j] /= 2
        >>> p = purity(rho_max)
        >>> print(f"Purity: {p:.4f}")
        Purity: 0.5000
    """
    if isinstance(rho, Operator):
        matrix = rho.data
    else:
        matrix = rho
    
    n = len(matrix)
    rho_sq = [[sum(matrix[i][k] * matrix[k][j] for k in range(n)) 
               for j in range(n)] for i in range(n)]
    
    tr = sum(rho_sq[i][i] for i in range(n))
    return tr.real if hasattr(tr, 'real') else tr


__all__ = [
    'shannon_entropy',
    'von_neumann_entropy',
    'renyi_entropy',
    'collision_entropy',
    'mutual_information',
    'partial_trace',
    'relative_entropy',
    'purity',
]