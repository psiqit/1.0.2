"""
psiqit/info/__init__.py
Quantum information theory module

This module provides fundamental measures and operations from quantum
information theory, including entropy measures, entanglement metrics,
and information-theoretic quantities.

Available functions:
    - Shannon entropy: Classical entropy for probability distributions
    - von Neumann entropy: Quantum entropy for density matrices
    - Rényi entropy: Family of generalized entropies
    - Collision entropy: Special case of Rényi entropy (α=2)
    - Mutual information: Correlation between subsystems
    - Partial trace: Tracing out subsystems from density matrices
    - Relative entropy: Distance measure between quantum states
    - Purity: Measure of mixedness (Tr(ρ²))

Example:
    >>> from psiqit.info import von_neumann_entropy, mutual_information
    >>> from psiqit.quantum.state import bell_phi_plus
    >>> 
    >>> # Bell state is maximally entangled, so reduced states are maximally mixed
    >>> bell = bell_phi_plus()
    >>> rho = bell.outer(bell)  # Density matrix
    >>> S = von_neumann_entropy(rho, base='2')
    >>> print(f"Entropy of Bell state: {S:.4f} bits")
    Entropy of Bell state: 0.0000 bits
    >>> 
    >>> # Mutual information for Bell state
    >>> I = mutual_information(rho, dim_a=2, dim_b=2, base='2')
    >>> print(f"Mutual information: {I:.4f} bits")
    Mutual information: 2.0000 bits

References:
    C. E. Shannon, "A mathematical theory of communication,"
    Bell System Technical Journal, 27(3):379-423, 1948.
    J. von Neumann, "Mathematical Foundations of Quantum Mechanics,"
    Princeton University Press, 1955.
    A. Rényi, "On measures of entropy and information,"
    Proceedings of the 4th Berkeley Symposium on Mathematics,
    Statistics and Probability, 1960.
"""

from .entropy import (
    shannon_entropy,
    von_neumann_entropy,
    renyi_entropy,
    collision_entropy,
    mutual_information,
    partial_trace,
    relative_entropy,
    purity,
)

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