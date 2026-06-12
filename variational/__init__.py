"""
psiqit/variational/__init__.py
============================================================
Variational Methods Package - Quantum Variational Algorithms
============================================================

This package provides variational quantum algorithms for optimization,
simulation, and machine learning:

    • Variational Quantum Eigensolver (VQE) - Find ground state energies
    • Quantum Approximate Optimization Algorithm (QAOA) - Solve combinatorial optimization
    • ADAPT-VQE and Subspace-Search VQE (SSVQE) - Advanced VQE variants
    • Time-Dependent Variational Principle (TDVP) - Simulate time evolution
    • Variational Monte Carlo - Stochastic optimization
    • Hartree-Fock method - Electronic structure
    • Rayleigh-Ritz method - Classical variational method

Variational quantum algorithms are hybrid quantum-classical algorithms
that use a parameterized quantum circuit (ansatz) and a classical optimizer
to minimize a cost function. They are particularly important for near-term
(NISQ) quantum computers.

Example:
    >>> from psiqit.variational import VQE, QAOA, SSVQE
    >>> from psiqit.variational import maxcut_hamiltonian
    >>> 
    >>> # VQE for ground state of a 2-qubit Hamiltonian
    >>> hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}
    >>> vqe = VQE(n_qubits=2, hamiltonian=hamiltonian)
    >>> result = vqe.run(n_iterations=100)
    >>> print(f"Ground state energy: {result.optimal_energy:.6f}")
    >>> 
    >>> # QAOA for Max-Cut on a triangle graph
    >>> edges = [(0,1), (1,2), (2,0)]
    >>> H = maxcut_hamiltonian(edges)
    >>> qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
    >>> result = qaoa.run(n_iterations=100)
    >>> 
    >>> # SSVQE for multiple excited states
    >>> ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=3)
    >>> results = ssvqe.run()

References:
    A. Peruzzo et al., "A variational eigenvalue solver on a photonic
    quantum processor," Nature Communications, 5:4213, 2014.
    E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate
    Optimization Algorithm," arXiv:1411.4028, 2014.
    J. R. McClean et al., "The theory of variational quantum algorithms,"
    arXiv:2108.08428, 2021.
"""

# ============================================================
# Variational Quantum Eigensolver (VQE)
# ============================================================

from .vqe import (
    VQEResult,
    VQE,
)

# ============================================================
# Quantum Approximate Optimization Algorithm (QAOA)
# ============================================================

from .qaoa import (
    QAOResult,
    QAOA,
    maxcut_hamiltonian,
    qubo_to_hamiltonian,
)

# ============================================================
# Advanced Variational Methods
# ============================================================

from .variational_advanced import (
    MultiStateResult,
    SSVQE,
    ADAPTVQE,
    VariationalQuantumDeflation,
)

# ============================================================
# General Variational Methods
# ============================================================

from .variational_methods import (
    VariationalResult,
    RayleighRitz,
    TDVP,
    VariationalMonteCarlo,
    HartreeFock,
)

# ============================================================
# Package Metadata
# ============================================================

__version__ = "1.0.0"
__author__ = "PSIQIT Contributors"

__all__ = [
    # VQE
    'VQEResult',
    'VQE',
    # QAOA
    'QAOResult',
    'QAOA',
    'maxcut_hamiltonian',
    'qubo_to_hamiltonian',
    # Advanced Variational
    'MultiStateResult',
    'SSVQE',
    'ADAPTVQE',
    'VariationalQuantumDeflation',
    # General Variational
    'VariationalResult',
    'RayleighRitz',
    'TDVP',
    'VariationalMonteCarlo',
    'HartreeFock',
]