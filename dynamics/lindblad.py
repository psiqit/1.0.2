"""
psiqit/dynamics/lindblad.py
============================================================
Lindblad Master Equation
============================================================

Time evolution of open quantum systems using the Lindblad master equation:
    dρ/dt = -i[H, ρ] + Σ γᵢ (Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})

The Lindblad master equation describes the dynamics of open quantum systems
that interact with their environment. It is the most general Markovian
evolution that preserves complete positivity and trace.

This module provides:
- Numerical integration of the Lindblad equation (Euler and RK4 methods)
- Steady state calculation
- Support for multiple collapse operators with different decay rates

Example:
    >>> from psiqit.dynamics.lindblad import LindbladSolver
    >>> from psiqit.quantum.operator import pauli_z, pauli_x
    >>> import numpy as np
    >>> 
    >>> # Hamiltonian
    >>> H = pauli_z()
    >>> 
    # Collapse operators (e.g., relaxation)
    >>> collapse_ops = [pauli_x()]
    >>> 
    >>> # Initial density matrix (|0⟩⟨0|)
    >>> rho0 = np.array([[1, 0], [0, 0]])
    >>> 
    >>> # Solve Lindblad equation
    >>> solver = LindbladSolver(H, collapse_ops, gamma=[0.1])
    >>> result = solver.evolve(rho0, t_max=10.0, dt=0.01)
    >>> 
    >>> # Find steady state
    >>> rho_ss = solver.steady_state()
    >>> print(f"Steady state: {rho_ss}")

References:
    G. Lindblad, "On the generators of quantum dynamical semigroups,"
    Communications in Mathematical Physics, 48(2):119-130, 1976.
    V. Gorini, A. Kossakowski, and E. C. G. Sudarshan,
    "Completely positive dynamical semigroups of N-level systems,"
    Journal of Mathematical Physics, 17(5):821-825, 1976.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from ..quantum.operator import Operator
from ..quantum.state import Ket
from ..math.qalgebra import mul, add, dagger, trace


@dataclass
class LindbladResult:
    """
    Result container for Lindblad master equation evolution.

    Attributes:
        times: List of time points.
        states: List of density matrices at each time point.
        populations: List of populations for each basis state at each time point.
        success: Whether the evolution was successful.

    Examples:
        >>> result = LindbladResult(times=[0, 1, 2], states=[], populations=[[1,0],[0.8,0.2]], success=True)
        >>> print(f"Final populations: {result.populations[-1]}")
        Final populations: [0.8, 0.2]
    """
    times: List[float]
    states: List[List[List[complex]]]
    populations: List[List[float]]
    success: bool


class LindbladSolver:
    """
    Solver for the Lindblad master equation.

    The Lindblad master equation describes the time evolution of the density
    matrix ρ(t) for an open quantum system:

        dρ/dt = -i[H, ρ] + Σ γᵢ (Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})

    where:
        - H is the system Hamiltonian
        - Lᵢ are collapse (jump) operators
        - γᵢ are decay rates (positive real numbers)

    This solver provides numerical integration using Euler or Runge-Kutta
    methods, as well as steady state calculation via finding the nullspace
    of the Liouvillian superoperator.

    Examples:
        >>> from psiqit.quantum.operator import pauli_z, pauli_x
        >>> import numpy as np
        >>> 
        >>> # Two-level system with decay
        >>> H = pauli_z()
        >>> collapse_ops = [pauli_x()]  # Relaxation operator
        >>> solver = LindbladSolver(H, collapse_ops, gamma=[0.1])
        >>> 
        >>> # Initial state |0⟩⟨0|
        >>> rho0 = np.array([[1, 0], [0, 0]])
        >>> 
        >>> # Evolve
        >>> result = solver.evolve(rho0, t_max=5.0, dt=0.01)
        >>> 
        >>> # Plot population dynamics
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result.times, [p[0] for p in result.populations])
        >>> plt.xlabel('Time')
        >>> plt.ylabel('Population')
        >>> plt.show()
    """
    
    def __init__(self, hamiltonian: Operator, collapse_ops: List[Operator],
                 gamma: List[float] = None):
        """
        Initialize the Lindblad solver.

        Args:
            hamiltonian: System Hamiltonian H (Hermitian operator).
            collapse_ops: List of collapse operators Lᵢ (jump operators).
            gamma: Decay rates for each collapse operator.
                  If None, all rates are set to 1.0.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> 
            >>> # Single decay channel
            >>> solver = LindbladSolver(pauli_z(), [pauli_x()], gamma=[0.1])
            >>> 
            >>> # Multiple decay channels
            >>> solver = LindbladSolver(pauli_z(), [pauli_x(), pauli_y()], gamma=[0.1, 0.05])
        """
        self.H = hamiltonian
        self.collapse_ops = collapse_ops
        self.gamma = gamma if gamma else [1.0] * len(collapse_ops)
        self.dim = hamiltonian.dim
    
    def _lindbladian(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute the Lindbladian superoperator acting on the density matrix.

        The Lindbladian is defined as:
            L(ρ) = -i[H, ρ] + Σ γᵢ (Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})

        Args:
            rho: Density matrix.

        Returns:
            The derivative dρ/dt.
        """
        # -i[H, ρ]
        H_rho = self.H.data @ rho
        rho_H = rho @ self.H.data
        drho = -1j * (H_rho - rho_H)
        
        # Σ γᵢ (Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})
        for L, gamma in zip(self.collapse_ops, self.gamma):
            L_mat = np.array(L.data)
            L_dag = L_mat.conj().T
            
            L_rho_Ld = L_mat @ rho @ L_dag
            LdL_rho = (L_dag @ L_mat) @ rho
            rho_LdL = rho @ (L_dag @ L_mat)
            
            drho += gamma * (L_rho_Ld - 0.5 * (LdL_rho + rho_LdL))
        
        return drho
    
    def _rho_to_vec(self, rho: np.ndarray) -> np.ndarray:
        """
        Vectorize a density matrix column-wise.

        Args:
            rho: Density matrix.

        Returns:
            Vectorized density matrix.
        """
        return rho.flatten('F')
    
    def _vec_to_rho(self, vec: np.ndarray) -> np.ndarray:
        """
        Unvectorize a vector back to a density matrix.

        Args:
            vec: Vectorized density matrix.

        Returns:
            Density matrix.
        """
        return vec.reshape(self.dim, self.dim, order='F')
    
    def evolve(self, rho0: np.ndarray, t_max: float, dt: float = 0.01,
               method: str = 'euler') -> LindbladResult:
        """
        Evolve the density matrix using the Lindblad master equation.

        Args:
            rho0: Initial density matrix.
            t_max: Maximum evolution time.
            dt: Time step for integration.
            method: Integration method ('euler' or 'rk4').

        Returns:
            LindbladResult containing time points, states, and populations.

        Examples:
            >>> import numpy as np
            >>> from psiqit.quantum.operator import pauli_z
            >>> 
            >>> solver = LindbladSolver(pauli_z(), [], gamma=[])
            >>> rho0 = np.array([[1, 0], [0, 0]])
            >>> result = solver.evolve(rho0, t_max=1.0, dt=0.01, method='rk4')
            >>> print(f"Number of time steps: {len(result.times)}")
            Number of time steps: 100
        """
        times = np.arange(0, t_max, dt)
        states = [rho0.copy()]
        
        rho = rho0.copy()
        
        for t in times[1:]:
            if method == 'euler':
                drho = self._lindbladian(rho)
                rho = rho + dt * drho
            elif method == 'rk4':
                k1 = self._lindbladian(rho)
                k2 = self._lindbladian(rho + dt/2 * k1)
                k3 = self._lindbladian(rho + dt/2 * k2)
                k4 = self._lindbladian(rho + dt * k3)
                rho = rho + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            
            # Ensure positivity (simplified)
            rho = (rho + rho.conj().T) / 2
            states.append(rho.copy())
        
        # Compute populations
        populations = [[abs(rho[i][i]) for i in range(self.dim)] for rho in states]
        
        return LindbladResult(
            times=times.tolist(),
            states=[s.tolist() for s in states],
            populations=populations,
            success=True
        )
    
    def steady_state(self, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
        """
        Find the steady state of the Lindblad master equation.

        The steady state satisfies dρ/dt = 0, which means it is the nullspace
        (eigenvector with eigenvalue 0) of the Liouvillian superoperator.

        Args:
            tol: Numerical tolerance for eigenvalue identification.
            max_iter: Maximum iterations (not used in current implementation).

        Returns:
            Steady state density matrix.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> 
            >>> solver = LindbladSolver(pauli_z(), [pauli_x()], gamma=[0.1])
            >>> rho_ss = solver.steady_state()
            >>> print(f"Trace: {np.trace(rho_ss):.6f}")
            Trace: 1.000000
        """
        # Vectorize Lindbladian
        dim = self.dim
        L = np.zeros((dim*dim, dim*dim), dtype=complex)
        
        # Build superoperator matrix
        for i in range(dim):
            for j in range(dim):
                rho_ij = np.zeros((dim, dim))
                rho_ij[i, j] = 1.0
                vec_ij = self._rho_to_vec(rho_ij)
                Lvec = self._rho_to_vec(self._lindbladian(rho_ij))
                L[:, i*dim + j] = Lvec
        
        # Find eigenvector with eigenvalue 0
        eigvals, eigvecs = np.linalg.eig(L)
        idx = np.argmin(np.abs(eigvals))
        rho_ss_vec = eigvecs[:, idx]
        rho_ss = self._vec_to_rho(rho_ss_vec)
        
        # Normalize
        rho_ss = rho_ss / trace(rho_ss.tolist())
        
        # Ensure Hermiticity
        rho_ss = (rho_ss + rho_ss.conj().T) / 2
        
        return rho_ss


__all__ = [
    'LindbladResult',
    'LindbladSolver',
]