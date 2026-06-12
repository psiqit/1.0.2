"""
psiqit/dynamics/heisenberg.py
Heisenberg Picture Dynamics - Simplified

This module provides tools for quantum dynamics in the Heisenberg picture,
where operators evolve in time while states remain fixed. This is in contrast
to the Schrödinger picture where states evolve and operators are constant.

The Heisenberg picture is particularly useful for:
- Computing time-dependent expectation values
- Studying the dynamics of observables
- Quantum control and optimal control theory

References:
    W. Heisenberg, "Über den anschaulichen Inhalt der quantentheoretischen Kinematik und Mechanik,"
    Zeitschrift für Physik, 43(3-4):172-198, 1927.
"""

import math
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from ..quantum.operator import Operator
from ..quantum.state import Ket
from ..math.qalgebra import commutator, mul, add, dagger, eye


@dataclass
class HeisenbergResult:
    """
    Result container for Heisenberg picture evolution.

    Attributes:
        times: List of time points.
        operators: List of evolved operators at each time point.
        success: Whether the evolution was successful.

    Examples:
        >>> result = HeisenbergResult(times=[0, 0.1, 0.2], operators=[], success=True)
        >>> print(f"Number of time steps: {len(result.times)}")
        Number of time steps: 3
    """
    times: List[float]
    operators: List[Operator]
    success: bool


class HeisenbergEvolution:
    """
    Time evolution of operators in the Heisenberg picture.

    In the Heisenberg picture, operators evolve according to:
        A(t) = e^{iHt} A e^{-iHt}

    This satisfies the Heisenberg equation of motion:
        dA/dt = i[H, A] + (∂A/∂t)

    This class provides methods to compute time-evolved operators using
    either matrix exponentiation (for small systems) or series expansion.

    Examples:
        >>> from psiqit.quantum.operator import pauli_x, pauli_z
        >>> 
        >>> # Hamiltonian for a single qubit
        >>> H = pauli_z()
        >>> 
        # Create Heisenberg evolution object
        >>> heisenberg = HeisenbergEvolution(H)
        >>> 
        # Evolve the Pauli X operator
        >>> X_t = heisenberg.evolve_operator(pauli_x(), t=1.0, method='matrix')
        >>> print(f"X(1) = {X_t.name}")
        X(1) = X(t)
    """
    
    def __init__(self, hamiltonian: Operator):
        """
        Initialize Heisenberg evolution with a given Hamiltonian.

        Args:
            hamiltonian: System Hamiltonian (Hermitian operator).

        Examples:
            >>> from psiqit.quantum.operator import pauli_z
            >>> H = pauli_z()
            >>> heisenberg = HeisenbergEvolution(H)
        """
        self.H = hamiltonian
    
    def evolve_operator(self, A: Operator, t: float, 
                        method: str = 'series', order: int = 5) -> Operator:
        """
        Evolve an operator in the Heisenberg picture.

        The evolved operator is computed as: A(t) = e^{iHt} A e^{-iHt}

        Args:
            A: Operator to evolve.
            t: Evolution time.
            method: Evolution method ('matrix' or 'series').
                   'matrix' uses matrix exponentiation (accurate for small systems).
                   'series' uses Taylor series expansion (approximate).
            order: Order of series expansion (used only for 'series' method).

        Returns:
            Operator: Time-evolved operator A(t).

        Examples:
            >>> from psiqit.quantum.operator import pauli_x, pauli_z
            >>> H = pauli_z()
            >>> heisenberg = HeisenbergEvolution(H)
            >>> 
            >>> # Using matrix exponentiation
            >>> X_t = heisenberg.evolve_operator(pauli_x(), t=np.pi/2, method='matrix')
            >>> 
            >>> # Using series expansion
            >>> X_t_approx = heisenberg.evolve_operator(pauli_x(), t=0.1, method='series', order=3)
        """
        if method == 'matrix':
            # For small systems, use matrix exponential
            import numpy as np
            H_mat = np.array(self.H.data, dtype=complex)
            A_mat = np.array(A.data, dtype=complex)
            
            U_mat = np.linalg.expm(1j * t * H_mat)
            U_dag_mat = np.linalg.expm(-1j * t * H_mat)
            
            A_t_mat = U_mat @ A_mat @ U_dag_mat
            return Operator(A_t_mat.tolist(), f"{A.name}(t)")
        
        else:
            # Series expansion: A(t) = A + it[H, A] + (it)²/2 [H, [H, A]] + ...
            A_t_mat = [row[:] for row in A.data]  # Copy
            term_mat = [row[:] for row in A.data]  # Copy
            factor = 1.0
            
            for n in range(1, order):
                factor *= 1j * t / n
                # Compute commutator [H, term]
                term_mat = commutator(self.H.data, term_mat)
                # Add to result
                for i in range(len(term_mat)):
                    for j in range(len(term_mat[0])):
                        A_t_mat[i][j] += factor * term_mat[i][j]
            
            return Operator(A_t_mat, f"{A.name}(t)")


class HeisenbergEquation:
    """
    Heisenberg equations of motion for quantum operators.

    The Heisenberg equation of motion describes the time evolution of an
    operator in the Heisenberg picture:
        dA/dt = i[H, A] + ∂A/∂t

    For operators with no explicit time dependence (∂A/∂t = 0), this simplifies
    to dA/dt = i[H, A].

    Examples:
        >>> from psiqit.quantum.operator import pauli_x, pauli_z
        >>> 
        >>> # Hamiltonian for a single qubit
        >>> H = pauli_z()
        >>> 
        >>> # Compute equation of motion for Pauli X
        >>> eq = HeisenbergEquation(H)
        >>> dX_dt = eq.equation_of_motion(pauli_x())
        >>> print(f"dX/dt = {dX_dt.name}")
        dX/dt = dX/dt
    """
    
    def __init__(self, hamiltonian: Operator):
        """
        Initialize Heisenberg equation with a given Hamiltonian.

        Args:
            hamiltonian: System Hamiltonian (Hermitian operator).

        Examples:
            >>> from psiqit.quantum.operator import pauli_z
            >>> H = pauli_z()
            >>> eq = HeisenbergEquation(H)
        """
        self.H = hamiltonian
    
    def equation_of_motion(self, A: Operator) -> Operator:
        """
        Compute the Heisenberg equation of motion for an operator.

        The equation of motion is: dA/dt = i[H, A] + ∂A/∂t
        For simplicity, this implementation assumes ∂A/∂t = 0.

        Args:
            A: Operator for which to compute the equation of motion.

        Returns:
            Operator: The time derivative dA/dt.

        Examples:
            >>> from psiqit.quantum.operator import pauli_x, pauli_z
            >>> H = pauli_z()
            >>> eq = HeisenbergEquation(H)
            >>> dX_dt = eq.equation_of_motion(pauli_x())
            >>> # For H = Z, [Z, X] = -2iY, so dX/dt = i[Z, X] = 2Y
        """
        comm = commutator(self.H.data, A.data)
        dA_dt_data = [[1j * comm[i][j] for j in range(len(comm))] 
                      for i in range(len(comm))]
        return Operator(dA_dt_data, f"d{A.name}/dt")


__all__ = [
    'HeisenbergResult',
    'HeisenbergEvolution',
    'HeisenbergEquation',
]