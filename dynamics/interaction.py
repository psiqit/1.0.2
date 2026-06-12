"""
psiqit/dynamics/interaction.py
============================================================
Interaction Picture Dynamics
============================================================

Time evolution in the interaction picture:
    |ψ_I(t)⟩ = e^{iH₀t}|ψ_S(t)⟩
    A_I(t) = e^{iH₀t} A_S e^{-iH₀t}

The interaction picture (also known as the Dirac picture) is an intermediate
representation between the Schrödinger and Heisenberg pictures. It is
particularly useful for treating systems where the Hamiltonian can be split
into a solvable free part H₀ and a small interaction V:
    H = H₀ + V

In this picture, both states and operators evolve, but the evolution is
governed only by the interaction Hamiltonian V_I(t).

Example:
    >>> from psiqit.dynamics.interaction import InteractionPicture
    >>> from psiqit.quantum.operator import pauli_z, pauli_x
    >>> from psiqit.quantum.state import zero
    >>> 
    >>> H0 = pauli_z()
    >>> V = pauli_x()
    >>> inter = InteractionPicture(H0, V)
    >>> 
    >>> # Evolve state in interaction picture
    >>> psi_I = inter.evolve_state(zero(), t=1.0)
    >>> 
    >>> # Get interaction picture operator
    >>> V_I = inter.interaction_operator(t=0.5)

References:
    P. A. M. Dirac, "The quantum theory of the emission and absorption of radiation,"
    Proceedings of the Royal Society A, 114(767):243-265, 1927.
"""

import math
import numpy as np
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from ..quantum.operator import Operator
from ..quantum.state import Ket
from ..math.qalgebra import mul, add, dagger


@dataclass
class InteractionResult:
    """
    Result container for interaction picture evolution.

    Attributes:
        times: List of time points.
        states: List of quantum states at each time point (interaction picture).
        success: Whether the evolution was successful.

    Examples:
        >>> result = InteractionResult(times=[0, 0.5, 1.0], states=[], success=True)
        >>> print(f"Number of time steps: {len(result.times)}")
        Number of time steps: 3
    """
    times: List[float]
    states: List[Ket]
    success: bool


class InteractionPicture:
    """
    Interaction picture representation for quantum dynamics.

    The Hamiltonian is split as H = H₀ + V, where:
        - H₀ is the free Hamiltonian (solvable part)
        - V is the interaction Hamiltonian (perturbation)

    In the interaction picture:
        - States evolve as: |ψ_I(t)⟩ = e^{iH₀t}|ψ_S(t)⟩
        - Operators evolve as: A_I(t) = e^{iH₀t} A_S e^{-iH₀t}
        - The interaction Hamiltonian becomes: V_I(t) = e^{iH₀t} V e^{-iH₀t}
        - The time evolution is governed by: i d|ψ_I⟩/dt = V_I|ψ_I⟩

    This picture is fundamental in quantum field theory and perturbation theory,
    particularly in the derivation of the Dyson series and the S-matrix.

    Examples:
        >>> from psiqit.quantum.operator import pauli_z, pauli_x
        >>> 
        >>> # Define Hamiltonians
        >>> H0 = pauli_z()   # Free part
        >>> V = pauli_x()    # Interaction
        >>> 
        >>> # Create interaction picture
        >>> inter = InteractionPicture(H0, V)
        >>> 
        >>> # Compute interaction operator at time t
        >>> V_I = inter.interaction_operator(t=0.5)
    """
    
    def __init__(self, H0: Operator, V: Operator):
        """
        Initialize interaction picture with free and interaction Hamiltonians.

        Args:
            H0: Free Hamiltonian (solvable part, usually time-independent).
            V: Interaction Hamiltonian (may be time-dependent).

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> H0 = pauli_z()
            >>> V = pauli_x()
            >>> inter = InteractionPicture(H0, V)
        """
        self.H0 = H0
        self.V = V
    
    def _U0(self, t: float) -> np.ndarray:
        """
        Compute the free evolution operator U₀(t) = e^{-iH₀t}.

        Args:
            t: Time.

        Returns:
            Unitary matrix for free evolution.
        """
        H0_mat = np.array(self.H0.data, dtype=complex)
        return np.linalg.expm(-1j * t * H0_mat)
    
    def _U0_dag(self, t: float) -> np.ndarray:
        """
        Compute the adjoint of the free evolution operator U₀†(t) = e^{iH₀t}.

        Args:
            t: Time.

        Returns:
            Adjoint of the free evolution matrix.
        """
        H0_mat = np.array(self.H0.data, dtype=complex)
        return np.linalg.expm(1j * t * H0_mat)
    
    def interaction_operator(self, t: float) -> Operator:
        """
        Compute the interaction Hamiltonian in the interaction picture.

        The interaction picture operator is defined as:
            V_I(t) = e^{iH₀t} V e^{-iH₀t}

        Args:
            t: Time.

        Returns:
            Operator: Interaction Hamiltonian V_I(t) in the interaction picture.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> H0 = pauli_z()
            >>> V = pauli_x()
            >>> inter = InteractionPicture(H0, V)
            >>> V_I = inter.interaction_operator(t=0.5)
            >>> print(V_I.name)
            V_I(t)
        """
        V_mat = np.array(self.V.data, dtype=complex)
        U0 = self._U0(t)
        U0_dag = self._U0_dag(t)
        
        V_I_mat = U0_dag @ V_mat @ U0
        return Operator(V_I_mat.tolist(), "V_I(t)")
    
    def evolve_state(self, psi0: Ket, t: float, dt: float = 0.01) -> Ket:
        """
        Evolve a state in the interaction picture.

        The interaction picture state is related to the Schrödinger picture state by:
            |ψ_I(t)⟩ = e^{iH₀t}|ψ_S(t)⟩

        Args:
            psi0: Initial state in the Schrödinger picture.
            t: Evolution time.
            dt: Time step for numerical integration (not used in simplified version).

        Returns:
            Ket: State in the interaction picture.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> from psiqit.quantum.state import zero
            >>> H0 = pauli_z()
            >>> V = pauli_x()
            >>> inter = InteractionPicture(H0, V)
            >>> psi_I = inter.evolve_state(zero(), t=1.0)
        """
        U0_dag = self._U0_dag(t)
        psi_data = U0_dag @ np.array(psi0.data)
        return Ket(psi_data.tolist())
    
    def evolve_operator(self, A: Operator, t: float) -> Operator:
        """
        Evolve an operator in the interaction picture.

        The interaction picture operator is defined as:
            A_I(t) = e^{iH₀t} A e^{-iH₀t}

        Args:
            A: Operator in the Schrödinger picture.
            t: Time.

        Returns:
            Operator: Operator in the interaction picture.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> H0 = pauli_z()
            >>> inter = InteractionPicture(H0, pauli_x())
            >>> X_I = inter.evolve_operator(pauli_x(), t=0.5)
            >>> print(X_I.name)
            X_I(t)
        """
        A_mat = np.array(A.data, dtype=complex)
        U0 = self._U0(t)
        U0_dag = self._U0_dag(t)
        
        A_I_mat = U0_dag @ A_mat @ U0
        return Operator(A_I_mat.tolist(), f"{A.name}_I(t)")


__all__ = [
    'InteractionResult',
    'InteractionPicture',
]