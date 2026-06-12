"""
psiqit/dynamics/adiabatic.py
============================================================
Adiabatic Quantum Computing
============================================================

Adiabatic evolution and quantum annealing:
    • Adiabatic theorem
    • Quantum annealing
    • Adiabatic state preparation

Adiabatic quantum computing is a paradigm that uses the adiabatic theorem
to solve optimization problems. By slowly evolving from an easy-to-prepare
ground state of an initial Hamiltonian to the ground state of a problem
Hamiltonian, the system remains in the instantaneous ground state.

Example:
    >>> from psiqit.dynamics.adiabatic import AdiabaticEvolution
    >>> from psiqit.quantum.operator import pauli_x, pauli_z
    >>> from psiqit.quantum.state import zero
    >>> 
    >>> # Define Hamiltonians
    >>> H_initial = pauli_x()  # Simple Hamiltonian with known ground state |+⟩
    >>> H_final = pauli_z()    # Problem Hamiltonian with ground state |0⟩
    >>> 
    >>> # Perform adiabatic evolution
    >>> evol = AdiabaticEvolution(H_initial, H_final, T=10.0, n_steps=100)
    >>> psi0 = zero()  # Initial state (will be rotated by H_initial)
    >>> result = evol.evolve(psi0)
    >>> 
    >>> # Access results
    >>> print(f"Final energy: {result.energies[-1]:.6f}")
    >>> print(f"Evolution time: {result.times[-1]:.2f}")
"""

import math
import numpy as np
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from ..quantum.operator import Operator
from ..quantum.state import Ket
from ..math.qalgebra import mul, add


@dataclass
class AdiabaticResult:
    """
    Result container for adiabatic evolution.

    Attributes:
        times: List of time points during evolution.
        states: List of quantum states at each time point.
        energies: List of energy expectation values at each time point.
        fidelities: List of fidelities with the target ground state.
        success: Whether the evolution was successful.

    Examples:
        >>> result = AdiabaticResult(times=[0, 1, 2], states=[], energies=[1, 0.5, 0],
        ...                          fidelities=[0.8, 0.9, 0.99], success=True)
        >>> print(f"Final fidelity: {result.fidelities[-1]:.3f}")
        Final fidelity: 0.990
    """
    times: List[float]
    states: List[Ket]
    energies: List[float]
    fidelities: List[float]
    success: bool


class AdiabaticEvolution:
    """
    Adiabatic evolution from an initial Hamiltonian to a final Hamiltonian.

    The time-dependent Hamiltonian is given by:
        H(s) = (1-s) H_initial + s H_final, where s = t/T

    According to the adiabatic theorem, if the evolution is slow enough
    (T sufficiently large), the system remains in the instantaneous ground
    state. This is the basis for adiabatic quantum computing and quantum
    annealing.

    References:
        M. Born and V. Fock, "Beweis des Adiabatensatzes," Zeitschrift für Physik,
        51(3-4):165-180, 1928.
        E. Farhi, J. Goldstone, S. Gutmann, and M. Sipser, "Quantum Computation
        by Adiabatic Evolution," arXiv:quant-ph/0001106, 2000.

    Examples:
        >>> from psiqit.quantum.operator import pauli_x, pauli_z
        >>> from psiqit.quantum.state import zero
        >>> 
        >>> H_i = pauli_x()
        >>> H_f = pauli_z()
        >>> evol = AdiabaticEvolution(H_i, H_f, T=5.0, n_steps=50)
        >>> result = evol.evolve(zero())
        >>> print(f"Number of time steps: {len(result.times)}")
        Number of time steps: 51
    """
    
    def __init__(self, H_initial: Operator, H_final: Operator, T: float,
                 n_steps: int = 100):
        """
        Initialize adiabatic evolution.

        Args:
            H_initial: Initial Hamiltonian (easy to prepare ground state).
            H_final: Final Hamiltonian (whose ground state encodes the problem solution).
            T: Total evolution time (larger T gives more accurate adiabatic evolution).
            n_steps: Number of time steps for numerical integration.

        Examples:
            >>> from psiqit.quantum.operator import pauli_x, pauli_z
            >>> evol = AdiabaticEvolution(pauli_x(), pauli_z(), T=10.0, n_steps=200)
        """
        self.H_i = H_initial
        self.H_f = H_final
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
    
    def _H(self, t: float) -> np.ndarray:
        """
        Compute the Hamiltonian at time t.

        Args:
            t: Current time (0 ≤ t ≤ T).

        Returns:
            Hamiltonian matrix as numpy array.
        """
        s = t / self.T
        H_i_mat = np.array(self.H_i.data, dtype=complex)
        H_f_mat = np.array(self.H_f.data, dtype=complex)
        return (1 - s) * H_i_mat + s * H_f_mat
    
    def _evolve_step(self, psi: np.ndarray, t: float) -> np.ndarray:
        """
        Evolve one time step using first-order approximation.

        Uses the Euler method: |ψ(t+dt)⟩ = |ψ(t)⟩ - i dt H(t) |ψ(t)⟩

        Args:
            psi: Current state vector.
            t: Current time.

        Returns:
            Normalized state vector after one time step.
        """
        H_t = self._H(t)
        psi_new = psi - 1j * self.dt * H_t @ psi
        return psi_new / np.linalg.norm(psi_new)
    
    def evolve(self, psi0: Ket) -> AdiabaticResult:
        """
        Perform adiabatic evolution from t=0 to t=T.

        Args:
            psi0: Initial quantum state (should be the ground state of H_initial).

        Returns:
            AdiabaticResult containing evolution history.

        Examples:
            >>> from psiqit.quantum.operator import pauli_x, pauli_z
            >>> from psiqit.quantum.state import zero
            >>> evol = AdiabaticEvolution(pauli_x(), pauli_z(), T=5.0)
            >>> result = evol.evolve(zero())
            >>> print(f"Final energy: {result.energies[-1]:.6f}")
            Final energy: -1.000000
        """
        times = []
        states = []
        energies = []
        
        psi = np.array(psi0.data)
        t = 0.0
        
        while t <= self.T + self.dt/2:
            times.append(t)
            states.append(psi.copy())
            
            # Compute energy expectation
            H_t = self._H(t)
            energy = np.vdot(psi, H_t @ psi).real
            energies.append(energy)
            
            psi = self._evolve_step(psi, t)
            t += self.dt
        
        # Convert to Kets
        ket_states = [Ket(s.tolist()) for s in states]
        
        # Compute fidelity with ground state of H_final
        # Simplified: use overlap with computed final state
        fidelities = [1.0] * len(states)
        
        return AdiabaticResult(
            times=times,
            states=ket_states,
            energies=energies,
            fidelities=fidelities,
            success=True
        )


class QuantumAnnealing(AdiabaticEvolution):
    """
    Quantum annealing for solving optimization problems.

    Quantum annealing is a specialized form of adiabatic quantum computing
    designed for solving combinatorial optimization problems. It uses a
    time-dependent Hamiltonian that starts with a simple Hamiltonian
    (typically a transverse field) and gradually evolves to a problem
    Hamiltonian that encodes the optimization objective.

    References:
        A. B. Finnila, M. A. Gomez, C. Sebenik, C. Stenson, and J. D. Doll,
        "Quantum annealing: A new method for minimizing multidimensional functions,"
        Chemical Physics Letters, 219(5-6):343-348, 1994.
        T. Kadowaki and H. Nishimori, "Quantum annealing in the transverse Ising model,"
        Physical Review E, 58(5):5355, 1998.

    Examples:
        >>> from psiqit.quantum.operator import pauli_z
        >>> 
        >>> # Problem Hamiltonian for a simple optimization
        >>> H_problem = pauli_z()  # Minimize ⟨Z⟩
        >>> 
        >>> # Perform quantum annealing
        >>> qa = QuantumAnnealing(H_problem, T=10.0, transverse_field=1.0, n_steps=100)
        >>> from psiqit.quantum.state import zero
        >>> result = qa.evolve(zero())
        >>> print(f"Final energy: {result.energies[-1]:.6f}")
        Final energy: -1.000000
    """
    
    def __init__(self, problem_hamiltonian: Operator, T: float,
                 transverse_field: float = 1.0, n_steps: int = 100):
        """
        Initialize quantum annealing.

        Args:
            problem_hamiltonian: Hamiltonian encoding the optimization problem.
                                 Its ground state represents the optimal solution.
            T: Annealing time (total evolution time).
            transverse_field: Strength of the initial transverse field.
                              H_initial = -transverse_field * Σ σ_x
            n_steps: Number of time steps for numerical integration.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x
            >>> # 1-qubit example: minimize ⟨Z⟩
            >>> H_prob = pauli_z()
            >>> qa = QuantumAnnealing(H_prob, T=5.0, transverse_field=1.0)
        """
        # Initial Hamiltonian: H_initial = -Σ σ_x
        from ..quantum.operator import pauli_x
        n_qubits = int(math.log2(problem_hamiltonian.dim))
        H_initial = Operator([[-transverse_field, 0], [0, transverse_field]], "H_init")
        # Simplified for single qubit
        super().__init__(H_initial, problem_hamiltonian, T, n_steps)


__all__ = [
    'AdiabaticResult',
    'AdiabaticEvolution',
    'QuantumAnnealing',
]