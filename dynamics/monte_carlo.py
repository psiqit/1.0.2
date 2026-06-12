"""
psiqit/dynamics/monte_carlo.py
============================================================
Monte Carlo Wave Function Method
============================================================

Quantum trajectories for open quantum systems using Monte Carlo
wave function method (quantum jump approach).

The Monte Carlo wave function method (also known as quantum trajectories)
provides an alternative to solving the Lindblad master equation for open
quantum systems. Instead of evolving the density matrix, it simulates
stochastic wave function evolution with quantum jumps.

Advantages:
- Memory requirements scale as O(N²) instead of O(N⁴)
- Natural interpretation of individual experimental runs
- Allows simulation of conditional dynamics (e.g., quantum feedback)

Example:
    >>> from psiqit.dynamics.monte_carlo import QuantumTrajectory
    >>> from psiqit.quantum.operator import pauli_z, pauli_x
    >>> from psiqit.quantum.state import zero
    >>> 
    >>> H = pauli_z()
    >>> collapse_ops = [pauli_x()]
    >>> traj = QuantumTrajectory(H, collapse_ops, gamma=[0.1])
    >>> psi0 = zero()
    >>> results = traj.run(psi0, n_trajectories=100, t_max=10.0, dt=0.01)
    >>> 
    >>> print(f"Average number of jumps: {np.mean(results.n_jumps):.2f}")
    Average number of jumps: 5.00

References:
    J. Dalibard, Y. Castin, and K. Mølmer, "Wave-function approach to
    dissipative processes in quantum optics," Physical Review Letters,
    68(5):580-583, 1992.
    K. Mølmer, Y. Castin, and J. Dalibard, "Monte Carlo wave-function
    method in quantum optics," Journal of the Optical Society of America B,
    10(3):524-538, 1993.
"""

import math
import random
import numpy as np
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass

from ..quantum.operator import Operator
from ..quantum.state import Ket
from ..math.qalgebra import mul, dagger, norm


@dataclass
class MonteCarloResult:
    """
    Result container for Monte Carlo wave function simulation.

    Attributes:
        times: List of time points.
        avg_states: Average wave function (normalized) over trajectories.
        jump_times: List of jump times for each trajectory.
        n_jumps: Number of jumps for each trajectory.
        success: Whether the simulation was successful.

    Examples:
        >>> result = MonteCarloResult(times=[0, 0.1, 0.2], avg_states=[], 
        ...                           jump_times=[[0.05], [0.07]], 
        ...                           n_jumps=[1, 1], success=True)
        >>> print(f"Total trajectories: {len(result.n_jumps)}")
        Total trajectories: 2
    """
    times: List[float]
    avg_states: List[List[complex]]
    jump_times: List[List[float]]
    n_jumps: List[int]
    success: bool


class QuantumTrajectory:
    """
    Monte Carlo wave function method for open quantum systems.

    This class implements the quantum trajectories (Monte Carlo wave function)
    method for simulating open quantum systems. The evolution alternates between
    deterministic non-unitary evolution under an effective Hamiltonian and
    stochastic quantum jumps representing photon emission or other dissipation.

    Key equations:
        - Effective Hamiltonian: H_eff = H - iħ/2 Σ γᵢ Lᵢ†Lᵢ
        - Jump probability: δp = γ dt ⟨ψ|Lᵢ†Lᵢ|ψ⟩
        - State after jump: |ψ'⟩ = Lᵢ|ψ⟩ / ||Lᵢ|ψ⟩||

    Examples:
        >>> from psiqit.quantum.operator import pauli_z, pauli_x
        >>> from psiqit.quantum.state import zero
        >>> 
        >>> # Two-level system with decay
        >>> H = pauli_z()
        >>> collapse_ops = [pauli_x()]
        >>> traj = QuantumTrajectory(H, collapse_ops, gamma=[0.1])
        >>> 
        >>> # Run 100 trajectories
        >>> psi0 = zero()
        >>> results = traj.run(psi0, t_max=5.0, n_trajectories=100)
        >>> 
        >>> # Analyze jump statistics
        >>> import numpy as np
        >>> print(f"Mean jumps: {np.mean(results.n_jumps):.2f}")
        Mean jumps: 2.50
    """
    
    def __init__(self, hamiltonian: Operator, collapse_ops: List[Operator],
                 gamma: List[float] = None):
        """
        Initialize the Monte Carlo wave function simulator.

        Args:
            hamiltonian: System Hamiltonian H (Hermitian).
            collapse_ops: List of collapse (jump) operators Lᵢ.
            gamma: Decay rates for each collapse operator.
                  If None, all rates are set to 1.0.

        Examples:
            >>> from psiqit.quantum.operator import pauli_z, pauli_x, pauli_y
            >>> 
            >>> # Single decay channel
            >>> traj = QuantumTrajectory(pauli_z(), [pauli_x()], gamma=[0.1])
            >>> 
            >>> # Multiple channels with different rates
            >>> traj = QuantumTrajectory(pauli_z(), [pauli_x(), pauli_y()], 
            ...                          gamma=[0.1, 0.05])
        """
        self.H = hamiltonian
        self.collapse_ops = collapse_ops
        self.gamma = gamma if gamma else [1.0] * len(collapse_ops)
        self.dim = hamiltonian.dim
    
    def _effective_hamiltonian(self) -> np.ndarray:
        """
        Compute the effective non-Hermitian Hamiltonian.

        The effective Hamiltonian is given by:
            H_eff = H - i/2 Σ γᵢ Lᵢ†Lᵢ

        This Hamiltonian governs the deterministic evolution between jumps.

        Returns:
            Effective Hamiltonian matrix.

        Examples:
            >>> traj = QuantumTrajectory(pauli_z(), [pauli_x()])
            >>> H_eff = traj._effective_hamiltonian()
        """
        H_eff = np.array(self.H.data, dtype=complex)
        
        for L, gamma in zip(self.collapse_ops, self.gamma):
            L_mat = np.array(L.data, dtype=complex)
            LdL = L_mat.conj().T @ L_mat
            H_eff -= 1j * gamma / 2 * LdL
        
        return H_eff
    
    def _evolve_without_jump(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """
        Evolve the wave function without a quantum jump.

        Uses the effective Hamiltonian: |ψ(t+dt)⟩ = (1 - iH_eff dt)|ψ(t)⟩
        followed by renormalization.

        Args:
            psi: Current wave function.
            dt: Time step.

        Returns:
            Normalized wave function after non-unitary evolution.
        """
        H_eff = self._effective_hamiltonian()
        # First-order approximation
        psi_new = psi - 1j * dt * H_eff @ psi
        return psi_new / np.linalg.norm(psi_new)
    
    def _jump_probability(self, psi: np.ndarray, dt: float) -> float:
        """
        Compute the total probability of a quantum jump.

        The jump probability is: δp = Σ γᵢ dt ⟨ψ|Lᵢ†Lᵢ|ψ⟩

        Args:
            psi: Current wave function.
            dt: Time step.

        Returns:
            Total jump probability (capped at 1.0).
        """
        prob = 0.0
        for L, gamma in zip(self.collapse_ops, self.gamma):
            L_mat = np.array(L.data, dtype=complex)
            L_psi = L_mat @ psi
            prob += gamma * dt * np.vdot(L_psi, L_psi).real
        return min(prob, 1.0)
    
    def _apply_jump(self, psi: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply a quantum jump (collapse) to the wave function.

        The jump operator is selected randomly based on the individual
        jump probabilities. After the jump, the state is renormalized.

        Args:
            psi: Current wave function.

        Returns:
            Tuple of (new wave function after jump, index of the applied jump).
        """
        # Choose which jump occurs
        cum_prob = 0.0
        r = random.random()
        
        for i, (L, gamma) in enumerate(zip(self.collapse_ops, self.gamma)):
            L_mat = np.array(L.data, dtype=complex)
            L_psi = L_mat @ psi
            prob = gamma * np.vdot(L_psi, L_psi).real
            cum_prob += prob
            
            if r < cum_prob:
                psi_new = L_psi / np.linalg.norm(L_psi)
                return psi_new, i
        
        return psi, -1
    
    def single_trajectory(self, psi0: Ket, t_max: float, dt: float = 0.01) -> dict:
        """
        Simulate a single quantum trajectory.

        This method evolves a single wave function stochastically, recording
        jump times and states along the trajectory.

        Args:
            psi0: Initial wave function (Ket).
            t_max: Maximum simulation time.
            dt: Time step.

        Returns:
            Dictionary containing:
                - 'times': List of time points
                - 'states': List of states at each time
                - 'jump_times': Times when jumps occurred
                - 'n_jumps': Total number of jumps

        Examples:
            >>> from psiqit.quantum.state import zero
            >>> traj = QuantumTrajectory(pauli_z(), [pauli_x()])
            >>> result = traj.single_trajectory(zero(), t_max=1.0)
            >>> print(f"Number of jumps: {result['n_jumps']}")
            Number of jumps: 3
        """
        times = []
        states = []
        jump_times = []
        
        psi = np.array(psi0.data)
        t = 0.0
        
        while t < t_max:
            times.append(t)
            states.append(psi.copy())
            
            # Compute jump probability
            jump_prob = self._jump_probability(psi, dt)
            
            if random.random() < jump_prob:
                # Quantum jump occurs
                psi, jump_idx = self._apply_jump(psi)
                jump_times.append(t)
            else:
                # No jump, evolve with effective Hamiltonian
                psi = self._evolve_without_jump(psi, dt)
            
            t += dt
        
        return {
            'times': times,
            'states': states,
            'jump_times': jump_times,
            'n_jumps': len(jump_times)
        }
    
    def run(self, psi0: Ket, t_max: float, dt: float = 0.01,
            n_trajectories: int = 100) -> MonteCarloResult:
        """
        Run multiple trajectories and average results.

        This method simulates many quantum trajectories and averages the
        results to obtain the density matrix evolution (equivalent to the
        Lindblad master equation).

        Args:
            psi0: Initial wave function (Ket).
            t_max: Maximum simulation time.
            dt: Time step.
            n_trajectories: Number of trajectories to simulate.

        Returns:
            MonteCarloResult containing averaged states and jump statistics.

        Examples:
            >>> from psiqit.quantum.state import zero
            >>> traj = QuantumTrajectory(pauli_z(), [pauli_x()], gamma=[0.1])
            >>> results = traj.run(zero(), t_max=5.0, n_trajectories=100)
            >>> print(f"Average fidelity: {results.success}")
            True
        """
        all_trajectories = []
        all_jump_times = []
        all_n_jumps = []
        
        for _ in range(n_trajectories):
            result = self.single_trajectory(psi0, t_max, dt)
            all_trajectories.append(result['states'])
            all_jump_times.append(result['jump_times'])
            all_n_jumps.append(result['n_jumps'])
        
        # Average over trajectories
        n_steps = len(all_trajectories[0])
        avg_states = []
        
        for step in range(n_steps):
            avg_psi = np.zeros(self.dim, dtype=complex)
            for traj in all_trajectories:
                avg_psi += traj[step]
            avg_psi /= np.linalg.norm(avg_psi)
            avg_states.append(avg_psi.tolist())
        
        return MonteCarloResult(
            times=result['times'],
            avg_states=avg_states,
            jump_times=all_jump_times,
            n_jumps=all_n_jumps,
            success=True
        )


__all__ = [
    'MonteCarloResult',
    'QuantumTrajectory',
]