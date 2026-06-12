"""
psiqit/variational/vqe.py
============================================================
Variational Quantum Eigensolver (VQE)
============================================================

Variational method for finding ground state energy of Hamiltonians.

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical
algorithm for finding the ground state energy of a Hamiltonian. It uses
a parameterized quantum circuit (ansatz) to prepare trial states and a
classical optimizer to minimize the energy expectation value.

The algorithm:
    1. Prepare trial state |ψ(θ)⟩ using parameterized circuit
    2. Measure expectation value ⟨ψ(θ)|H|ψ(θ)⟩
    3. Update parameters θ to minimize the energy
    4. Repeat until convergence

VQE is particularly important for quantum chemistry and materials science
on near-term quantum computers.

Example:
    >>> from psiqit.variational.vqe import VQE
    >>> 
    >>> # Hamiltonian for H2 molecule (simplified)
    >>> hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}
    >>> 
    >>> vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)
    >>> result = vqe.run(n_iterations=100, learning_rate=0.1)
    >>> print(f"Ground state energy: {result.optimal_energy:.6f}")

References:
    A. Peruzzo et al., "A variational eigenvalue solver on a photonic
    quantum processor," Nature Communications, 5:4213, 2014.
    J. R. McClean et al., "The theory of variational quantum algorithms,"
    arXiv:2108.08428, 2021.
"""

import math
import random
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import rx, ry, rz, hadamard, cnot
from ..quantum.measurement import expectation


@dataclass
class VQEResult:
    """
    Result container for VQE optimization.

    Attributes:
        optimal_params: Optimal parameters found.
        optimal_energy: Minimum energy (ground state energy) achieved.
        n_iterations: Number of optimization iterations.
        history: List of energy values at each iteration.
        success: Whether optimization completed successfully.

    Examples:
        >>> result = VQEResult(optimal_params=[0.1, 0.2], optimal_energy=-1.0,
        ...                    n_iterations=100, history=[-0.5, -0.8, -1.0],
        ...                    success=True)
        >>> print(result)
        VQE(energy=-1.000000, iterations=100)
    """
    optimal_params: List[float]
    optimal_energy: float
    n_iterations: int
    history: List[float]
    success: bool
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing optimal energy and number of iterations.
        """
        return f"VQE(energy={self.optimal_energy:.6f}, iterations={self.n_iterations})"


class VQE:
    """
    Variational Quantum Eigensolver (VQE).

    VQE finds the ground state energy of a Hamiltonian by minimizing the
    expectation value ⟨ψ(θ)|H|ψ(θ)⟩ over parameters θ of a quantum circuit.

    The ansatz circuit consists of alternating layers of:
        - RY rotations on each qubit (parameterized)
        - CNOT entangling gates in a linear chain

    The Hamiltonian is specified as a dictionary mapping Pauli strings to
    coefficients. For example:
        {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5} represents H = Z₀ + Z₁ + 0.5·Z₀Z₁

    Examples:
        >>> # Simple 1-qubit Hamiltonian: H = X + Z
        >>> hamiltonian = {'X': 1.0, 'Z': 1.0}
        >>> vqe = VQE(n_qubits=1, hamiltonian=hamiltonian, n_layers=2)
        >>> result = vqe.run(n_iterations=100)
        >>> print(f"Ground state energy: {result.optimal_energy:.6f}")
        >>> 
        >>> # 2-qubit Heisenberg model
        >>> hamiltonian = {'X0X1': 1.0, 'Y0Y1': 1.0, 'Z0Z1': 1.0}
        >>> vqe = VQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=3)
        >>> result = vqe.run(n_iterations=200, learning_rate=0.05)
    """
    
    def __init__(self, n_qubits: int, hamiltonian: Dict[str, float], 
                 n_layers: int = 2, optimizer: str = 'gradient'):
        """
        Initialize the VQE algorithm.

        Args:
            n_qubits: Number of qubits.
            hamiltonian: Hamiltonian as Pauli string dictionary.
            n_layers: Number of variational layers in the ansatz circuit.
            optimizer: Optimization method (currently only 'gradient').

        Examples:
            >>> vqe = VQE(n_qubits=2, hamiltonian=H, n_layers=3)
        """
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_layers = n_layers
        self.optimizer = optimizer
        
        self.n_params = n_qubits * n_layers
        self.params = None
        self._init_random()
    
    def _init_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly.

        Args:
            seed: Random seed for reproducibility.

        Examples:
            >>> vqe._init_random(seed=42)
        """
        if seed is not None:
            random.seed(seed)
        self.params = [random.uniform(-math.pi, math.pi) for _ in range(self.n_params)]
    
    def build_circuit(self, params: Optional[List[float]] = None) -> QuantumCircuit:
        """
        Build the variational quantum circuit.

        The circuit consists of:
            - n_layers of RY rotations on all qubits
            - CNOT gates connecting adjacent qubits

        Args:
            params: Parameters for the circuit. If None, uses current parameters.

        Returns:
            QuantumCircuit: The variational circuit.

        Examples:
            >>> circuit = vqe.build_circuit()
            >>> print(circuit.depth)
        """
        if params is None:
            params = self.params
        
        circuit = QuantumCircuit(self.n_qubits)
        p_idx = 0
        
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                circuit.ry(qubit, params[p_idx])
                p_idx += 1
            
            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        
        return circuit
    
    def evaluate_energy(self, params: Optional[List[float]] = None) -> float:
        """
        Evaluate the energy expectation value for given parameters.

        Args:
            params: Parameters to evaluate. If None, uses current parameters.

        Returns:
            Energy expectation value.

        Examples:
            >>> energy = vqe.evaluate_energy()
            >>> print(f"Current energy: {energy:.6f}")
        """
        circuit = self.build_circuit(params)
        state = circuit.run()
        
        total_energy = 0.0
        
        for pauli_str, coeff in self.hamiltonian.items():
            exp_val = self._expectation_pauli(state, pauli_str)
            total_energy += coeff * exp_val
        
        return total_energy
    
    def _expectation_pauli(self, state: Ket, pauli_str: str) -> float:
        """
        Compute the expectation value of a Pauli string.

        Args:
            state: Quantum state.
            pauli_str: Pauli string (e.g., 'Z', 'Z0', 'X0Y1', 'Z0Z1').

        Returns:
            Expectation value (real number between -1 and 1).

        Examples:
            >>> from psiqit.quantum.state import zero
            >>> exp = vqe._expectation_pauli(zero(), 'Z')
            >>> print(exp)  # 1.0
        """
        n = self.n_qubits
        dim = state.dim
        
        # Single qubit Pauli
        if len(pauli_str) == 2 and pauli_str[0] in ['Z', 'X', 'Y']:
            qubit = int(pauli_str[1])
            
            if pauli_str[0] == 'Z':
                exp_val = 0.0
                for i in range(dim):
                    bit = (i >> (n - 1 - qubit)) & 1
                    exp_val += (1 if bit == 0 else -1) * abs(state.data[i])**2
                return exp_val
            
            elif pauli_str[0] == 'X':
                exp_val = 0.0
                for i in range(dim):
                    flipped = i ^ (1 << (n - 1 - qubit))
                    exp_val += (state.data[i].conjugate() * state.data[flipped]).real
                return exp_val
            
            else:  # 'Y'
                exp_val = 0.0
                for i in range(dim):
                    bit = (i >> (n - 1 - qubit)) & 1
                    flipped = i ^ (1 << (n - 1 - qubit))
                    factor = -1j if bit == 0 else 1j
                    exp_val += (factor * state.data[i].conjugate() * state.data[flipped]).real
                return exp_val
        
        # Multi-qubit Pauli string (product of single-qubit Paulis)
        else:
            result = 1.0
            for idx in range(0, len(pauli_str), 2):
                gate = pauli_str[idx]
                qubit = int(pauli_str[idx+1])
                result *= self._expectation_pauli(state, gate + str(qubit))
            return result
    
    def _gradient(self, params: List[float], eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient using finite differences.

        Args:
            params: Parameters to differentiate.
            eps: Step size for finite differences.

        Returns:
            List of gradient values.
        """
        grads = []
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            
            e_plus = self.evaluate_energy(p_plus)
            e_minus = self.evaluate_energy(p_minus)
            grads.append((e_plus - e_minus) / (2 * eps))
        
        return grads
    
    def run(self, n_iterations: int = 100, learning_rate: float = 0.1,
            verbose: bool = True) -> VQEResult:
        """
        Run the VQE optimization.

        Args:
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            VQEResult with optimal parameters and energy.

        Examples:
            >>> vqe = VQE(n_qubits=2, hamiltonian=H)
            >>> result = vqe.run(n_iterations=200, learning_rate=0.05)
            >>> print(f"Ground state energy: {result.optimal_energy:.6f}")
        """
        history = []
        params = self.params.copy()
        
        for iteration in range(n_iterations):
            grads = self._gradient(params)
            
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]
            
            energy = self.evaluate_energy(params)
            history.append(energy)
            
            if verbose and (iteration % 20 == 0 or iteration == n_iterations - 1):
                print(f"  Iter {iteration}: energy = {energy:.6f}")
        
        self.params = params
        
        return VQEResult(
            optimal_params=params,
            optimal_energy=history[-1] if history else 0.0,
            n_iterations=n_iterations,
            history=history,
            success=True
        )


__all__ = [
    'VQEResult',
    'VQE',
]