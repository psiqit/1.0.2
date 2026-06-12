"""
psiqit/variational/qaoa.py
============================================================
Quantum Approximate Optimization Algorithm (QAOA)
============================================================

QAOA for solving combinatorial optimization problems:
    • Max-Cut problem
    • Traveling Salesman (simplified)
    • Quadratic unconstrained binary optimization (QUBO)

The Quantum Approximate Optimization Algorithm (QAOA) is a hybrid
quantum-classical algorithm for solving combinatorial optimization
problems. It uses a parameterized quantum circuit that alternates
between applying the cost Hamiltonian (encoding the problem) and a
mixer Hamiltonian (exploring the solution space).

The QAOA circuit consists of p layers, each with two parameters:
    - γ (gamma): angle for the cost Hamiltonian evolution
    - β (beta): angle for the mixer Hamiltonian evolution

As p increases, the approximation quality improves, approaching the
optimal solution in the limit p → ∞.

Example:
    >>> from psiqit.variational.qaoa import QAOA, maxcut_hamiltonian
    >>> 
    >>> # Max-Cut on a triangle graph
    >>> edges = [(0,1), (1,2), (2,0)]
    >>> H = maxcut_hamiltonian(edges)
    >>> 
    >>> qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
    >>> result = qaoa.run(n_iterations=100)
    >>> print(f"Max-Cut energy: {result.optimal_energy:.4f}")

References:
    E. Farhi, J. Goldstone, and S. Gutmann, "A Quantum Approximate
    Optimization Algorithm," arXiv:1411.4028, 2014.
    E. Farhi and A. W. Harrow, "Quantum Supremacy through the Quantum
    Approximate Optimization Algorithm," arXiv:1602.07674, 2016.
"""

import math
import random
from typing import List, Dict, Optional, Callable, Tuple, Set
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import rx, ry, rz, hadamard, cnot, pauli_z
from ..quantum.measurement import expectation


@dataclass
class QAOResult:
    """
    Result container for QAOA optimization.

    Attributes:
        optimal_params: List of optimal parameters [γ₁, β₁, γ₂, β₂, ...].
        optimal_energy: Minimum energy (cost) achieved.
        n_iterations: Number of optimization iterations.
        history: List of energy values at each iteration.
        success: Whether optimization completed successfully.

    Examples:
        >>> result = QAOResult(optimal_params=[0.5, 0.3, 0.7, 0.2],
        ...                    optimal_energy=-2.0, n_iterations=100,
        ...                    history=[-1.5, -1.8, -2.0], success=True)
        >>> print(result)
        QAOA(energy=-2.000000, iterations=100)
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
        return f"QAOA(energy={self.optimal_energy:.6f}, iterations={self.n_iterations})"


class QAOA:
    """
    Quantum Approximate Optimization Algorithm (QAOA).

    QAOA is a variational algorithm for solving combinatorial optimization
    problems. It alternates between applying the cost Hamiltonian (which
    encodes the problem) and a mixer Hamiltonian (which explores the
    solution space).

    The circuit for p layers is:
        |ψ(γ,β)⟩ = e^{-iβ_p H_M} e^{-iγ_p H_C} ... e^{-iβ_1 H_M} e^{-iγ_1 H_C} |+⟩^{⊗n}

    The cost Hamiltonian H_C encodes the objective function, and the mixer
    Hamiltonian H_M = Σ X_i (usually) ensures exploration of the space.

    Examples:
        >>> # Max-Cut on a 3-node graph
        >>> edges = [(0,1), (1,2), (2,0)]
        >>> H = maxcut_hamiltonian(edges)
        >>> qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
        >>> result = qaoa.run(n_iterations=100)
        >>> 
        >>> # QUBO problem
        >>> qubo = {(0,0): -1, (1,1): -1, (0,1): 2}
        >>> H = qubo_to_hamiltonian(qubo, n_qubits=2)
        >>> qaoa = QAOA(n_qubits=2, hamiltonian=H, p=1)
    """
    
    def __init__(self, n_qubits: int, hamiltonian: Dict[str, float], 
                 p: int = 2, optimizer: str = 'gradient'):
        """
        Initialize QAOA.

        Args:
            n_qubits: Number of qubits.
            hamiltonian: Problem Hamiltonian as Pauli string dictionary.
            p: Number of QAOA layers (higher p gives better approximation).
            optimizer: Optimization method (currently only 'gradient').

        Examples:
            >>> # 2-layer QAOA for 3-qubit Max-Cut
            >>> qaoa = QAOA(n_qubits=3, hamiltonian=H, p=2)
        """
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.p = p
        self.optimizer = optimizer
        
        # 2 parameters per layer (gamma and beta)
        self.n_params = 2 * p
        self.params = None
        self._init_random()
    
    def _init_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly.

        Args:
            seed: Random seed for reproducibility.

        Notes:
            gamma parameters are in [0, 2π), beta parameters in [0, π).
        """
        if seed is not None:
            random.seed(seed)
        # gamma in [0, 2π), beta in [0, π)
        self.params = [random.uniform(0, 2*math.pi) for _ in range(self.n_params)]
    
    def _apply_cost_layer(self, circuit: QuantumCircuit, gamma: float):
        """
        Apply the cost Hamiltonian layer: e^{-iγ H_C}.

        Args:
            circuit: Quantum circuit to add gates to.
            gamma: Parameter for the cost layer.
        """
        for term, coeff in self.hamiltonian.items():
            if term == 'Z0' or term == 'Z':
                circuit.rz(0, 2 * coeff * gamma)
            elif term == 'Z1':
                circuit.rz(1, 2 * coeff * gamma)
            elif term == 'Z0Z1':
                # ZZ interaction
                circuit.cx(0, 1)
                circuit.rz(1, 2 * coeff * gamma)
                circuit.cx(0, 1)
            else:
                # General Pauli string
                self._apply_pauli_string(circuit, term, gamma)
    
    def _apply_mixer_layer(self, circuit: QuantumCircuit, beta: float):
        """
        Apply the mixer Hamiltonian layer: e^{-iβ H_M} with H_M = Σ X_i.

        Args:
            circuit: Quantum circuit to add gates to.
            beta: Parameter for the mixer layer.
        """
        for i in range(self.n_qubits):
            circuit.rx(i, 2 * beta)
    
    def _apply_pauli_string(self, circuit: QuantumCircuit, pauli_str: str, angle: float):
        """
        Apply a rotation generated by a Pauli string.

        Args:
            circuit: Quantum circuit to add gates to.
            pauli_str: Pauli string (e.g., 'X0', 'Z0Z1', 'Y0X1').
            angle: Rotation angle.
        """
        # Find qubits with X or Y
        for i, p in enumerate(pauli_str):
            if p == 'X':
                circuit.h(i)
            elif p == 'Y':
                circuit.rx(i, math.pi/2)
        
        # Apply multi-controlled Z on qubits with Z
        target = None
        for i, p in enumerate(pauli_str):
            if p == 'Z':
                if target is None:
                    target = i
                else:
                    circuit.cx(target, i)
        
        if target is not None:
            circuit.rz(target, 2 * angle)
        
        # Uncompute
        for i, p in enumerate(pauli_str):
            if p == 'X':
                circuit.h(i)
            elif p == 'Y':
                circuit.rx(i, -math.pi/2)
    
    def build_circuit(self, params: Optional[List[float]] = None) -> QuantumCircuit:
        """
        Build the QAOA quantum circuit.

        Args:
            params: List of parameters [γ₁, β₁, γ₂, β₂, ...].
                    If None, uses current parameters.

        Returns:
            QuantumCircuit: The QAOA circuit ready for execution.

        Examples:
            >>> qaoa = QAOA(n_qubits=2, hamiltonian={'Z0Z1': 1.0}, p=1)
            >>> circuit = qaoa.build_circuit([0.5, 0.3])
            >>> print(circuit.depth)
        """
        if params is None:
            params = self.params
        
        circuit = QuantumCircuit(self.n_qubits)
        
        # Initialize in superposition |+...+⟩
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Apply alternating layers
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            
            self._apply_cost_layer(circuit, gamma)
            self._apply_mixer_layer(circuit, beta)
        
        return circuit
    
    def evaluate_energy(self, params: Optional[List[float]] = None) -> float:
        """
        Evaluate the cost function (energy) for given parameters.

        Args:
            params: List of parameters. If None, uses current parameters.

        Returns:
            Energy expectation value.

        Examples:
            >>> energy = qaoa.evaluate_energy([0.5, 0.3])
            >>> print(f"Energy: {energy:.6f}")
        """
        circuit = self.build_circuit(params)
        state = circuit.run()
        
        total_energy = 0.0
        for term, coeff in self.hamiltonian.items():
            exp_val = self._expectation_pauli(state, term)
            total_energy += coeff * exp_val
        
        return total_energy
    
    def _expectation_pauli(self, state: Ket, pauli_str: str) -> float:
        """
        Compute the expectation value of a Pauli string.

        Args:
            state: Quantum state.
            pauli_str: Pauli string (e.g., 'Z', 'Z0', 'Z0Z1').

        Returns:
            Expectation value (real number between -1 and 1).
        """
        n = self.n_qubits
        dim = state.dim
        
        if len(pauli_str) == 1 and pauli_str == 'Z':
            # Single qubit Z on first qubit
            exp_val = 0.0
            for i in range(dim):
                bit = (i >> (n - 1)) & 1
                exp_val += (1 if bit == 0 else -1) * abs(state.data[i])**2
            return exp_val
        
        elif len(pauli_str) == 2:
            # Single qubit Z on specific qubit
            qubit = int(pauli_str[1])
            exp_val = 0.0
            for i in range(dim):
                bit = (i >> (n - 1 - qubit)) & 1
                exp_val += (1 if bit == 0 else -1) * abs(state.data[i])**2
            return exp_val
        
        elif len(pauli_str) == 4 and pauli_str[0] == 'Z' and pauli_str[2] == 'Z':
            # ZZ interaction
            q1 = int(pauli_str[1])
            q2 = int(pauli_str[3])
            exp_val = 0.0
            for i in range(dim):
                b1 = (i >> (n - 1 - q1)) & 1
                b2 = (i >> (n - 1 - q2)) & 1
                sign = 1 if b1 == b2 else -1
                exp_val += sign * abs(state.data[i])**2
            return exp_val
        
        return 0.0
    
    def _gradient(self, params: List[float], eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient using finite differences.

        Args:
            params: List of parameters.
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
            verbose: bool = True) -> QAOResult:
        """
        Run QAOA optimization.

        Args:
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            QAOResult containing optimal parameters and energy.

        Examples:
            >>> qaoa = QAOA(n_qubits=2, hamiltonian={'Z0Z1': 1.0}, p=1)
            >>> result = qaoa.run(n_iterations=100, learning_rate=0.1)
            >>> print(f"Optimal energy: {result.optimal_energy:.6f}")
        """
        history = []
        params = self.params.copy()
        
        for iteration in range(n_iterations):
            grads = self._gradient(params)
            
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]
                # Keep parameters in reasonable range
                if i % 2 == 0:  # gamma
                    params[i] = params[i] % (2 * math.pi)
                else:  # beta
                    params[i] = max(0, min(math.pi, params[i]))
            
            energy = self.evaluate_energy(params)
            history.append(energy)
            
            if verbose and (iteration % 20 == 0 or iteration == n_iterations - 1):
                print(f"  Iter {iteration}: energy = {energy:.6f}")
        
        self.params = params
        
        return QAOResult(
            optimal_params=params,
            optimal_energy=history[-1] if history else 0.0,
            n_iterations=n_iterations,
            history=history,
            success=True
        )


def maxcut_hamiltonian(edges: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Create a Max-Cut Hamiltonian from graph edges.

    For Max-Cut, the Hamiltonian is:
        H = Σ_{(i,j)} (I - Z_i Z_j)/2

    Maximizing the cut is equivalent to minimizing the energy of this
    Hamiltonian.

    Args:
        edges: List of edges as (i, j) tuples.

    Returns:
        Hamiltonian as a Pauli string dictionary.

    Examples:
        >>> edges = [(0,1), (1,2), (2,0)]  # Triangle graph
        >>> H = maxcut_hamiltonian(edges)
        >>> print(H)
        {'Z0Z1': 0.5, 'I': 1.5, 'Z1Z2': 0.5, 'Z2Z0': 0.5}
    """
    hamiltonian = {}
    
    for i, j in edges:
        term = f"Z{i}Z{j}"
        hamiltonian[term] = 0.5
        hamiltonian['I'] = hamiltonian.get('I', 0) + 0.5
    
    return hamiltonian


def qubo_to_hamiltonian(qubo: Dict[Tuple[int, int], float], n_qubits: int) -> Dict[str, float]:
    """
    Convert a QUBO (Quadratic Unconstrained Binary Optimization) problem
    to a Hamiltonian.

    The transformation uses the mapping x_i → (I - Z_i)/2.

    Args:
        qubo: QUBO coefficients where (i,i) are linear terms and (i,j) are
              quadratic terms (i ≠ j).
        n_qubits: Number of qubits.

    Returns:
        Hamiltonian as a Pauli string dictionary.

    Examples:
        >>> # Minimize x₀ + x₁ - 2x₀x₁
        >>> qubo = {(0,0): 1, (1,1): 1, (0,1): -2}
        >>> H = qubo_to_hamiltonian(qubo, n_qubits=2)
        >>> print(H)
        {'Z0': -0.5, 'I': 1.0, 'Z1': -0.5, 'Z0Z1': 0.5}
    """
    hamiltonian = {}
    
    for (i, j), coeff in qubo.items():
        if i == j:
            # Linear term: -coeff * Z_i/2 + coeff/2 * I
            hamiltonian[f"Z{i}"] = hamiltonian.get(f"Z{i}", 0) - coeff / 2
            hamiltonian['I'] = hamiltonian.get('I', 0) + coeff / 2
        else:
            # Quadratic term: coeff * (I - Z_i Z_j)/4
            hamiltonian[f"Z{i}Z{j}"] = hamiltonian.get(f"Z{i}Z{j}", 0) - coeff / 4
            hamiltonian['I'] = hamiltonian.get('I', 0) + coeff / 4
    
    return hamiltonian


__all__ = [
    'QAOResult',
    'QAOA',
    'maxcut_hamiltonian',
    'qubo_to_hamiltonian',
]