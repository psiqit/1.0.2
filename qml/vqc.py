"""
psiqit/qml/vqc.py
============================================================
Variational Quantum Circuit (VQC)
============================================================

Variational quantum algorithms for optimization and machine learning:
    • Parameterized quantum circuits
    • Cost function evaluation
    • Gradient computation (parameter shift rule)
    • Classical optimizer integration

Variational Quantum Circuits (VQCs) are parameterized quantum circuits that
can be optimized to minimize a cost function. They are the building blocks
of many variational quantum algorithms, including:

    - Variational Quantum Eigensolver (VQE)
    - Quantum Approximate Optimization Algorithm (QAOA)
    - Quantum Neural Networks (QNNs)

Example:
    >>> from psiqit.qml.vqc import VQC, HamiltonianVQE
    >>> 
    >>> # Create variational circuit
    >>> vqc = VQC(n_qubits=2, n_layers=3)
    >>> 
    >>> # Define cost function
    >>> def cost(params):
    ...     return vqc.evaluate_cost(params)
    >>> 
    >>> # Optimize
    >>> result = vqc.optimize(cost, n_iterations=100)
    >>> print(f"Optimal cost: {result.optimal_cost:.6f}")

References:
    M. Benedetti et al., "Parameterized quantum circuits as machine learning models,"
    Quantum Science and Technology, 4(4):043001, 2019.
    J. R. McClean et al., "The theory of variational quantum algorithms,"
    arXiv:2108.08428, 2021.
"""

import math
import random
from typing import List, Union, Optional, Callable, Tuple, Dict
from dataclasses import dataclass, field

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, basis
from ..quantum.operator import hadamard, pauli_x, pauli_y, pauli_z, rx, ry, rz, cnot, cz
from ..quantum.measurement import expectation, measure
from ..utils.random import random_state


@dataclass
class VQCResult:
    """
    Result container for variational optimization.

    Attributes:
        optimal_params: List of optimal parameters found.
        optimal_cost: Minimum cost value achieved.
        n_iterations: Number of iterations performed.
        history: List of cost values at each iteration.
        success: Whether optimization completed successfully.

    Examples:
        >>> result = VQCResult(optimal_params=[0.1, 0.2], optimal_cost=-1.0,
        ...                    n_iterations=100, history=[-0.5, -0.8, -1.0],
        ...                    success=True)
        >>> print(result)
        VQC(optimal_cost=-1.000000, n_iterations=100, success=True)
    """
    optimal_params: List[float]
    optimal_cost: float
    n_iterations: int
    history: List[float]
    success: bool
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing optimal cost, iterations, and success status.
        """
        return f"VQC(optimal_cost={self.optimal_cost:.6f}, n_iterations={self.n_iterations}, success={self.success})"


class VariationalLayer:
    """
    A single variational layer for VQC.

    Each layer consists of:
        - RY and RZ rotations on each qubit (2n parameters)
        - Entangling gates (CNOT or CZ) in a cyclic topology

    Attributes:
        n_qubits: Number of qubits.
        entangler: Type of entangling gate ('cnot' or 'cz').
        n_params: Number of parameters per layer (2 * n_qubits).

    Examples:
        >>> layer = VariationalLayer(n_qubits=4, entangler='cnot')
        >>> circuit = QuantumCircuit(4)
        >>> params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        >>> layer.apply(circuit, params)
    """
    
    def __init__(self, n_qubits: int, entangler: str = 'cnot'):
        """
        Initialize a variational layer.

        Args:
            n_qubits: Number of qubits.
            entangler: Type of entangling gate ('cnot' or 'cz').

        Examples:
            >>> layer = VariationalLayer(4, entangler='cnot')
        """
        self.n_qubits = n_qubits
        self.entangler = entangler
        self.n_params = 2 * n_qubits  # Ry and Rz per qubit
    
    def apply(self, circuit: QuantumCircuit, params: List[float], start_idx: int = 0):
        """
        Apply the variational layer to a circuit.

        Args:
            circuit: Quantum circuit to add gates to.
            params: Parameter list for this layer.
            start_idx: Starting index in the parameter list.
        """
        # Rotation gates
        for i in range(self.n_qubits):
            circuit.ry(i, params[start_idx + i])
            circuit.rz(i, params[start_idx + self.n_qubits + i])
        
        # Entangling gates
        if self.entangler == 'cnot':
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.cx(self.n_qubits - 1, 0)
        elif self.entangler == 'cz':
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)
            circuit.cz(self.n_qubits - 1, 0)
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"VariationalLayer(n_qubits={self.n_qubits}, n_params={self.n_params})"


class VQC:
    """
    Variational Quantum Circuit (VQC).

    A VQC is a parameterized quantum circuit that can be optimized to
    minimize a cost function. It consists of multiple variational layers
    with RY/RZ rotations and entangling gates.

    The circuit structure:
        - Input: Initial state (default |0...0⟩)
        - Variational layers: Alternating rotations and entanglement
        - Measurement: Expectation value of Pauli-Z on the first qubit

    Examples:
        >>> # Create a VQC for 2 qubits with 3 layers
        >>> vqc = VQC(n_qubits=2, n_layers=3, entangler='cnot')
        >>> 
        >>> # Evaluate cost for current parameters
        >>> cost = vqc.evaluate_cost()
        >>> 
        >>> # Optimize to minimize a custom cost function
        >>> def my_cost(params):
        ...     return vqc.evaluate_cost(params)
        >>> result = vqc.optimize(my_cost, n_iterations=100)
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 3, 
                 entangler: str = 'cnot', measurement: str = 'z'):
        """
        Initialize a Variational Quantum Circuit.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of variational layers (default: 3).
            entangler: Type of entangling gate ('cnot' or 'cz').
            measurement: Measurement basis ('z', 'x', or 'y').

        Examples:
            >>> vqc = VQC(n_qubits=4, n_layers=3, measurement='z')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.measurement = measurement
        
        # Create layers
        self.layers = [VariationalLayer(n_qubits, entangler) for _ in range(n_layers)]
        
        # Total parameters
        self.n_params = n_layers * (2 * n_qubits)
        
        # Current parameters
        self.params = None
        self._init_random()
    
    def _init_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly in [-π, π].

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.params = [random.uniform(-math.pi, math.pi) for _ in range(self.n_params)]
    
    def set_params(self, params: List[float]):
        """
        Set the circuit parameters.

        Args:
            params: New parameter list.

        Raises:
            ValueError: If the number of parameters doesn't match.
        """
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {len(params)}")
        self.params = params
    
    def build_circuit(self, input_state: Optional[Ket] = None) -> QuantumCircuit:
        """
        Build the variational quantum circuit.

        Args:
            input_state: Initial state (default: |0...0⟩).

        Returns:
            Quantum circuit.

        Examples:
            >>> vqc = VQC(2, 2)
            >>> circuit = vqc.build_circuit()
            >>> print(circuit.depth)
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Apply variational layers
        param_idx = 0
        for layer in self.layers:
            layer.apply(circuit, self.params, param_idx)
            param_idx += 2 * self.n_qubits
        
        return circuit
    
    def get_state(self, input_state: Optional[Ket] = None) -> Ket:
        """
        Get the output state of the circuit.

        Args:
            input_state: Input state (default: |0...0⟩).

        Returns:
            Output quantum state.
        """
        circuit = self.build_circuit(input_state)
        return circuit.run()
    
    def measure(self, input_state: Optional[Ket] = None) -> float:
        """
        Measure the expectation value.

        Args:
            input_state: Input state (default: |0...0⟩).

        Returns:
            Expectation value in [-1, 1].

        Examples:
            >>> vqc = VQC(2, 2)
            >>> value = vqc.measure()
            >>> print(f"Expectation: {value:.4f}")
        """
        output_state = self.get_state(input_state)
        
        if not output_state.is_normalized:
            output_state = output_state.normalize()
        
        # Measure first qubit (Pauli-Z by default)
        n = self.n_qubits
        dim = output_state.dim
        exp_val = 0.0
        
        for i in range(dim):
            first_bit = (i >> (n - 1)) & 1
            sign = 1 if first_bit == 0 else -1
            exp_val += sign * abs(output_state.data[i]) ** 2
        
        return exp_val
    
    def evaluate_cost(self, params: Optional[List[float]] = None,
                      input_state: Optional[Ket] = None) -> float:
        """
        Evaluate the cost function for given parameters.

        Args:
            params: Parameters to evaluate (uses current if None).
            input_state: Input state (default: |0...0⟩).

        Returns:
            Cost value (expectation value).
        """
        if params is not None:
            self.set_params(params)
        
        return self.measure(input_state)
    
    def parameter_shift_gradient(self, param_idx: int, 
                                  input_state: Optional[Ket] = None,
                                  epsilon: float = 1e-5) -> float:
        """
        Compute gradient using the parameter shift rule (finite difference).

        Args:
            param_idx: Index of the parameter to differentiate.
            input_state: Input state.
            epsilon: Small step for finite difference.

        Returns:
            Gradient value.
        """
        original_params = self.params.copy()
        
        # Forward pass with +ε
        self.params = original_params.copy()
        self.params[param_idx] += epsilon
        cost_plus = self.evaluate_cost(input_state=input_state)
        
        # Forward pass with -ε
        self.params = original_params.copy()
        self.params[param_idx] -= epsilon
        cost_minus = self.evaluate_cost(input_state=input_state)
        
        # Restore parameters
        self.params = original_params
        
        return (cost_plus - cost_minus) / (2 * epsilon)
    
    def gradient(self, input_state: Optional[Ket] = None) -> List[float]:
        """
        Compute the full gradient vector.

        Args:
            input_state: Input state.

        Returns:
            List of gradient values for all parameters.
        """
        grads = []
        for i in range(self.n_params):
            grad = self.parameter_shift_gradient(i, input_state)
            grads.append(grad)
        return grads
    
    def optimize(self, cost_function: Optional[Callable] = None,
                 n_iterations: int = 100, learning_rate: float = 0.1,
                 input_state: Optional[Ket] = None,
                 verbose: bool = True) -> VQCResult:
        """
        Optimize the variational circuit.

        Args:
            cost_function: Custom cost function (uses evaluate_cost if None).
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            input_state: Input state.
            verbose: Whether to print progress.

        Returns:
            VQCResult with optimization results.

        Examples:
            >>> vqc = VQC(2, 2)
            >>> 
            >>> # Optimize using built-in cost
            >>> result = vqc.optimize(n_iterations=100, learning_rate=0.1)
            >>> 
            >>> # Optimize with custom cost function
            >>> def custom_cost(params):
            ...     return -vqc.evaluate_cost(params)
            >>> result = vqc.optimize(custom_cost, n_iterations=50)
        """
        if cost_function is None:
            cost_function = lambda params: self.evaluate_cost(params, input_state)
        
        history = []
        params = self.params.copy()
        
        for iteration in range(n_iterations):
            # Compute gradient
            grads = []
            for i in range(self.n_params):
                # Finite difference gradient
                params_plus = params.copy()
                params_minus = params.copy()
                eps = 1e-5
                
                params_plus[i] += eps
                params_minus[i] -= eps
                
                cost_plus = cost_function(params_plus)
                cost_minus = cost_function(params_minus)
                
                grad = (cost_plus - cost_minus) / (2 * eps)
                grads.append(grad)
            
            # Update parameters
            for i in range(self.n_params):
                params[i] -= learning_rate * grads[i]
            
            # Evaluate cost
            current_cost = cost_function(params)
            history.append(current_cost)
            
            if verbose and (iteration % 20 == 0 or iteration == n_iterations - 1):
                print(f"  Iteration {iteration}: cost = {current_cost:.6f}")
        
        self.params = params
        
        return VQCResult(
            optimal_params=params,
            optimal_cost=history[-1] if history else 0.0,
            n_iterations=n_iterations,
            history=history,
            success=True
        )
    
    def __repr__(self) -> str:
        """Return a string representation of the VQC."""
        return f"VQC(n_qubits={self.n_qubits}, n_layers={self.n_layers}, n_params={self.n_params})"


class HamiltonianVQE(VQC):
    """
    Variational Quantum Eigensolver (VQE) for finding ground state energy.

    VQE is a hybrid quantum-classical algorithm that finds the ground state
    energy of a given Hamiltonian. It uses a variational quantum circuit
    to prepare trial states and minimizes the energy expectation value.

    The Hamiltonian is specified as a dictionary mapping Pauli strings to
    coefficients. For example:
        {'X': 1.0, 'Z': 1.0} represents H = X + Z
        {'Z0Z1': 0.5, 'X0': 1.0} represents H = 0.5·Z₀Z₁ + X₀

    Example:
        >>> from psiqit.qml.vqc import HamiltonianVQE
        >>> 
        >>> # Define Hamiltonian for a single qubit (H = X + Z)
        >>> hamiltonian = {'X': 1.0, 'Z': 1.0}
        >>> 
        >>> # Create and run VQE
        >>> vqe = HamiltonianVQE(n_qubits=1, hamiltonian=hamiltonian, n_layers=2)
        >>> result = vqe.run(n_iterations=100)
        >>> print(f"Ground state energy: {result.optimal_cost:.6f}")

    References:
        A. Peruzzo et al., "A variational eigenvalue solver on a photonic
        quantum processor," Nature Communications, 5(1):4213, 2014.
        J. R. McClean et al., "The theory of variational quantum algorithms,"
        arXiv:2108.08428, 2021.
    """
    
    def __init__(self, n_qubits: int, hamiltonian: Dict[str, float],
                 n_layers: int = 3, entangler: str = 'cnot'):
        """
        Initialize VQE for a given Hamiltonian.

        Args:
            n_qubits: Number of qubits.
            hamiltonian: Dictionary mapping Pauli strings to coefficients.
                         e.g., {'X': 1.0, 'Z': 1.0} for H = X + Z.
            n_layers: Number of variational layers (default: 3).
            entangler: Type of entangling gate ('cnot' or 'cz').

        Examples:
            >>> # Hydrogen molecule (simplified)
            >>> hamiltonian = {'Z0': 0.5, 'Z1': 0.5, 'Z0Z1': 0.2}
            >>> vqe = HamiltonianVQE(n_qubits=2, hamiltonian=hamiltonian)
        """
        super().__init__(n_qubits, n_layers, entangler)
        self.hamiltonian = hamiltonian
    
    def evaluate_hamiltonian(self, params: Optional[List[float]] = None) -> float:
        """
        Evaluate the expectation value of the Hamiltonian.

        Args:
            params: Parameters to evaluate (uses current if None).

        Returns:
            Energy expectation value.

        Examples:
            >>> energy = vqe.evaluate_hamiltonian()
            >>> print(f"Energy: {energy:.6f}")
        """
        if params is not None:
            self.set_params(params)
        
        state = self.get_state()
        
        if not state.is_normalized:
            state = state.normalize()
        
        total_energy = 0.0
        
        for pauli_string, coeff in self.hamiltonian.items():
            # Compute expectation of Pauli string
            if pauli_string == 'I':
                exp_val = 1.0
            elif pauli_string == 'X':
                exp_val = self._expectation_x(state)
            elif pauli_string == 'Y':
                exp_val = self._expectation_y(state)
            elif pauli_string == 'Z':
                exp_val = self._expectation_z(state)
            else:
                # Multi-qubit Pauli string
                exp_val = self._expectation_pauli_string(state, pauli_string)
            
            total_energy += coeff * exp_val
        
        return total_energy
    
    def _expectation_z(self, state: Ket) -> float:
        """
        Compute expectation value of Pauli-Z on the first qubit.

        Args:
            state: Quantum state.

        Returns:
            ⟨Z₀⟩.
        """
        n = self.n_qubits
        dim = state.dim
        exp_val = 0.0
        for i in range(dim):
            first_bit = (i >> (n - 1)) & 1
            sign = 1 if first_bit == 0 else -1
            exp_val += sign * abs(state.data[i]) ** 2
        return exp_val
    
    def _expectation_x(self, state: Ket) -> float:
        """
        Compute expectation value of Pauli-X on the first qubit.

        Args:
            state: Quantum state.

        Returns:
            ⟨X₀⟩.
        """
        n = self.n_qubits
        dim = state.dim
        exp_val = 0.0
        for i in range(dim):
            # Apply X on first qubit
            first_bit = (i >> (n - 1)) & 1
            # Flip the bit
            flipped = i ^ (1 << (n - 1))
            exp_val += state.data[i].conjugate() * state.data[flipped]
        return exp_val.real
    
    def _expectation_y(self, state: Ket) -> float:
        """
        Compute expectation value of Pauli-Y on the first qubit.

        Args:
            state: Quantum state.

        Returns:
            ⟨Y₀⟩.
        """
        n = self.n_qubits
        dim = state.dim
        exp_val = 0.0
        for i in range(dim):
            first_bit = (i >> (n - 1)) & 1
            flipped = i ^ (1 << (n - 1))
            factor = -1j if first_bit == 0 else 1j
            exp_val += factor * state.data[i].conjugate() * state.data[flipped]
        return exp_val.real
    
    def _expectation_pauli_string(self, state: Ket, pauli_string: str) -> float:
        """
        Compute expectation value of a multi-qubit Pauli string.

        Args:
            state: Quantum state.
            pauli_string: String of Pauli operators (e.g., 'XIZY').

        Returns:
            ⟨P₀P₁...Pₙ⟩.
        """
        n = len(pauli_string)
        if n != self.n_qubits:
            raise ValueError(f"Pauli string length {n} != n_qubits {self.n_qubits}")
        
        dim = state.dim
        exp_val = 0.0
        
        for i in range(dim):
            # Determine sign from Pauli string
            sign = 1.0
            target_idx = i
            
            for qubit, p in enumerate(pauli_string):
                bit = (i >> (self.n_qubits - 1 - qubit)) & 1
                
                if p == 'X':
                    # Flip bit
                    target_idx ^= (1 << (self.n_qubits - 1 - qubit))
                elif p == 'Y':
                    # Flip bit and add phase
                    target_idx ^= (1 << (self.n_qubits - 1 - qubit))
                    sign *= 1j if bit == 0 else -1j
                elif p == 'Z':
                    sign *= 1 if bit == 0 else -1
                # 'I' does nothing
            
            exp_val += sign * state.data[i].conjugate() * state.data[target_idx]
        
        return exp_val.real
    
    def run(self, n_iterations: int = 100, learning_rate: float = 0.1,
            verbose: bool = True) -> VQCResult:
        """
        Run VQE to find the ground state energy.

        Args:
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            VQCResult with the ground state energy.

        Examples:
            >>> vqe = HamiltonianVQE(1, {'X': 1.0, 'Z': 1.0})
            >>> result = vqe.run(n_iterations=100)
            >>> print(f"Ground state energy: {result.optimal_cost:.6f}")
            Ground state energy: -1.414214
        """
        return self.optimize(
            cost_function=self.evaluate_hamiltonian,
            n_iterations=n_iterations,
            learning_rate=learning_rate,
            verbose=verbose
        )


__all__ = [
    'VQCResult',
    'VariationalLayer',
    'VQC',
    'HamiltonianVQE',
]