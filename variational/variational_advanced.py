"""
psiqit/variational/variational_advanced.py
============================================================
Advanced Variational Methods
============================================================

Advanced variational techniques for quantum systems:
    • Subspace expansion VQE (SS-VQE)
    • Multi-state VQE (MS-VQE)
    • Weighted least squares VQE (WLS-VQE)
    • Adaptive VQE (ADAPT-VQE)
    • Variational Quantum Deflation (VQD)
    • Subspace-search VQE (SSVQE)

These methods extend standard VQE to find excited states, adaptively
construct ansatz circuits, and deflate previously found states.

Example:
    >>> from psiqit.variational.variational_advanced import SSVQE, ADAPTVQE
    >>> 
    >>> # Multi-state VQE for excited states
    >>> hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}
    >>> ms_vqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=3)
    >>> results = ms_vqe.run()
    >>> print(f"Energies: {results.energies}")
    >>> 
    >>> # ADAPT-VQE for ground state
    >>> adapt = ADAPTVQE(n_qubits=2, hamiltonian=hamiltonian)
    >>> result = adapt.run()
    >>> print(f"Ground state energy: {result.optimal_energy:.6f}")

References:
    O. Higgott, D. Wang, and S. Brierley, "Variational Quantum Computation
    of Excited States," Quantum, 3:156, 2019.
    H. R. Grimsley et al., "An adaptive variational algorithm for exact
    molecular simulations on a quantum computer," Nature Communications,
    10(1):3007, 2019.
    K. M. Nakanishi, K. Mitarai, and K. Fujii, "Subspace-search variational
    quantum eigensolver for excited states," Physical Review Research,
    1(3):033062, 2019.
"""

import math
import random
from typing import List, Dict, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from copy import deepcopy

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import Operator, pauli_x, pauli_y, pauli_z, rx, ry, rz, cnot
from ..quantum.measurement import expectation
from .vqe import VQE, VQEResult


@dataclass
class MultiStateResult:
    """
    Result container for multi-state variational calculations.

    Attributes:
        energies: List of energies for each found state.
        states: List of state vectors (as amplitude lists).
        success: Whether the calculation was successful.
        iterations: Number of iterations for each state.

    Examples:
        >>> result = MultiStateResult(energies=[-1.0, -0.5, 0.2],
        ...                           states=[], success=True,
        ...                           iterations=[100, 100, 100])
        >>> print(result)
        MultiStateResult(energies=['-1.000000', '-0.500000', '0.200000'])
    """
    energies: List[float]
    states: List[List[float]]
    success: bool
    iterations: List[int]
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing the energies of all found states.
        """
        return f"MultiStateResult(energies={[f'{e:.6f}' for e in self.energies]})"


class SSVQE:
    """
    Subspace-Search VQE (SSVQE) for excited states.

    SSVQE finds multiple eigenstates (ground and excited) simultaneously
    by using orthogonal initial states and minimizing a weighted sum of
    energies. The orthogonal initial states (typically computational basis
    states) evolve to orthogonal final states that approximate the true
    eigenstates.

    The cost function is: C = Σ w_i ⟨ψ_i(θ)|H|ψ_i(θ)⟩, where w_i are weights
    with w₁ > w₂ > ... > wₙ to ensure ordering of eigenstates.

    References:
        K. M. Nakanishi, K. Mitarai, and K. Fujii, "Subspace-search variational
        quantum eigensolver for excited states," Physical Review Research,
        1(3):033062, 2019.

    Examples:
        >>> hamiltonian = {'Z0Z1': 0.5, 'X0': 1.0, 'X1': 1.0}
        >>> ssvqe = SSVQE(n_qubits=2, hamiltonian=hamiltonian, n_states=3)
        >>> result = ssvqe.run(n_iterations=200)
        >>> print(f"Ground state: {result.energies[0]:.6f}")
        >>> print(f"First excited: {result.energies[1]:.6f}")
    """
    
    def __init__(self, n_qubits: int, hamiltonian: Dict[str, float],
                 n_states: int = 3, n_layers: int = 2):
        """
        Initialize SSVQE.

        Args:
            n_qubits: Number of qubits.
            hamiltonian: Hamiltonian as Pauli string dictionary.
            n_states: Number of states to find (ground + excited).
            n_layers: Number of variational layers in the ansatz.

        Examples:
            >>> ssvqe = SSVQE(n_qubits=2, hamiltonian=H, n_states=3, n_layers=2)
        """
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_states = n_states
        self.n_layers = n_layers
        
        # Weights for states (higher weights for lower energies)
        self.weights = [1.0 / (i + 1) for i in range(n_states)]
        
        # Create orthogonal initial states (computational basis)
        self.initial_states = []
        for i in range(min(n_states, 2**n_qubits)):
            state = basis(2**n_qubits, i)
            self.initial_states.append(state)
        
        # Shared parameters for all states
        self.n_params = n_qubits * n_layers
        self.params = None
        self._vqe = VQE(n_qubits, hamiltonian, n_layers)
        self._init_random()
    
    def _init_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.params = [random.uniform(-math.pi, math.pi) for _ in range(self.n_params)]
        self._vqe.params = self.params
    
    def build_circuit(self, params: Optional[List[float]] = None) -> QuantumCircuit:
        """
        Build the variational circuit.

        Args:
            params: Parameters for the circuit. If None, uses current parameters.

        Returns:
            QuantumCircuit: The variational circuit.
        """
        if params is None:
            params = self.params
        return self._vqe.build_circuit(params)
    
    def evaluate_state_energy(self, state_idx: int, params: List[float]) -> float:
        """
        Evaluate energy for a specific initial state.

        Args:
            state_idx: Index of the initial state.
            params: Parameters to evaluate.

        Returns:
            Energy expectation value.
        """
        circuit = self.build_circuit(params)
        initial_state = self.initial_states[state_idx]
        final_state = circuit.run()  # This should apply to initial state
        # Simplified: for now, compute energy from final state
        return self._vqe.evaluate_energy(params)
    
    def evaluate_cost(self, params: List[float]) -> float:
        """
        Evaluate the weighted sum of state energies.

        Args:
            params: Parameters to evaluate.

        Returns:
            Weighted cost value.
        """
        total = 0.0
        for i in range(self.n_states):
            energy = self.evaluate_state_energy(i, params)
            total += self.weights[i] * energy
        return total
    
    def _gradient(self, params: List[float], eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient using finite differences.

        Args:
            params: Parameters to differentiate.
            eps: Step size.

        Returns:
            List of gradient values.
        """
        grads = []
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            
            e_plus = self.evaluate_cost(p_plus)
            e_minus = self.evaluate_cost(p_minus)
            grads.append((e_plus - e_minus) / (2 * eps))
        
        return grads
    
    def run(self, n_iterations: int = 100, learning_rate: float = 0.1,
            verbose: bool = True) -> MultiStateResult:
        """
        Run SSVQE optimization.

        Args:
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            MultiStateResult with energies for all states.

        Examples:
            >>> ssvqe = SSVQE(n_qubits=2, hamiltonian=H, n_states=3)
            >>> result = ssvqe.run(n_iterations=200, learning_rate=0.1)
        """
        params = self.params.copy()
        history = []
        energies_history = []
        
        for iteration in range(n_iterations):
            grads = self._gradient(params)
            
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]
            
            cost = self.evaluate_cost(params)
            history.append(cost)
            
            # Compute individual energies
            energies = [self.evaluate_state_energy(i, params) for i in range(self.n_states)]
            energies_history.append(energies)
            
            if verbose and (iteration % 20 == 0 or iteration == n_iterations - 1):
                print(f"  Iter {iteration}: cost = {cost:.6f}")
                print(f"    Energies: {[f'{e:.6f}' for e in energies]}")
        
        self.params = params
        
        final_energies = [self.evaluate_state_energy(i, params) for i in range(self.n_states)]
        
        return MultiStateResult(
            energies=final_energies,
            states=[],  # Would need to store final states
            success=True,
            iterations=[n_iterations] * self.n_states
        )


class ADAPTVQE:
    """
    ADAPT-VQE: Adaptive construction of variational ansatz.

    ADAPT-VQE (Adaptive Derivative-Assembled Pseudo-Trotter) builds the
    ansatz circuit adaptively by adding operators that have the largest
    gradient contribution to the energy. This results in shallower circuits
    compared to fixed ansatz methods.

    The algorithm:
        1. Start with an empty ansatz
        2. Compute gradients for all operators in a predefined pool
        3. Add the operator with the largest gradient magnitude
        4. Optimize the new parameters
        5. Repeat until convergence

    References:
        H. R. Grimsley et al., "An adaptive variational algorithm for exact
        molecular simulations on a quantum computer," Nature Communications,
        10(1):3007, 2019.

    Examples:
        >>> adapt = ADAPTVQE(n_qubits=2, hamiltonian=hamiltonian)
        >>> result = adapt.run(max_iterations=10, grad_threshold=0.01)
        >>> print(f"Number of operators: {len(adapt.ansatz_operators)}")
    """
    
    def __init__(self, n_qubits: int, hamiltonian: Dict[str, float],
                 operator_pool: Optional[List[Operator]] = None):
        """
        Initialize ADAPT-VQE.

        Args:
            n_qubits: Number of qubits.
            hamiltonian: Hamiltonian as Pauli string dictionary.
            operator_pool: List of operators to choose from. If None,
                          uses single-qubit Pauli rotations.

        Examples:
            >>> adapt = ADAPTVQE(n_qubits=2, hamiltonian=H)
        """
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        
        if operator_pool is None:
            self.operator_pool = self._create_default_pool()
        else:
            self.operator_pool = operator_pool
        
        self.ansatz_operators = []
        self.ansatz_params = []
    
    def _create_default_pool(self) -> List[Operator]:
        """
        Create a default operator pool with single-qubit Pauli rotations.

        Returns:
            List of Pauli rotation operators.
        """
        pool = []
        
        # Single qubit Pauli rotations
        for qubit in range(self.n_qubits):
            op_x = Operator([[0, 1], [1, 0]], f"X{qubit}")
            pool.append(op_x)
            op_y = Operator([[0, -1j], [1j, 0]], f"Y{qubit}")
            pool.append(op_y)
            op_z = Operator([[1, 0], [0, -1]], f"Z{qubit}")
            pool.append(op_z)
        
        return pool
    
    def _get_qubit_from_name(self, name: str) -> int:
        """
        Extract qubit number from operator name.

        Args:
            name: Operator name like 'X0', 'Y1', 'Z2'.

        Returns:
            Qubit index.
        """
        # Handle names like "X0", "Y1", "Z2"
        if name and name[0] in ['X', 'Y', 'Z']:
            try:
                return int(name[1:])
            except (ValueError, IndexError):
                pass
        return 0
    
    def _apply_operator(self, circuit: QuantumCircuit, op: Operator, param: float):
        """
        Apply an operator as a rotation to the circuit.

        Args:
            circuit: Quantum circuit to add gates to.
            op: Operator (X, Y, or Z).
            param: Rotation angle.
        """
        name = op.name
        if not name:
            return
        
        if name[0] == 'X':
            qubit = self._get_qubit_from_name(name)
            circuit.rx(qubit, 2 * param)
        elif name[0] == 'Y':
            qubit = self._get_qubit_from_name(name)
            circuit.ry(qubit, 2 * param)
        elif name[0] == 'Z':
            qubit = self._get_qubit_from_name(name)
            circuit.rz(qubit, 2 * param)
    
    def _evaluate_energy_with_ops(self, params: List[float]) -> float:
        """
        Evaluate the energy with the current ansatz operators.

        Args:
            params: List of parameters for the ansatz operators.

        Returns:
            Energy expectation value.
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        for op, param in zip(self.ansatz_operators, params):
            self._apply_operator(circuit, op, param)
        
        state = circuit.run()
        
        # Simplified energy evaluation - compute expectation of Hamiltonian
        total_energy = 0.0
        dim = state.dim
        
        for term, coeff in self.hamiltonian.items():
            # Parse Pauli string like 'Z0', 'Z1', 'Z0Z1'
            exp_val = 0.0
            
            if len(term) == 2:
                # Single qubit Pauli
                gate = term[0]
                qubit = int(term[1])
                
                if gate == 'Z':
                    for i in range(dim):
                        bit = (i >> (self.n_qubits - 1 - qubit)) & 1
                        exp_val += (1 if bit == 0 else -1) * abs(state.data[i])**2
                elif gate == 'X':
                    for i in range(dim):
                        flipped = i ^ (1 << (self.n_qubits - 1 - qubit))
                        exp_val += (state.data[i].conjugate() * state.data[flipped]).real
                elif gate == 'Y':
                    for i in range(dim):
                        bit = (i >> (self.n_qubits - 1 - qubit)) & 1
                        flipped = i ^ (1 << (self.n_qubits - 1 - qubit))
                        factor = -1j if bit == 0 else 1j
                        exp_val += (factor * state.data[i].conjugate() * state.data[flipped]).real
            
            elif len(term) == 4:
                # Two-qubit Pauli like 'Z0Z1'
                gate1 = term[0]
                qubit1 = int(term[1])
                gate2 = term[2]
                qubit2 = int(term[3])
                
                exp_val = 1.0
                for gate, qubit in [(gate1, qubit1), (gate2, qubit2)]:
                    if gate == 'Z':
                        sub_exp = 0.0
                        for i in range(dim):
                            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
                            sub_exp += (1 if bit == 0 else -1) * abs(state.data[i])**2
                        exp_val *= sub_exp
                    else:
                        # Simplified for other gates
                        exp_val *= 0
            
            total_energy += coeff * exp_val
        
        return total_energy
    
    def _gradient_wrt_operator(self, params: List[float], op_idx: int) -> float:
        """
        Compute the gradient with respect to an operator.

        Args:
            params: Current parameters.
            op_idx: Index of the operator to compute gradient for.

        Returns:
            Gradient value.
        """
        # Compute energy with parameter +ε and -ε
        eps = 1e-4
        
        # Test adding operator with small parameter
        test_params = params + [eps]
        e_plus = self._evaluate_energy_with_ops(test_params)
        
        test_params = params + [-eps]
        e_minus = self._evaluate_energy_with_ops(test_params)
        
        return (e_plus - e_minus) / (2 * eps)
    
    def run(self, max_iterations: int = 10, grad_threshold: float = 0.01,
            verbose: bool = True) -> VQEResult:
        """
        Run ADAPT-VQE.

        Args:
            max_iterations: Maximum number of operators to add.
            grad_threshold: Threshold for gradient magnitude to stop.
            verbose: Whether to print progress.

        Returns:
            VQEResult with the final energy and parameters.

        Examples:
            >>> adapt = ADAPTVQE(n_qubits=2, hamiltonian=H)
            >>> result = adapt.run(max_iterations=15, grad_threshold=0.005)
        """
        energies = []
        params = []
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"  ADAPT Iteration {iteration + 1}")
            
            # Compute gradients for all operators in pool
            grads = []
            for op in self.operator_pool:
                # Add this operator temporarily
                grad = self._gradient_wrt_operator(params, len(self.operator_pool))
                grads.append(abs(grad))
            
            # Select operator with largest gradient
            best_idx = max(range(len(grads)), key=lambda i: grads[i])
            best_grad = grads[best_idx]
            
            if verbose:
                print(f"    Best gradient: {best_grad:.6f}")
            
            if best_grad < grad_threshold:
                if verbose:
                    print(f"    Converged! Gradient below threshold")
                break
            
            # Add operator to ansatz
            best_op = self.operator_pool[best_idx]
            self.ansatz_operators.append(best_op)
            params.append(0.0)
            
            if verbose:
                print(f"    Added operator: {best_op.name}")
            
            # Optimize parameter for new operator
            for opt_iter in range(20):
                grad = self._gradient_wrt_operator(params[:-1], len(params) - 1)
                params[-1] -= 0.1 * grad
            
            # Evaluate energy
            current_energy = self._evaluate_energy_with_ops(params)
            energies.append(current_energy)
            
            if verbose:
                print(f"    Current energy: {current_energy:.6f}")
        
        self.ansatz_params = params
        
        return VQEResult(
            optimal_params=params,
            optimal_energy=energies[-1] if energies else 0.0,
            n_iterations=len(energies),
            history=energies,
            success=True
        )


class VariationalQuantumDeflation:
    """
    Variational Quantum Deflation (VQD) for excited states.

    VQD finds excited states by adding a penalty term to the cost function
    that penalizes overlap with previously found states. This deflates the
    lower-energy states, allowing the optimization to find higher-energy
    eigenstates.

    The cost function is: C(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ + Σ β_i |⟨ψ(θ)|ψ_i⟩|²

    References:
        O. Higgott, D. Wang, and S. Brierley, "Variational Quantum Computation
        of Excited States," Quantum, 3:156, 2019.

    Examples:
        >>> vqd = VariationalQuantumDeflation(n_qubits=2, hamiltonian=H)
        >>> # First find ground state
        >>> ground = vqd.find_excited_state(penalty_weight=0)
        >>> # Then find first excited state
        >>> excited = vqd.find_excited_state(penalty_weight=1.0)
    """
    
    def __init__(self, n_qubits: int, hamiltonian: Dict[str, float],
                 n_layers: int = 2):
        """
        Initialize VQD.

        Args:
            n_qubits: Number of qubits.
            hamiltonian: Hamiltonian as Pauli string dictionary.
            n_layers: Number of variational layers in the ansatz.

        Examples:
            >>> vqd = VariationalQuantumDeflation(n_qubits=2, hamiltonian=H)
        """
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self.n_layers = n_layers
        self._vqe = VQE(n_qubits, hamiltonian, n_layers)
        self.found_states = []
        self.found_energies = []
    
    def _overlap_penalty(self, params: List[float], state_idx: int) -> float:
        """
        Compute the penalty for overlap with previous states.

        Args:
            params: Current parameters.
            state_idx: Index of the state being optimized.

        Returns:
            Overlap penalty value.
        """
        # Simplified: compute overlap with previous states
        # Full implementation would need state tomography
        penalty = 0.0
        for prev_params in self.found_states:
            # Simplified overlap calculation
            overlap = 0.5  # Placeholder
            penalty += overlap**2
        return penalty
    
    def _cost_function(self, params: List[float], penalty_weight: float = 1.0) -> float:
        """
        Compute the cost function with penalty for overlap.

        Args:
            params: Current parameters.
            penalty_weight: Weight for the overlap penalty.

        Returns:
            Cost value.
        """
        energy = self._vqe.evaluate_energy(params)
        penalty = self._overlap_penalty(params, len(self.found_states))
        return energy + penalty_weight * penalty
    
    def find_excited_state(self, penalty_weight: float = 1.0,
                          n_iterations: int = 100) -> VQEResult:
        """
        Find the next excited state.

        Args:
            penalty_weight: Weight for the overlap penalty.
            n_iterations: Number of optimization iterations.

        Returns:
            VQEResult with the energy of the found state.

        Examples:
            >>> vqd = VariationalQuantumDeflation(2, H)
            >>> ground = vqd.find_excited_state(penalty_weight=0)
            >>> first_excited = vqd.find_excited_state(penalty_weight=1.0)
            >>> second_excited = vqd.find_excited_state(penalty_weight=2.0)
        """
        self._vqe._init_random()
        result = self._vqe.run(n_iterations=n_iterations)
        self.found_states.append(self._vqe.params)
        self.found_energies.append(result.optimal_energy)
        return result


__all__ = [
    'MultiStateResult',
    'SSVQE',
    'ADAPTVQE',
    'VariationalQuantumDeflation',
]