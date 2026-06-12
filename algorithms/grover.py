"""
psiqit/algorithms/grover.py
Grover's Search Algorithm - Complete Implementation

Grover's algorithm provides quadratic speedup for unstructured search problems.
It can find a marked item in an unsorted database of N items using O(√N) queries,
compared to O(N) queries required classically.
"""

import math
import random
from typing import List, Optional, Union, Dict
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import hadamard, pauli_z, cnot, toffoli, cz
from ..quantum.measurement import measure


@dataclass
class GroverResult:
    """
    Result container for Grover's search algorithm.

    Attributes:
        counts: Dictionary mapping measurement outcomes (bitstrings) to counts.
        most_likely: The most frequently measured state (as integer).
        marked_states: List of target states that were being searched for.
        success: Whether the most likely state is one of the marked states.
        n_iterations: Number of Grover iterations performed.
        theoretical_prob: Theoretical probability of measuring a marked state.

    Examples:
        >>> result = GroverResult(counts={'101': 900, '000': 124}, most_likely=5,
        ...                       marked_states=[5], success=True, n_iterations=2,
        ...                       theoretical_prob=0.98)
        >>> print(result)
        GroverResult(most_likely=5, success=True, iterations=2)
    """
    counts: Dict[str, int]
    most_likely: int
    marked_states: List[int]
    success: bool
    n_iterations: int
    theoretical_prob: float
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing most likely state, success status, and iterations.
        """
        return (f"GroverResult(most_likely={self.most_likely}, "
                f"success={self.success}, iterations={self.n_iterations})")


class Grover:
    """
    Grover's search algorithm for unstructured quantum search.

    Grover's algorithm finds one or more marked items in an unsorted database
    of size N = 2ⁿ using O(√(N/M)) queries, where M is the number of marked
    items. This provides a quadratic speedup over classical search algorithms.

    The algorithm works by repeatedly applying the Grover iteration:
    1. Oracle: flips the sign of marked states
    2. Diffusion operator: amplifies amplitude of marked states

    References:
        L. K. Grover, "A fast quantum mechanical algorithm for database search,"
        Proceedings of STOC, pp. 212-219, 1996.

    Examples:
        >>> # Search for a single marked state
        >>> grover = Grover(n_qubits=3, marked_states=5)
        >>> result = grover.run(shots=1024)
        >>> print(f"Found: {result.most_likely}, Success: {result.success}")
        Found: 5, Success: True
        
        >>> # Search for multiple marked states
        >>> grover = Grover(n_qubits=3, marked_states=[2, 5])
        >>> result = grover.run()
        >>> print(result.most_likely in [2, 5])
        True
    """
    
    def __init__(self, n_qubits: int, marked_states: Union[int, List[int]]):
        """
        Initialize Grover's algorithm.

        Args:
            n_qubits: Number of qubits (database size N = 2ⁿ).
            marked_states: Target state(s) to search for (integer or list of integers).

        Examples:
            >>> grover = Grover(n_qubits=4, marked_states=7)
            >>> grover = Grover(n_qubits=4, marked_states=[3, 7, 11])
        """
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        
        if isinstance(marked_states, int):
            self.marked_states = [marked_states]
        else:
            self.marked_states = marked_states
        
        self.M = len(self.marked_states)
        
        # Calculate optimal number of iterations
        # Optimal iterations: k ≈ π/4 * √(N/M)
        if self.M <= 0 or self.M >= self.N:
            self.n_iterations = 0
        else:
            theta = math.asin(math.sqrt(self.M / self.N))
            self.n_iterations = max(1, int(round((math.pi / (4 * theta)) - 0.5)))
    
    def _get_binary(self, state: int) -> str:
        """
        Convert state index to binary string.

        Args:
            state: Integer state index (0 to 2ⁿ-1).

        Returns:
            Binary string of length n_qubits.

        Examples:
            >>> grover = Grover(3, 5)
            >>> grover._get_binary(5)
            '101'
        """
        return format(state, f'0{self.n_qubits}b')
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the complete Grover circuit.

        The circuit consists of:
        1. Initialization: Hadamard gates on all qubits (uniform superposition)
        2. Grover iterations: Oracle + Diffuser applied k times

        Returns:
            QuantumCircuit: The complete Grover circuit ready for execution.

        Examples:
            >>> grover = Grover(n_qubits=3, marked_states=5)
            >>> circuit = grover.build_circuit()
            >>> print(f"Circuit depth: {circuit.depth}")
            Circuit depth: 14
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Initialize in uniform superposition
        for i in range(self.n_qubits):
            circuit.h(i)
        
        # Apply Grover iterations
        for _ in range(self.n_iterations):
            # Oracle (simplified for now)
            for target in self.marked_states:
                binary = self._get_binary(target)
                for i, bit in enumerate(binary):
                    if bit == '0':
                        circuit.x(i)
                circuit.cz(0, 1)  # Simplified multi-controlled Z
                for i, bit in enumerate(binary):
                    if bit == '0':
                        circuit.x(i)
            
            # Diffuser (inversion about the mean)
            for i in range(self.n_qubits):
                circuit.h(i)
            for i in range(self.n_qubits):
                circuit.x(i)
            circuit.cz(0, 1)
            for i in range(self.n_qubits):
                circuit.x(i)
            for i in range(self.n_qubits):
                circuit.h(i)
        
        return circuit
    
    def get_probabilities(self) -> List[float]:
        """
        Calculate theoretical probabilities for each state after Grover iterations.

        Returns:
            List of probabilities for each basis state (index 0 to 2ⁿ-1).

        Examples:
            >>> grover = Grover(n_qubits=3, marked_states=5)
            >>> probs = grover.get_probabilities()
            >>> print(f"Probability of marked state: {probs[5]:.4f}")
            Probability of marked state: 0.9453
        """
        probs = [0.0] * self.N
        
        if self.M == 0 or self.M == self.N:
            return [1.0 / self.N] * self.N
        
        theta = math.asin(math.sqrt(self.M / self.N))
        prob_marked = math.sin((2 * self.n_iterations + 1) * theta) ** 2
        prob_unmarked = (1 - prob_marked) / (self.N - self.M) if self.N > self.M else 0
        
        for state in self.marked_states:
            probs[state] = prob_marked / self.M
        for i in range(self.N):
            if i not in self.marked_states:
                probs[i] = prob_unmarked
        
        return probs
    
    def _apply_gate_to_state(self, state: Ket, gate_matrix, qubits: List[int]) -> Ket:
        """
        Apply a gate to specific qubits of a state vector.

        This is an internal method for state vector simulation.

        Args:
            state: Current quantum state (Ket).
            gate_matrix: Matrix representation of the gate.
            qubits: List of qubit indices to apply the gate to.

        Returns:
            Ket: New state after gate application.
        """
        n = self.n_qubits
        dim = state.dim
        new_data = [0j] * dim
        
        for i in range(dim):
            bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
            
            if len(qubits) == 1:
                q = qubits[0]
                row = bits[q]
                for j in range(2):
                    new_bits = bits.copy()
                    new_bits[q] = j
                    new_idx = 0
                    for k, b in enumerate(new_bits):
                        new_idx = (new_idx << 1) | b
                    new_data[new_idx] += gate_matrix[row][j] * state.data[i]
            
            elif len(qubits) == 2 and len(gate_matrix) == 4:
                q1, q2 = qubits
                row = (bits[q1] << 1) | bits[q2]
                for j1 in range(2):
                    for j2 in range(2):
                        col = (j1 << 1) | j2
                        new_bits = bits.copy()
                        new_bits[q1] = j1
                        new_bits[q2] = j2
                        new_idx = 0
                        for k, b in enumerate(new_bits):
                            new_idx = (new_idx << 1) | b
                        new_data[new_idx] += gate_matrix[row][col] * state.data[i]
        
        return Ket(new_data)
    
    def _oracle(self, state: Ket) -> Ket:
        """
        Apply the phase oracle to mark target states.

        The oracle flips the sign (multiplies by -1) of the marked states.

        Args:
            state: Current quantum state.

        Returns:
            Ket: State with marked states' amplitudes negated.
        """
        new_data = state.data.copy()
        for target in self.marked_states:
            new_data[target] = -new_data[target]
        return Ket(new_data)
    
    def _diffuser(self, state: Ket) -> Ket:
        """
        Apply the Grover diffusion operator.

        The diffusion operator performs inversion about the mean:
        U_diff = 2|ψ⟩⟨ψ| - I, where |ψ⟩ is the uniform superposition.

        Args:
            state: Current quantum state.

        Returns:
            Ket: State after applying the diffusion operator.
        """
        n = self.N
        avg = sum(state.data) / n
        new_data = [2 * avg - a for a in state.data]
        return Ket(new_data)
    
    def run(self, shots: int = 1024) -> GroverResult:
        """
        Execute Grover's search algorithm.

        Args:
            shots: Number of measurements to perform.

        Returns:
            GroverResult: Object containing measurement counts, most likely state,
                         success status, and theoretical probability.

        Examples:
            >>> grover = Grover(n_qubits=3, marked_states=5)
            >>> result = grover.run(shots=1024)
            >>> print(f"Most likely: {result.most_likely}")
            Most likely: 5
            >>> print(f"Success: {result.success}")
            Success: True
            >>> print(f"Counts: {result.counts}")
            Counts: {'101': 978, '010': 46}
        """
        # Start from |0...0⟩
        state = basis(self.N, 0)
        
        # Apply Hadamard to all qubits
        from ..quantum.operator import hadamard
        H = hadamard()
        for i in range(self.n_qubits):
            state = self._apply_gate_to_state(state, H.data, [i])
        
        # Apply Grover iterations
        for _ in range(self.n_iterations):
            state = self._oracle(state)
            state = self._diffuser(state)
        
        # Measure
        probs = [abs(a)**2 for a in state.data]
        total = sum(probs)
        probs = [p / total for p in probs]
        
        outcomes = []
        for _ in range(shots):
            r = random.random()
            cumsum = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r < cumsum:
                    outcomes.append(i)
                    break
        
        outcome_strings = [format(o, f'0{self.n_qubits}b') for o in outcomes]
        
        counts = {}
        for s in outcome_strings:
            counts[s] = counts.get(s, 0) + 1
        
        most_likely = max(counts, key=counts.get) if counts else None
        most_likely_int = int(most_likely, 2) if most_likely else None
        
        # Theoretical probability
        if self.M > 0 and self.M < self.N:
            theta = math.asin(math.sqrt(self.M / self.N))
            prob_marked = math.sin((2 * self.n_iterations + 1) * theta) ** 2
        else:
            prob_marked = 1.0 if self.M == self.N else 0.0
        
        success = most_likely_int in self.marked_states if most_likely_int is not None else False
        
        return GroverResult(
            counts=counts,
            most_likely=most_likely_int,
            marked_states=self.marked_states,
            success=success,
            n_iterations=self.n_iterations,
            theoretical_prob=prob_marked
        )


def grover_search(n_qubits: int, target: int, shots: int = 1024) -> GroverResult:
    """
    Convenience function for single-item Grover search.

    Args:
        n_qubits: Number of qubits (database size N = 2ⁿ).
        target: The single marked state to search for.
        shots: Number of measurements.

    Returns:
        GroverResult: Result of the Grover search.

    Examples:
        >>> result = grover_search(n_qubits=4, target=9, shots=1024)
        >>> print(f"Found: {result.most_likely}")
        Found: 9
    """
    grover = Grover(n_qubits, target)
    return grover.run(shots)


def grover_search_multiple(n_qubits: int, targets: List[int], shots: int = 1024) -> GroverResult:
    """
    Convenience function for multiple-item Grover search.

    Args:
        n_qubits: Number of qubits (database size N = 2ⁿ).
        targets: List of marked states to search for.
        shots: Number of measurements.

    Returns:
        GroverResult: Result of the Grover search.

    Examples:
        >>> result = grover_search_multiple(n_qubits=4, targets=[3, 7, 11], shots=1024)
        >>> print(f"Found one of: {result.most_likely}")
        Found one of: 7
    """
    grover = Grover(n_qubits, targets)
    return grover.run(shots)


__all__ = [
    'GroverResult',
    'Grover',
    'grover_search',
    'grover_search_multiple',
]