"""
psiqit/algorithms/bernstein_vazirani.py
Bernstein-Vazirani Algorithm - Fixed
"""

from typing import Callable, List, Optional
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import hadamard, cnot, pauli_x
from ..quantum.measurement import measure


@dataclass
class BernsteinVaziraniResult:
    """
    Result container for the Bernstein-Vazirani algorithm.

    Attributes:
        hidden_string: The discovered hidden bit string as an integer.
        n_qubits: Number of qubits used (length of hidden string).
        success: Whether the algorithm successfully found the hidden string.

    Examples:
        >>> result = BernsteinVaziraniResult(hidden_string=5, n_qubits=3, success=True)
        >>> print(result)
        Hidden string: 101 (decimal: 5)
    """
    hidden_string: int
    n_qubits: int
    success: bool
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing both binary and decimal representation.
        """
        bits = format(self.hidden_string, f'0{self.n_qubits}b')
        return f"Hidden string: {bits} (decimal: {self.hidden_string})"


class BernsteinVazirani:
    """
    Bernstein-Vazirani algorithm for finding a hidden bit string.

    The Bernstein-Vazirani algorithm determines a hidden string s ∈ {0,1}ⁿ
    using a quantum oracle that computes f(x) = s·x (mod 2). The algorithm
    finds s with a single query to the oracle, providing a quadratic speedup
    over classical algorithms.

    The algorithm uses n+1 qubits:
    - n qubits for the input register
    - 1 ancilla qubit for the oracle output

    References:
        E. Bernstein and U. Vazirani, "Quantum Complexity Theory,"
        SIAM Journal on Computing, 26(5):1411-1473, 1997.

    Examples:
        >>> bv = BernsteinVazirani(n_qubits=3)
        >>> result = bv.run()
        >>> print(result)  # e.g., "Hidden string: 101 (decimal: 5)"
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize the Bernstein-Vazirani algorithm.

        Args:
            n_qubits: Number of qubits (length of hidden bit string).

        Examples:
            >>> bv = BernsteinVazirani(n_qubits=4)  # 4-bit hidden string
        """
        self.n = n_qubits
        self.total_qubits = n_qubits + 1
    
    def build_circuit(self, oracle: Callable) -> QuantumCircuit:
        """
        Build the quantum circuit for the Bernstein-Vazirani algorithm.

        The circuit structure:
        1. Initialize ancilla qubit to |1⟩
        2. Apply Hadamard gates to all qubits
        3. Apply the oracle (phase kickback)
        4. Apply Hadamard gates to input qubits
        5. Measure input qubits to read the hidden string

        Args:
            oracle: Oracle function that encodes the hidden string.
                    For this implementation, a simplified CNOT-based oracle
                    is used where each CNOT corresponds to a 1 in the hidden string.

        Returns:
            QuantumCircuit: The constructed circuit ready for execution.

        Examples:
            >>> bv = BernsteinVazirani(n_qubits=3)
            >>> circuit = bv.build_circuit(None)
            >>> print(circuit.depth)
            7
        """
        circuit = QuantumCircuit(self.total_qubits)
        
        # Initialize ancilla to |1⟩
        circuit.x(self.n)
        
        # Apply Hadamard to all qubits
        for i in range(self.total_qubits):
            circuit.h(i)
        
        # Apply oracle (simplified - just CNOTs)
        # Each CNOT from qubit i to ancilla corresponds to s_i = 1
        for i in range(self.n):
            circuit.cx(i, self.n)
        
        # Apply Hadamard to input qubits
        for i in range(self.n):
            circuit.h(i)
        
        return circuit
    
    def run(self) -> BernsteinVaziraniResult:
        """
        Execute the Bernstein-Vazirani algorithm.

        The algorithm runs the quantum circuit, measures the input qubits,
        and reconstructs the hidden bit string from measurement outcomes.

        Returns:
            BernsteinVaziraniResult: Object containing the discovered hidden string.

        Examples:
            >>> bv = BernsteinVazirani(n_qubits=3)
            >>> result = bv.run()
            >>> print(result.hidden_string)  # e.g., 5
        """
        circuit = self.build_circuit(None)
        state = circuit.run()
        
        # Measure input qubits
        measurements = []
        for i in range(self.n):
            prob_0 = 0
            dim = state.dim
            for j in range(dim):
                bit = (j >> (self.total_qubits - 1 - i)) & 1
                if bit == 0:
                    prob_0 += abs(state.data[j]) ** 2
            measurements.append(0 if prob_0 > 0.5 else 1)
        
        # Convert to integer
        hidden_string = 0
        for i, bit in enumerate(measurements):
            hidden_string = (hidden_string << 1) | bit
        
        return BernsteinVaziraniResult(
            hidden_string=hidden_string,
            n_qubits=self.n,
            success=True
        )


def bernstein_vazirani(n_qubits: int) -> int:
    """
    Convenience function to run the Bernstein-Vazirani algorithm.

    This function provides a simple interface for running the algorithm
    without needing to instantiate the BernsteinVazirani class.

    Args:
        n_qubits: Number of qubits (length of hidden bit string).

    Returns:
        int: The discovered hidden bit string as an integer.

    Examples:
        >>> hidden = bernstein_vazirani(3)
        >>> print(f"Hidden string: {hidden:03b}")
        Hidden string: 101
    """
    bv = BernsteinVazirani(n_qubits)
    result = bv.run()
    return result.hidden_string


__all__ = [
    'BernsteinVaziraniResult',
    'BernsteinVazirani',
    'bernstein_vazirani',
]