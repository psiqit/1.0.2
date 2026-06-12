"""
psiit/algorithms/deutsch_jozsa.py
Deutsch-Jozsa Algorithm - Fixed with circuit.measure

The Deutsch-Jozsa algorithm determines whether a Boolean function
f: {0,1}ⁿ → {0,1} is constant or balanced using a single quantum query,
providing exponential speedup over classical algorithms.
"""

import math
from typing import List, Optional, Union, Callable, Dict
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, basis
from ..quantum.operator import hadamard, pauli_x, cnot, toffoli
from ..quantum.measurement import measure


@dataclass
class DeutschJozsaResult:
    """
    Result container for the Deutsch-Jozsa algorithm.

    Attributes:
        is_constant: True if the function is constant (always returns 0 or 1).
        is_balanced: True if the function is balanced (half 0s, half 1s).
        measurement_result: Integer representation of the measurement outcome.
        n_qubits: Number of input qubits.
        oracle_type: Type of oracle used ('constant', 'balanced', or 'custom').
        confidence: Confidence level of the result (0 to 1).

    Examples:
        >>> result = DeutschJozsaResult(is_constant=True, is_balanced=False, 
        ...                             measurement_result=0, n_qubits=3,
        ...                             oracle_type='constant', confidence=1.0)
        >>> print(result)
        Function is CONSTANT (f(x) = 0 for all x)
    """
    is_constant: bool
    is_balanced: bool
    measurement_result: int
    n_qubits: int
    oracle_type: str
    confidence: float = 1.0
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String indicating whether the function is constant or balanced.
        """
        if self.is_constant:
            return f"Function is CONSTANT (f(x) = 0 for all x)"
        else:
            return f"Function is BALANCED (exactly half 0s, half 1s)"


class DeutschJozsa:
    """
    Deutsch-Jozsa algorithm for determining if a Boolean function is constant or balanced.

    The algorithm distinguishes between constant and balanced functions using
    a single quantum query, providing an exponential speedup over classical
    algorithms that require 2ⁿ⁻¹+1 queries in the worst case.

    The algorithm uses n+1 qubits:
    - n qubits for the input register
    - 1 ancilla qubit for the oracle output (phase kickback)

    References:
        D. Deutsch and R. Jozsa, "Rapid solution of problems by quantum computation,"
        Proceedings of the Royal Society of London A, 439(1907):553-558, 1992.

    Examples:
        >>> # Test a constant function
        >>> dj = DeutschJozsa(n_qubits=3, oracle_type='constant', constant_value=0)
        >>> result = dj.run()
        >>> print(result)
        Function is CONSTANT (f(x) = 0 for all x)
        
        >>> # Test a balanced function
        >>> dj = DeutschJozsa(n_qubits=3, oracle_type='balanced')
        >>> result = dj.run()
        >>> print(result)
        Function is BALANCED (exactly half 0s, half 1s)
    """
    
    def __init__(self, n_qubits: int, 
                 oracle_type: str = 'constant',
                 constant_value: int = 0,
                 custom_oracle: Optional[Callable] = None):
        """
        Initialize the Deutsch-Jozsa algorithm.

        Args:
            n_qubits: Number of input qubits.
            oracle_type: Type of oracle ('constant', 'balanced', or 'custom').
            constant_value: For constant oracle, the constant output value (0 or 1).
            custom_oracle: Custom oracle function for advanced use cases.

        Raises:
            ValueError: If oracle_type is unknown.

        Examples:
            >>> dj = DeutschJozsa(n_qubits=4, oracle_type='constant', constant_value=1)
            >>> dj = DeutschJozsa(n_qubits=4, oracle_type='balanced')
        """
        self.n_qubits = n_qubits
        self.oracle_type = oracle_type
        self.constant_value = constant_value
        
        if custom_oracle:
            self.oracle = custom_oracle
            self.oracle_type = 'custom'
        else:
            self.oracle = self._build_oracle()
    
    def _build_oracle(self) -> Callable:
        """
        Build the oracle for the Deutsch-Jozsa algorithm.

        Returns:
            Callable: Oracle function that modifies the circuit.

        Raises:
            ValueError: If oracle_type is unknown.
        """
        def constant_oracle(circuit: QuantumCircuit):
            """
            Constant oracle: f(x) = constant_value for all x.
            
            Args:
                circuit: QuantumCircuit to apply the oracle to.
            """
            if self.constant_value == 1:
                circuit.x(self.n_qubits)
        
        def balanced_oracle(circuit: QuantumCircuit):
            """
            Balanced oracle: f(x) = x₀ (first input bit determines output).
            
            Args:
                circuit: QuantumCircuit to apply the oracle to.
            """
            circuit.cx(0, self.n_qubits)
        
        if self.oracle_type == 'constant':
            return constant_oracle
        elif self.oracle_type == 'balanced':
            return balanced_oracle
        else:
            raise ValueError(f"Unknown oracle type: {self.oracle_type}")
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the complete Deutsch-Jozsa circuit.

        The circuit structure:
        1. Initialize ancilla qubit to |1⟩
        2. Apply Hadamard gates to all qubits (create superposition)
        3. Apply the oracle (phase kickback)
        4. Apply Hadamard gates to input qubits only
        5. Measure input qubits

        Returns:
            QuantumCircuit: The constructed circuit ready for execution.

        Examples:
            >>> dj = DeutschJozsa(n_qubits=3, oracle_type='constant')
            >>> circuit = dj.build_circuit()
            >>> print(circuit.depth)
            8
        """
        total_qubits = self.n_qubits + 1
        circuit = QuantumCircuit(total_qubits)
        
        # Initialize ancilla qubit to |1⟩
        circuit.x(self.n_qubits)
        
        # Apply Hadamard to all qubits
        for i in range(total_qubits):
            circuit.h(i)
        
        # Apply oracle
        self.oracle(circuit)
        
        # Apply Hadamard to input qubits only
        for i in range(self.n_qubits):
            circuit.h(i)
        
        return circuit
    
    def run(self, shots: int = 100) -> DeutschJozsaResult:
        """
        Run the Deutsch-Jozsa algorithm.

        Args:
            shots: Number of measurements (higher gives more accurate confidence).

        Returns:
            DeutschJozsaResult: Object containing the classification result.

        Examples:
            >>> dj = DeutschJozsa(n_qubits=3, oracle_type='balanced')
            >>> result = dj.run(shots=1000)
            >>> print(f"Confidence: {result.confidence:.2%}")
            Confidence: 100.00%
        """
        circuit = self.build_circuit()
        
        # Measure all qubits
        results = circuit.measure(shots=shots)
        
        # For Deutsch-Jozsa, we only care about the input qubits
        # We check if all measurements of input qubits are 0
        n = self.n_qubits
        
        # Count how many measurements have all input qubits = 0
        zero_count = 0
        for bitstring, count in results.items():
            # Check if first n bits are all '0'
            if bitstring[:n] == '0' * n:
                zero_count += count
        
        # If more than 50% of measurements are all zeros, function is constant
        # (In theory, it should be 100% for constant, 0% for balanced)
        is_constant = (zero_count / shots) > 0.5
        
        return DeutschJozsaResult(
            is_constant=is_constant,
            is_balanced=not is_constant,
            measurement_result=0 if is_constant else 1,
            n_qubits=self.n_qubits,
            oracle_type=self.oracle_type,
            confidence=zero_count / shots
        )


# ============================================================
# Simplified version for 1-qubit (Deutsch algorithm)
# ============================================================

class DeutschAlgorithm:
    """
    Simplified Deutsch algorithm for 1-qubit functions (n=1).

    This is the simplest quantum algorithm that demonstrates quantum
    advantage by determining if a 1-bit Boolean function is constant
    or balanced using a single query.

    References:
        D. Deutsch, "Quantum theory, the Church-Turing principle and the
        universal quantum computer," Proceedings of the Royal Society of
        London A, 400(1818):97-117, 1985.

    Examples:
        >>> # Test a constant function
        >>> da = DeutschAlgorithm(oracle_type='constant', constant_value=0)
        >>> result = da.run()
        >>> print(result['is_constant'])
        True
        
        >>> # Test a balanced function
        >>> da = DeutschAlgorithm(oracle_type='balanced')
        >>> result = da.run()
        >>> print(result['is_balanced'])
        True
    """
    
    def __init__(self, oracle_type: str = 'constant', constant_value: int = 0):
        """
        Initialize the Deutsch algorithm.

        Args:
            oracle_type: Type of oracle ('constant' or 'balanced').
            constant_value: For constant oracle, the constant output value (0 or 1).
        """
        self.oracle_type = oracle_type
        self.constant_value = constant_value
    
    def _apply_oracle(self, circuit: QuantumCircuit):
        """
        Apply the 1-qubit oracle to the circuit.

        Args:
            circuit: QuantumCircuit to apply the oracle to.
        """
        if self.oracle_type == 'constant':
            if self.constant_value == 1:
                circuit.x(1)
        else:  # balanced
            circuit.cx(0, 1)
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the Deutsch algorithm circuit.

        The circuit structure:
        1. Initialize ancilla qubit to |1⟩
        2. Apply Hadamard to both qubits
        3. Apply the oracle
        4. Apply Hadamard to the first qubit
        5. Measure the first qubit

        Returns:
            QuantumCircuit: The constructed circuit ready for execution.

        Examples:
            >>> da = DeutschAlgorithm(oracle_type='balanced')
            >>> circuit = da.build_circuit()
            >>> print(circuit.depth)
            5
        """
        circuit = QuantumCircuit(2)
        
        circuit.x(1)
        circuit.h(0)
        circuit.h(1)
        self._apply_oracle(circuit)
        circuit.h(0)
        
        return circuit
    
    def run(self, shots: int = 100) -> Dict:
        """
        Run the Deutsch algorithm.

        Args:
            shots: Number of measurements.

        Returns:
            Dict containing:
                - 'is_constant': True if function is constant
                - 'is_balanced': True if function is balanced
                - 'measurement': Measurement outcome ('0' or '1')
                - 'oracle_type': Type of oracle used
                - 'confidence': Confidence level of the result

        Examples:
            >>> da = DeutschAlgorithm(oracle_type='balanced')
            >>> result = da.run(shots=1000)
            >>> print(result['measurement'])
            1
        """
        circuit = self.build_circuit()
        results = circuit.measure(shots=shots)
        
        # Count measurements where first qubit is 0
        zero_count = 0
        for bitstring, count in results.items():
            if bitstring[0] == '0':
                zero_count += count
        
        is_constant = (zero_count / shots) > 0.5
        
        return {
            'is_constant': is_constant,
            'is_balanced': not is_constant,
            'measurement': '0' if is_constant else '1',
            'oracle_type': self.oracle_type,
            'confidence': zero_count / shots if is_constant else (shots - zero_count) / shots
        }


# ============================================================
# Convenience Functions
# ============================================================

def deutsch_jozsa_constant(n_qubits: int = 3, constant_value: int = 0, shots: int = 100) -> DeutschJozsaResult:
    """
    Convenience function to test a constant function with Deutsch-Jozsa.

    Args:
        n_qubits: Number of input qubits.
        constant_value: Constant output value (0 or 1).
        shots: Number of measurements.

    Returns:
        DeutschJozsaResult: Result of the algorithm.

    Examples:
        >>> result = deutsch_jozsa_constant(n_qubits=4, constant_value=0)
        >>> print(result)
        Function is CONSTANT (f(x) = 0 for all x)
    """
    dj = DeutschJozsa(n_qubits, oracle_type='constant', constant_value=constant_value)
    return dj.run(shots)


def deutsch_jozsa_balanced(n_qubits: int = 3, shots: int = 100) -> DeutschJozsaResult:
    """
    Convenience function to test a balanced function with Deutsch-Jozsa.

    Args:
        n_qubits: Number of input qubits.
        shots: Number of measurements.

    Returns:
        DeutschJozsaResult: Result of the algorithm.

    Examples:
        >>> result = deutsch_jozsa_balanced(n_qubits=4)
        >>> print(result)
        Function is BALANCED (exactly half 0s, half 1s)
    """
    dj = DeutschJozsa(n_qubits, oracle_type='balanced')
    return dj.run(shots)


def deutsch_algorithm_constant(constant_value: int = 0, shots: int = 100) -> Dict:
    """
    Convenience function to test a constant function with Deutsch algorithm.

    Args:
        constant_value: Constant output value (0 or 1).
        shots: Number of measurements.

    Returns:
        Dict: Result dictionary from the Deutsch algorithm.

    Examples:
        >>> result = deutsch_algorithm_constant(constant_value=1)
        >>> print(result['is_constant'])
        True
    """
    da = DeutschAlgorithm(oracle_type='constant', constant_value=constant_value)
    return da.run(shots)


def deutsch_algorithm_balanced(shots: int = 100) -> Dict:
    """
    Convenience function to test a balanced function with Deutsch algorithm.

    Args:
        shots: Number of measurements.

    Returns:
        Dict: Result dictionary from the Deutsch algorithm.

    Examples:
        >>> result = deutsch_algorithm_balanced()
        >>> print(result['is_balanced'])
        True
    """
    da = DeutschAlgorithm(oracle_type='balanced')
    return da.run(shots)


__all__ = [
    'DeutschJozsaResult',
    'DeutschJozsa',
    'DeutschAlgorithm',
    'deutsch_jozsa_constant',
    'deutsch_jozsa_balanced',
    'deutsch_algorithm_constant',
    'deutsch_algorithm_balanced',
]