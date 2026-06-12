"""
psiqit/algorithms/qft.py
Quantum Fourier Transform - Fixed

The Quantum Fourier Transform (QFT) is the quantum analogue of the classical
Discrete Fourier Transform (DFT). It is a key subroutine in many quantum
algorithms including Shor's factoring algorithm, quantum phase estimation,
and quantum eigenvalue estimation.
"""

import math
from typing import List, Optional, Union
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import hadamard, phase, swap, cnot, pauli_z


@dataclass
class QFTResult:
    """
    Result container for the Quantum Fourier Transform.

    Attributes:
        input_state: The input quantum state before QFT.
        output_state: The output quantum state after applying QFT.
        n_qubits: Number of qubits.
        circuit_depth: Depth of the QFT circuit.

    Examples:
        >>> from psiqit.quantum.state import basis
        >>> state = basis(4, 0)  # |00⟩
        >>> qft = QFT(n_qubits=2)
        >>> result = qft.simulate(state)
        >>> print(result)
        QFT(2 qubits): 1.000|00⟩ → 0.500|00⟩ + 0.500|01⟩ + ...
    """
    input_state: Ket
    output_state: Ket
    n_qubits: int
    circuit_depth: int
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing input and output states.
        """
        return f"QFT({self.n_qubits} qubits): {self.input_state} → {self.output_state}"


class QFT:
    """
    Quantum Fourier Transform implementation.

    The QFT transforms a quantum state from the computational basis to the
    Fourier basis. For an n-qubit system, it performs the transformation:

    |j⟩ → (1/√(2ⁿ)) Σ_{k=0}^{2ⁿ-1} ω^{jk} |k⟩, where ω = e^{2πi/2ⁿ}

    The QFT can be implemented efficiently using O(n²) gates, exponentially
    faster than the classical FFT which requires O(n2ⁿ) gates.

    References:
        D. Coppersmith, "An approximate Fourier transform useful in quantum
        factoring," IBM Research Report RC19642, 1994.
        A. Y. Kitaev, "Quantum measurements and the Abelian stabilizer problem,"
        arXiv:quant-ph/9511026, 1995.

    Examples:
        >>> # Forward QFT
        >>> qft = QFT(n_qubits=3)
        >>> circuit = qft.build_circuit()
        >>> 
        >>> # Inverse QFT
        >>> iqft = QFT(n_qubits=3, inverse=True)
        >>> circuit = iqft.build_circuit()
        >>> 
        >>> # Apply QFT to a state
        >>> from psiqit.quantum.state import basis
        >>> state = basis(8, 3)  # |011⟩
        >>> qft = QFT(n_qubits=3)
        >>> transformed = qft.apply(state)
    """
    
    def __init__(self, n_qubits: int, inverse: bool = False):
        """
        Initialize the Quantum Fourier Transform.

        Args:
            n_qubits: Number of qubits.
            inverse: If True, construct the inverse QFT (IQFT).

        Examples:
            >>> qft = QFT(n_qubits=4)  # Forward QFT
            >>> iqft = QFT(n_qubits=4, inverse=True)  # Inverse QFT
        """
        self.n_qubits = n_qubits
        self.N = 2 ** n_qubits
        self.inverse = inverse
    
    def _phase_angle(self, k: int) -> float:
        """
        Calculate the phase angle for controlled phase gates.

        Args:
            k: Distance between qubits (j - i).

        Returns:
            Phase angle in radians.

        Examples:
            >>> qft = QFT(n_qubits=3)
            >>> qft._phase_angle(1)  # π/2 for adjacent qubits
            1.5707963267948966
        """
        if self.inverse:
            return -2 * math.pi / (2 ** (k + 1))
        else:
            return 2 * math.pi / (2 ** (k + 1))
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the QFT quantum circuit.

        The circuit structure for forward QFT:
        1. Apply Hadamard to qubit i
        2. Apply controlled-phase gates with qubit i as control and j>i as targets
        3. Swap qubits at the end to reverse order

        For inverse QFT, the operations are applied in reverse order.

        Returns:
            QuantumCircuit: The QFT circuit ready for execution.

        Examples:
            >>> qft = QFT(n_qubits=4)
            >>> circuit = qft.build_circuit()
            >>> print(f"Circuit depth: {circuit.depth}")
            Circuit depth: 10
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        if self.inverse:
            self._build_inverse_qft(circuit)
        else:
            self._build_forward_qft(circuit)
        
        return circuit
    
    def _build_forward_qft(self, circuit: QuantumCircuit):
        """
        Build the forward QFT circuit.

        Args:
            circuit: QuantumCircuit to build upon.
        """
        n = self.n_qubits
        
        for i in range(n):
            circuit.h(i)
            for j in range(i + 1, n):
                angle = self._phase_angle(j - i)
                self._controlled_phase(circuit, j, i, angle)
        
        # Swap qubits to correct the order
        for i in range(n // 2):
            circuit.swap(i, n - 1 - i)
    
    def _build_inverse_qft(self, circuit: QuantumCircuit):
        """
        Build the inverse QFT circuit.

        Args:
            circuit: QuantumCircuit to build upon.
        """
        n = self.n_qubits
        
        # Swap qubits first
        for i in range(n // 2):
            circuit.swap(i, n - 1 - i)
        
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                angle = self._phase_angle(j - i)
                self._controlled_phase(circuit, j, i, -angle)
            circuit.h(i)
    
    def _controlled_phase(self, circuit: QuantumCircuit, control: int, target: int, angle: float):
        """
        Apply a controlled-phase gate using CNOT and Rz.

        This implements the controlled-phase gate: CP(θ) = |0⟩⟨0|⊗I + |1⟩⟨1|⊗Rz(θ)
        using the decomposition: CNOT * (I ⊗ Rz(θ)) * CNOT

        Args:
            circuit: QuantumCircuit to add the gate to.
            control: Control qubit index.
            target: Target qubit index.
            angle: Phase angle in radians.
        """
        circuit.cx(control, target)
        circuit.rz(target, angle)
        circuit.cx(control, target)
    
    def apply(self, state: Ket) -> Ket:
        """
        Apply QFT to a given state using matrix multiplication.

        This method computes the QFT by directly multiplying the state vector
        by the QFT matrix. This is useful for small systems or for verification.

        Args:
            state: Input quantum state (must have dimension 2ⁿ).

        Returns:
            Ket: Transformed state after applying QFT.

        Raises:
            ValueError: If state dimension doesn't match.

        Examples:
            >>> from psiqit.quantum.state import basis
            >>> state = basis(4, 0)  # |00⟩
            >>> qft = QFT(n_qubits=2)
            >>> transformed = qft.apply(state)
            >>> print(transformed)
            0.500|00⟩ + 0.500|01⟩ + 0.500|10⟩ + 0.500|11⟩
        """
        if state.dim != self.N:
            raise ValueError(f"State dimension {state.dim} != {self.N}")
        
        # Get matrix representation
        matrix = self.get_matrix()
        
        # Apply matrix to state
        new_data = []
        for i in range(self.N):
            val = sum(matrix[i][j] * state.data[j] for j in range(self.N))
            new_data.append(val)
        
        return Ket(new_data)
    
    def get_matrix(self) -> List[List[complex]]:
        """
        Get the QFT matrix representation.

        The QFT matrix elements are: (U_{QFT})_{j,k} = ω^{jk} / √N,
        where ω = e^{2πi/N} and N = 2ⁿ.

        Returns:
            N×N matrix representing the QFT.

        Examples:
            >>> qft = QFT(n_qubits=2)
            >>> matrix = qft.get_matrix()
            >>> print(len(matrix))  # 4
            4
        """
        N = self.N
        omega = math.e ** (2j * math.pi / N)
        
        matrix = []
        for i in range(N):
            row = []
            for j in range(N):
                if self.inverse:
                    val = (omega ** (-i * j)) / math.sqrt(N)
                else:
                    val = (omega ** (i * j)) / math.sqrt(N)
                row.append(val)
            matrix.append(row)
        
        return matrix
    
    def simulate(self, input_state: Ket) -> QFTResult:
        """
        Simulate QFT on a given input state.

        This method applies QFT to the input state and returns a result object
        containing both input and output states.

        Args:
            input_state: Input quantum state.

        Returns:
            QFTResult: Object containing input state, output state, and metadata.

        Examples:
            >>> from psiqit.quantum.state import basis
            >>> state = basis(4, 1)  # |01⟩
            >>> qft = QFT(n_qubits=2)
            >>> result = qft.simulate(state)
            >>> print(result.n_qubits)
            2
        """
        output_state = self.apply(input_state)
        
        return QFTResult(
            input_state=input_state,
            output_state=output_state,
            n_qubits=self.n_qubits,
            circuit_depth=self._get_circuit_depth()
        )
    
    def _get_circuit_depth(self) -> int:
        """
        Calculate the theoretical circuit depth.

        Returns:
            Number of gates in the QFT circuit.
        """
        n = self.n_qubits
        phase_gates = n * (n - 1) // 2
        swaps = n // 2
        return n + phase_gates + swaps
    
    def __repr__(self) -> str:
        """Return a string representation of the QFT object."""
        return f"QFT(n_qubits={self.n_qubits}, inverse={self.inverse})"


class IQFT(QFT):
    """
    Inverse Quantum Fourier Transform.

    This class is a convenience wrapper around QFT with inverse=True.
    It computes the inverse of the QFT, which is used in algorithms like
    quantum phase estimation to convert phase information back to the
    computational basis.

    Examples:
        >>> iqft = IQFT(n_qubits=3)
        >>> circuit = iqft.build_circuit()
        >>> state = basis(8, 0)
        >>> result = iqft.simulate(state)
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize the Inverse Quantum Fourier Transform.

        Args:
            n_qubits: Number of qubits.

        Examples:
            >>> iqft = IQFT(n_qubits=4)
        """
        super().__init__(n_qubits, inverse=True)


# ============================================================
# Convenience Functions
# ============================================================

def qft_circuit(n_qubits: int, inverse: bool = False) -> QuantumCircuit:
    """
    Create a QFT circuit.

    Args:
        n_qubits: Number of qubits.
        inverse: If True, create inverse QFT circuit.

    Returns:
        QuantumCircuit: The QFT circuit.

    Examples:
        >>> circuit = qft_circuit(n_qubits=3)
        >>> circuit = qft_circuit(n_qubits=3, inverse=True)
    """
    qft = QFT(n_qubits, inverse=inverse)
    return qft.build_circuit()


def apply_qft(state: Ket, inverse: bool = False) -> Ket:
    """
    Apply QFT to a quantum state.

    Args:
        state: Input quantum state.
        inverse: If True, apply inverse QFT.

    Returns:
        Ket: Transformed state.

    Examples:
        >>> from psiqit.quantum.state import basis
        >>> state = basis(8, 3)
        >>> transformed = apply_qft(state)
        >>> original = apply_qft(transformed, inverse=True)
    """
    n_qubits = state.dim.bit_length() - 1
    qft = QFT(n_qubits, inverse=inverse)
    return qft.apply(state)


def qft_matrix(n_qubits: int, inverse: bool = False) -> List[List[complex]]:
    """
    Get the QFT matrix representation.

    Args:
        n_qubits: Number of qubits.
        inverse: If True, get inverse QFT matrix.

    Returns:
        List[List[complex]]: N×N QFT matrix where N = 2ⁿ.

    Examples:
        >>> matrix = qft_matrix(n_qubits=2)
        >>> print(len(matrix))
        4
    """
    qft = QFT(n_qubits, inverse=inverse)
    return qft.get_matrix()


__all__ = [
    'QFTResult',
    'QFT',
    'IQFT',
    'qft_circuit',
    'apply_qft',
    'qft_matrix',
]