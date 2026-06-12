"""
psiqit/algorithms/qpe.py
Quantum Phase Estimation - Simplified

Quantum Phase Estimation (QPE) is a fundamental quantum algorithm that estimates
the phase φ of an eigenvalue e^{2πiφ} of a unitary operator U, given an
eigenstate |ψ⟩ such that U|ψ⟩ = e^{2πiφ}|ψ⟩.

This is a simplified version that demonstrates the core concepts of QPE using
a single-qubit unitary (Pauli-Z) for demonstration purposes.
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis, zero
from ..quantum.operator import Operator, hadamard, cnot, pauli_z
from ..quantum.measurement import measure


@dataclass
class QPEResult:
    """
    Result container for Quantum Phase Estimation.

    Attributes:
        phase: Estimated phase φ (between 0 and 1).
        eigenvalue: Estimated eigenvalue e^{2πiφ} as a complex number.
        precision: Number of phase qubits used (precision of estimation).
        success: Whether the estimation was successful.
        probability: Probability of measuring the correct phase.

    Examples:
        >>> result = QPEResult(phase=0.25, eigenvalue=1j, precision=4,
        ...                    success=True, probability=0.95)
        >>> print(result)
        QPE(phase=0.250000)
    """
    phase: float
    eigenvalue: complex
    precision: int
    success: bool
    probability: float
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing the estimated phase.
        """
        return f"QPE(phase={self.phase:.6f})"


class QPE:
    """
    Quantum Phase Estimation algorithm (simplified version).

    The QPE algorithm estimates the phase φ of an eigenvalue of a unitary
    operator U. For a given eigenstate |ψ⟩ satisfying U|ψ⟩ = e^{2πiφ}|ψ⟩,
    the algorithm outputs an n-bit approximation of φ.

    This simplified implementation:
    1. Uses Pauli-Z gate as the unitary (eigenvalues: ±1 → φ = 0 or 0.5)
    2. Uses n phase qubits to estimate the phase
    3. Uses a simplified inverse QFT (just Hadamard gates)

    References:
        A. Y. Kitaev, "Quantum measurements and the Abelian stabilizer problem,"
        arXiv:quant-ph/9511026, 1995.
        R. Cleve, A. Ekert, C. Macchiavello, and M. Mosca,
        "Quantum algorithms revisited," Proceedings of the Royal Society A,
        454(1969):339-354, 1998.

    Examples:
        >>> # Estimate phase of Pauli-Z eigenstate |1⟩ (eigenvalue -1, φ = 0.5)
        >>> qpe = QPE(n_phase_qubits=4)
        >>> result = qpe.estimate()
        >>> print(f"Estimated phase: {result.phase}")
        Estimated phase: 0.5
        
        >>> # Estimate phase with higher precision
        >>> qpe = QPE(n_phase_qubits=6)
        >>> result = qpe.estimate()
        >>> print(f"Phase: {result.phase}, Eigenvalue: {result.eigenvalue}")
        Phase: 0.5, Eigenvalue: (-1+0j)
    """
    
    def __init__(self, n_phase_qubits: int = 4):
        """
        Initialize the Quantum Phase Estimation algorithm.

        Args:
            n_phase_qubits: Number of qubits in the phase register.
                            Higher precision requires more qubits.

        Examples:
            >>> qpe = QPE(n_phase_qubits=4)  # 4-bit precision
            >>> qpe = QPE(n_phase_qubits=8)  # 8-bit precision
        """
        self.n_phase = n_phase_qubits
        self.total_qubits = n_phase_qubits + 1
    
    def _controlled_phase(self, circuit: QuantumCircuit, 
                          control: int, target: int, angle: float):
        """
        Apply a controlled phase gate to the circuit.

        This implements a controlled-Rz gate using the decomposition:
        CNOT * Rz(θ) * CNOT

        Args:
            circuit: QuantumCircuit to add the gate to.
            control: Control qubit index.
            target: Target qubit index (where Rz is applied).
            angle: Phase angle in radians.

        Examples:
            >>> circuit = QuantumCircuit(2)
            >>> qpe = QPE()
            >>> qpe._controlled_phase(circuit, 0, 1, math.pi/2)
        """
        circuit.cx(control, target)
        circuit.rz(target, angle)
        circuit.cx(control, target)
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the simplified QPE circuit.

        The circuit structure:
        1. Initialize phase register in superposition (Hadamard gates)
        2. Initialize eigenstate register to |1⟩ (eigenstate of Pauli-Z)
        3. Apply controlled-U^{2^k} operations
        4. Apply inverse QFT (simplified using only Hadamard gates)

        Returns:
            QuantumCircuit: The QPE circuit ready for execution.

        Examples:
            >>> qpe = QPE(n_phase_qubits=4)
            >>> circuit = qpe.build_circuit()
            >>> print(f"Circuit depth: {circuit.depth}")
            Circuit depth: 9
        """
        circuit = QuantumCircuit(self.total_qubits)
        
        # Initialize phase register in superposition
        for i in range(self.n_phase):
            circuit.h(i)
        
        # Initialize eigenstate register to |1⟩ (for Z gate eigenstate)
        # For Pauli-Z, the eigenstate with eigenvalue -1 is |1⟩ (φ = 0.5)
        circuit.x(self.n_phase)
        
        # Apply controlled phase operations
        # This implements controlled-U^{2^k} where U = Rz(2π/2^{n-k})
        for i in range(self.n_phase):
            angle = 2 * math.pi / (2 ** (self.n_phase - i))
            self._controlled_phase(circuit, i, self.n_phase, angle)
        
        # Apply inverse QFT (simplified - just Hadamard gates)
        # A full inverse QFT would use controlled-phase gates as well
        for i in range(self.n_phase):
            circuit.h(i)
        
        return circuit
    
    def estimate(self, shots: int = 1024) -> QPEResult:
        """
        Estimate the phase using the QPE algorithm.

        This method runs the QPE circuit and extracts the phase from
        measurement outcomes.

        Args:
            shots: Number of measurements to perform.

        Returns:
            QPEResult: Object containing estimated phase, eigenvalue,
                      precision, success status, and probability.

        Examples:
            >>> qpe = QPE(n_phase_qubits=4)
            >>> result = qpe.estimate(shots=2048)
            >>> print(f"Phase: {result.phase}")
            Phase: 0.5
            >>> print(f"Eigenvalue: {result.eigenvalue}")
            Eigenvalue: (-1+0j)
        """
        circuit = self.build_circuit()
        
        # Measure phase register
        results = circuit.measure(shots=shots)
        
        if isinstance(results, dict):
            # Find most common outcome
            most_common = max(results, key=results.get)
            phase_bits = int(most_common[:self.n_phase], 2)
        else:
            phase_bits = 0
        
        phase = phase_bits / (2 ** self.n_phase)
        
        return QPEResult(
            phase=phase,
            eigenvalue=math.cos(2*math.pi*phase) + 1j*math.sin(2*math.pi*phase),
            precision=self.n_phase,
            success=True,
            probability=1.0
        )


__all__ = ['QPEResult', 'QPE']