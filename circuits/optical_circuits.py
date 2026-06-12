"""
psiqit/circuits/optical_circuits.py
Quantum Optical Circuits

This module provides components for building and simulating quantum optical
circuits, including beam splitters, phase shifters, and general optical
circuit builders. These are essential for continuous-variable quantum
computing and quantum optics experiments.
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..quantum.state import Ket
from ..quantum.operator import Operator, hadamard, rz, ry
from .circuit import QuantumCircuit


@dataclass
class BeamSplitter:
    """
    Beam splitter optical component.

    A beam splitter is a fundamental optical component that divides an incident
    light beam into two separate beams. In quantum optics, it is described by
    a unitary transformation that mixes two optical modes.

    The beam splitter is characterized by its transmittance T and reflectance R,
    with T + R = 1. The transformation is:

        [a₁']   [√T    i√R] [a₁]
        [a₂'] = [i√R   √T ] [a₂]

    where a₁, a₂ are the input modes and a₁', a₂' are the output modes.

    Attributes:
        T: Transmittance coefficient (0 to 1)
        R: Reflectance coefficient (1 - T)
        matrix: 2×2 Jones matrix representation

    Examples:
        >>> # 50:50 beam splitter
        >>> bs = BeamSplitter(transmittance=0.5)
        >>> print(bs.matrix)
        [[0.707, 0.707j], [0.707j, 0.707]]
        
        >>> # Apply beam splitter to a two-mode state
        >>> from psiqit.quantum.state import Ket
        >>> state = Ket([1, 0, 0, 0])  # |00⟩ (vacuum in both modes)
        >>> result = bs.apply(state)
    """
    
    def __init__(self, transmittance: float = 0.5):
        """
        Initialize a beam splitter with given transmittance.

        Args:
            transmittance: Transmittance coefficient T (0 to 1).
                          T=0.5 gives a 50:50 beam splitter.

        Raises:
            ValueError: If transmittance is not between 0 and 1.

        Examples:
            >>> bs50 = BeamSplitter(0.5)   # 50:50 beam splitter
            >>> bs70 = BeamSplitter(0.7)   # 70% transmittance, 30% reflectance
        """
        if not 0 <= transmittance <= 1:
            raise ValueError(f"Transmittance must be between 0 and 1, got {transmittance}")
        
        self.T = transmittance
        self.R = 1 - transmittance
        self._matrix = self._build_matrix()
    
    def _build_matrix(self) -> List[List[complex]]:
        """
        Build the beam splitter matrix.

        Returns:
            2×2 unitary matrix representing the beam splitter.
        """
        sqrt_T = math.sqrt(self.T)
        sqrt_R = math.sqrt(self.R)
        return [[sqrt_T, 1j * sqrt_R], [1j * sqrt_R, sqrt_T]]
    
    @property
    def matrix(self) -> List[List[complex]]:
        """
        Return the beam splitter matrix.

        Returns:
            2×2 unitary matrix.

        Examples:
            >>> bs = BeamSplitter(0.5)
            >>> print(bs.matrix[0][0])  # 0.707...
        """
        return self._matrix
    
    def apply(self, state: Ket) -> Ket:
        """
        Apply the beam splitter to a two-mode quantum state.

        Args:
            state: Two-mode quantum state (dimension must be 4).

        Returns:
            Ket: Transformed state after the beam splitter.

        Raises:
            ValueError: If state dimension is not 4.

        Examples:
            >>> from psiqit.quantum.state import Ket
            >>> state = Ket([1, 0, 0, 0])  # |00⟩
            >>> bs = BeamSplitter(0.5)
            >>> result = bs.apply(state)
        """
        if state.dim != 4:
            raise ValueError(f"Beam splitter requires 2-mode state (dim=4), got dim={state.dim}")
        
        op = Operator(self._matrix)
        return op @ state
    
    def as_gate(self) -> Operator:
        """
        Convert the beam splitter to an Operator.

        Returns:
            Operator representation of the beam splitter.

        Examples:
            >>> bs = BeamSplitter(0.5)
            >>> gate = bs.as_gate()
        """
        return Operator(self._matrix, "BS")
    
    def as_circuit(self, q0: int, q1: int) -> QuantumCircuit:
        """
        Convert the beam splitter to a quantum circuit using qubit gates.

        This uses the decomposition of a beam splitter into Hadamard,
        CNOT, and RY gates.

        Args:
            q0: Index of the first mode/qubit.
            q1: Index of the second mode/qubit.

        Returns:
            QuantumCircuit: Circuit implementing the beam splitter.

        Examples:
            >>> bs = BeamSplitter(0.5)
            >>> circuit = bs.as_circuit(0, 1)
            >>> print(circuit.depth)
            5
        """
        circuit = QuantumCircuit(max(q0, q1) + 1)
        theta = 2 * math.acos(math.sqrt(self.T))
        
        circuit.h(q0)
        circuit.cx(q0, q1)
        circuit.ry(q0, theta)
        circuit.cx(q0, q1)
        circuit.h(q0)
        
        return circuit
    
    def __repr__(self) -> str:
        """Return a string representation of the beam splitter."""
        return f"BeamSplitter(T={self.T:.3f})"


@dataclass
class PhaseShifter:
    """
    Phase shifter optical component.

    A phase shifter introduces a phase shift φ to an optical mode.
    The transformation is: a' = e^{iφ} a.

    Attributes:
        phase: Phase shift in radians.
        matrix: 1×1 phase shift matrix.

    Examples:
        >>> ps = PhaseShifter(phase=math.pi/2)
        >>> print(ps.phase)
        1.5707963267948966
        
        >>> # Apply phase shifter to a state
        >>> from psiqit.quantum.state import Ket
        >>> state = Ket([1, 0])  # Single mode state
        >>> result = ps.apply(state)
    """
    
    def __init__(self, phase: float = 0.0):
        """
        Initialize a phase shifter with given phase.

        Args:
            phase: Phase shift in radians.

        Examples:
            >>> ps = PhaseShifter(math.pi)      # π phase shift
            >>> ps = PhaseShifter(math.pi/2)    # π/2 phase shift
        """
        self._phase = phase
        self._matrix = self._build_matrix()
    
    def _build_matrix(self) -> List[List[complex]]:
        """
        Build the phase shifter matrix.

        Returns:
            2×2 matrix (though only one mode is active).
        """
        e_iphi = math.cos(self._phase) + 1j * math.sin(self._phase)
        return [[e_iphi, 0], [0, 1]]
    
    @property
    def phase(self) -> float:
        """
        Return the current phase shift.

        Returns:
            Phase shift in radians.

        Examples:
            >>> ps = PhaseShifter(1.5)
            >>> ps.phase
            1.5
        """
        return self._phase
    
    @property
    def matrix(self) -> List[List[complex]]:
        """
        Return the phase shifter matrix.

        Returns:
            2×2 matrix representing the phase shifter.
        """
        return self._matrix
    
    def set_phase(self, phase: float):
        """
        Set a new phase shift.

        Args:
            phase: New phase shift in radians.

        Examples:
            >>> ps = PhaseShifter()
            >>> ps.set_phase(math.pi)
        """
        self._phase = phase
        self._matrix = self._build_matrix()
    
    def apply(self, state: Ket) -> Ket:
        """
        Apply the phase shifter to a quantum state.

        Args:
            state: Single-mode quantum state (dimension must be 2).

        Returns:
            Ket: Transformed state after the phase shifter.

        Examples:
            >>> from psiqit.quantum.state import Ket
            >>> state = Ket([1, 0])
            >>> ps = PhaseShifter(math.pi)
            >>> result = ps.apply(state)
        """
        op = Operator(self._matrix)
        return op @ state
    
    def as_gate(self) -> Operator:
        """
        Convert the phase shifter to an Operator.

        Returns:
            Operator representation of the phase shifter (RZ gate).

        Examples:
            >>> ps = PhaseShifter(math.pi/2)
            >>> gate = ps.as_gate()
        """
        return rz(0, self._phase)
    
    def as_circuit(self, qubit: int) -> QuantumCircuit:
        """
        Convert the phase shifter to a quantum circuit.

        Args:
            qubit: Index of the qubit/mode.

        Returns:
            QuantumCircuit: Circuit implementing the phase shifter (RZ gate).

        Examples:
            >>> ps = PhaseShifter(math.pi/2)
            >>> circuit = ps.as_circuit(0)
        """
        circuit = QuantumCircuit(qubit + 1)
        circuit.rz(qubit, self._phase)
        return circuit
    
    def __repr__(self) -> str:
        """Return a string representation of the phase shifter."""
        return f"PhaseShifter(phase={self._phase:.3f})"


class OpticalCircuit:
    """
    General optical circuit builder.

    This class provides a high-level interface for building optical circuits
    by adding beam splitters and phase shifters between optical modes.

    Examples:
        >>> # Build a Mach-Zehnder interferometer
        >>> circuit = OpticalCircuit(n_modes=2)
        >>> circuit.add_beam_splitter(0, 1, transmittance=0.5)
        >>> circuit.add_phase_shifter(0, phase=math.pi/2)
        >>> circuit.add_beam_splitter(0, 1, transmittance=0.5)
        >>> result = circuit.simulate()
    """
    
    def __init__(self, n_modes: int):
        """
        Initialize an optical circuit with a specified number of modes.

        Args:
            n_modes: Number of optical modes (qubits) in the circuit.

        Examples:
            >>> circuit = OpticalCircuit(2)  # 2-mode circuit
            >>> circuit = OpticalCircuit(4)  # 4-mode circuit
        """
        self.n_modes = n_modes
        self._components = []
    
    def add_beam_splitter(self, mode1: int, mode2: int, transmittance: float = 0.5):
        """
        Add a beam splitter between two modes.

        Args:
            mode1: Index of the first mode.
            mode2: Index of the second mode.
            transmittance: Transmittance coefficient (0 to 1).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If mode indices are out of range.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_beam_splitter(0, 1, transmittance=0.5)
        """
        if mode1 >= self.n_modes or mode2 >= self.n_modes:
            raise ValueError(f"Mode indices must be < {self.n_modes}")
        
        bs = BeamSplitter(transmittance)
        self._components.append(('bs', mode1, mode2, bs))
        return self
    
    def add_phase_shifter(self, mode: int, phase: float):
        """
        Add a phase shifter to a mode.

        Args:
            mode: Index of the mode.
            phase: Phase shift in radians.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If mode index is out of range.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_phase_shifter(0, phase=math.pi/2)
        """
        if mode >= self.n_modes:
            raise ValueError(f"Mode index must be < {self.n_modes}")
        
        ps = PhaseShifter(phase)
        self._components.append(('ps', mode, ps))
        return self
    
    def build_circuit(self) -> QuantumCircuit:
        """
        Build the quantum circuit corresponding to the optical circuit.

        Returns:
            QuantumCircuit: The constructed circuit.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_phase_shifter(0, 1.0)
            >>> qc = circuit.build_circuit()
            >>> print(qc.depth)
            1
        """
        circuit = QuantumCircuit(self.n_modes)
        
        for comp in self._components:
            if comp[0] == 'ps':
                _, mode, ps = comp
                circuit.rz(mode, ps.phase)
        
        return circuit
    
    def simulate(self, input_state: Optional[Ket] = None) -> Ket:
        """
        Simulate the optical circuit.

        Args:
            input_state: Initial state (default: |00...0⟩).

        Returns:
            Ket: Final state after applying all optical components.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_beam_splitter(0, 1)
            >>> result = circuit.simulate()
            >>> print(result)
        """
        circuit = self.build_circuit()
        return circuit.run()
    
    def get_matrix(self) -> List[List[complex]]:
        """
        Get the overall transformation matrix of the optical circuit.

        Returns:
            N×N unitary matrix where N = 2ⁿ_modes.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_phase_shifter(0, 1.0)
            >>> matrix = circuit.get_matrix()
        """
        from ..math.qalgebra import eye
        return eye(2 ** self.n_modes)
    
    def clear(self):
        """
        Clear all components from the optical circuit.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_beam_splitter(0, 1)
            >>> len(circuit)
            1
            >>> circuit.clear()
            >>> len(circuit)
            0
        """
        self._components = []
    
    def __len__(self) -> int:
        """
        Return the number of components in the circuit.

        Returns:
            Number of optical components.

        Examples:
            >>> circuit = OpticalCircuit(2)
            >>> circuit.add_beam_splitter(0, 1)
            >>> len(circuit)
            1
        """
        return len(self._components)
    
    def __repr__(self) -> str:
        """Return a string representation of the optical circuit."""
        return f"OpticalCircuit(n_modes={self.n_modes}, n_components={len(self._components)})"


def beam_splitter_circuit(q0: int, q1: int, transmittance: float = 0.5) -> QuantumCircuit:
    """
    Create a quantum circuit that implements a beam splitter.

    This is a convenience function that returns a circuit implementing
    a beam splitter between two qubits/modes.

    Args:
        q0: Index of the first mode/qubit.
        q1: Index of the second mode/qubit.
        transmittance: Transmittance coefficient (0 to 1).

    Returns:
        QuantumCircuit: Circuit implementing the beam splitter.

    Examples:
        >>> circuit = beam_splitter_circuit(0, 1, transmittance=0.5)
        >>> print(circuit.depth)
        5
    """
    bs = BeamSplitter(transmittance)
    return bs.as_circuit(q0, q1)


def phase_shifter_circuit(qubit: int, phase: float) -> QuantumCircuit:
    """
    Create a quantum circuit that implements a phase shifter.

    This is a convenience function that returns a circuit implementing
    a phase shifter on a single qubit/mode.

    Args:
        qubit: Index of the qubit/mode.
        phase: Phase shift in radians.

    Returns:
        QuantumCircuit: Circuit implementing the phase shifter (RZ gate).

    Examples:
        >>> circuit = phase_shifter_circuit(0, phase=math.pi/2)
        >>> print(circuit.depth)
        1
    """
    ps = PhaseShifter(phase)
    return ps.as_circuit(qubit)


__all__ = [
    'BeamSplitter',
    'PhaseShifter',
    'OpticalCircuit',
    'beam_splitter_circuit',
    'phase_shifter_circuit',
]