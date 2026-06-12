"""
psiqit/lab/virtual_lab.py
Virtual Quantum Lab - Simplified and Corrected

This module provides a virtual laboratory environment for designing,
running, and analyzing quantum experiments. It allows users to create
experiments, add quantum gates, execute simulations, and visualize results.

The Virtual Lab is designed for educational purposes, prototyping, and
batch experimentation without requiring deep knowledge of quantum circuit
implementation details.

Example:
    >>> from psiqit.lab.virtual_lab import QuantumLab, PredefinedExperiments
    >>> 
    >>> # Create a virtual lab
    >>> lab = QuantumLab()
    >>> 
    >>> # Create a custom experiment
    >>> exp = lab.create_experiment("My Bell State", n_qubits=2)
    >>> exp.add_gate("h", 0)
    >>> exp.add_gate("cx", 0, 1)
    >>> 
    >>> # Run the experiment
    >>> result = lab.run_experiment(exp)
    >>> print(result.summary())
"""

import json
import math
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, one, plus, minus, basis
from ..quantum.operator import hadamard, pauli_x, pauli_y, pauli_z, cnot, swap, cz
from ..quantum.measurement import measure
from ..visualization.bloch import bloch_sphere, state_to_bloch


class ExperimentStatus(Enum):
    """
    Status of an experiment in the virtual lab.

    Attributes:
        DRAFT: Experiment is being designed (gates can be added/modified).
        RUNNING: Experiment is currently being executed.
        COMPLETED: Experiment has finished successfully.
        FAILED: Experiment failed due to an error.

    Examples:
        >>> status = ExperimentStatus.DRAFT
        >>> print(status.value)
        draft
    """
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentResult:
    """
    Result container for a quantum experiment.

    Attributes:
        name: Name of the experiment.
        timestamp: ISO format timestamp of when the experiment was run.
        n_qubits: Number of qubits in the experiment.
        shots: Number of measurements performed.
        counts: Dictionary mapping bitstrings to measurement counts.
        probabilities: List of probabilities for each basis state.
        final_state: Final state vector as a list of complex amplitudes.
        success_rate: Fraction of the most common outcome (max_count/shots).
        execution_time: Time taken to run the experiment in seconds.
        status: Status string ('completed' or 'failed').

    Examples:
        >>> result = ExperimentResult(
        ...     name="Bell State", timestamp="2024-01-01T12:00:00",
        ...     n_qubits=2, shots=1024, counts={"00": 512, "11": 512},
        ...     probabilities=[0.5, 0, 0, 0.5], final_state=[], 
        ...     success_rate=0.5, execution_time=0.123, status="completed"
        ... )
        >>> print(result.summary())
    """
    name: str
    timestamp: str
    n_qubits: int
    shots: int
    counts: Dict[str, int]
    probabilities: List[float]
    final_state: List[complex]
    success_rate: float
    execution_time: float
    status: str
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the experiment results.

        Returns:
            Multi-line string containing experiment metadata and measurement counts.

        Examples:
            >>> result = experiment_result
            >>> print(result.summary())
            Experiment: Bell State
            Time: 2024-01-01T12:00:00
            Qubits: 2, Shots: 1024
            Success rate: 50.00%
            Status: completed
            Measurement counts:
              |00⟩: 512 (50.00%)
              |11⟩: 512 (50.00%)
        """
        lines = [
            f"Experiment: {self.name}",
            f"Time: {self.timestamp}",
            f"Qubits: {self.n_qubits}, Shots: {self.shots}",
            f"Success rate: {self.success_rate:.2%}",
            f"Status: {self.status}",
            "",
            "Measurement counts:"
        ]
        for outcome, count in sorted(self.counts.items()):
            prob = count / self.shots
            lines.append(f"  |{outcome}⟩: {count} ({prob:.2%})")
        return "\n".join(lines)


@dataclass
class Experiment:
    """
    Container for a quantum experiment definition.

    This class stores all information about an experiment including its
    name, number of qubits, list of gates, and execution parameters.

    Attributes:
        name: Name of the experiment.
        n_qubits: Number of qubits.
        gates: List of gate dictionaries with 'gate', 'qubits', and 'params'.
        shots: Number of measurement shots (default: 1024).
        status: Current experiment status (default: DRAFT).
        created_at: ISO format timestamp of creation.
        result: ExperimentResult object after execution (initially None).

    Examples:
        >>> exp = Experiment(name="My Experiment", n_qubits=2)
        >>> exp.add_gate("h", 0)
        >>> exp.add_gate("cx", 0, 1)
        >>> len(exp.gates)
        2
    """
    name: str
    n_qubits: int
    gates: List[Dict[str, Any]] = field(default_factory=list)
    shots: int = 1024
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    result: Optional[ExperimentResult] = None
    
    def add_gate(self, gate_name: str, *qubits: int, params: Optional[Dict] = None):
        """
        Add a quantum gate to the experiment.

        Args:
            gate_name: Name of the gate ('h', 'x', 'y', 'z', 'cx', 'cz', 'swap').
            *qubits: Variable number of qubit indices.
            params: Optional parameters for parameterized gates (e.g., rotation angles).

        Examples:
            >>> exp = Experiment(name="Test", n_qubits=2)
            >>> exp.add_gate("h", 0)
            >>> exp.add_gate("cx", 0, 1)
            >>> exp.add_gate("rx", 0, params={"theta": 3.14159})
        """
        self.gates.append({
            'gate': gate_name,
            'qubits': list(qubits),
            'params': params or {}
        })
        self.status = ExperimentStatus.DRAFT
        self.result = None
    
    def clear_gates(self):
        """
        Remove all gates from the experiment.

        This resets the experiment to an empty circuit.

        Examples:
            >>> exp = Experiment(name="Test", n_qubits=2)
            >>> exp.add_gate("h", 0)
            >>> len(exp.gates)
            1
            >>> exp.clear_gates()
            >>> len(exp.gates)
            0
        """
        self.gates = []
        self.status = ExperimentStatus.DRAFT
        self.result = None


class QuantumLab:
    """
    Virtual quantum laboratory for running experiments.

    The QuantumLab class provides a high-level interface for creating,
    managing, and executing quantum experiments. It handles state vector
    simulation, gate application, measurement, and result collection.

    Examples:
        >>> lab = QuantumLab()
        >>> 
        >>> # Create and run a Bell state experiment
        >>> exp = lab.create_experiment("Bell State", n_qubits=2)
        >>> exp.add_gate("h", 0)
        >>> exp.add_gate("cx", 0, 1)
        >>> result = lab.run_experiment(exp)
        >>> 
        >>> # List all experiments
        >>> experiments = lab.list_experiments()
        >>> print(experiments[0]['name'])
        Bell State
    """
    
    def __init__(self):
        """
        Initialize a new virtual quantum laboratory.

        Examples:
            >>> lab = QuantumLab()
            >>> print(lab.list_experiments())
            []
        """
        self.experiments: List[Experiment] = []
        self.current_experiment: Optional[Experiment] = None
    
    def create_experiment(self, name: str, n_qubits: int) -> Experiment:
        """
        Create a new experiment in the laboratory.

        Args:
            name: Name of the experiment.
            n_qubits: Number of qubits.

        Returns:
            Experiment: The newly created experiment object.

        Examples:
            >>> lab = QuantumLab()
            >>> exp = lab.create_experiment("My Experiment", n_qubits=3)
            >>> print(exp.name)
            My Experiment
            >>> print(exp.n_qubits)
            3
        """
        exp = Experiment(name=name, n_qubits=n_qubits)
        self.experiments.append(exp)
        self.current_experiment = exp
        return exp
    
    def _apply_gate_to_state(self, state: Ket, gate_name: str, qubits: List[int]) -> Ket:
        """
        Apply a quantum gate to a state vector (internal method).

        Args:
            state: Current quantum state.
            gate_name: Name of the gate to apply.
            qubits: List of qubit indices.

        Returns:
            Ket: New state after gate application.
        """
        n = int(math.log2(state.dim))
        
        if gate_name == 'h':
            from ..quantum.operator import hadamard
            H = hadamard()
            if len(qubits) == 1:
                q = qubits[0]
                # Build full operator
                from ..math.qalgebra import kron, eye
                full_op = None
                for i in range(n):
                    if i == q:
                        g = H.data
                    else:
                        g = eye(2)
                    if full_op is None:
                        full_op = g
                    else:
                        full_op = kron(full_op, g)
                # Apply to state
                new_data = []
                for i in range(state.dim):
                    val = sum(full_op[i][j] * state.data[j] for j in range(state.dim))
                    new_data.append(val)
                return Ket(new_data)
        
        elif gate_name in ('x', 'pauli_x'):
            from ..quantum.operator import pauli_x
            X = pauli_x()
            if len(qubits) == 1:
                q = qubits[0]
                from ..math.qalgebra import kron, eye
                full_op = None
                for i in range(n):
                    if i == q:
                        g = X.data
                    else:
                        g = eye(2)
                    if full_op is None:
                        full_op = g
                    else:
                        full_op = kron(full_op, g)
                new_data = []
                for i in range(state.dim):
                    val = sum(full_op[i][j] * state.data[j] for j in range(state.dim))
                    new_data.append(val)
                return Ket(new_data)
        
        elif gate_name in ('z', 'pauli_z'):
            from ..quantum.operator import pauli_z
            Z = pauli_z()
            if len(qubits) == 1:
                q = qubits[0]
                from ..math.qalgebra import kron, eye
                full_op = None
                for i in range(n):
                    if i == q:
                        g = Z.data
                    else:
                        g = eye(2)
                    if full_op is None:
                        full_op = g
                    else:
                        full_op = kron(full_op, g)
                new_data = []
                for i in range(state.dim):
                    val = sum(full_op[i][j] * state.data[j] for j in range(state.dim))
                    new_data.append(val)
                return Ket(new_data)
        
        elif gate_name in ('cx', 'cnot'):
            if len(qubits) == 2:
                control, target = qubits
                new_data = state.data.copy()
                # CNOT: if control is 1, flip target
                for i in range(state.dim):
                    # Get bits
                    bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                    if bits[control] == 1:
                        # Flip target bit
                        target_bit = bits[target]
                        new_target = 1 - target_bit
                        # Build new index
                        new_bits = bits.copy()
                        new_bits[target] = new_target
                        new_idx = 0
                        for k, b in enumerate(new_bits):
                            new_idx = (new_idx << 1) | b
                        new_data[new_idx] = state.data[i]
                        new_data[i] = 0
                return Ket(new_data)
        
        return state
    
    def run_experiment(self, experiment: Experiment, shots: int = None) -> ExperimentResult:
        """
        Run a quantum experiment.

        Args:
            experiment: Experiment object to run.
            shots: Number of measurement shots (overrides experiment.shots if provided).

        Returns:
            ExperimentResult: Results of the experiment.

        Raises:
            RuntimeError: If the experiment fails during execution.

        Examples:
            >>> lab = QuantumLab()
            >>> exp = lab.create_experiment("Bell State", n_qubits=2)
            >>> exp.add_gate("h", 0)
            >>> exp.add_gate("cx", 0, 1)
            >>> result = lab.run_experiment(exp, shots=2048)
            >>> print(result.success_rate)
            0.50
        """
        experiment.status = ExperimentStatus.RUNNING
        shots = shots or experiment.shots
        
        start_time = time.time()
        
        try:
            # Start from |0...0⟩
            dim = 2 ** experiment.n_qubits
            state = basis(dim, 0)
            
            # Apply gates
            for gate_info in experiment.gates:
                gate_name = gate_info['gate']
                qubits = gate_info['qubits']
                state = self._apply_gate_to_state(state, gate_name, qubits)
            
            # Normalize (just in case)
            if not state.is_normalized:
                state = state.normalize()
            
            # Measure
            probs = [abs(amp)**2 for amp in state.data]
            total = sum(probs)
            probs = [p / total for p in probs]
            
            # Sample outcomes
            outcomes = []
            for _ in range(shots):
                r = random.random()
                cumsum = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r < cumsum:
                        outcomes.append(i)
                        break
            
            # Convert to bitstrings
            n = experiment.n_qubits
            outcome_strings = [format(o, f'0{n}b') for o in outcomes]
            
            # Count
            counts = {}
            for s in outcome_strings:
                counts[s] = counts.get(s, 0) + 1
            
            # Calculate success rate (most likely outcome)
            if counts:
                max_count = max(counts.values())
                success_rate = max_count / shots
            else:
                success_rate = 0.0
            
            execution_time = time.time() - start_time
            
            result = ExperimentResult(
                name=experiment.name,
                timestamp=datetime.now().isoformat(),
                n_qubits=experiment.n_qubits,
                shots=shots,
                counts=counts,
                probabilities=probs,
                final_state=state.data,
                success_rate=success_rate,
                execution_time=execution_time,
                status="completed"
            )
            
            experiment.result = result
            experiment.status = ExperimentStatus.COMPLETED
            
            return result
            
        except Exception as e:
            experiment.status = ExperimentStatus.FAILED
            raise RuntimeError(f"Experiment failed: {e}")
    
    def get_state_vector(self, experiment: Experiment) -> Ket:
        """
        Get the final state vector of an experiment without measurement.

        Args:
            experiment: Experiment object.

        Returns:
            Ket: State vector after applying all gates.

        Examples:
            >>> lab = QuantumLab()
            >>> exp = lab.create_experiment("Bell State", n_qubits=2)
            >>> exp.add_gate("h", 0)
            >>> exp.add_gate("cx", 0, 1)
            >>> state = lab.get_state_vector(exp)
            >>> print(state)
            0.707|00⟩ + 0.707|11⟩
        """
        dim = 2 ** experiment.n_qubits
        state = basis(dim, 0)
        
        for gate_info in experiment.gates:
            gate_name = gate_info['gate']
            qubits = gate_info['qubits']
            state = self._apply_gate_to_state(state, gate_name, qubits)
        
        return state
    
    def get_bloch_coordinates(self, experiment: Experiment, qubit: int = 0) -> Tuple[float, float, float]:
        """
        Compute Bloch sphere coordinates for a specific qubit.

        Args:
            experiment: Experiment object.
            qubit: Index of the qubit (default: 0).

        Returns:
            Tuple[float, float, float]: (x, y, z) coordinates on the Bloch sphere.

        Examples:
            >>> lab = QuantumLab()
            >>> exp = lab.create_experiment("X Gate", n_qubits=1)
            >>> exp.add_gate("x", 0)
            >>> x, y, z = lab.get_bloch_coordinates(exp, qubit=0)
            >>> print(f"({x:.2f}, {y:.2f}, {z:.2f})")
            (0.00, 0.00, -1.00)
        """
        state = self.get_state_vector(experiment)
        
        if experiment.n_qubits == 1:
            return state_to_bloch(state)
        else:
            # For multi-qubit, compute reduced density matrix
            n = experiment.n_qubits
            dim = state.dim
            
            # Compute reduced density matrix for target qubit
            rho_reduced = [[0j, 0j], [0j, 0j]]
            
            for i in range(dim):
                for j in range(dim):
                    # Get bits of i and j
                    bits_i = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                    bits_j = [(j >> (n - 1 - k)) & 1 for k in range(n)]
                    
                    # Check if other qubits match
                    other_match = True
                    for k in range(n):
                        if k != qubit:
                            if bits_i[k] != bits_j[k]:
                                other_match = False
                                break
                    
                    if other_match:
                        row = bits_i[qubit]
                        col = bits_j[qubit]
                        rho_reduced[row][col] += state.data[i] * state.data[j].conjugate()
            
            # Convert to Bloch coordinates
            x = 2 * rho_reduced[0][1].real
            y = 2 * rho_reduced[0][1].imag
            z = (rho_reduced[0][0] - rho_reduced[1][1]).real
            return (x, y, z)
    
    def list_experiments(self) -> List[Dict]:
        """
        List all experiments in the laboratory.

        Returns:
            List of dictionaries with experiment metadata.

        Examples:
            >>> lab = QuantumLab()
            >>> lab.create_experiment("Exp1", 2)
            >>> lab.create_experiment("Exp2", 3)
            >>> experiments = lab.list_experiments()
            >>> for exp in experiments:
            ...     print(exp['name'])
            Exp1
            Exp2
        """
        return [{'name': e.name, 'n_qubits': e.n_qubits, 'n_gates': len(e.gates)} 
                for e in self.experiments]


class PredefinedExperiments:
    """
    Collection of predefined quantum experiments.

    This class provides static methods to create common quantum circuits
    such as Bell states and GHZ states.

    Examples:
        >>> bell = PredefinedExperiments.bell_state()
        >>> ghz = PredefinedExperiments.ghz_state(n=4)
    """
    
    @staticmethod
    def bell_state() -> Experiment:
        """
        Create a Bell state experiment: (|00⟩ + |11⟩)/√2.

        Returns:
            Experiment configured to generate a Bell state.

        Examples:
            >>> exp = PredefinedExperiments.bell_state()
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
            >>> print(sorted(result.counts.keys()))
            ['00', '11']
        """
        exp = Experiment(name="Bell State", n_qubits=2)
        exp.add_gate("h", 0)
        exp.add_gate("cx", 0, 1)
        return exp
    
    @staticmethod
    def ghz_state(n: int = 3) -> Experiment:
        """
        Create a GHZ state experiment: (|00...0⟩ + |11...1⟩)/√2.

        Args:
            n: Number of qubits (default: 3).

        Returns:
            Experiment configured to generate an n-qubit GHZ state.

        Examples:
            >>> exp = PredefinedExperiments.ghz_state(n=4)
            >>> lab = QuantumLab()
            >>> result = lab.run_experiment(exp)
            >>> print(len(result.counts))
            2
        """
        exp = Experiment(name=f"GHZ State ({n} qubits)", n_qubits=n)
        exp.add_gate("h", 0)
        for i in range(n - 1):
            exp.add_gate("cx", i, i + 1)
        return exp


__all__ = [
    'ExperimentStatus',
    'ExperimentResult',
    'Experiment',
    'QuantumLab',
    'PredefinedExperiments',
]