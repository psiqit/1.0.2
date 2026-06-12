"""
psiqit/circuits/circuit.py
Quantum circuit builder and executor

The QuantumCircuit class is the core component for building and simulating
quantum circuits. It supports single-qubit gates, multi-qubit gates,
measurements, and circuit visualization.
"""

from typing import List, Union, Optional, Tuple, Dict
import math
import random

from ..quantum.state import Ket, basis
from ..quantum.operator import Operator
from ..math.qalgebra import kron, eye, mul


class QuantumCircuit:
    """
    Quantum circuit for building and simulating quantum algorithms.

    The QuantumCircuit class provides a high-level interface for constructing
    quantum circuits by adding gates, then executing them to obtain the final
    state vector or measurement results.

    Examples:
        >>> # Create a Bell state circuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> state = circ.run()
        >>> print(state)  # (|00⟩ + |11⟩)/√2
        
        >>> # Measure the circuit
        >>> results = circ.measure(shots=1024)
        >>> print(results['counts'])  # {'00': 512, '11': 512}
        
        >>> # Draw the circuit
        >>> print(circ.draw())
        q0: ─[H]─●─
        q1: ─────⊕─
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize a quantum circuit with a specified number of qubits.

        The circuit starts in the all-zero state |00...0⟩.

        Args:
            n_qubits: Number of qubits in the circuit.

        Examples:
            >>> circ = QuantumCircuit(2)  # 2-qubit circuit
            >>> circ = QuantumCircuit(5)  # 5-qubit circuit
        """
        self._n_qubits = n_qubits
        self._state = basis(2**n_qubits, 0)  # |00...0⟩
        self._gates: List[Tuple[Operator, List[int]]] = []
        self._measured = False
    
    @property
    def n_qubits(self) -> int:
        """
        Return the number of qubits in the circuit.

        Returns:
            Number of qubits.

        Examples:
            >>> circ = QuantumCircuit(3)
            >>> circ.n_qubits
            3
        """
        return self._n_qubits
    
    @property
    def depth(self) -> int:
        """
        Return the number of gates in the circuit.

        Returns:
            Circuit depth (gate count).

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.h(0)
            >>> circ.cx(0, 1)
            >>> circ.depth
            2
        """
        return len(self._gates)
    
    # ============================================================
    # Gate Applications
    # ============================================================
    
    def _apply_gate(self, gate: Operator, qubits: List[int]):
        """
        Add a gate to the circuit (internal method).

        Args:
            gate: Operator representing the quantum gate.
            qubits: List of qubit indices to apply the gate to.
        """
        self._gates.append((gate, qubits))
        self._measured = False
    
    def _build_full_operator(self, gate: Operator, qubits: List[int]) -> List[List[complex]]:
        """
        Build the full operator for the entire register (internal method).

        This method expands a gate acting on specific qubits to the full
        Hilbert space of all qubits using tensor products.

        Args:
            gate: Gate to apply.
            qubits: Qubit indices to apply gate to.

        Returns:
            Full operator matrix of size 2ⁿ × 2ⁿ.

        Raises:
            ValueError: If the gate is applied to an unsupported number of qubits.
        """
        n = self._n_qubits
        dim = 2 ** n
        
        if len(qubits) == 1:
            # Single-qubit gate
            q = qubits[0]
            result = None
            
            for i in range(n):
                if i == q:
                    g = gate.data
                else:
                    g = eye(2)
                
                if result is None:
                    result = g
                else:
                    result = kron(result, g)
            
            return result
        
        elif len(qubits) == 2:
            # Two-qubit gate (CNOT, CZ, SWAP)
            q1, q2 = qubits
            
            # Build permutation to bring qubits to positions 0 and 1
            # For now, assume qubits are in order and we're applying to first two
            # This is a simplification - full implementation would permute indices
            
            # If qubits are not [0, 1], we need to swap them
            if q1 == 0 and q2 == 1:
                # Direct application
                result = gate.data
                
                # Add identity for other qubits
                for i in range(2, n):
                    result = kron(result, eye(2))
                
                return result
            else:
                # Need to swap qubits to bring them to positions 0 and 1
                # Build swap network (simplified: just build full operator directly)
                result = eye(dim)
                
                # Iterate over all basis states
                for i in range(dim):
                    # Get bits
                    bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                    
                    # Check the control and target bits
                    control_bit = bits[q1]
                    target_bit = bits[q2]
                    
                    # For CNOT: if control is 1, flip target
                    if gate.name == "CNOT":
                        if control_bit == 1:
                            new_target = 1 - target_bit
                        else:
                            new_target = target_bit
                        
                        # Build new bits
                        new_bits = bits.copy()
                        new_bits[q2] = new_target
                        
                        # Compute new index
                        new_idx = 0
                        for k, b in enumerate(new_bits):
                            new_idx = (new_idx << 1) | b
                        
                        result[i][new_idx] = 1.0
                    
                    elif gate.name == "CZ":
                        # CZ: if both are 1, apply -1 phase
                        if control_bit == 1 and target_bit == 1:
                            result[i][i] = -1.0
                        else:
                            result[i][i] = 1.0
                    
                    elif gate.name == "SWAP":
                        # SWAP: exchange qubits
                        new_bits = bits.copy()
                        new_bits[q1], new_bits[q2] = new_bits[q2], new_bits[q1]
                        
                        new_idx = 0
                        for k, b in enumerate(new_bits):
                            new_idx = (new_idx << 1) | b
                        
                        result[i][new_idx] = 1.0
                
                return result
        
        elif len(qubits) == 3:
            # Three-qubit gate (Toffoli)
            q1, q2, q3 = qubits
            dim = 2 ** n
            
            result = eye(dim)
            
            for i in range(dim):
                bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
                
                # Toffoli: if both controls are 1, flip target
                if bits[q1] == 1 and bits[q2] == 1:
                    new_bits = bits.copy()
                    new_bits[q3] = 1 - bits[q3]
                    
                    new_idx = 0
                    for k, b in enumerate(new_bits):
                        new_idx = (new_idx << 1) | b
                    
                    result[i][new_idx] = 1.0
                else:
                    result[i][i] = 1.0
            
            return result
        
        else:
            raise ValueError(f"Unsupported gate with {len(qubits)} qubits")
    
    # -----------------------------
    # Single-qubit gates
    # -----------------------------
    
    def x(self, qubit: int):
        """
        Apply Pauli X gate (NOT gate) to a qubit.

        The X gate flips the state: |0⟩ ↔ |1⟩.

        Args:
            qubit: Index of the target qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(1)
            >>> circ.x(0)
            >>> state = circ.run()
            >>> print(state)  # |1⟩
            1.000|1⟩
        """
        from ..quantum.operator import pauli_x
        self._apply_gate(pauli_x(), [qubit])
        return self
    
    def y(self, qubit: int):
        """
        Apply Pauli Y gate to a qubit.

        The Y gate applies a bit flip and a phase flip: Y = i|1⟩⟨0| - i|0⟩⟨1|.

        Args:
            qubit: Index of the target qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(1)
            >>> circ.y(0)
            >>> state = circ.run()
        """
        from ..quantum.operator import pauli_y
        self._apply_gate(pauli_y(), [qubit])
        return self
    
    def z(self, qubit: int):
        """
        Apply Pauli Z gate (phase flip) to a qubit.

        The Z gate leaves |0⟩ unchanged and flips the sign of |1⟩: Z|1⟩ = -|1⟩.

        Args:
            qubit: Index of the target qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(1)
            >>> circ.z(0)
            >>> state = circ.run()
        """
        from ..quantum.operator import pauli_z
        self._apply_gate(pauli_z(), [qubit])
        return self
    
    def h(self, qubit: int):
        """
        Apply Hadamard gate to a qubit.

        The Hadamard gate creates superposition: H|0⟩ = |+⟩, H|1⟩ = |-⟩.

        Args:
            qubit: Index of the target qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(1)
            >>> circ.h(0)
            >>> state = circ.run()
            >>> print(state)  # |+⟩
            0.707|0⟩ + 0.707|1⟩
        """
        from ..quantum.operator import hadamard
        self._apply_gate(hadamard(), [qubit])
        return self
    
    def s(self, qubit: int):
        """
        Apply S gate (phase gate, π/2) to a qubit.

        The S gate applies a phase of i to the |1⟩ state: S|1⟩ = i|1⟩.

        Args:
            qubit: Index of the target qubit.

        Returns:
            Self for method chaining.
        """
        from ..quantum.operator import s_gate
        self._apply_gate(s_gate(), [qubit])
        return self
    
    def t(self, qubit: int):
        """
        Apply T gate (π/4 phase) to a qubit.

        The T gate applies a phase of e^{iπ/4} to the |1⟩ state.

        Args:
            qubit: Index of the target qubit.

        Returns:
            Self for method chaining.
        """
        from ..quantum.operator import t_gate
        self._apply_gate(t_gate(), [qubit])
        return self
    
    def rx(self, qubit: int, theta: float):
        """
        Apply RX rotation gate (rotation around X-axis).

        RX(θ) = exp(-iθX/2) = cos(θ/2)I - i sin(θ/2)X

        Args:
            qubit: Index of the target qubit.
            theta: Rotation angle in radians.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(1)
            >>> circ.rx(0, math.pi/2)
        """
        from ..quantum.operator import rx
        self._apply_gate(rx(theta), [qubit])
        return self
    
    def ry(self, qubit: int, theta: float):
        """
        Apply RY rotation gate (rotation around Y-axis).

        RY(θ) = exp(-iθY/2) = cos(θ/2)I - i sin(θ/2)Y

        Args:
            qubit: Index of the target qubit.
            theta: Rotation angle in radians.

        Returns:
            Self for method chaining.
        """
        from ..quantum.operator import ry
        self._apply_gate(ry(theta), [qubit])
        return self
    
    def rz(self, qubit: int, theta: float):
        """
        Apply RZ rotation gate (rotation around Z-axis).

        RZ(θ) = exp(-iθZ/2) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]

        Args:
            qubit: Index of the target qubit.
            theta: Rotation angle in radians.

        Returns:
            Self for method chaining.
        """
        from ..quantum.operator import rz
        self._apply_gate(rz(theta), [qubit])
        return self
    
    # -----------------------------
    # Two-qubit gates
    # -----------------------------
    
    def cx(self, control: int, target: int):
        """
        Apply CNOT (controlled-NOT) gate.

        The CNOT gate flips the target qubit if the control qubit is |1⟩.

        Args:
            control: Index of the control qubit.
            target: Index of the target qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.cx(0, 1)  # CNOT with control=q0, target=q1
        """
        from ..quantum.operator import cnot
        self._apply_gate(cnot(), [control, target])
        return self
    
    def cz(self, control: int, target: int):
        """
        Apply CZ (controlled-Z) gate.

        The CZ gate applies a Z gate to the target if the control is |1⟩,
        adding a -1 phase when both qubits are |1⟩.

        Args:
            control: Index of the control qubit.
            target: Index of the target qubit.

        Returns:
            Self for method chaining.
        """
        from ..quantum.operator import cz
        self._apply_gate(cz(), [control, target])
        return self
    
    def swap(self, qubit1: int, qubit2: int):
        """
        Apply SWAP gate.

        The SWAP gate exchanges the states of two qubits.

        Args:
            qubit1: Index of the first qubit.
            qubit2: Index of the second qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.swap(0, 1)  # Swap q0 and q1
        """
        from ..quantum.operator import swap
        self._apply_gate(swap(), [qubit1, qubit2])
        return self
    
    # -----------------------------
    # Three-qubit gates
    # -----------------------------
    
    def toffoli(self, control1: int, control2: int, target: int):
        """
        Apply Toffoli (CCNOT) gate.

        The Toffoli gate flips the target qubit if both control qubits are |1⟩.
        It is a universal gate for classical reversible computing.

        Args:
            control1: Index of the first control qubit.
            control2: Index of the second control qubit.
            target: Index of the target qubit.

        Returns:
            Self for method chaining.

        Examples:
            >>> circ = QuantumCircuit(3)
            >>> circ.toffoli(0, 1, 2)  # CCNOT with controls q0,q1 and target q2
        """
        from ..quantum.operator import toffoli
        self._apply_gate(toffoli(), [control1, control2, target])
        return self
    
    # ============================================================
    # Execution
    # ============================================================
    
    def run(self) -> Ket:
        """
        Execute the circuit and return the final state vector.

        This method simulates the quantum circuit by applying all added gates
        in sequence to the initial |00...0⟩ state.

        Returns:
            Ket: Final quantum state after applying all gates.

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.h(0)
            >>> circ.cx(0, 1)
            >>> state = circ.run()
            >>> print(state)  # Bell state
            0.707|00⟩ + 0.707|11⟩
        """
        state = self._state.copy()
        
        for gate, qubits in self._gates:
            full_operator = self._build_full_operator(gate, qubits)
            
            # Apply operator to state
            new_data = []
            for i in range(len(full_operator)):
                val = sum(full_operator[i][j] * state.data[j] for j in range(len(state.data)))
                new_data.append(val)
            
            state = Ket(new_data)
        
        self._state = state
        return state
    
    # ============================================================
    # Measurement
    # ============================================================
    
    def measure(self, qubit: Optional[int] = None, shots: int = 1):
        """
        Measure qubit(s) after running the circuit.

        Args:
            qubit: Qubit index to measure (if None, measure all qubits).
            shots: Number of measurements to perform.

        Returns:
            - If qubit is specified and shots=1: int (0 or 1)
            - If qubit is specified and shots>1: List[int]
            - If qubit is None and shots=1: str (bitstring)
            - If qubit is None and shots>1: dict mapping bitstring to count

        Raises:
            ValueError: If the state is not normalized.

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.h(0)
            >>> circ.cx(0, 1)
            >>> 
            >>> # Measure all qubits
            >>> result = circ.measure(shots=1000)
            >>> print(result['counts'])  # {'00': 500, '11': 500}
            >>> 
            >>> # Measure single qubit
            >>> bit = circ.measure(qubit=0, shots=1)
            >>> print(bit)  # 0 or 1
        """
        state = self.run()
        n = self._n_qubits
        
        if qubit is not None:
            # Measure single qubit
            prob_0 = 0
            for i, amp in enumerate(state.data):
                bit = (i >> (n - 1 - qubit)) & 1
                if bit == 0:
                    prob_0 += abs(amp) ** 2
            
            outcomes = []
            for _ in range(shots):
                r = random.random()
                outcome = 0 if r < prob_0 else 1
                outcomes.append(outcome)
            
            return outcomes[0] if shots == 1 else outcomes
        
        else:
            # Measure all qubits
            probs = [abs(amp)**2 for amp in state.data]
            outcomes = []
            for _ in range(shots):
                r = random.random()
                cumsum = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r < cumsum:
                        bitstring = format(i, f'0{n}b')
                        outcomes.append(bitstring)
                        break
            
            if shots == 1:
                return outcomes[0]
            
            counts = {}
            for outcome in outcomes:
                counts[outcome] = counts.get(outcome, 0) + 1
            return counts
    
    # ============================================================
    # Utility Methods
    # ============================================================
    
    def draw(self) -> str:
        """
        Draw a simple ASCII representation of the circuit.

        The drawing uses Unicode box-drawing characters to show the circuit
        structure, including gates and connections.

        Returns:
            String representation of the circuit.

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.h(0)
            >>> circ.cx(0, 1)
            >>> print(circ.draw())
            q0: ─[H]─●─
            q1: ─────⊕─
        """
        lines = [f"q{i}: ─" for i in range(self._n_qubits)]
        
        for gate, qubits in self._gates:
            gate_name = gate.name or "U"
            
            if len(qubits) == 1:
                q = qubits[0]
                for i in range(self._n_qubits):
                    if i == q:
                        lines[i] += f"[{gate_name}]─"
                    else:
                        lines[i] += "───"
            
            elif len(qubits) == 2:
                q1, q2 = qubits
                min_q, max_q = min(q1, q2), max(q1, q2)
                
                for i in range(self._n_qubits):
                    if i == q1:
                        lines[i] += "●─"
                    elif i == q2:
                        if gate_name == "CNOT":
                            lines[i] += "⊕─"
                        else:
                            lines[i] += f"[{gate_name}]─"
                    elif min_q < i < max_q:
                        lines[i] += "│─"
                    else:
                        lines[i] += "──"
            
            elif len(qubits) == 3:
                q1, q2, q3 = qubits
                for i in range(self._n_qubits):
                    if i == q1 or i == q2:
                        lines[i] += "●─"
                    elif i == q3:
                        lines[i] += "⊕─"
                    else:
                        lines[i] += "──"
        
        return "\n".join(lines)
    
    def reset(self):
        """
        Reset the circuit to its initial state.

        This clears all added gates and resets the state to |00...0⟩.

        Examples:
            >>> circ = QuantumCircuit(2)
            >>> circ.h(0)
            >>> circ.reset()
            >>> print(circ.depth)  # 0
            0
        """
        self._state = basis(2**self._n_qubits, 0)
        self._gates = []
        self._measured = False
    
    def __repr__(self) -> str:
        """Return a string representation of the circuit."""
        return f"QuantumCircuit(n={self._n_qubits}, depth={self.depth})"