"""
psiqit/circuits/register.py
Quantum register - collection of qubits

This module provides the QuantumRegister class for managing multiple qubits
as a single register, with support for state vector operations, individual
qubit access, and measurements.
"""

from typing import List, Union, Optional
from ..quantum.state import Ket, basis, zero
from .qubit import Qubit


class QuantumRegister:
    """
    A register of multiple qubits for quantum computing.

    The QuantumRegister class manages a collection of qubits and provides
    operations on the entire register, such as getting the full state vector,
    measuring individual or all qubits, and resetting the register.

    Examples:
        >>> # Create a 2-qubit register
        >>> reg = QuantumRegister(2)
        >>> reg.n_qubits
        2
        
        >>> # Access individual qubits
        >>> reg[0].state = pauli_x() @ reg[0].state  # Apply X to first qubit
        >>> print(reg.get_state_vector())
        
        >>> # Measure all qubits
        >>> results = reg.measure()
        >>> print(results)  # [0, 0] or [1, 0] etc.
        
        >>> # Reset the register
        >>> reg.reset()
    """
    
    def __init__(self, n_qubits: int, initial_states: Optional[List[Ket]] = None):
        """
        Initialize a quantum register with a specified number of qubits.

        Args:
            n_qubits: Number of qubits in the register.
            initial_states: Optional list of initial states for each qubit.
                            If not provided, all qubits start in |0⟩.

        Raises:
            ValueError: If length of initial_states does not match n_qubits.

        Examples:
            >>> # Register with all qubits in |0⟩
            >>> reg1 = QuantumRegister(3)
            
            >>> # Register with custom initial states
            >>> from psiqit.quantum.state import plus, zero, one
            >>> reg2 = QuantumRegister(3, initial_states=[plus(), zero(), one()])
        """
        self._n_qubits = n_qubits
        
        if initial_states:
            if len(initial_states) != n_qubits:
                raise ValueError("Number of initial states must match n_qubits")
            self._qubits = [Qubit(i, initial_states[i]) for i in range(n_qubits)]
        else:
            self._qubits = [Qubit(i, zero()) for i in range(n_qubits)]
    
    @property
    def n_qubits(self) -> int:
        """
        Return the number of qubits in the register.

        Returns:
            Number of qubits.

        Examples:
            >>> reg = QuantumRegister(4)
            >>> reg.n_qubits
            4
        """
        return self._n_qubits
    
    @property
    def dim(self) -> int:
        """
        Return the Hilbert space dimension (2ⁿ).

        Returns:
            Dimension of the combined Hilbert space.

        Examples:
            >>> reg = QuantumRegister(3)
            >>> reg.dim
            8
        """
        return 2 ** self._n_qubits
    
    def get_state_vector(self) -> Ket:
        """
        Get the full state vector of the register.

        This computes the tensor product of all individual qubit states:
        |ψ⟩ = |q₀⟩ ⊗ |q₁⟩ ⊗ ... ⊗ |q_{n-1}⟩

        Returns:
            Ket: Combined state vector of the entire register.

        Examples:
            >>> reg = QuantumRegister(2)
            >>> from psiqit.quantum.operator import pauli_x
            >>> reg[0].state = pauli_x() @ reg[0].state  # First qubit to |1⟩
            >>> state = reg.get_state_vector()
            >>> print(state)  # |10⟩
            1.000|10⟩
        """
        if self._n_qubits == 1:
            return self._qubits[0].state
        
        # Start with first qubit
        from ..math.qalgebra import kron
        
        result = self._qubits[0].state.data
        for qubit in self._qubits[1:]:
            # Tensor product
            new_data = []
            for a in result:
                for b in qubit.state.data:
                    new_data.append(a * b)
            result = new_data
        
        return Ket(result)
    
    def set_state_vector(self, state: Ket):
        """
        Set the full state vector of the register.

        Note: This is a simplified implementation that stores the combined
        state directly. A full implementation would decompose the state
        into individual qubit states.

        Args:
            state: Combined state vector for the entire register.

        Raises:
            ValueError: If state dimension does not match register dimension.

        Examples:
            >>> reg = QuantumRegister(2)
            >>> from psiqit.quantum.state import basis
            >>> bell_state = basis(4, 0) + basis(4, 3)
            >>> bell_state = bell_state.normalize()
            >>> reg.set_state_vector(bell_state)
        """
        if state.dim != self.dim:
            raise ValueError(f"State dimension {state.dim} does not match register dimension {self.dim}")
        
        # For now, just store as is
        # In a full implementation, we would decompose into individual qubit states
        self._combined_state = state
    
    def get_qubit(self, index: int) -> Qubit:
        """
        Get a qubit by its index.

        Args:
            index: Index of the qubit (0 to n_qubits-1).

        Returns:
            Qubit: The qubit at the specified index.

        Raises:
            IndexError: If index is out of range.

        Examples:
            >>> reg = QuantumRegister(3)
            >>> q = reg.get_qubit(1)  # Get the second qubit
            >>> print(q.index)
            1
        """
        if index < 0 or index >= self._n_qubits:
            raise IndexError(f"Qubit index {index} out of range")
        return self._qubits[index]
    
    def measure(self, qubit_index: Optional[int] = None):
        """
        Measure a specific qubit or all qubits in the register.

        Args:
            qubit_index: Index of qubit to measure. If None, measure all qubits.

        Returns:
            - If qubit_index is specified: int (0 or 1)
            - If qubit_index is None: List[int] of measurement outcomes

        Examples:
            >>> reg = QuantumRegister(2)
            >>> # Measure single qubit
            >>> result = reg.measure(qubit_index=0)
            >>> print(result)  # 0 or 1
            
            >>> # Measure all qubits
            >>> results = reg.measure()
            >>> print(results)  # [0, 0] or [0, 1] or [1, 0] or [1, 1]
        """
        if qubit_index is not None:
            return self._qubits[qubit_index].measure()
        else:
            return [q.measure() for q in self._qubits]
    
    def reset(self):
        """
        Reset all qubits in the register to the |0⟩ state.

        This clears the quantum state of every qubit and resets all
        measurement information.

        Examples:
            >>> reg = QuantumRegister(2)
            >>> from psiqit.quantum.operator import pauli_x
            >>> reg[0].state = pauli_x() @ reg[0].state  # First qubit to |1⟩
            >>> print(reg.get_state_vector())
            1.000|10⟩
            >>> reg.reset()
            >>> print(reg.get_state_vector())
            1.000|00⟩
        """
        for qubit in self._qubits:
            qubit.reset()
    
    def __getitem__(self, index: int) -> Qubit:
        """
        Access a qubit using indexing syntax.

        Args:
            index: Index of the qubit.

        Returns:
            Qubit: The qubit at the specified index.

        Examples:
            >>> reg = QuantumRegister(3)
            >>> qubit = reg[0]  # Same as reg.get_qubit(0)
            >>> qubit = reg[-1]  # Last qubit
        """
        return self.get_qubit(index)
    
    def __len__(self) -> int:
        """
        Return the number of qubits in the register.

        Returns:
            Number of qubits.

        Examples:
            >>> reg = QuantumRegister(3)
            >>> len(reg)
            3
        """
        return self._n_qubits
    
    def __repr__(self) -> str:
        """
        Return a string representation of the register.

        Returns:
            String showing the number of qubits.

        Examples:
            >>> reg = QuantumRegister(4)
            >>> repr(reg)
            'QuantumRegister(n=4)'
        """
        return f"QuantumRegister(n={self._n_qubits})"
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the register.

        Returns:
            Multi-line string showing each qubit's state.

        Examples:
            >>> reg = QuantumRegister(2)
            >>> print(reg)
            q0: 1.000|0⟩
            q1: 1.000|0⟩
        """
        return "\n".join(str(q) for q in self._qubits)