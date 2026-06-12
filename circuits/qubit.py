"""
psiqit/circuits/qubit.py
Single qubit representation

This module provides the Qubit class for representing and manipulating
individual quantum bits. It supports state initialization, gate operations
(via state updates), measurement, and reset functionality.
"""

from typing import Optional, List
from ..quantum.state import Ket, zero


class Qubit:
    """
    A single quantum bit (qubit) with state and measurement capabilities.

    The Qubit class represents a single qubit in a quantum circuit or register.
    It maintains its quantum state as a Ket vector and provides methods for
    measurement, reset, and state manipulation.

    Note: Gate operations are applied by directly modifying the state through
    the state setter or by using operator methods from the quantum module.

    Examples:
        >>> # Create a qubit in |0⟩ state
        >>> q = Qubit(0)
        >>> print(q.state)
        1.000|0⟩
        
        >>> # Apply X gate (using state setter)
        >>> from psiqit.quantum.operator import pauli_x
        >>> q.state = pauli_x() @ q.state
        >>> print(q.state)
        1.000|1⟩
        
        >>> # Measure the qubit
        >>> result = q.measure()
        >>> print(f"Measured: {result}, Value: {q.value}")
        Measured: 1, Value: 1
        
        >>> # Reset the qubit
        >>> q.reset()
        >>> print(q.state)
        1.000|0⟩
    """
    
    def __init__(self, index: int, initial_state: Optional[Ket] = None):
        """
        Initialize a qubit with a given index and optional initial state.

        Args:
            index: Qubit index in the register (used for identification).
            initial_state: Initial quantum state (default: |0⟩ if None).

        Examples:
            >>> q1 = Qubit(0)                    # Qubit 0, initial |0⟩
            >>> from psiqit.quantum.state import plus
            >>> q2 = Qubit(1, initial_state=plus())  # Qubit 1, initial |+⟩
        """
        self._index = index
        self._state = initial_state if initial_state else zero()
        self._measured = False
        self._classical_value = None
    
    @property
    def index(self) -> int:
        """
        Return the qubit's index in the register.

        Returns:
            Integer index of the qubit.

        Examples:
            >>> q = Qubit(3)
            >>> q.index
            3
        """
        return self._index
    
    @property
    def state(self) -> Ket:
        """
        Return the current quantum state of the qubit.

        Returns:
            Ket: Current state vector.

        Examples:
            >>> q = Qubit(0)
            >>> print(q.state)
            1.000|0⟩
        """
        return self._state
    
    @state.setter
    def state(self, new_state: Ket):
        """
        Set the quantum state of the qubit.

        This method is used to apply gate operations by updating the state.

        Args:
            new_state: New Ket state for the qubit.

        Examples:
            >>> from psiqit.quantum.operator import pauli_x, hadamard
            >>> q = Qubit(0)
            >>> q.state = pauli_x() @ q.state  # Apply X gate
            >>> q.state = hadamard() @ q.state  # Apply H gate
        """
        self._state = new_state
        self._measured = False
        self._classical_value = None
    
    @property
    def is_measured(self) -> bool:
        """
        Check if the qubit has been measured.

        Returns:
            True if the qubit has been measured, False otherwise.

        Examples:
            >>> q = Qubit(0)
            >>> q.is_measured
            False
            >>> q.measure()
            0
            >>> q.is_measured
            True
        """
        return self._measured
    
    @property
    def value(self) -> Optional[int]:
        """
        Return the classical measurement value (if measured).

        Returns:
            Integer (0 or 1) if measured, None otherwise.

        Examples:
            >>> q = Qubit(0)
            >>> print(q.value)
            None
            >>> q.measure()
            0
            >>> q.value
            0
        """
        return self._classical_value
    
    def measure(self) -> int:
        """
        Measure the qubit in the computational basis.

        This collapses the qubit's state to either |0⟩ or |1⟩ and returns
        the measurement outcome. After measurement, the qubit's state is
        updated to the measured basis state.

        Returns:
            int: Measurement outcome (0 or 1).

        Examples:
            >>> from psiqit.quantum.state import plus
            >>> q = Qubit(0, initial_state=plus())
            >>> result = q.measure()
            >>> print(f"Outcome: {result}")
            Outcome: 0  # or 1 with 50% probability
            >>> print(q.value)
            0
        """
        result = self._state.measure(shots=1)
        outcome = result['outcomes'][0]
        self._measured = True
        self._classical_value = outcome
        return outcome
    
    def reset(self):
        """
        Reset the qubit to the |0⟩ state.

        This clears the quantum state and measurement information, allowing
        the qubit to be reused.

        Examples:
            >>> q = Qubit(0)
            >>> from psiqit.quantum.operator import pauli_x
            >>> q.state = pauli_x() @ q.state
            >>> print(q.state)  # |1⟩
            1.000|1⟩
            >>> q.reset()
            >>> print(q.state)  # |0⟩
            1.000|0⟩
            >>> q.is_measured
            False
        """
        self._state = zero()
        self._measured = False
        self._classical_value = None
    
    def __repr__(self) -> str:
        """
        Return a string representation of the qubit.

        Returns:
            String showing index and measurement status.

        Examples:
            >>> q = Qubit(0)
            >>> repr(q)
            'Qubit(0)'
            >>> q.measure()
            0
            >>> repr(q)
            'Qubit(0, measured=0)'
        """
        if self._measured:
            return f"Qubit({self._index}, measured={self._classical_value})"
        return f"Qubit({self._index})"
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the qubit.

        Returns:
            String showing qubit index and its quantum state.

        Examples:
            >>> q = Qubit(0)
            >>> print(q)
            q0: 1.000|0⟩
        """
        return f"q{self._index}: {self._state}"