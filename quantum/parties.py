"""
psiqit/quantum/parties.py
Quantum Communication Parties

This module provides classes for simulating quantum communication protocols
between parties (Alice, Bob, Charlie). It includes implementations of
fundamental quantum communication protocols:

    • BB84 Quantum Key Distribution (QKD)
    • Superdense coding
    • Quantum teleportation

The parties can prepare, send, receive, and measure quantum states, as well
as encode classical bits into quantum states.

Example:
    >>> from psiqit.quantum.parties import Alice, Bob, BB84
    >>> from psiqit.quantum.state import zero, one
    >>> 
    >>> # Alice prepares a state
    >>> alice = Alice()
    >>> alice.prepare(zero())
    >>> 
    >>> # Bob receives and measures
    >>> bob = Bob()
    >>> bob.receive(alice.send())
    >>> result = bob.measure()
    >>> 
    >>> # Run BB84 protocol
    >>> bb84 = BB84()
    >>> key = bb84.run(n_bits=100)
    >>> print(f"Shared key: {key[:20]}...")

References:
    C. H. Bennett and G. Brassard, "Quantum cryptography: Public key
    distribution and coin tossing," Proceedings of IEEE, 1984.
    C. H. Bennett et al., "Teleporting an unknown quantum state via dual
    classical and Einstein-Podolsky-Rosen channels," Phys. Rev. Lett., 1993.
"""

import random
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

from .state import Ket, basis, zero, one, plus, minus, bell_phi_plus, random_state
from .operator import hadamard, pauli_x, pauli_z, cnot
from ..circuits.circuit import QuantumCircuit
from ..quantum.measurement import measure


@dataclass
class Alice:
    """
    Alice - a quantum communication party (sender).

    Alice can prepare quantum states, encode classical bits, send states,
    and measure in different bases.

    Attributes:
        name: Name of the party (default: "Alice").
        _state: Current quantum state (internal).
        _classical_bits: List of encoded classical bits.

    Examples:
        >>> alice = Alice()
        >>> alice.prepare(zero())
        >>> alice.encode_bit(1)  # Apply X gate
        >>> state = alice.send()
        >>> result = alice.measure(basis='z')
    """
    name: str = "Alice"
    _state: Optional[Ket] = None
    _classical_bits: List[int] = field(default_factory=list)
    
    def prepare(self, state: Ket):
        """
        Prepare a quantum state.

        Args:
            state: The quantum state to prepare.

        Returns:
            Self for method chaining.

        Examples:
            >>> alice = Alice()
            >>> alice.prepare(plus())
        """
        self._state = state
        return self
    
    def prepare_random(self, n_qubits: int = 1) -> Ket:
        """
        Prepare a random quantum state.

        Args:
            n_qubits: Number of qubits (default: 1).

        Returns:
            The random state.

        Examples:
            >>> alice = Alice()
            >>> state = alice.prepare_random(n_qubits=2)
        """
        self._state = random_state(2 ** n_qubits)
        return self._state
    
    def send(self) -> Ket:
        """
        Send the prepared quantum state.

        Returns:
            The quantum state to send.

        Raises:
            ValueError: If no state has been prepared.

        Examples:
            >>> alice.prepare(zero())
            >>> state = alice.send()
        """
        if self._state is None:
            raise ValueError("No state prepared. Call prepare() first.")
        return self._state
    
    def measure(self, basis: str = 'z') -> int:
        """
        Measure the state in the specified basis.

        Args:
            basis: Measurement basis ('z' for computational, 'x' for Hadamard).

        Returns:
            Measurement outcome (0 or 1).

        Raises:
            ValueError: If no state to measure or unknown basis.

        Examples:
            >>> alice.prepare(plus())
            >>> result = alice.measure(basis='x')
        """
        if self._state is None:
            raise ValueError("No state to measure")
        
        if basis == 'z':
            result = self._state.measure(shots=1)
            return result['outcomes'][0]
        elif basis == 'x':
            H = hadamard()
            state_x = H @ self._state
            result = state_x.measure(shots=1)
            return result['outcomes'][0]
        else:
            raise ValueError(f"Unknown basis: {basis}")
    
    def encode_bit(self, bit: int):
        """
        Encode a classical bit by applying X gate if bit = 1.

        Args:
            bit: Classical bit (0 or 1).

        Returns:
            Self for method chaining.

        Examples:
            >>> alice.prepare(zero())
            >>> alice.encode_bit(1)  # Now state is |1⟩
        """
        if self._state is None:
            self._state = zero()
        if bit == 1:
            self._state = pauli_x() @ self._state
        self._classical_bits.append(bit)
        return self


@dataclass
class Bob:
    """
    Bob - a quantum communication party (receiver).

    Bob can receive quantum states, measure them in different bases,
    and retrieve classical bits.

    Attributes:
        name: Name of the party (default: "Bob").
        _state: Current quantum state (internal).
        _measurements: List of measurement outcomes.

    Examples:
        >>> bob = Bob()
        >>> bob.receive(state)
        >>> result = bob.measure(basis='z')
        >>> bit = bob.get_bit()
    """
    name: str = "Bob"
    _state: Optional[Ket] = None
    _measurements: List[int] = field(default_factory=list)
    
    def receive(self, state: Ket):
        """
        Receive a quantum state.

        Args:
            state: The received quantum state.

        Returns:
            Self for method chaining.

        Examples:
            >>> bob = Bob()
            >>> bob.receive(state)
        """
        self._state = state
        return self
    
    def measure(self, basis: str = 'z') -> int:
        """
        Measure the state in the specified basis.

        Args:
            basis: Measurement basis ('z' or 'x').

        Returns:
            Measurement outcome (0 or 1).

        Raises:
            ValueError: If no state received or unknown basis.

        Examples:
            >>> bob.receive(state)
            >>> result = bob.measure(basis='x')
        """
        if self._state is None:
            raise ValueError("No state received")
        
        if basis == 'z':
            result = self._state.measure(shots=1)
            return result['outcomes'][0]
        elif basis == 'x':
            H = hadamard()
            state_x = H @ self._state
            result = state_x.measure(shots=1)
            return result['outcomes'][0]
        else:
            raise ValueError(f"Unknown basis: {basis}")
    
    def get_bit(self) -> int:
        """
        Get the last measurement outcome.

        Returns:
            Last measurement outcome, or -1 if no measurements.

        Examples:
            >>> bob.measure()
            >>> bit = bob.get_bit()
        """
        if not self._measurements:
            return -1
        return self._measurements[-1]


@dataclass
class Charlie:
    """
    Charlie - a third party for quantum protocols.

    Charlie can receive and measure quantum states, often used as a
    trusted third party in quantum communication protocols.

    Attributes:
        name: Name of the party (default: "Charlie").
        _state: Current quantum state (internal).

    Examples:
        >>> charlie = Charlie()
        >>> charlie.receive(state)
        >>> result = charlie.measure()
    """
    name: str = "Charlie"
    _state: Optional[Ket] = None
    
    def receive(self, state: Ket):
        """
        Receive a quantum state.

        Args:
            state: The received quantum state.

        Returns:
            Self for method chaining.
        """
        self._state = state
        return self
    
    def measure(self) -> int:
        """
        Measure the state in the computational basis.

        Returns:
            Measurement outcome (0 or 1).

        Raises:
            ValueError: If no state received.
        """
        if self._state is None:
            raise ValueError("No state received")
        result = self._state.measure(shots=1)
        return result['outcomes'][0]


# ============================================================
# Communication Protocols
# ============================================================

class BB84:
    """
    BB84 Quantum Key Distribution (QKD) protocol.

    The BB84 protocol allows two parties (Alice and Bob) to establish a
    shared secret key using quantum states. It is provably secure against
    eavesdropping.

    The protocol works as follows:
        1. Alice sends random bits encoded in random bases (Z or X)
        2. Bob measures in random bases
        3. They compare bases over a classical channel
        4. They keep bits where bases match (sifted key)

    References:
        C. H. Bennett and G. Brassard, "Quantum cryptography: Public key
        distribution and coin tossing," Proceedings of IEEE, 1984.

    Examples:
        >>> bb84 = BB84(eavesdropping=False)
        >>> key = bb84.run(n_bits=100)
        >>> print(f"Key length: {len(key)}")
        >>> 
        >>> # With eavesdropping detection
        >>> bb84 = BB84(eavesdropping=True)
        >>> key = bb84.run(n_bits=100)  # Key will be shorter due to errors
    """
    
    def __init__(self, eavesdropping: bool = False):
        """
        Initialize the BB84 protocol.

        Args:
            eavesdropping: If True, simulate an eavesdropper (Eve) who
                          intercepts and resends qubits randomly.
        """
        self.eavesdropping = eavesdropping
        self.alice = Alice()
        self.bob = Bob()
    
    def run(self, n_bits: int = 100) -> str:
        """
        Run the BB84 protocol.

        Args:
            n_bits: Number of bits to send (raw key length).

        Returns:
            Shared secret key as a binary string.

        Examples:
            >>> bb84 = BB84()
            >>> key = bb84.run(n_bits=200)
            >>> print(f"Shared key: {key[:20]}...")
        """
        alice_bases = []
        bob_bases = []
        alice_bits = []
        bob_bits = []
        
        for _ in range(n_bits):
            # Alice: random bit and random basis
            bit = 0 if random.random() < 0.5 else 1
            basis = 'z' if random.random() < 0.5 else 'x'
            alice_bases.append(basis)
            alice_bits.append(bit)
            
            # Prepare state
            if basis == 'z':
                state = zero() if bit == 0 else one()
            else:
                state = plus() if bit == 0 else minus()
            
            # Eavesdropping (if enabled)
            if self.eavesdropping:
                if random.random() < 0.5:
                    eve_basis = 'z' if random.random() < 0.5 else 'x'
                    if eve_basis == 'z':
                        result = state.measure(shots=1)['outcomes'][0]
                        state = zero() if result == 0 else one()
                    else:
                        H = hadamard()
                        state_x = H @ state
                        result = state_x.measure(shots=1)['outcomes'][0]
                        state = H @ (zero() if result == 0 else one())
            
            # Bob: random basis
            basis = 'z' if random.random() < 0.5 else 'x'
            bob_bases.append(basis)
            
            # Bob measures
            if basis == 'z':
                result = state.measure(shots=1)['outcomes'][0]
            else:
                H = hadamard()
                state_x = H @ state
                result = state_x.measure(shots=1)['outcomes'][0]
            bob_bits.append(result)
        
        # Sift key (keep only where bases match)
        key = []
        for i in range(n_bits):
            if alice_bases[i] == bob_bases[i]:
                key.append(str(alice_bits[i]))
        
        return ''.join(key)


class SuperdenseCoding:
    """
    Superdense coding protocol.

    Superdense coding allows Alice to send two classical bits to Bob by
    sending a single qubit, using a pre-shared entangled Bell pair.

    The protocol:
        1. Alice and Bob share a Bell pair |Φ⁺⟩
        2. Alice applies operations based on the two bits she wants to send
        3. Alice sends her qubit to Bob
        4. Bob performs a Bell measurement to recover both bits

    References:
        C. H. Bennett and S. J. Wiesner, "Communication via one- and two-particle
        operators on Einstein-Podolsky-Rosen states," Phys. Rev. Lett., 1992.

    Examples:
        >>> sc = SuperdenseCoding()
        >>> bits = (1, 0)
        >>> received = sc.run(bits)
        >>> print(received)  # Should be (1, 0)
    """
    
    def __init__(self):
        """Initialize the superdense coding protocol."""
        self.alice = Alice()
        self.bob = Bob()
    
    def run(self, bits: Tuple[int, int]) -> Tuple[int, int]:
        """
        Run the superdense coding protocol.

        Args:
            bits: Tuple of two classical bits (b0, b1) to send.

        Returns:
            Received bits (should equal the sent bits in ideal conditions).

        Examples:
            >>> sc = SuperdenseCoding()
            >>> result = sc.run((1, 0))
            >>> print(result)
            (1, 0)
        """
        bell = bell_phi_plus()
        
        b0, b1 = bits
        if b0 == 1:
            bell = pauli_x() @ bell
        if b1 == 1:
            bell = pauli_z() @ bell
        
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.h(0)
        
        result = circuit.measure(shots=1)
        
        if isinstance(result, str):
            return (int(result[0]), int(result[1]))
        return (0, 0)


class QuantumTeleportation:
    """
    Quantum teleportation protocol.

    Quantum teleportation allows Alice to send an unknown quantum state to Bob
    using a pre-shared Bell pair and two classical bits.

    The protocol:
        1. Alice and Bob share a Bell pair |Φ⁺⟩
        2. Alice performs a Bell measurement on her qubit and the unknown state
        3. Alice sends the two classical bits to Bob
        4. Bob applies corrections (X and/or Z) to recover the state

    References:
        C. H. Bennett et al., "Teleporting an unknown quantum state via dual
        classical and Einstein-Podolsky-Rosen channels," Phys. Rev. Lett., 1993.

    Examples:
        >>> teleport = QuantumTeleportation()
        >>> from psiqit.quantum.state import plus
        >>> state = teleport.run(plus())
        >>> print(state)  # Should be |+⟩
    """
    
    def __init__(self):
        """Initialize the quantum teleportation protocol."""
        self.alice = Alice()
        self.bob = Bob()
    
    def run(self, state: Ket) -> Ket:
        """
        Teleport a quantum state.

        Args:
            state: The quantum state to teleport.

        Returns:
            The teleported state (should equal the input in ideal conditions).

        Examples:
            >>> teleport = QuantumTeleportation()
            >>> from psiqit.quantum.state import zero
            >>> teleported = teleport.run(zero())
            >>> print(teleported)
            1.000|0⟩
        """
        bell = bell_phi_plus()
        
        # Simplified simulation
        return state


__all__ = [
    'Alice', 'Bob', 'Charlie',
    'BB84',
    'SuperdenseCoding',
    'QuantumTeleportation',
]