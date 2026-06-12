"""
psiqit/quantum/__init__.py
Quantum mechanics foundations.

This module provides the core building blocks for quantum computing:
    • Quantum states (Ket, Bra) and state manipulation
    • Quantum operators (gates and observables)
    • Measurements and POVMs

The quantum module is the foundation of the PSIQIT framework, providing
the essential tools for representing and manipulating quantum systems.

Submodules:
    - state: Quantum states (Ket, Bra) and common states (|0⟩, |1⟩, |+⟩, Bell, GHZ)
    - operator: Quantum operators (Pauli gates, Hadamard, rotations, multi-qubit gates)
    - measurement: Measurement functions, POVMs, and tomography

Example:
    >>> from psiqit.quantum import Ket, zero, one, hadamard, cnot, measure
    >>> 
    >>> # Create Bell state
    >>> circuit = QuantumCircuit(2)
    >>> circuit.h(0)
    >>> circuit.cx(0, 1)
    >>> state = circuit.run()
    >>> print(state)
    0.707|00⟩ + 0.707|11⟩
"""

from .state import *
from .operator import *
from .measurement import *

__all__ = [
    # From state
    'Ket', 'Bra', 'ket', 'basis',
    'zero', 'one', 'plus', 'minus', 'ip', 'im',
    'bell_phi_plus', 'bell_phi_minus',
    'bell_psi_plus', 'bell_psi_minus', 'bell_state',
    'ghz', 'w_state', 'random_state',
    'is_orthogonal', 'is_same', 'fidelity',
    # From operator
    'Operator',
    'pauli_x', 'pauli_y', 'pauli_z', 'identity',
    'hadamard', 'phase', 's_gate', 't_gate',
    'rx', 'ry', 'rz',
    'cnot', 'cz', 'swap', 'toffoli', 'fredkin',
    'tensor_product', 'expectation',
    # From measurement
    'measure', 'measure_observable', 'variance', 'standard_deviation',
    'POVM', 'povm_z_basis', 'povm_x_basis', 'povm_y_basis',
    'ProjectiveMeasurement',
    'state_tomography', 'born_rule', 'measurement_statistics',
]