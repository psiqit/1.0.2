"""
psiqit - a Quantum Information Toolkit
"""

__version__ = "1.0.1"
__author__ = "mahdi azadmarzabadi , Dr.mahdi mirzaee , Dr.karim ghorbani"

# Core exports
from .quantum.state import Ket, Bra, zero, one, plus, minus
from .quantum.operator import hadamard, pauli_x, pauli_y, pauli_z, cnot
from .circuits.circuit import QuantumCircuit

__all__ = [
    '__version__',
    '__author__',
    'Ket', 'Bra', 'zero', 'one', 'plus', 'minus',
    'hadamard', 'pauli_x', 'pauli_y', 'pauli_z', 'cnot',
    'QuantumCircuit',
]