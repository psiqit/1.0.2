"""
psiqit/algorithms/__init__.py
Quantum algorithms module
"""

from .grover import Grover, grover_search, grover_search_multiple
from .deutsch_jozsa import (
    DeutschJozsaResult,
    DeutschJozsa,
    DeutschAlgorithm,
    deutsch_jozsa_constant,
    deutsch_jozsa_balanced,
    deutsch_algorithm_constant,
    deutsch_algorithm_balanced,
)

__all__ = [
    # Grover
    'Grover',
    'grover_search',
    'grover_search_multiple',
    # Deutsch-Jozsa
    'DeutschJozsaResult',
    'DeutschJozsa',
    'DeutschAlgorithm',
    'deutsch_jozsa_constant',
    'deutsch_jozsa_balanced',
    'deutsch_algorithm_constant',
    'deutsch_algorithm_balanced',
]