"""
psiqit/interface/__init__.py
============================================================
Interface Package - CLI, LaTeX, and Jupyter Notebook Tools
============================================================

This package provides interfaces for interacting with PSIQIT:
- Command Line Interface (CLI)
- LaTeX output generation
- Jupyter notebook tools

The interface module makes PSIQIT accessible from different environments:
    - Terminal/command line for quick simulations and batch processing
    - LaTeX documents for generating publication-ready equations and circuits
    - Jupyter notebooks for interactive quantum computing tutorials and research

Example:
    >>> # CLI usage (from terminal)
    >>> $ python -m psiqit.interface.cli circuit --n-qubits 2 --gates h0,cx01 --shots 100
    >>> 
    >>> # LaTeX generation
    >>> from psiqit.interface.latex import matrix_to_latex, circuit_to_latex
    >>> H = [[1, 1], [1, -1]]
    >>> print(matrix_to_latex(H, name="H"))
    >>> 
    >>> # Jupyter notebook
    >>> from psiqit.interface.notebook import init_notebook, display_circuit
    >>> init_notebook()
    >>> display_circuit(circuit)
"""

from .cli import main as cli_main, parse_gates, run_circuit, run_algorithm
from .latex import (
    matrix_to_latex,
    state_to_latex,
    circuit_to_latex,
    equation_to_latex,
    table_to_latex,
    generate_report,
)
from .notebook import (
    init_notebook,
    display_circuit,
    display_state,
    display_histogram,
    display_bloch,
    display_matrix,
    display_notebook_version,
)

__all__ = [
    # CLI
    'cli_main',
    'parse_gates',
    'run_circuit',
    'run_algorithm',
    # LaTeX
    'matrix_to_latex',
    'state_to_latex',
    'circuit_to_latex',
    'equation_to_latex',
    'table_to_latex',
    'generate_report',
    # Notebook
    'init_notebook',
    'display_circuit',
    'display_state',
    'display_histogram',
    'display_bloch',
    'display_matrix',
    'display_notebook_version',
]

__version__ = "1.0.2"
__author__ = "PSIQIT Contributors"