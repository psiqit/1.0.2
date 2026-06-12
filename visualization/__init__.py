"""
psiqit/visualization/__init__.py
============================================================
Visualization Package - Quantum State and Circuit Visualization
============================================================

This package provides visualization tools for quantum computing:
    • Bloch sphere visualization
    • Quantum circuit drawing (ASCII/Unicode)
    • Wigner function and Husimi Q-function plots
    • Quantum animations
    • Symbolic equation rendering

Example:
    >>> from psiqit.visualization import bloch_sphere, draw_circuit, plot_wigner
    >>> from psiqit.circuits.circuit import QuantumCircuit
    >>> from psiqit.quantum.state import plus
    >>> 
    >>> # Plot Bloch sphere
    >>> bloch_sphere(plus())
    >>> 
    >>> # Draw quantum circuit
    >>> circ = QuantumCircuit(2)
    >>> circ.h(0)
    >>> circ.cx(0, 1)
    >>> print(draw_circuit(circ))
"""

# ============================================================
# Bloch Sphere Visualization
# ============================================================

from .bloch import (
    state_to_bloch,
    bloch_to_state,
    spherical_to_bloch,
    bloch_to_spherical,
    bloch_sphere,
    plot_multiple_states,
    animate_bloch,
)

# ============================================================
# Circuit Drawing
# ============================================================

from .circuit_drawer import (
    draw_circuit,
    circuit_to_text,
    circuit_statistics,
)

# ============================================================
# Wigner Function
# ============================================================

from .wigner import (
    wigner_function,
    wigner_function_analytic,
    wigner_function_gaussian,
    wigner_function_coherent_state,
    wigner_function_squeezed_state,
    plot_wigner,
    plot_wigner_3d,
    wigner_negativity,
)

# ============================================================
# Husimi Q-Function
# ============================================================

from .husimi import (
    husimi_function_gaussian,
    husimi_function_coherent_state,
    plot_husimi,
)

# ============================================================
# Animation
# ============================================================

from .animation import (
    animate_wavefunction,
    animate_bloch_sphere,
)

# ============================================================
# Symbolic Equation Plotting
# ============================================================

from .symbolic_plot import (
    render_equation,
    schrodinger_equation,
    dirac_notation,
    pauli_matrices,
    bloch_sphere_equation,
    heisenberg_uncertainty,
    commutation_relation,
    maxwell_equations,
)

# ============================================================
# Package Metadata
# ============================================================

__version__ = "1.0.0"
__author__ = "PSIQIT Contributors"

__all__ = [
    # Bloch
    'state_to_bloch',
    'bloch_to_state',
    'spherical_to_bloch',
    'bloch_to_spherical',
    'bloch_sphere',
    'plot_multiple_states',
    'animate_bloch',
    # Circuit drawing
    'draw_circuit',
    'circuit_to_text',
    'circuit_statistics',
    # Wigner
    'wigner_function',
    'wigner_function_analytic',
    'wigner_function_gaussian',
    'wigner_function_coherent_state',
    'wigner_function_squeezed_state',
    'plot_wigner',
    'plot_wigner_3d',
    'wigner_negativity',
    # Husimi
    'husimi_function_gaussian',
    'husimi_function_coherent_state',
    'plot_husimi',
    # Animation
    'animate_wavefunction',
    'animate_bloch_sphere',
    # Symbolic
    'render_equation',
    'schrodinger_equation',
    'dirac_notation',
    'pauli_matrices',
    'bloch_sphere_equation',
    'heisenberg_uncertainty',
    'commutation_relation',
    'maxwell_equations',
]