"""
psiqit/dynamics/__init__.py
Quantum dynamics module

This module provides tools for simulating quantum dynamics, including
solving the Schrödinger equation (time-dependent and time-independent),
computing time evolution operators, and calculating expectation values
and uncertainty relations.

The module includes:
- WaveFunction class for representing and manipulating wave functions
- Numerical solvers for the Schrödinger equation
- Time evolution operators for finite-dimensional systems
- Utility functions for common quantum states (Gaussian wavepackets, etc.)

Example:
    >>> from psiqit.dynamics import solve_time_independent, gaussian_wavepacket
    >>> import numpy as np
    >>> 
    >>> # Solve for harmonic oscillator eigenstates
    >>> V = lambda x: 0.5 * x**2
    >>> energies, wavefunctions = solve_time_independent(V, x_range=(-5, 5), n_states=5)
    >>> print(f"Ground state energy: {energies[0]:.4f}")
    Ground state energy: 0.5000
    >>> 
    >>> # Create a Gaussian wavepacket
    >>> x = np.linspace(-10, 10, 500)
    >>> psi = gaussian_wavepacket(x, x0=-3, sigma=0.5, k0=5.0)
"""

from .schrodinger import (
    WaveFunction,
    solve_time_independent,
    solve_time_dependent,
    time_evolution_operator,
    propagate_state,
    expectation_value,
    uncertainty_relation,
    gaussian_wavepacket,
    plane_wave,
    square_well_ground_state,
    harmonic_oscillator_state,
)

__all__ = [
    # Wave function class
    'WaveFunction',
    # Schrödinger equation solvers
    'solve_time_independent',
    'solve_time_dependent',
    # Time evolution
    'time_evolution_operator',
    'propagate_state',
    # Quantum mechanics utilities
    'expectation_value',
    'uncertainty_relation',
    # Common wave functions
    'gaussian_wavepacket',
    'plane_wave',
    'square_well_ground_state',
    'harmonic_oscillator_state',
]