"""
psiqit/lab/__init__.py
============================================================
Virtual Lab Package
============================================================

This package provides a virtual quantum laboratory environment
for running experiments and simulating quantum circuits.

The Virtual Lab allows users to:
    - Create and manage quantum experiments
    - Add gates to circuits interactively
    - Run simulations and collect results
    - Access predefined experiments (Bell state, GHZ state)
    - Visualize results and state vectors

This is particularly useful for:
    - Teaching quantum computing concepts
    - Prototyping quantum circuits
    - Running batch experiments
    - Educational demonstrations

Example:
    >>> from psiqit.lab import QuantumLab, PredefinedExperiments
    >>> 
    >>> # Create a virtual lab
    >>> lab = QuantumLab()
    >>> 
    >>> # Create a custom experiment
    >>> exp = lab.create_experiment("My Experiment", n_qubits=2)
    >>> exp.add_gate("h", 0)
    >>> exp.add_gate("cx", 0, 1)
    >>> 
    >>> # Run the experiment
    >>> result = lab.run_experiment(exp)
    >>> print(result.summary())
    Experiment: My Experiment
    Time: 2024-01-01T12:00:00
    Qubits: 2, Shots: 1024
    Success rate: 50.00%
    Status: completed
    Measurement counts:
      |00⟩: 512 (50.00%)
      |11⟩: 512 (50.00%)
    >>> 
    >>> # Use predefined experiments
    >>> bell_exp = PredefinedExperiments.bell_state()
    >>> result = lab.run_experiment(bell_exp)

References:
    The Virtual Lab is inspired by quantum computing education platforms
    like IBM Quantum Experience and Quirk.
"""

from .virtual_lab import (
    ExperimentStatus,
    ExperimentResult,
    Experiment,
    QuantumLab,
    PredefinedExperiments,
)

__all__ = [
    'ExperimentStatus',
    'ExperimentResult',
    'Experiment',
    'QuantumLab',
    'PredefinedExperiments',
]

__version__ = "1.0.2"