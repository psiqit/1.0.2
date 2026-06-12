"""
psiqit/quantum/interference.py
============================================================
Quantum Interference
============================================================

Simulation of quantum interference phenomena:
    • Double-slit experiment
    • Mach-Zehnder interferometer
    • Interference patterns

This module provides classical and quantum simulations of fundamental
interference experiments that demonstrate the wave-particle duality
of quantum mechanics.

The double-slit experiment shows how quantum particles interfere with
themselves, while the Mach-Zehnder interferometer is a key device in
quantum optics and quantum information processing.

Example:
    >>> from psiqit.quantum.interference import DoubleSlit, MachZehnderInterferometer
    >>> import math
    >>> 
    >>> # Double-slit experiment
    >>> double_slit = DoubleSlit(slit_distance=1.0, slit_width=0.2, wavelength=0.5)
    >>> x, intensity, visibility, spacing = double_slit.pattern(x_range=(-0.1, 0.1), n_points=500)
    >>> print(f"Fringe visibility: {visibility:.3f}")
    >>> 
    >>> # Mach-Zehnder interferometer
    >>> mz = MachZehnderInterferometer()
    >>> mz.set_phase(math.pi/2)
    >>> prob0 = mz.output_probability(0)
    >>> prob1 = mz.output_probability(1)
    >>> print(f"P0={prob0:.3f}, P1={prob1:.3f}")

References:
    R. P. Feynman, R. B. Leighton, and M. Sands, "The Feynman Lectures on Physics,"
    Vol. III, Quantum Mechanics, 1965.
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Fix imports
from ..circuits.circuit import QuantumCircuit
from .state import Ket
from .operator import hadamard, rz


@dataclass
class InterferenceResult:
    """
    Result container for interference calculations.

    Attributes:
        x: List of positions on the screen.
        intensity: List of normalized intensity values.
        visibility: Interference fringe visibility (0 to 1).
        fringe_spacing: Distance between adjacent fringes.

    Examples:
        >>> result = InterferenceResult(x=[-0.1, 0, 0.1], intensity=[0.2, 1.0, 0.2],
        ...                             visibility=0.8, fringe_spacing=0.05)
        >>> print(f"Visibility: {result.visibility:.3f}")
    """
    x: List[float]
    intensity: List[float]
    visibility: float
    fringe_spacing: float


class DoubleSlit:
    """
    Double-slit experiment simulation.

    This class simulates the classical interference pattern from a double-slit
    setup, including both single-slit diffraction and double-slit interference.

    The intensity pattern is given by:
        I(θ) = I₀ [sin(β)/β]² [cos(δ/2)]²
    where β = (π a sin θ)/λ and δ = (2π d sin θ)/λ

    Attributes:
        d: Distance between slits.
        a: Width of each slit.
        lam: Wavelength of light.
        L: Distance from slits to screen.
        k: Wave number (2π/λ).

    Examples:
        >>> # Create a double-slit experiment
        >>> ds = DoubleSlit(slit_distance=1.0e-3, slit_width=0.1e-3,
        ...                 wavelength=632.8e-9, distance_to_screen=1.0)
        >>> x, I, vis, spacing = ds.pattern(x_range=(-0.05, 0.05), n_points=1000)
        >>> 
        >>> # Plot the pattern
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(x, I)
        >>> plt.xlabel('Position (m)')
        >>> plt.ylabel('Intensity')
        >>> plt.title('Double-Slit Interference Pattern')
        >>> plt.show()
    """
    
    def __init__(self, slit_distance: float, slit_width: float, 
                 wavelength: float, distance_to_screen: float = 1.0):
        """
        Initialize the double-slit experiment.

        Args:
            slit_distance: Distance between slits (d).
            slit_width: Width of each slit (a).
            wavelength: Wavelength of light (λ).
            distance_to_screen: Distance from slits to screen (L).

        Examples:
            >>> ds = DoubleSlit(slit_distance=1.0e-3, slit_width=0.1e-3,
            ...                 wavelength=632.8e-9, distance_to_screen=1.0)
        """
        self.d = slit_distance
        self.a = slit_width
        self.lam = wavelength
        self.L = distance_to_screen
        self.k = 2 * math.pi / wavelength
    
    def _single_slit_diffraction(self, theta: float) -> float:
        """
        Compute the single-slit diffraction pattern.

        Args:
            theta: Angle from the optical axis.

        Returns:
            Normalized diffraction intensity.
        """
        if abs(theta) < 1e-10:
            return 1.0
        beta = self.k * self.a * math.sin(theta) / 2
        return (math.sin(beta) / beta) ** 2
    
    def _double_slit_interference(self, theta: float) -> float:
        """
        Compute the double-slit interference term.

        Args:
            theta: Angle from the optical axis.

        Returns:
            Interference intensity modulation.
        """
        if abs(theta) < 1e-10:
            return 1.0
        delta = self.k * self.d * math.sin(theta)
        return math.cos(delta / 2) ** 2
    
    def intensity(self, theta: float) -> float:
        """
        Compute total intensity at a given angle.

        Args:
            theta: Angle from the optical axis.

        Returns:
            Relative intensity (normalized).
        """
        return self._single_slit_diffraction(theta) * self._double_slit_interference(theta)
    
    def pattern(self, x_range: Tuple[float, float] = (-0.1, 0.1), 
                n_points: int = 500) -> Tuple[List[float], List[float], float, float]:
        """
        Calculate the interference pattern on the screen.

        Args:
            x_range: Tuple (x_min, x_max) for positions on the screen.
            n_points: Number of points to evaluate.

        Returns:
            Tuple of (x_positions, intensities, visibility, fringe_spacing).

        Examples:
            >>> x, I, vis, spacing = ds.pattern(x_range=(-0.05, 0.05), n_points=1000)
            >>> print(f"Fringe spacing: {spacing:.6f} m")
        """
        x_min, x_max = x_range
        x = [x_min + i * (x_max - x_min) / (n_points - 1) for i in range(n_points)]
        
        intensities = []
        for xi in x:
            theta = math.atan(xi / self.L)
            intensities.append(self.intensity(theta))
        
        # Normalize
        max_int = max(intensities)
        if max_int > 0:
            intensities = [i / max_int for i in intensities]
        
        # Calculate fringe spacing
        fringe_spacing = self._calculate_fringe_spacing(x, intensities)
        
        # Calculate visibility
        max_intensity = max(intensities)
        min_intensity = min(intensities)
        visibility = (max_intensity - min_intensity) / (max_intensity + min_intensity) if max_intensity + min_intensity > 0 else 0
        
        return x, intensities, visibility, fringe_spacing
    
    def _calculate_fringe_spacing(self, x: List[float], intensities: List[float]) -> float:
        """
        Calculate fringe spacing from the intensity pattern.

        Args:
            x: List of positions.
            intensities: List of intensity values.

        Returns:
            Average distance between adjacent fringes.
        """
        if len(x) < 3:
            return 0.0
        
        # Find peaks
        peaks = []
        for i in range(1, len(intensities) - 1):
            if intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1]:
                peaks.append(x[i])
        
        if len(peaks) >= 2:
            return peaks[1] - peaks[0]
        return 0.0
    
    def quantum_probability(self, position: int, n_qubits: int = 8) -> float:
        """
        Quantum circuit simulation of the double-slit experiment.

        This method uses a quantum circuit to simulate a particle passing
        through a double-slit, creating a superposition of paths.

        Args:
            position: Position index on the screen (0 to 2ⁿ - 1).
            n_qubits: Number of qubits for position basis.

        Returns:
            Probability of detection at that position.

        Examples:
            >>> prob = ds.quantum_probability(position=100, n_qubits=10)
            >>> print(f"Probability: {prob:.4f}")
        """
        dim = 2 ** n_qubits
        # Create superposition of paths
        amplitudes = [1.0 / math.sqrt(dim)] * dim
        
        # Add phase from path difference
        phase = 2 * math.pi * position / dim
        amplitudes[position] *= math.cos(phase)
        
        # Normalize
        norm = math.sqrt(sum(a**2 for a in amplitudes))
        if norm > 0:
            amplitudes = [a / norm for a in amplitudes]
        
        return amplitudes[position] ** 2


class MachZehnderInterferometer:
    """
    Mach-Zehnder interferometer simulation.

    The Mach-Zehnder interferometer is a fundamental device in quantum optics.
    It consists of two beam splitters and two mirrors, with a phase shifter
    in one arm. This class simulates its quantum behavior using a single qubit.

    The quantum circuit equivalent uses:
        - Hadamard gate for the first beam splitter
        - RZ gate for the phase shifter
        - Hadamard gate for the second beam splitter

    Examples:
        >>> mz = MachZehnderInterferometer()
        >>> 
        >>> # Scan phase and record probabilities
        >>> phases = [i * math.pi/50 for i in range(101)]
        >>> probs = []
        >>> for phi in phases:
        ...     mz.set_phase(phi)
        ...     probs.append(mz.output_probability(0))
        >>> 
        >>> # Plot interference fringe
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(phases, probs)
        >>> plt.xlabel('Phase Shift (rad)')
        >>> plt.ylabel('Probability')
        >>> plt.title('Mach-Zehnder Interference')
        >>> plt.show()
    """
    
    def __init__(self):
        """Initialize the Mach-Zehnder interferometer with zero phase shift."""
        self._phase_shift = 0.0
    
    def set_phase(self, phi: float):
        """
        Set the phase shift in one arm of the interferometer.

        Args:
            phi: Phase shift in radians.

        Examples:
            >>> mz.set_phase(math.pi)   # π phase shift
            >>> mz.set_phase(math.pi/2) # π/2 phase shift
        """
        self._phase_shift = phi
    
    def get_phase(self) -> float:
        """
        Get the current phase shift.

        Returns:
            Current phase shift in radians.
        """
        return self._phase_shift
    
    def quantum_circuit(self) -> QuantumCircuit:
        """
        Build the quantum circuit equivalent of the Mach-Zehnder interferometer.

        The circuit consists of:
            1. Hadamard (first beam splitter)
            2. RZ phase shift
            3. Hadamard (second beam splitter)

        Returns:
            QuantumCircuit: The interferometer circuit.
        """
        circuit = QuantumCircuit(1)
        
        # First beam splitter (Hadamard)
        circuit.h(0)
        
        # Phase shifter
        circuit.rz(0, self._phase_shift)
        
        # Second beam splitter (Hadamard)
        circuit.h(0)
        
        return circuit
    
    def get_state(self) -> Ket:
        """
        Get the quantum state after the interferometer.

        Returns:
            Ket: Final quantum state.
        """
        circuit = self.quantum_circuit()
        return circuit.run()
    
    def output_probability(self, output_port: int = 0) -> float:
        """
        Calculate the probability of the photon exiting a specific port.

        Args:
            output_port: 0 for transmitted port, 1 for reflected port.

        Returns:
            Probability between 0 and 1.

        Examples:
            >>> mz.set_phase(0)
            >>> print(mz.output_probability(0))  # Should be 1.0
            1.0
            >>> mz.set_phase(math.pi)
            >>> print(mz.output_probability(0))  # Should be 0.0
            0.0
        """
        state = self.get_state()
        
        if output_port == 0:
            return abs(state.data[0]) ** 2
        else:
            return abs(state.data[1]) ** 2
    
    def interference_fringe(self, phase_range: Tuple[float, float] = (0, 2*math.pi),
                           n_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Calculate the interference fringe pattern as a function of phase.

        Args:
            phase_range: Tuple (phase_min, phase_max) in radians.
            n_points: Number of points to evaluate.

        Returns:
            Tuple of (phase_values, probabilities) for the transmitted port.

        Examples:
            >>> phases, probs = mz.interference_fringe(phase_range=(0, 2*math.pi), n_points=200)
        """
        phases = [phase_range[0] + i * (phase_range[1] - phase_range[0]) / (n_points - 1) 
                  for i in range(n_points)]
        probs = []
        
        for phi in phases:
            self.set_phase(phi)
            probs.append(self.output_probability(0))
        
        return phases, probs
    
    def visibility(self, phase_range: Tuple[float, float] = (0, 2*math.pi),
                  n_points: int = 100) -> float:
        """
        Calculate the interference fringe visibility.

        Visibility is defined as: V = (I_max - I_min) / (I_max + I_min)

        Args:
            phase_range: Tuple (phase_min, phase_max) in radians.
            n_points: Number of points to evaluate.

        Returns:
            Visibility value between 0 and 1.
        """
        phases, probs = self.interference_fringe(phase_range, n_points)
        max_prob = max(probs)
        min_prob = min(probs)
        
        if max_prob + min_prob == 0:
            return 0.0
        return (max_prob - min_prob) / (max_prob + min_prob)


__all__ = [
    'InterferenceResult',
    'DoubleSlit',
    'MachZehnderInterferometer',
]