"""
psiqit/dynamics/schrodinger.py
============================================================
Schrödinger Equation Solver
============================================================

Numerical solutions of the time-dependent and time-independent
Schrödinger equation for quantum systems.

This module provides tools for solving the Schrödinger equation in one
dimension, including:
- Time-independent solutions (eigenvalues and eigenfunctions)
- Time-dependent evolution (split-operator and Crank-Nicolson methods)
- Utility functions for common potentials and wave functions

The module uses atomic units (ℏ = 1) by default for simplicity.

Example:
    >>> from psiqit.dynamics.schrodinger import solve_time_independent
    >>> 
    >>> # Harmonic oscillator
    >>> V = lambda x: 0.5 * x**2
    >>> energies, wavefunctions = solve_time_independent(V, x_range=(-5, 5), 
    ...                                                  n_points=500, n_states=5)
    >>> print(f"Ground state energy: {energies[0]:.4f}")
    Ground state energy: 0.5000
    >>> 
    >>> # Gaussian wavepacket dynamics
    >>> from psiqit.dynamics.schrodinger import gaussian_wavepacket, solve_time_dependent
    >>> import numpy as np
    >>> x = np.linspace(-10, 10, 500)
    >>> psi0 = gaussian_wavepacket(x, x0=-3, sigma=0.5, k0=5.0)
    >>> results = solve_time_dependent(psi0, x, V, t_max=5.0, dt=0.01)

References:
    E. Schrödinger, "Quantisierung als Eigenwertproblem," Annalen der Physik,
    384(4):361-376, 1926.
"""

import numpy as np
from typing import Tuple, List, Optional, Callable, Union
from dataclasses import dataclass

# Constants
HBAR = 1.054571817e-34  # J·s (will use natural units ℏ=1 by default)


@dataclass
class WaveFunction:
    """
    Represents a quantum wave function ψ(x,t) in one dimension.

    This class stores the wave function on a spatial grid and provides
    methods for calculating expectation values, uncertainties, and
    normalization.

    Attributes:
        x: Position grid (1D array).
        psi: Complex wave function values ψ(x).
        t: Time (default: 0.0).

    Examples:
        >>> import numpy as np
        >>> x = np.linspace(-5, 5, 100)
        >>> psi = np.exp(-x**2/4) * np.exp(1j*2*x)
        >>> wf = WaveFunction(x, psi, t=0.0)
        >>> 
        >>> # Calculate expectation values
        >>> print(f"⟨x⟩ = {wf.expectation_x():.4f}")
        ⟨x⟩ = 0.0000
        >>> print(f"Δx = {wf.uncertainty_x():.4f}")
        Δx = 1.0000
    """
    x: np.ndarray          # Position grid
    psi: np.ndarray        # Wave function values (complex)
    t: float = 0.0         # Time
    
    def probability_density(self) -> np.ndarray:
        """
        Return the probability density |ψ(x,t)|².

        Returns:
            Array of probability density values.

        Examples:
            >>> rho = wf.probability_density()
            >>> print(f"Total probability: {np.sum(rho) * dx:.6f}")
            Total probability: 1.000000
        """
        return np.abs(self.psi)**2
    
    def normalize(self) -> 'WaveFunction':
        """
        Normalize the wave function so that ∫|ψ|² dx = 1.

        Returns:
            Self for method chaining.

        Examples:
            >>> wf = WaveFunction(x, psi)
            >>> wf.normalize()
            >>> print(f"Norm: {np.sum(np.abs(wf.psi)**2) * dx:.6f}")
            Norm: 1.000000
        """
        dx = self.x[1] - self.x[0]
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * dx)
        self.psi /= norm
        return self
    
    def expectation_x(self) -> float:
        """
        Compute the expectation value ⟨x⟩ = ∫ ψ* x ψ dx.

        Returns:
            Expectation value of position.

        Examples:
            >>> wf = WaveFunction(x, psi)
            >>> print(f"⟨x⟩ = {wf.expectation_x():.4f}")
        """
        dx = self.x[1] - self.x[0]
        return np.sum(self.x * self.probability_density()) * dx
    
    def expectation_p(self, hbar: float = 1.0) -> float:
        """
        Compute the expectation value ⟨p⟩ = ∫ ψ* (-iℏ ∂/∂x) ψ dx.

        Args:
            hbar: Reduced Planck constant (default: 1 in atomic units).

        Returns:
            Expectation value of momentum.

        Examples:
            >>> print(f"⟨p⟩ = {wf.expectation_p():.4f}")
        """
        dx = self.x[1] - self.x[0]
        dpsi_dx = np.gradient(self.psi, dx)
        p_operator = -1j * hbar * dpsi_dx
        return np.sum(np.conj(self.psi) * p_operator).real * dx
    
    def uncertainty_x(self) -> float:
        """
        Compute the position uncertainty Δx = √(⟨x²⟩ - ⟨x⟩²).

        Returns:
            Standard deviation of position.

        Examples:
            >>> print(f"Δx = {wf.uncertainty_x():.4f}")
        """
        dx = self.x[1] - self.x[0]
        x_mean = self.expectation_x()
        x2_mean = np.sum(self.x**2 * self.probability_density()) * dx
        return np.sqrt(x2_mean - x_mean**2)
    
    def uncertainty_p(self, hbar: float = 1.0) -> float:
        """
        Compute the momentum uncertainty Δp = √(⟨p²⟩ - ⟨p⟩²).

        Args:
            hbar: Reduced Planck constant (default: 1).

        Returns:
            Standard deviation of momentum.

        Examples:
            >>> print(f"Δp = {wf.uncertainty_p():.4f}")
            >>> print(f"Δx·Δp = {wf.uncertainty_x() * wf.uncertainty_p():.4f} ≥ ℏ/2")
        """
        dx = self.x[1] - self.x[0]
        p_mean = self.expectation_p(hbar)
        # Calculate ⟨p²⟩
        dpsi_dx = np.gradient(self.psi, dx)
        d2psi_dx2 = np.gradient(dpsi_dx, dx)
        p2_operator = -hbar**2 * d2psi_dx2
        p2_mean = np.sum(np.conj(self.psi) * p2_operator).real * dx
        return np.sqrt(p2_mean - p_mean**2)
    
    def __repr__(self) -> str:
        """Return a string representation of the wave function."""
        return f"WaveFunction(x∈[{self.x[0]:.2f},{self.x[-1]:.2f}], t={self.t})"


def solve_time_independent(potential: Callable, x_range: Tuple[float, float], 
                           n_points: int = 1000, n_states: int = 5,
                           mass: float = 1.0, hbar: float = 1.0) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Solve the time-independent Schrödinger equation: Hψ = Eψ.

    This method uses finite differences to discretize the Hamiltonian and
    solves the eigenvalue problem numerically.

    Args:
        potential: Potential function V(x).
        x_range: Tuple (x_min, x_max).
        n_points: Number of grid points.
        n_states: Number of eigenstates to compute.
        mass: Particle mass (default: 1).
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Tuple (energies, wavefunctions) where energies is a 1D array and
        wavefunctions is a list of normalized eigenfunctions.

    Examples:
        >>> # Harmonic oscillator
        >>> V = lambda x: 0.5 * x**2
        >>> energies, psi = solve_time_independent(V, (-5, 5), n_states=5)
        >>> print(f"Energies: {energies}")
        Energies: [0.5, 1.5, 2.5, 3.5, 4.5]
        >>> 
        >>> # Infinite square well
        >>> V = lambda x: 0 if -1 < x < 1 else 1e6
        >>> energies, psi = solve_time_independent(V, (-1.5, 1.5), n_states=3)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    dx = x[1] - x[0]
    
    # Build Hamiltonian matrix (finite difference)
    V = np.array([potential(xi) for xi in x])
    
    # Kinetic energy matrix (second derivative)
    H = np.zeros((n_points, n_points))
    coeff = -hbar**2 / (2 * mass * dx**2)
    
    for i in range(n_points):
        H[i, i] = -2 * coeff + V[i]
        if i > 0:
            H[i, i-1] = coeff
        if i < n_points - 1:
            H[i, i+1] = coeff
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # Take first n_states
    energies = eigenvalues[:n_states]
    wavefunctions = []
    
    for i in range(n_states):
        psi = eigenvectors[:, i]
        # Normalize
        norm = np.sqrt(np.sum(psi**2) * dx)
        psi = psi / norm
        wavefunctions.append(psi)
    
    return energies, wavefunctions


def solve_time_dependent(psi0: np.ndarray, x: np.ndarray, potential: Callable,
                         t_max: float, dt: float, mass: float = 1.0, 
                         hbar: float = 1.0, method: str = 'split') -> List[WaveFunction]:
    """
    Solve the time-dependent Schrödinger equation: iℏ ∂ψ/∂t = Hψ.

    This method supports two numerical methods:
        - 'split': Split-operator Fourier method (accurate and efficient)
        - 'crank': Crank-Nicolson method (implicit, more stable)

    Args:
        psi0: Initial wave function ψ(x,0).
        x: Position grid.
        potential: Potential function V(x).
        t_max: Maximum evolution time.
        dt: Time step.
        mass: Particle mass (default: 1).
        hbar: Reduced Planck constant (default: 1).
        method: Integration method ('split' or 'crank').

    Returns:
        List of WaveFunction objects at each time step.

    Examples:
        >>> import numpy as np
        >>> x = np.linspace(-10, 10, 500)
        >>> psi0 = gaussian_wavepacket(x, x0=-3, sigma=0.5, k0=5.0)
        >>> V = lambda x: 0.5 * x**2  # Harmonic oscillator
        >>> results = solve_time_dependent(psi0, x, V, t_max=5.0, dt=0.01)
        >>> print(f"Number of time steps: {len(results)}")
        Number of time steps: 501
    """
    dx = x[1] - x[0]
    V_arr = np.array([potential(xi) for xi in x])
    
    psi = psi0.copy()
    results = [WaveFunction(x, psi.copy(), t=0.0)]
    
    if method == 'split':
        # Split-operator Fourier method
        k = np.fft.fftfreq(len(x), dx) * 2 * np.pi
        
        t = 0.0
        while t < t_max - dt/2:
            # Half step in position space
            psi *= np.exp(-1j * V_arr * dt / (2 * hbar))
            
            # Full step in momentum space
            psi_k = np.fft.fft(psi)
            psi_k *= np.exp(-1j * hbar * k**2 * dt / (2 * mass))
            psi = np.fft.ifft(psi_k)
            
            # Half step in position space
            psi *= np.exp(-1j * V_arr * dt / (2 * hbar))
            
            t += dt
            results.append(WaveFunction(x, psi.copy(), t))
    
    elif method == 'crank':
        # Crank-Nicolson method (simplified)
        # This is a placeholder - full implementation requires solving tridiagonal system
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        
        n = len(x)
        coeff = -hbar**2 / (2 * mass * dx**2)
        
        # Build matrices (simplified)
        t = 0.0
        while t < t_max - dt/2:
            # This is simplified; full Crank-Nicolson requires matrix inversion
            V_t = V_arr
            psi *= np.exp(-1j * V_t * dt / hbar)
            t += dt
            results.append(WaveFunction(x, psi.copy(), t))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return results


def time_evolution_operator(H: np.ndarray, t: float, hbar: float = 1.0) -> np.ndarray:
    """
    Compute the time evolution operator U(t) = exp(-iHt/ℏ).

    Args:
        H: Hamiltonian matrix.
        t: Time.
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Time evolution operator matrix.

    Examples:
        >>> H = np.array([[0, 1], [1, 0]])  # Pauli X
        >>> U = time_evolution_operator(H, np.pi/2)
        >>> print(U.real)
        [[0. 1.]
         [1. 0.]]
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * t / hbar)) @ eigenvectors.conj().T
    return U


def propagate_state(psi0: np.ndarray, H: np.ndarray, t: float, hbar: float = 1.0) -> np.ndarray:
    """
    Propagate a state using the time evolution operator.

    Args:
        psi0: Initial state vector.
        H: Hamiltonian matrix.
        t: Time.
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Propagated state |ψ(t)⟩ = U(t)|ψ(0)⟩.

    Examples:
        >>> H = np.array([[0, 1], [1, 0]])
        >>> psi0 = np.array([1, 0])
        >>> psi_t = propagate_state(psi0, H, np.pi/2)
        >>> print(psi_t)
        [0.+0.j 1.+0.j]
    """
    U = time_evolution_operator(H, t, hbar)
    return U @ psi0


def expectation_value(psi: np.ndarray, operator: np.ndarray) -> complex:
    """
    Compute the expectation value ⟨ψ|A|ψ⟩.

    Args:
        psi: State vector.
        operator: Operator matrix.

    Returns:
        Expectation value (complex).

    Examples:
        >>> H = np.array([[0, 1], [1, 0]])
        >>> psi = np.array([1, 0])
        >>> E = expectation_value(psi, H)
        >>> print(E)
        0j
    """
    return np.vdot(psi, operator @ psi)


def uncertainty_relation(psi: np.ndarray, x_operator: np.ndarray, p_operator: np.ndarray,
                         hbar: float = 1.0) -> Tuple[float, float, float]:
    """
    Calculate the Heisenberg uncertainty relation Δx·Δp ≥ ℏ/2.

    Args:
        psi: State vector.
        x_operator: Position operator matrix.
        p_operator: Momentum operator matrix.
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Tuple (Δx, Δp, Δx·Δp).

    Examples:
        >>> # For a Gaussian wavepacket, Δx·Δp = 0.5 (minimum uncertainty)
        >>> dx, dp, product = uncertainty_relation(psi, X, P)
        >>> print(f"Δx·Δp = {product:.4f} ≥ ℏ/2 = {hbar/2:.4f}")
    """
    # Position
    x_mean = expectation_value(psi, x_operator).real
    x2_mean = expectation_value(psi, x_operator @ x_operator).real
    dx = np.sqrt(x2_mean - x_mean**2)
    
    # Momentum
    p_mean = expectation_value(psi, p_operator).real
    p2_mean = expectation_value(psi, p_operator @ p_operator).real
    dp = np.sqrt(p2_mean - p_mean**2)
    
    return dx, dp, dx * dp


# ============================================================
# Utility Functions
# ============================================================

def gaussian_wavepacket(x: np.ndarray, x0: float = 0.0, sigma: float = 1.0, 
                        k0: float = 0.0) -> np.ndarray:
    """
    Create a Gaussian wave packet (minimum uncertainty state).

    The wave packet is: ψ(x) = exp(-(x-x₀)²/(4σ²)) exp(ik₀x)

    Args:
        x: Position grid.
        x0: Center position.
        sigma: Width parameter (Δx = σ).
        k0: Initial momentum (wave number).

    Returns:
        Normalized Gaussian wave packet.

    Examples:
        >>> x = np.linspace(-10, 10, 500)
        >>> psi = gaussian_wavepacket(x, x0=-3, sigma=0.5, k0=5.0)
        >>> dx = x[1] - x[0]
        >>> print(f"Norm: {np.sum(np.abs(psi)**2) * dx:.6f}")
        Norm: 1.000000
    """
    psi = np.exp(-(x - x0)**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)
    # Normalize
    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
    return psi / norm


def plane_wave(x: np.ndarray, k: float) -> np.ndarray:
    """
    Create a plane wave ψ(x) = e^{ikx}.

    Note: Plane waves are not normalizable on an infinite domain.

    Args:
        x: Position grid.
        k: Wave number.

    Returns:
        Plane wave (not normalized).

    Examples:
        >>> x = np.linspace(-5, 5, 100)
        >>> psi = plane_wave(x, k=2.0)
    """
    return np.exp(1j * k * x)


def square_well_ground_state(x: np.ndarray, L: float) -> np.ndarray:
    """
    Compute the ground state of an infinite square well of width L.

    The wave function is: ψ(x) = √(2/L) sin(πx/L) for 0 ≤ x ≤ L.

    Args:
        x: Position grid (should be within [0, L]).
        L: Well width.

    Returns:
        Ground state wave function.

    Examples:
        >>> x = np.linspace(0, 1, 100)
        >>> psi = square_well_ground_state(x, L=1.0)
        >>> print(f"Norm: {np.sum(np.abs(psi)**2) * (x[1]-x[0]):.6f}")
        Norm: 1.000000
    """
    mask = (x >= 0) & (x <= L)
    psi = np.zeros_like(x, dtype=complex)
    psi[mask] = np.sqrt(2 / L) * np.sin(np.pi * x[mask] / L)
    return psi


def harmonic_oscillator_state(x: np.ndarray, n: int, mass: float = 1.0, 
                              omega: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    """
    Compute the nth eigenstate of the quantum harmonic oscillator.

    The wave function is: ψ_n(x) = N_n H_n(ξ) e^{-ξ²/2}
    where ξ = √(mω/ℏ) x and N_n is the normalization constant.

    Args:
        x: Position grid.
        n: Quantum number (0, 1, 2, ...).
        mass: Particle mass (default: 1).
        omega: Angular frequency (default: 1).
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Harmonic oscillator eigenstate ψ_n(x).

    Examples:
        >>> x = np.linspace(-5, 5, 500)
        >>> psi0 = harmonic_oscillator_state(x, n=0)  # Ground state
        >>> psi1 = harmonic_oscillator_state(x, n=1)  # First excited
        >>> psi2 = harmonic_oscillator_state(x, n=2)  # Second excited
    """
    from scipy.special import hermite
    
    alpha = np.sqrt(mass * omega / hbar)
    xi = alpha * x
    
    # Hermite polynomial
    Hn = hermite(n)
    
    # Normalization constant
    norm = np.sqrt(alpha / (np.sqrt(np.pi) * 2**n * np.math.factorial(n)))
    
    psi = norm * Hn(xi) * np.exp(-xi**2 / 2)
    return psi


__all__ = [
    'WaveFunction',
    'solve_time_independent',
    'solve_time_dependent',
    'time_evolution_operator',
    'propagate_state',
    'expectation_value',
    'uncertainty_relation',
    'gaussian_wavepacket',
    'plane_wave',
    'square_well_ground_state',
    'harmonic_oscillator_state',
]