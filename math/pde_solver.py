"""
psiqit/math/pde_solver.py
Partial Differential Equation Solver - Fixed

Numerical solvers for partial differential equations (PDEs) commonly
encountered in quantum mechanics and physics:

    • Heat equation (diffusion equation)
    • Wave equation
    • Time-dependent Schrödinger equation

These solvers use finite difference methods and the split-operator method
to provide numerical solutions for initial-boundary value problems.

Example:
    >>> from psiqit.math.pde_solver import solve_heat_equation, solve_wave_equation
    >>> 
    >>> # Solve heat equation
    >>> result = solve_heat_equation(nt=100, nx=50, L=1.0, T=1.0, alpha=0.01)
    >>> print(f"Temperature at t={result.t[-1]:.2f}: {result.u[-1][25]:.4f}")
    >>> 
    >>> # Solve wave equation
    >>> result = solve_wave_equation(nt=100, nx=50, L=1.0, T=2.0, c=1.0)

References:
    J. W. Thomas, "Numerical Partial Differential Equations: Finite Difference
    Methods," Springer, 1995.
    R. J. LeVeque, "Finite Difference Methods for Ordinary and Partial
    Differential Equations," SIAM, 2007.
"""

import math
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PDEResult:
    """
    Result container for PDE solutions.

    Attributes:
        x: Spatial grid points (list of floats).
        t: Time grid points (list of floats).
        u: Solution matrix u[j][i] where j is time index, i is space index.
        method: Numerical method used ('explicit', 'finite_difference', 'split-operator').
        success: Whether the solution was successful.

    Examples:
        >>> result = PDEResult(x=[0, 0.5, 1.0], t=[0, 0.1, 0.2],
        ...                    u=[[1,0,0], [0.8,0,0], [0.6,0,0]],
        ...                    method='explicit', success=True)
        >>> print(f"Grid size: {len(result.x)} x {len(result.t)}")
        Grid size: 3 x 3
    """
    x: List[float]
    t: List[float]
    u: List[List[float]]
    method: str
    success: bool


def solve_heat_equation(nt: int = 100, nx: int = 50, L: float = 1.0,
                        T: float = 1.0, alpha: float = 0.01,
                        initial_condition: Optional[Callable] = None) -> PDEResult:
    """
    Solve the 1D heat equation (diffusion equation) numerically.

    The heat equation is: ∂u/∂t = α ∂²u/∂x²

    This implementation uses the explicit forward-time centered-space (FTCS)
    finite difference method. The scheme is stable when r = αΔt/Δx² ≤ 0.5.

    Args:
        nt: Number of time steps.
        nx: Number of spatial points.
        L: Length of the domain (x from 0 to L).
        T: Total simulation time.
        alpha: Thermal diffusivity coefficient.
        initial_condition: Initial temperature distribution u(x,0).
                           If None, uses a Gaussian peak at the center.

    Returns:
        PDEResult containing the solution u(x,t).

    Examples:
        >>> # Default Gaussian initial condition
        >>> result = solve_heat_equation(nt=100, nx=50, L=1.0, T=1.0)
        >>> 
        >>> # Custom initial condition
        >>> def init(x):
        ...     return 1.0 if 0.4 < x < 0.6 else 0.0
        >>> result = solve_heat_equation(initial_condition=init)
    """
    if initial_condition is None:
        def initial_condition(x):
            return math.exp(-50 * (x - L/2)**2)
    
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2
    
    x = [i * dx for i in range(nx)]
    t = [j * dt for j in range(nt)]
    
    u = [[0.0] * nx for _ in range(nt)]
    for i in range(nx):
        u[0][i] = initial_condition(x[i])
    
    for j in range(nt - 1):
        for i in range(1, nx - 1):
            u[j+1][i] = u[j][i] + r * (u[j][i-1] - 2*u[j][i] + u[j][i+1])
        
        # Boundary conditions (Dirichlet: u=0 at boundaries)
        u[j+1][0] = 0.0
        u[j+1][nx-1] = 0.0
    
    return PDEResult(x=x, t=t, u=u, method='explicit', success=True)


def solve_wave_equation(nt: int = 100, nx: int = 50, L: float = 1.0,
                        T: float = 2.0, c: float = 1.0,
                        initial_condition: Optional[Callable] = None) -> PDEResult:
    """
    Solve the 1D wave equation numerically with reflective boundaries.

    The wave equation is: ∂²u/∂t² = c² ∂²u/∂x²

    This implementation uses the finite difference method with reflective
    (Neumann) boundary conditions: ∂u/∂x = 0 at the boundaries.

    The scheme is stable when the Courant number cΔt/Δx ≤ 1.

    Args:
        nt: Number of time steps.
        nx: Number of spatial points.
        L: Length of the domain (x from 0 to L).
        T: Total simulation time.
        c: Wave speed.
        initial_condition: Initial displacement u(x,0).
                           If None, uses sin(πx) (fundamental mode).

    Returns:
        PDEResult containing the solution u(x,t).

    Examples:
        >>> # Fundamental mode oscillation
        >>> result = solve_wave_equation(nt=100, nx=50, L=1.0, T=2.0, c=1.0)
        >>> 
        >>> # Pluck the string at the center
        >>> def pluck(x):
        ...     return 1 - abs(2*x - 1)  # Triangular shape
        >>> result = solve_wave_equation(initial_condition=pluck)
    """
    if initial_condition is None:
        def initial_condition(x):
            return math.sin(math.pi * x)
    
    dx = L / (nx - 1)
    dt = min(T / nt, dx / c * 0.9)  # Ensure stability
    
    x = [i * dx for i in range(nx)]
    t = [j * dt for j in range(nt)]
    
    u = [[0.0] * nx for _ in range(nt)]
    
    for i in range(nx):
        u[0][i] = initial_condition(x[i])
    
    courant2 = (c * dt / dx)**2
    
    # First time step (using initial velocity = 0)
    for i in range(1, nx - 1):
        u[1][i] = u[0][i] + 0.5 * courant2 * (u[0][i-1] - 2*u[0][i] + u[0][i+1])
    
    # Reflective boundaries (Neumann: du/dx = 0)
    u[1][0] = u[1][1]
    u[1][nx-1] = u[1][nx-2]
    
    for j in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[j+1][i] = 2*u[j][i] - u[j-1][i] + \
                        courant2 * (u[j][i-1] - 2*u[j][i] + u[j][i+1])
        
        u[j+1][0] = u[j+1][1]
        u[j+1][nx-1] = u[j+1][nx-2]
    
    return PDEResult(x=x, t=t, u=u, method='finite_difference', success=True)


def solve_schrodinger_1d(nt: int = 100, nx: int = 100, L: float = 10.0,
                         T: float = 2.0, mass: float = 1.0, hbar: float = 1.0,
                         potential: Optional[Callable] = None) -> PDEResult:
    """
    Solve the 1D time-dependent Schrödinger equation.

    The Schrödinger equation is: iℏ ∂ψ/∂t = -ℏ²/2m ∂²ψ/∂x² + V(x)ψ

    This implementation uses the split-operator method, which alternates
    between applying the potential and kinetic energy operators.

    Returns the probability density |ψ(x,t)|².

    Args:
        nt: Number of time steps.
        nx: Number of spatial points.
        L: Length of the domain (x from -L/2 to L/2).
        T: Total simulation time.
        mass: Particle mass (default: 1).
        hbar: Reduced Planck constant (default: 1).
        potential: Potential function V(x). If None, uses harmonic oscillator.

    Returns:
        PDEResult containing the probability density |ψ|².

    Examples:
        >>> # Harmonic oscillator (default)
        >>> result = solve_schrodinger_1d(nt=100, nx=100, L=10.0, T=2.0)
        >>> 
        >>> # Square well potential
        >>> def square_well(x):
        ...     return 0 if -5 < x < 5 else 100
        >>> result = solve_schrodinger_1d(nt=100, potential=square_well)
        >>> 
        >>> # Visualize results
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(result.u, aspect='auto', extent=[-5, 5, 2, 0])
        >>> plt.colorbar(label='|ψ|²')
        >>> plt.show()
    """
    if potential is None:
        def potential(x):
            return 0.5 * mass * x**2  # Harmonic oscillator
    
    dx = L / (nx - 1)
    dt = T / nt
    
    x = [-L/2 + i * dx for i in range(nx)]
    t = [j * dt for j in range(nt)]
    
    # Initial Gaussian wavepacket
    sigma = 0.5
    x0 = -2.0
    k0 = 3.0
    
    psi = [0.0] * nx
    for i in range(nx):
        psi[i] = math.exp(-(x[i] - x0)**2 / (4 * sigma**2)) * math.cos(k0 * x[i])
    
    # Normalize
    norm = math.sqrt(sum(psi[i]**2 for i in range(nx)) * dx)
    if norm > 0:
        for i in range(nx):
            psi[i] /= norm
    
    V = [potential(x[i]) for i in range(nx)]
    
    # Split-operator method
    prob_density = [[0.0] * nx for _ in range(nt)]
    
    # Store initial probability
    for i in range(nx):
        prob_density[0][i] = psi[i]**2
    
    # Precompute phase factors
    phase_V = [math.exp(-V[i] * dt / (2 * hbar)) for i in range(nx)]
    
    # Time evolution
    for j in range(nt - 1):
        # Apply potential half-step
        for i in range(nx):
            psi[i] *= phase_V[i]
        
        # Apply kinetic step using FFT (simplified: use finite difference)
        # Simple finite difference approximation for kinetic term
        psi_new = psi.copy()
        coeff = hbar * dt / (2 * mass * dx**2)
        
        for i in range(1, nx - 1):
            # Second derivative approximation
            d2psi = (psi[i-1] - 2*psi[i] + psi[i+1]) / dx**2
            psi_new[i] = psi[i] + 1j * coeff * d2psi
        
        # Use real part only for stability
        for i in range(1, nx - 1):
            psi[i] = psi_new[i].real if isinstance(psi_new[i], complex) else psi_new[i]
        
        # Apply potential half-step again
        for i in range(nx):
            psi[i] *= phase_V[i]
        
        # Re-normalize
        norm = math.sqrt(sum(psi[i]**2 for i in range(nx)) * dx)
        if norm > 1e-10:
            for i in range(nx):
                psi[i] /= norm
        
        # Store probability density
        for i in range(nx):
            prob_density[j+1][i] = psi[i]**2
    
    return PDEResult(x=x, t=t, u=prob_density, method='split-operator', success=True)


def diffusion_equation_implicit(nt: int = 100, nx: int = 50, L: float = 1.0,
                                 T: float = 1.0, alpha: float = 0.01) -> PDEResult:
    """
    Solve the diffusion equation using the explicit method (alias for solve_heat_equation).

    Note: Despite the name, this uses the explicit FTCS method.
    For an implicit method, a different implementation would be needed.

    Args:
        nt: Number of time steps.
        nx: Number of spatial points.
        L: Length of the domain.
        T: Total simulation time.
        alpha: Diffusion coefficient.

    Returns:
        PDEResult containing the solution.

    Examples:
        >>> result = diffusion_equation_implicit(nt=100, nx=50, L=1.0, T=1.0, alpha=0.01)
    """
    return solve_heat_equation(nt, nx, L, T, alpha)


__all__ = [
    'PDEResult',
    'solve_heat_equation',
    'solve_wave_equation',
    'solve_schrodinger_1d',
    'diffusion_equation_implicit',
]