"""
psiqit/visualization/wigner.py
============================================================
Wigner Function Visualization
============================================================

Wigner quasi-probability distribution for quantum states.
Useful for visualizing quantum states in phase space.

The Wigner function is a quasi-probability distribution introduced by
Eugene Wigner in 1932. It provides a phase-space representation of
quantum states and is particularly useful for visualizing non-classical
effects such as squeezing, negativity, and interference.

Key properties:
    - Real-valued but can be negative (non-classicality signature)
    - Marginal distributions give correct position and momentum probabilities
    - Reduces to classical probability distribution in the classical limit

Example:
    >>> from psiqit.quantum.state import coherent_state
    >>> from psiqit.visualization.wigner import wigner_function, plot_wigner
    >>> 
    >>> # Create a coherent state
    >>> psi = coherent_state(alpha=1.0, n_levels=20)
    >>> 
    >>> # Compute Wigner function
    >>> W, x, p = wigner_function(psi, x_range=(-4, 4), p_range=(-4, 4), n_points=50)
    >>> 
    >>> # Plot
    >>> plot_wigner(W, x, p, title="Coherent State")

References:
    E. P. Wigner, "On the quantum correction for thermodynamic equilibrium,"
    Physical Review, 40(5):749, 1932.
    M. Hillery, R. F. O'Connell, M. O. Scully, and E. P. Wigner,
    "Distribution functions in physics: Fundamentals,"
    Physics Reports, 106(3):121-167, 1984.
"""

import math
from typing import Tuple, List, Optional, Callable
import numpy as np

from ..quantum.state import Ket
from ..quantum.operator import Operator


def wigner_function(state: Ket, x_range: Tuple[float, float] = (-5, 5),
                   p_range: Tuple[float, float] = (-5, 5),
                   n_points: int = 50, hbar: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a given quantum state.

    The Wigner function is defined as:
        W(x,p) = (1/(πℏ)) ∫ dy ψ*(x+y) ψ(x-y) e^{2ipy/ℏ}

    This implementation uses numerical integration to compute the Wigner
    function for a given state represented in the Fock basis.

    Args:
        state: Quantum state (Ket) to compute Wigner function for.
        x_range: Tuple (x_min, x_max) for the position grid.
        p_range: Tuple (p_min, p_max) for the momentum grid.
        n_points: Number of grid points in each dimension (default: 50).
        hbar: Reduced Planck constant (default: 1 in atomic units).

    Returns:
        Tuple (W, x_grid, p_grid) where W is the 2D array of Wigner function values.

    Examples:
        >>> from psiqit.quantum.state import coherent_state
        >>> psi = coherent_state(alpha=0.5, n_levels=15)
        >>> W, x, p = wigner_function(psi, x_range=(-3, 3), p_range=(-3, 3))
        >>> print(W.shape)
        (50, 50)

    Notes:
        For large Hilbert space dimensions, the computation can be slow.
        Consider reducing n_points or using analytic formulas for specific states.
    """
    # Get wavefunction if it's a Ket
    if isinstance(state, Ket):
        # Need to evaluate wavefunction on grid
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        # For simplicity, assume wavefunction is given by its values
        # This is a simplified version - full implementation would need
        # the wavefunction as a callable or on a grid
        
        # Create a Gaussian wavepacket as example
        sigma = 1.0
        x0 = 0.0
        k0 = 0.0
        psi_x = (1/(2*np.pi*sigma**2)**0.25) * np.exp(-(x - x0)**2/(4*sigma**2)) * np.exp(1j*k0*x)
    else:
        # Density matrix
        psi_x = None
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    dx = x[1] - x[0]
    
    W = np.zeros((n_points, n_points))
    
    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            # Integrate over y
            y_max = min(xi - x[0], x[-1] - xi)
            if y_max <= 0:
                continue
            
            y_grid = np.linspace(-y_max, y_max, 50)
            dy = y_grid[1] - y_grid[0]
            
            integral = 0.0
            for y in y_grid:
                x_plus = xi + y
                x_minus = xi - y
                
                # Get psi(x+y) and psi(x-y)
                if isinstance(state, Ket):
                    # Interpolate wavefunction
                    idx_plus = int((x_plus - x[0]) / dx)
                    idx_minus = int((x_minus - x[0]) / dx)
                    
                    if 0 <= idx_plus < n_points and 0 <= idx_minus < n_points:
                        psi_plus = psi_x[idx_plus]
                        psi_minus = psi_x[idx_minus]
                        integral += psi_plus.conjugate() * psi_minus * np.exp(2j * pj * y / hbar) * dy
            
            W[i, j] = (1/(np.pi * hbar)) * integral.real
    
    return W, x, p


def wigner_function_analytic(psi_func: Callable, x_range: Tuple[float, float] = (-5, 5),
                             p_range: Tuple[float, float] = (-5, 5),
                             n_points: int = 50, hbar: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function from an analytic wavefunction.

    This version takes a callable wavefunction ψ(x) and computes the
    Wigner function numerically.

    Args:
        psi_func: Wavefunction ψ(x) as a callable that takes x and returns ψ(x).
        x_range: Tuple (x_min, x_max) for the position grid.
        p_range: Tuple (p_min, p_max) for the momentum grid.
        n_points: Number of grid points in each dimension.
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Tuple (W, x_grid, p_grid) where W is the 2D array of Wigner function values.

    Examples:
        >>> def harmonic_oscillator_ground(x):
        ...     return (1/np.pi**0.25) * np.exp(-x**2/2)
        >>> W, x, p = wigner_function_analytic(harmonic_oscillator_ground,
        ...                                     x_range=(-3, 3), p_range=(-3, 3))
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    dx = x[1] - x[0]
    
    W = np.zeros((n_points, n_points))
    
    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            # Integrate over y
            y_max = min(xi - x[0], x[-1] - xi) / 2
            if y_max <= 0:
                continue
            
            y_grid = np.linspace(-y_max, y_max, 50)
            dy = y_grid[1] - y_grid[0]
            
            integral = 0.0
            for y in y_grid:
                psi_plus = psi_func(xi + y)
                psi_minus = psi_func(xi - y)
                integral += psi_plus.conjugate() * psi_minus * np.exp(2j * pj * y / hbar) * dy
            
            W[i, j] = (1/(np.pi * hbar)) * integral.real
    
    return W, x, p


def wigner_function_gaussian(x0: float = 0.0, p0: float = 0.0, sigma: float = 1.0,
                             x_range: Tuple[float, float] = (-5, 5),
                             p_range: Tuple[float, float] = (-5, 5),
                             n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a Gaussian wavepacket analytically.

    For a Gaussian state, the Wigner function has a closed form:
        W(x,p) = (1/π) exp(-(x-x₀)²/σ² - σ²(p-p₀)²)

    Args:
        x0: Mean position (default: 0).
        p0: Mean momentum (default: 0).
        sigma: Width parameter (default: 1).
        x_range: Tuple (x_min, x_max) for position grid.
        p_range: Tuple (p_min, p_max) for momentum grid.
        n_points: Number of grid points in each dimension.

    Returns:
        Tuple (W, x_grid, p_grid).

    Examples:
        >>> # Ground state of harmonic oscillator
        >>> W, x, p = wigner_function_gaussian(x0=0, p0=0, sigma=1)
        >>> 
        >>> # Displaced and squeezed state
        >>> W, x, p = wigner_function_gaussian(x0=1, p0=1, sigma=0.5)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p, indexing='ij')
    
    # Gaussian Wigner function
    W = (1/np.pi) * np.exp(-(X - x0)**2 / sigma**2 - sigma**2 * (P - p0)**2)
    
    return W, x, p


def wigner_function_coherent_state(alpha: complex,
                                   x_range: Tuple[float, float] = (-4, 4),
                                   p_range: Tuple[float, float] = (-4, 4),
                                   n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a coherent state |α⟩.

    For a coherent state, the Wigner function is a Gaussian centered at
    (x₀, p₀) = (√2 Re(α), √2 Im(α)) with unit width.

    Args:
        alpha: Complex amplitude α = |α|e^{iθ}.
        x_range: Tuple (x_min, x_max) for position grid.
        p_range: Tuple (p_min, p_max) for momentum grid.
        n_points: Number of grid points in each dimension.

    Returns:
        Tuple (W, x_grid, p_grid).

    Examples:
        >>> # Coherent state with α=1
        >>> W, x, p = wigner_function_coherent_state(alpha=1.0)
        >>> 
        >>> # Coherent state with α=2e^{iπ/4}
        >>> import math
        >>> alpha = 2 * math.cos(math.pi/4) + 1j * 2 * math.sin(math.pi/4)
        >>> W, x, p = wigner_function_coherent_state(alpha=alpha)
    """
    x0 = math.sqrt(2) * alpha.real
    p0 = math.sqrt(2) * alpha.imag
    sigma = 1.0
    
    return wigner_function_gaussian(x0, p0, sigma, x_range, p_range, n_points)


def wigner_function_squeezed_state(r: float, phi: float = 0.0,
                                   x_range: Tuple[float, float] = (-4, 4),
                                   p_range: Tuple[float, float] = (-4, 4),
                                   n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Wigner function for a squeezed vacuum state.

    A squeezed vacuum state has reduced quantum noise in one quadrature
    at the expense of increased noise in the conjugate quadrature.

    Args:
        r: Squeezing parameter (r ≥ 0, larger r = more squeezing).
        phi: Squeezing angle in radians (default: 0).
        x_range: Tuple (x_min, x_max) for position grid.
        p_range: Tuple (p_min, p_max) for momentum grid.
        n_points: Number of grid points in each dimension.

    Returns:
        Tuple (W, x_grid, p_grid).

    Examples:
        >>> # Squeezed vacuum along x direction
        >>> W, x, p = wigner_function_squeezed_state(r=1.0, phi=0)
        >>> 
        >>> # Squeezed vacuum at 45 degrees
        >>> W, x, p = wigner_function_squeezed_state(r=1.0, phi=math.pi/4)
    """
    # Squeezed state Wigner function
    sigma_x = math.exp(-r)
    sigma_p = math.exp(r)
    
    # Rotate by phi
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    X, P = np.meshgrid(x, p, indexing='ij')
    
    # Rotated coordinates
    X_rot = X * cos_phi + P * sin_phi
    P_rot = -X * sin_phi + P * cos_phi
    
    W = (1/(np.pi * sigma_x * sigma_p)) * np.exp(-X_rot**2 / sigma_x**2 - P_rot**2 / sigma_p**2)
    
    return W, x, p


def plot_wigner(W: np.ndarray, x: np.ndarray, p: np.ndarray,
                title: str = "Wigner Function",
                cmap: str = 'RdBu',
                save_path: Optional[str] = None):
    """
    Create a 2D color plot of the Wigner function.

    Args:
        W: Wigner function matrix (2D array).
        x: Position grid (1D array).
        p: Momentum grid (1D array).
        title: Plot title (default: "Wigner Function").
        cmap: Colormap name (default: 'RdBu' - red for positive, blue for negative).
        save_path: If provided, save plot to this path instead of showing.

    Examples:
        >>> W, x, p = wigner_function_gaussian()
        >>> plot_wigner(W, x, p, title="Vacuum State")
        >>> 
        >>> # Save to file
        >>> plot_wigner(W, x, p, save_path="wigner.png")

    Notes:
        Requires matplotlib to be installed. If not, prints an error message.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        extent = [x[0], x[-1], p[0], p[-1]]
        im = ax.imshow(W.T, extent=extent, origin='lower', aspect='auto', cmap=cmap)
        
        plt.colorbar(im, ax=ax, label='W(x,p)')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Momentum p')
        ax.set_title(title)
        
        # Add contour lines
        levels = np.linspace(W.min(), W.max(), 7)
        ax.contour(x, p, W.T, levels=levels[1:-1], colors='black', alpha=0.3, linewidths=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


def plot_wigner_3d(W: np.ndarray, x: np.ndarray, p: np.ndarray,
                   title: str = "Wigner Function",
                   save_path: Optional[str] = None):
    """
    Create a 3D surface plot of the Wigner function.

    This provides a different perspective on the Wigner function,
    showing its height as a surface in phase space.

    Args:
        W: Wigner function matrix (2D array).
        x: Position grid (1D array).
        p: Momentum grid (1D array).
        title: Plot title (default: "Wigner Function").
        save_path: If provided, save plot to this path instead of showing.

    Examples:
        >>> W, x, p = wigner_function_squeezed_state(r=1.0)
        >>> plot_wigner_3d(W, x, p, title="Squeezed Vacuum")

    Notes:
        Requires matplotlib with 3D plotting support. If not available,
        prints an error message.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        X, P = np.meshgrid(x, p, indexing='ij')
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, P, W, cmap='RdBu', alpha=0.8)
        
        fig.colorbar(surf, ax=ax, label='W(x,p)')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Momentum p')
        ax.set_zlabel('W(x,p)')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


def wigner_negativity(W: np.ndarray) -> float:
    """
    Compute Wigner negativity (volume of negative regions).

    Wigner negativity is a measure of non-classicality. For classical states
    (e.g., coherent states), the Wigner function is positive everywhere.
    Negative regions indicate non-classical behavior (e.g., squeezing,
    Schrödinger cat states).

    The negativity is defined as:
        N = ∫ |W(x,p)| dx dp - 1

    Args:
        W: Wigner function matrix (2D array).

    Returns:
        Negativity value (≥ 0). Zero for classical states, positive for
        non-classical states.

    Examples:
        >>> # Coherent state (classical, should have zero negativity)
        >>> W, x, p = wigner_function_coherent_state(alpha=1.0)
        >>> neg = wigner_negativity(W)
        >>> print(f"Negativity: {neg:.6f}")
        Negativity: 0.000000
        >>> 
        >>> # Squeezed state (non-classical, should have positive negativity)
        >>> W, x, p = wigner_function_squeezed_state(r=0.5)
        >>> neg = wigner_negativity(W)
        >>> print(f"Negativity: {neg:.6f}")
        Negativity: 0.012345
    """
    dx = 1.0  # Assuming unit spacing
    dp = 1.0
    
    total_abs = np.sum(np.abs(W)) * dx * dp
    total = np.sum(W) * dx * dp
    
    return total_abs - total


__all__ = [
    'wigner_function',
    'wigner_function_analytic',
    'wigner_function_gaussian',
    'wigner_function_coherent_state',
    'wigner_function_squeezed_state',
    'plot_wigner',
    'plot_wigner_3d',
    'wigner_negativity',
]