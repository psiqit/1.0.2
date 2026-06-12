"""
psiqit/visualization/husimi.py
============================================================
Husimi Q-Function
============================================================

Husimi Q-function (Husimi distribution) for quantum states.

The Husimi Q-function is a quasiprobability distribution that represents
a quantum state in phase space. Unlike the Wigner function, the Q-function
is always non-negative and can be interpreted as a probability distribution
for simultaneous position and momentum measurements (though not simultaneously
sharp due to uncertainty principle).

Definition:
    Q(α) = ⟨α|ρ|α⟩ / π

where |α⟩ is a coherent state. For a pure state |ψ⟩, Q(α) = |⟨α|ψ⟩|²/π.

The Q-function is particularly useful for visualizing:
    - Coherent states (Gaussian distributions)
    - Squeezed states
    - Quantum state overlap and fidelity

Example:
    >>> from psiqit.visualization.husimi import husimi_function_gaussian, plot_husimi
    >>> 
    >>> # Compute Husimi function for a Gaussian state
    >>> Q, x, p = husimi_function_gaussian(x0=0, p0=0, sigma=1)
    >>> plot_husimi(Q, x, p, title="Gaussian State")

References:
    K. Husimi, "Some formal properties of the density matrix,"
    Proceedings of the Physico-Mathematical Society of Japan, 22(4):264-314, 1940.
    M. Hillery, R. F. O'Connell, M. O. Scully, and E. P. Wigner,
    "Distribution functions in physics: Fundamentals,"
    Physics Reports, 106(3):121-167, 1984.
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Callable


def husimi_function_gaussian(x0: float = 0.0, p0: float = 0.0, sigma: float = 1.0,
                            x_range: Tuple[float, float] = (-4, 4),
                            p_range: Tuple[float, float] = (-4, 4),
                            n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function for a Gaussian state.

    For a Gaussian state centered at (x₀, p₀) with width σ = 1,
    the Q-function is a Gaussian distribution:
        Q(x, p) = (1/π) exp(-(x-x₀)² - (p-p₀)²)

    This is also the Q-function for a coherent state |α⟩ with
    α = (x₀ + i p₀)/√2.

    Args:
        x0: Mean position (default: 0).
        p0: Mean momentum (default: 0).
        sigma: Width parameter (default: 1).
        x_range: Tuple (x_min, x_max) for position grid.
        p_range: Tuple (p_min, p_max) for momentum grid.
        n_points: Number of grid points in each dimension.

    Returns:
        Tuple (Q, x_grid, p_grid) where Q is the 2D array of Q-function values.

    Examples:
        >>> # Ground state of harmonic oscillator (coherent state with α=0)
        >>> Q, x, p = husimi_function_gaussian(x0=0, p0=0, sigma=1)
        >>> print(Q.shape)
        (100, 100)
        >>> 
        >>> # Displaced coherent state
        >>> Q, x, p = husimi_function_gaussian(x0=2, p0=1, sigma=1)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    p = np.linspace(p_range[0], p_range[1], n_points)
    
    X, P = np.meshgrid(x, p, indexing='ij')
    
    Q = (1/np.pi) * np.exp(-(X - x0)**2 - (P - p0)**2)
    
    return Q, x, p


def husimi_function_coherent_state(alpha: complex,
                                   x_range: Tuple[float, float] = (-4, 4),
                                   p_range: Tuple[float, float] = (-4, 4),
                                   n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Husimi Q-function for a coherent state |α⟩.

    For a coherent state |α⟩, the Q-function is a Gaussian centered at
    (x₀, p₀) = (√2 Re(α), √2 Im(α)).

    Args:
        alpha: Complex amplitude α = |α|e^{iθ}.
        x_range: Tuple (x_min, x_max) for position grid.
        p_range: Tuple (p_min, p_max) for momentum grid.
        n_points: Number of grid points in each dimension.

    Returns:
        Tuple (Q, x_grid, p_grid).

    Examples:
        >>> # Coherent state |α=1⟩
        >>> Q, x, p = husimi_function_coherent_state(alpha=1.0)
        >>> 
        >>> # Coherent state with phase α = 2e^{iπ/4}
        >>> import math
        >>> alpha = 2 * math.cos(math.pi/4) + 1j * 2 * math.sin(math.pi/4)
        >>> Q, x, p = husimi_function_coherent_state(alpha=alpha)
    """
    x0 = math.sqrt(2) * alpha.real
    p0 = math.sqrt(2) * alpha.imag
    
    return husimi_function_gaussian(x0, p0, 1.0, x_range, p_range, n_points)


def plot_husimi(Q: np.ndarray, x: np.ndarray, p: np.ndarray,
                title: str = "Husimi Q-Function",
                cmap: str = 'hot',
                save_path: Optional[str] = None):
    """
    Plot the Husimi Q-function as a 2D color map.

    This function creates a 2D plot of the Husimi Q-function in phase space,
    with position on the x-axis and momentum on the y-axis.

    Args:
        Q: 2D array of Q-function values (from husimi_function_*).
        x: Position grid (1D array).
        p: Momentum grid (1D array).
        title: Plot title (default: "Husimi Q-Function").
        cmap: Colormap name (default: 'hot').
        save_path: If provided, save plot to this path instead of showing.

    Examples:
        >>> # Plot Gaussian state
        >>> Q, x, p = husimi_function_gaussian()
        >>> plot_husimi(Q, x, p, title="Vacuum State")
        >>> 
        >>> # Plot coherent state and save
        >>> Q, x, p = husimi_function_coherent_state(alpha=1.5)
        >>> plot_husimi(Q, x, p, save_path="husimi_coherent.png")

    Notes:
        Requires matplotlib to be installed. If not, prints an error message.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        extent = [x[0], x[-1], p[0], p[-1]]
        im = ax.imshow(Q.T, extent=extent, origin='lower', aspect='auto', cmap=cmap)
        
        plt.colorbar(im, ax=ax, label='Q(x,p)')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Momentum p')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


__all__ = [
    'husimi_function_gaussian',
    'husimi_function_coherent_state',
    'plot_husimi',
]