"""
psiqit/visualization/animation.py
============================================================
Quantum Animation
============================================================

Animation of quantum states and time evolution.

This module provides tools for creating animations of quantum systems,
including wavefunction evolution and Bloch sphere dynamics. These animations
are useful for visualizing time-dependent quantum phenomena and for
educational purposes.

The module includes:
    - Wavefunction evolution animation (real, imaginary parts, and probability density)
    - Bloch sphere animation for single qubit states

Example:
    >>> from psiqit.visualization.animation import animate_wavefunction
    >>> import numpy as np
    >>> 
    >>> # Define a moving Gaussian wavepacket
    >>> def psi(x, t):
    ...     return np.exp(-(x - t)**2/4) * np.exp(1j*2*x)
    >>> 
    >>> # Animate the wavefunction
    >>> animate_wavefunction(psi, x_range=(-5, 5), t_range=(0, 5), n_frames=100)

References:
    H. Goldstein, "Classical Mechanics," 3rd ed., Addison-Wesley, 2001.
    R. P. Feynman, "The Feynman Lectures on Physics," Vol. III, 1965.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


def animate_wavefunction(psi_func: Callable, x_range: Tuple[float, float] = (-5, 5),
                        t_range: Tuple[float, float] = (0, 5),
                        n_x: int = 200, n_frames: int = 100,
                        interval: int = 50, save_path: Optional[str] = None):
    """
    Animate the time evolution of a wavefunction.

    This function creates an animation showing the real part, imaginary part,
    and probability density of a wavefunction as it evolves in time.

    Args:
        psi_func: Wavefunction ψ(x, t) as a callable taking (x, t) arguments.
        x_range: Tuple (x_min, x_max) for the spatial domain.
        t_range: Tuple (t_min, t_max) for the time domain.
        n_x: Number of spatial points (grid resolution).
        n_frames: Number of animation frames (time steps).
        interval: Time between frames in milliseconds.
        save_path: Optional path to save the animation (e.g., 'animation.gif').

    Examples:
        >>> # Free particle Gaussian wavepacket
        >>> def psi(x, t):
        ...     return np.exp(-(x - t)**2/4) * np.exp(1j*2*x)
        >>> animate_wavefunction(psi, x_range=(-5, 5), t_range=(0, 5))
        >>> 
        >>> # Stationary state of harmonic oscillator
        >>> def psi(x, t):
        ...     return np.exp(-x**2/4) * np.exp(-1j*0.5*t)  # Ground state
        >>> animate_wavefunction(psi, x_range=(-5, 5), t_range=(0, 10))

    Notes:
        Requires matplotlib to be installed. If not, prints an error message.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        x = np.linspace(x_range[0], x_range[1], n_x)
        times = np.linspace(t_range[0], t_range[1], n_frames)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Initial plot
        psi0 = psi_func(x, 0)
        line_real, = ax1.plot(x, np.real(psi0), 'b-', label='Re(ψ)')
        line_imag, = ax1.plot(x, np.imag(psi0), 'r-', label='Im(ψ)')
        line_prob, = ax2.plot(x, np.abs(psi0)**2, 'g-', label='|ψ|²')
        
        ax1.set_xlim(x_range[0], x_range[-1])
        ax1.set_ylim(-1, 1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('ψ(x,t)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlim(x_range[0], x_range[-1])
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('x')
        ax2.set_ylabel('|ψ(x,t)|²')
        ax2.legend()
        ax2.grid(True)
        
        fig.suptitle('Wavefunction Evolution')
        
        def update(frame):
            t = times[frame]
            psi = psi_func(x, t)
            line_real.set_ydata(np.real(psi))
            line_imag.set_ydata(np.imag(psi))
            line_prob.set_ydata(np.abs(psi)**2)
            ax1.set_title(f't = {t:.2f}')
            return line_real, line_imag, line_prob
        
        anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000/interval)
            print(f"Saved animation to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


def animate_bloch_sphere(states: List, interval: int = 200,
                        save_path: Optional[str] = None):
    """
    Animate a sequence of quantum states on the Bloch sphere.

    This function creates a 3D animation showing how a qubit state evolves
    on the Bloch sphere. Each state in the list is displayed sequentially.

    Args:
        states: List of quantum states (Ket objects) to animate.
        interval: Time between frames in milliseconds.
        save_path: Optional path to save the animation (e.g., 'bloch.gif').

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus, minus, ip
        >>> from psiqit.visualization.animation import animate_bloch_sphere
        >>> 
        >>> # Create a sequence of states
        >>> states = [zero(), plus(), one(), minus(), zero()]
        >>> animate_bloch_sphere(states, interval=500)
        >>> 
        >>> # Save animation
        >>> animate_bloch_sphere(states, save_path="bloch_rotation.gif")

    Notes:
        Requires matplotlib with 3D plotting support. If not available,
        prints an error message.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation
        
        from .bloch import state_to_bloch
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        xs = np.outer(np.sin(v), np.cos(u))
        ys = np.outer(np.sin(v), np.sin(u))
        zs = np.outer(np.cos(v), np.ones_like(u))
        ax.plot_surface(xs, ys, zs, alpha=0.1, color='cyan')
        
        # Draw axes
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.3)
        
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Bloch Sphere Animation')
        
        # Get coordinates
        coords = [state_to_bloch(s) for s in states]
        
        # Initial point
        x0, y0, z0 = coords[0]
        point = ax.scatter([x0], [y0], [z0], c='red', s=100)
        
        def update(frame):
            x, y, z = coords[frame]
            point._offsets3d = ([x], [y], [z])
            return [point]
        
        anim = FuncAnimation(fig, update, frames=len(states), interval=interval, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
            print(f"Saved animation to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


__all__ = [
    'animate_wavefunction',
    'animate_bloch_sphere',
]