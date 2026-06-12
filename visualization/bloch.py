"""
psiqit/visualization/bloch.py
============================================================
Bloch Sphere Visualization
============================================================

Visual representation of single-qubit states on the Bloch sphere.

The Bloch sphere is a geometric representation of a single qubit state.
Any pure state of a qubit can be represented as a point on the unit sphere:

    |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩

where:
    - θ ∈ [0, π] is the polar angle (from +Z axis)
    - φ ∈ [0, 2π) is the azimuthal angle (from +X axis)
    - Coordinates: (x, y, z) = (sin θ cos φ, sin θ sin φ, cos θ)

The Bloch sphere provides an intuitive visualization of qubit states,
quantum gates (as rotations), and quantum dynamics.

Example:
    >>> from psiqit.quantum.state import zero, one, plus, ip
    >>> from psiqit.visualization.bloch import bloch_sphere, state_to_bloch
    >>> 
    >>> # Get Bloch coordinates
    >>> coords = state_to_bloch(plus())
    >>> print(coords)  # (1.0, 0.0, 0.0)
    >>> 
    >>> # Plot Bloch sphere with a state
    >>> bloch_sphere(plus())
    >>> 
    >>> # Plot multiple states
    >>> plot_multiple_states([zero(), one(), plus(), ip()])

References:
    F. Bloch, "Nuclear induction," Physical Review, 70(7-8):460, 1946.
    M. A. Nielsen and I. L. Chuang, "Quantum Computation and Quantum Information,"
    Cambridge University Press, 2010.
"""

import math
from typing import Tuple, Optional, List, Union

from ..quantum.state import Ket, zero, one, plus, minus, ip, im
from ..quantum.operator import Operator


def state_to_bloch(state: Ket) -> Tuple[float, float, float]:
    """
    Convert a single-qubit state to Bloch sphere coordinates.

    The mapping is: |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    gives coordinates: (x, y, z) = (sin θ cos φ, sin θ sin φ, cos θ)

    Args:
        state: Single-qubit quantum state (must be normalized).

    Returns:
        Tuple (x, y, z) coordinates on the Bloch sphere.

    Raises:
        ValueError: If the state is not a single qubit.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus, ip
        >>> print(state_to_bloch(zero()))   # (0, 0, 1)
        (0.0, 0.0, 1.0)
        >>> print(state_to_bloch(one()))    # (0, 0, -1)
        (0.0, 0.0, -1.0)
        >>> print(state_to_bloch(plus()))   # (1, 0, 0)
        (1.0, 0.0, 0.0)
        >>> print(state_to_bloch(ip()))     # (0, 1, 0)
        (0.0, 1.0, 0.0)
    """
    if state.dim != 2:
        raise ValueError(f"Bloch sphere only for single qubit, got dim={state.dim}")
    
    if not state.is_normalized:
        state = state.normalize()
    
    # Get amplitudes
    a = state.data[0]  # amplitude for |0⟩
    b = state.data[1]  # amplitude for |1⟩
    
    # Calculate Bloch coordinates
    x = 2 * (a.conjugate() * b).real
    y = 2 * (a.conjugate() * b).imag
    z = (a.conjugate() * a - b.conjugate() * b).real
    
    # Normalize (should already be on sphere, but just in case)
    r = math.sqrt(x**2 + y**2 + z**2)
    if r > 1e-10:
        x, y, z = x/r, y/r, z/r
    
    return (x, y, z)


def bloch_to_state(x: float, y: float, z: float) -> Ket:
    """
    Convert Bloch sphere coordinates to a quantum state.

    Args:
        x, y, z: Coordinates on the Bloch sphere (must satisfy x²+y²+z² = 1).

    Returns:
        Quantum state |ψ⟩ on the Bloch sphere.

    Raises:
        ValueError: If the point is not on the unit sphere.

    Examples:
        >>> psi = bloch_to_state(1, 0, 0)   # |+⟩
        >>> psi = bloch_to_state(0, 1, 0)   # |i+⟩
        >>> psi = bloch_to_state(0, 0, 1)   # |0⟩
        >>> psi = bloch_to_state(0, 0, -1)  # |1⟩
    """
    # Check normalization
    r = math.sqrt(x**2 + y**2 + z**2)
    if abs(r - 1.0) > 1e-6:
        raise ValueError(f"Point ({x},{y},{z}) is not on unit sphere (r={r})")
    
    # Calculate spherical coordinates
    theta = math.acos(max(-1, min(1, z)))  # θ = arccos(z)
    phi = math.atan2(y, x)  # φ = atan2(y, x)
    
    # Construct state
    cos_half = math.cos(theta / 2)
    sin_half = math.sin(theta / 2)
    
    a = complex(cos_half, 0)
    b = complex(sin_half * math.cos(phi), sin_half * math.sin(phi))
    
    return Ket([a, b], _normalized=True)


def spherical_to_bloch(theta: float, phi: float) -> Tuple[float, float, float]:
    """
    Convert spherical coordinates to Bloch (Cartesian) coordinates.

    Args:
        theta: Polar angle (0 to π).
        phi: Azimuthal angle (0 to 2π).

    Returns:
        (x, y, z) Cartesian coordinates.

    Examples:
        >>> spherical_to_bloch(math.pi/2, 0)   # (1, 0, 0)
        (1.0, 0.0, 0.0)
        >>> spherical_to_bloch(0, 0)           # (0, 0, 1)
        (0.0, 0.0, 1.0)
        >>> spherical_to_bloch(math.pi, 0)     # (0, 0, -1)
        (0.0, 0.0, -1.0)
    """
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    return (x, y, z)


def bloch_to_spherical(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Convert Bloch (Cartesian) coordinates to spherical coordinates.

    Args:
        x, y, z: Cartesian coordinates.

    Returns:
        (theta, phi) spherical coordinates.

    Examples:
        >>> bloch_to_spherical(1, 0, 0)   # (π/2, 0)
        (1.5707963267948966, 0.0)
    """
    r = math.sqrt(x**2 + y**2 + z**2)
    if r > 0:
        x, y, z = x/r, y/r, z/r
    
    theta = math.acos(max(-1, min(1, z)))
    phi = math.atan2(y, x)
    return (theta, phi)


# ============================================================
# Matplotlib Visualization (Optional)
# ============================================================

def _has_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return True
    except ImportError:
        return False


def bloch_sphere(state: Optional[Ket] = None, 
                points: Optional[List[Tuple[float, float, float]]] = None,
                title: str = "Bloch Sphere",
                show_axes: bool = True,
                show_labels: bool = True,
                figsize: Tuple[int, int] = (8, 8),
                filename: Optional[str] = None):
    """
    Plot the Bloch sphere with an optional state vector and additional points.

    This function creates a 3D visualization of the Bloch sphere,
    showing the sphere, axes, cardinal points, and optionally a state
    vector and custom points.

    Args:
        state: Single-qubit state to visualize (optional).
        points: List of (x, y, z) points to plot (optional).
        title: Plot title.
        show_axes: Whether to show X, Y, Z axes.
        show_labels: Whether to show axis labels and cardinal point labels.
        figsize: Figure size (width, height) in inches.
        filename: If provided, save to file instead of showing.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus, ip, im
        >>> 
        >>> # Plot a single state
        >>> bloch_sphere(plus())
        >>> 
        >>> # Plot with custom points
        >>> bloch_sphere(points=[(1,0,0), (0,1,0), (0,0,1)])
        >>> 
        >>> # Save to file
        >>> bloch_sphere(plus(), filename="bloch.png")
    """
    if not _has_matplotlib():
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.sin(v), np.cos(u))
    ys = np.outer(np.sin(v), np.sin(u))
    zs = np.outer(np.cos(v), np.ones_like(u))
    
    ax.plot_surface(xs, ys, zs, alpha=0.15, color='cyan', edgecolor='none')
    
    # Draw axes
    if show_axes:
        # X axis (red)
        ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'r-', linewidth=1, alpha=0.5)
        # Y axis (green)
        ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'g-', linewidth=1, alpha=0.5)
        # Z axis (blue)
        ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'b-', linewidth=1, alpha=0.5)
        
        if show_labels:
            ax.text(1.3, 0, 0, 'X', color='red', fontsize=12)
            ax.text(0, 1.3, 0, 'Y', color='green', fontsize=12)
            ax.text(0, 0, 1.3, 'Z', color='blue', fontsize=12)
    
    # Draw cardinal points
    cardinal_points = {
        '|0⟩': (0, 0, 1),
        '|1⟩': (0, 0, -1),
        '|+⟩': (1, 0, 0),
        '|-⟩': (-1, 0, 0),
        '|i+⟩': (0, 1, 0),
        '|i-⟩': (0, -1, 0),
    }
    
    for label, (cx, cy, cz) in cardinal_points.items():
        ax.scatter([cx], [cy], [cz], c='black', s=50, alpha=0.5)
        if show_labels:
            # Offset labels slightly
            ax.text(cx*1.1, cy*1.1, cz*1.1, label, fontsize=9, alpha=0.7)
    
    # Draw state vector
    if state is not None:
        x, y, z = state_to_bloch(state)
        # Draw line from origin to point
        ax.quiver(0, 0, 0, x, y, z, color='red', 
                  arrow_length_ratio=0.1, linewidth=3)
        # Draw point
        ax.scatter([x], [y], [z], c='red', s=100)
    
    # Draw additional points
    if points is not None:
        for px, py, pz in points:
            ax.scatter([px], [py], [pz], c='blue', s=50)
    
    # Set labels and limits
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make axes equal
    ax.set_box_aspect([1, 1, 1])
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to {filename}")
        plt.close()
    else:
        plt.show()


def plot_multiple_states(states: List[Ket], 
                         colors: Optional[List[str]] = None,
                         title: str = "Bloch Sphere - Multiple States",
                         filename: Optional[str] = None):
    """
    Plot multiple quantum states on the same Bloch sphere.

    This is useful for comparing different states or visualizing
    state trajectories.

    Args:
        states: List of quantum states to plot.
        colors: List of colors for each state (optional).
        title: Plot title.
        filename: If provided, save to file instead of showing.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus, ip
        >>> plot_multiple_states([zero(), one(), plus(), ip()])
        >>> 
        >>> # With custom colors
        >>> plot_multiple_states([zero(), one()], colors=['blue', 'red'])
    """
    if not _has_matplotlib():
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    if colors is None:
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        colors = default_colors[:len(states)]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.sin(v), np.cos(u))
    ys = np.outer(np.sin(v), np.sin(u))
    zs = np.outer(np.cos(v), np.ones_like(u))
    
    ax.plot_surface(xs, ys, zs, alpha=0.1, color='cyan', edgecolor='none')
    
    # Draw axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', linewidth=0.5, alpha=0.3)
    
    # Draw states
    for i, state in enumerate(states):
        x, y, z = state_to_bloch(state)
        color = colors[i % len(colors)]
        ax.quiver(0, 0, 0, x, y, z, color=color, 
                  arrow_length_ratio=0.1, linewidth=2)
        ax.scatter([x], [y], [z], c=color, s=80)
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to {filename}")
        plt.close()
    else:
        plt.show()


def animate_bloch(states: List[Ket], 
                  interval: int = 200,
                  filename: Optional[str] = None):
    """
    Create an animation of states on the Bloch sphere.

    This function animates a sequence of quantum states, showing how the
    state vector moves on the Bloch sphere over time.

    Args:
        states: List of states to animate (in order).
        interval: Time between frames in milliseconds.
        filename: If provided, save animation to file (e.g., 'bloch.gif').

    Examples:
        >>> from psiqit.quantum.state import zero, plus, minus, one
        >>> 
        >>> # Create a sequence of states
        >>> states = [zero(), plus(), one(), minus(), zero()]
        >>> animate_bloch(states, interval=500)
        >>> 
        >>> # Save animation
        >>> animate_bloch(states, filename="bloch_animation.gif")
    """
    if not _has_matplotlib():
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    import numpy as np
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere (static)
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.sin(v), np.cos(u))
    ys = np.outer(np.sin(v), np.sin(u))
    zs = np.outer(np.cos(v), np.ones_like(u))
    ax.plot_surface(xs, ys, zs, alpha=0.1, color='cyan', edgecolor='none')
    
    # Draw axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', linewidth=0.5, alpha=0.3)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere Animation')
    ax.set_box_aspect([1, 1, 1])
    
    # Get all coordinates
    coords = [state_to_bloch(state) for state in states]
    
    # Initialize quiver
    x0, y0, z0 = coords[0]
    quiver = ax.quiver(0, 0, 0, x0, y0, z0, color='red', 
                       arrow_length_ratio=0.1, linewidth=3)
    point = ax.scatter([x0], [y0], [z0], c='red', s=100)
    
    def update(frame):
        x, y, z = coords[frame]
        quiver.remove()
        new_quiver = ax.quiver(0, 0, 0, x, y, z, color='red',
                                arrow_length_ratio=0.1, linewidth=3)
        point._offsets3d = ([x], [y], [z])
        return [new_quiver, point]
    
    anim = FuncAnimation(fig, update, frames=len(states), interval=interval, blit=False)
    
    if filename:
        anim.save(filename, writer='pillow', fps=1000/interval)
        print(f"Saved animation to {filename}")
    else:
        plt.show()


# ============================================================
# Exports
# ============================================================

__all__ = [
    'state_to_bloch',
    'bloch_to_state',
    'spherical_to_bloch',
    'bloch_to_spherical',
    'bloch_sphere',
    'plot_multiple_states',
    'animate_bloch',
]