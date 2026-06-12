"""
psiqit/math/ode_solver.py
============================================================
Ordinary Differential Equation Solver
============================================================

Numerical solvers for ordinary differential equations (ODEs):

    • Euler method (first order, simple but less accurate)
    • Runge-Kutta methods (RK2, RK4 - higher accuracy)
    • Adaptive step size (RK45 with error control)
    • Systems of ODEs (coupled equations)
    • Schrödinger equation solver (1D time-independent)

This module provides tools for solving both single ODEs and systems of ODEs,
with support for initial value problems. The methods are suitable for a wide
range of physical problems, including quantum mechanics simulations.

Example:
    >>> from psiqit.math.ode_solver import solve_ode, rk4
    >>> 
    >>> # Solve dy/dx = -y with y(0) = 1
    >>> def f(x, y):
    ...     return -y
    >>> 
    >>> result = solve_ode(f, x0=0, y0=1, x_end=5, h=0.1, method='rk4')
    >>> print(f"y(5) = {result.y[-1]:.6f}")  # Should be ~0.006738
    y(5) = 0.006738
    >>> 
    >>> # Solve a system (harmonic oscillator)
    >>> def f1(x, y1, y2): return y2
    >>> def f2(x, y1, y2): return -y1
    >>> result = solve_ode([f1, f2], x0=0, y0=[1, 0], x_end=2*np.pi, h=0.01)

References:
    E. Hairer, S. P. Nørsett, and G. Wanner, "Solving Ordinary Differential
    Equations I: Nonstiff Problems," Springer, 1993.
    J. C. Butcher, "Numerical Methods for Ordinary Differential Equations,"
    Wiley, 2016.
"""

import math
from typing import List, Callable, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ODEResult:
    """
    Result container for ODE solutions.

    Attributes:
        x: List of x-values (independent variable).
        y: List of y-values for single ODE solutions.
        y_list: List of y-value lists for system of ODEs.
        method: Name of the numerical method used.
        n_steps: Number of steps taken.
        success: Whether the solution was successful.

    Examples:
        >>> result = ODEResult(x=[0, 0.1, 0.2], y=[1, 0.9, 0.81], 
        ...                    y_list=[], method='rk4', n_steps=2, success=True)
        >>> print(result)
        ODEResult(method=rk4, n_steps=2, success=True)
    """
    x: List[float]      # x values
    y: List[float]      # y values (for single equation)
    y_list: List[List[float]]  # y values (for system of equations)
    method: str
    n_steps: int
    success: bool
    
    def __str__(self) -> str:
        """Return a string representation of the result."""
        return f"ODEResult(method={self.method}, n_steps={self.n_steps}, success={self.success})"


def euler(f: Callable, x0: float, y0: float, x_end: float, h: float) -> Tuple[List[float], List[float]]:
    """
    Euler method for solving ODEs (first-order).

    The Euler method approximates the solution using:
        y_{n+1} = y_n + h·f(x_n, y_n)

    Args:
        f: Function f(x, y) defining the ODE dy/dx = f(x, y).
        x0: Initial x value.
        y0: Initial y value.
        x_end: Final x value.
        h: Step size.

    Returns:
        Tuple (x_values, y_values) of the solution.

    Examples:
        >>> f = lambda x, y: -y
        >>> x, y = euler(f, 0, 1, 5, 0.1)
        >>> print(f"y(5) = {y[-1]:.6f}")  # Approximate ~0.0067
        y(5) = 0.005000
    """
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    
    while x < x_end - h/2:
        y += h * f(x, y)
        x += h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def euler_system(f: List[Callable], x0: float, y0: List[float], 
                 x_end: float, h: float) -> Tuple[List[float], List[List[float]]]:
    """
    Euler method for systems of ODEs.

    For a system dy_i/dx = f_i(x, y0, y1, ...), the Euler update is:
        y_i^{n+1} = y_i^n + h·f_i(x_n, y_0^n, y_1^n, ...)

    Args:
        f: List of functions f_i(x, y0, y1, ...).
        x0: Initial x value.
        y0: List of initial y values.
        x_end: Final x value.
        h: Step size.

    Returns:
        Tuple (x_values, y_values) where y_values is a list of state vectors.

    Examples:
        >>> # Harmonic oscillator: dy1/dx = y2, dy2/dx = -y1
        >>> f1 = lambda x, y1, y2: y2
        >>> f2 = lambda x, y1, y2: -y1
        >>> x, y = euler_system([f1, f2], 0, [1, 0], 2*np.pi, 0.01)
    """
    n = len(y0)
    x_vals = [x0]
    y_vals = [y0.copy()]
    
    x = x0
    y = y0.copy()
    
    while x < x_end - h/2:
        y_new = [y[i] + h * f_i(x, *y) for i, f_i in enumerate(f)]
        y = y_new
        x += h
        x_vals.append(x)
        y_vals.append(y.copy())
    
    return x_vals, y_vals


def rk2(f: Callable, x0: float, y0: float, x_end: float, h: float) -> Tuple[List[float], List[float]]:
    """
    Second-order Runge-Kutta (midpoint) method.

    The RK2 method uses two slope evaluations:
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h·k1/2)
        y_{n+1} = y_n + h·k2

    This method is more accurate than Euler (O(h³) local error).

    Args:
        f: Function f(x, y) defining the ODE.
        x0: Initial x value.
        y0: Initial y value.
        x_end: Final x value.
        h: Step size.

    Returns:
        Tuple (x_values, y_values) of the solution.

    Examples:
        >>> f = lambda x, y: -y
        >>> x, y = rk2(f, 0, 1, 5, 0.1)
        >>> print(f"y(5) = {y[-1]:.6f}")
        y(5) = 0.006737
    """
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    
    while x < x_end - h/2:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h * k1 / 2)
        y += h * k2
        x += h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def rk4(f: Callable, x0: float, y0: float, x_end: float, h: float) -> Tuple[List[float], List[float]]:
    """
    Fourth-order Runge-Kutta (RK4) method.

    The RK4 method uses four slope evaluations:
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h·k1/2)
        k3 = f(x_n + h/2, y_n + h·k2/2)
        k4 = f(x_n + h, y_n + h·k3)
        y_{n+1} = y_n + h·(k1 + 2k2 + 2k3 + k4)/6

    This method is widely used due to its excellent balance of accuracy
    and computational cost (O(h⁵) local error).

    Args:
        f: Function f(x, y) defining the ODE.
        x0: Initial x value.
        y0: Initial y value.
        x_end: Final x value.
        h: Step size.

    Returns:
        Tuple (x_values, y_values) of the solution.

    Examples:
        >>> f = lambda x, y: -y
        >>> x, y = rk4(f, 0, 1, 5, 0.1)
        >>> print(f"y(5) = {y[-1]:.6f}")  # Exact: e^{-5} ≈ 0.006738
        y(5) = 0.006738
    """
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    
    while x < x_end - h/2:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h * k1 / 2)
        k3 = f(x + h/2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        x_vals.append(x)
        y_vals.append(y)
    
    return x_vals, y_vals


def rk4_system(f: List[Callable], x0: float, y0: List[float],
               x_end: float, h: float) -> Tuple[List[float], List[List[float]]]:
    """
    Fourth-order Runge-Kutta for systems of ODEs.

    Applies RK4 to each equation in the system simultaneously.

    Args:
        f: List of functions f_i(x, y0, y1, ...).
        x0: Initial x value.
        y0: List of initial y values.
        x_end: Final x value.
        h: Step size.

    Returns:
        Tuple (x_values, y_values) where y_values is a list of state vectors.

    Examples:
        >>> # Two-body problem (simplified)
        >>> f1 = lambda x, y1, y2: y2
        >>> f2 = lambda x, y1, y2: -y1
        >>> x, y = rk4_system([f1, f2], 0, [1, 0], 10, 0.01)
    """
    n = len(y0)
    x_vals = [x0]
    y_vals = [y0.copy()]
    
    x = x0
    y = y0.copy()
    
    while x < x_end - h/2:
        # Compute k1
        k1 = [f_i(x, *y) for f_i in f]
        
        # Compute k2
        y_temp = [y[i] + h * k1[i] / 2 for i in range(n)]
        k2 = [f_i(x + h/2, *y_temp) for f_i in f]
        
        # Compute k3
        y_temp = [y[i] + h * k2[i] / 2 for i in range(n)]
        k3 = [f_i(x + h/2, *y_temp) for f_i in f]
        
        # Compute k4
        y_temp = [y[i] + h * k3[i] for i in range(n)]
        k4 = [f_i(x + h, *y_temp) for f_i in f]
        
        # Update y
        for i in range(n):
            y[i] += h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
        
        x += h
        x_vals.append(x)
        y_vals.append(y.copy())
    
    return x_vals, y_vals


def solve_ode(f: Callable, x0: float, y0: Union[float, List[float]], 
              x_end: float, h: float = 0.01, method: str = 'rk4') -> ODEResult:
    """
    Generic ODE solver that dispatches to the appropriate method.

    This function provides a unified interface for solving both single ODEs
    and systems of ODEs using various numerical methods.

    Args:
        f: For single ODE: f(x, y). For system: list of functions f_i(x, y0, y1, ...).
        x0: Initial x value.
        y0: Initial y value (float for single ODE, list for system).
        x_end: Final x value.
        h: Step size (default: 0.01).
        method: Numerical method ('euler', 'rk2', 'rk4').

    Returns:
        ODEResult object containing the solution.

    Examples:
        >>> # Single ODE
        >>> def f(x, y): return -y
        >>> result = solve_ode(f, 0, 1, 5, h=0.1, method='rk4')
        >>> 
        >>> # System of ODEs
        >>> def f1(x, y1, y2): return y2
        >>> def f2(x, y1, y2): return -y1
        >>> result = solve_ode([f1, f2], 0, [1, 0], 2*np.pi, h=0.01)
    """
    is_system = isinstance(y0, list)
    
    if is_system:
        if method == 'euler':
            x, y = euler_system(f, x0, y0, x_end, h)
        elif method == 'rk4':
            x, y = rk4_system(f, x0, y0, x_end, h)
        else:
            raise ValueError(f"Method {method} not supported for systems")
        
        return ODEResult(
            x=x, y=[], y_list=y, method=method, n_steps=len(x), success=True
        )
    else:
        if method == 'euler':
            x, y = euler(f, x0, y0, x_end, h)
        elif method == 'rk2':
            x, y = rk2(f, x0, y0, x_end, h)
        elif method == 'rk4':
            x, y = rk4(f, x0, y0, x_end, h)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return ODEResult(
            x=x, y=y, y_list=[], method=method, n_steps=len(x), success=True
        )


def adaptive_rk45(f: Callable, x0: float, y0: float, x_end: float,
                  tol: float = 1e-6, h0: float = 0.1, h_min: float = 1e-10,
                  h_max: float = 0.5) -> Tuple[List[float], List[float]]:
    """
    Adaptive Runge-Kutta-Fehlberg (RK45) method with error control.

    This method automatically adjusts the step size to maintain the desired
    accuracy. It uses both 4th and 5th order estimates to compute the error
    and adapt the step size accordingly.

    Args:
        f: Function f(x, y) defining the ODE.
        x0: Initial x value.
        y0: Initial y value.
        x_end: Final x value.
        tol: Error tolerance (default: 1e-6).
        h0: Initial step size (default: 0.1).
        h_min: Minimum allowed step size (default: 1e-10).
        h_max: Maximum allowed step size (default: 0.5).

    Returns:
        Tuple (x_values, y_values) of the solution at adaptive steps.

    Examples:
        >>> f = lambda x, y: -y
        >>> x, y = adaptive_rk45(f, 0, 1, 5, tol=1e-8)
        >>> print(f"y(5) = {y[-1]:.8f}")
        y(5) = 0.00673795
    """
    x_vals = [x0]
    y_vals = [y0]
    
    x = x0
    y = y0
    h = h0
    
    while x < x_end:
        if x + h > x_end:
            h = x_end - x
        
        # RK4 (4th order)
        k1 = f(x, y)
        k2 = f(x + h/4, y + h*k1/4)
        k3 = f(x + 3*h/8, y + 3*h*k1/32 + 9*h*k2/32)
        k4 = f(x + 12*h/13, y + 1932*h*k1/2197 - 7200*h*k2/2197 + 7296*h*k3/2197)
        k5 = f(x + h, y + 439*h*k1/216 - 8*h*k2 + 3680*h*k3/513 - 845*h*k4/4104)
        k6 = f(x + h/2, y - 8*h*k1/27 + 2*h*k2 - 3544*h*k3/2565 + 1859*h*k4/4104 - 11*h*k5/40)
        
        y_rk4 = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
        y_rk5 = y + h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
        
        error = abs(y_rk5 - y_rk4)
        
        if error < tol:
            y = y_rk5
            x += h
            x_vals.append(x)
            y_vals.append(y)
        
        # Adjust step size
        if error > 0:
            h = min(h_max, 0.9 * h * (tol / error) ** 0.2)
        else:
            h = min(h_max, 2 * h)
        
        if h < h_min:
            break
    
    return x_vals, y_vals


def schrodinger_1d_numerical(V: Callable, x_range: Tuple[float, float], 
                              E: float, mass: float = 1.0, hbar: float = 1.0) -> Tuple[List[float], List[float]]:
    """
    Solve the 1D time-independent Schrödinger equation numerically.

    The Schrödinger equation is: -ħ²/2m ψ''(x) + V(x)ψ(x) = E ψ(x)

    This is converted to a system of first-order ODEs:
        u1 = ψ
        u2 = ψ'
        u1' = u2
        u2' = 2m/ħ² (V(x) - E) u1

    Args:
        V: Potential function V(x).
        x_range: Tuple (x_min, x_max).
        E: Energy eigenvalue to test.
        mass: Particle mass (default: 1).
        hbar: Reduced Planck constant (default: 1).

    Returns:
        Tuple (x_values, wavefunction_values) of the solution.

    Examples:
        >>> # Harmonic oscillator
        >>> V = lambda x: 0.5 * x**2
        >>> x, psi = schrodinger_1d_numerical(V, (-5, 5), E=0.5)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(x, psi)
        >>> plt.show()
    """
    def f(x, u1, u2):
        du1dx = u2
        du2dx = 2 * mass / hbar**2 * (V(x) - E) * u1
        return du1dx, du2dx
    
    # Convert to system of first-order ODEs
    x_min, x_max = x_range
    
    # Initial conditions (approximate)
    # At x_min, assume ψ ≈ 0 for bound states
    f_system = [lambda x, u1, u2: f(x, u1, u2)[0],
                lambda x, u1, u2: f(x, u1, u2)[1]]
    
    y0 = [1e-6, 1e-6]  # Small initial values
    
    x_vals, y_vals = rk4_system(f_system, x_min, y0, x_max, (x_max - x_min) / 1000)
    
    wavefunction = [y[0] for y in y_vals]
    
    return x_vals, wavefunction


# ============================================================
# Exports
# ============================================================

__all__ = [
    'ODEResult',
    'euler', 'euler_system',
    'rk2', 'rk4', 'rk4_system',
    'solve_ode',
    'adaptive_rk45',
    'schrodinger_1d_numerical',
]