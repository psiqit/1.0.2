"""
psiqit/math/calculus.py
============================================================
Calculus Operations for Quantum Mechanics
============================================================

Numerical calculus operations essential for quantum mechanics and
quantum computing applications:

    • Derivatives (first, second, partial, gradient, Jacobian, Hessian, Laplacian)
    • Integrals (definite, indefinite, double, triple)
    • Special derivatives for quantum wave functions (Gaussian, sine, cosine, plane wave)
    • Taylor series expansion and differential operators

This module provides numerical approximations using finite difference methods,
suitable for problems where analytical derivatives are not available or
when working with experimental data.

Example:
    >>> from psiqit.math.calculus import derivative, integral
    >>> 
    >>> # Derivative of x² at x=2
    >>> f = lambda x: x**2
    >>> df = derivative(f, 2.0)  # Should be 4.0
    >>> print(f"{df:.6f}")
    4.000000
    >>> 
    >>> # Definite integral of x² from 0 to 1
    >>> int_f = integral(f, 0, 1)  # Should be 1/3 ≈ 0.333333
    >>> print(f"{int_f:.6f}")
    0.333333
    >>> 
    >>> # Gradient of a 2D function
    >>> g = lambda x, y: x**2 + y**2
    >>> grad = gradient(g, [1.0, 2.0])  # [2x, 2y] = [2, 4]
    >>> print(grad)
    [2.0, 4.0]

References:
    Numerical methods are based on standard finite difference approximations.
    For quantum mechanics applications, see:
    R. P. Feynman and A. R. Hibbs, "Quantum Mechanics and Path Integrals,"
    McGraw-Hill, 1965.
"""

import math
from typing import Callable, List, Union, Optional, Tuple


# ============================================================
# Derivatives
# ============================================================

def derivative(f: Callable, x: float, h: float = 1e-5, order: int = 1) -> float:
    """
    Compute numerical derivative using central finite differences.

    For order=1 (first derivative): (f(x+h) - f(x-h)) / (2h)
    For order=2 (second derivative): (f(x+h) - 2f(x) + f(x-h)) / h²
    For order=3: (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h³)

    Args:
        f: Function to differentiate.
        x: Point at which to evaluate the derivative.
        h: Step size (default: 1e-5).
        order: Derivative order (1, 2, or 3).

    Returns:
        Numerical approximation of the derivative.

    Examples:
        >>> f = lambda x: x**3
        >>> derivative(f, 2.0, order=1)  # 3x² = 12
        12.0
        >>> derivative(f, 2.0, order=2)  # 6x = 12
        12.0
        >>> derivative(f, 2.0, order=3)  # 6
        6.0
    """
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    
    elif order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    
    elif order == 3:
        return (f(x + 2*h) - 2*f(x + h) + 2*f(x - h) - f(x - 2*h)) / (2 * h**3)
    
    else:
        def first_derivative(x):
            return derivative(f, x, h, 1)
        return derivative(first_derivative, x, h, order - 1)


def hessian(f: Callable, point: List[float], h: float = 1e-5) -> List[List[float]]:
    """
    Compute the Hessian matrix of a multivariate function.

    The Hessian matrix H contains second partial derivatives:
        H_ij = ∂²f/∂x_i∂x_j

    Args:
        f: Function f(x0, x1, ..., xn) returning a scalar.
        point: List of coordinates [x0, x1, ..., xn].
        h: Step size for finite differences (default: 1e-5).

    Returns:
        n×n Hessian matrix as a list of lists.

    Examples:
        >>> f = lambda x, y: x**2 * y**2
        >>> H = hessian(f, [1.0, 1.0])
        >>> print(H[0][0])  # ∂²f/∂x² = 2y² = 2
        2.0
        >>> print(H[0][1])  # ∂²f/∂x∂y = 4xy = 4
        4.0
    """
    n = len(point)
    hess = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Second derivative ∂²f/∂x_i²
                point_plus = point.copy()
                point_minus = point.copy()
                point_plus[i] += h
                point_minus[i] -= h
                
                d2f = (f(*point_plus) - 2 * f(*point) + f(*point_minus)) / (h * h)
                hess[i][j] = d2f
                hess[j][i] = d2f
            else:
                # Mixed partial ∂²f/∂x_i∂x_j
                point_pp = point.copy()
                point_pm = point.copy()
                point_mp = point.copy()
                point_mm = point.copy()
                
                point_pp[i] += h; point_pp[j] += h
                point_pm[i] += h; point_pm[j] -= h
                point_mp[i] -= h; point_mp[j] += h
                point_mm[i] -= h; point_mm[j] -= h
                
                d2f = (f(*point_pp) - f(*point_pm) - f(*point_mp) + f(*point_mm)) / (4 * h * h)
                hess[i][j] = d2f
                hess[j][i] = d2f
    
    return hess


def laplacian(f: Callable, point: List[float], h: float = 1e-5) -> float:
    """
    Compute the Laplacian ∇²f = Σ ∂²f/∂x_i² at a point.

    The Laplacian is the sum of second partial derivatives.

    Args:
        f: Function f(x0, x1, ..., xn) returning a scalar.
        point: List of coordinates [x0, x1, ..., xn].
        h: Step size for finite differences (default: 1e-5).

    Returns:
        Laplacian value.

    Examples:
        >>> f = lambda x, y: x**2 + y**2
        >>> laplacian(f, [1.0, 1.0])  # ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4
        4.0
    """
    laplacian_val = 0.0
    for i in range(len(point)):
        point_plus = point.copy()
        point_minus = point.copy()
        point_plus[i] += h
        point_minus[i] -= h
        d2f = (f(*point_plus) - 2 * f(*point) + f(*point_minus)) / (h * h)
        laplacian_val += d2f
    return laplacian_val


def partial_derivative(f: Callable, point: List[float], var_index: int, 
                       h: float = 1e-7) -> float:
    """
    Compute partial derivative ∂f/∂x_i at a given point.

    Args:
        f: Function f(x0, x1, ..., xn).
        point: List of coordinates.
        var_index: Index of variable to differentiate.
        h: Step size (default: 1e-7).

    Returns:
        Partial derivative value.

    Examples:
        >>> f = lambda x, y: x**2 * y
        >>> partial_derivative(f, [2.0, 3.0], 0)  # ∂f/∂x = 2xy = 12
        12.0
        >>> partial_derivative(f, [2.0, 3.0], 1)  # ∂f/∂y = x² = 4
        4.0
    """
    point_plus = point.copy()
    point_minus = point.copy()
    point_plus[var_index] += h
    point_minus[var_index] -= h
    
    return (f(*point_plus) - f(*point_minus)) / (2 * h)


def gradient(f: Callable, point: List[float], h: float = 1e-7) -> List[float]:
    """
    Compute the gradient vector ∇f at a given point.

    The gradient is a vector of all first partial derivatives.

    Args:
        f: Function f(x0, x1, ..., xn).
        point: List of coordinates.
        h: Step size (default: 1e-7).

    Returns:
        List of partial derivatives [∂f/∂x0, ∂f/∂x1, ...].

    Examples:
        >>> f = lambda x, y: x**2 + y**2
        >>> gradient(f, [1.0, 2.0])  # [2x, 2y] = [2, 4]
        [2.0, 4.0]
    """
    return [partial_derivative(f, point, i, h) for i in range(len(point))]


def jacobian(f: Callable, point: List[float], h: float = 1e-7) -> List[List[float]]:
    """
    Compute the Jacobian matrix of a vector-valued function.

    The Jacobian matrix J_ij = ∂f_i/∂x_j.

    Args:
        f: Function returning list of values f0(x), f1(x), ...
        point: List of coordinates.
        h: Step size (default: 1e-7).

    Returns:
        Jacobian matrix (m × n) where m is output dimension, n is input dimension.

    Examples:
        >>> f = lambda x, y: [x**2 * y, x * y**2]
        >>> J = jacobian(f, [2.0, 3.0])
        >>> print(J[0][0])  # ∂f0/∂x = 2xy = 12
        12.0
    """
    n = len(point)
    f0 = f(*point)
    m = len(f0) if isinstance(f0, (list, tuple)) else 1
    
    if m == 1:
        # Single output function
        return [gradient(f, point, h)]
    
    # Multi-output function
    jac = []
    for i in range(m):
        def fi(*args):
            return f(*args)[i]
        jac.append(gradient(fi, point, h))
    
    return jac


# ============================================================
# Integrals
# ============================================================

def integral(f: Callable, a: float, b: float, method: str = 'simpson', 
             n: int = 1000) -> float:
    """
    Compute definite integral using numerical methods.

    Supported methods:
        - 'simpson': Simpson's rule (most accurate for smooth functions)
        - 'trapezoid': Trapezoidal rule
        - 'midpoint': Midpoint rule

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.
        method: Integration method ('simpson', 'trapezoid', or 'midpoint').
        n: Number of intervals (higher = more accurate).

    Returns:
        ∫_a^b f(x) dx.

    Examples:
        >>> integral(lambda x: x**2, 0, 1)  # 1/3 ≈ 0.333333
        0.3333333333333333
        >>> integral(lambda x: math.sin(x), 0, math.pi, method='simpson')
        2.0
    """
    if method == 'trapezoid':
        h = (b - a) / n
        result = (f(a) + f(b)) / 2
        for i in range(1, n):
            result += f(a + i * h)
        return result * h
    
    elif method == 'simpson':
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        result = f(a) + f(b)
        for i in range(1, n):
            if i % 2 == 0:
                result += 2 * f(a + i * h)
            else:
                result += 4 * f(a + i * h)
        return result * h / 3
    
    else:  # midpoint
        h = (b - a) / n
        result = 0.0
        for i in range(n):
            result += f(a + (i + 0.5) * h)
        return result * h


def integral_2d(f: Callable, x_range: Tuple[float, float], 
                y_range: Tuple[float, float], 
                nx: int = 100, ny: int = 100) -> float:
    """
    Compute double integral ∫∫ f(x,y) dx dy using the midpoint method.

    Args:
        f: Function f(x, y).
        x_range: (x_min, x_max).
        y_range: (y_min, y_max).
        nx: Number of intervals in x-direction.
        ny: Number of intervals in y-direction.

    Returns:
        Double integral value.

    Examples:
        >>> f = lambda x, y: x * y
        >>> integral_2d(f, (0, 1), (0, 1))  # 0.25
        0.25
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    result = 0.0
    for i in range(nx):
        x = x_min + (i + 0.5) * dx
        for j in range(ny):
            y = y_min + (j + 0.5) * dy
            result += f(x, y) * dx * dy
    
    return result


def integral_3d(f: Callable, x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                z_range: Tuple[float, float],
                nx: int = 50, ny: int = 50, nz: int = 50) -> float:
    """
    Compute triple integral ∫∫∫ f(x,y,z) dx dy dz using the midpoint method.

    Args:
        f: Function f(x, y, z).
        x_range, y_range, z_range: (min, max) for each dimension.
        nx, ny, nz: Number of intervals in each dimension.

    Returns:
        Triple integral value.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz
    
    result = 0.0
    for i in range(nx):
        x = x_min + (i + 0.5) * dx
        for j in range(ny):
            y = y_min + (j + 0.5) * dy
            for k in range(nz):
                z = z_min + (k + 0.5) * dz
                result += f(x, y, z) * dx * dy * dz
    
    return result


def indefinite_integral(f: Callable, x: float, C: float = 0.0, 
                        a: float = 0.0, method: str = 'simpson') -> float:
    """
    Compute indefinite integral F(x) = ∫_a^x f(t) dt + C.

    Args:
        f: Function to integrate.
        x: Upper limit.
        C: Integration constant.
        a: Lower limit (default: 0).
        method: Integration method ('simpson', 'trapezoid', or 'midpoint').

    Returns:
        F(x).

    Examples:
        >>> indefinite_integral(lambda x: 2*x, 2)  # x² from 0 to 2 = 4
        4.0
    """
    return integral(f, a, x, method) + C


# ============================================================
# Derivatives of Common Quantum Functions
# ============================================================

def derivative_gaussian(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    Compute derivative of a Gaussian wavepacket.

    d/dx [exp(-(x-μ)²/(2σ²))] = -((x-μ)/σ²) * exp(-(x-μ)²/(2σ²))

    Args:
        x: Position.
        mu: Mean (center) of the Gaussian.
        sigma: Standard deviation (width).

    Returns:
        Derivative value.

    Examples:
        >>> derivative_gaussian(0)  # At center, derivative is 0
        0.0
    """
    gauss = math.exp(-(x - mu)**2 / (2 * sigma**2))
    return -gauss * (x - mu) / sigma**2


def derivative_sine_wave(x: float, k: float) -> float:
    """
    Compute derivative of a sine wave.

    d/dx sin(kx) = k cos(kx)

    Args:
        x: Position.
        k: Wave number.

    Returns:
        Derivative value.
    """
    return k * math.cos(k * x)


def derivative_cosine_wave(x: float, k: float) -> float:
    """
    Compute derivative of a cosine wave.

    d/dx cos(kx) = -k sin(kx)

    Args:
        x: Position.
        k: Wave number.

    Returns:
        Derivative value.
    """
    return -k * math.sin(k * x)


def derivative_plane_wave(x: float, k: float) -> complex:
    """
    Compute derivative of a plane wave (complex exponential).

    d/dx e^{ikx} = ik e^{ikx}

    Args:
        x: Position.
        k: Wave number.

    Returns:
        Complex derivative value.
    """
    return 1j * k * math.cos(k * x) - k * math.sin(k * x)


# ============================================================
# Utility Functions
# ============================================================

def taylor_series(f: Callable, x0: float, n: int = 5, x: float = None) -> str:
    """
    Generate Taylor series expansion of a function around x0.

    The Taylor series is: P(x) = f(x0) + f'(x0)(x-x0) + f''(x0)(x-x0)²/2! + ...

    Args:
        f: Function to expand.
        x0: Expansion point.
        n: Number of terms.
        x: If provided, evaluate the series at this point.

    Returns:
        String representation of the series or evaluated value.

    Examples:
        >>> taylor_series(lambda x: math.sin(x), 0, n=4)
        'P(x) = 0.0000 + 1.0000·(x-0) + 0.0000·(x-0)^2/2! + -1.0000·(x-0)^3/3!'
    """
    terms = []
    coeffs = []
    
    for i in range(n):
        df = derivative(f, x0, order=i) if i > 0 else f(x0)
        coeffs.append(df)
        if i == 0:
            terms.append(f"{df:.4f}")
        elif i == 1:
            terms.append(f"{df:.4f}·(x-{x0})")
        else:
            terms.append(f"{df:.4f}·(x-{x0})^{i}/{i}!")
    
    series = "P(x) = " + " + ".join(terms)
    
    if x is not None:
        result = 0.0
        for i, coeff in enumerate(coeffs):
            result += coeff * (x - x0)**i / math.factorial(i)
        return f"{series}\nP({x}) = {result:.6f}"
    
    return series


def diff_operator(f: Callable, x: float, h: float = 1e-7) -> Callable:
    """
    Create a differential operator function.

    Returns a function that computes the derivative of f at any point.

    Args:
        f: Function to differentiate.
        x: Initial point (not used in returned function).
        h: Step size for finite differences.

    Returns:
        Function that computes derivative at a given point.

    Examples:
        >>> d_dx = diff_operator(lambda x: x**3, 0)
        >>> d_dx(2)  # derivative of x³ at x=2 is 12
        12.0
    """
    def derivative_at_point(x0: float):
        return derivative(f, x0, h)
    return derivative_at_point


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Derivatives
    'derivative', 'partial_derivative', 'gradient', 'jacobian', 'hessian',
    'laplacian',
    # Integrals
    'integral', 'integral_2d', 'integral_3d', 'indefinite_integral',
    # Quantum derivatives
    'derivative_gaussian', 'derivative_sine_wave', 'derivative_cosine_wave',
    'derivative_plane_wave',
    # Utilities
    'taylor_series', 'diff_operator',
]