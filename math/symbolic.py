"""
psiqit/math/symbolic.py
============================================================
Symbolic Mathematics for Quantum Mechanics
============================================================

Symbolic operations using SymPy for quantum mechanics and quantum computing:
    • Symbolic variables and expressions
    • Equation solving (algebraic and differential)
    • Differentiation and integration
    • Matrix operations with symbolic entries
    • Quantum operator algebra (Pauli matrices, commutators)
    • Schrödinger equation solvers

This module provides a convenient interface to SymPy's symbolic capabilities,
tailored for quantum mechanics applications. It requires SymPy to be installed.

Example:
    >>> from psiqit.math.symbolic import Symbol, solve, diff, integrate
    >>> x = Symbol('x')
    >>> expr = x**2 + 2*x + 1
    >>> solve(expr, x)  # [-1]
    >>> diff(expr, x)   # 2*x + 2
    >>> integrate(x**2, x)  # x**3/3

References:
    SymPy documentation: https://docs.sympy.org/
    A. Meurer et al., "SymPy: symbolic computing in Python,"
    PeerJ Computer Science, 3:e103, 2017.
"""

from typing import List, Union, Optional, Any, Dict, Tuple

# Try to import SymPy
try:
    import sympy as sp
    from sympy import (
        Symbol, symbols, Matrix, Function, 
        Eq, solve, diff, integrate, simplify, expand,
        I, pi, E, oo, sin, cos, tan, exp, log, sqrt,
        Rational, Integer, Float
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    # Create dummy classes for when SymPy is not available
    class Symbol:
        def __init__(self, *args, **kwargs):
            raise ImportError("SymPy not installed. Install with: pip install sympy")
    
    def dummy_func(*args, **kwargs):
        raise ImportError("SymPy not installed. Install with: pip install sympy")
    
    Symbol = dummy_func
    symbols = dummy_func
    Matrix = dummy_func
    Function = dummy_func
    Eq = dummy_func
    solve = dummy_func
    diff = dummy_func
    integrate = dummy_func
    simplify = dummy_func
    expand = dummy_func
    I = None
    pi = None
    E = None


# ============================================================
# Basic Symbolic Operations
# ============================================================

def create_symbols(names: str) -> List:
    """
    Create multiple symbolic variables.

    Args:
        names: Space-separated variable names, e.g., 'x y z'.

    Returns:
        List of Symbol objects.

    Examples:
        >>> x, y, z = create_symbols('x y z')
        >>> expr = x**2 + y**2 + z**2
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return symbols(names)


def create_matrix(rows: List[List], name: str = "M") -> Matrix:
    """
    Create a symbolic matrix.

    Args:
        rows: 2D list of expressions.
        name: Matrix name for display (not used in output).

    Returns:
        SymPy Matrix object.

    Examples:
        >>> M = create_matrix([[1, 2], [3, 4]])
        >>> M.det()
        -2
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return Matrix(rows)


def solve_equation(expr, var, **kwargs) -> List:
    """
    Solve an equation symbolically.

    Args:
        expr: Expression or Eq object (e.g., x**2 - 4 or Eq(x**2, 4)).
        var: Variable to solve for.
        **kwargs: Additional arguments passed to sympy.solve().

    Returns:
        List of solutions.

    Examples:
        >>> x = Symbol('x')
        >>> solve_equation(x**2 - 4, x)  # [-2, 2]
        >>> solve_equation(x**2 + 1, x)  # [-I, I]
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return solve(expr, var, **kwargs)


def solve_linear_system(A: Matrix, b: Matrix) -> List:
    """
    Solve a linear system Ax = b symbolically.

    Args:
        A: Coefficient matrix.
        b: Right-hand side vector.

    Returns:
        Solution vector as a set of tuples.

    Examples:
        >>> x, y = symbols('x y')
        >>> A = Matrix([[1, 1], [1, -1]])
        >>> b = Matrix([2, 0])
        >>> solve_linear_system(A, b)  # {(1, 1)}
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return sp.linsolve((A, b))


def differentiate(expr, var, order: int = 1):
    """
    Differentiate an expression symbolically.

    Args:
        expr: Expression to differentiate.
        var: Variable to differentiate with respect to.
        order: Derivative order (default: 1).

    Returns:
        Differentiated expression.

    Examples:
        >>> x = Symbol('x')
        >>> differentiate(x**3, x)  # 3*x**2
        >>> differentiate(x**3, x, order=2)  # 6*x
        >>> differentiate(sin(x), x)  # cos(x)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return diff(expr, var, order)


def integrate_expression(expr, var, limits: Optional[Tuple] = None):
    """
    Integrate an expression symbolically.

    Args:
        expr: Expression to integrate.
        var: Variable to integrate with respect to.
        limits: Optional tuple (lower, upper) for definite integral.

    Returns:
        Integrated expression (indefinite or definite).

    Examples:
        >>> x = Symbol('x')
        >>> integrate_expression(x**2, x)  # x**3/3
        >>> integrate_expression(x**2, x, (0, 1))  # 1/3
        >>> integrate_expression(sin(x), x)  # -cos(x)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    if limits:
        return integrate(expr, (var, limits[0], limits[1]))
    return integrate(expr, var)


def simplify_expression(expr) -> Any:
    """
    Simplify an expression symbolically.

    Args:
        expr: Expression to simplify.

    Returns:
        Simplified expression.

    Examples:
        >>> x = Symbol('x')
        >>> simplify_expression((x**2 - 1)/(x - 1))  # x + 1
        >>> simplify_expression(sin(x)**2 + cos(x)**2)  # 1
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return simplify(expr)


def expand_expression(expr) -> Any:
    """
    Expand an expression symbolically.

    Args:
        expr: Expression to expand.

    Returns:
        Expanded expression.

    Examples:
        >>> x = Symbol('x')
        >>> expand_expression((x + 1)**2)  # x**2 + 2*x + 1
        >>> expand_expression((x + y)**3)  # x**3 + 3*x**2*y + 3*x*y**2 + y**3
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return expand(expr)


# ============================================================
# Quantum Specific Symbolic Operations
# ============================================================

def pauli_algebra():
    """
    Create symbolic Pauli matrices.

    Returns:
        Dictionary containing:
            - 'sigma_x': Pauli X matrix
            - 'sigma_y': Pauli Y matrix
            - 'sigma_z': Pauli Z matrix
            - 'I': 2x2 identity matrix

    Examples:
        >>> pauli = pauli_algebra()
        >>> sx = pauli['sigma_x']
        >>> sy = pauli['sigma_y']
        >>> sz = pauli['sigma_z']
        >>> sx * sy  # i*sz
        Matrix([[I, 0], [0, -I]])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    sigma_x = Matrix([[0, 1], [1, 0]])
    sigma_y = Matrix([[0, -I], [I, 0]])
    sigma_z = Matrix([[1, 0], [0, -1]])
    
    return {
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'I': Matrix([[1, 0], [0, 1]])
    }


def commutator_symbolic(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the symbolic commutator [A, B] = AB - BA.

    Args:
        A: First operator (symbolic matrix).
        B: Second operator (symbolic matrix).

    Returns:
        Commutator matrix.

    Examples:
        >>> sigma = pauli_algebra()
        >>> sx = sigma['sigma_x']
        >>> sy = sigma['sigma_y']
        >>> commutator_symbolic(sx, sy)  # 2i*sigma_z
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return A * B - B * A


def anticommutator_symbolic(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the symbolic anticommutator {A, B} = AB + BA.

    Args:
        A: First operator (symbolic matrix).
        B: Second operator (symbolic matrix).

    Returns:
        Anticommutator matrix.

    Examples:
        >>> sigma = pauli_algebra()
        >>> sx = sigma['sigma_x']
        >>> anticommutator_symbolic(sx, sx)  # 2*I
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return A * B + B * A


def commutator_algebra_symbolic():
    """
    Demonstrate symbolic commutation relations for Pauli matrices.

    Returns:
        Dictionary of commutation relations:
            - '[σx, σy]': 2iσz
            - '[σy, σz]': 2iσx
            - '[σz, σx]': 2iσy

    Examples:
        >>> relations = commutator_algebra_symbolic()
        >>> relations['[σx, σy]']
        Matrix([[0, 2*I], [2*I, 0]])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    sigma = pauli_algebra()
    sx, sy, sz = sigma['sigma_x'], sigma['sigma_y'], sigma['sigma_z']
    
    relations = {
        '[σx, σy]': commutator_symbolic(sx, sy),
        '[σy, σz]': commutator_symbolic(sy, sz),
        '[σz, σx]': commutator_symbolic(sz, sx),
    }
    
    return relations


def hamiltonian_symbolic(terms: Dict[str, float]) -> Matrix:
    """
    Create a symbolic Hamiltonian from Pauli terms.

    Args:
        terms: Dictionary mapping Pauli strings to coefficients.
               e.g., {'X': 1.0, 'Z': 1.0} for H = X + Z
               Supports 'I', 'X', 'Y', 'Z'.

    Returns:
        Symbolic 2x2 Hamiltonian matrix.

    Examples:
        >>> H = hamiltonian_symbolic({'X': 1.0, 'Z': 1.0})
        >>> H
        Matrix([[1, 1], [1, -1]])
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    sigma = pauli_algebra()
    H = sp.zeros(2, 2)
    
    for pauli, coeff in terms.items():
        if pauli == 'I':
            H += coeff * sigma['I']
        elif pauli == 'X':
            H += coeff * sigma['sigma_x']
        elif pauli == 'Y':
            H += coeff * sigma['sigma_y']
        elif pauli == 'Z':
            H += coeff * sigma['sigma_z']
    
    return H


def eigenvalues_symbolic(matrix: Matrix) -> List:
    """
    Compute symbolic eigenvalues of a matrix.

    Args:
        matrix: Symbolic matrix.

    Returns:
        Dictionary of eigenvalues with their multiplicities.

    Examples:
        >>> H = hamiltonian_symbolic({'X': 1.0, 'Z': 1.0})
        >>> eigenvalues_symbolic(H)
        {sqrt(2): 1, -sqrt(2): 1}
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return matrix.eigenvals()


def eigenvectors_symbolic(matrix: Matrix) -> Dict:
    """
    Compute symbolic eigenvectors of a matrix.

    Args:
        matrix: Symbolic matrix.

    Returns:
        List of tuples (eigenvalue, multiplicity, eigenvectors).

    Examples:
        >>> H = hamiltonian_symbolic({'X': 1.0})
        >>> eigenvectors_symbolic(H)
        [(-1, 1, [Matrix([[-1], [1]])]), (1, 1, [Matrix([[1], [1]])])]
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    return matrix.eigenvects()


# ============================================================
# Schrödinger Equation Solvers (Symbolic)
# ============================================================

def schrodinger_1d_symbolic(potential: Any, mass: float = 1.0, hbar: float = 1.0):
    """
    Set up the 1D time-independent Schrödinger equation symbolically.

    The Schrödinger equation is: -ħ²/2m ψ''(x) + V(x)ψ(x) = E ψ(x)

    Args:
        potential: Symbolic potential V(x) (as SymPy expression).
        mass: Particle mass (default: 1).
        hbar: Reduced Planck constant (default: 1).

    Returns:
        SymPy Eq object representing the differential equation.

    Examples:
        >>> x = Symbol('x')
        >>> V = 0.5 * x**2  # Harmonic oscillator
        >>> eq = schrodinger_1d_symbolic(V)
        >>> eq
        Eq(-0.5*Derivative(psi(x), x, x) + 0.5*x**2*psi(x), E*psi(x))
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    x = Symbol('x')
    psi = Function('psi')(x)
    E = Symbol('E')
    
    # Kinetic term: -ħ²/2m * d²ψ/dx²
    kinetic = -hbar**2 / (2 * mass) * diff(psi, x, 2)
    
    # Schrödinger equation: Hψ = Eψ
    schrodinger_eq = Eq(kinetic + potential * psi, E * psi)
    
    return schrodinger_eq


def infinite_well_states(n: int, L: float = 1.0) -> Any:
    """
    Get the symbolic wavefunction for the infinite square well.

    The wavefunction is: ψ_n(x) = √(2/L) sin(nπx/L)

    Args:
        n: Quantum number (1, 2, 3, ...).
        L: Well width (default: 1).

    Returns:
        SymPy expression for ψ_n(x).

    Examples:
        >>> psi1 = infinite_well_states(1, L=1)
        >>> psi1
        sqrt(2)*sin(pi*x)
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    x = Symbol('x')
    return sqrt(2/L) * sin(n * pi * x / L)


def harmonic_oscillator_states(n: int, mass: float = 1.0, omega: float = 1.0) -> Any:
    """
    Get the symbolic wavefunction for the quantum harmonic oscillator.

    The wavefunction is: ψ_n(x) = N_n H_n(ξ) e^{-ξ²/2}
    where ξ = √(mω/ħ) x and N_n is the normalization constant.

    Args:
        n: Quantum number (0, 1, 2, ...).
        mass: Particle mass (default: 1).
        omega: Angular frequency (default: 1).

    Returns:
        SymPy expression for ψ_n(x).

    Examples:
        >>> psi0 = harmonic_oscillator_states(0)  # Ground state
        >>> psi1 = harmonic_oscillator_states(1)  # First excited
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy required for symbolic operations")
    
    x = Symbol('x')
    from sympy import hermite, factorial
    
    alpha = sqrt(mass * omega / hbar) if 'hbar' in dir() else sqrt(mass * omega)
    xi = alpha * x
    
    norm = (alpha / (sqrt(pi) * 2**n * factorial(n)))**(1/2)
    psi = norm * hermite(n, xi) * exp(-xi**2 / 2)
    
    return psi


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Basic
    'create_symbols', 'create_matrix',
    'solve_equation', 'solve_linear_system',
    'differentiate', 'integrate_expression',
    'simplify_expression', 'expand_expression',
    # Quantum
    'pauli_algebra', 'commutator_symbolic', 'anticommutator_symbolic',
    'commutator_algebra_symbolic', 'hamiltonian_symbolic',
    'eigenvalues_symbolic', 'eigenvectors_symbolic',
    # Schrödinger
    'schrodinger_1d_symbolic', 'infinite_well_states', 'harmonic_oscillator_states',
    # SymPy exports (if available)
    'Symbol', 'symbols', 'Matrix', 'Function', 'Eq', 'I', 'pi', 'E',
]