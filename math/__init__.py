"""
psiqit/math/__init__.py
============================================================
Mathematics Package - Complete Mathematical Toolkit for Quantum Computing
============================================================

This package provides a comprehensive set of mathematical operations
for quantum computing, including:

    • Linear algebra (matrices, vectors, eigenvalues, etc.)
    • Calculus (derivatives, integrals, gradient, Jacobian, Hessian)
    • ODE solvers (Euler, Runge-Kutta, adaptive methods)
    • PDE solvers (heat, wave, Schrödinger equations)
    • Symbolic mathematics (via SymPy integration)
    • Physical constants and unit conversions

The math package is designed to be self-contained with minimal dependencies.
Core linear algebra operations (qalgebra) have no external dependencies,
while advanced features (symbolic math) optionally use SymPy.

Example:
    >>> from psiqit.math import eye, kron, derivative, integral, solve_ode
    >>> 
    >>> # Matrix operations
    >>> I = eye(2)
    >>> H = [[1, 1], [1, -1]]
    >>> 
    >>> # Calculus
    >>> df = derivative(lambda x: x**2, 2.0)  # 4.0
    >>> 
    >>> # ODE solving
    >>> def f(x, y): return -y
    >>> result = solve_ode(f, x0=0, y0=1, x_end=5, method='rk4')
"""

# ============================================================
# Quantum Algebra (Core)
# ============================================================

from .qalgebra import (
    # Types
    Vector,
    Matrix,
    # Constants
    PI, TAU, EULER, INF,
    SQRT2, SQRT3, SQRT2_INV, SQRT3_INV,
    LN2, LN10, LOG2_E,
    PHI, PHI_INV,
    # Physical constants
    HBAR, H_PLANCK, C, E_CHARGE,
    M_ELECTRON, M_PROTON, M_NEUTRON, K_B,
    # Complex operations
    is_real, is_imag, phase, magnitude, conjugate,
    from_polar, to_polar, cis,
    # Matrix operations
    eye, zeros, mul, add, transpose, conj, dagger, trace, kron,
    # Vector operations
    inner, outer, norm, normalize,
    # Advanced
    determinant, inverse, eigenvalues, eigenvectors, expm, sqrt_matrix,
    # Validation
    is_unitary, is_hermitian, is_positive,
    # Probability & statistics
    is_distribution, normalize_probs, sample, entropy, max_entropy,
    born_rule, expectation, variance,
    # Utilities
    dimension, hilbert_space, display, commutator, anticommutator,
    # Unit conversions
    eV_to_J, J_to_eV, nm_to_m, m_to_nm,
    energy_to_frequency, frequency_to_energy,
)

# ============================================================
# Calculus Operations
# ============================================================

from .calculus import (
    # Derivatives
    derivative, partial_derivative, gradient, jacobian, hessian, laplacian,
    # Integrals
    integral, integral_2d, integral_3d, indefinite_integral,
    # Quantum derivatives
    derivative_gaussian, derivative_sine_wave,
    derivative_cosine_wave, derivative_plane_wave,
    # Utilities
    taylor_series, diff_operator,
)

# ============================================================
# ODE Solvers
# ============================================================

from .ode_solver import (
    ODEResult,
    euler, euler_system,
    rk2, rk4, rk4_system,
    solve_ode,
    adaptive_rk45,
    schrodinger_1d_numerical,
)

# ============================================================
# PDE Solvers
# ============================================================

from .pde_solver import (
    PDEResult,
    solve_heat_equation,
    solve_wave_equation,
    solve_schrodinger_1d,
    diffusion_equation_implicit,
)

# ============================================================
# Symbolic Mathematics (Optional - requires SymPy)
# ============================================================

# Try to import symbolic operations
try:
    from .symbolic import (
        create_symbols, create_matrix,
        solve_equation, solve_linear_system,
        differentiate, integrate_expression,
        simplify_expression, expand_expression,
        pauli_algebra, commutator_symbolic, anticommutator_symbolic,
        commutator_algebra_symbolic, hamiltonian_symbolic,
        eigenvalues_symbolic, eigenvectors_symbolic,
        schrodinger_1d_symbolic, infinite_well_states, harmonic_oscillator_states,
        Symbol, symbols, Matrix as SymMatrix, Function, Eq,
        I, pi, E,
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ============================================================
# Package Metadata
# ============================================================

__version__ = "1.0.2"
__author__ = "PSIQIT Contributors"

__all__ = [
    # Algebra
    'Vector', 'Matrix',
    'PI', 'TAU', 'EULER', 'INF',
    'SQRT2', 'SQRT3', 'SQRT2_INV', 'SQRT3_INV',
    'LN2', 'LN10', 'LOG2_E',
    'PHI', 'PHI_INV',
    'HBAR', 'H_PLANCK', 'C', 'E_CHARGE',
    'M_ELECTRON', 'M_PROTON', 'M_NEUTRON', 'K_B',
    'is_real', 'is_imag', 'phase', 'magnitude', 'conjugate',
    'from_polar', 'to_polar', 'cis',
    'eye', 'zeros', 'mul', 'add', 'transpose', 'conj', 'dagger', 'trace', 'kron',
    'inner', 'outer', 'norm', 'normalize',
    'determinant', 'inverse', 'eigenvalues', 'eigenvectors', 'expm', 'sqrt_matrix',
    'is_unitary', 'is_hermitian', 'is_positive',
    'is_distribution', 'normalize_probs', 'sample', 'entropy', 'max_entropy',
    'born_rule', 'expectation', 'variance',
    'dimension', 'hilbert_space', 'display', 'commutator', 'anticommutator',
    'eV_to_J', 'J_to_eV', 'nm_to_m', 'm_to_nm',
    'energy_to_frequency', 'frequency_to_energy',
    # Calculus
    'derivative', 'partial_derivative', 'gradient', 'jacobian', 'hessian', 'laplacian',
    'integral', 'integral_2d', 'integral_3d', 'indefinite_integral',
    'derivative_gaussian', 'derivative_sine_wave', 'derivative_cosine_wave', 'derivative_plane_wave',
    'taylor_series', 'diff_operator',
    # ODE
    'ODEResult', 'euler', 'euler_system', 'rk2', 'rk4', 'rk4_system',
    'solve_ode', 'adaptive_rk45', 'schrodinger_1d_numerical',
    # PDE
    'PDEResult', 'solve_heat_equation', 'solve_wave_equation',
    'solve_schrodinger_1d', 'diffusion_equation_implicit',
]

# Add symbolic exports if SymPy is available
if SYMPY_AVAILABLE:
    __all__ += [
        'create_symbols', 'create_matrix', 'solve_equation', 'solve_linear_system',
        'differentiate', 'integrate_expression', 'simplify_expression', 'expand_expression',
        'pauli_algebra', 'commutator_symbolic', 'anticommutator_symbolic',
        'commutator_algebra_symbolic', 'hamiltonian_symbolic',
        'eigenvalues_symbolic', 'eigenvectors_symbolic',
        'schrodinger_1d_symbolic', 'infinite_well_states', 'harmonic_oscillator_states',
        'Symbol', 'symbols', 'SymMatrix', 'Function', 'Eq', 'I', 'pi', 'E',
        'SYMPY_AVAILABLE',
    ]