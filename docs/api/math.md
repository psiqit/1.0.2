# Mathematics API

## Module: `psiqit.math`

This module provides comprehensive mathematical tools for quantum computing, including linear algebra (matrices, vectors, eigenvalues), calculus (derivatives, integrals, gradient, Jacobian, Hessian), ODE solvers (Euler, RK2, RK4, adaptive), PDE solvers (heat, wave, Schrödinger), symbolic mathematics (SymPy integration), physical constants, and unit conversions.

**No external dependencies for core functionality!** Only symbolic math requires SymPy (optional).

---

## Quantum Algebra (`qalgebra.py`)

**Self-contained linear algebra for quantum computing** - No external dependencies. Provides matrix operations, vector operations, eigenvalues/eigenvectors, matrix exponential, and more.

### Type Aliases

| Type | Description |
|------|-------------|
| `Vector` | `List[complex]` - Complex vector |
| `Matrix` | `List[List[complex]]` - Complex matrix |

### Constants

#### Mathematical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PI` | 3.141592653589793 | π |
| `TAU` | 6.283185307179586 | 2π |
| `EULER` | 2.718281828459045 | e |
| `INF` | `float('inf')` | Infinity |
| `SQRT2` | 1.4142135623730951 | √2 |
| `SQRT3` | 1.7320508075688772 | √3 |
| `SQRT2_INV` | 0.7071067811865475 | 1/√2 |
| `SQRT3_INV` | 0.5773502691896258 | 1/√3 |
| `PHI` | 1.618033988749895 | Golden ratio |

#### Physical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `HBAR` | 1.054571817e-34 | Reduced Planck constant (J·s) |
| `H_PLANCK` | 6.62607015e-34 | Planck constant (J·s) |
| `C` | 299792458.0 | Speed of light (m/s) |
| `E_CHARGE` | 1.602176634e-19 | Elementary charge (C) |
| `M_ELECTRON` | 9.1093837015e-31 | Electron mass (kg) |
| `M_PROTON` | 1.67262192369e-27 | Proton mass (kg) |
| `M_NEUTRON` | 1.67492749804e-27 | Neutron mass (kg) |
| `K_B` | 1.380649e-23 | Boltzmann constant (J/K) |

### Complex Number Operations

| Function | Description | Example |
|----------|-------------|---------|
| `is_real(z, tol)` | Check if number is real | `is_real(3+0j)` → `True` |
| `is_imag(z, tol)` | Check if number is imaginary | `is_imag(0+2j)` → `True` |
| `phase(z)` | Argument/phase angle | `phase(1+1j)` → `π/4` |
| `magnitude(z)` | Modulus | `magnitude(3+4j)` → `5.0` |
| `from_polar(r, theta)` | Polar to complex | `from_polar(1, π)` → `-1` |
| `to_polar(z)` | Complex to polar | `to_polar(-1)` → `(1, π)` |
| `cis(theta)` | e^{iθ} = cosθ + i sinθ | `cis(π)` → `-1` |

### Matrix Operations

| Function | Description | Example |
|----------|-------------|---------|
| `eye(n)` | Identity matrix Iₙ | `eye(2)` → `[[1,0],[0,1]]` |
| `zeros(n, m)` | Zero matrix | `zeros(2,3)` → 2×3 zero matrix |
| `mul(A, B)` | Matrix multiplication | `mul(A, B)` |
| `add(A, B)` | Matrix addition | `add(A, B)` |
| `transpose(A)` | Transpose A^T | `transpose(A)` |
| `conj(A)` | Complex conjugate | `conj(A)` |
| `dagger(A)` | Hermitian conjugate A† | `dagger(A)` |
| `trace(A)` | Trace Tr(A) | `trace(A)` |
| `kron(A, B)` | Kronecker product A ⊗ B | `kron(A, B)` |

### Vector Operations

| Function | Description | Example |
|----------|-------------|---------|
| `inner(bra, ket)` | Inner product ⟨bra|ket⟩ | `inner([1,0], [1,0])` → `1` |
| `outer(ket, bra)` | Outer product |ket⟩⟨bra| | `outer([1,0], [1,0])` → `[[1,0],[0,0]]` |
| `norm(ket)` | Euclidean norm | `norm([1,1])` → `√2` |
| `normalize(ket)` | Normalize to unit norm | `normalize([1,1])` → `[1/√2, 1/√2]` |

### Advanced Operations

| Function | Description | Example |
|----------|-------------|---------|
| `determinant(A)` | Matrix determinant | `determinant([[1,2],[3,4]])` → `-2` |
| `inverse(A)` | Matrix inverse | `inverse(A)` |
| `eigenvalues(A)` | List of eigenvalues | `eigenvalues([[0,1],[1,0]])` → `[1, -1]` |
| `eigenvectors(A)` | Dict with values and vectors | `eigenvectors(A)` |
| `expm(A, terms)` | Matrix exponential e^A | `expm([[0,1],[0,0]])` → `[[1,1],[0,1]]` |
| `sqrt_matrix(A, tol)` | Square root √A | `sqrt_matrix(eye(2))` → `eye(2)` |

### Validation Functions

| Function | Description | Example |
|----------|-------------|---------|
| `is_unitary(U, tol)` | Check if U†U = I | `is_unitary(hadamard())` → `True` |
| `is_hermitian(H, tol)` | Check if H = H† | `is_hermitian(pauli_z())` → `True` |
| `is_positive(M, tol)` | Check if all eigenvalues ≥ 0 | `is_positive(rho)` |

### Probability & Statistics

| Function | Description | Example |
|----------|-------------|---------|
| `born_rule(amplitudes)` | Probability = |amplitude|² | `born_rule([0.707,0.707])` → `[0.5,0.5]` |
| `entropy(probs, base)` | Shannon entropy | `entropy([0.5,0.5], '2')` → `1.0` |
| `expectation(values, probs)` | Expectation value | `expectation([-1,1], [0.5,0.5])` → `0` |
| `variance(values, probs)` | Variance | `variance([-1,1], [0.5,0.5])` → `1` |

### Utilities

| Function | Description | Example |
|----------|-------------|---------|
| `commutator(A, B)` | [A, B] = AB - BA | `commutator(X, Z)` |
| `anticommutator(A, B)` | {A, B} = AB + BA | `anticommutator(X, X)` |
| `display(M, precision)` | Pretty-print matrix | `display(H)` |

### Unit Conversions

| Function | Description |
|----------|-------------|
| `eV_to_J(eV)` | Electronvolts → Joules |
| `J_to_eV(J)` | Joules → Electronvolts |
| `nm_to_m(nm)` | Nanometers → Meters |
| `m_to_nm(m)` | Meters → Nanometers |
| `energy_to_frequency(energy)` | Energy → Frequency (E = hν) |
| `frequency_to_energy(freq)` | Frequency → Energy |

### Example 1: Basic Matrix Operations

```python
from psiqit.math.qalgebra import eye, mul, kron, dagger

# Identity matrix
I = eye(2)
print(I)  # [[1.0, 0.0], [0.0, 1.0]]

# Pauli matrices
X = [[0, 1], [1, 0]]
Z = [[1, 0], [0, -1]]

# Matrix multiplication
XZ = mul(X, Z)
print(XZ)  # [[0, -1], [1, 0]]

# Kronecker product (X ⊗ I)
X_I = kron(X, I)
print(X_I)  # 4x4 matrix
Example 2: Eigenvalues and Eigenvectors
python
from psiqit.math.qalgebra import eigenvalues, eigenvectors

# Pauli X matrix
X = [[0, 1], [1, 0]]

# Eigenvalues
vals = eigenvalues(X)
print(f"Eigenvalues: {vals}")  # [1.0, -1.0]

# Eigenvectors
result = eigenvectors(X)
print(f"Eigenvectors: {result['vectors']}")
# [[0.707, 0.707], [0.707, -0.707]]
Example 3: Unitary and Hermitian Checks
python
from psiqit.math.qalgebra import is_unitary, is_hermitian

# Hadamard gate
H = [[1/1.414, 1/1.414], [1/1.414, -1/1.414]]

print(f"Unitary: {is_unitary(H)}")   # True
print(f"Hermitian: {is_hermitian(H)}")  # True

# Pauli X
X = [[0, 1], [1, 0]]
print(f"Unitary: {is_unitary(X)}")   # True
print(f"Hermitian: {is_hermitian(X)}")  # True
Example 4: Born Rule and Entropy
python
from psiqit.math.qalgebra import born_rule, entropy

# Quantum state amplitudes
amplitudes = [0.707, 0.707]  # |+⟩ state
probs = born_rule(amplitudes)
print(f"Probabilities: {probs}")  # [0.5, 0.5]

## Calculus (`calculus.py`)

Numerical calculus operations - Derivatives, integrals, gradient, Jacobian, Hessian, Laplacian.

### Derivatives

| Function | Description | Example |
|----------|-------------|---------|
| `derivative(f, x, h, order)` | Numerical derivative (order 1-3) | `derivative(lambda x: x**2, 2)` → `4.0` |
| `partial_derivative(f, point, var_index, h)` | Partial derivative | `∂f/∂x` at point |
| `gradient(f, point, h)` | Gradient vector ∇f | `[∂f/∂x, ∂f/∂y]` |
| `jacobian(f, point, h)` | Jacobian matrix J_ij = ∂f_i/∂x_j | For vector functions |
| `hessian(f, point, h)` | Hessian matrix H_ij = ∂²f/∂x_i∂x_j | Second derivatives |
| `laplacian(f, point, h)` | Laplacian ∇²f = Σ ∂²f/∂x_i² | Sum of second derivatives |

### Integrals

| Function | Description | Example |
|----------|-------------|---------|
| `integral(f, a, b, method, n)` | Definite integral | `∫₀¹ x² dx = 1/3` |
| `integral_2d(f, x_range, y_range, nx, ny)` | Double integral | `∫∫ f(x,y) dx dy` |
| `integral_3d(f, x_range, y_range, z_range, nx, ny, nz)` | Triple integral | `∫∫∫ f(x,y,z) dx dy dz` |
| `indefinite_integral(f, x, C, a, method)` | Indefinite integral | `F(x) = ∫ₐˣ f(t) dt + C` |

### Quantum Derivatives

| Function | Description |
|----------|-------------|
| `derivative_gaussian(x, mu, sigma)` | d/dx exp(-(x-μ)²/(2σ²)) |
| `derivative_sine_wave(x, k)` | d/dx sin(kx) = k cos(kx) |
| `derivative_cosine_wave(x, k)` | d/dx cos(kx) = -k sin(kx) |
| `derivative_plane_wave(x, k)` | d/dx e^{ikx} = ik e^{ikx} |

### Utilities

| Function | Description |
|----------|-------------|
| `taylor_series(f, x0, n, x)` | Generate Taylor series expansion |
| `diff_operator(f, x, h)` | Create differential operator function |

### Example 1: Derivatives

```python
from psiqit.math.calculus import derivative, gradient
Example 2: Shannon Entropy
python
# Shannon entropy
H = entropy(probs, base='2')
print(f"Entropy: {H:.4f} bits")  # 1.0000 bits
# First derivative
f = lambda x: x**3
df = derivative(f, 2.0)
print(f"f'(2) = {df}")  # 12.0

# Second derivative
d2f = derivative(f, 2.0, order=2)
print(f"f''(2) = {d2f}")  # 12.0

# Gradient of 2D function
g = lambda x, y: x**2 + y**2
grad = gradient(g, [1.0, 2.0])
print(f"∇f(1,2) = {grad}")  # [2.0, 4.0]
Example 2: Integrals
python
from psiqit.math.calculus import integral, integral_2d

# Definite integral
f = lambda x: x**2
result = integral(f, 0, 1, method='simpson')
print(f"∫₀¹ x² dx = {result:.6f}")  # 0.333333

# Double integral
g = lambda x, y: x * y
result = integral_2d(g, (0, 1), (0, 1))
print(f"∫∫ xy dx dy = {result:.6f}")  # 0.250000
Example 3: Taylor Series
python
from psiqit.math.calculus import taylor_series
import math

f = lambda x: math.sin(x)
series = taylor_series(f, 0, n=5)
print(series)
# P(x) = 0.0000 + 1.0000·(x-0) + 0.0000·(x-0)^2/2! + -1.0000·(x-0)^3/3! + 0.0000·(x-0)^4/4!
ODE Solver (ode_solver.py)
Ordinary differential equation solvers - Euler, Runge-Kutta (RK2, RK4), adaptive RK45, and systems of ODEs.

Class
Class	Description
ODEResult	Result container with x, y, y_list, method, n_steps, success
Methods
Function	Description	Example
euler(f, x0, y0, x_end, h)	Euler method (1st order)	dy/dx = f(x,y)
euler_system(f, x0, y0, x_end, h)	Euler for systems	System of ODEs
rk2(f, x0, y0, x_end, h)	2nd order Runge-Kutta (midpoint)	More accurate than Euler
rk4(f, x0, y0, x_end, h)	4th order Runge-Kutta	Standard method
rk4_system(f, x0, y0, x_end, h)	RK4 for systems	System of ODEs
solve_ode(f, x0, y0, x_end, h, method)	Generic solver	Unified interface
adaptive_rk45(f, x0, y0, x_end, tol, h0, h_min, h_max)	Adaptive RK45	Error control
schrodinger_1d_numerical(V, x_range, E, mass, hbar)	1D Schrödinger solver	TISE numerical solution
Example 1: Solving dy/dx = -y
python
from psiqit.math.ode_solver import solve_ode

def f(x, y):
    return -y

result = solve_ode(f, x0=0, y0=1, x_end=5, h=0.01, method='rk4')

print(f"y(5) = {result.y[-1]:.8f}")  # ~0.00673795 (exact: e^{-5})
Example 2: Harmonic Oscillator (System of ODEs)
python
from psiqit.math.ode_solver import solve_ode
import numpy as np

# dy1/dx = y2, dy2/dx = -y1
def f1(x, y1, y2): return y2
def f2(x, y1, y2): return -y1

result = solve_ode([f1, f2], x0=0, y0=[1, 0], x_end=2*np.pi, h=0.01)

# Should return to (1, 0) after 2π
print(f"y1(2π) = {result.y_list[-1][0]:.6f}")  # ~1.000000
print(f"y2(2π) = {result.y_list[-1][1]:.6f}")  # ~0.000000
Example 3: Adaptive RK45
python
from psiqit.math.ode_solver import adaptive_rk45

def f(x, y):
    return -y

x, y = adaptive_rk45(f, x0=0, y0=1, x_end=5, tol=1e-8)

print(f"Number of steps: {len(x)}")  # Adaptive steps
print(f"y(5) = {y[-1]:.8f}")
PDE Solver (pde_solver.py)
Partial differential equation solvers - Heat equation, wave equation, time-dependent Schrödinger equation.

Class
Class	Description
PDEResult	Result container with x, t, u, method, success
Functions
Function	Description
solve_heat_equation(nt, nx, L, T, alpha, initial_condition)	∂u/∂t = α ∂²u/∂x²
solve_wave_equation(nt, nx, L, T, c, initial_condition)	∂²u/∂t² = c² ∂²u/∂x²
solve_schrodinger_1d(nt, nx, L, T, mass, hbar, potential)	iℏ ∂ψ/∂t = Hψ
diffusion_equation_implicit(nt, nx, L, T, alpha)	Alias for heat equation
Example 1: Heat Equation
python
from psiqit.math.pde_solver import solve_heat_equation

# Solve heat equation on [0,1] for t in [0,1]
result = solve_heat_equation(nt=100, nx=50, L=1.0, T=1.0, alpha=0.01)

print(f"Temperature at center: {result.u[-1][25]:.4f}")
Example 2: Wave Equation
python
from psiqit.math.pde_solver import solve_wave_equation

# Solve wave equation on [0,1] for t in [0,2]
result = solve_wave_equation(nt=100, nx=50, L=1.0, T=2.0, c=1.0)

print(f"Wave amplitude at center: {result.u[-1][25]:.4f}")
Example 3: Schrödinger Equation
python
from psiqit.math.pde_solver import solve_schrodinger_1d

# Harmonic oscillator potential
potential = lambda x: 0.5 * x**2

result = solve_schrodinger_1d(nt=100, nx=100, L=10.0, T=2.0, potential=potential)

# Returns probability density |ψ|²
print(f"Probability density at center: {result.u[-1][50]:.4f}")
Symbolic Mathematics (symbolic.py)
Symbolic algebra using SymPy - Requires pip install sympy. Provides symbolic variables, equation solving, differentiation, integration, and quantum operator algebra.

Availability
python
from psiqit.math.symbolic import SYMPY_AVAILABLE

if SYMPY_AVAILABLE:
    print("SymPy is available")
else:
    print("Install SymPy: pip install sympy")
Basic Operations
Function	Description	Example
create_symbols(names)	Create symbolic variables	x, y = create_symbols('x y')
create_matrix(rows, name)	Create symbolic matrix	M = create_matrix([[1, x], [y, 0]])
solve_equation(expr, var)	Solve equation	solve_equation(x**2 - 4, x) → [-2, 2]
solve_linear_system(A, b)	Solve Ax = b	solve_linear_system(A, b)
differentiate(expr, var, order)	Symbolic derivative	differentiate(x**3, x) → 3*x**2
integrate_expression(expr, var, limits)	Symbolic integral	integrate_expression(x**2, x) → x**3/3
simplify_expression(expr)	Simplify expression	simplify_expression((x**2-1)/(x-1)) → x+1
expand_expression(expr)	Expand expression	expand_expression((x+1)**2) → x**2+2*x+1
Quantum Operations
Function	Description
pauli_algebra()	Symbolic Pauli matrices σₓ, σᵧ, σ₂, I
commutator_symbolic(A, B)	Symbolic commutator [A, B]
anticommutator_symbolic(A, B)	Symbolic anticommutator {A, B}
hamiltonian_symbolic(terms)	Build Hamiltonian from Pauli terms
eigenvalues_symbolic(matrix)	Symbolic eigenvalues
eigenvectors_symbolic(matrix)	Symbolic eigenvectors
Schrödinger Equation
Function	Description
schrodinger_1d_symbolic(potential, mass, hbar)	Symbolic TISE
infinite_well_states(n, L)	Infinite well wavefunctions
harmonic_oscillator_states(n, mass, omega)	HO wavefunctions
Example 1: Symbolic Algebra
python
from psiqit.math.symbolic import Symbol, differentiate, integrate_expression

x = Symbol('x')

# Derivative
expr = x**3
deriv = differentiate(expr, x)
print(f"d/dx x^3 = {deriv}")  # 3*x**2

# Integral
integral = integrate_expression(x**2, x)
print(f"∫ x² dx = {integral}")  # x**3/3
Example 2: Pauli Algebra
python
from psiqit.math.symbolic import pauli_algebra, commutator_symbolic

pauli = pauli_algebra()
sx = pauli['sigma_x']
sy = pauli['sigma_y']

# Commutator [σx, σy] = 2iσz
comm = commutator_symbolic(sx, sy)
print(f"[σx, σy] = {comm}")
Example 3: Hamiltonian and Eigenvalues
python
from psiqit.math.symbolic import hamiltonian_symbolic, eigenvalues_symbolic

# H = X + Z
H = hamiltonian_symbolic({'X': 1.0, 'Z': 1.0})
print(f"H = {H}")

# Eigenvalues
eigvals = eigenvalues_symbolic(H)
print(f"Eigenvalues: {eigvals}")  # {sqrt(2): 1, -sqrt(2): 1}
Module Contents
python
__all__ = [
    # Algebra
    'Vector', 'Matrix', 'PI', 'TAU', 'HBAR', 'eye', 'zeros', 'mul', 'add',
    'transpose', 'dagger', 'trace', 'kron', 'inner', 'outer', 'norm',
    'determinant', 'inverse', 'eigenvalues', 'eigenvectors', 'expm',
    'is_unitary', 'is_hermitian', 'born_rule', 'commutator',
    'eV_to_J', 'J_to_eV', 'nm_to_m', 'm_to_nm',
    # Calculus
    'derivative', 'gradient', 'jacobian', 'hessian', 'laplacian',
    'integral', 'integral_2d', 'integral_3d', 'taylor_series',
    # ODE
    'ODEResult', 'euler', 'rk2', 'rk4', 'solve_ode', 'adaptive_rk45',
    # PDE
    'PDEResult', 'solve_heat_equation', 'solve_wave_equation', 'solve_schrodinger_1d',
    # Symbolic
    'Symbol', 'symbols', 'create_symbols', 'differentiate', 'integrate_expression',
    'pauli_algebra', 'commutator_symbolic', 'hamiltonian_symbolic',
    'eigenvalues_symbolic', 'schrodinger_1d_symbolic',
    'infinite_well_states', 'harmonic_oscillator_states',
]
Complete Example: Quantum Harmonic Oscillator Analysis
python
import numpy as np
import matplotlib.pyplot as plt
from psiqit.math.ode_solver import schrodinger_1d_numerical

# Harmonic oscillator potential
V = lambda x: 0.5 * x**2

# Energy eigenvalues: E_n = n + 0.5
energies = [0.5, 1.5, 2.5, 3.5, 4.5]
colors = ['blue', 'red', 'green', 'orange', 'purple']

# Solve for each energy
x_range = (-5, 5)
plt.figure(figsize=(10, 6))

for n, E in enumerate(energies):
    x, psi = schrodinger_1d_numerical(V, x_range, E=E)
    # Normalize
    dx = x[1] - x[0]
    norm = np.sqrt(np.sum(psi**2) * dx)
    psi = psi / norm
    plt.plot(x, psi + E, color=colors[n], label=f'n={n}, E={E:.1f}')

plt.plot(x, V(x), 'k--', label='V(x) = 0.5x²')
plt.xlabel('Position x')
plt.ylabel('Energy')
plt.title('Harmonic Oscillator Eigenstates')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
References
Topic	Reference
Linear Algebra	G. Strang, "Linear Algebra and Its Applications," 2006
Numerical Methods	W. H. Press et al., "Numerical Recipes," 2007
ODE Solvers	E. Hairer, S. P. Nørsett, G. Wanner, "Solving Ordinary Differential Equations," 1993
PDE Solvers	J. W. Thomas, "Numerical Partial Differential Equations," 1995
SymPy	A. Meurer et al., "SymPy: symbolic computing in Python," PeerJ Computer Science, 2017