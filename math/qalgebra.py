"""
psiqit/math/qalgebra.py
============================================================
Quantum Algebra - Complete Mathematical Toolkit
============================================================

A self-contained module for all mathematical operations in quantum computing.
This module has NO external dependencies and provides:

    • Mathematical and physical constants
    • Complex number operations
    • Matrix operations (multiplication, transpose, trace, Kronecker product, etc.)
    • Vector operations (inner product, outer product, norm, normalization)
    • Advanced linear algebra (determinant, inverse, eigenvalues, eigenvectors, matrix exponential)
    • Validation functions (unitary, Hermitian, positive semidefinite)
    • Probability and statistics (distribution, sampling, entropy, expectation, variance)
    • Unit conversions (energy, length, frequency)

Example:
    >>> from psiqit.math.qalgebra import eye, inner, PI, HBAR
    >>> I = eye(2)
    >>> amp = inner([1,0], [1,0])  # ⟨0|0⟩ = 1
    >>> print(amp)
    (1+0j)

References:
    G. Strang, "Linear Algebra and Its Applications," 4th ed., Cengage Learning, 2006.
    R. A. Horn and C. R. Johnson, "Matrix Analysis," Cambridge University Press, 2012.
"""

import math
import random
from typing import List, Union, Optional, Tuple

# ============================================================
# Part 1: Type Aliases
# ============================================================

Vector = List[complex]
Matrix = List[List[complex]]


# ============================================================
# Part 2: Mathematical Constants
# ============================================================

PI = math.pi
"""π ≈ 3.141592653589793"""

TAU = 2 * math.pi
"""τ = 2π ≈ 6.283185307179586"""

EULER = math.e
"""e ≈ 2.718281828459045"""

INF = float('inf')
"""Infinity"""

SQRT2 = math.sqrt(2)
"""√2 ≈ 1.4142135623730951"""

SQRT3 = math.sqrt(3)
"""√3 ≈ 1.7320508075688772"""

SQRT2_INV = 1 / SQRT2
"""1/√2 ≈ 0.7071067811865475"""

SQRT3_INV = 1 / SQRT3
"""1/√3 ≈ 0.5773502691896258"""

LN2 = math.log(2)
"""ln 2 ≈ 0.6931471805599453"""

LN10 = math.log(10)
"""ln 10 ≈ 2.302585092994046"""

LOG2_E = math.log2(math.e)
"""log₂ e ≈ 1.4426950408889634"""

PHI = (1 + SQRT2) / 2
"""Golden ratio φ ≈ 1.618033988749895"""

PHI_INV = 2 / (1 + SQRT2)
"""1/φ ≈ 0.6180339887498949"""


# ============================================================
# Part 3: Physical Constants
# ============================================================

HBAR = 1.054571817e-34
"""Reduced Planck constant ħ in J·s"""

H_PLANCK = 6.62607015e-34
"""Planck constant h in J·s"""

C = 299792458.0
"""Speed of light in vacuum in m/s"""

E_CHARGE = 1.602176634e-19
"""Elementary charge e in C"""

M_ELECTRON = 9.1093837015e-31
"""Electron mass in kg"""

M_PROTON = 1.67262192369e-27
"""Proton mass in kg"""

M_NEUTRON = 1.67492749804e-27
"""Neutron mass in kg"""

K_B = 1.380649e-23
"""Boltzmann constant in J/K"""


# ============================================================
# Part 4: Complex Number Operations
# ============================================================

def is_real(z: complex, tol: float = 1e-12) -> bool:
    """
    Check if a complex number is real (imaginary part is zero).

    Args:
        z: Complex number.
        tol: Numerical tolerance.

    Returns:
        True if imaginary part is within tolerance.

    Examples:
        >>> is_real(3.0 + 0j)
        True
        >>> is_real(3.0 + 1e-15j)
        True
        >>> is_real(3.0 + 0.1j)
        False
    """
    return abs(z.imag) < tol


def is_imag(z: complex, tol: float = 1e-12) -> bool:
    """
    Check if a complex number is purely imaginary (real part is zero).

    Args:
        z: Complex number.
        tol: Numerical tolerance.

    Returns:
        True if real part is within tolerance.

    Examples:
        >>> is_imag(1j)
        True
        >>> is_imag(1e-15 + 2j)
        True
        >>> is_imag(1 + 2j)
        False
    """
    return abs(z.real) < tol


def phase(z: complex) -> float:
    """
    Compute the phase (argument) of a complex number.

    Args:
        z: Complex number.

    Returns:
        Phase angle in radians in the range [-π, π].

    Examples:
        >>> phase(1 + 0j)
        0.0
        >>> phase(0 + 1j)
        1.5707963267948966
    """
    return math.atan2(z.imag, z.real)


def magnitude(z: complex) -> float:
    """
    Compute the magnitude (modulus) of a complex number.

    Args:
        z: Complex number.

    Returns:
        |z| = √(Re(z)² + Im(z)²).

    Examples:
        >>> magnitude(3 + 4j)
        5.0
    """
    return abs(z)


def conjugate(z: complex) -> complex:
    """
    Compute the complex conjugate.

    Args:
        z: Complex number.

    Returns:
        Complex conjugate z* = Re(z) - i Im(z).

    Examples:
        >>> conjugate(1 + 2j)
        (1-2j)
    """
    return z.conjugate()


def from_polar(r: float, theta: float) -> complex:
    """
    Convert polar coordinates (r, θ) to complex number.

    Args:
        r: Radius (magnitude).
        theta: Angle in radians.

    Returns:
        Complex number r·e^{iθ} = r·(cos θ + i sin θ).

    Examples:
        >>> from_polar(1, math.pi)
        (-1+0j)
        >>> from_polar(1, math.pi/2)
        1j
    """
    return r * (math.cos(theta) + 1j * math.sin(theta))


def to_polar(z: complex) -> Tuple[float, float]:
    """
    Convert complex number to polar coordinates (r, θ).

    Args:
        z: Complex number.

    Returns:
        Tuple (r, θ) where r = |z| and θ = arg(z).

    Examples:
        >>> to_polar(-1 + 0j)
        (1.0, 3.141592653589793)
    """
    return abs(z), math.atan2(z.imag, z.real)


def cis(theta: float) -> complex:
    """
    Compute e^{iθ} = cos θ + i sin θ.

    Args:
        theta: Angle in radians.

    Returns:
        Complex number on the unit circle.

    Examples:
        >>> cis(math.pi)
        (-1+0j)
        >>> cis(math.pi/2)
        1j
    """
    return math.cos(theta) + 1j * math.sin(theta)


# ============================================================
# Part 5: Basic Matrix Operations
# ============================================================

def eye(n: int) -> Matrix:
    """
    Create an n×n identity matrix.

    Args:
        n: Matrix dimension.

    Returns:
        n×n identity matrix I with ones on the diagonal and zeros elsewhere.

    Examples:
        >>> eye(2)
        [[1.0, 0.0], [0.0, 1.0]]
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(n: int, m: Optional[int] = None) -> Matrix:
    """
    Create an n×m zero matrix.

    Args:
        n: Number of rows.
        m: Number of columns (if None, m = n).

    Returns:
        n×m matrix filled with zeros.

    Examples:
        >>> zeros(2)
        [[0.0, 0.0], [0.0, 0.0]]
        >>> zeros(2, 3)
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    """
    if m is None:
        m = n
    return [[0.0 for _ in range(m)] for _ in range(n)]


def mul(A: Matrix, B: Matrix) -> Matrix:
    """
    Multiply two matrices: C = A·B.

    Args:
        A: First matrix (m×n).
        B: Second matrix (n×p).

    Returns:
        Product matrix (m×p).

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> mul(A, B)
        [[19, 22], [43, 50]]
    """
    n, m, p = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


def add(A: Matrix, B: Matrix) -> Matrix:
    """
    Add two matrices: C = A + B.

    Args:
        A: First matrix.
        B: Second matrix (same dimensions as A).

    Returns:
        Sum matrix.

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> B = [[5, 6], [7, 8]]
        >>> add(A, B)
        [[6, 8], [10, 12]]
    """
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have same dimensions")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def transpose(A: Matrix) -> Matrix:
    """
    Compute the transpose of a matrix: A^T.

    Args:
        A: Input matrix.

    Returns:
        Transposed matrix.

    Examples:
        >>> A = [[1, 2, 3], [4, 5, 6]]
        >>> transpose(A)
        [[1, 4], [2, 5], [3, 6]]
    """
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def conj(A: Matrix) -> Matrix:
    """
    Compute the complex conjugate of a matrix.

    Args:
        A: Input matrix.

    Returns:
        Matrix with each element complex conjugated.

    Examples:
        >>> A = [[1+2j, 3-4j]]
        >>> conj(A)
        [[(1-2j), (3+4j)]]
    """
    return [[x.conjugate() for x in row] for row in A]


def dagger(A: Matrix) -> Matrix:
    """
    Compute the Hermitian conjugate (adjoint) of a matrix: A^† = (A^T)*.

    Args:
        A: Input matrix.

    Returns:
        Hermitian conjugate matrix.

    Examples:
        >>> A = [[1+2j, 3+4j], [5+6j, 7+8j]]
        >>> H = dagger(A)
    """
    return conj(transpose(A))


def trace(A: Matrix) -> complex:
    """
    Compute the trace of a square matrix: Tr(A) = Σ A_{ii}.

    Args:
        A: Square matrix.

    Returns:
        Sum of diagonal elements.

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> trace(A)
        5
    """
    return sum(A[i][i] for i in range(len(A)))


def kron(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the Kronecker product (tensor product): A ⊗ B.

    Args:
        A: First matrix (m×n).
        B: Second matrix (p×q).

    Returns:
        Kronecker product matrix (mp × nq).

    Examples:
        >>> I = eye(2)
        >>> X = [[0, 1], [1, 0]]
        >>> kron(X, I)
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    """
    rA, cA = len(A), len(A[0])
    rB, cB = len(B), len(B[0])
    return [
        [A[i // rB][j // cB] * B[i % rB][j % cB]
         for j in range(cA * cB)]
        for i in range(rA * rB)
    ]


# ============================================================
# Part 6: Vector Operations
# ============================================================

def inner(bra: Vector, ket: Vector) -> complex:
    """
    Compute the inner product ⟨bra|ket⟩ = Σ bra_i* · ket_i.

    Args:
        bra: Bra vector (row vector of amplitudes).
        ket: Ket vector (column vector of amplitudes).

    Returns:
        Complex inner product.

    Examples:
        >>> inner([1, 0], [1, 0])  # ⟨0|0⟩
        1.0
        >>> inner([1, 0], [0, 1])  # ⟨0|1⟩
        0j
    """
    if len(bra) != len(ket):
        raise ValueError(f"Dimension mismatch: {len(bra)} vs {len(ket)}")
    return sum(bra[i].conjugate() * ket[i] for i in range(len(bra)))


def outer(ket: Vector, bra: Vector) -> Matrix:
    """
    Compute the outer product |ket⟩⟨bra|.

    Args:
        ket: Ket vector (column vector).
        bra: Bra vector (row vector).

    Returns:
        Matrix of size len(ket) × len(bra).

    Examples:
        >>> zero = [1, 0]
        >>> outer(zero, zero)  # |0⟩⟨0|
        [[1, 0], [0, 0]]
    """
    return [[ket[i] * bra[j].conjugate() for j in range(len(bra))] for i in range(len(ket))]


def norm(ket: Vector) -> float:
    """
    Compute the Euclidean norm of a vector: |||ψ⟩|| = √(⟨ψ|ψ⟩).

    Args:
        ket: Ket vector.

    Returns:
        Norm (non-negative real number).

    Examples:
        >>> norm([1, 0])
        1.0
        >>> norm([1, 1])
        1.4142135623730951
    """
    return math.sqrt(sum(abs(x) ** 2 for x in ket))


def normalize(ket: Vector) -> Vector:
    """
    Normalize a vector to unit norm.

    Args:
        ket: Ket vector.

    Returns:
        Normalized vector |ψ⟩/|||ψ⟩||.

    Examples:
        >>> normalize([1, 1])
        [0.7071067811865475, 0.7071067811865475]
    """
    n = norm(ket)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / n for x in ket]


# ============================================================
# Part 7: Determinant (Recursive - Laplace expansion)
# ============================================================

def determinant(A: Matrix) -> complex:
    """
    Compute the determinant of a square matrix using Laplace expansion.

    Args:
        A: Square matrix.

    Returns:
        Determinant value.

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> determinant(A)
        -2.0
    """
    n = len(A)
    if n == 1:
        return A[0][0]
    if n == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    
    det = 0
    for j in range(n):
        # Create minor
        minor = [row[:j] + row[j+1:] for row in A[1:]]
        sign = 1 if j % 2 == 0 else -1
        det += sign * A[0][j] * determinant(minor)
    return det


# ============================================================
# Part 8: Matrix Inverse (Gauss-Jordan elimination)
# ============================================================

def inverse(A: Matrix) -> Matrix:
    """
    Compute the inverse of a square matrix using Gauss-Jordan elimination.

    Args:
        A: Square invertible matrix.

    Returns:
        Inverse matrix A⁻¹.

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> inverse(A)
        [[-2.0, 1.0], [1.5, -0.5]]
    """
    n = len(A)
    # Create augmented matrix [A | I]
    aug = [A[i][:] + eye(n)[i] for i in range(n)]
    
    for col in range(n):
        # Find pivot
        pivot = col
        while pivot < n and abs(aug[pivot][col]) < 1e-12:
            pivot += 1
        if pivot == n:
            raise ValueError("Matrix is singular")
        
        # Swap rows
        aug[col], aug[pivot] = aug[pivot], aug[col]
        
        # Normalize pivot row
        pivot_val = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot_val
        
        # Eliminate other rows
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]
    
    # Extract inverse
    return [row[n:] for row in aug]


# ============================================================
# Part 9: Eigenvalues and Eigenvectors (QR Algorithm)
# ============================================================

def eigenvalues(A: Matrix, max_iter: int = 100, tol: float = 1e-10) -> List[complex]:
    """
    Compute eigenvalues of a square matrix using the QR algorithm.

    Args:
        A: Square matrix.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.

    Returns:
        List of eigenvalues.

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> eigenvalues(A)
        [-0.3722813232690143, 5.372281323269014]
    """
    n = len(A)
    # Make a copy
    Ak = [row[:] for row in A]
    
    for _ in range(max_iter):
        Q, R = _qr_decomposition(Ak)
        Ak = mul(R, Q)
        
        # Check convergence (off-diagonal elements)
        off_diag = sum(abs(Ak[i][j]) for i in range(n) for j in range(n) if i != j)
        if off_diag < tol:
            break
    
    return [Ak[i][i] for i in range(n)]


def eigenvectors(A: Matrix, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """
    Compute eigenvalues and eigenvectors of a square matrix.

    Args:
        A: Square matrix.
        max_iter: Maximum number of iterations.
        tol: Numerical tolerance.

    Returns:
        Dictionary with keys 'values' (list of eigenvalues) and 'vectors'
        (list of eigenvectors as lists).

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> result = eigenvectors(A)
        >>> print(result['values'])
        [-0.372, 5.372]
    """
    vals = eigenvalues(A, max_iter, tol)
    
    # Simple inverse iteration for eigenvectors
    vecs = []
    for lam in vals:
        # Solve (A - λI)v = 0
        M = [[A[i][j] - (lam if i == j else 0) for j in range(len(A))] for i in range(len(A))]
        
        # Start with random vector
        v = [1.0] * len(A)
        for _ in range(50):
            # Solve Mv = v_old (simplified)
            try:
                Minv = inverse([[M[i][j] + (tol if i == j else 0) for j in range(len(M))] 
                               for i in range(len(M))])
                v = [sum(Minv[i][j] * v[j] for j in range(len(v))) for i in range(len(v))]
            except:
                break
            # Normalize
            norm_v = math.sqrt(sum(abs(x)**2 for x in v))
            if norm_v > 1e-12:
                v = [x / norm_v for x in v]
        vecs.append(v)
    
    return {'values': vals, 'vectors': vecs}


# ============================================================
# Part 10: Matrix Exponential (Taylor Series)
# ============================================================

def expm(A: Matrix, terms: int = 20) -> Matrix:
    """
    Compute the matrix exponential e^A using Taylor series.

    e^A = Σ_{k=0}^∞ A^k / k!

    Args:
        A: Square matrix.
        terms: Number of terms in the series expansion.

    Returns:
        Matrix exponential e^A.

    Examples:
        >>> A = [[0, 1], [0, 0]]
        >>> expm(A)
        [[1.0, 1.0], [0.0, 1.0]]
    """
    n = len(A)
    result = eye(n)
    term = eye(n)
    
    for k in range(1, terms):
        term = mul(term, A)
        for i in range(n):
            for j in range(n):
                term[i][j] /= k
        for i in range(n):
            for j in range(n):
                result[i][j] += term[i][j]
    
    return result


# ============================================================
# Part 11: Validation Functions
# ============================================================

def is_unitary(U: Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is unitary: U^†U = I.

    Args:
        U: Square matrix.
        tol: Numerical tolerance.

    Returns:
        True if U is unitary, False otherwise.
    """
    n = len(U)
    I = eye(n)
    U_dag = dagger(U)
    product = mul(U_dag, U)
    for i in range(n):
        for j in range(n):
            if abs(product[i][j] - I[i][j]) > tol:
                return False
    return True


def is_hermitian(H: Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is Hermitian: H^† = H.

    Args:
        H: Square matrix.
        tol: Numerical tolerance.

    Returns:
        True if H is Hermitian, False otherwise.
    """
    H_dag = dagger(H)
    for i in range(len(H)):
        for j in range(len(H)):
            if abs(H[i][j] - H_dag[i][j]) > tol:
                return False
    return True


def is_positive(M: Matrix, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is positive semidefinite (all eigenvalues ≥ 0).

    Args:
        M: Square Hermitian matrix.
        tol: Numerical tolerance.

    Returns:
        True if M is positive semidefinite, False otherwise.
    """
    if not is_hermitian(M, tol):
        return False
    vals = eigenvalues(M)
    return all(v.real >= -tol for v in vals)


# ============================================================
# Part 12: Probability and Statistics
# ============================================================

def is_distribution(probs: List[float], tol: float = 1e-10) -> bool:
    """
    Check if a list of numbers forms a valid probability distribution.

    A valid distribution must:
        - Have non-negative entries (within tolerance)
        - Sum to 1 (within tolerance)

    Args:
        probs: List of probabilities.
        tol: Numerical tolerance.

    Returns:
        True if valid distribution, False otherwise.
    """
    if not probs:
        return False
    if any(p < -tol for p in probs):
        return False
    return abs(sum(probs) - 1.0) < tol


def normalize_probs(probs: List[float]) -> List[float]:
    """
    Normalize a list of probabilities to sum to 1.

    Args:
        probs: List of probabilities.

    Returns:
        Normalized probabilities.

    Examples:
        >>> normalize_probs([0.5, 0.5])
        [0.5, 0.5]
        >>> normalize_probs([1, 2])
        [0.3333333333333333, 0.6666666666666666]
    """
    total = sum(probs)
    if total == 0:
        raise ValueError("Cannot normalize zero distribution")
    return [p / total for p in probs]


def sample(probs: List[float], shots: int = 1) -> List[int]:
    """
    Sample outcomes from a probability distribution.

    Args:
        probs: List of probabilities.
        shots: Number of samples.

    Returns:
        List of sampled indices.

    Examples:
        >>> sample([0.5, 0.5], shots=10)
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    """
    if not is_distribution(probs):
        probs = normalize_probs(probs)
    
    outcomes = []
    for _ in range(shots):
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                outcomes.append(i)
                break
    return outcomes


def entropy(probs: List[float], base: str = 'e') -> float:
    """
    Compute Shannon entropy: H = -Σ p_i log(p_i).

    Args:
        probs: List of probabilities.
        base: Logarithm base ('e', '2', or '10').

    Returns:
        Entropy in the specified units.

    Examples:
        >>> entropy([0.5, 0.5], base='2')
        1.0
    """
    if not is_distribution(probs):
        probs = normalize_probs(probs)
    
    H = 0.0
    for p in probs:
        if p > 0:
            if base == 'e':
                H -= p * math.log(p)
            elif base == '2':
                H -= p * math.log2(p)
            else:
                H -= p * math.log10(p)
    return H


def max_entropy(n: int, base: str = 'e') -> float:
    """
    Compute maximum entropy for a distribution with n outcomes.

    Args:
        n: Number of outcomes.
        base: Logarithm base.

    Returns:
        Maximum entropy = log(n).

    Examples:
        >>> max_entropy(2, base='2')
        1.0
    """
    if base == 'e':
        return math.log(n)
    elif base == '2':
        return math.log2(n)
    else:
        return math.log10(n)


def born_rule(amplitudes: Vector) -> List[float]:
    """
    Apply the Born rule: probability = |amplitude|².

    Args:
        amplitudes: List of complex amplitudes.

    Returns:
        Normalized probabilities.

    Examples:
        >>> born_rule([0.707, 0.707])
        [0.5, 0.5]
    """
    probs = [abs(a)**2 for a in amplitudes]
    return normalize_probs(probs)


def expectation(values: List[float], probs: List[float]) -> float:
    """
    Compute expectation value: ⟨A⟩ = Σ a_i p_i.

    Args:
        values: List of possible outcomes.
        probs: List of corresponding probabilities.

    Returns:
        Expectation value.

    Examples:
        >>> expectation([-1, 1], [0.5, 0.5])
        0.0
    """
    if len(values) != len(probs):
        raise ValueError("Values and probabilities must have same length")
    return sum(v * p for v, p in zip(values, probs))


def variance(values: List[float], probs: List[float]) -> float:
    """
    Compute variance: Var(A) = ⟨A²⟩ - ⟨A⟩².

    Args:
        values: List of possible outcomes.
        probs: List of corresponding probabilities.

    Returns:
        Variance.

    Examples:
        >>> variance([-1, 1], [0.5, 0.5])
        1.0
    """
    mu = expectation(values, probs)
    mu2 = expectation([v**2 for v in values], probs)
    return mu2 - mu**2


# ============================================================
# Part 13: Utility Functions
# ============================================================

def dimension(obj: Union[Vector, Matrix]) -> int:
    """
    Get the dimension of a vector or square matrix.

    Args:
        obj: Vector or square matrix.

    Returns:
        Length of vector or matrix size.

    Examples:
        >>> dimension([1, 2, 3])
        3
        >>> dimension([[1, 2], [3, 4]])
        2
    """
    if isinstance(obj[0], list):
        return len(obj)
    return len(obj)


def hilbert_space(n: int) -> str:
    """
    Get a formatted string representation of Hilbert space dimension.

    Args:
        n: Dimension.

    Returns:
        String like 'ℂⁿ' with superscript.

    Examples:
        >>> hilbert_space(4)
        'ℂ⁴'
    """
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"ℂ{str(n).translate(sup)}"


def display(M: Matrix, precision: int = 3) -> None:
    """
    Pretty-print a matrix.

    Args:
        M: Matrix to display.
        precision: Number of decimal places.

    Examples:
        >>> A = [[1, 2], [3, 4]]
        >>> display(A)
        [ 1.000  2.000 ]
        [ 3.000  4.000 ]
    """
    for row in M:
        row_str = "["
        for val in row:
            if isinstance(val, complex):
                if abs(val.imag) < 10**(-precision):
                    row_str += f" {val.real:>{precision+4}.{precision}f}"
                else:
                    sign = "+" if val.imag >= 0 else ""
                    row_str += f" {val.real:>{precision+4}.{precision}f}{sign}{val.imag:.{precision}f}j"
            else:
                row_str += f" {val:>{precision+4}.{precision}f}"
        row_str += " ]"
        print(row_str)


def commutator(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the commutator [A, B] = AB - BA.

    Args:
        A: First matrix.
        B: Second matrix.

    Returns:
        Commutator matrix.

    Examples:
        >>> X = [[0, 1], [1, 0]]
        >>> Z = [[1, 0], [0, -1]]
        >>> commutator(X, Z)
        [[0, -2], [2, 0]]
    """
    AB = mul(A, B)
    BA = mul(B, A)
    n = len(AB)
    return [[AB[i][j] - BA[i][j] for j in range(n)] for i in range(n)]


def anticommutator(A: Matrix, B: Matrix) -> Matrix:
    """
    Compute the anticommutator {A, B} = AB + BA.

    Args:
        A: First matrix.
        B: Second matrix.

    Returns:
        Anticommutator matrix.
    """
    AB = mul(A, B)
    BA = mul(B, A)
    n = len(AB)
    return [[AB[i][j] + BA[i][j] for j in range(n)] for i in range(n)]


# ============================================================
# Part 14: Unit Conversions
# ============================================================

def eV_to_J(eV: float) -> float:
    """Convert electronvolts to Joules."""
    return eV * E_CHARGE


def J_to_eV(J: float) -> float:
    """Convert Joules to electronvolts."""
    return J / E_CHARGE


def nm_to_m(nm: float) -> float:
    """Convert nanometers to meters."""
    return nm * 1e-9


def m_to_nm(m: float) -> float:
    """Convert meters to nanometers."""
    return m * 1e9


def energy_to_frequency(energy: float) -> float:
    """Convert energy to frequency using E = hν."""
    return energy / H_PLANCK


def frequency_to_energy(freq: float) -> float:
    """Convert frequency to energy using E = hν."""
    return H_PLANCK * freq


def sqrt_matrix(A: Matrix, tol: float = 1e-10) -> Matrix:
    """
    Compute the square root of a positive semidefinite matrix.

    Args:
        A: Positive semidefinite matrix.
        tol: Numerical tolerance.

    Returns:
        Matrix √A such that (√A)² = A.

    Examples:
        >>> I = eye(2)
        >>> sqrt_matrix(I)  # Should return I
        [[1.0, 0.0], [0.0, 1.0]]
    """
    # Get eigenvalues and eigenvectors
    eig_result = eigenvectors(A)
    eigvals = eig_result['values']
    eigvecs = eig_result['vectors']
    
    # Take square root of eigenvalues
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        lam = eigvals[i]
        if isinstance(lam, complex):
            lam = lam.real
        if lam > tol:
            sqrt_lam = math.sqrt(lam)
            vec = eigvecs[i]
            for j in range(n):
                for k in range(n):
                    result[j][k] += sqrt_lam * vec[j] * vec[k].conjugate()
    
    return result


# ============================================================
# Exports
# ============================================================

__all__ = [
    'Vector', 'Matrix',
    'PI', 'TAU', 'EULER','sqrt_matrix', 'INF', 'SQRT2', 'SQRT3', 'SQRT2_INV', 'SQRT3_INV',
    'LN2', 'LN10', 'LOG2_E', 'PHI', 'PHI_INV',
    'HBAR', 'H_PLANCK', 'C', 'E_CHARGE', 'M_ELECTRON', 'M_PROTON', 'M_NEUTRON', 'K_B',
    'is_real', 'is_imag', 'phase', 'magnitude', 'conjugate', 'from_polar', 'to_polar', 'cis',
    'eye', 'zeros', 'mul', 'add', 'transpose', 'conj', 'dagger', 'trace', 'kron',
    'inner', 'outer', 'norm', 'normalize',
    'determinant', 'inverse', 'eigenvalues', 'eigenvectors', 'expm',
    'is_unitary', 'is_hermitian', 'is_positive',
    'is_distribution', 'normalize_probs', 'sample', 'entropy', 'max_entropy',
    'born_rule', 'expectation', 'variance',
    'dimension', 'hilbert_space', 'display', 'commutator', 'anticommutator',
    'eV_to_J', 'J_to_eV', 'nm_to_m', 'm_to_nm', 'energy_to_frequency', 'frequency_to_energy',
]