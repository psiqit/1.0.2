"""
psiqit/utils/polarization.py
============================================================
Jones Matrices and Polarization Optics
============================================================

Jones calculus for polarization optics:
    • Jones vectors for polarization states
    • Jones matrices for optical elements
    • Polarization transformations

This module implements Jones calculus, a mathematical formalism for
describing the polarization of light and its transformation by optical
elements. It is widely used in classical and quantum optics.

The main components are:
    - Jones vectors: 2-component complex vectors representing polarization states
    - Jones matrices: 2x2 complex matrices representing optical elements
    - Stokes parameters: Real parameters describing partially polarized light
    - Poincaré sphere: Geometric representation of polarization states

Example:
    >>> from psiqit.utils.polarization import (
    ...     horizontal, vertical, quarter_waveplate, circular_right
    ... )
    >>> 
    >>> # Create horizontally polarized light
    >>> H = horizontal()
    >>> 
    >>> # Apply quarter-wave plate at 45°
    >>> QWP = quarter_waveplate(angle=45)
    >>> output = apply_jones(QWP, H)
    >>> 
    >>> # Check if output is right-circular
    >>> circular = circular_right()
    >>> print(output[0][0] / output[1][0])  # Should be approximately -1j

References:
    R. C. Jones, "A new calculus for the treatment of optical systems,"
    J. Opt. Soc. Am., 31(7):488-493, 1941.
    E. Hecht, "Optics," 5th ed., Pearson, 2017.
"""

import math
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..math.qalgebra import mul, dagger, trace


# ============================================================
# Jones Vectors (Polarization States)
# ============================================================

def horizontal() -> List[List[complex]]:
    """
    Horizontally polarized light |H⟩ = [1; 0].

    Returns:
        2x1 Jones vector.

    Examples:
        >>> H = horizontal()
        >>> print(H[0][0])
        1
    """
    return [[1], [0]]


def vertical() -> List[List[complex]]:
    """
    Vertically polarized light |V⟩ = [0; 1].

    Returns:
        2x1 Jones vector.

    Examples:
        >>> V = vertical()
        >>> print(V[1][0])
        1
    """
    return [[0], [1]]


def diagonal(angle: float = 45) -> List[List[complex]]:
    """
    Linearly polarized light at a given angle.

    Args:
        angle: Polarization angle in degrees (0° = horizontal, 90° = vertical).

    Returns:
        2x1 Jones vector.

    Examples:
        >>> D45 = diagonal(45)  # 45° linear polarization
        >>> D30 = diagonal(30)  # 30° linear polarization
    """
    rad = math.radians(angle)
    return [[math.cos(rad)], [math.sin(rad)]]


def anti_diagonal() -> List[List[complex]]:
    """
    Anti-diagonally polarized light |A⟩ = [1/√2; -1/√2].

    Returns:
        2x1 Jones vector.

    Examples:
        >>> A = anti_diagonal()
        >>> print(f"{A[0][0]:.3f}, {A[1][0]:.3f}")
        0.707, -0.707
    """
    v = 1 / math.sqrt(2)
    return [[v], [-v]]


def circular_right() -> List[List[complex]]:
    """
    Right-circularly polarized light |R⟩ = [1; -i]/√2.

    Returns:
        2x1 Jones vector.

    Examples:
        >>> R = circular_right()
        >>> print(f"{R[0][0]:.3f}, {R[1][0]:.3f}")
        0.707, -0.707j
    """
    v = 1 / math.sqrt(2)
    return [[v], [-1j * v]]


def circular_left() -> List[List[complex]]:
    """
    Left-circularly polarized light |L⟩ = [1; i]/√2.

    Returns:
        2x1 Jones vector.

    Examples:
        >>> L = circular_left()
        >>> print(f"{L[0][0]:.3f}, {L[1][0]:.3f}")
        0.707, 0.707j
    """
    v = 1 / math.sqrt(2)
    return [[v], [1j * v]]


def elliptical(a: float, b: float, delta: float) -> List[List[complex]]:
    """
    Elliptically polarized light.

    Args:
        a: Amplitude in the x-direction.
        b: Amplitude in the y-direction.
        delta: Phase difference between x and y components.

    Returns:
        2x1 Jones vector (normalized).

    Examples:
        >>> # General elliptical polarization
        >>> psi = elliptical(a=2, b=1, delta=math.pi/4)
    """
    norm = math.sqrt(a**2 + b**2)
    return [[a / norm], [b * math.cos(delta) + 1j * b * math.sin(delta) / norm]]


# ============================================================
# Jones Matrices (Optical Elements)
# ============================================================

def polarizer(angle: float = 0) -> List[List[complex]]:
    """
    Linear polarizer with transmission axis at a given angle.

    Args:
        angle: Transmission axis angle in degrees.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> P0 = polarizer(0)   # Horizontal polarizer
        >>> P90 = polarizer(90) # Vertical polarizer
        >>> P45 = polarizer(45) # 45° polarizer
    """
    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)
    return [[c**2, c*s], [c*s, s**2]]


def waveplate(retardance: float, angle: float = 0) -> List[List[complex]]:
    """
    General waveplate (phase retarder).

    Args:
        retardance: Phase retardation in radians.
        angle: Fast axis angle in degrees.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> QWP = waveplate(math.pi/2, 0)  # Quarter-wave plate at 0°
        >>> HWP = waveplate(math.pi, 45)   # Half-wave plate at 45°
    """
    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)
    
    # Rotation matrix
    R = [[c, -s], [s, c]]
    R_inv = [[c, s], [-s, c]]
    
    # Retarder matrix
    W = [[1, 0], [0, math.cos(retardance) + 1j * math.sin(retardance)]]
    
    # R * W * R^(-1)
    temp = mul(R, W)
    return mul(temp, R_inv)


def quarter_waveplate(angle: float = 0) -> List[List[complex]]:
    """
    Quarter-wave plate (retardance = π/2).

    A quarter-wave plate converts linear polarization to circular
    polarization when the fast axis is at 45° to the polarization direction.

    Args:
        angle: Fast axis angle in degrees.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> QWP0 = quarter_waveplate(0)
        >>> QWP45 = quarter_waveplate(45)
    """
    rad = math.radians(angle)
    cos2 = math.cos(rad)
    sin2 = math.sin(rad)
    
    Q11 = cos2**2 + 1j * sin2**2
    Q12 = (1 - 1j) * sin2 * cos2
    Q21 = (1 - 1j) * sin2 * cos2
    Q22 = sin2**2 + 1j * cos2**2
    
    return [[Q11, Q12], [Q21, Q22]]


def half_waveplate(angle: float = 0) -> List[List[complex]]:
    """
    Half-wave plate (retardance = π).

    A half-wave plate rotates the polarization direction by twice the
    angle between the fast axis and the input polarization.

    Args:
        angle: Fast axis angle in degrees.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> HWP0 = half_waveplate(0)
        >>> HWP22_5 = half_waveplate(22.5)  # Rotates by 45°
    """
    return waveplate(math.pi, angle)


def full_waveplate(angle: float = 0) -> List[List[complex]]:
    """
    Full-wave plate (retardance = 2π).

    A full-wave plate has no net effect on polarization (effectively identity).

    Args:
        angle: Fast axis angle in degrees (ignored).

    Returns:
        2x2 Jones matrix (effectively identity).

    Examples:
        >>> FWP = full_waveplate(0)  # Identity matrix
    """
    return waveplate(2 * math.pi, angle)


def rotator(angle: float) -> List[List[complex]]:
    """
    Optical rotator (Faraday rotator).

    Rotates the polarization direction by a fixed angle regardless of input.

    Args:
        angle: Rotation angle in degrees.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> R45 = rotator(45)  # Rotates polarization by 45°
    """
    rad = math.radians(angle)
    c = math.cos(rad)
    s = math.sin(rad)
    return [[c, -s], [s, c]]


def linear_retarder(delta: float, angle: float = 0) -> List[List[complex]]:
    """
    Linear phase retarder (alias for waveplate).

    Args:
        delta: Phase retardation.
        angle: Fast axis angle in degrees.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> LR = linear_retarder(delta=math.pi/2, angle=0)
    """
    return waveplate(delta, angle)


def circular_dichroism(theta: float) -> List[List[complex]]:
    """
    Circular dichroism element (circular birefringence).

    Args:
        theta: Rotation angle from circular birefringence.

    Returns:
        2x2 Jones matrix.

    Examples:
        >>> CD = circular_dichroism(theta=math.pi/4)
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, -s], [s, c]]


# ============================================================
# Polarization State Analysis
# ============================================================

def stokes_parameters(jones: List[List[complex]]) -> Tuple[float, float, float, float]:
    """
    Compute Stokes parameters from a Jones vector.

    The Stokes parameters are:
        S₀ = |Eₓ|² + |Eᵧ|²  (total intensity)
        S₁ = |Eₓ|² - |Eᵧ|²  (horizontal vs vertical)
        S₂ = 2 Re(Eₓ* Eᵧ)   (45° vs 135°)
        S₃ = 2 Im(Eₓ* Eᵧ)   (right vs left circular)

    Args:
        jones: 2x1 Jones vector.

    Returns:
        Tuple (S₀, S₁, S₂, S₃) of Stokes parameters.

    Examples:
        >>> H = horizontal()
        >>> S0, S1, S2, S3 = stokes_parameters(H)
        >>> print(f"S₁ = {S1}, S₂ = {S2}, S₃ = {S3}")
        S₁ = 1.0, S₂ = 0.0, S₃ = 0.0
    """
    if isinstance(jones[0], list):
        Ex = jones[0][0]
        Ey = jones[1][0]
    else:
        Ex = jones[0]
        Ey = jones[1]
    
    S0 = abs(Ex)**2 + abs(Ey)**2
    S1 = abs(Ex)**2 - abs(Ey)**2
    S2 = 2 * (Ex * Ey.conjugate()).real
    S3 = 2 * (Ex * Ey.conjugate()).imag
    
    return (S0, S1, S2, S3)


def degree_of_polarization(jones: List[List[complex]]) -> float:
    """
    Compute the degree of polarization (DOP).

    For fully polarized light, DOP = 1.
    For unpolarized light, DOP = 0.
    For partially polarized light, 0 < DOP < 1.

    Args:
        jones: Jones vector.

    Returns:
        Degree of polarization between 0 and 1.

    Examples:
        >>> H = horizontal()
        >>> dop = degree_of_polarization(H)
        >>> print(f"{dop:.1f}")
        1.0
    """
    S0, S1, S2, S3 = stokes_parameters(jones)
    if S0 == 0:
        return 0
    return math.sqrt(S1**2 + S2**2 + S3**2) / S0


def polarization_angle(jones: List[List[complex]]) -> float:
    """
    Compute the polarization angle (orientation of the major axis).

    Returns:
        Angle in degrees (0° to 180°).

    Examples:
        >>> H = horizontal()
        >>> print(polarization_angle(H))
        0.0
        >>> D45 = diagonal(45)
        >>> print(polarization_angle(D45))
        45.0
    """
    if isinstance(jones[0], list):
        Ex = jones[0][0]
        Ey = jones[1][0]
    else:
        Ex = jones[0]
        Ey = jones[1]
    
    angle = 0.5 * math.atan2(2 * (Ex * Ey.conjugate()).real, 
                              abs(Ex)**2 - abs(Ey)**2)
    return math.degrees(angle)


def ellipticity(jones: List[List[complex]]) -> float:
    """
    Compute the ellipticity angle.

    The ellipticity angle ε satisfies tan ε = b/a, where a and b are the
    semi-major and semi-minor axes of the polarization ellipse.
    ε = 0° for linear polarization, ε = ±45° for circular polarization.

    Returns:
        Ellipticity angle in degrees (-45° to +45°).

    Examples:
        >>> R = circular_right()
        >>> print(f"{ellipticity(R):.1f}")
        -45.0
        >>> L = circular_left()
        >>> print(f"{ellipticity(L):.1f}")
        45.0
    """
    S0, S1, S2, S3 = stokes_parameters(jones)
    if S0 == 0:
        return 0
    return math.degrees(0.5 * math.asin(S3 / S0))


# ============================================================
# Poincaré Sphere
# ============================================================

def to_poincare(jones: List[List[complex]]) -> Tuple[float, float, float]:
    """
    Convert a Jones vector to Poincaré sphere coordinates.

    The Poincaré sphere is a geometric representation of polarization
    states, where:
        - North pole: right-circular
        - South pole: left-circular
        - Equator: linear polarizations

    Args:
        jones: 2x1 Jones vector.

    Returns:
        (x, y, z) coordinates on the Poincaré sphere.

    Examples:
        >>> H = horizontal()
        >>> x, y, z = to_poincare(H)
        >>> print(f"({x:.1f}, {y:.1f}, {z:.1f})")
        (1.0, 0.0, 0.0)
    """
    S0, S1, S2, S3 = stokes_parameters(jones)
    if S0 == 0:
        return (0, 0, 0)
    return (S1 / S0, S2 / S0, S3 / S0)


def from_poincare(x: float, y: float, z: float) -> List[List[complex]]:
    """
    Convert Poincaré sphere coordinates to a Jones vector.

    Args:
        x, y, z: Coordinates on the Poincaré sphere (must satisfy x²+y²+z² = 1).

    Returns:
        2x1 Jones vector.

    Examples:
        >>> # North pole = right-circular
        >>> R = from_poincare(0, 0, 1)
        >>> 
        >>> # Equator along x = horizontal
        >>> H = from_poincare(1, 0, 0)
    """
    r = math.sqrt(x**2 + y**2 + z**2)
    if r > 0:
        x, y, z = x/r, y/r, z/r
    
    # Spherical to Cartesian
    theta = math.acos(z)  # Polar angle
    phi = math.atan2(y, x)  # Azimuthal angle
    
    # Jones vector from spherical coordinates
    a = math.cos(theta / 2)
    b = math.sin(theta / 2) * (math.cos(phi) + 1j * math.sin(phi))
    
    return [[a], [b]]


# ============================================================
# Jones Matrix Operations
# ============================================================

def apply_jones(matrix: List[List[complex]], jones: List[List[complex]]) -> List[List[complex]]:
    """
    Apply a Jones matrix to a Jones vector.

    Args:
        matrix: 2x2 Jones matrix.
        jones: 2x1 Jones vector.

    Returns:
        Transformed Jones vector.

    Examples:
        >>> H = horizontal()
        >>> HWP = half_waveplate(22.5)
        >>> output = apply_jones(HWP, H)
    """
    return mul(matrix, jones)


def cascade_jones(matrices: List[List[List[complex]]]) -> List[List[complex]]:
    """
    Cascade multiple Jones matrices (applied in order).

    The combined matrix is M = Mₙ ... M₂ M₁.

    Args:
        matrices: List of 2x2 Jones matrices.

    Returns:
        Combined 2x2 Jones matrix.

    Examples:
        >>> QWP = quarter_waveplate(45)
        >>> HWP = half_waveplate(0)
        >>> combined = cascade_jones([QWP, HWP])
    """
    result = [[1, 0], [0, 1]]
    for M in matrices:
        result = mul(M, result)
    return result


def is_unitary_jones(matrix: List[List[complex]], tol: float = 1e-10) -> bool:
    """
    Check if a Jones matrix is unitary (lossless optical element).

    A lossless optical element preserves intensity, so its Jones matrix
    must be unitary: U^†U = I.

    Args:
        matrix: 2x2 Jones matrix.
        tol: Numerical tolerance.

    Returns:
        True if the matrix is unitary, False otherwise.

    Examples:
        >>> HWP = half_waveplate(0)
        >>> print(is_unitary_jones(HWP))
        True
        >>> polarizer_matrix = polarizer(0)
        >>> print(is_unitary_jones(polarizer_matrix))  # Polarizers are lossy
        False
    """
    from ..math.qalgebra import is_unitary
    return is_unitary(matrix, tol)


__all__ = [
    # Jones vectors
    'horizontal', 'vertical', 'diagonal', 'anti_diagonal',
    'circular_right', 'circular_left', 'elliptical',
    # Jones matrices
    'polarizer', 'waveplate', 'quarter_waveplate', 'half_waveplate',
    'full_waveplate', 'rotator', 'linear_retarder', 'circular_dichroism',
    # Analysis
    'stokes_parameters', 'degree_of_polarization',
    'polarization_angle', 'ellipticity',
    # Poincaré sphere
    'to_poincare', 'from_poincare',
    # Operations
    'apply_jones', 'cascade_jones', 'is_unitary_jones',
]