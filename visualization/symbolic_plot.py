"""
psiqit/visualization/symbolic_plot.py
============================================================
Symbolic Equation Visualization
============================================================

Render mathematical equations as images or text using LaTeX.

This module provides tools for generating LaTeX representations of
common quantum mechanics equations and rendering them as images.
It is useful for:
    - Including equations in Jupyter notebooks
    - Generating figures for presentations and papers
    - Creating LaTeX documents programmatically

The module includes predefined equations for:
    - Schrödinger equation (time-dependent and time-independent)
    - Dirac notation (ket, bra, inner product, outer product, expectation)
    - Pauli matrices
    - Bloch sphere representation
    - Heisenberg uncertainty principle
    - Commutation relations
    - Maxwell's equations

Example:
    >>> from psiqit.visualization.symbolic_plot import render_equation, schrodinger_equation
    >>> 
    >>> # Get Schrödinger equation
    >>> eq = schrodinger_equation()
    >>> print(eq)
    i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)
    >>> 
    >>> # Render as image
    >>> render_equation(eq, output_file="schrodinger.png")

References:
    LaTeX documentation: https://www.latex-project.org/
    Matplotlib text rendering: https://matplotlib.org/stable/users/explain/text/text_intro.html
"""

import os
from typing import Optional, Dict, List


def render_equation(latex: str, output_file: Optional[str] = None, 
                   fontsize: int = 16) -> str:
    """
    Render a LaTeX equation to an image or return it as a string.

    This function uses matplotlib to render LaTeX equations as high-quality
    images. If matplotlib is not available, it returns the LaTeX string
    wrapped in dollar signs for inline rendering.

    Args:
        latex: LaTeX equation string (without $$ delimiters).
        output_file: Optional file path for output (PNG format).
        fontsize: Font size for rendering (default: 16).

    Returns:
        - If output_file is provided: path to the saved image file
        - If output_file is None and matplotlib is available: displays the image
        - If matplotlib is not available: returns LaTeX string wrapped in $$.

    Examples:
        >>> # Display equation in notebook
        >>> render_equation(r"E = mc^2")
        >>> 
        >>> # Save to file
        >>> render_equation(r"i\hbar\frac{\partial}{\partial t}\psi = H\psi",
        ...                 output_file="schrodinger.png")
        >>> 
        >>> # Get LaTeX string
        >>> eq = render_equation(r"\hat{H}|\psi\rangle = E|\psi\rangle")
        >>> print(eq)
        $\hat{H}|\psi\rangle = E|\psi\rangle$

    Notes:
        Requires matplotlib to be installed for image rendering.
    """
    # Try to use matplotlib if available
    try:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, f'${latex}$', 
                 fontsize=fontsize, ha='center', va='center')
        plt.axis('off')
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            return output_file
        else:
            plt.show()
            return latex
    
    except ImportError:
        return f"${latex}$"


def schrodinger_equation(time_dependent: bool = True) -> str:
    """
    Return the Schrödinger equation in LaTeX format.

    Args:
        time_dependent: If True, return time-dependent version (default).
                        If False, return time-independent version.

    Returns:
        LaTeX string of the Schrödinger equation.

    Examples:
        >>> # Time-dependent Schrödinger equation
        >>> eq = schrodinger_equation(time_dependent=True)
        >>> print(eq)
        i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)
        >>> 
        >>> # Time-independent Schrödinger equation
        >>> eq = schrodinger_equation(time_dependent=False)
        >>> print(eq)
        \hat{H}\psi(\mathbf{r}) = E\psi(\mathbf{r})
    """
    if time_dependent:
        return r"i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)"
    else:
        return r"\hat{H}\psi(\mathbf{r}) = E\psi(\mathbf{r})"


def dirac_notation() -> Dict[str, str]:
    """
    Return common Dirac notation equations.

    Returns:
        Dictionary with keys:
            - 'ket': |ψ⟩
            - 'bra': ⟨ψ|
            - 'inner': ⟨φ|ψ⟩
            - 'outer': |ψ⟩⟨φ|
            - 'expectation': ⟨ψ|Â|ψ⟩

    Examples:
        >>> dirac = dirac_notation()
        >>> print(dirac['ket'])
        |\psi\rangle
        >>> print(dirac['expectation'])
        \langle\psi|\hat{A}|\psi\rangle
    """
    return {
        'ket': r"|\psi\rangle",
        'bra': r"\langle\psi|",
        'inner': r"\langle\phi|\psi\rangle",
        'outer': r"|\psi\rangle\langle\phi|",
        'expectation': r"\langle\psi|\hat{A}|\psi\rangle",
    }


def pauli_matrices() -> Dict[str, str]:
    """
    Return Pauli matrix equations in LaTeX format.

    Returns:
        Dictionary with keys:
            - 'sigma_x': σₓ matrix
            - 'sigma_y': σᵧ matrix
            - 'sigma_z': σ₂ matrix

    Examples:
        >>> pauli = pauli_matrices()
        >>> print(pauli['sigma_x'])
        \sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
    """
    return {
        'sigma_x': r"\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}",
        'sigma_y': r"\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}",
        'sigma_z': r"\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}",
    }


def bloch_sphere_equation(theta: float = None, phi: float = None) -> str:
    """
    Return the Bloch sphere representation of a qubit state.

    Args:
        theta: Polar angle (0 to π). If provided, includes numeric value.
        phi: Azimuthal angle (0 to 2π). If provided, includes numeric value.

    Returns:
        LaTeX string of the Bloch sphere state.

    Examples:
        >>> # General form
        >>> print(bloch_sphere_equation())
        |\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle
        >>> 
        >>> # With specific angles
        >>> print(bloch_sphere_equation(theta=1.57, phi=0))
        |\psi\rangle = \cos\left(\frac{1.57}{2}\right)|0\rangle + e^{0.00i}\sin\left(\frac{1.57}{2}\right)|1\rangle
    """
    if theta is not None and phi is not None:
        return f"|\\psi\\rangle = \\cos\\left(\\frac{{{theta:.2f}}}{{2}}\\right)|0\\rangle + e^{{{phi:.2f}i}}\\sin\\left(\\frac{{{theta:.2f}}}{{2}}\\right)|1\\rangle"
    return r"|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle"


def heisenberg_uncertainty() -> str:
    """
    Return the Heisenberg uncertainty principle.

    Returns:
        LaTeX string: Δx Δp ≥ ℏ/2

    Examples:
        >>> print(heisenberg_uncertainty())
        \Delta x \Delta p \geq \frac{\hbar}{2}
    """
    return r"\Delta x \Delta p \geq \frac{\hbar}{2}"


def commutation_relation() -> str:
    """
    Return the canonical commutation relation.

    Returns:
        LaTeX string: [x, p] = iℏ

    Examples:
        >>> print(commutation_relation())
        [x, p] = i\hbar
    """
    return r"[x, p] = i\hbar"


def maxwell_equations() -> List[str]:
    """
    Return Maxwell's equations in LaTeX format.

    Returns:
        List of four strings representing:
            - Gauss's law for electricity
            - Gauss's law for magnetism
            - Faraday's law of induction
            - Ampère's law (with Maxwell's correction)

    Examples:
        >>> eqs = maxwell_equations()
        >>> for eq in eqs:
        ...     print(eq)
        \nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}
        \nabla \cdot \mathbf{B} = 0
        \nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}
        \nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial \mathbf{E}}{\partial t}
    """
    return [
        r"\nabla \cdot \mathbf{E} = \frac{\rho}{\varepsilon_0}",
        r"\nabla \cdot \mathbf{B} = 0",
        r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}",
        r"\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial \mathbf{E}}{\partial t}",
    ]


__all__ = [
    'render_equation',
    'schrodinger_equation',
    'dirac_notation',
    'pauli_matrices',
    'bloch_sphere_equation',
    'heisenberg_uncertainty',
    'commutation_relation',
    'maxwell_equations',
]