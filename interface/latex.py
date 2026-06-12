"""
psiqit/interface/latex.py
============================================================
LaTeX Output Generator
============================================================

Generate LaTeX code for quantum circuits, matrices, and equations.

This module provides functions to convert quantum objects (matrices, states,
circuits, equations, tables) into LaTeX code suitable for inclusion in
scientific papers, reports, and presentations.

The generated LaTeX uses:
    - quantikz package for quantum circuits (recommended)
    - qcircuit package as an alternative
    - amsmath for matrices and equations

Example:
    >>> from psiqit.interface.latex import matrix_to_latex, circuit_to_latex
    >>> from psiqit.circuits.circuit import QuantumCircuit
    >>> 
    >>> # Convert matrix to LaTeX
    >>> H = [[1, 1], [1, -1]]
    >>> print(matrix_to_latex(H, name="H"))
    H = \begin{pmatrix}
      1.0000 & 1.0000 \\
      1.0000 & -1.0000
    \end{pmatrix}
    >>> 
    >>> # Convert circuit to LaTeX (using quantikz)
    >>> circ = QuantumCircuit(2)
    >>> circ.h(0)
    >>> circ.cx(0, 1)
    >>> print(circuit_to_latex(circ))
    \\begin{quantikz}
        \\lstick{q_0} & \\gate{H} & \\ctrl{1} & \\qw \\\\
        \\lstick{q_1} & \\qw & \\targ{-1} & \\qw \\\\
    \\end{quantikz}
"""

from typing import List, Union, Optional, Dict, Any
from ..quantum.operator import Operator


def matrix_to_latex(matrix: Union[List[List], Operator], 
                    name: str = "",
                    precision: int = 4,
                    as_array: bool = False) -> str:
    """
    Convert a matrix to LaTeX format.

    This function handles both real and complex matrices, formatting complex
    numbers appropriately. It supports both pmatrix (default) and array
    environments.

    Args:
        matrix: Matrix as list of lists or Operator object.
        name: Optional matrix name (e.g., "H", "X", "U") for inline labeling.
        precision: Number of decimal places for formatting numbers.
        as_array: If True, use array environment; otherwise use pmatrix.

    Returns:
        LaTeX string representing the matrix.

    Examples:
        >>> # Pauli X matrix
        >>> X = [[0, 1], [1, 0]]
        >>> print(matrix_to_latex(X, name="X"))
        X = \begin{pmatrix}
          0.0000 & 1.0000 \\
          1.0000 & 0.0000
        \end{pmatrix}
        >>> 
        >>> # Complex matrix
        >>> Y = [[0, -1j], [1j, 0]]
        >>> print(matrix_to_latex(Y, name="Y", precision=2))
        Y = \begin{pmatrix}
          0.00 & -1.00i \\
          1.00i & 0.00
        \end{pmatrix}
        >>> 
        >>> # Using array environment
        >>> print(matrix_to_latex(X, as_array=True))
        \begin{array}{cc}
          0.0000 & 1.0000 \\
          1.0000 & 0.0000
        \end{array}
    """
    if isinstance(matrix, Operator):
        mat = matrix.data
    else:
        mat = matrix
    
    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0
    
    env = "array" if as_array else "pmatrix"
    latex = f"\\begin{{{env}}}\n"
    
    for i in range(rows):
        row = []
        for j in range(cols):
            val = mat[i][j]
            if isinstance(val, complex):
                if abs(val.imag) < 10**(-precision):
                    row.append(f"{val.real:.{precision}f}")
                elif abs(val.real) < 10**(-precision):
                    if val.imag > 0:
                        row.append(f"{val.imag:.{precision}f}i")
                    else:
                        row.append(f"-{abs(val.imag):.{precision}f}i")
                else:
                    sign = "+" if val.imag >= 0 else "-"
                    row.append(f"{val.real:.{precision}f}{sign}{abs(val.imag):.{precision}f}i")
            else:
                row.append(f"{val:.{precision}f}")
        
        latex += "  " + " & ".join(row) + " \\\\\n"
    
    latex += f"\\end{{{env}}}"
    
    if name:
        latex = f"{name} = {latex}"
    
    return latex


def state_to_latex(state, name: str = "\\psi", precision: int = 4) -> str:
    """
    Convert a quantum state to LaTeX Dirac notation.

    Args:
        state: Ket state (from psiqit.quantum.state) or list of amplitudes.
        name: State name (e.g., "\\psi", "\\phi", "\\alpha").
        precision: Number of decimal places for formatting amplitudes.

    Returns:
        LaTeX string in Dirac notation.

    Examples:
        >>> from psiqit.quantum.state import plus, zero
        >>> print(state_to_latex(zero(), name="0"))
        |0\rangle = 1.0000|0\rangle
        >>> 
        >>> print(state_to_latex(plus(), name="\\psi"))
        |\\psi\rangle = 0.7071|0\rangle + 0.7071|1\rangle
        >>> 
        >>> # Custom amplitude list
        >>> print(state_to_latex([0.6, 0.8], name="\\phi"))
        |\\phi\rangle = 0.6000|0\rangle + 0.8000|1\rangle
    """
    if hasattr(state, 'data'):
        amplitudes = state.data
    else:
        amplitudes = state
    
    terms = []
    for i, a in enumerate(amplitudes):
        if abs(a) > 1e-10:
            if isinstance(a, complex):
                if abs(a.imag) < 10**(-precision):
                    coeff = f"{a.real:.{precision}f}"
                elif abs(a.real) < 10**(-precision):
                    if a.imag > 0:
                        coeff = f"{a.imag:.{precision}f}i"
                    else:
                        coeff = f"-{abs(a.imag):.{precision}f}i"
                else:
                    sign = "+" if a.imag >= 0 else ""
                    coeff = f"{a.real:.{precision}f}{sign}{abs(a.imag):.{precision}f}i"
            else:
                coeff = f"{a:.{precision}f}"
            
            if coeff == "1.0000":
                coeff = ""
            elif coeff == "-1.0000":
                coeff = "-"
            
            terms.append(f"{coeff}|{i}\\rangle")
    
    latex = " + ".join(terms).replace("+ -", "- ")
    
    if name:
        latex = f"|{name}\\rangle = {latex}"
    
    return latex


def circuit_to_latex(circuit, style: str = 'quantikz', 
                     wire_labels: Optional[List[str]] = None) -> str:
    """
    Convert a quantum circuit to LaTeX code.

    This function generates LaTeX code for quantum circuits using either
    the quantikz package (recommended) or the older qcircuit package.

    Args:
        circuit: QuantumCircuit instance.
        style: 'quantikz' (modern, recommended) or 'qcircuit' (legacy).
        wire_labels: Custom labels for qubit wires (e.g., ["|0⟩", "|1⟩"]).
                     If None, uses default "q_0", "q_1", etc.

    Returns:
        LaTeX string for the circuit.

    Examples:
        >>> from psiqit.circuits.circuit import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> 
        >>> # Using quantikz (recommended)
        >>> print(circuit_to_latex(circ, style='quantikz'))
        \\begin{quantikz}
            \\lstick{q_0} & \\gate{H} & \\ctrl{1} & \\qw \\\\
            \\lstick{q_1} & \\qw & \\targ{-1} & \\qw \\\\
        \\end{quantikz}
        >>> 
        >>> # Using qcircuit
        >>> print(circuit_to_latex(circ, style='qcircuit'))
        \\Qcircuit @C=1em @R=1em {
            & \\lstick{q_0} & \\gate{H} & \\ctrl{1} & \\qw \\\\
            & \\lstick{q_1} & \\qw & \\targ & \\qw \\\\
        }
        >>> 
        >>> # Custom wire labels
        >>> print(circuit_to_latex(circ, wire_labels=["|0⟩", "|1⟩"]))
    """
    n = circuit.n_qubits
    
    if wire_labels is None:
        wire_labels = [f"q_{i}" for i in range(n)]
    
    if style == 'quantikz':
        latex = "\\begin{quantikz}\n"
        
        # Add wire labels
        for i in range(n):
            latex += f"    \\lstick{{{wire_labels[i]}}} & "
        
        latex = latex.rstrip(" & ") + " \\\\\n"
        
        # Add gates
        for gate, qubits in circuit._gates:
            gate_name = gate.name or "U"
            
            # Create gate string for each qubit
            row = []
            for i in range(n):
                if i in qubits:
                    if len(qubits) == 1:
                        row.append(f"\\gate{{{gate_name}}}")
                    elif len(qubits) == 2:
                        if i == qubits[0]:
                            row.append(f"\\ctrl{{{qubits[1] - qubits[0]}}}")
                        else:
                            row.append(f"\\targ{{{qubits[0] - qubits[1]}}}")
                else:
                    row.append("\\qw")
            
            latex += "    " + " & ".join(row) + " \\\\\n"
        
        latex += "\\end{quantikz}"
        
    else:  # qcircuit style
        latex = "\\Qcircuit @C=1em @R=1em {\n"
        
        for i in range(n):
            latex += f"    & \\lstick{{{wire_labels[i]}}} "
        
        latex += "\\\\\n"
        
        for gate, qubits in circuit._gates:
            gate_name = gate.name or "U"
            
            latex += "    & "
            for i in range(n):
                if i in qubits:
                    if len(qubits) == 1:
                        latex += f"\\gate{{{gate_name}}}"
                    elif len(qubits) == 2:
                        if i == qubits[0]:
                            latex += f"\\ctrl{{{qubits[1] - qubits[0]}}}"
                        else:
                            latex += f"\\targ"
                else:
                    latex += "\\qw"
                latex += " & "
            
            latex = latex.rstrip(" & ") + "\\\\\n"
        
        latex += "}"
    
    return latex


def equation_to_latex(expr: str, name: str = "", align: bool = False) -> str:
    """
    Wrap an equation in a LaTeX math environment.

    Args:
        expr: LaTeX expression to be displayed.
        name: Optional equation label for referencing (e.g., "eq:schrodinger").
        align: If True, use align environment; otherwise use equation.

    Returns:
        LaTeX string with the equation environment.

    Examples:
        >>> print(equation_to_latex(r"e^{i\pi} = -1", name="euler"))
        \\begin{equation}
            e^{i\\pi} = -1
            \\label{euler}
        \\end{equation}
        >>> 
        >>> # Multi-line equation with align
        >>> expr = r"a &= b + c \\\\ d &= e + f"
        >>> print(equation_to_latex(expr, align=True))
        \\begin{align}
            a &= b + c \\\\ d &= e + f
        \\end{align}
    """
    env = "align" if align else "equation"
    latex = f"\\begin{{{env}}}\n"
    latex += f"    {expr}\n"
    
    if name:
        latex += f"    \\label{{{name}}}\n"
    
    latex += f"\\end{{{env}}}"
    
    return latex


def table_to_latex(data: List[List], 
                   headers: Optional[List[str]] = None,
                   caption: str = "",
                   label: str = "") -> str:
    """
    Convert a data table to LaTeX tabular format.

    Args:
        data: 2D list of data (rows, columns).
        headers: Optional list of column headers.
        caption: Optional table caption.
        label: Optional table label for referencing.

    Returns:
        LaTeX string with the table environment.

    Examples:
        >>> data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> headers = ["A", "B", "C"]
        >>> print(table_to_latex(data, headers=headers, caption="Sample Table"))
        \\begin{table}[htbp]
        \\centering
        \\begin{tabular}{|c|c|c|}
        \\hline
        A & B & C \\\\
        \\hline
        1 & 2 & 3 \\\\
        \\hline
        4 & 5 & 6 \\\\
        \\hline
        7 & 8 & 9 \\\\
        \\hline
        \\end{tabular}
        \\caption{Sample Table}
        \\end{table}
    """
    if not data:
        return ""
    
    n_cols = len(data[0])
    col_format = "|" + "c|" * n_cols
    
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    latex += "\\hline\n"
    
    if headers:
        latex += " & ".join(headers) + " \\\\\n"
        latex += "\\hline\n"
    
    for row in data:
        row_str = []
        for cell in row:
            if isinstance(cell, float):
                row_str.append(f"{cell:.4f}")
            else:
                row_str.append(str(cell))
        latex += " & ".join(row_str) + " \\\\\n"
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    if label:
        latex += f"\\label{{{label}}}\n"
    
    latex += "\\end{table}"
    
    return latex


def generate_report(results: Dict[str, Any], title: str = "Quantum Report") -> str:
    """
    Generate a complete LaTeX report from a results dictionary.

    This function creates a full LaTeX document with sections, subsections,
    equations, tables, and itemized lists based on the structure of the
    results dictionary.

    Args:
        results: Dictionary containing the results to be formatted.
        title: Title of the report.

    Returns:
        Complete LaTeX document as a string.

    Examples:
        >>> results = {
        ...     "Experiment": {
        ...         "Fidelity": 0.98,
        ...         "Counts": {"00": 512, "11": 512}
        ...     },
        ...     "Parameters": ["r=2.0", "g=1.0"]
        ... }
        >>> report = generate_report(results, title="Teleportation Report")
        >>> print(report[:200])  # First 200 characters
        \\documentclass{article}
        \\usepackage{amsmath, amssymb}
        \\usepackage{quantikz}
        \\usepackage{graphicx}
        ...
    """
    latex = f"""\\documentclass{{article}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{quantikz}}
\\usepackage{{graphicx}}

\\title{{{title}}}
\\author{{psiqit}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{Introduction}}
This report was generated automatically by the psiqit (Quantum Information Toolkit).

"""

    for key, value in results.items():
        latex += f"\\section{{{key}}}\n"
        
        if isinstance(value, dict):
            for subkey, subval in value.items():
                latex += f"\\subsection{{{subkey}}}\n"
                if isinstance(subval, list) and len(subval) > 0 and isinstance(subval[0], list):
                    latex += table_to_latex(subval, caption=subkey)
                else:
                    latex += f"\\[\n    {subval}\n\\]\n"
        elif isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], (int, float)):
                latex += "\\begin{itemize}\n"
                for v in value:
                    latex += f"    \\item {v}\n"
                latex += "\\end{itemize}\n"
            elif len(value) > 0 and isinstance(value[0], list):
                latex += table_to_latex(value, caption=key)
        elif isinstance(value, (int, float)):
            latex += f"\\[\n    {key} = {value}\n\\]\n"
        else:
            latex += f"\\[\n    {value}\n\\]\n"
    
    latex += """
\\end{document}
"""
    
    return latex


__all__ = [
    'matrix_to_latex',
    'state_to_latex',
    'circuit_to_latex',
    'equation_to_latex',
    'table_to_latex',
    'generate_report',
]