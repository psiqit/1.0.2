"""
psiqit/interface/notebook.py
============================================================
Jupyter Notebook Tools
============================================================

Interactive tools for Jupyter notebooks.

This module provides interactive visualization tools for Jupyter notebooks,
allowing users to display quantum circuits, states, measurement histograms,
Bloch spheres, and matrices directly within the notebook environment.

The tools are designed to work with IPython's rich display system and
matplotlib for high-quality visualizations.

Example:
    >>> from psiqit.interface.notebook import display_circuit, display_state, init_notebook
    >>> from psiqit.circuits.circuit import QuantumCircuit
    >>> 
    >>> init_notebook()
    >>> circ = QuantumCircuit(2)
    >>> circ.h(0)
    >>> circ.cx(0, 1)
    >>> display_circuit(circ)
    >>> 
    >>> from psiqit.quantum.state import plus
    >>> display_state(plus(), title="|+⟩ State")
"""

from typing import Optional, List, Dict, Any
from IPython.display import display, HTML, Latex, SVG
import matplotlib.pyplot as plt
from matplotlib import rcParams


def init_notebook():
    """
    Initialize Jupyter notebook display settings.

    This function sets up matplotlib styles and figure parameters for
    optimal display in Jupyter notebooks. It should be called at the
    beginning of a notebook session.

    Examples:
        >>> init_notebook()
        ✅ psiqit notebook mode initialized
        Available functions:
          - display_circuit()   : Draw quantum circuit
          - display_state()     : Show quantum state
          - display_histogram() : Plot measurement histogram
          - display_bloch()     : Show Bloch sphere
          - display_matrix()    : Show matrix with color map
    """
    # Set matplotlib style for notebooks
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    
    rcParams['figure.figsize'] = (10, 6)
    rcParams['font.size'] = 12
    
    print("✅ psiqit notebook mode initialized")
    print("Available functions:")
    print("  - display_circuit()   : Draw quantum circuit")
    print("  - display_state()     : Show quantum state")
    print("  - display_histogram() : Plot measurement histogram")
    print("  - display_bloch()     : Show Bloch sphere")
    print("  - display_matrix()    : Show matrix with color map")


def display_circuit(circuit, style: str = 'text'):
    """
    Display a quantum circuit in the notebook.

    Args:
        circuit: QuantumCircuit instance to display.
        style: Display style - 'text' (ASCII/Unicode), 'matplotlib', or 'latex'.

    Examples:
        >>> from psiqit.circuits.circuit import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> 
        >>> # Text style (default)
        >>> display_circuit(circ, style='text')
        >>> 
        >>> # Unicode style
        >>> display_circuit(circ, style='unicode')
    """
    from ..visualization.circuit_drawer import draw_circuit, circuit_to_text
    
    if style == 'text':
        print(circuit_to_text(circuit))
        print("\n" + draw_circuit(circuit, style='unicode'))
    
    elif style == 'matplotlib':
        try:
            from qiskit.visualization import circuit_drawer
            # This is a placeholder - would need qiskit
            print("Matplotlib drawing requires qiskit. Use style='text' instead.")
        except ImportError:
            print(draw_circuit(circuit, style='unicode'))
    
    else:
        display(HTML(f"<pre>{draw_circuit(circuit, style='unicode')}</pre>"))


def display_state(state, title: str = "Quantum State"):
    """
    Display a quantum state in Dirac notation.

    Args:
        state: Ket state to display.
        title: Title for the display box.

    Examples:
        >>> from psiqit.quantum.state import plus, bell_phi_plus
        >>> 
        >>> # Single qubit state
        >>> display_state(plus(), title="|+⟩ State")
        >>> 
        >>> # Bell state
        >>> display_state(bell_phi_plus(), title="Bell State |Φ⁺⟩")
    """
    from ..visualization.circuit_drawer import circuit_statistics
    
    amplitudes = state.data
    n = len(amplitudes)
    n_qubits = int(math.log2(n)) if n > 0 else 0
    
    html = f"""
    <div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">
        <h4>{title}</h4>
        <p><b>Number of qubits:</b> {n_qubits}</p>
        <p><b>Dimension:</b> {n}</p>
        <p><b>Amplitudes:</b></p>
        <ul>
    """
    
    for i, a in enumerate(amplitudes):
        if abs(a) > 1e-6:
            binary = format(i, f'0{n_qubits}b')
            html += f"<li>|{binary}⟩: {a.real:.4f} + {a.imag:.4f}i</li>\n"
    
    html += """
        </ul>
    </div>
    """
    
    display(HTML(html))


def display_histogram(counts: Dict[str, int], title: str = "Measurement Results"):
    """
    Display a measurement result histogram.

    Args:
        counts: Dictionary mapping bitstrings to measurement counts.
        title: Plot title.

    Examples:
        >>> from psiqit.circuits.circuit import QuantumCircuit
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> result = circ.measure(shots=1024)
        >>> display_histogram(result['counts'], title="Bell State Measurements")
    """
    outcomes = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(outcomes, values, color='steelblue', edgecolor='black')
    plt.xlabel('Outcome')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def display_bloch(state, title: str = "Bloch Sphere"):
    """
    Display a Bloch sphere visualization for a single qubit state.

    Args:
        state: Single qubit state (Ket).
        title: Plot title.

    Examples:
        >>> from psiqit.quantum.state import zero, one, plus, ip
        >>> 
        >>> display_bloch(zero(), title="|0⟩ State")
        >>> display_bloch(plus(), title="|+⟩ State")
        >>> display_bloch(ip(), title="|i+⟩ State")
    """
    from ..visualization.bloch import bloch_sphere
    bloch_sphere(state, title=title)


def display_matrix(matrix, title: str = "Matrix", cmap: str = 'RdBu'):
    """
    Display a matrix with color maps for real and imaginary parts.

    Args:
        matrix: 2D list or Operator object.
        title: Plot title.
        cmap: Colormap name (default: 'RdBu').

    Examples:
        >>> from psiqit.quantum.operator import hadamard, pauli_x
        >>> 
        >>> # Display Hadamard matrix
        >>> display_matrix(hadamard(), title="Hadamard Gate")
        >>> 
        >>> # Display custom matrix
        >>> H = [[1, 1], [1, -1]]
        >>> display_matrix(H, title="Hadamard", cmap='coolwarm')
    """
    import numpy as np
    
    if hasattr(matrix, 'data'):
        mat = matrix.data
    else:
        mat = matrix
    
    n = len(mat)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real part
    real_mat = [[mat[i][j].real for j in range(n)] for i in range(n)]
    im1 = ax1.imshow(real_mat, cmap=cmap, aspect='equal')
    ax1.set_title(f"{title} - Real Part")
    plt.colorbar(im1, ax=ax1)
    
    # Imaginary part
    imag_mat = [[mat[i][j].imag for j in range(n)] for i in range(n)]
    im2 = ax2.imshow(imag_mat, cmap=cmap, aspect='equal')
    ax2.set_title(f"{title} - Imaginary Part")
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def display_notebook_version():
    """
    Display notebook version information and available modules.

    This function shows the PSIQIT version, author, and a list of
    available modules for quick reference in the notebook.

    Examples:
        >>> display_notebook_version()
        ╔══════════════════════════════════════════════════════════════╗
        ║  psiqit - Quantum Information Toolkit                      ║
        ║  Version: 1.0.0                                            ║
        ║  Author: PSIQIT Contributors                                ║
        ╠══════════════════════════════════════════════════════════════╣
        ║  Available modules:                                         ║
        ║    - psiqit.math       - Mathematical operations           ║
        ║    - psiqit.quantum    - Quantum states and operators       ║
        ║    - psiqit.circuits   - Quantum circuits                   ║
        ║    - psiqit.algorithms - Quantum algorithms                 ║
        ║    - psiqit.visualization - Visualization tools            ║
        ║    - psiqit.qml        - Quantum machine learning           ║
        ╚══════════════════════════════════════════════════════════════╝
    """
    from ..version import __version__, __author__
    
    html = f"""
    <div style="background-color: #f0f0f0; padding: 15px; border-radius: 10px;">
        <h2>psiqit - Quantum Information Toolkit</h2>
        <p><b>Version:</b> {__version__}</p>
        <p><b>Author:</b> {__author__}</p>
        <hr>
        <p><b>Available modules:</b></p>
        <ul>
            <li><code>psiqit.math</code> - Mathematical operations</li>
            <li><code>psiqit.quantum</code> - Quantum states and operators</li>
            <li><code>psiqit.circuits</code> - Quantum circuits</li>
            <li><code>psiqit.algorithms</code> - Quantum algorithms</li>
            <li><code>psiqit.visualization</code> - Visualization tools</li>
            <li><code>psiqit.qml</code> - Quantum machine learning</li>
        </ul>
    </div>
    """
    
    display(HTML(html))


__all__ = [
    'init_notebook',
    'display_circuit',
    'display_state',
    'display_histogram',
    'display_bloch',
    'display_matrix',
    'display_notebook_version',
]