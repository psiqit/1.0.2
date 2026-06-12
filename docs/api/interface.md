
# Interface API

## Module: `psiqit.interface`

This module provides interfaces for interacting with PSIQIT from different environments:
- **Command Line Interface (CLI)** - Run quantum circuits and algorithms from terminal
- **LaTeX Output Generation** - Generate publication-ready LaTeX code for matrices, states, circuits, and equations
- **Jupyter Notebook Tools** - Interactive visualization and display tools for notebooks

---

## Command Line Interface (`cli.py`)

**Run quantum circuits and algorithms from the terminal** - Execute quantum circuits with custom gates or run predefined quantum algorithms without writing Python code.

### Functions

| Function | Description |
|----------|-------------|
| `main()` | Main CLI entry point |
| `parse_gates(gates_str)` | Parse gate string (e.g., "h0,cx01") into gate list |
| `run_circuit(args)` | Execute a quantum circuit |
| `run_algorithm(args)` | Execute a quantum algorithm |

### CLI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `circuit` | Run a custom quantum circuit | `psiqit circuit --n-qubits 2 --gates h0,cx01 --shots 100` |
| `algorithm` | Run a predefined quantum algorithm | `psiqit algorithm --algorithm grover --n-qubits 3 --target 5` |
| `info` | Display version and information | `psiqit info` |

### Gate String Format

| Gate | Format | Example |
|------|--------|---------|
| Hadamard | `h{qubit}` | `h0` |
| Pauli X | `x{qubit}` | `x1` |
| Pauli Y | `y{qubit}` | `y2` |
| Pauli Z | `z{qubit}` | `z3` |
| CNOT | `cx{control}{target}` | `cx01` |
| CZ | `cz{control}{target}` | `cz02` |
| SWAP | `swap{q1}{q2}` | `swap01` |

### Example 1: Run Circuit from CLI

```bash
# Create a Bell state circuit
python -m psiqit.interface.cli circuit --n-qubits 2 --gates h0,cx01 --shots 1024

# Output:
# ==================================================
# Quantum Circuit Results
# ==================================================
# Number of qubits: 2
# Circuit depth: 2
# Shots: 1024
#
# Measurement counts:
#   |00⟩: 512 (50.00%)
#   |11⟩: 512 (50.00%)
```

### Example 2: Run Grover's Algorithm from CLI

```bash
# Search for state 5 in 3-qubit system
python -m psiqit.interface.cli algorithm --algorithm grover --n-qubits 3 --target 5 --shots 1024

# Output:
# ==================================================
# Running GROVER Algorithm
# ==================================================
# Most likely state: 5
# Success: True
```

### Example 3: Parse Gates Programmatically

```python
from psiqit.interface.cli import parse_gates

# Parse gate string
gates = parse_gates("h0,cx01,swap23")
print(gates)
# [('h', 0), ('cx', 0, 1), ('swap', 2, 3)]
```

---

## LaTeX Output Generator (`latex.py`)

Generate LaTeX code for quantum objects - Convert matrices, quantum states, circuits, equations, and tables to LaTeX for inclusion in scientific papers and reports.

### Functions

| Function | Description |
|----------|-------------|
| `matrix_to_latex(matrix, name, precision, as_array)` | Convert matrix to LaTeX |
| `state_to_latex(state, name, precision)` | Convert quantum state to Dirac notation |
| `circuit_to_latex(circuit, style, wire_labels)` | Convert circuit to LaTeX (quantikz/qcircuit) |
| `equation_to_latex(expr, name, align)` | Wrap equation in LaTeX environment |
| `table_to_latex(data, headers, caption, label)` | Convert data table to LaTeX tabular |
| `generate_report(results, title)` | Generate complete LaTeX report |

### Example 1: Matrix to LaTeX

```python
from psiqit.interface.latex import matrix_to_latex
from psiqit.quantum.operator import hadamard

# Hadamard matrix
H = hadamard()
latex = matrix_to_latex(H, name="H", precision=3)
print(latex)
```

Output:
```latex
H = \begin{pmatrix}
  0.707 & 0.707 \\
  0.707 & -0.707
\end{pmatrix}
```

### Example 2: State to LaTeX (Dirac Notation)

```python
from psiqit.interface.latex import state_to_latex
from psiqit.quantum.state import plus, bell_phi_plus

# Single qubit state
latex = state_to_latex(plus(), name="\\psi")
print(latex)
# |\psi\rangle = 0.707|0\rangle + 0.707|1\rangle

# Bell state
latex = state_to_latex(bell_phi_plus(), name="\\Phi")
print(latex)
# |\Phi\rangle = 0.707|00\rangle + 0.707|11\rangle
```

### Example 3: Circuit to LaTeX (Quantikz)

```python
from psiqit.interface.latex import circuit_to_latex
from psiqit.circuits.circuit import QuantumCircuit

# Create Bell state circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Generate LaTeX with quantikz
latex = circuit_to_latex(circ, style='quantikz')
print(latex)
```

Output:
```latex
\begin{quantikz}
    \lstick{q_0} & \gate{H} & \ctrl{1} & \qw \\
    \lstick{q_1} & \qw & \targ{-1} & \qw \\
\end{quantikz}
```

### Example 4: Custom Wire Labels

```python
# Custom wire labels
latex = circuit_to_latex(circ, wire_labels=["|0⟩", "|1⟩"])
print(latex)
# \lstick{|0⟩} & \gate{H} & \ctrl{1} & \qw \\
# \lstick{|1⟩} & \qw & \targ{-1} & \qw \\
```

### Example 5: Equation to LaTeX

```python
from psiqit.interface.latex import equation_to_latex

# Simple equation
latex = equation_to_latex(r"E = mc^2", name="einstein")
print(latex)
```

Output:
```latex
\begin{equation}
    E = mc^2
    \label{einstein}
\end{equation}
```

### Example 6: Table to LaTeX

```python
from psiqit.interface.latex import table_to_latex

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
headers = ["A", "B", "C"]

latex = table_to_latex(data, headers=headers, caption="Sample Table", label="tab:sample")
print(latex)
```

Output:
```latex
\begin{table}[htbp]
\centering
\begin{tabular}{|c|c|c|}
\hline
A & B & C \\
\hline
1 & 2 & 3 \\
\hline
4 & 5 & 6 \\
\hline
7 & 8 & 9 \\
\hline
\end{tabular}
\caption{Sample Table}
\label{tab:sample}
\end{table}
```

### Example 7: Complete LaTeX Report

```python
from psiqit.interface.latex import generate_report

results = {
    "Experiment": {
        "Fidelity": 0.98,
        "Energy": -1.0
    },
    "Parameters": ["r=2.0", "g=1.0"]
}

report = generate_report(results, title="Quantum Teleportation Report")
print(report[:500])  # First 500 characters
```

---

## Jupyter Notebook Tools (`notebook.py`)

Interactive tools for Jupyter notebooks - Display quantum circuits, states, histograms, Bloch spheres, and matrices directly in notebooks.

### Functions

| Function | Description |
|----------|-------------|
| `init_notebook()` | Initialize notebook display settings |
| `display_circuit(circuit, style)` | Draw quantum circuit |
| `display_state(state, title)` | Show quantum state in Dirac notation |
| `display_histogram(counts, title)` | Plot measurement histogram |
| `display_bloch(state, title)` | Show Bloch sphere visualization |
| `display_matrix(matrix, title, cmap)` | Display matrix with color map |
| `display_notebook_version()` | Show version information |

### Example 1: Initialize Notebook

```python
from psiqit.interface.notebook import init_notebook

init_notebook()
# ✅ psiqit notebook mode initialized
# Available functions:
#   - display_circuit()   : Draw quantum circuit
#   - display_state()     : Show quantum state
#   - display_histogram() : Plot measurement histogram
#   - display_bloch()     : Show Bloch sphere
#   - display_matrix()    : Show matrix with color map
```

### Example 2: Display Circuit

```python
from psiqit.interface.notebook import display_circuit
from psiqit.circuits.circuit import QuantumCircuit

circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

display_circuit(circ, style='text')
```

### Example 3: Display Quantum State

```python
from psiqit.interface.notebook import display_state
from psiqit.quantum.state import bell_phi_plus

bell = bell_phi_plus()
display_state(bell, title="Bell State |Φ⁺⟩")
```

Output (HTML):
```
┌─────────────────────────────────────┐
│ Bell State |Φ⁺⟩                    │
│ Number of qubits: 2                │
│ Dimension: 4                       │
│ Amplitudes:                        │
│   • |00⟩: 0.7071 + 0.0000i         │
│   • |11⟩: 0.7071 + 0.0000i         │
└─────────────────────────────────────┘
```

### Example 4: Display Histogram

```python
from psiqit.interface.notebook import display_histogram
from psiqit.circuits.circuit import QuantumCircuit

circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
results = circ.measure(shots=1024)

display_histogram(results['counts'], title="Bell State Measurements")
```

### Example 5: Display Bloch Sphere

```python
from psiqit.interface.notebook import display_bloch
from psiqit.quantum.state import plus, ip, zero

# Single state
display_bloch(plus(), title="|+⟩ State")

# Multiple states can be displayed using plot_multiple_states
from psiqit.visualization.bloch import plot_multiple_states
plot_multiple_states([zero(), plus(), ip()], colors=['blue', 'red', 'green'])
```

### Example 6: Display Matrix

```python
from psiqit.interface.notebook import display_matrix
from psiqit.quantum.operator import hadamard, cnot

# Hadamard gate
display_matrix(hadamard(), title="Hadamard Gate")

# CNOT gate
display_matrix(cnot(), title="CNOT Gate")
```

### Example 7: Complete Notebook Example

```python
from psiqit.interface.notebook import (
    init_notebook, display_circuit, display_state,
    display_histogram, display_bloch, display_notebook_version
)
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.quantum.state import plus

# Initialize
init_notebook()
display_notebook_version()

# Create Bell state
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Display circuit
display_circuit(circ, style='unicode')

# Display state
state = circ.run()
display_state(state, title="Bell State")

# Measure and display histogram
results = circ.measure(shots=1024)
display_histogram(results['counts'], title="Bell State Measurements")

# Display Bloch sphere
display_bloch(plus(), title="|+⟩ State")
```

---

## Complete Example: Generate LaTeX for a Paper

```python
from psiqit.interface.latex import matrix_to_latex, circuit_to_latex, generate_report
from psiqit.interface.notebook import init_notebook, display_circuit
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.quantum.operator import hadamard, pauli_x, pauli_z
from psiqit.quantum.state import bell_phi_plus

# Create Bell state circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Generate LaTeX for paper
latex_document = r"""
\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{quantikz}

\begin{document}

\title{Quantum Teleportation Study}
\author{PSIQIT}
\maketitle

\section{Hadamard Gate}
""" + matrix_to_latex(hadamard(), name="H") + r"""

\section{Bell State Circuit}
""" + circuit_to_latex(circ, style='quantikz') + r"""

\section{Bell State}
""" + state_to_latex(bell_phi_plus(), name="\Phi^+") + r"""

\end{document}
"""

print(latex_document)
```

---

## Module Contents

```python
__all__ = [
    # CLI
    'cli_main', 'parse_gates', 'run_circuit', 'run_algorithm',
    # LaTeX
    'matrix_to_latex', 'state_to_latex', 'circuit_to_latex',
    'equation_to_latex', 'table_to_latex', 'generate_report',
    # Notebook
    'init_notebook', 'display_circuit', 'display_state',
    'display_histogram', 'display_bloch', 'display_matrix',
    'display_notebook_version',
]
```

---

## References

| Tool | Reference |
|------|-----------|
| LaTeX | https://www.latex-project.org/ |
| Quantikz | https://github.com/Quantikz/Quantikz |
| Qcircuit | https://qcircuit.github.io/ |
| Jupyter | https://jupyter.org/ |
| Matplotlib | https://matplotlib.org/ |
