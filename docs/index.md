
# PSIQIT - Quantum Information Toolkit

<div align="center">
  <img src="assets/logo.jpg" alt="PSIQIT Logo" width="200"/>
  <br><br>
</div>

**PSIQIT** (Quantum Information Toolkit) is a comprehensive Python library for quantum computing, quantum information theory, and quantum mechanics simulations. It provides a complete set of tools for research, education, and development in quantum technologies.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/psiqit/psiqit/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/psiqit.svg)](https://badge.fury.io/py/psiqit)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://psiqit.github.io)
[![Tests](https://github.com/psiqit/psiqit/actions/workflows/tests.yml/badge.svg)](https://github.com/psiqit/psiqit/actions)

---

## ✨ Key Features

<div class="grid cards" markdown>

-   :material-calculator: **Mathematics**
    Linear algebra, calculus, ODE/PDE solvers, symbolic math

-   :material-atom: **Quantum States**
    Ket/Bra, Bell states, GHZ, W, coherent, squeezed, Fock states

-   :material-gate: **Quantum Gates**
    Pauli, Hadamard, rotations, CNOT, Toffoli, Fredkin

-   :material-circuit-board: **Quantum Circuits**
    Circuit builder, simulation, visualization

-   :material-chart-bell-curve: **Measurements**
    POVM, projective measurements, tomography

-   :material-algorithm: **Algorithms**
    Grover, Shor, Deutsch-Jozsa, QFT, QPE, Simon

-   :material-chart-line: **Variational Methods**
    VQE, QAOA, SSVQE, ADAPT-VQE, TDVP

-   :material-connection: **Entanglement**
    Concurrence, negativity, Schmidt decomposition

-   :material-wave: **Open Systems**
    Lindblad equation, quantum trajectories, Monte Carlo

-   :material-shield: **Error Correction**
    Bit flip, phase flip, Shor code, Steane code

-   :material-robot: **Quantum ML**
    QSVM, QNN, VQC, QGAN, quantum kernels

-   :material-chart-scatter-3d: **Visualization**
    Bloch sphere, Wigner function, circuit drawing

</div>

---

## 📦 Installation

```bash
pip install psiqit
```

### With Optional Dependencies

```bash
# For visualization
pip install psiqit[viz]

# For machine learning
pip install psiqit[ml]

# For symbolic mathematics
pip install psiqit[symbolic]

# Install all
pip install psiqit[full]
```

---

## 🚀 Quick Example

```python
from psiqit.circuits.circuit import QuantumCircuit
from psiqit.quantum.state import zero
from psiqit.quantum.operator import hadamard, cnot

# Create Bell state
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
state = circ.run()

print(state)  # 0.707|00⟩ + 0.707|11⟩
```

---

## 📚 Documentation Sections

| Section | Description |
|---------|-------------|
| Getting Started | Installation and quick start guide |
| User Guide | In-depth tutorials on quantum computing concepts |
| API Reference | Complete API documentation |
| Examples | Code examples for common tasks |
| Tutorials | Jupyter notebook tutorials |

---

## 📁 Project Structure

```
psiqit/
│
├── algorithms/       # Quantum algorithms
├── circuits/         # Quantum circuits
├── dynamics/         # Quantum dynamics
├── error_correction/ # Error correction codes
├── info/             # Information theory
├── interface/        # CLI, LaTeX, Jupyter tools
├── lab/              # Virtual quantum laboratory
├── math/             # Mathematical toolkit
├── noise_canceling/  # Noise models and mitigation
├── qml/              # Quantum machine learning
├── quantum/          # Quantum foundations
├── utils/            # Utilities
├── variational/      # Variational methods
└── visualization/    # Visualization tools
```

---

## 🤝 Contributing

Contributions are welcome! Please see our Contributing Guide.

### Development Setup

```bash
git clone https://github.com/psiqit/psiqit.git
cd psiqit
pip install -e .[dev]
pytest tests/
```

---

## 📄 License

PSIQIT is released under the MIT License.

---

## 📧 Contact

**Author:** Mahdi Azadmarzabadi

**Maintainer:** PSIQIT Developer Team

**Email:** psiqitofficial@protonmail.com

**GitHub:** https://github.com/psiqit/psiqit

---

## 🌟 Citation

If you use PSIQIT in your research, please cite:

```bibtex
@software{psiqit2025,
  title     = {PSIQIT: Quantum Information Toolkit},
  author    = {Mahdi Azadmarzabadi and PSIQIT Contributors},
  year      = {2025},
  url       = {https://github.com/psiqit/psiqit},
  version   = {1.0.2},
  publisher = {GitHub},
  month     = jan
}
```
