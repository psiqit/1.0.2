# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Basic Installation

The simplest way to install PSIQIT is via pip:

```bash
pip install psiqit
Installation with Optional Dependencies
PSIQIT offers several optional dependency groups for extended functionality:

For Visualization
Install matplotlib for Bloch sphere plots, Wigner functions, and circuit drawing:

bash
pip install psiqit[viz]
For Machine Learning
Install scikit-learn for classical ML components in QSVM and QNN:

bash
pip install psiqit[ml]
For Symbolic Mathematics
Install SymPy for symbolic algebra and equation solving:

bash
pip install psiqit[symbolic]
For Advanced Scientific Computing
Install SciPy for advanced ODE/PDE solvers:

bash
pip install psiqit[scipy]
Complete Installation
Install all optional dependencies:

bash
pip install psiqit[full]
Installation from Source
To install the latest development version:

bash
git clone https://github.com/psiqit/psiqit.git
cd psiqit
pip install -e .
For development with all dependencies:

bash
pip install -e .[dev]
Verification
Verify the installation by running:

python
import psiqit
print(psiqit.__version__)  # Should print version number
Docker Installation
Pull the PSIQIT Docker image:

bash
docker pull psiqit/psiqit:latest
Run a container:

bash
docker run -it psiqit/psiqit:latest python
Common Issues
matplotlib not found
If you get ModuleNotFoundError: No module named 'matplotlib', install it with:

bash
pip install matplotlib
SymPy import errors
If symbolic operations fail, install SymPy:

bash
pip install sympy
NumPy version conflicts
PSIQIT requires NumPy 1.21.0 or higher. Upgrade if needed:

bash
pip install --upgrade numpy
Next Steps
Check out the Quick Start Guide for basic usage examples

Explore the User Guide for detailed tutorials

Browse the API Reference for complete documentation