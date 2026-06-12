
# Contributing to PSIQIT

Thank you for your interest in contributing to PSIQIT! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Community](#community)

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors. Please:

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of quantum computing concepts

### First Time Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/psiqit.git
   cd psiqit
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/psiqit/psiqit.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

---

## How to Contribute

### Types of Contributions

| Type | Description | Difficulty |
|------|-------------|------------|
| 🐛 **Bug Reports** | Report issues you encounter | Beginner |
| 📝 **Documentation** | Improve docs, examples, tutorials | Beginner |
| 🎨 **Code Review** | Review pull requests | Intermediate |
| ✨ **Feature Requests** | Suggest new features | Beginner |
| 🔧 **Bug Fixes** | Fix reported issues | Intermediate |
| 🚀 **New Features** | Implement new functionality | Advanced |
| 🧪 **Tests** | Write unit tests | Intermediate |
| 📊 **Examples** | Add new examples | Beginner |

### Good First Issues

Look for issues labeled `good-first-issue` or `help-wanted`. These are perfect for new contributors.

---

## Development Setup

### Install Dependencies

```bash
# Development dependencies
pip install -e .[dev]

# With all optional dependencies
pip install -e .[full]
```

### Verify Installation

```bash
python -c "import psiqit; print(psiqit.__version__)"
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_circuits.py

# Run with coverage
pytest --cov=psiqit tests/

# Run specific test
pytest tests/test_circuits.py::TestQuantumCircuit::test_bell_state
```

### Running Linters

```bash
# Check code style
flake8 psiqit/

# Type checking
mypy psiqit/

# Format code
black psiqit/
```

---

## Code Style Guidelines

### Python Style (PEP 8)

- Use 4 spaces for indentation
- Maximum line length: 88 characters (Black default)
- Use descriptive variable names

```python
# Good
def calculate_fidelity(state1, state2):
    return abs(state1.inner(state2))**2

# Avoid
def calc_fid(s1, s2):
    return abs(s1.inner(s2))**2
```

### Type Hints

Always include type hints for function arguments and return values:

```python
from typing import List, Optional, Union
from psiqit.quantum.state import Ket

def apply_gate(
    gate: np.ndarray,
    state: Ket,
    qubits: List[int]
) -> Ket:
    """Apply gate to specific qubits."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something goes wrong.

    Examples:
        >>> my_function(1, "test")
        True
    """
    ...
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `QuantumCircuit` |
| Functions | snake_case | `compute_entropy` |
| Variables | snake_case | `n_qubits` |
| Constants | UPPER_SNAKE | `PLANCK_CONSTANT` |
| Private members | _leading_underscore | `_internal_method` |

### Imports Order

1. Standard library imports
2. Third-party imports
3. Local imports

```python
# Standard library
import math
from typing import List

# Third-party
import numpy as np

# Local
from psiqit.quantum.state import Ket
```

---

## Testing

### Writing Tests

All new features should include tests. Place tests in the `tests/` directory:

```python
# tests/test_example.py
import pytest
from psiqit.example import my_function

def test_my_function():
    result = my_function(5)
    assert result == 10

def test_my_function_raises_error():
    with pytest.raises(ValueError):
        my_function(-1)
```

### Test Coverage

Aim for at least 80% test coverage for new code.

```bash
pytest --cov=psiqit --cov-report=html tests/
```

---

## Documentation

### Building Documentation

```bash
cd docs
mkdocs serve
```

### Documentation Structure

```
docs/
├── index.md              # Home page
├── getting_started/      # Installation, quick start
├── user_guide/           # Tutorials and explanations
├── api/                  # API reference
├── examples/             # Code examples
└── tutorials/            # Jupyter notebooks
```

### Writing Examples

Examples should be complete, runnable, and well-commented:

```python
"""
Example title
=============

Brief description of what this example demonstrates.
"""

# Import required modules
from psiqit.circuits.circuit import QuantumCircuit

# Create circuit
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)

# Run simulation
state = circ.run()
print(state)  # 0.707|00⟩ + 0.707|11⟩
```

---

## Pull Request Process

### Step 1: Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Step 2: Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation as needed

### Step 3: Run Checks

```bash
# Format code
black psiqit/

# Run linter
flake8 psiqit/

# Run tests
pytest tests/

# Build docs locally
mkdocs build
```

### Step 4: Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

**Commit Message Format:**

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `style:` | Code style (formatting, etc.) |
| `refactor:` | Code refactoring |
| `test:` | Adding or updating tests |
| `chore:` | Maintenance tasks |

### Step 5: Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

### Step 6: PR Review

- Wait for CI checks to pass
- Address review comments
- Keep PR focused on a single topic

---

## Reporting Issues

### Bug Reports

Include the following information:

```markdown
**Description:**
Clear description of the bug

**Steps to Reproduce:**
1. Do this
2. Do that
3. See error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10]
- PSIQIT version: [e.g., 1.0.0]

**Code Example:**
```python
# Minimal code that reproduces the issue
```
```

### Issue Labels

| Label | Purpose |
|-------|---------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Docs improvements |
| `good-first-issue` | Good for newcomers |
| `help-wanted` | Extra attention needed |
| `question` | Further information requested |

---

## Feature Requests

We welcome feature suggestions! Please provide:

1. **Clear description** of the feature
2. **Use case** - why is this needed?
3. **Proposed API** - how would it look?
4. **Example** - code example of usage

---

## Community

### Communication Channels

| Channel | Purpose |
|---------|---------|
| [GitHub Issues](https://github.com/psiqit/psiqit/issues) | Bug reports, feature requests |
| [GitHub Discussions](https://github.com/psiqit/psiqit/discussions) | Questions, ideas |
| [Email](mailto:psiqitofficial@protonmail.com) | Direct contact |

### Maintainers

| Name | Role | Area |
|------|------|------|
| Mahdi Azadmarzabadi | Lead Developer | Core architecture |
| PSIQIT Contributors | Maintainers | Various modules |

---

## Recognition

Contributors will be recognized in:

- **GitHub Contributors** page
- **AUTHORS.md** file
- **Release notes** for major versions
- **Documentation** acknowledgments

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have any questions, please:

1. Check existing [Issues](https://github.com/psiqit/psiqit/issues)
2. Start a [Discussion](https://github.com/psiqit/psiqit/discussions)
3. Email the maintainers

Thank you for contributing to PSIQIT! 🚀
