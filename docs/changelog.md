
# Changelog

All notable changes to PSIQIT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.2] - 2025-06-12 (22 Khordad 1404)

### Added
- Quantum error correction with surface code (experimental)
- GPU acceleration for large-scale simulations
- Integration with IBM Qiskit backend
- Quantum chemistry module with molecular Hamiltonians

### Changed
- Improved performance of state vector simulation
- Updated documentation structure

### Fixed
- Bug in partial trace for multi-qubit systems
- Memory leak in Wigner function computation

### Deprecated
- Old `QuantumCircuit.draw()` method (use `draw_circuit()` instead)

---

## [1.0.1] - 2025-06-07 (17 Khordad 1404)

### Added
- New visualization: Husimi Q-function plots
- `QuantumKernelEstimator` class for swap test implementation
- `VariationalQuantumDeflation` for excited state VQE
- Support for Python 3.12

### Changed
- Improved error messages in circuit validation
- Optimized matrix exponential for large Hamiltonians
- Updated documentation with more examples

### Fixed
- Fix negativity calculation for mixed states
- Fix memory usage in Wigner function for large Fock spaces
- Correct Bloch sphere coordinates for states with global phase

### Documentation
- Added tutorial for quantum kernel methods
- Added example for BB84 with eavesdropping detection
- Updated API reference with missing functions

---

## [1.0.0] - 2025-06-02 (12 Khordad 1404)

### Added
- Initial stable release of PSIQIT
- Complete quantum circuit framework with all standard gates
- Quantum algorithms: Grover, Shor, Deutsch-Jozsa, Bernstein-Vazirani, Simon, QFT, QPE
- Variational methods: VQE, QAOA, SSVQE, ADAPT-VQE, VQD
- Open quantum systems: Lindblad master equation, Monte Carlo trajectories
- Quantum information: entropy measures, entanglement quantification
- Visualization: Bloch sphere, Wigner function, circuit drawing
- Quantum machine learning: QNN, QSVM, QGAN, quantum kernels
- Error correction: Bit-flip, Phase-flip, Shor, Steane codes
- Quantum communication: BB84, teleportation, superdense coding
- Comprehensive documentation with API reference and examples

---

## Versioning Scheme

PSIQIT follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): Backward-compatible new features
- **PATCH** version (0.0.X): Backward-compatible bug fixes

---

## Upgrade Guide

### From 0.9.x to 1.0.0

Most APIs are backward compatible. Major changes:

| Old | New | Notes |
|-----|-----|-------|
| `circuit.draw()` | `draw_circuit(circuit)` | Moved to visualization module |
| `State.measure()` | `measure(state)` | Now returns dictionary |
| `Operator.dagger()` | `operator.dagger()` | Same, but more efficient |

---

## Credits

Special thanks to all contributors who made this release possible:

- Mahdi Azadmarzabadi (Lead Developer)
- PSIQIT Contributors (See GitHub)

---

## Links

- [GitHub Releases](https://github.com/psiqit/psiqit/releases)
- [Issue Tracker](https://github.com/psiqit/psiqit/issues)
- [Documentation](https://psiqit.github.io)
- [PyPI](https://pypi.org/project/psiqit/)
