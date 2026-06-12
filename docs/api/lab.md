```markdown
# Virtual Lab API

## Module: `psiqit.lab`

This module provides a virtual quantum laboratory environment for designing, running, and analyzing quantum experiments. It allows users to create experiments, add quantum gates, execute simulations, and collect results without needing deep knowledge of quantum circuit implementation details.

**Key Features:**
- Create and manage quantum experiments
- Add gates to circuits interactively
- Run simulations and collect results
- Access predefined experiments (Bell state, GHZ state)
- Visualize results and state vectors

---

## ExperimentStatus

**Enum representing the status of an experiment.**

### Values

| Value | Description |
|-------|-------------|
| `DRAFT` | Experiment is being designed (gates can be added/modified) |
| `RUNNING` | Experiment is currently being executed |
| `COMPLETED` | Experiment has finished successfully |
| `FAILED` | Experiment failed due to an error |

### Example

```python
from psiqit.lab import ExperimentStatus

print(ExperimentStatus.DRAFT)       # ExperimentStatus.DRAFT
print(ExperimentStatus.DRAFT.value) # "draft"
```

---

## ExperimentResult

Result container for a quantum experiment.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the experiment |
| `timestamp` | `str` | ISO format timestamp |
| `n_qubits` | `int` | Number of qubits |
| `shots` | `int` | Number of measurements |
| `counts` | `Dict[str, int]` | Measurement counts (bitstring → count) |
| `probabilities` | `List[float]` | Probabilities for each basis state |
| `final_state` | `List[complex]` | Final state vector |
| `success_rate` | `float` | Success rate (max_count/shots) |
| `execution_time` | `float` | Execution time in seconds |
| `status` | `str` | Status string ('completed' or 'failed') |

### Methods

| Method | Description |
|--------|-------------|
| `summary()` | Return human-readable summary |

### Example

```python
from psiqit.lab import ExperimentResult

result = ExperimentResult(
    name="Bell State",
    timestamp="2024-01-01T12:00:00",
    n_qubits=2,
    shots=1024,
    counts={"00": 512, "11": 512},
    probabilities=[0.5, 0, 0, 0.5],
    final_state=[],
    success_rate=0.5,
    execution_time=0.123,
    status="completed"
)

print(result.summary())
```

### Output

```text
Experiment: Bell State
Time: 2024-01-01T12:00:00
Qubits: 2, Shots: 1024
Success rate: 50.00%
Status: completed

Measurement counts:
  |00⟩: 512 (50.00%)
  |11⟩: 512 (50.00%)
```

---

## Experiment

Container for a quantum experiment definition.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Name of the experiment |
| `n_qubits` | `int` | Number of qubits |
| `gates` | `List[Dict]` | List of gate dictionaries |
| `shots` | `int` | Number of measurement shots (default: 1024) |
| `status` | `ExperimentStatus` | Current status (default: DRAFT) |
| `created_at` | `str` | Creation timestamp |
| `result` | `Optional[ExperimentResult]` | Result after execution |

### Methods

| Method | Description |
|--------|-------------|
| `add_gate(gate_name, *qubits, params)` | Add a quantum gate to the experiment |
| `clear_gates()` | Remove all gates from the experiment |

### Supported Gates

| Gate Name | Description | Example |
|-----------|-------------|---------|
| `h` | Hadamard | `add_gate("h", 0)` |
| `x` | Pauli X (NOT) | `add_gate("x", 0)` |
| `y` | Pauli Y | `add_gate("y", 0)` |
| `z` | Pauli Z | `add_gate("z", 0)` |
| `cx` or `cnot` | CNOT | `add_gate("cx", 0, 1)` |
| `cz` | Controlled-Z | `add_gate("cz", 0, 1)` |
| `swap` | SWAP | `add_gate("swap", 0, 1)` |
| `rx` | Rotation around X | `add_gate("rx", 0, params={"theta": 3.14})` |
| `ry` | Rotation around Y | `add_gate("ry", 0, params={"theta": 3.14})` |
| `rz` | Rotation around Z | `add_gate("rz", 0, params={"theta": 3.14})` |

### Example 1: Create and Configure Experiment

```python
from psiqit.lab import Experiment

# Create experiment
exp = Experiment(name="My Bell State", n_qubits=2)

# Add gates
exp.add_gate("h", 0)
exp.add_gate("cx", 0, 1)

print(f"Experiment: {exp.name}")
print(f"Number of gates: {len(exp.gates)}")
print(f"Status: {exp.status.value}")  # draft
```

### Example 2: Clear Gates

```python
exp = Experiment(name="Test", n_qubits=2)
exp.add_gate("h", 0)
print(f"Gates before clear: {len(exp.gates)}")  # 1

exp.clear_gates()
print(f"Gates after clear: {len(exp.gates)}")   # 0
```

---

## QuantumLab

Virtual quantum laboratory for running experiments.

### Methods

| Method | Description |
|--------|-------------|
| `create_experiment(name, n_qubits)` | Create a new experiment |
| `run_experiment(experiment, shots)` | Run an experiment |
| `get_state_vector(experiment)` | Get final state vector without measurement |
| `get_bloch_coordinates(experiment, qubit)` | Get Bloch coordinates for a qubit |
| `list_experiments()` | List all experiments in the lab |

### Example 1: Create and Run Experiment

```python
from psiqit.lab import QuantumLab

# Create virtual lab
lab = QuantumLab()

# Create experiment
exp = lab.create_experiment("Bell State", n_qubits=2)
exp.add_gate("h", 0)
exp.add_gate("cx", 0, 1)

# Run experiment
result = lab.run_experiment(exp, shots=1024)

print(f"Experiment: {result.name}")
print(f"Success rate: {result.success_rate:.2%}")
print(f"Counts: {result.counts}")
# Counts: {'00': 512, '11': 512}
```

### Example 2: Get State Vector

```python
from psiqit.lab import QuantumLab

lab = QuantumLab()
exp = lab.create_experiment("Bell State", n_qubits=2)
exp.add_gate("h", 0)
exp.add_gate("cx", 0, 1)

state = lab.get_state_vector(exp)
print(f"State vector: {state}")
# 0.707|00⟩ + 0.707|11⟩
```

### Example 3: Get Bloch Coordinates

```python
from psiqit.lab import QuantumLab

lab = QuantumLab()

# Single qubit experiment
exp = lab.create_experiment("X Gate", n_qubits=1)
exp.add_gate("x", 0)

x, y, z = lab.get_bloch_coordinates(exp, qubit=0)
print(f"Bloch coordinates: ({x:.2f}, {y:.2f}, {z:.2f})")
# (0.00, 0.00, -1.00)  # |1⟩ state
```

### Example 4: List Experiments

```python
from psiqit.lab import QuantumLab

lab = QuantumLab()

# Create multiple experiments
lab.create_experiment("Exp1", 2)
lab.create_experiment("Exp2", 3)
lab.create_experiment("Exp3", 4)

# List all experiments
experiments = lab.list_experiments()
for exp in experiments:
    print(f"Name: {exp['name']}, Qubits: {exp['n_qubits']}, Gates: {exp['n_gates']}")
```

**Output:**

```text
Name: Exp1, Qubits: 2, Gates: 0
Name: Exp2, Qubits: 3, Gates: 0
Name: Exp3, Qubits: 4, Gates: 0
```

### Example 5: Complete Workflow with Error Handling

```python
from psiqit.lab import QuantumLab

lab = QuantumLab()

try:
    # Create experiment
    exp = lab.create_experiment("GHZ State", n_qubits=3)
    exp.add_gate("h", 0)
    exp.add_gate("cx", 0, 1)
    exp.add_gate("cx", 1, 2)
    
    # Run with custom shots
    result = lab.run_experiment(exp, shots=2048)
    
    print(f"Status: {result.status}")
    print(f"Execution time: {result.execution_time:.4f}s")
    print(f"Most common outcome: {max(result.counts, key=result.counts.get)}")
    
except RuntimeError as e:
    print(f"Experiment failed: {e}")
```

---

## PredefinedExperiments

Collection of predefined quantum experiments.

### Static Methods

| Method | Description |
|--------|-------------|
| `bell_state()` | Create Bell state experiment: (|00⟩ + |11⟩)/√2 |
| `ghz_state(n)` | Create GHZ state experiment: (|00...0⟩ + |11...1⟩)/√2 |

### Example 1: Bell State Experiment

```python
from psiqit.lab import QuantumLab, PredefinedExperiments

lab = QuantumLab()
bell_exp = PredefinedExperiments.bell_state()

result = lab.run_experiment(bell_exp)
print(result.summary())
```

**Output:**

```text
Experiment: Bell State
Time: 2024-01-01T12:00:00
Qubits: 2, Shots: 1024
Success rate: 50.00%
Status: completed

Measurement counts:
  |00⟩: 512 (50.00%)
  |11⟩: 512 (50.00%)
```

### Example 2: GHZ State Experiment

```python
from psiqit.lab import QuantumLab, PredefinedExperiments

lab = QuantumLab()

# 3-qubit GHZ state
ghz_exp = PredefinedExperiments.ghz_state(n=3)
result = lab.run_experiment(ghz_exp)

print(f"GHZ state results:")
for outcome, count in result.counts.items():
    print(f"  |{outcome}⟩: {count} ({count/result.shots:.1%})")
```

**Output:**

```text
GHZ state results:
  |000⟩: 512 (50.0%)
  |111⟩: 512 (50.0%)
```

---

## Complete Example: Virtual Lab Tutorial

```python
from psiqit.lab import QuantumLab, PredefinedExperiments
from psiqit.visualization.bloch import bloch_sphere

# Initialize lab
lab = QuantumLab()

print("=" * 50)
print("PSIQIT Virtual Lab Tutorial")
print("=" * 50)

# 1. Create custom experiment
print("\n1. Creating custom Bell state experiment...")
exp = lab.create_experiment("My Bell State", n_qubits=2)
exp.add_gate("h", 0)
exp.add_gate("cx", 0, 1)

# 2. Run experiment
print("\n2. Running experiment...")
result = lab.run_experiment(exp, shots=1024)

print(f"\nResults:")
print(f"  Name: {result.name}")
print(f"  Success rate: {result.success_rate:.2%}")
print(f"  Execution time: {result.execution_time:.4f}s")

print(f"\n  Measurement counts:")
for outcome, count in sorted(result.counts.items()):
    prob = count / result.shots
    print(f"    |{outcome}⟩: {count} ({prob:.1%})")

# 3. Get state vector
print("\n3. State vector (without measurement):")
state = lab.get_state_vector(exp)
print(f"  {state}")

# 4. Use predefined experiment
print("\n4. Using predefined GHZ state...")
ghz = PredefinedExperiments.ghz_state(n=3)
result = lab.run_experiment(ghz)

print(f"\n  GHZ State Results:")
for outcome, count in sorted(result.counts.items()):
    prob = count / result.shots
    print(f"    |{outcome}⟩: {count} ({prob:.1%})")

# 5. List all experiments
print("\n5. All experiments in lab:")
for exp_info in lab.list_experiments():
    print(f"  - {exp_info['name']} ({exp_info['n_qubits']} qubits, {exp_info['n_gates']} gates)")
```

### Expected Output

```text
==================================================
PSIQIT Virtual Lab Tutorial
==================================================

1. Creating custom Bell state experiment...

2. Running experiment...

Results:
  Name: My Bell State
  Success rate: 50.00%
  Execution time: 0.0123s

  Measurement counts:
    |00⟩: 512 (50.0%)
    |11⟩: 512 (50.0%)

3. State vector (without measurement):
  0.707|00⟩ + 0.707|11⟩

4. Using predefined GHZ state...

  GHZ State Results:
    |000⟩: 512 (50.0%)
    |111⟩: 512 (50.0%)

5. All experiments in lab:
  - My Bell State (2 qubits, 2 gates)
  - Bell State (2 qubits, 2 gates)
  - GHZ State (3 qubits) (3 qubits, 3 gates)
```

---

## Module Contents

```python
__all__ = [
    'ExperimentStatus',
    'ExperimentResult',
    'Experiment',
    'QuantumLab',
    'PredefinedExperiments',
]
```

---

## References

| Source | Description |
|--------|-------------|
| IBM Quantum Experience | Inspiration for virtual lab interface |
| Quirk | Quantum circuit simulator |
| Qiskit Tutorials | Educational quantum computing examples |


