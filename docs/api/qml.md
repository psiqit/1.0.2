# Quantum Machine Learning API

## Module: `psiqit.qml`

This module provides quantum machine learning algorithms and models that leverage quantum computing for enhanced machine learning capabilities.

**Available Models:**
- **Quantum Kernel** - Feature maps for kernel methods
- **Quantum SVM (QSVM)** - Support vector machines with quantum kernels
- **Quantum Neural Network (QNN)** - Variational quantum circuits for classification
- **Variational Quantum Circuit (VQC)** - General-purpose variational circuits
- **Quantum GAN (QGAN)** - Generative adversarial networks with quantum generators

---

## Quantum Kernel (`quantum_kernel.py`)

**Quantum kernel for machine learning** - Maps classical data to quantum states and computes kernel values as the squared overlap: k(x, x') = |⟨φ(x)|φ(x')⟩|².

### Classes

| Class | Description |
|-------|-------------|
| `KernelResult` | Result container for kernel computation |
| `QuantumKernel` | Quantum kernel with ZZ or XX feature maps |
| `QuantumKernelEstimator` | SWAP test based kernel estimator |

### Feature Maps

| Type | Description | Gates |
|------|-------------|-------|
| `'zz'` | ZZ feature map | RZ + CZ |
| `'xx'` | XX feature map | RX + CNOT |

### Methods

| Method | Description |
|--------|-------------|
| `evaluate(x1, x2, shots)` | Compute kernel value between two data points |
| `kernel_matrix(X, shots)` | Compute full kernel matrix for a dataset |
| `kernel_alignment(X, y, shots)` | Compute kernel alignment with labels |

### Example 1: Basic Kernel Evaluation

```python
from psiqit.qml.quantum_kernel import QuantumKernel

# Create quantum kernel with 2 qubits
kernel = QuantumKernel(n_qubits=2, feature_map='zz', n_layers=2)

# Two data points
x1 = [0.2, 0.5]
x2 = [0.7, 0.3]

# Compute kernel value
k = kernel.evaluate(x1, x2)
print(f"k(x1, x2) = {k:.6f}")
Example 2: Kernel Matrix
python
from psiqit.qml.quantum_kernel import QuantumKernel

# Dataset (XOR pattern)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

kernel = QuantumKernel(n_qubits=2, feature_map='zz')
K = kernel.kernel_matrix(X)

print("Kernel Matrix:")
for row in K:
    print([f"{x:.4f}" for x in row])
Example 3: Kernel Alignment
python
from psiqit.qml.quantum_kernel import QuantumKernel

# Dataset with labels (XOR)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

kernel = QuantumKernel(n_qubits=2)
alignment = kernel.kernel_alignment(X, y)
print(f"Kernel alignment: {alignment:.4f}")  # Higher is better for classification
Quantum SVM (qsvm.py)
Quantum Support Vector Machine - Uses quantum kernels for feature mapping. Supports linear, RBF, and quantum kernels.

Classes
Class	Description
SVMResult	Training result with support vectors, alphas, bias, accuracy
QSVM	Quantum Support Vector Machine
QuantumKernelSVM	Alias for QSVM with quantum kernel
Methods
Method	Description
fit(X, y, max_iter)	Train the SVM model
predict(X)	Predict labels for test data
decision_function(X)	Compute decision function values
score(X, y)	Calculate accuracy on test data
Example 1: Quantum Kernel SVM
python
from psiqit.qml.qsvm import QSVM

# XOR dataset
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]

# Create QSVM with quantum kernel
qsvm = QSVM(n_qubits=2, kernel_type='quantum', C=1.0)

# Train
result = qsvm.fit(X_train, y_train)
print(f"Training accuracy: {result.accuracy:.2%}")
print(f"Number of support vectors: {result.n_support}")
Example 2: Linear and RBF Kernels
python
from psiqit.qml.qsvm import QSVM

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
y = [0, 0, 1, 1]

# Linear kernel
svm_linear = QSVM(kernel_type='linear')
result_linear = svm_linear.fit(X, y)
print(f"Linear SVM accuracy: {result_linear.accuracy:.2%}")

# RBF kernel
svm_rbf = QSVM(kernel_type='rbf')
result_rbf = svm_rbf.fit(X, y)
print(f"RBF SVM accuracy: {result_rbf.accuracy:.2%}")
Example 3: Prediction and Scoring
python
from psiqit.qml.qsvm import QSVM

# Train
X_train = [[0, 0], [1, 1], [0, 1], [1, 0]]
y_train = [0, 0, 1, 1]

qsvm = QSVM(n_qubits=2, kernel_type='quantum')
qsvm.fit(X_train, y_train)

# Test
X_test = [[0.2, 0.3], [0.8, 0.7]]
predictions = qsvm.predict(X_test)
print(f"Predictions: {predictions}")

# Decision function values
decisions = qsvm.decision_function(X_test)
print(f"Decision values: {decisions}")
Quantum Neural Network (qnn.py)
Quantum Neural Network - Variational quantum circuit for machine learning tasks with multiple layers of parameterized rotations and entangling gates.

Classes
Class	Description
QNNResult	Training result with loss, accuracy, parameters
QNNLayer	Single layer of QNN (rotations + entanglement)
QNN	Quantum Neural Network
QuantumClassifier	QNN for classification tasks
VariationalQuantumCircuit	Alias for VQC
Methods
Method	Description
forward(input_state)	Forward pass through the network
predict(X, as_probability)	Make predictions on classical data
train(X, y, epochs, learning_rate)	Train the network
accuracy(X, y)	Calculate accuracy
loss(X, y)	Calculate MSE loss
Example 1: Create and Train QNN
python
from psiqit.qml.qnn import QNN

# XOR dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# Create QNN
qnn = QNN(n_qubits=2, n_layers=3, entangler='cnot')

# Train
result = qnn.train(X, y, epochs=100, learning_rate=0.1)
print(f"Final loss: {result.final_loss:.6f}")
print(f"Accuracy: {result.accuracy:.2%}")
print(f"Predictions: {result.predictions}")
Example 2: QNN Layer Configuration
python
from psiqit.qml.qnn import QNNLayer, QNN

# Create a single layer
layer = QNNLayer(n_qubits=4, entangler='cz')
print(f"Parameters per layer: {layer.n_params}")

# Create QNN with custom layers
qnn = QNN(n_qubits=4, n_layers=3, entangler='cnot', measurement='z')
print(f"Total parameters: {qnn.n_params}")
Example 3: Forward Pass
python
from psiqit.qml.qnn import QNN
from psiqit.quantum.state import zero

qnn = QNN(n_qubits=2, n_layers=2)

# Forward pass with default |0...0⟩ state
output_state = qnn.forward()
print(f"Output state: {output_state}")

# Measure expectation value
exp_val = qnn.measure_expectation()
print(f"Expectation value: {exp_val:.4f}")
Variational Quantum Circuit (vqc.py)
Variational Quantum Circuit - Parameterized quantum circuit for optimization and machine learning. Can be used for VQE, QAOA, and other variational algorithms.

Classes
Class	Description
VQCResult	Optimization result with optimal parameters and cost
VariationalLayer	Single variational layer
VQC	Variational Quantum Circuit
HamiltonianVQE	VQE for finding ground state energies
Methods
Method	Description
build_circuit(params)	Build VQC circuit
evaluate_cost(params, input_state)	Evaluate cost function
gradient(input_state)	Compute parameter gradients
optimize(cost_function, n_iterations)	Optimize parameters
run(n_iterations, learning_rate)	Run VQE optimization
Example 1: Basic VQC Optimization
python
from psiqit.qml.vqc import VQC

# Create VQC
vqc = VQC(n_qubits=2, n_layers=3)

# Define custom cost function (minimize expectation)
def cost(params):
    return vqc.evaluate_cost(params)

# Optimize
result = vqc.optimize(cost, n_iterations=100, learning_rate=0.1)
print(f"Optimal cost: {result.optimal_cost:.6f}")
print(f"Iterations: {result.n_iterations}")
Example 2: Hamiltonian VQE
python
from psiqit.qml.vqc import HamiltonianVQE

# Hamiltonian for H₂ molecule (simplified)
hamiltonian = {'Z0': 1.0, 'Z1': 1.0, 'Z0Z1': 0.5}

# Create VQE
vqe = HamiltonianVQE(n_qubits=2, hamiltonian=hamiltonian, n_layers=2)

# Run optimization
result = vqe.run(n_iterations=100, learning_rate=0.1)
print(f"Ground state energy: {result.optimal_cost:.6f}")
Example 3: Custom Cost Function
python
from psiqit.qml.vqc import VQC

vqc = VQC(n_qubits=2, n_layers=2)

# Custom cost: minimize |⟨Z⟩|
def custom_cost(params):
    vqc.set_params(params)
    return abs(vqc.measure())

result = vqc.optimize(custom_cost, n_iterations=50)
print(f"Optimal value: {result.optimal_cost:.6f}")
Quantum GAN (qgan.py)
Quantum Generative Adversarial Network - Trains a quantum generator to produce states that resemble target states, while a discriminator (quantum or classical) tries to distinguish real from generated states.

Classes
Class	Description
QGANResult	Training result with losses and fidelity history
QuantumGenerator	Quantum state generator (variational circuit)
QuantumDiscriminator	Quantum discriminator
ClassicalDiscriminator	Classical neural network discriminator
QGAN	Quantum Generative Adversarial Network
Methods
Method	Description
train(target_states, epochs, batch_size, learning_rate)	Train the QGAN
generate(n_samples, latent_vectors)	Generate new quantum states
Example 1: Train QGAN on Bell States
python
from psiqit.qml.qgan import QGAN
from psiqit.quantum.state import bell_phi_plus

# Create target states (Bell states)
target_states = [bell_phi_plus() for _ in range(100)]

# Initialize QGAN
qgan = QGAN(n_qubits=2, n_latent=2, generator_layers=2,
            discriminator_type='quantum')

# Train
result = qgan.train(target_states, epochs=50, learning_rate=0.01)
print(f"Final fidelity: {result.fidelity_history[-1]:.4f}")
print(f"Generator loss: {result.generator_loss[-1]:.4f}")
print(f"Discriminator loss: {result.discriminator_loss[-1]:.4f}")
Example 2: Generate States After Training
python
from psiqit.qml.qgan import QGAN

# Train first (see example above)
qgan = QGAN(n_qubits=2, n_latent=2)
# ... training code ...

# Generate new states
generated_states = qgan.generate(n_samples=5)

for i, state in enumerate(generated_states):
    print(f"State {i+1}: {state}")
Example 3: Classical Discriminator
python
from psiqit.qml.qgan import QGAN

# Use classical neural network as discriminator
qgan = QGAN(n_qubits=2, n_latent=2,
            discriminator_type='classical')

# Train (classical discriminator is often faster)
target_states = [bell_phi_plus() for _ in range(100)]
result = qgan.train(target_states, epochs=30)
Complete Example: QSVM vs Classical SVM
python
from psiqit.qml.qsvm import QSVM
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("=" * 60)
print("QSVM vs Classical SVM Comparison")
print("=" * 60)

# Classical SVM
print("\n1. Classical SVM (RBF kernel):")
svm_classical = SVC(kernel='rbf', C=1.0)
svm_classical.fit(X_train, y_train)
acc_classical = svm_classical.score(X_test, y_test)
print(f"   Accuracy: {acc_classical:.2%}")

# Quantum SVM
print("\n2. Quantum SVM (ZZ feature map):")
qsvm = QSVM(n_qubits=2, kernel_type='quantum', feature_map='zz')
result = qsvm.fit(X_train, y_train)
acc_qsvm = qsvm.score(X_test, y_test)
print(f"   Training accuracy: {result.accuracy:.2%}")
print(f"   Test accuracy: {acc_qsvm:.2%}")
print(f"   Support vectors: {result.n_support}")

# Compare
print("\n" + "-" * 40)
print(f"Classical SVM: {acc_classical:.2%}")
print(f"Quantum SVM:   {acc_qsvm:.2%}")
Complete Example: QNN for XOR Classification
python
from psiqit.qml.qnn import QNN
import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = [0, 1, 1, 0]

print("=" * 50)
print("QNN for XOR Classification")
print("=" * 50)

# Create QNN
qnn = QNN(n_qubits=2, n_layers=3, entangler='cnot')

print(f"\nQNN Architecture:")
print(f"  Qubits: {qnn.n_qubits}")
print(f"  Layers: {qnn.n_layers}")
print(f"  Parameters: {qnn.n_params}")

# Train
print("\nTraining...")
result = qnn.train(X, y, epochs=100, learning_rate=0.1, verbose=True)

print(f"\nResults:")
print(f"  Final loss: {result.final_loss:.6f}")
print(f"  Accuracy: {result.accuracy:.2%}")
print(f"  Predictions: {result.predictions}")

# Test individual predictions
print("\nIndividual predictions:")
for i, x in enumerate(X):
    pred = qnn.predict([x])[0]
    print(f"  {x} -> {pred:.3f} (expected: {y[i]})")
Module Contents
python
__all__ = [
    # Quantum Kernel
    'KernelResult', 'QuantumKernel', 'QuantumKernelEstimator',
    # QSVM
    'SVMResult', 'QSVM', 'QuantumKernelSVM',
    # QNN
    'QNNResult', 'QNNLayer', 'QNN', 'QuantumClassifier', 'VariationalQuantumCircuit',
    # VQC
    'VQCResult', 'VariationalLayer', 'VQC', 'HamiltonianVQE',
    # QGAN
    'QGANResult', 'QuantumGenerator', 'QuantumDiscriminator', 
    'ClassicalDiscriminator', 'QGAN',
]
## Summary Table

| Model | Type | Use Case | Quantum Advantage |
|-------|------|----------|-------------------|
| Quantum Kernel | Feature map | Kernel methods | High-dimensional feature space |
| QSVM | Classification | Supervised learning | Quantum feature maps |
| QNN | Classification/Regression | General ML | Variational circuits |
| VQC | Optimization | VQE, QAOA | Parameterized circuits |
| QGAN | Generation | Unsupervised learning | Quantum state generation |

## References

| Model | Reference |
|-------|-----------|
| Quantum Kernel | V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," Nature, 2019 |
| QSVM | Rebentrost et al., "Quantum support vector machine for big data classification," Phys. Rev. Lett., 2014 |
| QNN | M. Schuld and F. Petruccione, "Machine learning with quantum computers," Springer, 2021 |
| VQC | M. Benedetti et al., "Parameterized quantum circuits as machine learning models," Quantum Sci. Technol., 2019 |
| QGAN | P. L. Dallaire-Demers and N. Killoran, "Quantum generative adversarial networks," Phys. Rev. A, 2018 |
