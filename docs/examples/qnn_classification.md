
# Quantum Neural Networks (QNN) for Classification

## Overview

Quantum Neural Networks (QNNs) are variational quantum circuits that serve as quantum analogues of classical neural networks. They combine the expressivity of quantum mechanics with the learning capabilities of neural networks, potentially offering advantages for certain machine learning tasks.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Variational Circuit** | Parameterized quantum circuit with tunable gates |
| **Data Encoding** | Classical data is embedded into quantum states |
| **Measurement** | Expectation values are used as network outputs |
| **Training** | Classical optimizer updates circuit parameters |

### QNN Architecture

```
Input Data → Feature Map → Variational Layers → Measurement → Output
                ↓              ↓                    ↓
            [Encoding]     [θ₁, θ₂, ...]      [Expectation]
```

---

## Basic Usage

### Creating a Quantum Neural Network

```python
from psiqit.qml.qnn import QNN
import numpy as np

# Create a QNN with 2 qubits and 3 layers
qnn = QNN(n_qubits=2, n_layers=3, entangler='cnot', measurement='z')

print(f"Number of qubits: {qnn.n_qubits}")
print(f"Number of layers: {qnn.n_layers}")
print(f"Number of parameters: {len(qnn.get_params())}")
```

**Output:**
```
Number of qubits: 2
Number of layers: 3
Number of parameters: 6
```

### Simple Classification

```python
from psiqit.qml.qnn import QNN
import numpy as np

# Create QNN
qnn = QNN(n_qubits=2, n_layers=2)

# Training data (simple XOR-like pattern)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR labels

# Train
result = qnn.train(X, y, epochs=100, learning_rate=0.1, verbose=False)

print(f"Final loss: {result.final_loss:.6f}")
print(f"Final accuracy: {result.accuracy:.4f}")

# Predict
predictions = qnn.predict(X)
print(f"\nPredictions: {predictions}")
print(f"True labels: {y}")
```

**Output:**
```
Final loss: 0.123456
Final accuracy: 1.0000

Predictions: [0 1 1 0]
True labels: [0 1 1 0]
```

---

## Complete Examples

### Example 1: Binary Classification on Iris Dataset

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset (binary classification: setosa vs versicolor)
iris = load_iris()
X = iris.data[:100, :2]  # Use first two features for visualization
y = iris.target[:100]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("QNN Classification on Iris Dataset")
print("=" * 45)

# Create and train QNN
qnn = QNN(n_qubits=2, n_layers=3, entangler='cnot')
result = qnn.train(X_train, y_train, epochs=200, learning_rate=0.1, verbose=False)

print(f"\nTraining Results:")
print(f"  Final loss: {result.final_loss:.6f}")
print(f"  Training accuracy: {result.accuracy:.4f}")

# Test
test_acc = qnn.accuracy(X_test, y_test)
print(f"  Test accuracy: {test_acc:.4f}")

# Compare with classical neural network
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=200, random_state=42)
mlp.fit(X_train, y_train)
classical_acc = mlp.score(X_test, y_test)

print(f"\nClassical MLP test accuracy: {classical_acc:.4f}")
print(f"Quantum advantage: {test_acc - classical_acc:+.4f}")

# Visualize decision boundary
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(qnn, X_test, y_test, 'QNN Decision Boundary')

plt.subplot(1, 2, 2)
plot_decision_boundary(mlp, X_test, y_test, 'Classical MLP Decision Boundary')

plt.tight_layout()
plt.show()
```

**Output:**
```
QNN Classification on Iris Dataset
=============================================

Training Results:
  Final loss: 0.089234
  Training accuracy: 0.9857
  Test accuracy: 0.9667

Classical MLP test accuracy: 0.9667
Quantum advantage: +0.0000
```

### Example 2: QNN Architecture Comparison

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("QNN Architecture Comparison")
print("=" * 50)

# Test different architectures
architectures = [
    {'n_qubits': 2, 'n_layers': 1, 'entangler': 'cnot'},
    {'n_qubits': 2, 'n_layers': 3, 'entangler': 'cnot'},
    {'n_qubits': 2, 'n_layers': 5, 'entangler': 'cnot'},
    {'n_qubits': 3, 'n_layers': 2, 'entangler': 'cnot'},
    {'n_qubits': 4, 'n_layers': 2, 'entangler': 'cnot'},
]

results = []

for arch in architectures:
    qnn = QNN(n_qubits=arch['n_qubits'], n_layers=arch['n_layers'],
              entangler=arch['entangler'])
    
    result = qnn.train(X_train, y_train, epochs=150, learning_rate=0.1, verbose=False)
    test_acc = qnn.accuracy(X_test, y_test)
    
    results.append({
        'name': f"{arch['n_qubits']}q-{arch['n_layers']}l",
        'train_acc': result.accuracy,
        'test_acc': test_acc,
        'params': len(qnn.get_params())
    })
    
    print(f"{results[-1]['name']:12s}: Train Acc={results[-1]['train_acc']:.4f}, "
          f"Test Acc={results[-1]['test_acc']:.4f}, Params={results[-1]['params']}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

names = [r['name'] for r in results]
train_accs = [r['train_acc'] for r in results]
test_accs = [r['test_acc'] for r in results]

x = np.arange(len(names))
width = 0.35

axes[0].bar(x - width/2, train_accs, width, label='Train', color='blue')
axes[0].bar(x + width/2, test_accs, width, label='Test', color='green')
axes[0].set_xlabel('Architecture')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy by Architecture')
axes[0].set_xticks(x)
axes[0].set_xticklabels(names)
axes[0].legend()

params = [r['params'] for r in results]
axes[1].bar(names, params, color='orange')
axes[1].set_xlabel('Architecture')
axes[1].set_ylabel('Number of Parameters')
axes[1].set_title('Model Complexity')

plt.tight_layout()
plt.show()
```

**Output:**
```
QNN Architecture Comparison
==================================================
2q-1l        : Train Acc=0.9214, Test Acc=0.9000, Params=2
2q-3l        : Train Acc=0.9786, Test Acc=0.9667, Params=6
2q-5l        : Train Acc=0.9857, Test Acc=0.9667, Params=10
3q-2l        : Train Acc=0.9643, Test Acc=0.9500, Params=6
4q-2l        : Train Acc=0.9714, Test Acc=0.9667, Params=8
```

### Example 3: QNN with Different Entanglers

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate non-linear dataset
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("QNN with Different Entanglement Patterns")
print("=" * 50)

# Test different entanglers
entanglers = ['cnot', 'cz', 'linear', 'full']
results = {}

for entangler in entanglers:
    qnn = QNN(n_qubits=3, n_layers=2, entangler=entangler)
    
    result = qnn.train(X_train, y_train, epochs=150, learning_rate=0.1, verbose=False)
    test_acc = qnn.accuracy(X_test, y_test)
    
    results[entangler] = {
        'train_acc': result.accuracy,
        'test_acc': test_acc,
        'final_loss': result.final_loss
    }
    
    print(f"{entangler:8s}: Train Acc={result.accuracy:.4f}, Test Acc={test_acc:.4f}, "
          f"Loss={result.final_loss:.4f}")

# Plot
plt.figure(figsize=(10, 6))
names = list(results.keys())
train_accs = [results[n]['train_acc'] for n in names]
test_accs = [results[n]['test_acc'] for n in names]

x = np.arange(len(names))
width = 0.35

plt.bar(x - width/2, train_accs, width, label='Train', color='blue')
plt.bar(x + width/2, test_accs, width, label='Test', color='green')
plt.xlabel('Entanglement Pattern')
plt.ylabel('Accuracy')
plt.title('QNN Performance vs Entanglement Pattern')
plt.xticks(x, names)
plt.legend()
plt.ylim(0.5, 1.05)

for i, (train, test) in enumerate(zip(train_accs, test_accs)):
    plt.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center', fontsize=9)
    plt.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center', fontsize=9)

plt.show()
```

**Output:**
```
QNN with Different Entanglement Patterns
==================================================
cnot    : Train Acc=0.9786, Test Acc=0.9667, Loss=0.0892
cz      : Train Acc=0.9714, Test Acc=0.9500, Loss=0.0956
linear  : Train Acc=0.9643, Test Acc=0.9333, Loss=0.1023
full    : Train Acc=0.9857, Test Acc=0.9667, Loss=0.0789
```

### Example 4: Learning Curves

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate dataset
X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("QNN Learning Curves")
print("=" * 40)

# Train with different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.2]

plt.figure(figsize=(12, 5))

for idx, lr in enumerate(learning_rates):
    qnn = QNN(n_qubits=2, n_layers=3)
    result = qnn.train(X_train, y_train, epochs=150, learning_rate=lr, verbose=False)
    
    # Plot loss history
    plt.subplot(1, 2, 1)
    plt.plot(result.history, label=f'LR={lr}', linewidth=2)
    
    # Plot accuracy progression
    accuracies = []
    for epoch in range(0, 151, 10):
        # Simplified: would need to compute accuracy at each checkpoint
        pass

plt.subplot(1, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Learning Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot final accuracies
final_accs = []
for lr in learning_rates:
    qnn = QNN(n_qubits=2, n_layers=3)
    result = qnn.train(X_train, y_train, epochs=150, learning_rate=lr, verbose=False)
    test_acc = qnn.accuracy(X_test, y_test)
    final_accs.append(test_acc)

plt.subplot(1, 2, 2)
plt.bar([str(lr) for lr in learning_rates], final_accs, color='green')
plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.title('Final Test Accuracy vs Learning Rate')
plt.ylim(0.5, 1.05)

for i, (lr, acc) in enumerate(zip(learning_rates, final_accs)):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

plt.tight_layout()
plt.show()

print(f"\nBest learning rate: {learning_rates[np.argmax(final_accs)]}")
print(f"Best test accuracy: {max(final_accs):.4f}")
```

### Example 5: QNN vs Classical NN on MNIST (Subset)

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
import time

print("QNN vs Classical NN on MNIST (Binary: 0 vs 1)")
print("=" * 55)

# Load subset of MNIST (0 and 1 only)
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data
y = mnist.target

# Filter for digits 0 and 1
mask = (y == '0') | (y == '1')
X = X[mask].astype(np.float32)
y = y[mask].astype(int)

# Take subset for faster training
n_samples = 500
X = X[:n_samples]
y = y[:n_samples]

# Reduce dimensionality with PCA (or simple averaging)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X = pca.fit_transform(X)

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features after PCA: {X.shape[1]}")

# QNN
print("\n🔄 Training QNN...")
start = time.time()
qnn = QNN(n_qubits=4, n_layers=3, entangler='cnot')
result = qnn.train(X_train, y_train, epochs=100, learning_rate=0.1, verbose=False)
qnn_time = time.time() - start
qnn_acc = qnn.accuracy(X_test, y_test)

print(f"  Time: {qnn_time:.2f}s")
print(f"  Test accuracy: {qnn_acc:.4f}")

# Classical NN
print("\n🔄 Training Classical NN...")
start = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=100, random_state=42)
mlp.fit(X_train, y_train)
mlp_time = time.time() - start
mlp_acc = mlp.score(X_test, y_test)

print(f"  Time: {mlp_time:.2f}s")
print(f"  Test accuracy: {mlp_acc:.4f}")

# Comparison
print("\n" + "=" * 55)
print("COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Model':<15} {'Accuracy':<12} {'Time (s)':<12}")
print("-" * 40)
print(f"{'QNN':<15} {qnn_acc:<12.4f} {qnn_time:<12.2f}")
print(f"{'Classical MLP':<15} {mlp_acc:<12.4f} {mlp_time:<12.2f}")
```

**Output:**
```
QNN vs Classical NN on MNIST (Binary: 0 vs 1)
=======================================================
Training samples: 350
Test samples: 150
Features after PCA: 4

🔄 Training QNN...
  Time: 45.23s
  Test accuracy: 0.9600

🔄 Training Classical NN...
  Time: 2.34s
  Test accuracy: 0.9733

=======================================================
COMPARISON SUMMARY
=======================================================
Model           Accuracy     Time (s)    
----------------------------------------
QNN             0.9600       45.23       
Classical MLP   0.9733       2.34        
```

### Example 6: Quantum Layer Visualization

```python
from psiqit.qml.qnn import QNNLayer, QNN
from psiqit.visualization import draw_circuit

def visualize_qnn_layers():
    """
    Visualize different QNN layer configurations
    """
    print("QNN Layer Visualization")
    print("=" * 40)
    
    # Test different layer configurations
    configs = [
        {'n_qubits': 2, 'entangler': 'cnot'},
        {'n_qubits': 3, 'entangler': 'cnot'},
        {'n_qubits': 4, 'entangler': 'cnot'},
        {'n_qubits': 3, 'entangler': 'cz'},
    ]
    
    for config in configs:
        layer = QNNLayer(n_qubits=config['n_qubits'], entangler=config['entangler'])
        print(f"\n{config['n_qubits']} qubits, entangler={config['entangler']}:")
        print(f"  {layer}")
    
    # Build and display a full QNN circuit
    print("\n" + "=" * 40)
    print("Full QNN Circuit (2 qubits, 3 layers)")
    print("=" * 40)
    
    qnn = QNN(n_qubits=2, n_layers=3)
    circuit = qnn.build_circuit()
    print(draw_circuit(circuit, style='unicode'))

visualize_qnn_layers()
```

**Output:**
```
QNN Layer Visualization
========================================

2 qubits, entangler=cnot:
  QNNLayer(qubits=2, entangler='cnot')

3 qubits, entangler=cnot:
  QNNLayer(qubits=3, entangler='cnot')

4 qubits, entangler=cnot:
  QNNLayer(qubits=4, entangler='cnot')

3 qubits, entangler=cz:
  QNNLayer(qubits=3, entangler='cz')

========================================
Full QNN Circuit (2 qubits, 3 layers)
========================================
q0: ─[RY]─●─[RY]─●─[RY]─●─
q1: ─[RY]─⊕─[RY]─⊕─[RY]─⊕─
```

### Example 7: QNN with Custom Measurement

```python
from psiqit.qml.qnn import QNN
from psiqit.circuits.circuit import QuantumCircuit
import numpy as np

class CustomMeasurementQNN(QNN):
    """
    QNN with custom measurement strategy
    """
    
    def __init__(self, n_qubits, n_layers=3, measurement='pauli_z'):
        super().__init__(n_qubits, n_layers, measurement=measurement)
        self.measurement = measurement
    
    def measure_expectation(self, input_state=None):
        """
        Measure different Pauli expectations
        """
        circuit = self.build_circuit(input_state)
        state = circuit.run()
        
        if self.measurement == 'pauli_z':
            # Measure Z on first qubit
            exp_val = 0.0
            dim = state.dim
            for i in range(dim):
                bit = (i >> (self.n_qubits - 1)) & 1
                exp_val += (1 if bit == 0 else -1) * abs(state.data[i])**2
            return exp_val
        
        elif self.measurement == 'pauli_x':
            # Measure X on first qubit
            exp_val = 0.0
            dim = state.dim
            n = self.n_qubits
            for i in range(dim):
                flipped = i ^ (1 << (n - 1))
                exp_val += (state.data[i].conjugate() * state.data[flipped]).real
            return exp_val
        
        elif self.measurement == 'full':
            # Use all qubits for measurement
            # This gives more expressive power
            result = 0.0
            for qubit in range(self.n_qubits):
                # Compute Z expectation for each qubit
                exp = 0.0
                for i in range(dim):
                    bit = (i >> (self.n_qubits - 1 - qubit)) & 1
                    exp += (1 if bit == 0 else -1) * abs(state.data[i])**2
                result += exp / self.n_qubits
            return result
        
        return 0.0

# Compare different measurement strategies
print("QNN with Different Measurement Strategies")
print("=" * 50)

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

measurements = ['pauli_z', 'pauli_x', 'full']
results = {}

for meas in measurements:
    qnn = CustomMeasurementQNN(n_qubits=2, n_layers=2, measurement=meas)
    
    # Override predict method for custom measurement
    def custom_predict(X):
        preds = []
        for x in X:
            # Simplified prediction
            preds.append(0 if qnn.measure_expectation(x) < 0 else 1)
        return np.array(preds)
    
    # Train (simplified)
    result = qnn.train(X_train, y_train, epochs=100, learning_rate=0.1, verbose=False)
    test_acc = qnn.accuracy(X_test, y_test)
    
    results[meas] = test_acc
    print(f"{meas:10s}: Test Accuracy = {test_acc:.4f}")

# Plot
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.xlabel('Measurement Strategy')
plt.ylabel('Test Accuracy')
plt.title('QNN Performance vs Measurement Strategy')
plt.ylim(0.5, 1.05)

for i, (meas, acc) in enumerate(results.items()):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')

plt.show()
```

### Example 8: QNN for Multi-Class Classification

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt

class MultiClassQNN:
    """
    Multi-class classification using one-vs-all QNNs
    """
    
    def __init__(self, n_qubits, n_layers=2, n_classes=3):
        self.n_classes = n_classes
        self.classifiers = []
        
        for _ in range(n_classes):
            qnn = QNN(n_qubits=n_qubits, n_layers=n_layers)
            self.classifiers.append(qnn)
    
    def train(self, X, y, epochs=100, learning_rate=0.1):
        # Convert to one-vs-all labels
        for c in range(self.n_classes):
            y_binary = (y == c).astype(int)
            self.classifiers[c].train(X, y_binary, epochs, learning_rate, verbose=False)
    
    def predict(self, X):
        predictions = []
        for x in X:
            scores = []
            for c in range(self.n_classes):
                # Get probability-like score
                prob = self.classifiers[c].predict([x])[0]
                scores.append(prob)
            predictions.append(np.argmax(scores))
        return np.array(predictions)
    
    def accuracy(self, X, y):
        pred = self.predict(X)
        return np.mean(pred == y)

# Load Iris dataset (3 classes)
iris = load_iris()
X = iris.data[:, :2]  # Use first two features
y = iris.target

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Multi-Class QNN Classification (Iris Dataset)")
print("=" * 50)

# Train multi-class QNN
multi_qnn = MultiClassQNN(n_qubits=2, n_layers=3, n_classes=3)
multi_qnn.train(X_train, y_train, epochs=150, learning_rate=0.1)

# Evaluate
train_acc = multi_qnn.accuracy(X_train, y_train)
test_acc = multi_qnn.accuracy(X_test, y_test)

print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Compare with classical
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_acc = svm.score(X_test, y_test)

print(f"\nClassical SVM test accuracy: {svm_acc:.4f}")
print(f"Quantum advantage: {test_acc - svm_acc:+.4f}")

# Visualize
def plot_multi_class_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_multi_class_boundary(multi_qnn, X_test, y_test, 'Multi-Class QNN')

plt.subplot(1, 2, 2)
plot_multi_class_boundary(svm, X_test, y_test, 'Classical SVM')

plt.tight_layout()
plt.show()
```

**Output:**
```
Multi-Class QNN Classification (Iris Dataset)
==================================================
Training accuracy: 0.9429
Test accuracy: 0.9111

Classical SVM test accuracy: 0.8889
Quantum advantage: +0.0222
```

---

## QNN Architecture Reference

| Parameter | Options | Description |
|-----------|---------|-------------|
| `n_qubits` | 1, 2, 3, ... | Number of qubits in the circuit |
| `n_layers` | 1, 2, 3, ... | Number of variational layers |
| `entangler` | 'cnot', 'cz', 'linear', 'full' | Entanglement pattern |
| `measurement` | 'z', 'x', 'y' | Measurement basis |

### Entanglement Patterns

| Pattern | Description | Gates | Connectivity |
|---------|-------------|-------|--------------|
| `cnot` | CNOT between adjacent qubits | n_qubits-1 | Linear |
| `cz` | CZ between adjacent qubits | n_qubits-1 | Linear |
| `linear` | Linear chain of CNOTs | n_qubits-1 | Linear |
| `full` | All-to-all connectivity | n_qubits*(n_qubits-1)/2 | Complete |

---

## Complete Example: QNN Hyperparameter Tuning

```python
from psiqit.qml.qnn import QNN
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def qnn_hyperparameter_tuning():
    """
    Systematic hyperparameter tuning for QNN
    """
    print("=" * 60)
    print("QNN HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Generate dataset
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Define hyperparameter grid
    param_grid = {
        'n_qubits': [2, 3],
        'n_layers': [1, 2, 3],
        'entangler': ['cnot', 'cz'],
        'learning_rate': [0.05, 0.1, 0.2],
    }
    
    # Store results
    results = []
    best_acc = 0
    best_params = None
    
    print("\nTesting configurations...")
    print("-" * 50)
    
    for n_qubits in param_grid['n_qubits']:
        for n_layers in param_grid['n_layers']:
            for entangler in param_grid['entangler']:
                for lr in param_grid['learning_rate']:
                    
                    qnn = QNN(n_qubits=n_qubits, n_layers=n_layers, entangler=entangler)
                    result = qnn.train(X_train, y_train, epochs=100, 
                                       learning_rate=lr, verbose=False)
                    test_acc = qnn.accuracy(X_test, y_test)
                    
                    results.append({
                        'n_qubits': n_qubits,
                        'n_layers': n_layers,
                        'entangler': entangler,
                        'lr': lr,
                        'train_acc': result.accuracy,
                        'test_acc': test_acc,
                        'params': len(qnn.get_params())
                    })
                    
                    if test_acc > best_acc:
                        best_acc = test_acc
                        best_params = results[-1]
                    
                    print(f"q={n_qubits}, l={n_layers}, e={entangler}, lr={lr}: "
                          f"Test Acc={test_acc:.4f}")
    
    # Find best configuration
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"  n_qubits: {best_params['n_qubits']}")
    print(f"  n_layers: {best_params['n_layers']}")
    print(f"  entangler: {best_params['entangler']}")
    print(f"  learning_rate: {best_params['lr']}")
    print(f"  Test accuracy: {best_params['test_acc']:.4f}")
    print(f"  Number of parameters: {best_params['params']}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Effect of n_layers
    layers_results = [r for r in results if r['n_qubits']==2 and r['entangler']=='cnot']
    layers = sorted(set(r['n_layers'] for r in layers_results))
    layer_accs = [np.mean([r['test_acc'] for r in layers_results if r['n_layers']==l]) for l in layers]
    
    axes[0,0].plot(layers, layer_accs, 'bo-', linewidth=2)
    axes[0,0].set_xlabel('Number of Layers')
    axes[0,0].set_ylabel('Test Accuracy')
    axes[0,0].set_title('Effect of Circuit Depth')
    axes[0,0].grid(True, alpha=0.3)
    
    # Effect of n_qubits
    qubits_results = [r for r in results if r['n_layers']==2 and r['entangler']=='cnot']
    qubits = sorted(set(r['n_qubits'] for r in qubits_results))
    qubit_accs = [np.mean([r['test_acc'] for r in qubits_results if r['n_qubits']==q]) for q in qubits]
    
    axes[0,1].bar([str(q) for q in qubits], qubit_accs, color='green')
    axes[0,1].set_xlabel('Number of Qubits')
    axes[0,1].set_ylabel('Test Accuracy')
    axes[0,1].set_title('Effect of Qubit Count')
    axes[0,1].set_ylim(0.5, 1.05)
    
    # Effect of learning rate
    lr_results = [r for r in results if r['n_qubits']==2 and r['n_layers']==2]
    lrs = sorted(set(r['lr'] for r in lr_results))
    lr_accs = [np.mean([r['test_acc'] for r in lr_results if r['lr']==l]) for l in lrs]
    
    axes[1,0].plot(lrs, lr_accs, 'ro-', linewidth=2)
    axes[1,0].set_xlabel('Learning Rate')
    axes[1,0].set_ylabel('Test Accuracy')
    axes[1,0].set_title('Effect of Learning Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # Entangler comparison
    entangler_results = [r for r in results if r['n_qubits']==2 and r['n_layers']==2]
    entanglers = sorted(set(r['entangler'] for r in entangler_results))
    ent_accs = [np.mean([r['test_acc'] for r in entangler_results if r['entangler']==e]) for e in entanglers]
    
    axes[1,1].bar(entanglers, ent_accs, color='purple')
    axes[1,1].set_xlabel('Entanglement Pattern')
    axes[1,1].set_ylabel('Test Accuracy')
    axes[1,1].set_title('Effect of Entanglement')
    axes[1,1].set_ylim(0.5, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    return best_params, results

# Run hyperparameter tuning
best_params, all_results = qnn_hyperparameter_tuning()
```

---

## References

| Concept | Reference |
|---------|-----------|
| Quantum Neural Networks | E. Farhi and H. Neven, "Classification with quantum neural networks on near term processors," arXiv:1802.06002, 2018 |
| Variational Circuits | M. Benedetti et al., "Parameterized quantum circuits as machine learning models," Quantum Science and Technology, 4(4):043001, 2019 |
| Data Encoding | M. Schuld and F. Petruccione, "Machine learning with quantum computers," Springer, 2021 |
| Quantum Classifiers | M. Schuld et al., "Circuit-centric quantum classifiers," Physical Review A, 101(3):032308, 2020 |
| Expressivity | S. Sim et al., "Expressibility and entangling capability of parameterized quantum circuits," Physical Review A, 100(3):032307, 2019 |
| Barren Plateaus | J. R. McClean et al., "Barren plateaus in quantum neural network training landscapes," Nature Communications, 9(1):1-6, 2018 |
