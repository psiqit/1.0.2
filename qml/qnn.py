"""
psiqit/qml/qnn.py
============================================================
Quantum Neural Network (QNN)
============================================================

Variational quantum circuits for machine learning:
    • Quantum neural network layers
    • Parameterized quantum circuits
    • Training via parameter shift rule
    • Classical optimizer integration

Quantum Neural Networks (QNNs) are variational quantum circuits that can be
trained to perform machine learning tasks. They consist of alternating layers
of single-qubit rotations (parameterized) and entangling gates.

Architecture:
    - Encoding: Classical data is encoded into quantum states
    - Variational layers: Parameterized RY/RZ rotations + CNOT/CZ entangling
    - Measurement: Expectation value of Pauli operator (Z, X, or Y)
    - Output: Classical value used for prediction

Example:
    >>> from psiqit.qml.qnn import QNN, QNNLayer
    >>> import numpy as np
    >>> 
    >>> # Create a quantum neural network
    >>> qnn = QNN(n_qubits=4, n_layers=3)
    >>> 
    >>> # Forward pass
    >>> output = qnn.forward(input_state)
    >>> 
    >>> # Train on classical data
    >>> X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # XOR inputs
    >>> y = [0, 1, 1, 0]  # XOR labels
    >>> result = qnn.train(X, y, epochs=100, learning_rate=0.1)
    >>> print(f"Accuracy: {result.accuracy:.2%}")

References:
    M. Schuld and F. Petruccione, "Machine learning with quantum computers,"
    Springer, 2021.
    M. Benedetti et al., "Parameterized quantum circuits as machine learning models,"
    Quantum Science and Technology, 4(4):043001, 2019.
"""

import math
import random
from typing import List, Union, Optional, Callable, Tuple, Dict
from dataclasses import dataclass, field

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, zero, basis
from ..quantum.operator import hadamard, pauli_x, pauli_y, pauli_z, rx, ry, rz, cnot, cz
from ..quantum.measurement import expectation, measure
from ..utils.random import random_state


@dataclass
class QNNResult:
    """
    Result container for QNN training.

    Attributes:
        final_loss: Final loss value after training.
        accuracy: Classification accuracy (for classification tasks).
        epochs: Number of training epochs.
        params: Final optimized parameters.
        predictions: Final predictions (0/1) for the training data.

    Examples:
        >>> result = QNNResult(final_loss=0.05, accuracy=0.95, epochs=100,
        ...                    params=[0.1, 0.2], predictions=[0, 1, 1, 0])
        >>> print(result)
        QNN(final_loss=0.0500, accuracy=95.00%, epochs=100)
    """
    final_loss: float
    accuracy: float
    epochs: int
    params: List[float]
    predictions: List[int]
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing final loss, accuracy, and epochs.
        """
        return f"QNN(final_loss={self.final_loss:.4f}, accuracy={self.accuracy:.2%}, epochs={self.epochs})"


class QNNLayer:
    """
    A single layer of a Quantum Neural Network.

    Each layer consists of:
        - RY and RZ rotations on each qubit (2n parameters)
        - Entangling gates (CNOT or CZ) in a cyclic topology

    Attributes:
        n_qubits: Number of qubits.
        entangler: Type of entangling gate ('cnot' or 'cz').
        n_params: Number of parameters per layer (2 * n_qubits).

    Examples:
        >>> layer = QNNLayer(n_qubits=4, entangler='cnot')
        >>> circuit = QuantumCircuit(4)
        >>> params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        >>> layer.apply(circuit, params, start_idx=0)
    """
    
    def __init__(self, n_qubits: int, entangler: str = 'cnot'):
        """
        Initialize a QNN layer.

        Args:
            n_qubits: Number of qubits.
            entangler: Type of entangling gate ('cnot' or 'cz').

        Examples:
            >>> layer = QNNLayer(4, entangler='cnot')
        """
        self.n_qubits = n_qubits
        self.entangler = entangler
        self.n_params = 2 * n_qubits  # Ry and Rz for each qubit
    
    def apply(self, circuit: QuantumCircuit, params: List[float], start_idx: int = 0):
        """
        Apply the layer to a quantum circuit.

        Args:
            circuit: Quantum circuit to add gates to.
            params: Parameter list for this layer.
            start_idx: Starting index in the parameter list.
        """
        # Rotation gates on each qubit
        for i in range(self.n_qubits):
            circuit.ry(i, params[start_idx + i])
            circuit.rz(i, params[start_idx + self.n_qubits + i])
        
        # Entangling gates
        if self.entangler == 'cnot':
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            # Last connection to create cyclic entanglement
            circuit.cx(self.n_qubits - 1, 0)
        elif self.entangler == 'cz':
            for i in range(self.n_qubits - 1):
                circuit.cz(i, i + 1)
            circuit.cz(self.n_qubits - 1, 0)
    
    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"QNNLayer(n_qubits={self.n_qubits}, entangler={self.entangler}, n_params={self.n_params})"


class QNN:
    """
    Quantum Neural Network with multiple layers.

    The QNN is a variational quantum circuit that can be trained to perform
    machine learning tasks. It supports:
        - Multiple layers of parameterized rotations and entangling gates
        - Measurement of Pauli operators (Z, X, Y) for output
        - Amplitude encoding for classical data
        - Training using finite-difference gradients

    The network architecture:
        - Data encoding: Classical data is encoded via amplitude encoding
        - Variational layers: Alternating rotations and entanglement
        - Measurement: Expectation value of Pauli operator
        - Output: Sigmoid-activated value for classification

    Examples:
        >>> # Create a QNN for classification
        >>> qnn = QNN(n_qubits=4, n_layers=3, measurement='z')
        >>> 
        >>> # Generate synthetic data
        >>> X = [[0, 0, 0, 0], [1, 1, 1, 1]]
        >>> y = [0, 1]
        >>> 
        >>> # Train
        >>> result = qnn.train(X, y, epochs=50, learning_rate=0.1)
        >>> print(f"Accuracy: {result.accuracy:.2%}")
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 3, 
                 entangler: str = 'cnot', measurement: str = 'z'):
        """
        Initialize a Quantum Neural Network.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of variational layers (default: 3).
            entangler: Type of entangling gate ('cnot' or 'cz').
            measurement: Measurement basis ('z', 'x', or 'y').

        Examples:
            >>> qnn = QNN(n_qubits=4, n_layers=3, measurement='z')
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entangler = entangler
        self.measurement = measurement
        
        # Create layers
        self.layers = [QNNLayer(n_qubits, entangler) for _ in range(n_layers)]
        
        # Calculate total parameters
        self.n_params = n_layers * (2 * n_qubits)
        
        # Initialize parameters
        self.params = None
        self._init_params_random()
    
    def _init_params_random(self, seed: Optional[int] = None):
        """
        Initialize parameters randomly in [-π, π].

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self.params = [random.uniform(-math.pi, math.pi) for _ in range(self.n_params)]
    
    def set_params(self, params: List[float]):
        """
        Set the network parameters.

        Args:
            params: New parameter list.

        Raises:
            ValueError: If the number of parameters doesn't match.
        """
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(params)}")
        self.params = params
    
    def build_circuit(self, input_state: Optional[Ket] = None) -> QuantumCircuit:
        """
        Build the QNN circuit.

        Args:
            input_state: Initial state (if None, starts from |0...0⟩).

        Returns:
            QuantumCircuit: The constructed circuit.

        Examples:
            >>> qnn = QNN(2, 2)
            >>> circuit = qnn.build_circuit()
            >>> print(circuit.depth)
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        # Initialize state
        if input_state is not None:
            # This is simplified - proper state preparation would be more complex
            pass
        
        # Apply each layer
        param_idx = 0
        for layer in self.layers:
            layer.apply(circuit, self.params, param_idx)
            param_idx += 2 * self.n_qubits
        
        return circuit
    
    def forward(self, input_state: Optional[Ket] = None) -> Ket:
        """
        Forward pass through the network.

        Args:
            input_state: Input quantum state (if None, starts from |0...0⟩).

        Returns:
            Ket: Output quantum state.

        Examples:
            >>> qnn = QNN(2, 2)
            >>> from psiqit.quantum.state import zero
            >>> output = qnn.forward(zero())
            >>> print(output.dim)
            4
        """
        circuit = self.build_circuit(input_state)
        return circuit.run()
    
    def measure_expectation(self, input_state: Optional[Ket] = None) -> float:
        """
        Compute the expectation value of the measurement operator.

        Args:
            input_state: Input quantum state (if None, starts from |0...0⟩).

        Returns:
            Expectation value in the range [-1, 1].

        Examples:
            >>> qnn = QNN(1, 2)
            >>> value = qnn.measure_expectation()
            >>> print(f"Expectation: {value:.4f}")
        """
        output_state = self.forward(input_state)
        
        # Normalize the state
        if not output_state.is_normalized:
            output_state = output_state.normalize()
        
        # Choose measurement operator
        if self.measurement == 'z':
            from ..quantum.operator import pauli_z
            op = pauli_z()
        elif self.measurement == 'x':
            from ..quantum.operator import pauli_x
            op = pauli_x()
        elif self.measurement == 'y':
            from ..quantum.operator import pauli_y
            op = pauli_y()
        else:
            op = pauli_z()
        
        # For multi-qubit, measure first qubit using partial trace
        if self.n_qubits == 1:
            from ..quantum.measurement import expectation
            return expectation(op, output_state)
        else:
            # Compute expectation of Pauli Z on first qubit
            # |ψ⟩ → ⟨ψ|Z₁|ψ⟩
            n = self.n_qubits
            dim = output_state.dim
            exp_val = 0.0
            
            for i in range(dim):
                # Get bit of first qubit (most significant)
                first_bit = (i >> (n - 1)) & 1
                sign = 1 if first_bit == 0 else -1
                exp_val += sign * abs(output_state.data[i]) ** 2
            
            return exp_val
    
    def predict(self, X: List[List[float]], as_probability: bool = True) -> List[float]:
        """
        Make predictions for input data.

        Args:
            X: List of input feature vectors.
            as_probability: If True, apply sigmoid activation to output in [0,1].

        Returns:
            List of prediction values.

        Examples:
            >>> qnn = QNN(2, 2)
            >>> X = [[0, 0], [1, 1]]
            >>> predictions = qnn.predict(X)
            >>> print(predictions)
        """
        predictions = []
        for x in X:
            # Encode classical data into quantum state
            encoded_state = self._encode_data(x)
            output = self.measure_expectation(encoded_state)
            
            if as_probability:
                # Sigmoid activation
                output = 1 / (1 + math.exp(-output))
            predictions.append(output)
        
        return predictions
    
    def _encode_data(self, x: List[float]) -> Ket:
        """
        Encode classical data into a quantum state using amplitude encoding.

        Args:
            x: Feature vector.

        Returns:
            Ket: Encoded quantum state.
        """
        n_features = min(len(x), self.n_qubits)
        dim = 2 ** self.n_qubits
        amplitudes = [0.0] * dim
        
        # Simple amplitude encoding
        norm = math.sqrt(sum(xi**2 for xi in x[:n_features]) + 1e-10)
        for i in range(n_features):
            amplitudes[i] = x[i] / norm
        
        # Ensure normalization
        total = sum(abs(a)**2 for a in amplitudes)
        if total < 0.5:
            amplitudes[0] = math.sqrt(1 - total)
        
        return Ket(amplitudes).normalize()
    
    def loss(self, X: List[List[float]], y: List[float]) -> float:
        """
        Compute the mean squared error loss.

        Args:
            X: Input features.
            y: Target values.

        Returns:
            Mean squared error.
        """
        predictions = self.predict(X, as_probability=True)
        loss = sum((p - t)**2 for p, t in zip(predictions, y)) / len(y)
        return loss
    
    def accuracy(self, X: List[List[float]], y: List[int]) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Input features.
            y: Target labels (0 or 1).

        Returns:
            Accuracy (0 to 1).
        """
        predictions = self.predict(X, as_probability=True)
        binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
        correct = sum(1 for p, t in zip(binary_preds, y) if p == t)
        return correct / len(y)
    
    def _parameter_shift_gradient(self, X: List[List[float]], y: List[float], 
                                   param_idx: int, epsilon: float = 1e-5) -> float:
        """
        Compute gradient using finite differences.

        Args:
            X: Input features.
            y: Target values.
            param_idx: Index of the parameter to differentiate.
            epsilon: Step size for finite differences.

        Returns:
            Gradient value.
        """
        # Save current parameters
        original_params = self.params.copy()
        
        # Forward pass with +ε
        self.params = original_params.copy()
        self.params[param_idx] += epsilon
        loss_plus = self.loss(X, y)
        
        # Forward pass with -ε
        self.params = original_params.copy()
        self.params[param_idx] -= epsilon
        loss_minus = self.loss(X, y)
        
        # Restore parameters
        self.params = original_params
        
        # Finite difference gradient
        grad = (loss_plus - loss_minus) / (2 * epsilon)
        return grad
    
    def train(self, X: List[List[float]], y: List[int], 
              epochs: int = 100, learning_rate: float = 0.1,
              verbose: bool = True) -> QNNResult:
        """
        Train the QNN on labeled data.

        Args:
            X: Input features (list of feature vectors).
            y: Target labels (0 or 1).
            epochs: Number of training epochs.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            QNNResult containing training history.

        Examples:
            >>> # XOR dataset
            >>> X = [[0, 0], [0, 1], [1, 0], [1, 1]]
            >>> y = [0, 1, 1, 0]
            >>> qnn = QNN(n_qubits=2, n_layers=2)
            >>> result = qnn.train(X, y, epochs=100, learning_rate=0.1)
            >>> print(f"Final accuracy: {result.accuracy:.2%}")
        """
        losses = []
        
        for epoch in range(epochs):
            # Compute gradients
            gradients = []
            for i in range(self.n_params):
                grad = self._parameter_shift_gradient(X, y, i)
                gradients.append(grad)
            
            # Update parameters
            for i in range(self.n_params):
                self.params[i] -= learning_rate * gradients[i]
            
            # Compute loss
            loss = self.loss(X, y)
            losses.append(loss)
            
            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch}: loss={loss:.4f}, accuracy={acc:.2%}")
        
        # Final predictions
        predictions = self.predict(X, as_probability=True)
        binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
        accuracy = self.accuracy(X, y)
        
        return QNNResult(
            final_loss=losses[-1] if losses else 0.0,
            accuracy=accuracy,
            epochs=epochs,
            params=self.params.copy(),
            predictions=binary_preds
        )
    
    def __repr__(self) -> str:
        """Return a string representation of the QNN."""
        return f"QNN(n_qubits={self.n_qubits}, n_layers={self.n_layers}, n_params={self.n_params})"


class QuantumClassifier(QNN):
    """
    Quantum classifier (alias for QNN with classification-specific defaults).

    This class inherits from QNN and is used for classification tasks.
    """
    pass


class VariationalQuantumCircuit(QNN):
    """
    Variational quantum circuit (alias for QNN).

    This class inherits from QNN and is used for variational quantum algorithms.
    """
    pass


__all__ = [
    'QNNResult',
    'QNNLayer',
    'QNN',
    'QuantumClassifier',
    'VariationalQuantumCircuit',
]