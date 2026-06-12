"""
psiqit/qml/qsvm.py
============================================================
Quantum Support Vector Machine (QSVM)
============================================================

Quantum SVM implementation using quantum kernels.
Based on the standard SVM formulation but using quantum
kernel estimation for feature maps.

Quantum Support Vector Machines leverage quantum feature maps to compute
kernel functions that may be difficult to compute classically. The quantum
kernel is defined as: k(x, x') = |⟨φ(x)|φ(x')⟩|², where |φ(x)⟩ is a
quantum feature map.

This implementation supports:
    - Linear kernel (classical)
    - RBF kernel (classical)
    - Quantum kernel (using quantum feature maps)

The optimization uses a simplified SMO (Sequential Minimal Optimization)
algorithm to solve the dual problem.

Example:
    >>> from psiqit.qml.qsvm import QSVM
    >>> 
    >>> # Training data (XOR pattern)
    >>> X = [[0, 0], [1, 1], [0, 1], [1, 0]]
    >>> y = [0, 0, 1, 1]
    >>> 
    >>> # Create and train QSVM
    >>> qsvm = QSVM(n_qubits=2, kernel_type='quantum', feature_map='zz')
    >>> qsvm.fit(X, y)
    >>> 
    >>> # Predict
    >>> predictions = qsvm.predict([[0.5, 0.5], [0.2, 0.8]])
    >>> print(predictions)

References:
    V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces,"
    Nature, 567(7747):209-212, 2019.
    M. Schuld and F. Petruccione, "Machine learning with quantum computers,"
    Springer, 2021.
"""

import math
import random
from typing import List, Union, Optional, Tuple, Dict
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import hadamard, pauli_x, pauli_y, pauli_z, cnot, rx, ry, rz
from ..quantum.measurement import expectation, measure
from .quantum_kernel import QuantumKernel


@dataclass
class SVMResult:
    """
    Result container for SVM training.

    Attributes:
        support_vectors: List of support vectors (feature vectors).
        alphas: Lagrange multipliers for each support vector.
        bias: Bias term of the decision function.
        n_support: Number of support vectors.
        accuracy: Training accuracy.

    Examples:
        >>> result = SVMResult(support_vectors=[[0,0], [1,1]], alphas=[0.5, 0.5],
        ...                    bias=0.0, n_support=2, accuracy=1.0)
        >>> print(result)
        QSVM(n_support=2, bias=0.0000, accuracy=1.0000)
    """
    support_vectors: List[List[float]]
    alphas: List[float]
    bias: float
    n_support: int
    accuracy: float
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing number of support vectors, bias, and accuracy.
        """
        return f"QSVM(n_support={self.n_support}, bias={self.bias:.4f}, accuracy={self.accuracy:.4f})"


class QSVM:
    """
    Quantum Support Vector Machine.

    A QSVM uses quantum kernel estimation to compute kernel values that
    may be difficult to compute classically. The quantum kernel is defined
    as the squared overlap between quantum feature map states.

    The SVM optimization solves the dual problem:
        maximize Σ αᵢ - ½ Σ αᵢαⱼ yᵢyⱼ K(xᵢ, xⱼ)
        subject to 0 ≤ αᵢ ≤ C, Σ αᵢ yᵢ = 0

    The decision function is: f(x) = sign(Σ αᵢ yᵢ K(xᵢ, x) + b)

    Examples:
        >>> # Create a quantum SVM
        >>> qsvm = QSVM(n_qubits=2, kernel_type='quantum', C=1.0)
        >>> 
        >>> # Generate synthetic data
        >>> X = [[0, 0], [1, 1], [0, 1], [1, 0]]
        >>> y = [0, 0, 1, 1]
        >>> 
        >>> # Train
        >>> result = qsvm.fit(X, y)
        >>> print(f"Training accuracy: {result.accuracy:.2%}")
        >>> 
        >>> # Evaluate on test data
        >>> X_test = [[0.2, 0.3], [0.8, 0.7]]
        >>> predictions = qsvm.predict(X_test)
    """
    
    def __init__(self, n_qubits: int = 2, kernel_type: str = 'quantum', 
                 C: float = 1.0, feature_map: str = 'zz'):
        """
        Initialize the Quantum Support Vector Machine.

        Args:
            n_qubits: Number of qubits for the quantum kernel (default: 2).
            kernel_type: Type of kernel to use ('linear', 'rbf', or 'quantum').
            C: Regularization parameter (default: 1.0).
            feature_map: Type of quantum feature map ('zz', 'xx').

        Examples:
            >>> # Quantum kernel with ZZ feature map
            >>> qsvm1 = QSVM(n_qubits=4, kernel_type='quantum', feature_map='zz')
            >>> 
            >>> # Classical linear SVM
            >>> qsvm2 = QSVM(kernel_type='linear')
            >>> 
            >>> # Classical RBF SVM
            >>> qsvm3 = QSVM(kernel_type='rbf')
        """
        self.n_qubits = n_qubits
        self.kernel_type = kernel_type
        self.C = C
        self.feature_map = feature_map
        
        self._support_vectors = []
        self._alphas = []
        self._bias = 0.0
        self._kernel = None
        
        if kernel_type == 'quantum':
            self._kernel = QuantumKernel(n_qubits, feature_map)
    
    def _linear_kernel(self, x1: List[float], x2: List[float]) -> float:
        """
        Compute linear kernel: k(x, x') = x·x'.

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Dot product.
        """
        return sum(x1[i] * x2[i] for i in range(len(x1)))
    
    def _rbf_kernel(self, x1: List[float], x2: List[float], gamma: float = 1.0) -> float:
        """
        Compute RBF (Gaussian) kernel: k(x, x') = exp(-γ||x - x'||²).

        Args:
            x1: First feature vector.
            x2: Second feature vector.
            gamma: Kernel width parameter (default: 1.0).

        Returns:
            RBF kernel value in (0, 1].
        """
        diff = sum((x1[i] - x2[i])**2 for i in range(len(x1)))
        return math.exp(-gamma * diff)
    
    def _quantum_kernel(self, x1: List[float], x2: List[float]) -> float:
        """
        Compute quantum kernel: k(x, x') = |⟨φ(x)|φ(x')⟩|².

        Args:
            x1: First feature vector.
            x2: Second feature vector.

        Returns:
            Quantum kernel value in [0, 1].
        """
        return self._kernel.evaluate(x1, x2)
    
    def _compute_kernel(self, X: List[List[float]]) -> List[List[float]]:
        """
        Compute the full kernel matrix for a dataset.

        Args:
            X: List of feature vectors.

        Returns:
            n×n kernel matrix.
        """
        n = len(X)
        K = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                if self.kernel_type == 'linear':
                    k = self._linear_kernel(X[i], X[j])
                elif self.kernel_type == 'rbf':
                    k = self._rbf_kernel(X[i], X[j])
                elif self.kernel_type == 'quantum':
                    k = self._quantum_kernel(X[i], X[j])
                else:
                    raise ValueError(f"Unknown kernel type: {self.kernel_type}")
                
                K[i][j] = k
                K[j][i] = k
        
        return K
    
    def _solve_svm(self, K: List[List[float]], y: List[int]) -> Tuple[List[float], float]:
        """
        Solve the SVM dual problem using a simplified SMO algorithm.

        Args:
            K: Kernel matrix.
            y: Labels (0 or 1, internally converted to ±1).

        Returns:
            Tuple of (alphas, bias).
        """
        n = len(y)
        alphas = [0.0] * n
        bias = 0.0
        max_iter = 100
        tol = 1e-5
        
        # Convert labels to ±1
        y_bin = [1 if yi == 1 else -1 for yi in y]
        
        for _ in range(max_iter):
            alpha_old = alphas.copy()
            
            for i in range(n):
                # Compute error for i
                E_i = sum(alphas[j] * y_bin[j] * K[i][j] for j in range(n)) + bias - y_bin[i]
                
                # Check KKT conditions
                if (y_bin[i] * E_i < -tol and alphas[i] < self.C) or \
                (y_bin[i] * E_i > tol and alphas[i] > 0):
                    
                    # Select second alpha (j != i)
                    j = (i + 1) % n
                    while j == i:
                        j = (j + 1) % n
                    
                    # Compute error for j
                    E_j = sum(alphas[k] * y_bin[k] * K[j][k] for k in range(n)) + bias - y_bin[j]
                    
                    # Compute bounds
                    if y_bin[i] != y_bin[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if abs(L - H) < tol:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue
                    
                    # Update alphas
                    alpha_j_new = alphas[j] - y_bin[j] * (E_i - E_j) / eta
                    alpha_j_new = max(L, min(H, alpha_j_new))
                    
                    if abs(alpha_j_new - alphas[j]) < tol:
                        continue
                    
                    alpha_i_new = alphas[i] + y_bin[i] * y_bin[j] * (alphas[j] - alpha_j_new)
                    
                    # Update bias
                    b1 = bias - E_i - y_bin[i] * (alpha_i_new - alphas[i]) * K[i][i] - y_bin[j] * (alpha_j_new - alphas[j]) * K[i][j]
                    b2 = bias - E_j - y_bin[i] * (alpha_i_new - alphas[i]) * K[i][j] - y_bin[j] * (alpha_j_new - alphas[j]) * K[j][j]
                    
                    if 0 < alpha_i_new < self.C:
                        bias = b1
                    elif 0 < alpha_j_new < self.C:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2
                    
                    alphas[i] = alpha_i_new
                    alphas[j] = alpha_j_new
        
        return alphas, bias
    
    def fit(self, X: List[List[float]], y: List[int], max_iter: int = 100) -> SVMResult:
        """
        Train the QSVM on a dataset.

        Args:
            X: Training data (list of feature vectors).
            y: Labels (0 or 1).
            max_iter: Maximum iterations for the SMO algorithm (default: 100).

        Returns:
            SVMResult containing support vectors, alphas, bias, and accuracy.

        Examples:
            >>> X = [[0, 0], [1, 1], [0, 1], [1, 0]]
            >>> y = [0, 0, 1, 1]
            >>> qsvm = QSVM()
            >>> result = qsvm.fit(X, y)
            >>> print(f"Number of support vectors: {result.n_support}")
        """
        n = len(X)
        
        # Convert labels to 0/1 if needed
        y_labels = [1 if yi == 1 else -1 for yi in y]
        
        # Compute kernel matrix
        K = self._compute_kernel(X)
        
        # Solve SVM
        alphas, bias = self._solve_svm(K, y)
        
        # Extract support vectors
        support_indices = [i for i in range(n) if alphas[i] > 1e-5]
        support_vectors = [X[i] for i in support_indices]
        support_alphas = [alphas[i] for i in support_indices]
        support_y = [y_labels[i] for i in support_indices]
        
        self._support_vectors = support_vectors
        self._alphas = support_alphas
        self._support_y = support_y
        self._bias = bias
        
        # Calculate training accuracy
        predictions = [self._predict_single(x) for x in X]
        accuracy = sum(1 for p, t in zip(predictions, y) if p == t) / n
        
        return SVMResult(
            support_vectors=support_vectors,
            alphas=support_alphas,
            bias=bias,
            n_support=len(support_vectors),
            accuracy=accuracy
        )
    
    def _predict_single(self, x: List[float]) -> int:
        """
        Predict the label for a single data point.

        Args:
            x: Feature vector.

        Returns:
            Predicted label (0 or 1).
        """
        # Compute decision function
        decision = self._bias
        for i, sv in enumerate(self._support_vectors):
            if self.kernel_type == 'linear':
                k = self._linear_kernel(x, sv)
            elif self.kernel_type == 'rbf':
                k = self._rbf_kernel(x, sv)
            elif self.kernel_type == 'quantum':
                k = self._quantum_kernel(x, sv)
            else:
                k = 0
            
            decision += self._alphas[i] * self._support_y[i] * k
        
        # Return class (0 or 1)
        return 1 if decision >= 0 else 0
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """
        Predict labels for multiple data points.

        Args:
            X: List of feature vectors.

        Returns:
            List of predicted labels.

        Examples:
            >>> qsvm = QSVM()
            >>> qsvm.fit(X_train, y_train)
            >>> predictions = qsvm.predict(X_test)
            >>> print(predictions)
        """
        return [self._predict_single(x) for x in X]
    
    def decision_function(self, X: List[List[float]]) -> List[float]:
        """
        Compute the decision function values.

        Args:
            X: List of feature vectors.

        Returns:
            List of decision values (signed distance to the hyperplane).

        Examples:
            >>> decisions = qsvm.decision_function(X_test)
            >>> print(decisions)
        """
        decisions = []
        for x in X:
            decision = self._bias
            for i, sv in enumerate(self._support_vectors):
                if self.kernel_type == 'linear':
                    k = self._linear_kernel(x, sv)
                elif self.kernel_type == 'rbf':
                    k = self._rbf_kernel(x, sv)
                elif self.kernel_type == 'quantum':
                    k = self._quantum_kernel(x, sv)
                else:
                    k = 0
                decision += self._alphas[i] * self._support_y[i] * k
            decisions.append(decision)
        return decisions
    
    def score(self, X: List[List[float]], y: List[int]) -> float:
        """
        Calculate the accuracy on test data.

        Args:
            X: Test data (feature vectors).
            y: True labels.

        Returns:
            Accuracy score between 0 and 1.

        Examples:
            >>> accuracy = qsvm.score(X_test, y_test)
            >>> print(f"Test accuracy: {accuracy:.2%}")
        """
        predictions = self.predict(X)
        return sum(1 for p, t in zip(predictions, y) if p == t) / len(y)


class QuantumKernelSVM(QSVM):
    """
    Alias for QSVM with quantum kernel.

    This class is identical to QSVM but provides a more specific name
    when using quantum kernels.
    """
    pass


__all__ = [
    'SVMResult',
    'QSVM',
    'QuantumKernelSVM',
]