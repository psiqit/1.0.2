#psiqit/qml/qgan.py
#Quantum Generative Adversarial Networks - Fixed

"""
Quantum Generative Adversarial Networks (QGAN)

Quantum GANs are a hybrid quantum-classical machine learning model that extends
the classical GAN framework to quantum computing. A quantum generator produces
quantum states, while a discriminator (either quantum or classical) tries to
distinguish between real and generated states.

The QGAN framework consists of:
    - QuantumGenerator: Parameterized quantum circuit that generates states
    - QuantumDiscriminator: Quantum circuit that classifies states
    - ClassicalDiscriminator: Classical neural network alternative

The training alternates between:
    1. Training the discriminator to distinguish real from generated states
    2. Training the generator to fool the discriminator

References:
    P. L. Dallaire-Demers and N. Killoran, "Quantum generative adversarial networks,"
    Physical Review A, 98(1):012324, 2018.
    S. Lloyd and C. Weedbrook, "Quantum generative adversarial learning,"
    Physical Review Letters, 121(4):040502, 2018.

Example:
    >>> from psiqit.qml.qgan import QGAN
    >>> from psiqit.quantum.state import random_state
    >>> 
    >>> # Create target states (e.g., random states to learn)
    >>> target_states = [random_state(2) for _ in range(100)]
    >>> 
    >>> # Initialize QGAN
    >>> qgan = QGAN(n_qubits=1, n_latent=2, generator_layers=2,
    ...             discriminator_type='quantum')
    >>> 
    >>> # Train the QGAN
    >>> result = qgan.train(target_states, epochs=50, learning_rate=0.01)
    >>> 
    >>> # Generate new states
    >>> generated_states = qgan.generate(n_samples=10)
"""

import math
import random
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis, random_state
from ..quantum.operator import Operator, rx, ry, rz, hadamard, pauli_z
from ..quantum.measurement import expectation, measure
from ..utils.random import random_unitary


@dataclass
class QGANResult:
    """
    Result container for QGAN training.

    Attributes:
        generator_loss: List of generator loss values per epoch.
        discriminator_loss: List of discriminator loss values per epoch.
        fidelity_history: List of fidelity values between generated and target states.
        epoch_history: List of epoch numbers.
        success: Whether training completed successfully.
        final_generator: Trained generator object.
        final_discriminator: Trained discriminator object.

    Examples:
        >>> result = QGANResult(generator_loss=[0.5, 0.4], discriminator_loss=[0.6, 0.5],
        ...                     fidelity_history=[0.7, 0.8], epoch_history=[0, 1],
        ...                     success=True, final_generator=gen, final_discriminator=disc)
        >>> print(result)
        QGANResult(final_fidelity=0.8000, epochs=2)
    """
    generator_loss: List[float]
    discriminator_loss: List[float]
    fidelity_history: List[float]
    epoch_history: List[int]
    success: bool
    final_generator: 'QuantumGenerator'
    final_discriminator: 'QuantumDiscriminator'
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing final fidelity and number of epochs.
        """
        return (f"QGANResult(final_fidelity={self.fidelity_history[-1]:.4f}, "
                f"epochs={len(self.epoch_history)})")


class QuantumGenerator:
    """
    Quantum generator for QGAN.

    The generator is a parameterized quantum circuit that maps a latent vector
    z to a quantum state. It consists of:
        - Latent encoding: RY gates with angles based on z
        - Variational layers: RY and RZ rotations with entangling CNOT gates

    Attributes:
        n_qubits: Number of qubits in the generated state.
        n_latent: Dimension of the latent vector.
        n_layers: Number of variational layers.
        n_params: Total number of trainable parameters.
        params: Current parameter values.

    Examples:
        >>> generator = QuantumGenerator(n_qubits=2, n_latent=2, n_layers=3)
        >>> z = [0.5, -0.3]
        >>> state = generator.generate(z)
        >>> print(state.dim)
        4
    """
    
    def __init__(self, n_qubits: int, n_latent: int, n_layers: int = 2):
        """
        Initialize the quantum generator.

        Args:
            n_qubits: Number of qubits in the generated state.
            n_latent: Dimension of the latent vector (input noise).
            n_layers: Number of variational layers.

        Examples:
            >>> gen = QuantumGenerator(n_qubits=2, n_latent=2, n_layers=3)
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 2
        self.params = None
        self._init_params()
    
    def _init_params(self):
        """Initialize parameters randomly in [-π, π]."""
        self.params = [random.uniform(-math.pi, math.pi) for _ in range(self.n_params)]
    
    def _encode_latent(self, circuit: QuantumCircuit, z: List[float]):
        """
        Encode the latent vector into the circuit using RY gates.

        Args:
            circuit: Quantum circuit to add gates to.
            z: Latent vector (list of floats).
        """
        for i, zi in enumerate(z):
            if i < self.n_qubits:
                circuit.ry(i, math.pi * zi)
    
    def _variational_layer(self, circuit: QuantumCircuit, params: List[float], start_idx: int):
        """
        Add a variational layer to the circuit.

        Each layer consists of:
            - RY and RZ rotations on each qubit
            - CNOT entangling gates in a ring topology

        Args:
            circuit: Quantum circuit to add gates to.
            params: Parameter list for this layer.
            start_idx: Starting index in the parameter list.
        """
        n = self.n_qubits
        for i in range(n):
            circuit.ry(i, params[start_idx + i])
            circuit.rz(i, params[start_idx + n + i])
        
        # Use cx instead of cnot
        for i in range(n - 1):
            circuit.cx(i, i + 1)
        circuit.cx(n - 1, 0)
    
    def generate(self, z: List[float]) -> Ket:
        """
        Generate a quantum state from a latent vector.

        Args:
            z: Latent vector (list of floats of length n_latent).

        Returns:
            Ket: Generated quantum state.

        Examples:
            >>> gen = QuantumGenerator(2, 2)
            >>> state = gen.generate([0.5, -0.2])
        """
        circuit = QuantumCircuit(self.n_qubits)
        self._encode_latent(circuit, z)
        
        for layer in range(self.n_layers):
            self._variational_layer(circuit, self.params, layer * 2 * self.n_qubits)
        
        return circuit.run()
    
    def set_params(self, params: List[float]):
        """
        Set the generator parameters.

        Args:
            params: New parameter list.

        Raises:
            ValueError: If the number of parameters doesn't match.
        """
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {len(params)}")
        self.params = params
    
    def get_params(self) -> List[float]:
        """
        Get a copy of the current parameters.

        Returns:
            Copy of the parameter list.
        """
        return self.params.copy()


class QuantumDiscriminator:
    """
    Quantum discriminator for QGAN.

    The quantum discriminator is a parameterized quantum circuit that
    outputs a score (0 to 1) indicating whether a state is real or fake.
    The score is derived from the expectation value of Pauli-Z.

    Attributes:
        n_qubits: Number of qubits in the input state.
        n_layers: Number of variational layers.
        n_params: Total number of trainable parameters.
        params: Current parameter values.

    Examples:
        >>> discriminator = QuantumDiscriminator(n_qubits=2, n_layers=2)
        >>> from psiqit.quantum.state import random_state
        >>> score = discriminator.discriminate(random_state(4))
        >>> print(f"Score: {score:.4f}")
    """
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        """
        Initialize the quantum discriminator.

        Args:
            n_qubits: Number of qubits in the input state.
            n_layers: Number of variational layers.

        Examples:
            >>> disc = QuantumDiscriminator(n_qubits=2, n_layers=3)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_layers * n_qubits * 2
        self.params = None
        self._init_params()
    
    def _init_params(self):
        """Initialize parameters randomly in [-π, π]."""
        self.params = [random.uniform(-math.pi, math.pi) for _ in range(self.n_params)]
    
    def _discriminator_circuit(self) -> QuantumCircuit:
        """
        Build the discriminator quantum circuit.

        Returns:
            QuantumCircuit: The discriminator circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                circuit.ry(i, self.params[layer * 2 * self.n_qubits + i])
                circuit.rz(i, self.params[layer * 2 * self.n_qubits + self.n_qubits + i])
            
            # Use cx instead of cnot
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
        
        return circuit
    
    def discriminate(self, state: Ket) -> float:
        """
        Compute the discriminator score for a quantum state.

        Args:
            state: Input quantum state.

        Returns:
            Score between 0 and 1 (1 = real, 0 = fake).

        Examples:
            >>> disc = QuantumDiscriminator(1)
            >>> from psiqit.quantum.state import zero
            >>> score = disc.discriminate(zero())
            >>> print(f"Score: {score:.4f}")
        """
        # Normalize state first
        if not state.is_normalized:
            state = state.normalize()
        
        exp_val = expectation(pauli_z(), state)
        score = 1 / (1 + math.exp(-exp_val))
        return score
    
    def set_params(self, params: List[float]):
        """
        Set the discriminator parameters.

        Args:
            params: New parameter list.

        Raises:
            ValueError: If the number of parameters doesn't match.
        """
        if len(params) != self.n_params:
            raise ValueError(f"Expected {self.n_params} params, got {len(params)}")
        self.params = params
    
    def get_params(self) -> List[float]:
        """
        Get a copy of the current parameters.

        Returns:
            Copy of the parameter list.
        """
        return self.params.copy()


class ClassicalDiscriminator:
    """
    Classical neural network discriminator for QGAN.

    This is a classical feed-forward neural network that takes the
    real and imaginary parts of the state vector as input and outputs
    a score indicating whether the state is real or fake.

    Architecture:
        - Input: 2 * 2ⁿ (real and imag parts of amplitudes)
        - Hidden layers: Configurable (default: [64, 32])
        - Output: Single value (after sigmoid activation)

    Attributes:
        input_dim: Input dimension.
        hidden_dims: List of hidden layer dimensions.
        weights: List of weight matrices.
        biases: List of bias vectors.

    Examples:
        >>> disc = ClassicalDiscriminator(input_dim=4, hidden_dims=[64, 32])
        >>> from psiqit.quantum.state import zero
        >>> score = disc.discriminate(zero())
        >>> print(f"Score: {score:.4f}")
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        """
        Initialize the classical discriminator.

        Args:
            input_dim: Input dimension (2 * 2ⁿ for n qubits).
            hidden_dims: List of hidden layer dimensions (default: [64, 32]).

        Examples:
            >>> # For 2-qubit states (dim=4, input_dim=8)
            >>> disc = ClassicalDiscriminator(input_dim=8, hidden_dims=[128, 64])
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        self.weights = []
        self.biases = []
        
        prev_dim = input_dim
        for hdim in hidden_dims:
            self.weights.append(np.random.randn(prev_dim, hdim) * 0.01)
            self.biases.append(np.zeros(hdim))
            prev_dim = hdim
        
        self.weights.append(np.random.randn(prev_dim, 1) * 0.01)
        self.biases.append(np.zeros(1))
    
    def _relu(self, x):
        """ReLU activation function: max(0, x)."""
        return np.maximum(0, x)
    
    def _sigmoid(self, x):
        """Sigmoid activation function: 1/(1+e^{-x})."""
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass through the neural network.

        Args:
            x: Input vector.

        Returns:
            Output score (after sigmoid).
        """
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            x = self._relu(x @ W + b)
        x = x @ self.weights[-1] + self.biases[-1]
        return float(self._sigmoid(x))
    
    def discriminate(self, state: Ket) -> float:
        """
        Compute the discriminator score for a quantum state.

        Args:
            state: Input quantum state.

        Returns:
            Score between 0 and 1 (1 = real, 0 = fake).

        Examples:
            >>> disc = ClassicalDiscriminator(input_dim=4)
            >>> from psiqit.quantum.state import zero
            >>> score = disc.discriminate(zero())
        """
        vec = np.array([state.data[i].real for i in range(len(state.data))])
        vec = np.append(vec, [state.data[i].imag for i in range(len(state.data))])
        return self.forward(vec)


class QGAN:
    """
    Quantum Generative Adversarial Network (QGAN).

    The QGAN trains a quantum generator to produce states that resemble
    target states, while a discriminator (quantum or classical) tries to
    distinguish real from generated states.

    The training alternates between:
        1. Updating the discriminator to better distinguish real/fake
        2. Updating the generator to better fool the discriminator

    Attributes:
        n_qubits: Number of qubits for generated states.
        n_latent: Dimension of the latent vector.
        generator: QuantumGenerator instance.
        discriminator: QuantumDiscriminator or ClassicalDiscriminator.
        discriminator_type: 'quantum' or 'classical'.

    Examples:
        >>> from psiqit.quantum.state import random_state, bell_phi_plus
        >>> 
        >>> # Create target states (e.g., Bell states)
        >>> target_states = [bell_phi_plus() for _ in range(50)]
        >>> 
        >>> # Initialize QGAN
        >>> qgan = QGAN(n_qubits=2, n_latent=2, generator_layers=2,
        ...             discriminator_type='quantum')
        >>> 
        >>> # Train
        >>> result = qgan.train(target_states, epochs=100, learning_rate=0.01)
        >>> 
        >>> # Generate new states
        >>> generated = qgan.generate(n_samples=5)
    """
    
    def __init__(self, n_qubits: int, n_latent: int = 2,
                 generator_layers: int = 2, discriminator_layers: int = 2,
                 discriminator_type: str = 'quantum'):
        """
        Initialize the QGAN.

        Args:
            n_qubits: Number of qubits for generated states.
            n_latent: Dimension of the latent vector (default: 2).
            generator_layers: Number of layers in the generator (default: 2).
            discriminator_layers: Number of layers in the discriminator (default: 2).
            discriminator_type: 'quantum' or 'classical' (default: 'quantum').

        Examples:
            >>> qgan = QGAN(n_qubits=2, n_latent=2, discriminator_type='classical')
        """
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        
        self.generator = QuantumGenerator(n_qubits, n_latent, generator_layers)
        
        if discriminator_type == 'quantum':
            self.discriminator = QuantumDiscriminator(n_qubits, discriminator_layers)
        else:
            self.discriminator = ClassicalDiscriminator(2 * (2 ** n_qubits))
        
        self.discriminator_type = discriminator_type
    
    def _generator_loss(self, z: List[float]) -> float:
        """
        Compute the generator loss for a single latent vector.

        The generator wants to maximize the discriminator's score for
        generated states, which corresponds to minimizing -log(D(G(z))).

        Args:
            z: Latent vector.

        Returns:
            Generator loss value.
        """
        generated = self.generator.generate(z)
        score = self.discriminator.discriminate(generated)
        return -math.log(score + 1e-8)
    
    def _discriminator_loss(self, real_states: List[Ket], z_list: List[List[float]]) -> float:
        """
        Compute the discriminator loss for a batch.

        The discriminator wants to:
            - Maximize score for real states: minimize -log(D(real))
            - Minimize score for fake states: minimize -log(1 - D(G(z)))

        Args:
            real_states: List of real target states.
            z_list: List of latent vectors for fake states.

        Returns:
            Average discriminator loss.
        """
        loss = 0.0
        
        for state in real_states:
            score_real = self.discriminator.discriminate(state)
            loss -= math.log(score_real + 1e-8)
        
        for z in z_list:
            generated = self.generator.generate(z)
            score_fake = self.discriminator.discriminate(generated)
            loss -= math.log(1 - score_fake + 1e-8)
        
        return loss / (len(real_states) + len(z_list))
    
    def _gradient(self, loss_fn: Callable, params: List[float], 
                  eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient of a loss function using finite differences.

        Args:
            loss_fn: Loss function that depends on the parameters.
            params: Current parameter values.
            eps: Finite difference step size.

        Returns:
            List of gradient values.
        """
        grads = []
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            
            if hasattr(loss_fn, 'generator'):
                self.generator.set_params(params_plus)
                loss_plus = loss_fn()
                self.generator.set_params(params_minus)
                loss_minus = loss_fn()
                self.generator.set_params(params)
            else:
                self.discriminator.set_params(params_plus)
                loss_plus = loss_fn()
                self.discriminator.set_params(params_minus)
                loss_minus = loss_fn()
                self.discriminator.set_params(params)
            
            grad = (loss_plus - loss_minus) / (2 * eps)
            grads.append(grad)
        
        return grads
    
    def train(self, target_states: List[Ket], epochs: int = 50,
              batch_size: int = 10, learning_rate: float = 0.01,
              verbose: bool = True) -> QGANResult:
        """
        Train the QGAN on target states.

        Args:
            target_states: List of target quantum states to learn.
            epochs: Number of training epochs.
            batch_size: Batch size for each training step.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            QGANResult containing training history.

        Examples:
            >>> from psiqit.quantum.state import bell_phi_plus
            >>> target = [bell_phi_plus() for _ in range(100)]
            >>> qgan = QGAN(n_qubits=2, n_latent=2)
            >>> result = qgan.train(target, epochs=50, learning_rate=0.01)
        """
        generator_losses = []
        discriminator_losses = []
        fidelities = []
        
        for epoch in range(epochs):
            batch_targets = random.sample(target_states, min(batch_size, len(target_states)))
            z_batch = [[random.uniform(-1, 1) for _ in range(self.n_latent)] 
                       for _ in range(len(batch_targets))]
            
            def disc_loss():
                return self._discriminator_loss(batch_targets, z_batch)
            
            disc_grads = self._gradient(disc_loss, self.discriminator.get_params())
            disc_params = self.discriminator.get_params()
            for i in range(len(disc_params)):
                disc_params[i] -= learning_rate * disc_grads[i]
            self.discriminator.set_params(disc_params)
            
            def gen_loss():
                total_loss = 0.0
                for z in z_batch:
                    total_loss += self._generator_loss(z)
                return total_loss / len(z_batch)
            
            gen_grads = self._gradient(gen_loss, self.generator.get_params())
            gen_params = self.generator.get_params()
            for i in range(len(gen_params)):
                gen_params[i] -= learning_rate * gen_grads[i]
            self.generator.set_params(gen_params)
            
            disc_loss_val = disc_loss()
            gen_loss_val = gen_loss()
            generator_losses.append(gen_loss_val)
            discriminator_losses.append(disc_loss_val)
            
            avg_fidelity = 0.0
            for target in batch_targets:
                z = [random.uniform(-1, 1) for _ in range(self.n_latent)]
                generated = self.generator.generate(z)
                fidelity = abs(generated.inner(target)) ** 2
                avg_fidelity += fidelity
            avg_fidelity /= len(batch_targets)
            fidelities.append(avg_fidelity)
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  Epoch {epoch}: G_loss={gen_loss_val:.4f}, "
                      f"D_loss={disc_loss_val:.4f}, fidelity={avg_fidelity:.4f}")
        
        return QGANResult(
            generator_loss=generator_losses,
            discriminator_loss=discriminator_losses,
            fidelity_history=fidelities,
            epoch_history=list(range(epochs)),
            success=True,
            final_generator=self.generator,
            final_discriminator=self.discriminator
        )
    
    def generate(self, n_samples: int = 1, latent_vectors: Optional[List[List[float]]] = None) -> List[Ket]:
        """
        Generate quantum states using the trained generator.

        Args:
            n_samples: Number of states to generate (default: 1).
            latent_vectors: Optional list of latent vectors. If not provided,
                           random vectors are sampled uniformly from [-1, 1].

        Returns:
            List of generated quantum states.

        Examples:
            >>> qgan = QGAN(2, 2)
            >>> # Train first...
            >>> states = qgan.generate(n_samples=5)
            >>> for s in states:
            ...     print(s)
        """
        if latent_vectors is None:
            latent_vectors = [[random.uniform(-1, 1) for _ in range(self.n_latent)] 
                             for _ in range(n_samples)]
        
        return [self.generator.generate(z) for z in latent_vectors]


__all__ = [
    'QGANResult',
    'QuantumGenerator',
    'QuantumDiscriminator',
    'ClassicalDiscriminator',
    'QGAN',
]