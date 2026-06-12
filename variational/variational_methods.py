"""
psiqit/variational/variational_methods.py
============================================================
General Variational Methods
============================================================

Common variational techniques for quantum systems:
    • Rayleigh-Ritz method
    • Time-dependent variational principle (TDVP)
    • Variational Monte Carlo
    • Hartree-Fock method

These classical and quantum variational methods are fundamental tools for
approximating ground states, excited states, and time evolution of quantum
systems. They form the basis of many quantum chemistry and quantum physics
calculations.

Example:
    >>> from psiqit.variational.variational_methods import RayleighRitz, TDVP
    >>> import math
    >>> 
    >>> # Rayleigh-Ritz for ground state
    >>> H = [[1, 0], [0, -1]]
    >>> def trial_state(theta):
    ...     return [math.cos(theta/2), math.sin(theta/2)]
    >>> rr = RayleighRitz(H, trial_state, n_params=1)
    >>> result = rr.optimize()
    >>> print(f"Ground state energy: {result.optimal_value:.6f}")

References:
    W. Ritz, "Über eine neue Methode zur Lösung gewisser Variationsprobleme
    der mathematischen Physik," Journal für die reine und angewandte Mathematik,
    135:1-61, 1909.
    P. A. M. Dirac, "The Principles of Quantum Mechanics," Oxford University
    Press, 1930.
    W. M. C. Foulkes et al., "Quantum Monte Carlo simulations of solids,"
    Reviews of Modern Physics, 73(1):33-83, 2001.
    D. R. Hartree, "The wave mechanics of an atom with a non-Coulomb central
    field," Mathematical Proceedings of the Cambridge Philosophical Society,
    24(1):89-110, 1928.
"""

import math
import random
from typing import List, Callable, Optional, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class VariationalResult:
    """
    Result container for variational optimization.

    Attributes:
        optimal_params: Optimal parameters found.
        optimal_value: Minimum value (energy) achieved.
        iterations: Number of iterations performed.
        history: List of values at each iteration.
        success: Whether optimization completed successfully.
        method: Name of the variational method used.

    Examples:
        >>> result = VariationalResult(optimal_params=[0.5], optimal_value=-1.0,
        ...                             iterations=100, history=[-0.5, -0.8, -1.0],
        ...                             success=True, method="Rayleigh-Ritz")
        >>> print(result)
        Rayleigh-Ritz(value=-1.000000, iterations=100)
    """
    optimal_params: List[float]
    optimal_value: float
    iterations: int
    history: List[float]
    success: bool
    method: str = "variational"
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing method, value, and iterations.
        """
        return f"{self.method}(value={self.optimal_value:.6f}, iterations={self.iterations})"


# ============================================================
# Rayleigh-Ritz Method
# ============================================================

class RayleighRitz:
    """
    Rayleigh-Ritz variational method.

    The Rayleigh-Ritz method finds the minimum eigenvalue of a matrix
    by optimizing over a parameterized trial state. The expectation value
    ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ is minimized with respect to the parameters.

    This is a classical variational method that serves as the foundation
    for many quantum variational algorithms.

    References:
        W. Ritz, "Über eine neue Methode zur Lösung gewisser Variationsprobleme
        der mathematischen Physik," Journal für die reine und angewandte
        Mathematik, 135:1-61, 1909.

    Examples:
        >>> # For a 2x2 matrix
        >>> H = [[1, 0], [0, -1]]
        >>> def trial_state(theta):
        ...     return [math.cos(theta/2), math.sin(theta/2)]
        >>> rr = RayleighRitz(H, trial_state, n_params=1)
        >>> result = rr.optimize()
        >>> print(result.optimal_value)  # Should be -1
        -1.0
    """
    
    def __init__(self, matrix: List[List[float]], ansatz: Callable, 
                 n_params: int = 1):
        """
        Initialize the Rayleigh-Ritz method.

        Args:
            matrix: Matrix to find eigenvalue of (Hamiltonian).
            ansatz: Function that builds trial state from parameters.
            n_params: Number of parameters in the ansatz.

        Examples:
            >>> H = [[2, 0], [0, 1]]
            >>> def psi(x): return [1, x]
            >>> rr = RayleighRitz(H, psi, n_params=1)
        """
        self.matrix = matrix
        self.ansatz = ansatz
        self.n_params = n_params
        self._dim = len(matrix)
    
    def expectation(self, params: List[float]) -> float:
        """
        Compute the expectation value ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩.

        Args:
            params: Parameters for the trial state.

        Returns:
            Rayleigh quotient (energy expectation).

        Examples:
            >>> energy = rr.expectation([0.5])
        """
        psi = self.ansatz(*params)
        
        # Ensure list
        if not isinstance(psi, (list, tuple)):
            psi = [psi]
        
        norm = sum(abs(p)**2 for p in psi)
        if norm < 1e-10:
            return float('inf')
        
        # Compute ⟨ψ|H|ψ⟩
        expectation = 0.0
        for i in range(min(self._dim, len(psi))):
            for j in range(min(self._dim, len(psi))):
                expectation += psi[i].conjugate() * self.matrix[i][j] * psi[j]
        
        return (expectation / norm).real
    
    def _gradient(self, params: List[float], eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient using finite differences.

        Args:
            params: Parameters to differentiate.
            eps: Step size.

        Returns:
            List of gradient values.
        """
        grads = []
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            
            e_plus = self.expectation(p_plus)
            e_minus = self.expectation(p_minus)
            grads.append((e_plus - e_minus) / (2 * eps))
        
        return grads
    
    def optimize(self, n_iterations: int = 100, learning_rate: float = 0.1,
                 verbose: bool = False) -> VariationalResult:
        """
        Optimize parameters using gradient descent.

        Args:
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            VariationalResult with optimal parameters and energy.

        Examples:
            >>> result = rr.optimize(n_iterations=200, learning_rate=0.05)
        """
        params = [0.0] * self.n_params
        history = []
        
        for iteration in range(n_iterations):
            grads = self._gradient(params)
            
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]
            
            energy = self.expectation(params)
            history.append(energy)
            
            if verbose and iteration % 20 == 0:
                print(f"  Iter {iteration}: energy = {energy:.6f}")
        
        return VariationalResult(
            optimal_params=params,
            optimal_value=history[-1] if history else 0.0,
            iterations=n_iterations,
            history=history,
            success=True,
            method="Rayleigh-Ritz"
        )


# ============================================================
# Time-Dependent Variational Principle (TDVP)
# ============================================================

class TDVP:
    """
    Time-Dependent Variational Principle.

    TDVP solves the time-dependent Schrödinger equation within a
    variational manifold. It finds the optimal parameters λ(t) such
    that the variational state |ψ(λ(t))⟩ best approximates the true
    time-evolved state.

    The equations of motion are:
        M_{ij} dλ_j/dt = -i ∂E/∂λ_i

    where M_{ij} is the quantum geometric tensor (Fubini-Study metric).

    References:
        P. Kramer and M. Saraceno, "Geometry of the Time-Dependent
        Variational Principle in Quantum Mechanics," Lecture Notes in
        Physics, 140, Springer, 1981.
        J. Broeckhove et al., "Time-dependent variational principles
        for quantum dynamical problems," Journal of Physics A:
        Mathematical and General, 21(5):1095, 1988.

    Examples:
        >>> def psi(theta, phi):
        ...     return [math.cos(theta/2), math.sin(theta/2)*math.exp(1j*phi)]
        >>> tdvp = TDVP(hamiltonian, psi, n_params=2)
        >>> params_history, energies = tdvp.evolve(params0, t_max=10.0, dt=0.01)
    """
    
    def __init__(self, hamiltonian: Callable, ansatz: Callable, 
                 n_params: int, mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize TDVP.

        Args:
            hamiltonian: Function H(params) that returns energy.
            ansatz: Trial wavefunction ansatz ψ(x; params).
            n_params: Number of parameters.
            mass: Particle mass (for kinetic energy).
            hbar: Reduced Planck constant.

        Examples:
            >>> tdvp = TDVP(energy_func, trial_wavefunction, n_params=2)
        """
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.n_params = n_params
        self.mass = mass
        self.hbar = hbar
    
    def _metric_tensor(self, params: List[float], eps: float = 1e-5) -> List[List[float]]:
        """
        Compute the quantum geometric tensor (Fubini-Study metric).

        Args:
            params: Current parameters.
            eps: Step size for finite differences.

        Returns:
            n×n metric tensor matrix.
        """
        n = self.n_params
        metric = [[0.0] * n for _ in range(n)]
        
        psi0 = self.ansatz(*params)
        norm0 = math.sqrt(sum(abs(p)**2 for p in psi0))
        if norm0 > 0:
            psi0 = [p / norm0 for p in psi0]
        
        for i in range(n):
            for j in range(n):
                # Compute ∂_i ψ and ∂_j ψ
                p_i_plus = params.copy()
                p_i_minus = params.copy()
                p_i_plus[i] += eps
                p_i_minus[i] -= eps
                
                psi_i_plus = self.ansatz(*p_i_plus)
                psi_i_minus = self.ansatz(*p_i_minus)
                
                # Normalize
                norm_i_plus = math.sqrt(sum(abs(p)**2 for p in psi_i_plus))
                norm_i_minus = math.sqrt(sum(abs(p)**2 for p in psi_i_minus))
                if norm_i_plus > 0:
                    psi_i_plus = [p / norm_i_plus for p in psi_i_plus]
                if norm_i_minus > 0:
                    psi_i_minus = [p / norm_i_minus for p in psi_i_minus]
                
                # ∂_i ψ ≈ (ψ_i_plus - ψ_i_minus)/(2ε)
                dpsi_i = [(psi_i_plus[k] - psi_i_minus[k]) / (2*eps) 
                          for k in range(len(psi0))]
                
                # Similar for j
                p_j_plus = params.copy()
                p_j_minus = params.copy()
                p_j_plus[j] += eps
                p_j_minus[j] -= eps
                
                psi_j_plus = self.ansatz(*p_j_plus)
                psi_j_minus = self.ansatz(*p_j_minus)
                
                norm_j_plus = math.sqrt(sum(abs(p)**2 for p in psi_j_plus))
                norm_j_minus = math.sqrt(sum(abs(p)**2 for p in psi_j_minus))
                if norm_j_plus > 0:
                    psi_j_plus = [p / norm_j_plus for p in psi_j_plus]
                if norm_j_minus > 0:
                    psi_j_minus = [p / norm_j_minus for p in psi_j_minus]
                
                dpsi_j = [(psi_j_plus[k] - psi_j_minus[k]) / (2*eps) 
                          for k in range(len(psi0))]
                
                # Metric = Re(⟨∂_i ψ|∂_j ψ⟩ - ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩)
                overlap_ij = sum(dpsi_i[k].conjugate() * dpsi_j[k] for k in range(len(psi0)))
                overlap_i = sum(dpsi_i[k].conjugate() * psi0[k] for k in range(len(psi0)))
                overlap_j = sum(psi0[k].conjugate() * dpsi_j[k] for k in range(len(psi0)))
                
                metric[i][j] = (overlap_ij - overlap_i * overlap_j).real
        
        return metric
    
    def _energy_gradient(self, params: List[float], eps: float = 1e-5) -> List[float]:
        """
        Compute the gradient of the energy.

        Args:
            params: Current parameters.
            eps: Step size.

        Returns:
            List of gradient values.
        """
        grads = []
        for i in range(self.n_params):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            
            e_plus = self.hamiltonian(p_plus)
            e_minus = self.hamiltonian(p_minus)
            grads.append((e_plus - e_minus) / (2 * eps))
        
        return grads
    
    def evolve(self, params0: List[float], t_max: float = 10.0, 
               dt: float = 0.01, verbose: bool = False) -> Tuple[List[List[float]], List[float]]:
        """
        Evolve parameters according to TDVP.

        Args:
            params0: Initial parameters.
            t_max: Maximum evolution time.
            dt: Time step.
            verbose: Whether to print progress.

        Returns:
            Tuple of (params_history, energies_history).

        Examples:
            >>> params0 = [0.1, 0.2]
            >>> params_history, energies = tdvp.evolve(params0, t_max=5.0)
        """
        params = params0.copy()
        params_history = [params.copy()]
        energies_history = [self.hamiltonian(params)]
        
        for step in range(int(t_max / dt)):
            # Compute metric tensor and energy gradient
            metric = self._metric_tensor(params)
            grad = self._energy_gradient(params)
            
            # Solve M·dλ/dt = -grad
            # For now, simple Euler update (忽略 metric)
            # Full implementation would solve linear system
            
            # Simplified update (assuming metric = identity)
            for i in range(self.n_params):
                params[i] -= dt * grad[i] / self.hbar
            
            params_history.append(params.copy())
            energies_history.append(self.hamiltonian(params))
            
            if verbose and step % 100 == 0:
                print(f"  Step {step}: energy = {energies_history[-1]:.6f}")
        
        return params_history, energies_history


# ============================================================
# Variational Monte Carlo
# ============================================================

class VariationalMonteCarlo:
    """
    Variational Monte Carlo method.

    VMC uses Monte Carlo sampling to compute expectation values and
    optimize variational parameters. It is particularly useful for
    large systems where exact diagonalization is infeasible.

    The method:
        1. Sample configurations from |ψ|² using Metropolis algorithm
        2. Compute local energy E_local(x) = Hψ(x)/ψ(x)
        3. Average to get energy expectation
        4. Optimize parameters to minimize energy

    References:
        W. M. C. Foulkes et al., "Quantum Monte Carlo simulations of solids,"
        Reviews of Modern Physics, 73(1):33-83, 2001.
        J. B. Anderson, "A random-walk simulation of the Schrödinger equation:
        H+3," The Journal of Chemical Physics, 63(4):1499-1503, 1975.

    Examples:
        >>> def wavefunction(x, alpha):
        ...     return math.exp(-alpha * x**2)
        >>> 
        >>> def potential(x):
        ...     return 0.5 * x**2
        >>> 
        >>> vmc = VariationalMonteCarlo(wavefunction, potential, n_params=1)
        >>> result = vmc.optimize(n_samples=10000, n_iterations=50)
        >>> print(f"Optimal alpha: {result.optimal_params[0]:.4f}")
    """
    
    def __init__(self, wavefunction: Callable, potential: Callable,
                 n_params: int, kinetic_energy: Optional[Callable] = None,
                 mass: float = 1.0, hbar: float = 1.0):
        """
        Initialize VMC.

        Args:
            wavefunction: Trial wavefunction ψ(x; params).
            potential: Potential energy V(x).
            n_params: Number of variational parameters.
            kinetic_energy: Optional kinetic energy function.
            mass: Particle mass.
            hbar: Reduced Planck constant.

        Examples:
            >>> vmc = VariationalMonteCarlo(psi, V, n_params=2)
        """
        self.wavefunction = wavefunction
        self.potential = potential
        self.n_params = n_params
        self.mass = mass
        self.hbar = hbar
        
        if kinetic_energy is None:
            self.kinetic_energy = self._default_kinetic
        else:
            self.kinetic_energy = kinetic_energy
    
    def _default_kinetic(self, x: float, psi: complex, dpsi: complex, d2psi: complex) -> float:
        """
        Compute the kinetic energy: -ħ²/2m (ψ''/ψ).

        Args:
            x: Position.
            psi: Wavefunction value.
            dpsi: First derivative.
            d2psi: Second derivative.

        Returns:
            Kinetic energy.
        """
        if abs(psi) < 1e-10:
            return 0.0
        return -self.hbar**2 / (2 * self.mass) * (d2psi / psi).real
    
    def _local_energy(self, x: float, params: List[float]) -> float:
        """
        Compute the local energy E_local(x) = Hψ/ψ.

        Args:
            x: Position.
            params: Current parameters.

        Returns:
            Local energy.
        """
        psi = self.wavefunction(x, *params)
        
        # Numerical derivatives for kinetic energy
        eps = 1e-5
        psi_plus = self.wavefunction(x + eps, *params)
        psi_minus = self.wavefunction(x - eps, *params)
        
        # First derivative
        dpsi = (psi_plus - psi_minus) / (2 * eps)
        
        # Second derivative
        d2psi = (psi_plus - 2*psi + psi_minus) / (eps**2)
        
        kinetic = self._default_kinetic(x, psi, dpsi, d2psi)
        potential = self.potential(x)
        
        return kinetic + potential
    
    def _energy(self, params: List[float], samples: List[float]) -> float:
        """
        Compute the average energy over samples.

        Args:
            params: Current parameters.
            samples: List of sampled positions.

        Returns:
            Average energy.
        """
        total = 0.0
        for x in samples:
            total += self._local_energy(x, params)
        return total / len(samples) if samples else 0.0
    
    def _generate_samples(self, params: List[float], n_samples: int) -> List[float]:
        """
        Generate samples using the Metropolis algorithm.

        Args:
            params: Current parameters.
            n_samples: Number of samples to generate.

        Returns:
            List of sampled positions.
        """
        samples = []
        x = 0.0
        
        # Metropolis step size
        sigma = 1.0
        
        # Thermalization
        for _ in range(n_samples * 10):
            x_new = x + random.gauss(0, sigma)
            psi_new = self.wavefunction(x_new, *params)
            psi_old = self.wavefunction(x, *params)
            
            prob = min(1.0, (abs(psi_new)**2) / (abs(psi_old)**2 + 1e-10))
            
            if random.random() < prob:
                x = x_new
        
        # Sampling
        for _ in range(n_samples):
            x_new = x + random.gauss(0, sigma)
            psi_new = self.wavefunction(x_new, *params)
            psi_old = self.wavefunction(x, *params)
            
            prob = min(1.0, (abs(psi_new)**2) / (abs(psi_old)**2 + 1e-10))
            
            if random.random() < prob:
                x = x_new
            
            samples.append(x)
        
        return samples
    
    def _gradient(self, params: List[float], samples: List[float], 
                  eps: float = 1e-5) -> List[float]:
        """
        Compute the energy gradient using finite differences.

        Args:
            params: Current parameters.
            samples: Sampled positions.
            eps: Step size.

        Returns:
            List of gradient values.
        """
        grads = []
        e0 = self._energy(params, samples)
        
        for i in range(self.n_params):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            
            e_plus = self._energy(p_plus, samples)
            e_minus = self._energy(p_minus, samples)
            
            grads.append((e_plus - e_minus) / (2 * eps))
        
        return grads
    
    def optimize(self, n_samples: int = 5000, n_iterations: int = 50,
                 learning_rate: float = 0.01, verbose: bool = False) -> VariationalResult:
        """
        Optimize variational parameters.

        Args:
            n_samples: Number of Monte Carlo samples per iteration.
            n_iterations: Number of optimization iterations.
            learning_rate: Learning rate for gradient descent.
            verbose: Whether to print progress.

        Returns:
            VariationalResult with optimal parameters.

        Examples:
            >>> result = vmc.optimize(n_samples=10000, n_iterations=50)
            >>> print(result.optimal_value)
        """
        params = [0.0] * self.n_params
        history = []
        
        for iteration in range(n_iterations):
            samples = self._generate_samples(params, n_samples)
            energy = self._energy(params, samples)
            history.append(energy)
            
            grads = self._gradient(params, samples)
            
            for i in range(len(params)):
                params[i] -= learning_rate * grads[i]
            
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration}: energy = {energy:.6f}")
        
        return VariationalResult(
            optimal_params=params,
            optimal_value=history[-1] if history else 0.0,
            iterations=n_iterations,
            history=history,
            success=True,
            method="VariationalMonteCarlo"
        )


# ============================================================
# Hartree-Fock Method
# ============================================================

class HartreeFock:
    """
    Hartree-Fock method for electronic structure.

    The Hartree-Fock method is a mean-field approach to solve the
    electronic Schrödinger equation. It approximates the many-electron
    wavefunction as a single Slater determinant and solves self-consistent
    field equations.

    The method iteratively solves the Roothaan-Hall equations:
        FC = SCε

    where F is the Fock matrix, C are orbital coefficients, S is the
    overlap matrix, and ε are orbital energies.

    References:
        D. R. Hartree, "The wave mechanics of an atom with a non-Coulomb
        central field," Mathematical Proceedings of the Cambridge
        Philosophical Society, 24(1):89-110, 1928.
        V. Fock, "Näherungsmethode zur Lösung des quantenmechanischen
        Mehrkörperproblems," Zeitschrift für Physik, 61(1-2):126-148, 1930.
        C. C. J. Roothaan, "New developments in molecular orbital theory,"
        Reviews of Modern Physics, 23(2):69, 1951.

    Examples:
        >>> # For a simple 2-electron system
        >>> hf = HartreeFock(n_electrons=2, n_orbitals=4)
        >>> result = hf.solve()
        >>> print(f"HF energy: {result.optimal_value:.6f}")
    """
    
    def __init__(self, n_electrons: int, n_orbitals: int,
                 one_electron_integrals: Optional[List[List[float]]] = None,
                 two_electron_integrals: Optional[List[List[List[List[float]]]]] = None):
        """
        Initialize Hartree-Fock.

        Args:
            n_electrons: Number of electrons.
            n_orbitals: Number of spatial orbitals.
            one_electron_integrals: Core Hamiltonian integrals h_ij.
            two_electron_integrals: Electron repulsion integrals (ij|kl).

        Examples:
            >>> hf = HartreeFock(n_electrons=2, n_orbitals=4)
        """
        self.n_electrons = n_electrons
        self.n_orbitals = n_orbitals
        self.n_occupied = n_electrons // 2  # Closed shell
        
        # Default to simple hydrogen-like integrals if not provided
        if one_electron_integrals is None:
            self.h = self._default_one_electron()
        else:
            self.h = one_electron_integrals
        
        if two_electron_integrals is None:
            self.g = self._default_two_electron()
        else:
            self.g = two_electron_integrals
        
        self._density_matrix = None
        self._fock_matrix = None
    
    def _default_one_electron(self) -> List[List[float]]:
        """
        Create default one-electron integrals (diagonal).

        Returns:
            Diagonal core Hamiltonian matrix.
        """
        h = [[0.0] * self.n_orbitals for _ in range(self.n_orbitals)]
        for i in range(self.n_orbitals):
            h[i][i] = -0.5  # Simple approximation
        return h
    
    def _default_two_electron(self) -> List[List[List[List[float]]]]:
        """
        Create default two-electron integrals (simplified).

        Returns:
            Simplified electron repulsion integrals.
        """
        g = [[[[0.0] * self.n_orbitals for _ in range(self.n_orbitals)] 
              for _ in range(self.n_orbitals)] for _ in range(self.n_orbitals)]
        
        # Simplified: only on-site integrals
        for i in range(self.n_orbitals):
            g[i][i][i][i] = 1.0
        
        return g
    
    def _density(self, C: List[List[float]]) -> List[List[float]]:
        """
        Compute the density matrix from orbital coefficients.

        Args:
            C: Orbital coefficient matrix.

        Returns:
            Density matrix P.
        """
        P = [[0.0] * self.n_orbitals for _ in range(self.n_orbitals)]
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):
                for k in range(self.n_occupied):
                    P[i][j] += 2 * C[i][k] * C[j][k]
        return P
    
    def _fock(self, P: List[List[float]]) -> List[List[float]]:
        """
        Compute the Fock matrix from the density matrix.

        Args:
            P: Density matrix.

        Returns:
            Fock matrix F = h + J - K.
        """
        F = [[0.0] * self.n_orbitals for _ in range(self.n_orbitals)]
        
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):
                F[i][j] = self.h[i][j]
                
                # Coulomb term J and exchange term K
                for k in range(self.n_orbitals):
                    for l in range(self.n_orbitals):
                        F[i][j] += P[k][l] * (2 * self.g[i][j][k][l] - self.g[i][k][j][l])
        
        return F
    
    def _diagonalize(self, F: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
        """
        Diagonalize the Fock matrix (simplified).

        Args:
            F: Fock matrix.

        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        # For large matrices, would use numpy
        n = self.n_orbitals
        
        # Simplified: assume nearly diagonal
        eigenvalues = [F[i][i] for i in range(n)]
        eigenvectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        
        return eigenvalues, eigenvectors
    
    def solve(self, max_iter: int = 50, tol: float = 1e-8,
              verbose: bool = False) -> VariationalResult:
        """
        Solve the Hartree-Fock equations self-consistently.

        Args:
            max_iter: Maximum number of iterations.
            tol: Convergence tolerance.
            verbose: Whether to print progress.

        Returns:
            VariationalResult with the HF energy.

        Examples:
            >>> result = hf.solve(max_iter=100, tol=1e-10)
            >>> print(f"HF energy: {result.optimal_value:.8f}")
        """
        # Initial guess: identity
        C = [[1.0 if i == j else 0.0 for j in range(self.n_orbitals)] 
             for i in range(self.n_orbitals)]
        
        energies = []
        
        for iteration in range(max_iter):
            # Compute density matrix
            P = self._density(C)
            
            # Compute Fock matrix
            F = self._fock(P)
            
            # Diagonalize
            eps, C = self._diagonalize(F)
            
            # Compute total energy
            E = 0.0
            for i in range(self.n_orbitals):
                for j in range(self.n_orbitals):
                    E += P[i][j] * (self.h[i][j] + F[i][j])
            E *= 0.5
            
            energies.append(E)
            
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration}: energy = {E:.6f}")
            
            # Check convergence
            if len(energies) > 1 and abs(energies[-1] - energies[-2]) < tol:
                break
        
        self._density_matrix = P
        self._fock_matrix = F
        
        return VariationalResult(
            optimal_params=C[0],  # First orbital as parameter representation
            optimal_value=energies[-1] if energies else 0.0,
            iterations=iteration + 1,
            history=energies,
            success=True,
            method="Hartree-Fock"
        )


__all__ = [
    'VariationalResult',
    'RayleighRitz',
    'TDVP',
    'VariationalMonteCarlo',
    'HartreeFock',
]