
# Dynamics API

## Module: `psiqit.dynamics`

This module provides tools for simulating quantum dynamics, including solving the Schrödinger equation (time-dependent and time-independent), Lindblad master equation for open quantum systems, Monte Carlo wave function method (quantum trajectories), adiabatic evolution, and time evolution methods (Trotter, Chebyshev, Krylov).

---

## Schrödinger Equation (`schrodinger.py`)

**Time-dependent and time-independent Schrödinger equation solvers** - Numerical solutions for 1D quantum systems using finite difference and split-operator methods.

### WaveFunction Class

Represents a quantum wave function ψ(x,t) with methods for calculating expectation values and uncertainties.

| Method | Description |
|--------|-------------|
| `probability_density()` | Return \|ψ\|² probability density |
| `normalize()` | Normalize wave function to unit norm |
| `expectation_x()` | Compute ⟨x⟩ = ∫ ψ* x ψ dx |
| `expectation_p(hbar)` | Compute ⟨p⟩ = ∫ ψ* (-iℏ ∂/∂x) ψ dx |
| `uncertainty_x()` | Compute Δx = √(⟨x²⟩ - ⟨x⟩²) |
| `uncertainty_p(hbar)` | Compute Δp = √(⟨p²⟩ - ⟨p⟩²) |

### Functions

| Function | Description |
|----------|-------------|
| `solve_time_independent(potential, x_range, n_points, n_states)` | Solve TISE: Hψ = Eψ |
| `solve_time_dependent(psi0, x, potential, t_max, dt, method)` | Solve TDSE: iℏ ∂ψ/∂t = Hψ |
| `time_evolution_operator(H, t)` | Compute U(t) = exp(-iHt/ℏ) |
| `propagate_state(psi0, H, t)` | Propagate state using U(t) |
| `expectation_value(psi, operator)` | Compute ⟨ψ\|A\|ψ⟩ |
| `uncertainty_relation(psi, x_op, p_op, hbar)` | Compute Δx·Δp (Heisenberg uncertainty) |

### Utility Functions

| Function | Description |
|----------|-------------|
| `gaussian_wavepacket(x, x0, sigma, k0)` | Create Gaussian wave packet |
| `plane_wave(x, k)` | Create plane wave e^{ikx} |
| `square_well_ground_state(x, L)` | Infinite square well ground state |
| `harmonic_oscillator_state(x, n, mass, omega, hbar)` | HO eigenstate ψ_n(x) |

### Example 1: Harmonic Oscillator Eigenstates

```python
from psiqit.dynamics.schrodinger import solve_time_independent
import numpy as np

# Harmonic oscillator potential
V = lambda x: 0.5 * x**2

# Solve for first 5 eigenstates
energies, wavefunctions = solve_time_independent(
    V, x_range=(-5, 5), n_points=1000, n_states=5
)

print(f"Energies: {energies}")
# Energies: [0.5, 1.5, 2.5, 3.5, 4.5]

# Ground state wavefunction
psi_0 = wavefunctions[0]
```

### Example 2: Gaussian Wavepacket Dynamics

```python
from psiqit.dynamics.schrodinger import (
    gaussian_wavepacket, solve_time_dependent
)
import numpy as np

# Create grid and initial wavepacket
x = np.linspace(-10, 10, 500)
psi0 = gaussian_wavepacket(x, x0=-3, sigma=0.5, k0=5.0)

# Free particle (V=0) evolution
V = lambda x: 0
results = solve_time_dependent(psi0, x, V, t_max=5.0, dt=0.01)

print(f"Number of time steps: {len(results)}")
print(f"Final position expectation: {results[-1].expectation_x():.4f}")
```

### Example 3: Expectation Values and Uncertainty

```python
from psiqit.dynamics.schrodinger import (
    gaussian_wavepacket, expectation_value, uncertainty_relation
)
import numpy as np

x = np.linspace(-10, 10, 500)
psi = gaussian_wavepacket(x, x0=0, sigma=0.5, k0=2.0)

# Create position and momentum operators (finite difference representation)
X_op = np.diag(x)
dx = x[1] - x[0]
n = len(x)
P_op = -1j * (np.eye(n, k=1) - np.eye(n, k=-1)) / (2 * dx)

# Calculate uncertainty product
dx_val, dp_val, product = uncertainty_relation(psi, X_op, P_op)
print(f"Δx = {dx_val:.4f}, Δp = {dp_val:.4f}")
print(f"Δx·Δp = {product:.4f} ≥ ℏ/2 = {0.5:.4f}")
```

---

## Lindblad Master Equation (`lindblad.py`)

**Open quantum system dynamics** - Simulates time evolution of density matrices under the Lindblad master equation: dρ/dt = -i[H, ρ] + Σ γᵢ (Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ}).

### Classes

| Class | Description |
|-------|-------------|
| `LindbladResult` | Result container with times, states, populations |
| `LindbladSolver` | Lindblad equation solver (Euler and RK4 methods) |

### Methods

| Method | Description |
|--------|-------------|
| `evolve(rho0, t_max, dt, method)` | Evolve density matrix over time |
| `steady_state(tol, max_iter)` | Find steady state (dρ/dt = 0) |

### Example 1: Qubit Relaxation (Amplitude Damping)

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x, identity
import numpy as np

# Hamiltonian: H = Z
H = pauli_z()

# Collapse operator: relaxation (σ₋ = (X - iY)/2)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

solver = LindbladSolver(H, [sigma_minus], gamma=[0.1])

# Initial state: |1⟩⟨1|
rho0 = np.array([[0, 0], [0, 1]])

# Evolve
result = solver.evolve(rho0, t_max=10.0, dt=0.01, method='rk4')

# Plot population dynamics
import matplotlib.pyplot as plt
plt.plot(result.times, [p[1] for p in result.populations])
plt.xlabel('Time')
plt.ylabel('Population in |1⟩')
plt.title('Qubit Relaxation (T1)')
plt.show()
```

### Example 2: Find Steady State

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x
import numpy as np

H = pauli_z()
L = pauli_x()  # Dephasing

solver = LindbladSolver(H, [L], gamma=[0.1])
rho_ss = solver.steady_state()

print(f"Steady state:\n{rho_ss}")
# Should approach maximally mixed state I/2
```

---

## Monte Carlo Wave Function (`monte_carlo.py`)

**Quantum trajectories for open systems** - Simulates stochastic wave function evolution with quantum jumps. Alternative to Lindblad master equation with O(N²) memory scaling.

### Classes

| Class | Description |
|-------|-------------|
| `MonteCarloResult` | Result with avg_states, jump_times, n_jumps |
| `QuantumTrajectory` | Monte Carlo wave function simulator |

### Methods

| Method | Description |
|--------|-------------|
| `single_trajectory(psi0, t_max, dt)` | Simulate one quantum trajectory |
| `run(psi0, t_max, dt, n_trajectories)` | Run multiple trajectories and average |

### Example

```python
from psiqit.dynamics.monte_carlo import QuantumTrajectory
from psiqit.quantum.operator import pauli_z, pauli_x
from psiqit.quantum.state import zero
import numpy as np

H = pauli_z()
L = pauli_x()  # Jump operator

traj = QuantumTrajectory(H, [L], gamma=[0.1])
psi0 = zero()

# Run 100 trajectories
results = traj.run(psi0, t_max=10.0, dt=0.01, n_trajectories=100)

print(f"Average number of jumps: {np.mean(results.n_jumps):.2f}")
print(f"Jump times: {results.jump_times[0][:5]}")  # First 5 jumps of first trajectory
```

---

## Adiabatic Evolution (`adiabatic.py`)

**Adiabatic quantum computing and quantum annealing** - Simulates slow evolution from initial to final Hamiltonian, remaining in the instantaneous ground state.

### Classes

| Class | Description |
|-------|-------------|
| `AdiabaticResult` | Result with times, states, energies, fidelities |
| `AdiabaticEvolution` | Adiabatic evolution: H(s) = (1-s)H_i + s H_f |
| `QuantumAnnealing` | Quantum annealing for optimization problems |

### Methods

| Method | Description |
|--------|-------------|
| `evolve(psi0)` | Perform adiabatic evolution |

### Example

```python
from psiqit.dynamics.adiabatic import AdiabaticEvolution, QuantumAnnealing
from psiqit.quantum.operator import pauli_x, pauli_z
from psiqit.quantum.state import zero

# Adiabatic evolution from X to Z
H_i = pauli_x()
H_f = pauli_z()
evol = AdiabaticEvolution(H_i, H_f, T=10.0, n_steps=100)

psi0 = zero()  # Starting state
result = evol.evolve(psi0)

print(f"Final energy: {result.energies[-1]:.6f}")
print(f"Number of time steps: {len(result.times)}")
```

---

## Heisenberg Picture (`heisenberg.py`)

**Time evolution of operators** - Evolves operators in the Heisenberg picture: A(t) = e^{iHt} A e^{-iHt}.

### Classes

| Class | Description |
|-------|-------------|
| `HeisenbergResult` | Result with times, operators, success |
| `HeisenbergEvolution` | Evolve operators using matrix exponentiation or series |
| `HeisenbergEquation` | Compute dA/dt = i[H, A] + ∂A/∂t |

### Methods

| Method | Description |
|--------|-------------|
| `evolve_operator(A, t, method, order)` | Evolve operator using specified method |
| `equation_of_motion(A)` | Compute Heisenberg equation of motion |

### Example

```python
from psiqit.dynamics.heisenberg import HeisenbergEvolution
from psiqit.quantum.operator import pauli_x, pauli_z

H = pauli_z()
heisenberg = HeisenbergEvolution(H)

# Evolve Pauli X operator
X_t = heisenberg.evolve_operator(pauli_x(), t=0.5, method='matrix')
print(f"X(0.5) = {X_t}")
```

---

## Interaction Picture (`interaction.py`)

**Interaction picture representation** - For Hamiltonians split as H = H₀ + V, where H₀ is solvable and V is interaction.

### Classes

| Class | Description |
|-------|-------------|
| `InteractionResult` | Result with times, states, success |
| `InteractionPicture` | Interaction picture representation |

### Methods

| Method | Description |
|--------|-------------|
| `interaction_operator(t)` | Compute V_I(t) = e^{iH₀t} V e^{-iH₀t} |
| `evolve_state(psi0, t)` | Evolve state to interaction picture |
| `evolve_operator(A, t)` | Evolve operator to interaction picture |

### Example

```python
from psiqit.dynamics.interaction import InteractionPicture
from psiqit.quantum.operator import pauli_z, pauli_x
from psiqit.quantum.state import zero

H0 = pauli_z()
V = pauli_x()
inter = InteractionPicture(H0, V)

# Compute interaction picture operator
V_I = inter.interaction_operator(t=0.5)
print(V_I)
```

---

## Time Evolution Methods (`time_evolution.py`)

**Numerical methods for time evolution** - Trotter-Suzuki decomposition, Chebyshev polynomial expansion, and Krylov subspace methods.

### Classes

| Class | Description |
|-------|-------------|
| `EvolutionResult` | Result with times, states, fidelities |
| `TrotterEvolution` | Trotter-Suzuki decomposition |
| `ChebyshevEvolution` | Chebyshev polynomial expansion |
| `KrylovEvolution` | Krylov subspace (Lanczos) method |

### Methods

| Method | Description |
|--------|-------------|
| `evolve(psi0, t)` | Evolve state for time t |

### Example

```python
from psiqit.dynamics.time_evolution import TrotterEvolution
from psiqit.quantum.operator import pauli_z
from psiqit.quantum.state import zero

H = pauli_z()
evol = TrotterEvolution(H, n_steps=100, decomposition='first')

psi0 = zero()
psi_t = evol.evolve(psi0, t=1.0)

print(f"Evolved state: {psi_t}")
```

---

## Module Contents

```python
__all__ = [
    # Schrödinger
    'WaveFunction', 'solve_time_independent', 'solve_time_dependent',
    'time_evolution_operator', 'propagate_state', 'expectation_value',
    'uncertainty_relation', 'gaussian_wavepacket', 'plane_wave',
    'square_well_ground_state', 'harmonic_oscillator_state',
    # Lindblad
    'LindbladResult', 'LindbladSolver',
    # Monte Carlo
    'MonteCarloResult', 'QuantumTrajectory',
    # Adiabatic
    'AdiabaticResult', 'AdiabaticEvolution', 'QuantumAnnealing',
    # Heisenberg
    'HeisenbergResult', 'HeisenbergEvolution', 'HeisenbergEquation',
    # Interaction
    'InteractionResult', 'InteractionPicture',
    # Time Evolution
    'EvolutionResult', 'TrotterEvolution', 'ChebyshevEvolution', 'KrylovEvolution',
]
```

---

## Complete Example: Open System Dynamics Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.dynamics.monte_carlo import QuantumTrajectory
from psiqit.quantum.operator import pauli_z, pauli_x
from psiqit.quantum.state import zero

# Parameters
H = pauli_z()
L = pauli_x()
gamma = 0.1
t_max = 10.0

# Lindblad master equation
solver = LindbladSolver(H, [L], gamma=[gamma])
rho0 = np.array([[1, 0], [0, 0]])
result_lindblad = solver.evolve(rho0, t_max=t_max, dt=0.01)

# Monte Carlo trajectories
traj = QuantumTrajectory(H, [L], gamma=[gamma])
psi0 = zero()
result_mc = traj.run(psi0, t_max=t_max, dt=0.01, n_trajectories=100)

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(result_lindblad.times, [p[0] for p in result_lindblad.populations],
         'b-', linewidth=2, label='Lindblad (exact)')

# For Monte Carlo, compute population from averaged states (simplified)
plt.plot(result_mc.times, [abs(s[0])**2 for s in result_mc.avg_states],
         'r--', linewidth=2, label='Monte Carlo (100 trajectories)')

plt.xlabel('Time')
plt.ylabel('Population in |0⟩')
plt.title('Open System Dynamics: Lindblad vs Monte Carlo')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## References

| Method | Reference |
|--------|-----------|
| Schrödinger equation | E. Schrödinger, "Quantisierung als Eigenwertproblem," Annalen der Physik, 1926 |
| Lindblad equation | G. Lindblad, "On the generators of quantum dynamical semigroups," Comm. Math. Phys., 1976 |
| Monte Carlo wave function | J. Dalibard, Y. Castin, K. Mølmer, "Wave-function approach to dissipative processes," Phys. Rev. Lett., 1992 |
| Adiabatic theorem | M. Born and V. Fock, "Beweis des Adiabatensatzes," Zeitschrift für Physik, 1928 |
| Trotter-Suzuki | M. Suzuki, "Generalized Trotter's formula," Comm. Math. Phys., 1976 |
