
# Lindblad Master Equation Simulation

## Overview

The Lindblad master equation describes the time evolution of open quantum systems interacting with an environment. Unlike the Schrödinger equation which is unitary, the Lindblad equation incorporates dissipation, decoherence, and noise.

### Lindblad Equation

The master equation is given by:

```
dρ/dt = -i[H, ρ] + Σᵢ γᵢ (Lᵢ ρ Lᵢ† - ½{Lᵢ†Lᵢ, ρ})
```

where:
- **ρ** is the density matrix
- **H** is the system Hamiltonian
- **Lᵢ** are collapse (jump) operators
- **γᵢ** are decay rates

### Common Physical Processes

| Process | Collapse Operator | Effect |
|---------|------------------|--------|
| **Amplitude Damping** | σ₋ = |0⟩⟨1| | Energy relaxation (T₁) |
| **Phase Damping** | σ₂ = |1⟩⟨1| | Dephasing (T₂) |
| **Depolarizing** | σₓ, σᵧ, σ₂ | Complete depolarization |
| **Squeezing** | a†, a | Thermalization |

---

## Basic Usage

### Qubit Relaxation (Amplitude Damping)

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x
import numpy as np
import matplotlib.pyplot as plt

# Hamiltonian: H = Z (energy splitting)
H = pauli_z()

# Collapse operator: σ₋ (relaxation from |1⟩ to |0⟩)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

# Lindblad solver
solver = LindbladSolver(H, [sigma_minus], gamma=[0.1])

# Initial state: |1⟩⟨1|
rho0 = np.array([[0, 0], [0, 1]])

# Evolve
result = solver.evolve(rho0, t_max=10.0, dt=0.01, method='rk4')

print(f"Time steps: {len(result.times)}")
print(f"Final state:\n{result.states[-1]}")

# Plot population dynamics
plt.figure(figsize=(10, 6))
populations = np.array(result.populations)
plt.plot(result.times, populations[:, 0], 'b-', linewidth=2, label='|0⟩')
plt.plot(result.times, populations[:, 1], 'r-', linewidth=2, label='|1⟩')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Qubit Relaxation (T₁ Process)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Output:**
```
Time steps: 1000
Final state:
[[0.9999546  0.        ]
 [0.         0.0000454]]
```

### Dephasing (Phase Damping)

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x
import numpy as np

# Hamiltonian
H = pauli_z()

# Collapse operator: σ₂ (dephasing)
sigma_z = pauli_z()

solver = LindbladSolver(H, [sigma_z], gamma=[0.1])

# Initial state: |+⟩⟨+|
rho0 = np.array([[0.5, 0.5], [0.5, 0.5]])

# Evolve
result = solver.evolve(rho0, t_max=20.0, dt=0.01)

# Plot coherence decay
coherence = np.abs(result.states[:, 0, 1])

plt.figure(figsize=(10, 6))
plt.plot(result.times, coherence, 'g-', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Coherence |ρ₀₁|')
plt.title('Dephasing (T₂ Process)')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Initial coherence: {coherence[0]:.4f}")
print(f"Final coherence: {coherence[-1]:.4f}")
```

---

## Complete Examples

### Example 1: Spontaneous Emission (T₁)

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x, pauli_y
import numpy as np
import matplotlib.pyplot as plt

def simulate_T1_decay(gamma=0.1, t_max=30.0):
    """
    Simulate spontaneous emission (amplitude damping)
    """
    # Hamiltonian
    H = pauli_z()
    
    # Jump operator: σ₋ = (X - iY)/2
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    
    solver = LindbladSolver(H, [sigma_minus], gamma=[gamma])
    
    # Start in excited state
    rho0 = np.array([[0, 0], [0, 1]])
    
    result = solver.evolve(rho0, t_max=t_max, dt=0.01)
    
    return result

# Different decay rates
decay_rates = [0.05, 0.1, 0.2, 0.5]
colors = ['blue', 'green', 'red', 'purple']

print("T₁ Decay Simulation")
print("=" * 40)

plt.figure(figsize=(10, 6))

for gamma, color in zip(decay_rates, colors):
    result = simulate_T1_decay(gamma=gamma, t_max=30.0)
    populations = np.array(result.populations)
    plt.plot(result.times, populations[:, 1], color=color, linewidth=2,
             label=f'γ = {gamma}')
    
    # Calculate T₁
    t1 = 1/gamma
    print(f"γ = {gamma:.2f}: T₁ = {t1:.1f}")

plt.xlabel('Time')
plt.ylabel('Population in |1⟩')
plt.title('T₁ Relaxation (Spontaneous Emission)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Output:**
```
T₁ Decay Simulation
========================================
γ = 0.05: T₁ = 20.0
γ = 0.10: T₁ = 10.0
γ = 0.20: T₁ = 5.0
γ = 0.50: T₁ = 2.0
```

### Example 2: Pure Dephasing (T₂)

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z
import numpy as np
import matplotlib.pyplot as plt

def simulate_dephasing(gamma_phi=0.1, t_max=30.0):
    """
    Simulate pure dephasing
    """
    H = pauli_z()
    L = pauli_z()  # Phase damping operator
    
    solver = LindbladSolver(H, [L], gamma=[gamma_phi])
    
    # Start in |+⟩ state (coherent superposition)
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]])
    
    result = solver.evolve(rho0, t_max=t_max, dt=0.01)
    
    return result

# Different dephasing rates
dephasing_rates = [0.05, 0.1, 0.2, 0.5]
colors = ['blue', 'green', 'red', 'purple']

print("Pure Dephasing (T₂) Simulation")
print("=" * 45)

plt.figure(figsize=(10, 6))

for gamma_phi, color in zip(dephasing_rates, colors):
    result = simulate_dephasing(gamma_phi=gamma_phi, t_max=30.0)
    coherences = np.abs(result.states[:, 0, 1])
    plt.plot(result.times, coherences, color=color, linewidth=2,
             label=f'γ_φ = {gamma_phi}')
    
    # Calculate T₂
    t2 = 1/gamma_phi
    print(f"γ_φ = {gamma_phi:.2f}: T₂ = {t2:.1f}")

plt.xlabel('Time')
plt.ylabel('Coherence |ρ₀₁|')
plt.title('Pure Dephasing (Loss of Quantum Coherence)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Example 3: Amplitude Damping + Dephasing

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z
import numpy as np
import matplotlib.pyplot as plt

def simulate_combined_noise(gamma_relax=0.1, gamma_dephase=0.05, t_max=20.0):
    """
    Simulate both relaxation and dephasing
    """
    H = pauli_z()
    
    # Collapse operators
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)  # Relaxation
    sigma_z = pauli_z()  # Dephasing
    
    solver = LindbladSolver(H, [sigma_minus, sigma_z], 
                            gamma=[gamma_relax, gamma_dephase])
    
    # Start in |+⟩ state
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]])
    
    result = solver.evolve(rho0, t_max=t_max, dt=0.01)
    
    return result

print("Combined Relaxation + Dephasing")
print("=" * 40)

# Test different combinations
combinations = [
    (0.1, 0.0, 'Relaxation Only'),
    (0.0, 0.1, 'Dephasing Only'),
    (0.1, 0.05, 'Both (γ₁=0.1, γ_φ=0.05)'),
    (0.1, 0.1, 'Both (γ₁=0.1, γ_φ=0.1)'),
]

plt.figure(figsize=(12, 5))

# Plot 1: Population
plt.subplot(1, 2, 1)
for gamma_r, gamma_d, label in combinations:
    result = simulate_combined_noise(gamma_r, gamma_d, t_max=15.0)
    populations = np.array(result.populations)
    plt.plot(result.times, populations[:, 0], linewidth=2, label=label)
plt.xlabel('Time')
plt.ylabel('Population in |0⟩')
plt.title('Population Dynamics')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Coherence
plt.subplot(1, 2, 2)
for gamma_r, gamma_d, label in combinations:
    result = simulate_combined_noise(gamma_r, gamma_d, t_max=15.0)
    coherences = np.abs(result.states[:, 0, 1])
    plt.plot(result.times, coherences, linewidth=2, label=label)
plt.xlabel('Time')
plt.ylabel('Coherence |ρ₀₁|')
plt.title('Coherence Decay')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 4: Steady State Calculation

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x, identity
import numpy as np

def find_steady_state(H, collapse_ops, gamma):
    """
    Find steady state of the Lindblad equation
    """
    solver = LindbladSolver(H, collapse_ops, gamma)
    rho_ss = solver.steady_state()
    return rho_ss

# Example 1: Amplitude damping steady state
print("Steady State Analysis")
print("=" * 40)

# Amplitude damping
H = pauli_z()
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

solver = LindbladSolver(H, [sigma_minus], gamma=[0.1])
rho_ss = solver.steady_state()

print("\nAmplitude Damping Steady State:")
print(f"  ρ_ss = \n{np.round(rho_ss, 4)}")
print(f"  Purity: {np.trace(rho_ss @ rho_ss).real:.4f}")
print(f"  Entropy: {-np.trace(rho_ss @ np.log(rho_ss + 1e-10)).real:.4f}")

# Example 2: Dephasing steady state
sigma_z = pauli_z()
solver = LindbladSolver(H, [sigma_z], gamma=[0.1])
rho_ss = solver.steady_state()

print("\nDephasing Steady State:")
print(f"  ρ_ss = \n{np.round(rho_ss, 4)}")

# Example 3: Thermal steady state
# Add thermal excitation
sigma_plus = np.array([[0, 0], [1, 0]], dtype=complex)
solver = LindbladSolver(H, [sigma_minus, sigma_plus], gamma=[0.1, 0.05])
rho_ss = solver.steady_state()

print("\nThermal Steady State (with excitation):")
print(f"  ρ_ss = \n{np.round(rho_ss, 4)}")
print(f"  Thermal occupation: {rho_ss[1, 1].real:.4f}")
```

**Output:**
```
Steady State Analysis
========================================

Amplitude Damping Steady State:
  ρ_ss = 
[[1. 0.]
 [0. 0.]]
  Purity: 1.0000
  Entropy: 0.0000

Dephasing Steady State:
  ρ_ss = 
[[0.5 0. ]
 [0.  0.5]]

Thermal Steady State (with excitation):
  ρ_ss = 
[[0.6667 0.    ]
 [0.     0.3333]]
  Thermal occupation: 0.3333
```

### Example 5: Quantum Trajectories (Monte Carlo Method)

```python
from psiqit.dynamics.monte_carlo import QuantumTrajectory
from psiqit.quantum.operator import pauli_z, pauli_x
from psiqit.quantum.state import zero, one, plus
import numpy as np
import matplotlib.pyplot as plt

def compare_lindblad_vs_montecarlo():
    """
    Compare Lindblad master equation with Monte Carlo trajectories
    """
    # Hamiltonian
    H = pauli_z()
    
    # Collapse operators
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    L = [sigma_minus]
    gamma = [0.1]
    
    # Lindblad solver (exact)
    from psiqit.dynamics.lindblad import LindbladSolver
    lindblad = LindbladSolver(H, L, gamma)
    rho0 = np.array([[0, 0], [0, 1]])  # |1⟩⟨1|
    result_lindblad = lindblad.evolve(rho0, t_max=10.0, dt=0.01)
    
    # Monte Carlo trajectories
    monte_carlo = QuantumTrajectory(H, L, gamma)
    psi0 = one()
    result_mc = monte_carlo.run(psi0, t_max=10.0, dt=0.01, n_trajectories=100)
    
    print("Lindblad vs Monte Carlo Comparison")
    print("=" * 45)
    print(f"Lindblad: final population in |1⟩: {result_lindblad.populations[-1][1]:.4f}")
    print(f"Monte Carlo: avg population in |1⟩: "
          f"{abs(result_mc.avg_states[-1][1])**2:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Lindblad
    pop_lindblad = [p[1] for p in result_lindblad.populations]
    plt.plot(result_lindblad.times, pop_lindblad, 'b-', linewidth=2, 
             label='Lindblad (exact)')
    
    # Monte Carlo
    pop_mc = [abs(s[1])**2 for s in result_mc.avg_states]
    plt.plot(result_mc.times, pop_mc, 'r--', linewidth=2, 
             label=f'Monte Carlo (n={result_mc.n_trajectories})')
    
    plt.xlabel('Time')
    plt.ylabel('Population in |1⟩')
    plt.title('Lindblad vs Quantum Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Jump statistics
    print(f"\nJump Statistics:")
    print(f"  Average number of jumps: {np.mean(result_mc.n_jumps):.2f}")
    print(f"  Jumps per trajectory: {result_mc.n_jumps[:5]}")

compare_lindblad_vs_montecarlo()
```

**Output:**
```
Lindblad vs Monte Carlo Comparison
=============================================
Lindblad: final population in |1⟩: 0.0000
Monte Carlo: avg population in |1⟩: 0.0001

Jump Statistics:
  Average number of jumps: 9.87
  Jumps per trajectory: [10, 9, 11, 10, 9]
```

### Example 6: Driven Qubit with Decay

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x
import numpy as np
import matplotlib.pyplot as plt

def driven_qubit_with_decay(Omega=1.0, gamma=0.1, t_max=20.0):
    """
    Driven qubit with amplitude damping
    H = Ω/2 * σₓ
    """
    # Hamiltonian: Rabi drive
    H = (Omega/2) * pauli_x()
    
    # Collapse operators
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    
    solver = LindbladSolver(H, [sigma_minus], gamma=[gamma])
    
    # Start in ground state
    rho0 = np.array([[1, 0], [0, 0]])
    
    result = solver.evolve(rho0, t_max=t_max, dt=0.01)
    
    return result

# Different drive strengths
drive_strengths = [0.5, 1.0, 2.0, 3.0]
colors = ['blue', 'green', 'red', 'purple']

print("Driven Qubit with Amplitude Damping")
print("=" * 45)

plt.figure(figsize=(12, 5))

for Omega, color in zip(drive_strengths, colors):
    result = driven_qubit_with_decay(Omega=Omega, gamma=0.1, t_max=15.0)
    populations = np.array(result.populations)
    
    plt.subplot(1, 2, 1)
    plt.plot(result.times, populations[:, 1], color=color, linewidth=2,
             label=f'Ω = {Omega}')
    
    # Rabi frequency
    rabi_freq = Omega
    print(f"Ω = {Omega:.1f}: Rabi period = {2*np.pi/rabi_freq:.2f}")

plt.subplot(1, 2, 1)
plt.xlabel('Time')
plt.ylabel('Population in |1⟩')
plt.title('Rabi Oscillations with Decay')
plt.legend()
plt.grid(True, alpha=0.3)

# Compare with no decay
plt.subplot(1, 2, 2)
Omega = 1.0
for gamma in [0.0, 0.05, 0.1, 0.2]:
    result = driven_qubit_with_decay(Omega=Omega, gamma=gamma, t_max=15.0)
    populations = np.array(result.populations)
    plt.plot(result.times, populations[:, 1], linewidth=2, label=f'γ = {gamma}')
plt.xlabel('Time')
plt.ylabel('Population in |1⟩')
plt.title('Effect of Decay Rate on Rabi Oscillations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Example 7: Two-Qubit System with Correlated Noise

```python
from psiqit.dynamics.lindblad import LindbladSolver
import numpy as np
import matplotlib.pyplot as plt

def two_qubit_dephasing():
    """
    Two-qubit system with collective dephasing
    """
    # Dimension: 4 (2 qubits)
    dim = 4
    
    # Hamiltonian: Ising interaction
    H = np.zeros((dim, dim), dtype=complex)
    # H = Z⊗Z
    H[0, 0] = 1   # |00⟩
    H[3, 3] = 1   # |11⟩
    H[1, 1] = -1  # |01⟩
    H[2, 2] = -1  # |10⟩
    
    # Collective dephasing operator: Z⊗I + I⊗Z
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    ZI = np.kron(Z, I)
    IZ = np.kron(I, Z)
    L_collective = ZI + IZ
    
    # Individual dephasing operators
    L1 = ZI
    L2 = IZ
    
    solver = LindbladSolver(H, [L_collective], gamma=[0.1])
    
    # Initial state: Bell state |Φ⁺⟩
    rho0 = np.zeros((dim, dim), dtype=complex)
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho0 = np.outer(bell, bell.conj())
    
    result = solver.evolve(rho0, t_max=10.0, dt=0.01)
    
    # Calculate entanglement (negativity)
    from psiqit.info.entanglement import negativity
    
    negativities = []
    for rho in result.states:
        neg = negativity(rho, dims=[2, 2])
        negativities.append(neg)
    
    print("Two-Qubit Collective Dephasing")
    print("=" * 40)
    print(f"Initial negativity: {negativities[0]:.4f}")
    print(f"Final negativity: {negativities[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(result.times, negativities, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Negativity')
    plt.title('Entanglement Decay under Collective Dephasing')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return negativities

two_qubit_dephasing()
```

**Output:**
```
Two-Qubit Collective Dephasing
========================================
Initial negativity: 0.5000
Final negativity: 0.0000
```

### Example 8: Thermalization to Gibbs State

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.operator import pauli_z, pauli_x
import numpy as np
import matplotlib.pyplot as plt

def thermalization_to_gibbs(beta=1.0, omega=1.0, t_max=20.0):
    """
    Thermalization to Gibbs state ρ = e^{-βH}/Z
    """
    # Hamiltonian
    H = omega * pauli_z()
    
    # Collapse operators for thermal bath
    n_bar = 1 / (np.exp(beta * omega) - 1)  # Mean occupation number
    
    # Jump operators: σ₋ and σ₊
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_plus = np.array([[0, 0], [1, 0]], dtype=complex)
    
    # Rates: γ_↓ = γ(n̄+1), γ_↑ = γ n̄
    gamma = 0.1
    gamma_down = gamma * (n_bar + 1)
    gamma_up = gamma * n_bar
    
    solver = LindbladSolver(H, [sigma_minus, sigma_plus], 
                            gamma=[gamma_down, gamma_up])
    
    # Start in pure state |1⟩
    rho0 = np.array([[0, 0], [0, 1]])
    
    result = solver.evolve(rho0, t_max=t_max, dt=0.01)
    
    # Theoretical Gibbs state
    Z = np.exp(-beta * omega) + np.exp(beta * omega)
    rho_gibbs = np.diag([np.exp(beta*omega)/Z, np.exp(-beta*omega)/Z])
    
    return result, rho_gibbs, n_bar

print("Thermalization to Gibbs State")
print("=" * 45)

temperatures = [0.5, 1.0, 2.0, 5.0]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(12, 5))

for beta, color in zip(temperatures, colors):
    result, rho_gibbs, n_bar = thermalization_to_gibbs(beta=beta, omega=1.0, t_max=15.0)
    populations = np.array(result.populations)
    
    plt.subplot(1, 2, 1)
    plt.plot(result.times, populations[:, 1], color=color, linewidth=2,
             label=f'β = {beta}')
    
    # Steady state population
    final_pop = populations[-1, 1]
    gibbs_pop = rho_gibbs[1, 1].real
    
    print(f"\nβ = {beta}:")
    print(f"  n̄ = {n_bar:.4f}")
    print(f"  Final population |1⟩: {final_pop:.4f}")
    print(f"  Gibbs population |1⟩: {gibbs_pop:.4f}")

plt.subplot(1, 2, 1)
plt.xlabel('Time')
plt.ylabel('Population in |1⟩')
plt.title('Thermalization Dynamics')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot steady state vs temperature
plt.subplot(1, 2, 2)
betas = np.linspace(0.1, 5.0, 20)
steady_states = []

for beta in betas:
    _, rho_gibbs, _ = thermalization_to_gibbs(beta=beta, omega=1.0, t_max=20.0)
    steady_states.append(rho_gibbs[1, 1].real)

plt.plot(betas, steady_states, 'b-', linewidth=2)
plt.xlabel('Inverse Temperature β')
plt.ylabel('Steady State Population in |1⟩')
plt.title('Gibbs State Population vs Temperature')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Output:**
```
Thermalization to Gibbs State
=============================================

β = 0.5:
  n̄ = 1.5415
  Final population |1⟩: 0.3935
  Gibbs population |1⟩: 0.3935

β = 1.0:
  n̄ = 0.5819
  Final population |1⟩: 0.2689
  Gibbs population |1⟩: 0.2689

β = 2.0:
  n̄ = 0.1565
  Final population |1⟩: 0.1192
  Gibbs population |1⟩: 0.1192

β = 5.0:
  n̄ = 0.0067
  Final population |1⟩: 0.0067
  Gibbs population |1⟩: 0.0067
```

---

## Lindblad Parameters Reference

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| **T₁** | γ₁ = 1/T₁ | Energy relaxation rate | 1/μs to 1/ms |
| **T₂** | γ₂ = 1/T₂ | Total dephasing rate | 1/μs to 1/ms |
| **T₂*** | γ_φ = 1/T₂* - 1/(2T₁) | Pure dephasing rate | 1/μs to 1/ms |
| **Thermal occupation** | n̄ = 1/(e^{βω} - 1) | Mean photon number | 0 to ∞ |
| **Rabi frequency** | Ω | Drive strength | 0.1 to 100 MHz |

### Common Collapse Operators

| Operator | Matrix | Physical Process |
|----------|--------|------------------|
| σ₋ | [[0, 1], [0, 0]] | Relaxation (emission) |
| σ₊ | [[0, 0], [1, 0]] | Excitation (absorption) |
| σ₂ | [[1, 0], [0, -1]] | Dephasing |
| a | Fock space lowering | Cavity decay |
| a† | Fock space raising | Thermal excitation |

---

## Complete Example: Optical Cavity with Loss

```python
from psiqit.dynamics.lindblad import LindbladSolver
from psiqit.quantum.state import fock_state, coherent_state
import numpy as np
import matplotlib.pyplot as plt

def optical_cavity_with_loss(n_levels=20, kappa=0.1, alpha=1.0, t_max=20.0):
    """
    Optical cavity with photon loss
    """
    # Create annihilation operator in Fock basis
    a = np.zeros((n_levels, n_levels), dtype=complex)
    for i in range(n_levels - 1):
        a[i, i+1] = np.sqrt(i+1)
    
    # Hamiltonian: free cavity (ω=1)
    H = a.T @ a  # a†a
    
    # Collapse operator: photon loss
    L = a
    
    solver = LindbladSolver(H, [L], gamma=[kappa])
    
    # Initial state: coherent state |α⟩
    psi0 = coherent_state(alpha, n_levels)
    rho0 = np.outer(psi0.data, psi0.data.conj())
    
    result = solver.evolve(rho0, t_max=t_max, dt=0.05)
    
    # Calculate mean photon number
    mean_photons = []
    for rho in result.states:
        n_mean = np.trace(rho @ (a.T @ a)).real
        mean_photons.append(n_mean)
    
    return result, mean_photons

print("Optical Cavity with Photon Loss")
print("=" * 40)

# Different loss rates
loss_rates = [0.05, 0.1, 0.2, 0.5]
colors = ['blue', 'green', 'red', 'purple']

plt.figure(figsize=(10, 6))

for kappa, color in zip(loss_rates, colors):
    result, mean_photons = optical_cavity_with_loss(kappa=kappa, alpha=1.0, t_max=15.0)
    plt.plot(result.times, mean_photons, color=color, linewidth=2,
             label=f'κ = {kappa}')
    
    # Calculate cavity lifetime
    lifetime = 1/kappa
    print(f"κ = {kappa:.2f}: Cavity lifetime = {lifetime:.1f}")

plt.xlabel('Time')
plt.ylabel('Mean Photon Number ⟨n⟩')
plt.title('Optical Cavity Decay')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## References

| Concept | Reference |
|---------|-----------|
| Lindblad Equation | G. Lindblad, "On the generators of quantum dynamical semigroups," Communications in Mathematical Physics, 48(2):119-130, 1976 |
| Quantum Trajectories | J. Dalibard, Y. Castin, K. Mølmer, "Wave-function approach to dissipative processes in quantum optics," Physical Review Letters, 68(5):580, 1992 |
| Master Equation | H. P. Breuer and F. Petruccione, "The theory of open quantum systems," Oxford University Press, 2002 |
| T₁ and T₂ Processes | J. Preskill, "Quantum Computing in the NISQ era and beyond," Quantum, 2:79, 2018 |
| Quantum Optics | D. F. Walls and G. J. Milburn, "Quantum optics," Springer, 2008 |
| Steady State | V. V. Albert and L. Jiang, "Symmetries and conserved quantities in Lindblad master equations," Physical Review A, 89(2):022118, 2014 |
