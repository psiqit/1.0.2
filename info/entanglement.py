"""
psiqit/info/entanglement.py
Entanglement Measures - Simplified for Pure States

This module provides various measures and quantification tools for quantum
entanglement, which is a key resource in quantum information processing.

Entanglement measures quantify how much a quantum state is entangled.
For bipartite systems, several measures are implemented:
    - Concurrence: For 2-qubit states (both pure and mixed)
    - Negativity: Based on partial transpose criterion
    - Logarithmic Negativity: Log version of negativity
    - Entanglement Entropy: von Neumann entropy of reduced density matrix
    - Schmidt decomposition and Schmidt rank for pure states

References:
    W. K. Wootters, "Entanglement of formation of an arbitrary state of two qubits,"
    Physical Review Letters, 80(10):2245, 1998.
    G. Vidal and R. F. Werner, "Computable measure of entanglement,"
    Physical Review A, 65(3):032314, 2002.
    A. Peres, "Separability criterion for density matrices,"
    Physical Review Letters, 77(8):1413, 1996.

Example:
    >>> from psiqit.info.entanglement import concurrence, negativity, is_entangled
    >>> from psiqit.quantum.state import bell_phi_plus, bell_phi_minus, zero
    >>> 
    >>> # Bell state is maximally entangled
    >>> bell = bell_phi_plus()
    >>> C = concurrence(bell)
    >>> N = negativity(bell)
    >>> print(f"Concurrence: {C:.4f}, Negativity: {N:.4f}")
    Concurrence: 1.0000, Negativity: 0.5000
    >>> 
    >>> # Check if entangled
    >>> print(is_entangled(bell))
    True
    >>> print(is_entangled(zero()))
    False
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..math.qalgebra import eigenvalues, trace, mul
from ..quantum.state import Ket
from ..quantum.operator import Operator, pauli_y


@dataclass
class EntanglementResult:
    """
    Result container for entanglement measures.

    Attributes:
        value: Numerical value of the entanglement measure.
        measure: Name of the measure used (e.g., 'Concurrence', 'Negativity').
        is_entangled: Boolean indicating whether the state is entangled.
        details: Optional dictionary with additional information.

    Examples:
        >>> result = EntanglementResult(value=1.0, measure='Concurrence',
        ...                             is_entangled=True, details={'schmidt_rank': 2})
        >>> print(result)
        Concurrence = 1.000000 (entangled)
    """
    value: float
    measure: str
    is_entangled: bool
    details: Optional[dict] = None
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing measure value and entanglement status.
        """
        status = "entangled" if self.is_entangled else "separable"
        return f"{self.measure} = {self.value:.6f} ({status})"


def concurrence_pure(state: Ket, dims: List[int] = None) -> float:
    """
    Compute concurrence for pure bipartite states.

    For a pure state |ψ⟩, concurrence is defined as:
        C = √(2(1 - Tr(ρ_A²)))

    where ρ_A is the reduced density matrix of subsystem A.

    Args:
        state: Pure bipartite quantum state.
        dims: Dimensions of subsystems [dA, dB]. Default is [2, 2].

    Returns:
        Concurrence value between 0 and 1.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus, bell_phi_minus
        >>> C = concurrence_pure(bell_phi_plus())
        >>> print(f"{C:.4f}")
        1.0000
    """
    if dims is None:
        dims = [2, 2]
    
    dA, dB = dims
    
    # Get state vector
    psi = state.data
    dim = len(psi)
    
    if dim != dA * dB:
        raise ValueError(f"State dimension {dim} != {dA * dB}")
    
    # Reshape into matrix
    psi_mat = [[0.0] * dB for _ in range(dA)]
    for i in range(dA):
        for j in range(dB):
            psi_mat[i][j] = psi[i * dB + j]
    
    # Compute reduced density matrix for A
    rho_A = [[0.0] * dA for _ in range(dA)]
    for i in range(dA):
        for j in range(dA):
            for k in range(dB):
                rho_A[i][j] += psi_mat[i][k] * psi_mat[j][k].conjugate()
    
    # Compute purity Tr(ρ_A²)
    purity = 0.0
    for i in range(dA):
        for j in range(dA):
            purity += rho_A[i][j] * rho_A[j][i]
    purity = purity.real
    
    # Concurrence = √(2(1 - purity))
    C = math.sqrt(max(0, 2 * (1 - purity)))
    
    return min(1.0, C)


def concurrence_mixed(rho, dims: List[int] = None) -> float:
    """
    Compute concurrence for mixed 2-qubit states (Wootters formula).

    For a 2-qubit density matrix ρ, concurrence is given by:
        C = max(0, λ₁ - λ₂ - λ₃ - λ₄)

    where λᵢ are the square roots of eigenvalues of R = ρ (σ_y⊗σ_y) ρ* (σ_y⊗σ_y).

    Args:
        rho: Density matrix (2x2x2x2) or Operator for 2 qubits.
        dims: Dimensions of subsystems. Must be [2, 2].

    Returns:
        Concurrence value between 0 and 1.

    Raises:
        ValueError: If dimensions are not 2x2.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> rho = bell_phi_plus().outer(bell_phi_plus())
        >>> C = concurrence_mixed(rho)
        >>> print(f"{C:.4f}")
        1.0000
    """
    if dims is None:
        dims = [2, 2]
    
    dA, dB = dims
    
    if dA != 2 or dB != 2:
        raise ValueError("Concurrence for mixed states only defined for 2-qubit systems")
    
    if isinstance(rho, Operator):
        mat = rho.data
    else:
        mat = rho
    
    # Pauli Y matrix
    Y = [[0, -1j], [1j, 0]]
    
    # Y ⊗ Y
    YY = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    YY[i*2 + k][j*2 + l] = Y[i][j] * Y[k][l]
    
    # ρ_tilde = (Y⊗Y) ρ* (Y⊗Y)
    rho_conj = [[x.conjugate() for x in row] for row in mat]
    
    # YY @ rho_conj
    temp = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            temp[i][j] = sum(YY[i][k] * rho_conj[k][j] for k in range(4))
    
    # temp @ YY
    rho_tilde = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            rho_tilde[i][j] = sum(temp[i][k] * YY[k][j] for k in range(4))
    
    # R = ρ @ ρ_tilde
    R = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            R[i][j] = sum(mat[i][k] * rho_tilde[k][j] for k in range(4))
    
    # Eigenvalues of R
    eigvals = eigenvalues(R)
    lam = [max(0, v.real) for v in eigvals]
    lam.sort(reverse=True)
    
    # Concurrence = max(0, √λ₁ - √λ₂ - √λ₃ - √λ₄)
    sqrt_lam = [math.sqrt(l) for l in lam]
    C = max(0.0, sqrt_lam[0] - sqrt_lam[1] - sqrt_lam[2] - sqrt_lam[3])
    
    return min(1.0, C)


def concurrence(state_or_rho, dims: List[int] = None) -> float:
    """
    Compute concurrence (auto-detects pure or mixed state).

    Args:
        state_or_rho: Ket (pure state) or density matrix (mixed state).
        dims: Dimensions of subsystems [dA, dB]. Default is [2, 2].

    Returns:
        Concurrence value between 0 and 1.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus, zero
        >>> C = concurrence(bell_phi_plus())
        >>> print(f"{C:.4f}")
        1.0000
        >>> C = concurrence(zero())
        >>> print(f"{C:.4f}")
        0.0000
    """
    if isinstance(state_or_rho, Ket):
        return concurrence_pure(state_or_rho, dims)
    else:
        return concurrence_mixed(state_or_rho, dims)


def negativity(state_or_rho, dims: List[int] = None) -> float:
    """
    Compute negativity for bipartite states.

    Negativity is defined as:
        N(ρ) = (||ρ^{T_A}||₁ - 1) / 2

    where ρ^{T_A} is the partial transpose with respect to subsystem A,
    and ||·||₁ is the trace norm.

    Args:
        state_or_rho: Ket (pure state) or density matrix (mixed state).
        dims: Dimensions of subsystems [dA, dB]. Default is [2, 2].

    Returns:
        Negativity value (0 for separable states, >0 for entangled states).

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> N = negativity(bell_phi_plus())
        >>> print(f"{N:.4f}")
        0.5000
    """
    if dims is None:
        dims = [2, 2]
    
    dA, dB = dims
    
    # Get density matrix
    if isinstance(state_or_rho, Ket):
        # Pure state: convert to density matrix
        data = state_or_rho.data
        rho_mat = [[data[i] * data[j].conjugate() 
                    for j in range(len(data))] 
                   for i in range(len(data))]
    elif isinstance(state_or_rho, Operator):
        rho_mat = state_or_rho.data
    else:
        rho_mat = state_or_rho
    
    # Partial transpose on subsystem A
    size = dA * dB
    rho_pt = [[0.0] * size for _ in range(size)]
    
    for i in range(dA):
        for j in range(dA):
            for k in range(dB):
                for l in range(dB):
                    row = i * dB + k
                    col = j * dB + l
                    # Partial transpose: swap k and l indices
                    rho_pt[row][col] = rho_mat[i * dB + l][j * dB + k]
    
    # For 4x4 matrix, compute eigenvalues via numpy if available
    try:
        import numpy as np
        rho_pt_np = np.array(rho_pt, dtype=complex)
        eigvals = np.linalg.eigvals(rho_pt_np)
        neg_sum = sum(-v.real for v in eigvals if v.real < -1e-10)
        return neg_sum
    except ImportError:
        # Fallback: manual calculation for 2-qubit case
        if dA == 2 and dB == 2:
            import numpy as np
            rho_pt_np = np.array(rho_pt, dtype=complex)
            eigvals = np.linalg.eigvals(rho_pt_np)
            min_eig = min(v.real for v in eigvals)
            return max(0.0, -min_eig)
        else:
            return 0.0


def logarithmic_negativity(state_or_rho, dims: List[int] = None) -> float:
    """
    Compute logarithmic negativity.

    Logarithmic negativity is defined as:
        E_N(ρ) = log₂(2N(ρ) + 1)

    where N(ρ) is the negativity.

    Args:
        state_or_rho: Ket (pure state) or density matrix (mixed state).
        dims: Dimensions of subsystems [dA, dB]. Default is [2, 2].

    Returns:
        Logarithmic negativity.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> EN = logarithmic_negativity(bell_phi_plus())
        >>> print(f"{EN:.4f}")
        1.0000
    """
    N = negativity(state_or_rho, dims)
    return math.log2(2 * N + 1) if N > 1e-10 else 0.0


def entanglement_entropy(rho, dims: List[int]) -> float:
    """
    Compute entanglement entropy via partial trace.

    For a bipartite state, entanglement entropy is the von Neumann entropy
    of the reduced density matrix: S(ρ_A) = -Tr(ρ_A log ρ_A).

    Args:
        rho: Density matrix of the bipartite system.
        dims: Dimensions of subsystems [dA, dB].

    Returns:
        Entanglement entropy in bits (base 2).

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> rho = bell_phi_plus().outer(bell_phi_plus())
        >>> S = entanglement_entropy(rho, [2, 2])
        >>> print(f"{S:.4f}")
        1.0000
    """
    from .entropy import von_neumann_entropy
    
    dA, dB = dims
    
    if isinstance(rho, Operator):
        mat = rho.data
    else:
        mat = rho
    
    # Partial trace over subsystem B
    rho_A = [[0.0] * dA for _ in range(dA)]
    for i in range(dA):
        for j in range(dA):
            for k in range(dB):
                rho_A[i][j] += mat[i * dB + k][j * dB + k]
    
    return von_neumann_entropy(rho_A, base='2')


def schmidt_decomposition(state: Ket, dims: List[int]) -> dict:
    """
    Compute Schmidt decomposition of a bipartite pure state.

    Any bipartite pure state can be written as:
        |ψ⟩ = Σ λ_i |i_A⟩|i_B⟩

    where λ_i are the Schmidt coefficients.

    Args:
        state: Bipartite pure state.
        dims: Dimensions of subsystems [dA, dB].

    Returns:
        Dictionary containing:
            - 'coefficients': List of Schmidt coefficients (sorted descending)
            - 'schmidt_rank': Number of non-zero coefficients
            - 'entanglement_entropy': Entanglement entropy in bits

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus
        >>> result = schmidt_decomposition(bell_phi_plus(), [2, 2])
        >>> print(f"Schmidt coefficients: {result['coefficients']}")
        Schmidt coefficients: [0.7071067811865475, 0.7071067811865475]
        >>> print(f"Schmidt rank: {result['schmidt_rank']}")
        Schmidt rank: 2
    """
    dA, dB = dims
    
    # Reshape state into matrix
    psi_mat = [[0.0] * dB for _ in range(dA)]
    for i in range(dA):
        for j in range(dB):
            psi_mat[i][j] = state.data[i * dB + j]
    
    # Compute reduced density matrix for A
    rho_A = [[0.0] * dA for _ in range(dA)]
    for i in range(dA):
        for j in range(dA):
            for k in range(dB):
                rho_A[i][j] += psi_mat[i][k] * psi_mat[j][k].conjugate()
    
    eigvals = eigenvalues(rho_A)
    schmidt_coeffs = [math.sqrt(max(0, v.real)) for v in eigvals]
    schmidt_coeffs.sort(reverse=True)
    
    entropy = -sum(c**2 * math.log2(c**2) for c in schmidt_coeffs if c > 1e-10)
    
    return {
        'coefficients': schmidt_coeffs,
        'schmidt_rank': sum(1 for c in schmidt_coeffs if c > 1e-10),
        'entanglement_entropy': entropy
    }


def schmidt_rank(state: Ket, dims: List[int], tol: float = 1e-10) -> int:
    """
    Compute Schmidt rank of a bipartite pure state.

    The Schmidt rank is the number of non-zero Schmidt coefficients.

    Args:
        state: Bipartite pure state.
        dims: Dimensions of subsystems [dA, dB].
        tol: Numerical tolerance for zero coefficients.

    Returns:
        Schmidt rank (≥ 1). Rank 1 means the state is separable.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus, zero
        >>> r = schmidt_rank(bell_phi_plus(), [2, 2])
        >>> print(r)
        2
        >>> r = schmidt_rank(zero(), [2, 2])
        >>> print(r)
        1
    """
    result = schmidt_decomposition(state, dims)
    return result['schmidt_rank']


def is_entangled(state_or_rho, dims: List[int] = None, tol: float = 1e-10) -> bool:
    """
    Check if a bipartite state is entangled.

    For pure states, checks Schmidt rank > 1.
    For mixed states, checks negativity > tolerance.

    Args:
        state_or_rho: Ket (pure state) or density matrix (mixed state).
        dims: Dimensions of subsystems [dA, dB]. Default is [2, 2].
        tol: Numerical tolerance.

    Returns:
        True if the state is entangled, False otherwise.

    Examples:
        >>> from psiqit.quantum.state import bell_phi_plus, zero
        >>> print(is_entangled(bell_phi_plus()))
        True
        >>> print(is_entangled(zero()))
        False
    """
    if dims is None:
        dims = [2, 2]
    
    # Try negativity criterion for mixed states
    N = negativity(state_or_rho, dims)
    
    # For pure states, also check Schmidt rank
    if isinstance(state_or_rho, Ket):
        rank = schmidt_rank(state_or_rho, dims)
        return rank > 1 or N > tol
    
    return N > tol


__all__ = [
    'EntanglementResult',
    'concurrence',
    'concurrence_pure',
    'concurrence_mixed',
    'negativity',
    'logarithmic_negativity',
    'entanglement_entropy',
    'schmidt_decomposition',
    'schmidt_rank',
    'is_entangled',
]