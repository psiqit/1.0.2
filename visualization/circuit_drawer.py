"""
psiqit/visualization/circuit_drawer.py
============================================================
Quantum Circuit Drawer
============================================================

ASCII and text-based drawing of quantum circuits.

This module provides tools for visualizing quantum circuits using
ASCII/Unicode characters. It supports both a simple text representation
and a visual diagram with box-drawing characters.

The drawing styles include:
    - ASCII: Uses standard ASCII characters (+, -, |, o, X)
    - Unicode: Uses Unicode box-drawing characters (─, │, ●, ⊕)

Example:
    >>> from psiqit.circuits.circuit import QuantumCircuit
    >>> from psiqit.visualization.circuit_drawer import draw_circuit, circuit_to_text
    >>> 
    >>> # Create a Bell state circuit
    >>> circ = QuantumCircuit(2)
    >>> circ.h(0)
    >>> circ.cx(0, 1)
    >>> 
    >>> # Draw with Unicode (more beautiful)
    >>> print(draw_circuit(circ, style='unicode'))
    q0: ─[H]─●─
    q1: ─────⊕─
    >>> 
    >>> # Get text representation
    >>> print(circuit_to_text(circ))
    Quantum Circuit: 2 qubits, depth 2
    --------------------------------------------------
      1. H        on q0
      2. CNOT     on q0, q1

References:
    Qiskit circuit drawer inspiration, IBM Quantum.
"""

from typing import List, Dict, Optional, Tuple


def draw_circuit(circuit, style: str = 'ascii') -> str:
    """
    Draw a quantum circuit using ASCII or Unicode characters.

    Args:
        circuit: QuantumCircuit instance to draw.
        style: Drawing style - 'ascii' or 'unicode'.

    Returns:
        String representation of the circuit.

    Examples:
        >>> from psiqit.circuits.circuit import QuantumCircuit
        >>> 
        >>> circ = QuantumCircuit(3)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> circ.x(2)
        >>> 
        >>> # ASCII style
        >>> print(draw_circuit(circ, style='ascii'))
        q0: ---[H]---o-----------
        q1: ---------X-----------
        q2: ------------[X]------
        >>> 
        >>> # Unicode style (more beautiful)
        >>> print(draw_circuit(circ, style='unicode'))
        q0: ─[H]─●─
        q1: ─────⊕─
        q2: ─[X]──
    """
    if style == 'ascii':
        return _draw_ascii(circuit)
    else:
        return _draw_unicode(circuit)


def _draw_ascii(circuit) -> str:
    """
    Draw circuit using ASCII characters.

    ASCII symbols used:
        - '-' for horizontal lines
        - 'o' for CNOT control
        - 'X' for CNOT target
        - '[gate]' for gate labels

    Args:
        circuit: QuantumCircuit instance.

    Returns:
        ASCII circuit diagram as string.
    """
    n_qubits = circuit.n_qubits
    gates = circuit._gates
    
    lines = [f"q{i}: ---" for i in range(n_qubits)]
    
    for gate, qubits in gates:
        gate_name = gate.name or "U"
        
        if len(qubits) == 1:
            q = qubits[0]
            for i in range(n_qubits):
                if i == q:
                    lines[i] += f"[{gate_name}]---"
                else:
                    lines[i] += "----"
        
        elif len(qubits) == 2:
            q1, q2 = qubits
            min_q, max_q = min(q1, q2), max(q1, q2)
            
            for i in range(n_qubits):
                if i == q1:
                    lines[i] += "o---"
                elif i == q2:
                    lines[i] += "X---"
                elif min_q < i < max_q:
                    lines[i] += "|---"
                else:
                    lines[i] += "----"
        
        elif len(qubits) == 3:
            q1, q2, q3 = qubits
            for i in range(n_qubits):
                if i == q1 or i == q2:
                    lines[i] += "o---"
                elif i == q3:
                    lines[i] += "X---"
                else:
                    lines[i] += "----"
    
    return "\n".join(lines)


def _draw_unicode(circuit) -> str:
    """
    Draw circuit using Unicode box-drawing characters.

    Unicode symbols used:
        - '─' for horizontal lines
        - '│' for vertical connections
        - '●' for CNOT control
        - '⊕' for CNOT target
        - '[gate]' for gate labels

    Args:
        circuit: QuantumCircuit instance.

    Returns:
        Unicode circuit diagram as string.
    """
    n_qubits = circuit.n_qubits
    gates = circuit._gates
    
    # Unicode box-drawing characters
    HORIZ = "─"
    VERT = "│"
    CROSS = "┼"
    
    lines = [f"q{i}: {HORIZ}" for i in range(n_qubits)]
    
    for gate, qubits in gates:
        gate_name = gate.name or "U"
        
        if len(qubits) == 1:
            q = qubits[0]
            for i in range(n_qubits):
                if i == q:
                    lines[i] += f"[{gate_name}]{HORIZ}"
                else:
                    lines[i] += f"{HORIZ* (len(gate_name)+2)}"
        
        elif len(qubits) == 2:
            q1, q2 = qubits
            min_q, max_q = min(q1, q2), max(q1, q2)
            
            for i in range(n_qubits):
                if i == q1:
                    lines[i] += f"●{HORIZ}"
                elif i == q2:
                    lines[i] += f"⊕{HORIZ}"
                elif min_q < i < max_q:
                    lines[i] += f"{VERT}{HORIZ}"
                else:
                    lines[i] += f"{HORIZ*2}"
    
    return "\n".join(lines)


def circuit_to_text(circuit, show_wires: bool = True) -> str:
    """
    Convert a circuit to a text representation with a list of gates.

    This provides a simple, readable list of all gates in the circuit
    with their gate names and target qubits.

    Args:
        circuit: QuantumCircuit instance.
        show_wires: Whether to show wire connections (currently unused).

    Returns:
        Text representation of the circuit.

    Examples:
        >>> from psiqit.circuits.circuit import QuantumCircuit
        >>> 
        >>> circ = QuantumCircuit(2)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> print(circuit_to_text(circ))
        Quantum Circuit: 2 qubits, depth 2
        --------------------------------------------------
          1. H        on q0
          2. CNOT     on q0, q1
    """
    lines = []
    lines.append(f"Quantum Circuit: {circuit.n_qubits} qubits, depth {circuit.depth}")
    lines.append("-" * 50)
    
    for i, (gate, qubits) in enumerate(circuit._gates):
        gate_name = gate.name or "U"
        qubits_str = ", ".join(f"q{q}" for q in qubits)
        lines.append(f"{i+1:3d}. {gate_name:8s} on {qubits_str}")
    
    return "\n".join(lines)


def circuit_statistics(circuit) -> Dict:
    """
    Get statistics about the circuit.

    This function returns useful metrics about the circuit, including
    the number of qubits, circuit depth, total gate count, and a breakdown
    of gate types.

    Args:
        circuit: QuantumCircuit instance.

    Returns:
        Dictionary containing:
            - 'n_qubits': Number of qubits
            - 'depth': Circuit depth (number of gates)
            - 'n_gates': Total number of gates
            - 'gate_types': Dictionary mapping gate names to counts

    Examples:
        >>> circ = QuantumCircuit(3)
        >>> circ.h(0)
        >>> circ.cx(0, 1)
        >>> circ.x(2)
        >>> 
        >>> stats = circuit_statistics(circ)
        >>> print(f"Gates: {stats['n_gates']}")
        Gates: 3
        >>> print(f"Gate types: {stats['gate_types']}")
        Gate types: {'H': 1, 'CNOT': 1, 'X': 1}
    """
    gate_types = {}
    for gate, _ in circuit._gates:
        gate_name = gate.name or "U"
        gate_types[gate_name] = gate_types.get(gate_name, 0) + 1
    
    return {
        'n_qubits': circuit.n_qubits,
        'depth': circuit.depth,
        'n_gates': len(circuit._gates),
        'gate_types': gate_types,
    }


__all__ = [
    'draw_circuit',
    'circuit_to_text',
    'circuit_statistics',
]