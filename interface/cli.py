"""
psiqit/interface/cli.py
============================================================
Command Line Interface for psiqit
============================================================

CLI tool for running quantum circuits and algorithms from terminal.

This module provides a command-line interface to PSIQIT, allowing users to
run quantum circuits and algorithms without writing Python code. It supports:
    - Building and executing quantum circuits with various gates
    - Running quantum algorithms (Grover, Deutsch-Jozsa, QFT, Shor)
    - Saving results to JSON files

Example:
    $ python -m psiqit.interface.cli circuit --n-qubits 2 --gates h0,cx01 --shots 100
    $ python -m psiqit.interface.cli algorithm --algorithm grover --n-qubits 3 --target 5
    $ python -m psiqit.interface.cli info
"""

import argparse
import sys
import json
from typing import List, Dict, Any, Optional


def parse_gates(gates_str: str) -> List[tuple]:
    """
    Parse gate string from command line.

    The gate string format supports:
        - Single-qubit gates: h0, x1, y2, z3 (gate letter + qubit index)
        - Two-qubit gates: cx01, cz23 (cx/cz + control + target)
        - SWAP gate: swap01 (swap + qubit1 + qubit2)

    Args:
        gates_str: Comma-separated gate string (e.g., "h0,cx01,swap23")

    Returns:
        List of gate tuples in the format:
            - ('h', qubit) for Hadamard
            - ('x', qubit) for Pauli X
            - ('y', qubit) for Pauli Y
            - ('z', qubit) for Pauli Z
            - ('cx', control, target) for CNOT
            - ('cz', control, target) for CZ
            - ('swap', qubit1, qubit2) for SWAP

    Examples:
        >>> parse_gates("h0,cx01")
        [('h', 0), ('cx', 0, 1)]
        >>> parse_gates("x1,z2,swap23")
        [('x', 1), ('z', 2), ('swap', 2, 3)]
    """
    gates = []
    for g in gates_str.split(','):
        g = g.strip()
        if g[0] == 'h':
            gates.append(('h', int(g[1:])))
        elif g[0] == 'x':
            gates.append(('x', int(g[1:])))
        elif g[0] == 'y':
            gates.append(('y', int(g[1:])))
        elif g[0] == 'z':
            gates.append(('z', int(g[1:])))
        elif g.startswith('cx'):
            gates.append(('cx', int(g[2]), int(g[3])))
        elif g.startswith('cz'):
            gates.append(('cz', int(g[2]), int(g[3])))
        elif g.startswith('swap'):
            gates.append(('swap', int(g[4]), int(g[5])))
    return gates


def run_circuit(args):
    """
    Run a quantum circuit from command line arguments.

    This function builds a circuit based on the parsed gate string,
    executes it, and prints measurement results.

    Args:
        args: Parsed command line arguments containing:
            - n_qubits: Number of qubits
            - gates: Gate string (e.g., "h0,cx01")
            - shots: Number of measurements
            - output: Optional output file path

    Examples:
        >>> # Simulate from command line:
        >>> $ python -m psiqit.interface.cli circuit --n-qubits 2 --gates h0,cx01 --shots 100
    """
    from ..circuits.circuit import QuantumCircuit
    from ..quantum.measurement import measure
    
    circuit = QuantumCircuit(args.n_qubits)
    gates = parse_gates(args.gates)
    
    for gate in gates:
        if gate[0] == 'h':
            circuit.h(gate[1])
        elif gate[0] == 'x':
            circuit.x(gate[1])
        elif gate[0] == 'y':
            circuit.y(gate[1])
        elif gate[0] == 'z':
            circuit.z(gate[1])
        elif gate[0] == 'cx':
            circuit.cx(gate[1], gate[2])
        elif gate[0] == 'cz':
            circuit.cz(gate[1], gate[2])
        elif gate[0] == 'swap':
            circuit.swap(gate[1], gate[2])
    
    result = circuit.measure(shots=args.shots)
    
    print("\n" + "=" * 50)
    print("Quantum Circuit Results")
    print("=" * 50)
    print(f"Number of qubits: {args.n_qubits}")
    print(f"Circuit depth: {circuit.depth}")
    print(f"Shots: {args.shots}")
    print("\nMeasurement counts:")
    for outcome, count in sorted(result.items()):
        prob = count / args.shots
        print(f"  |{outcome}⟩: {count} ({prob:.2%})")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


def run_algorithm(args):
    """
    Run a quantum algorithm from command line arguments.

    Supported algorithms:
        - grover: Grover's search algorithm
        - deutsch-jozsa: Deutsch-Jozsa algorithm
        - qft: Quantum Fourier Transform
        - shor: Shor's factoring algorithm

    Args:
        args: Parsed command line arguments containing algorithm-specific parameters.

    Examples:
        >>> # Run Grover's algorithm
        >>> $ python -m psiqit.interface.cli algorithm --algorithm grover --n-qubits 3 --target 5
        >>> 
        >>> # Run Shor's algorithm
        >>> $ python -m psiqit.interface.cli algorithm --algorithm shor --number 15
    """
    from ..algorithms.grover import grover_search
    from ..algorithms.deutsch_jozsa import deutsch_jozsa_constant, deutsch_jozsa_balanced
    from ..algorithms.qft import QFT
    from ..algorithms.shor import Shor
    
    print("\n" + "=" * 50)
    print(f"Running {args.algorithm.upper()} Algorithm")
    print("=" * 50)
    
    if args.algorithm == 'grover':
        result = grover_search(args.n_qubits, args.target, shots=args.shots)
        print(result.summary())
        
    elif args.algorithm == 'deutsch-jozsa':
        if args.oracle_type == 'constant':
            result = deutsch_jozsa_constant(args.n_qubits, args.constant_value)
        else:
            result = deutsch_jozsa_balanced(args.n_qubits)
        print(result)
        
    elif args.algorithm == 'qft':
        qft = QFT(args.n_qubits)
        circuit = qft.build_circuit()
        print(f"QFT circuit created with {args.n_qubits} qubits")
        print(f"Circuit depth: {circuit.depth}")
        
    elif args.algorithm == 'shor':
        shor = Shor()
        result = shor.factor(args.number)
        print(result)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'algorithm': args.algorithm, 'result': str(result)}, f, indent=2)
        print(f"\nResults saved to {args.output}")


def main():
    """
    Main entry point for the PSIQIT command-line interface.

    This function parses command-line arguments and dispatches to the
    appropriate subcommand (circuit, algorithm, or info).

    Subcommands:
        circuit: Run a custom quantum circuit
        algorithm: Run a predefined quantum algorithm
        info: Display version and author information

    Examples:
        >>> $ python -m psiqit.interface.cli circuit --n-qubits 2 --gates h0,cx01
        >>> $ python -m psiqit.interface.cli algorithm --algorithm grover --n-qubits 3 --target 5
        >>> $ python -m psiqit.interface.cli info
    """
    parser = argparse.ArgumentParser(description='psiqit - Quantum Information Toolkit CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Circuit command
    circuit_parser = subparsers.add_parser('circuit', help='Run quantum circuit')
    circuit_parser.add_argument('--n-qubits', type=int, required=True, help='Number of qubits')
    circuit_parser.add_argument('--gates', type=str, required=True, help='Gates (e.g., "h0,cx01")')
    circuit_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    circuit_parser.add_argument('--output', type=str, help='Output file')
    
    # Algorithm command
    algo_parser = subparsers.add_parser('algorithm', help='Run quantum algorithm')
    algo_parser.add_argument('--algorithm', type=str, required=True, 
                             choices=['grover', 'deutsch-jozsa', 'qft', 'shor'],
                             help='Algorithm to run')
    algo_parser.add_argument('--n-qubits', type=int, default=3, help='Number of qubits')
    algo_parser.add_argument('--target', type=int, default=0, help='Target state (for Grover)')
    algo_parser.add_argument('--oracle-type', type=str, default='constant',
                             choices=['constant', 'balanced'], help='Oracle type')
    algo_parser.add_argument('--constant-value', type=int, default=0, help='Constant value')
    algo_parser.add_argument('--number', type=int, default=15, help='Number to factor (for Shor)')
    algo_parser.add_argument('--shots', type=int, default=1024, help='Number of shots')
    algo_parser.add_argument('--output', type=str, help='Output file')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show psiqit information')
    
    args = parser.parse_args()
    
    if args.command == 'circuit':
        run_circuit(args)
    elif args.command == 'algorithm':
        run_algorithm(args)
    elif args.command == 'info':
        from ..version import __version__, __author__
        print("\n" + "=" * 50)
        print("psiqit - a Quantum Information Toolkit")
        print("=" * 50)
        print(f"Version: {__version__}")
        print(f"Author: {__author__}")
        print("\nAvailable commands:")
        print("  circuit   - Run quantum circuit")
        print("  algorithm - Run quantum algorithm")
        print("  info      - Show information")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()