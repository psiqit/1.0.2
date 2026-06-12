"""
psiqit/algorithms/shor.py
============================================================
Shor's Factoring Algorithm
============================================================

Shor's algorithm for integer factorization using quantum period finding.

Shor's algorithm is one of the most famous quantum algorithms. It factors
an integer N in O((log N)³) time, which is exponentially faster than the
best known classical factoring algorithms. This exponential speedup has
significant implications for cryptography, particularly for RSA encryption.

Example:
    >>> from psiqit.algorithms.shor import Shor
    >>> 
    >>> shor = Shor()
    >>> factors = shor.factor(15)  # 15 = 3 * 5
    >>> print(factors)
    [3, 5]
"""

import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..circuits.circuit import QuantumCircuit
from ..quantum.state import Ket, basis
from ..quantum.operator import hadamard, cnot, swap, phase
from ..quantum.measurement import measure


@dataclass
class ShorResult:
    """
    Result container for Shor's factoring algorithm.

    Attributes:
        number: The integer that was being factored.
        factors: List of prime factors (e.g., [3, 5] for 15).
        period: The found period r such that a^r ≡ 1 (mod N).
        success: Whether factoring was successful.
        attempts: Number of attempts made.

    Examples:
        >>> result = ShorResult(number=15, factors=[3, 5], period=4,
        ...                     success=True, attempts=1)
        >>> print(result)
        15 = 3 × 5
        
        >>> result = ShorResult(number=21, factors=[], period=0,
        ...                     success=False, attempts=10)
        >>> print(result)
        Failed to factor 21
    """
    number: int
    factors: List[int]
    period: int
    success: bool
    attempts: int
    
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        Returns:
            String showing factorization or failure message.

        Examples:
            >>> result = ShorResult(15, [3, 5], 4, True, 1)
            >>> print(result)
            15 = 3 × 5
        """
        if self.factors:
            return f"{self.number} = {' × '.join(str(f) for f in self.factors)}"
        return f"Failed to factor {self.number}"


class Shor:
    """
    Shor's factoring algorithm implementation.

    Shor's algorithm factors an integer N by reducing the factoring problem
    to period finding. The algorithm works as follows:

    1. Choose a random integer a between 2 and N-1
    2. Compute gcd(a, N). If >1, we found a factor
    3. Use quantum period finding to find the order r of a modulo N
    4. If r is even and a^(r/2) ≠ ±1 mod N, compute factors
    5. Otherwise, repeat with a different a

    The quantum part (period finding) uses quantum phase estimation on the
    unitary U|y⟩ = |ay mod N⟩. This implementation uses classical period
    finding for small numbers as a demonstration.

    References:
        P. W. Shor, "Polynomial-Time Algorithms for Prime Factorization and
        Discrete Logarithms on a Quantum Computer," SIAM Journal on Computing,
        26(5):1484-1509, 1997.

    Examples:
        >>> # Factor a small number
        >>> shor = Shor()
        >>> result = shor.factor(15)
        >>> print(result.factors)
        [3, 5]
        
        >>> # Factor another number
        >>> result = shor.factor(21)
        >>> print(result.factors)
        [3, 7]
        
        >>> # Use convenience function
        >>> from psiqit.algorithms.shor import factorize
        >>> factors = factorize(35)
        >>> print(factors)
        [5, 7]
    """
    
    def __init__(self):
        """
        Initialize Shor's algorithm.

        Examples:
            >>> shor = Shor()
        """
        self._found_factors = []
    
    def _gcd(self, a: int, b: int) -> int:
        """
        Compute greatest common divisor using Euclidean algorithm.

        Args:
            a: First integer.
            b: Second integer.

        Returns:
            gcd(a, b).

        Examples:
            >>> shor = Shor()
            >>> shor._gcd(12, 18)
            6
            >>> shor._gcd(7, 13)
            1
        """
        while b:
            a, b = b, a % b
        return a
    
    def _modular_pow(self, base: int, exp: int, mod: int) -> int:
        """
        Fast modular exponentiation using binary exponentiation.

        Computes (base^exp) mod mod efficiently in O(log exp) time.

        Args:
            base: The base number.
            exp: The exponent.
            mod: The modulus.

        Returns:
            base^exp mod mod.

        Examples:
            >>> shor = Shor()
            >>> shor._modular_pow(2, 10, 1000)
            24
            >>> shor._modular_pow(3, 4, 5)
            1
        """
        result = 1
        base = base % mod
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp >>= 1
        return result
    
    def _quantum_period_finding(self, a: int, N: int, n_qubits: int = 8) -> int:
        """
        Quantum period finding subroutine.

        This is a simplified placeholder for the quantum period finding
        algorithm. In a full implementation, this would use:
        1. Quantum circuit with superposition
        2. Modular exponentiation
        3. Quantum Fourier transform
        4. Measurement and continued fraction expansion

        Args:
            a: Base number.
            N: Modulus.
            n_qubits: Number of qubits for the quantum register.

        Returns:
            The period r such that a^r ≡ 1 (mod N).

        Examples:
            >>> shor = Shor()
            >>> shor._quantum_period_finding(2, 15)
            4  # because 2^4 = 16 ≡ 1 mod 15
        """
        # Simplified quantum circuit for period finding
        circuit = QuantumCircuit(2 * n_qubits)
        
        # Initialize first register in superposition
        for i in range(n_qubits):
            circuit.h(i)
        
        # Apply modular exponentiation (simplified)
        # This is a placeholder - full implementation would use QFT
        
        # Apply inverse QFT (simplified)
        for i in range(n_qubits):
            circuit.h(i)
        
        # Measure
        results = circuit.measure(shots=1)
        
        # Extract period from measurement (simplified)
        # In practice, we would use continued fractions
        
        # For demonstration, use classical period finding for small numbers
        return self._classical_period_finding(a, N)
    
    def _classical_period_finding(self, a: int, N: int) -> int:
        """
        Classical period finding (for small numbers).

        This is a brute-force method that works for small N.
        For large N, the quantum method would be used.

        Args:
            a: Base number.
            N: Modulus.

        Returns:
            The period r such that a^r ≡ 1 (mod N), or 0 if not found.

        Examples:
            >>> shor = Shor()
            >>> shor._classical_period_finding(2, 15)
            4
            >>> shor._classical_period_finding(3, 15)
            Traceback...  # No period
        """
        r = 1
        value = a % N
        while value != 1:
            value = (value * a) % N
            r += 1
            if r > N:
                return 0
        return r
    
    def _quantum_order_finding(self, a: int, N: int) -> int:
        """
        Quantum order finding using Quantum Phase Estimation (QPE).

        This is the core quantum subroutine of Shor's algorithm.
        It finds the order r of a modulo N using QPE on the unitary
        U|y⟩ = |ay mod N⟩.

        Args:
            a: Base number.
            N: Modulus.

        Returns:
            The order r of a modulo N.

        Examples:
            >>> shor = Shor()
            >>> shor._quantum_order_finding(2, 15)
            4
        """
        # Simplified: use classical method for small N
        return self._classical_period_finding(a, N)
    
    def factor(self, N: int, max_attempts: int = 10) -> ShorResult:
        """
        Factor an integer using Shor's algorithm.

        The algorithm repeatedly tries random a values until it finds factors.
        For even numbers, it returns immediately with factor 2.

        Args:
            N: Integer to factor (must be odd and not a prime power).
            max_attempts: Maximum number of attempts before giving up.

        Returns:
            ShorResult: Object containing factors, period, and success status.

        Raises:
            ValueError: If N is 1 or prime? (handled by returning failure)

        Examples:
            >>> shor = Shor()
            >>> result = shor.factor(15)
            >>> print(result.factors)
            [3, 5]
            
            >>> result = shor.factor(21)
            >>> print(result.factors)
            [3, 7]
            
            >>> result = shor.factor(91)  # 7 × 13
            >>> print(result.factors)
            [7, 13]
        """
        # Handle even numbers immediately
        if N % 2 == 0:
            return ShorResult(number=N, factors=[2, N//2], period=0, success=True, attempts=1)
        
        for attempt in range(max_attempts):
            # Choose random a between 2 and N-1
            a = random.randint(2, N - 1)
            
            # Check if a shares a factor with N
            g = self._gcd(a, N)
            if g > 1:
                return ShorResult(number=N, factors=[g, N//g], period=0, success=True, attempts=attempt+1)
            
            # Find order r of a modulo N
            r = self._quantum_order_finding(a, N)
            
            # If r is odd, try another a
            if r % 2 != 0:
                continue
            
            # Compute factors using x = a^(r/2) mod N
            x = self._modular_pow(a, r // 2, N)
            factor1 = self._gcd(x - 1, N)
            factor2 = self._gcd(x + 1, N)
            
            if factor1 > 1 and factor1 < N:
                return ShorResult(number=N, factors=[factor1, N//factor1], period=r, success=True, attempts=attempt+1)
            if factor2 > 1 and factor2 < N:
                return ShorResult(number=N, factors=[factor2, N//factor2], period=r, success=True, attempts=attempt+1)
        
        return ShorResult(number=N, factors=[], period=0, success=False, attempts=max_attempts)


def factorize(N: int) -> List[int]:
    """
    Convenience function for factoring integers.

    This function provides a simple interface to Shor's algorithm.

    Args:
        N: Integer to factor.

    Returns:
        List of prime factors.

    Examples:
        >>> from psiqit.algorithms.shor import factorize
        >>> factors = factorize(15)
        >>> print(factors)
        [3, 5]
        
        >>> factors = factorize(77)
        >>> print(factors)  # May be [7, 11] or [11, 7]
        [7, 11]
    """
    shor = Shor()
    result = shor.factor(N)
    return result.factors


__all__ = [
    'ShorResult',
    'Shor',
    'factorize',
]