"""
psiqit/version.py
Version information for Quantum Information Toolkit
"""

# ============================================================
# Version following Semantic Versioning (MAJOR.MINOR.PATCH)
# ============================================================

__version__ = "1.0.2"
__version_info__ = (1, 0, 2)

# ============================================================
# Release status
# ============================================================

__status__ = "Production/Stable"  # Alpha, Beta, Production/Stable
__build__ = "1.0.2"

# ============================================================
# Author Information
# ============================================================

__author__ = "mahdi azadmarzabadi"
__maintainer__ = "psiqit Develo"
__email__ = "psiqitofficial@protonmail.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025-2026 psiqit Development Team"

# ============================================================
# Project Information
# ============================================================

__project__ = "psiqit"
__title__ = "Quantum Information Toolkit"
__description__ = "Comprehensive Python library for quantum computing"
__url__ = "https://github.com/psiqit"

# ============================================================
# Dependency Information
# ============================================================

PYTHON_REQUIRED = (3, 8)

REQUIRED_PACKAGES = {
    "numpy": "1.21.0",
    "scipy": "1.7.0",
}

OPTIONAL_PACKAGES = {
    "matplotlib": "3.4.0",
    "sympy": "1.10",
    "jupyter": "1.0.0",
}

DEV_PACKAGES = {
    "pytest": "7.0.0",
    "black": "22.0.0",
    "sphinx": "5.0.0",
}

# ============================================================
# Helper Functions
# ============================================================

def check_python_version() -> bool:
    """
    Check if current Python version meets requirements.
    
    Returns:
        True if version is compatible, False otherwise
    """
    import sys
    if sys.version_info < PYTHON_REQUIRED:
        print(f"Warning: psiqit requires Python {PYTHON_REQUIRED[0]}.{PYTHON_REQUIRED[1]}+")
        print(f"Current version: {sys.version_info[0]}.{sys.version_info[1]}")
        return False
    return True


def get_version_string() -> str:
    
    return f"psiqit version {__version__} ({__status__})"


def get_version_info() -> dict:
    
    return {
        "version": __version__,
        "version_info": __version_info__,
        "status": __status__,
        "build": __build__,
        "author": __author__,
        "license": __license__,
        "project": __project__,
        "title": __title__,
        "description": __description__,
        "url": __url__,
    }


def show_version() -> None:
    
    print("=" * 50)
    print(get_version_string())
    print("=" * 50)
    print(f"Project: {__title__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"Python Required: {PYTHON_REQUIRED[0]}.{PYTHON_REQUIRED[1]}+")
    print(f"Status: {__status__}")
    print("=" * 50)


def require_minimum_version(package: str, min_version: str) -> bool:
    
    try:
        import importlib.metadata
        installed = importlib.metadata.version(package)
    except ImportError:
        try:
            import pkg_resources
            installed = pkg_resources.get_distribution(package).version
        except Exception:
            return False
    
    def parse_version(v):
        return tuple(int(x) for x in v.split('.')[:3])
    
    return parse_version(installed) >= parse_version(min_version)


# ============================================================
# Exports
# ============================================================

__all__ = [
    "__version__",
    "__version_info__",
    "__status__",
    "__build__",
    "__author__",
    "__maintainer__",
    "__email__",
    "__license__",
    "__copyright__",
    "__project__",
    "__title__",
    "__description__",
    "__url__",
    "PYTHON_REQUIRED",
    "REQUIRED_PACKAGES",
    "OPTIONAL_PACKAGES",
    "DEV_PACKAGES",
    "check_python_version",
    "get_version_string",
    "get_version_info",
    "show_version",
    "require_minimum_version",
]