from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="psiqit",
    version="1.0.2",
    author="Mahdi Azadmarzabadi",
    author_email="psiqitofficial@protonmail.com",
    description="PSIQIT - Quantum Information Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/psiqit/1.0.2",
    project_urls={
        "Documentation": "https://psiqit.github.io/1.0.2/",
        "Source": "https://github.com/psiqit/1.0.2",
    },
    packages=find_packages(exclude=["docs", "tutorials", "examples", "docs.*", "tutorials.*", "examples.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "sympy>=1.9.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4.0"],
        "ml": ["scikit-learn>=1.0.0", "jupyter>=1.0.0"],
        "full": [
            "matplotlib>=3.4.0",
            "scikit-learn>=1.0.0",
            "jupyter>=1.0.0",
            "sympy>=1.9.0",
        ],
    },
    include_package_data=False,
    zip_safe=False,
)