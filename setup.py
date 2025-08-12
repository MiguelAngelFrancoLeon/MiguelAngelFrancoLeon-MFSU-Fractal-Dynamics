#!/usr/bin/env python3
"""
Setup script for MFSU - Unified Fractal-Stochastic Model
Author: Miguel Ángel Franco León  
Date: August 2025

Installation:
    pip install -e .
    
Or for development:
    git clone https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/
    cd MFSU-Universal-Fractal-Constant
    pip install -e .[dev,docs,all]
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Extension

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("MFSU requires Python 3.8 or higher")

# Get the long description from README
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py
def get_version():
    version_file = here / "mfsu" / "__init__.py"
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split('"')[1]
    raise RuntimeError("Unable to find version string")

# Core dependencies - minimal for basic functionality
CORE_DEPS = [
    "numpy>=1.20.0",
    "scipy>=1.8.0", 
    "matplotlib>=3.5.0",
    "pandas>=1.4.0",
    "scikit-learn>=1.1.0",
    "tqdm>=4.60.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
]

# Astronomy and cosmology specific dependencies
ASTRO_DEPS = [
    "astropy>=5.0.0",
    "healpy>=1.15.0",
    "camb>=1.4.0",
]

# Statistical analysis dependencies  
STATS_DEPS = [
    "emcee>=3.1.0",
    "corner>=2.2.0",
    "statsmodels>=0.13.0",
    "arviz>=0.12.0",
]

# Visualization dependencies
VIZ_DEPS = [
    "plotly>=5.10.0",
    "seaborn>=0.11.0", 
    "bokeh>=2.4.0",
    "ipywidgets>=8.0.0",
]

# Development dependencies
DEV_DEPS = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "flake8>=5.0.0",
    "mypy>=0.991",
    "pre-commit>=2.20.0",
    "isort>=5.10.0",
    "bandit>=1.7.0",
]

# Documentation dependencies
DOCS_DEPS = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0",
    "sphinx-autodoc-typehints>=1.19.0",
    "sphinx-copybutton>=0.5.0",
]

# Performance and parallel computing
PERF_DEPS = [
    "numba>=0.56.0",
    "dask>=2022.8.0", 
    "joblib>=1.2.0",
    "cython>=0.29.0",
]

# Optional scientific dependencies
SCIENCE_DEPS = [
    "sympy>=1.10.0",
    "h5py>=3.7.0",
    "netcdf4>=1.6.0",
    "xarray>=2022.6.0",
]

# All optional dependencies combined
ALL_DEPS = (ASTRO_DEPS + STATS_DEPS + VIZ_DEPS + 
           DEV_DEPS + DOCS_DEPS + PERF_DEPS + SCIENCE_DEPS)

# C extensions for performance-critical fractal calculations
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

ext_modules = []
if USE_CYTHON:
    ext_modules = [
        Extension(
            "mfsu.core._fractal_fast",
            ["mfsu/core/_fractal_fast.pyx"],
            include_dirs=["mfsu/core/"],
        ),
        Extension(
            "mfsu.analysis._stats_fast", 
            ["mfsu/analysis/_stats_fast.pyx"],
            include_dirs=["mfsu/analysis/"],
        ),
    ]
    ext_modules = cythonize(ext_modules, compiler_directives={'language_level': 3})

# Entry points for command-line tools
entry_points = {
    "console_scripts": [
        "mfsu=mfsu.cli.main:main",
        "mfsu-analyze=mfsu.cli.analyze:main",
        "mfsu-simulate=mfsu.cli.simulate:main", 
        "mfsu-validate=mfsu.cli.validate:main",
        "mfsu-plot=mfsu.cli.plot:main",
        "mfsu-cmb=mfsu.cli.cmb:main",
        "mfsu-fractal=mfsu.cli.fractal:main",
    ],
}

# Package data
package_data = {
    "mfsu": [
        "data/reference/*.csv",
        "data/reference/*.json", 
        "data/templates/*.yaml",
        "config/*.yaml",
        "tests/data/*.fits",
        "tests/data/*.h5",
    ],
}

# Additional package metadata
keywords = [
    "fractal", "stochastic", "cosmology", "physics", "CMB", 
    "superconductivity", "diffusion", "scaling", "universality",
    "delta-F", "MFSU", "scientific-computing", "astrophysics"
]

classifiers = [
    # Development Status
    "Development Status :: 4 - Beta",
    
    # Intended Audience
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education", 
    "Intended Audience :: Developers",
    
    # Topic classifications
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Astronomy",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Visualization",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Programming Language
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9", 
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    
    # Operating System
    "Operating System :: OS Independent",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X", 
    "Operating System :: Microsoft :: Windows",
    
    # Natural Language
    "Natural Language :: English",
    
    # Environment
    "Environment :: Console",
    "Environment :: Web Environment",
]

# Project URLs
project_urls = {
    "Homepage": "https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/",
    "Documentation": "https://mfsu-model.org/",
    "Repository": "https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/",
    "Bug Reports": "https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/issues",
    "Funding": "https://github.com/sponsors/MiguelAngelFrancoLeon",
    "Zenodo Dataset": "https://doi.org/10.5281/zenodo.16316882",
    "arXiv Paper": "https://arxiv.org/abs/",
}

setup(
    # Basic package information
    name="mfsu",
    version=get_version(),
    author="Miguel Ángel Franco León",
    author_email="miguelfranco@mfsu-model.org",
    description="Unified Fractal-Stochastic Model for complex systems in physics and cosmology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/",
    project_urls=project_urls,
    
    # Package structure
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_data=package_data,
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=CORE_DEPS,
    extras_require={
        "astro": ASTRO_DEPS,
        "stats": STATS_DEPS, 
        "viz": VIZ_DEPS,
        "dev": DEV_DEPS,
        "docs": DOCS_DEPS,
        "perf": PERF_DEPS,
        "science": SCIENCE_DEPS,
        "all": ALL_DEPS,
    },
    
    # C extensions
    ext_modules=ext_modules,
    
    # Entry points
    entry_points=entry_points,
    
    # Metadata
    keywords=keywords,
    classifiers=classifiers,
    license="MIT",
    
    # Additional options
    zip_safe=False,
    platforms=["any"],
    
    # Testing
    test_suite="tests",
    tests_require=DEV_DEPS,
    
    # Command classes for custom commands
    cmdclass={},
)

# Post-install message
print("""
╔══════════════════════════════════════════════════════════╗
║                    MFSU Installation                     ║
║        Unified Fractal-Stochastic Model (δF ≈ 0.921)    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Successfully installed MFSU package!                   ║
║                                                          ║
║  Next steps:                                            ║
║  1. conda activate mfsu                                 ║
║  2. python -c "import mfsu; mfsu.test_installation()"   ║
║  3. mfsu --help                                         ║
║  4. jupyter lab notebooks/                              ║
║                                                          ║
║  Documentation: https://mfsu.readthedocs.io/            ║
║  Repository: github.com/MiguelAngelFrancoLeon/MFSU...   ║
║                                                          ║
║  Citation: Franco León, M.A. (2025). MFSU Framework     ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")
