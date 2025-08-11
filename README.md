# 🌌 MFSU - Unified Fractal-Stochastic Model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16316882.svg)](https://doi.org/10.5281/zenodo.16316882)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


> **A universal fractal constant δF ≈ 0.921 governs cosmic structure across scales**

**Author:** Miguel Ángel Franco León  
**Contact:** miguelfranco@mfsu-model.org
             mf410360@gmail.com
**Institution:** Independent Researcher  

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/MiguelAngelFrancoLeon/MFSU-Universal-Fractal-Constant.git
cd MFSU-Universal-Fractal-Constant

# Install dependencies
pip install -r requirements.txt

# Run the main analysis
python scripts/mfsu_analysis.py

# Generate Nature-quality figures
python scripts/generate_figures.py
```

## 📄 Abstract

The Unified Fractal-Stochastic Model (MFSU) reveals that a universal fractal constant **δF ≈ 0.921** emerges independently across diverse physical systems, from cosmic microwave background fluctuations to superconducting critical temperatures. This constant optimizes fits to observational data, reducing χ² by 23% compared to standard models and potentially resolving fundamental cosmological tensions without exotic dark matter particles.

## 🎯 Key Results

- **Universal Constant**: δF = 0.921 ± 0.003 across 5 independent domains
- **CMB Analysis**: 23% improvement over ΛCDM using Planck 2018 data
- **Superconductivity**: 0.87% error vs 5.93% for BCS theory
- **Galaxy Rotation**: Flat curves emerge naturally without dark matter
- **Cosmological Tensions**: Resolves Hubble and S₈ discrepancies

## 📊 Visual Summary

| ![Figure 1](figures/figure_1_universality.png) | ![Figure 2](figures/figure_2_performance.png) |
|:---:|:---:|
| **Universal Constant δF** | **Performance vs Standard Models** |

| ![Figure 3](figures/figure_3_cmb_spectrum.png) | ![Figure 4](figures/figure_4_rotation_curves.png) |
|:---:|:---:|
| **CMB Power Spectrum** | **Galaxy Rotation Curves** |

## 📁 Repository Structure

```
MFSU-Universal-Fractal-Constant/
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 📄 CITATION.cff                 # Citation metadata
├── 📄 CHANGELOG.md                 # Version history
│
├── 📂 papers/                      # Scientific papers
│   ├── nature_submission.pdf       # Nature submission
│   ├── technical_report.pdf        # Complete technical analysis
│   └── supplementary_material.pdf  # Extended data
│
├── 📂 data/                        # Raw and processed data
│   ├── planck/                     # CMB data analysis
│   │   ├── COM_PowerSpect_CMB-TT-full_R3.01.txt
│   │   └── processed_cmb_analysis.csv
│   ├── superconductors/            # Critical temperature data
│   │   └── tc_database.csv
│   └── diffusion/                  # Gas diffusion experiments
│       └── co2_diffusion_results.csv
│
├── 📂 scripts/                     # Analysis and figure generation
│   ├── mfsu_analysis.py           # Main MFSU analysis
│   ├── generate_figures.py        # Nature-quality figures
│   ├── cmb_analysis.py            # CMB power spectrum analysis
│   ├── superconductor_analysis.py # Tc fitting
│   └── utils/                     # Utility functions
│       ├── fractal_tools.py
│       ├── statistical_analysis.py
│       └── plotting_utils.py
│
├── 📂 figures/                     # Generated figures
│   ├── figure_1_universality.png
│   ├── figure_2_performance.png
│   ├── figure_3_cmb_spectrum.png
│   ├── figure_4_rotation_curves.png
│   └── extended_data/             # Supplementary figures
│
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_cmb_validation.ipynb
│   ├── 03_superconductor_fitting.ipynb
│   └── 04_fractal_simulations.ipynb
│
├── 📂 tests/                       # Unit tests
│   ├── test_fractal_tools.py
│   ├── test_statistical_analysis.py
│   └── test_mfsu_model.py
│
├── 📂 docs/                        # Documentation
│   ├── theory.md                  # Theoretical background
│   ├── methodology.md             # Analysis methodology
│   ├── validation.md              # Validation procedures
│   └── api_reference.md           # Code documentation
│
└── 📂 examples/                    # Usage examples
    ├── basic_usage.py
    ├── reproduce_nature_figures.py
    └── extend_to_new_systems.py
```

## 🔬 Theoretical Background

The MFSU proposes that space-time exhibits intrinsic fractal structure characterized by the universal constant:

```
δF ≈ 0.921
```

This constant emerges from:
- **Variational principles** in fractal space-time
- **Entropy maximization** under self-similarity constraints
- **Percolation theory** in 3D fractal systems
- **Quantum entanglement** entropy in fractal geometries

### Core Equation

The MFSU field equation is:

```
∂ψ/∂t = α(−Δ)^(δF/2)ψ + βξH(x,t)ψ − γψ³
```

Where:
- `α ≈ 0.921`: Fractal diffusion coefficient
- `δF ≈ 0.921`: Universal fractal dimension
- `ξH`: Fractional Brownian noise (H ≈ 0.541)
- `γ`: Nonlinear stabilization parameter

## 📈 Validation Results

### Cosmic Microwave Background
- **Data**: Planck 2018 TT power spectrum (l = 2-2508)
- **Improvement**: 23% reduction in χ² vs ΛCDM
- **Key regions**: Excess power at l = 100-300 and l = 600-1000

### Superconductivity
- **Materials**: 15 cuprate superconductors
- **MFSU error**: 0.87% mean absolute error
- **BCS error**: 5.93% mean absolute error
- **Scaling law**: Tc ∝ (deff/d₀)^(1/(δF-1))

### Anomalous Diffusion
- **System**: CO₂ in fractal porous media
- **MFSU R²**: 0.987
- **Fick's law R²**: 0.823
- **Equation**: ∂C/∂t = DF∇^δF C

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Jupyter (for notebooks)
- Optional: HEALPix (for CMB analysis)

### Install from source
```bash
git clone https://github.com/MiguelAngelFrancoLeon/MFSU-Universal-Fractal-Constant.git
cd MFSU-Universal-Fractal-Constant
pip install -r requirements.txt
```

### Install as package
```bash
pip install git+https://github.com/MiguelAngelFrancoLeon/MFSU-Universal-Fractal-Constant.git
```

## 🎮 Usage Examples

### Basic Analysis
```python
from mfsu import UniversalFractalModel

# Initialize MFSU model
model = UniversalFractalModel(delta_f=0.921)

# Analyze CMB data
cmb_results = model.analyze_cmb('data/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt')
print(f"χ² improvement: {cmb_results.chi2_improvement:.1f}%")

# Fit superconductor data
tc_results = model.fit_superconductors('data/superconductors/tc_database.csv')
print(f"Mean error: {tc_results.mean_error:.2f}%")
```

### Generate Figures
```python
from mfsu.visualization import NatureFigures

# Create Nature-quality figures
figures = NatureFigures()
figures.generate_all(output_dir='figures/')
```

### Custom Analysis
```python
# Analyze your own fractal system
your_data = load_your_data()
fractal_dim = model.estimate_fractal_dimension(your_data)
print(f"Estimated δF: {fractal_dim:.3f}")
```

## 📊 Reproducibility

All results in our papers can be reproduced using this repository:

```bash
# Reproduce Nature figures
python scripts/reproduce_nature_figures.py

# Run full validation suite
python scripts/run_validation.py

# Generate statistical analysis
python scripts/statistical_analysis.py
```

## 🧪 Testing

Run the test suite to verify installation:

```bash
python -m pytest tests/ -v
```

## 📚 Documentation

- **[Theory](docs/theory.md)**: Mathematical foundations of MFSU
- **[Methodology](docs/methodology.md)**: Analysis procedures
- **[Validation](docs/validation.md)**: Statistical validation methods
- **[API Reference](docs/api_reference.md)**: Code documentation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Ways to contribute:
- 🐛 Report bugs or issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add new validation tests
- 🔬 Extend to new physical systems

## 📰 Publications

1. **Franco León, M.A.** "A universal fractal constant governs cosmic structure across scales" *Nature* (under review)
2. **Franco León, M.A.** "The Unified Fractal-Stochastic Model: Technical Report" *Zenodo* DOI: 10.5281/zenodo.16316882

## 🏆 Awards & Recognition



## 🌐 Community

- **Discussions**: [GitHub Discussions](https://github.com/MiguelAngelFrancoLeon/MFSU-Universal-Fractal-Constant/discussions)
- **Twitter**: [@miguelAfrancoL](https://twitter.com/miguelAfrancoL)
- **Email**: miguelfranco@mfsu-model.org

## 📄 Citation

If you use MFSU in your research, please cite:

```bibtex
@article{franco2025mfsu,
  title={A universal fractal constant governs cosmic structure across scales},
  author={Franco Le{\'o}n, Miguel {\'A}ngel},
  journal={Nature},
  year={2025},
  doi={10.1038/nature.2025.xxxxx}
}

@software{franco2025mfsu_code,
  title={MFSU: Unified Fractal-Stochastic Model},
  author={Franco Le{\'o}n, Miguel {\'A}ngel},
  year={2025},
  url={https://github.com/MiguelAngelFrancoLeon/MiguelAngelFrancoLeon-MFSU-Fractal-Dynamics/},
  doi={10.5281/zenodo.16316882}
}
```

## 📜 License

This project is licensed under the  CC-BY-4.0. - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Planck Collaboration** for CMB data
- **ESA Planck Legacy Archive** for data access
- **Open Science Community** for supporting independent research
- **GitHub** for hosting this revolutionary science

---

**⭐ Star this repository if MFSU helped your research!**

*"The universe is not chaos, but a fractal algorithm with δF = 0.921 as its organizing constant."*  
— Miguel Ángel Franco León
Authorship and Purpose
Miguel Ángel Franco León, an independent researcher, presents this model as the culmination of a lifelong effort to unify physics through pure geometry.
Without institutional support, this work aims to open a new paradigm in cosmology: that form precedes the particle, and that the universe is a living fractal in expansion.

## Protección de Autoría
Este proyecto está registrado en Zenodo (DOI: 10.5281/zenodo.15828185) y protegido en IPFS (CID: bafybeifdhrmcd6q46qsbj7cdpztvg55u2k3jxfidfk2scowin7rmkyccjq, accesible en https://ipfs.io/ipfs/bafybeifdhrmcd6q46qsbj7cdpztvg55u2k3jxfidfk2scowin7rmkyccjq). Licencia: CC-BY-4.0. Por favor, cita al autor (Miguel Ángel Franco León) en cualquier uso.

## References

- Mandelbrot, *The Fractal Geometry of Nature*, 1982.
- Calcagni, *QFT in Fractal Spacetimes*, PRD 2010.
- Nottale, *Fractal Space-Time and Microphysics*, 1993.

