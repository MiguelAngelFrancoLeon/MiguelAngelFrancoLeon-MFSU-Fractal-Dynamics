# Unified Stochastic Fractal Model (MFSU)

The Unified Stochastic Fractal Model (MFSU) is a theoretical-experimental framework that integrates fractal geometry, stochastic processes, and quantum field theory to model complex systems in superconductivity, gas dynamics, and cosmology. It unifies classical and quantum dynamics, predicting critical properties like superconducting transition temperatures (\( T_c \)) and cosmological density fields.

## Formulations
### Original Formula
The MFSU models dynamics as a stochastic process on a fractal structure:
dq/dt = -∇V(q) + η(t) + κ D_f q
where \( q \) is the state vector, \( V(q) \) is the potential, \( η(t) \) is Gaussian noise, \( D_f \) is the fractal dimension, and \( κ \) is the coupling constant.

### Refined Formula
The refined MFSU incorporates quantum dynamics via the Wigner function:
∂W/∂t = -p·∇_q W + ∇_q V·∇_p W + (ħ²/2) Σ (-1)ⁿ/(n!) (1/2)ⁿ ∇_q^(2n+1) V·∇_p^(2n+1) W + κ D_f Δ_f W
See `Documentation/MFSU_Theory.pdf` for details and `Code/MFSU_Wigner.py` for the implementation.

# Unified Fractal-Stochastic Model (MFSU) for CMB Low-ℓ Anomalies

This repository contains the code, data, and documentation for the Unified Fractal-Stochastic Model (MFSU), which introduces a fractal dimension (\(d_f = 1.53 \pm 0.03\)) to explain the low-\( \ell \) anomaly in the cosmic microwave background (CMB). The model derives \( C_\ell \propto \ell^{-(d_f - 1)} \), outperforming ΛCDM by 33.5% in RMSE (0.0123 vs 0.0185) for \( \ell \leq 30 \) (Planck 2018 TT data).

## Repository Structure
- **Cosmo_Code/**: Simulation scripts for CMB analysis.
  - `CMB_Simulation_Low_Ell.py`: Fits \( d_f \) to Planck 2018 TT data (\( \ell \leq 30 \)).
  - `preprocess_planck.py`: Preprocesses Planck 2018 TT, EE, TE data.
- **Data/CMB/**: Preprocessed Planck 2018 data.
  - `Planck_TT_2018_low_ell.csv`: TT data for \( \ell \leq 30 \).
  - `Planck_TT_2018.csv`: Full TT data (\( \ell \leq 2508 \)).
- **Results/CMB/**: Output figures.
  - `cmb_mfsu_comparison_low_ell.png`: Comparison of MFSU vs ΛCDM.
- **Docs/**: Documentation.
  - `CMB_MFSU_Final.tex`: LaTeX source.
  - `CMB_MFSU_Final.pdf`: Compiled paper.

## Reproducing the Results
1. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib scipy scikit-learn

## Resources
- **Data**: `/Data/Perovskites/RvT_300K.csv` (superconductivity data)
- **Code**: 
  - `/Code/MFSU_Simulation.py` (original formula)
  - `/Code/MFSU_Wigner.py` (refined formula)
- **Examples**: 
  - `/Examples/Cosmo_Simulation.ipynb` (cosmology)
  - `/Examples/Gas_Dynamics.py` (gas dynamics)
- **Documentation**: `/Documentation/MFSU_Theory.pdf`
## Contents

- Interactive simulations (Jupyter)
- Symbolic action derivation
- Propagator and critical temperature visualizations

## Citation
Franco León, M. Á. (2025). Unified Stochastic Fractal Model (MFSU). Zenodo. [DOI:0009-0003-9492-385X]


## Requirements

See `requirements.txt`.

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0).


## Contact
Miguel Ángel Franco León, mf410360@gmail.com [0009-0003-9492-385X](https://orcid.org/0009-0003-9492-385X)

## Citation
Franco León, M. Á. (2025). Unified Stochastic Fractal Model (MFSU). Zenodo. [DOI:0009-0003-9492-385X]

## References

- Mandelbrot, *The Fractal Geometry of Nature*, 1982.
- Calcagni, *QFT in Fractal Spacetimes*, PRD 2010.
- Nottale, *Fractal Space-Time and Microphysics*, 1993.

