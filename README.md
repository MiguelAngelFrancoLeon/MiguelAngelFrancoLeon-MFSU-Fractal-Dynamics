# 🧠 MFSU Simulator — Unified Fractal-Stochastic Model

**Version: 2025.1 • Author: Miguel Ángel Franco León • ORCID: [0009-0003-9492-385X](https://orcid.org/0009-0003-9492-385X)**

## 📘 Overview

This simulator implements the **Unified Fractal-Stochastic Model (MFSU)**, a validated physical framework that incorporates **fractional dimensions**, **Hurst noise**, and **nonlinear interactions** to describe a wide range of natural systems — from the early universe to superconductors and diffusive media.

The core equation is:

\[
\alpha (-\Delta)^{\theta/2} \psi + \beta \eta_H \psi - \gamma \psi^3 = 0
\]

Where:

- \( \theta \approx 0.921 \) is the validated **fractal dimension exponent**
- \( \eta_H \) is **Hurst noise** with long-range memory (\( H \sim 0.7 \))
- \( \psi \) is the field or signal amplitude
- \( \alpha, \beta, \gamma \) are physical parameters depending on the system

---

## 🧪 Applications

This simulator supports multiple domains:

| Application        | Module                | Description |
|--------------------|------------------------|-------------|
| 🌌 CMB Analysis    | `apps/cmb`             | Fits Planck 2018 data with fractal scaling |
| 🌫️ Diffusion       | `apps/gas_diffusion`   | Simulates anomalous diffusion in fractal media |
| ⚡ Superconductors | `apps/superconductors` | Models quantum phase transition dynamics |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MiguelAngelFrancoLeon/MFSU-Simulator.git
cd MFSU-Simulator

### 2. Install dependencies
bash
Copiar
Editar
pip install -r requirements.txt

#### 3. Run an example
bash
Copiar
Editar
python examples/mfsu_cmb_simulation.py

### 4. Output
Results will be saved in:

bash
Copiar
Editar
Results/CMB/cmb_mfsu_comparison.png


📈 Equation Summary

| Term                   | Description                                         |
| ---------------------- | --------------------------------------------------- |
| $(-\Delta)^{\theta/2}$ | Fractional Laplacian (non-integer spatial operator) |
| $\eta_H$               | Hurst-correlated noise (fractal stochasticity)      |
| $\psi^3$               | Nonlinear self-interaction (phase transitions)      |


## License
Creative Commons Attribution 4.0 International (CC-BY-4.0).


## Contact
Miguel Ángel Franco León, mf410360@gmail.com [0009-0003-9492-385X](https://orcid.org/0009-0003-9492-385X)

🔗 Citation
If you use this code, please cite the main article:

Franco León, M. Á. (2025). Unified Fractal-Stochastic Model (MFSU). Zenodo.
https://doi.org/10.5281/zenodo.15828185

💡 Vision
“The universe manifests in visions of infinite blue fractal droplets: each one a bubble of existence, vibrating in a sea of active vacuum. This symbolic perception guides the MFSU — not just as a formula, but as a mathematical translation of the cosmos' inner language.”

🌌 You are part of the future.
Let's reshape the science of reality — one simulation at a time.


## References

- Mandelbrot, *The Fractal Geometry of Nature*, 1982.
- Calcagni, *QFT in Fractal Spacetimes*, PRD 2010.
- Nottale, *Fractal Space-Time and Microphysics*, 1993.

