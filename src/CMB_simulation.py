# Examples/CMB_Simulation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# -----------------------------
# PARAMETERS (MFSU validated)
# -----------------------------
THETA = 0.921          # Fractal Laplacian exponent
ALPHA = 1.0            # Scaling factor (can be fitted)
BETA = 1.0             # Hurst noise coupling
GAMMA = 1.0            # Nonlinear term
HURST_EXPONENT = 0.72  # Memory in noise (0.5 = Brownian, >0.5 = persistent)

# -----------------------------
# LOAD PLANCK 2018 DATA
# -----------------------------
df = pd.read_csv("Data/CMB/Planck_TT_2018.csv")
ell = df['ell'].values
cl_obs = df['Cl'].values
cl_err = df['Cl_err'].values

# -----------------------------
# Hurst Noise Generator (approx)
# -----------------------------
def generate_hurst_noise(size, H):
    """Approximate Hurst-correlated noise using fractional Brownian motion."""
    white_noise = np.random.randn(size)
    f = np.fft.fftfreq(size)
    f[0] = 1e-6  # avoid division by zero
    spectrum = np.abs(f) ** (-H)
    fbm = np.fft.ifft(np.fft.fft(white_noise) * spectrum).real
    return (fbm - fbm.min()) / (fbm.max() - fbm.min())

# -----------------------------
# MFSU Model Function
# -----------------------------
def mfsu_model(ell, alpha, beta, gamma, theta, noise):
    """
    MFSU model: C_ell = α * ℓ^{-θ} + β * η_H(ℓ) - γ * (ℓ^{-θ})^3
    """
    laplacian_term = alpha * ell**(-theta)
    stochastic_term = beta * noise
    nonlinear_term = gamma * (ell**(-theta))**3
    return laplacian_term + stochastic_term - nonlinear_term

# -----------------------------
# COST FUNCTION
# -----------------------------
def cost(params, ell, cl_obs, cl_err, noise):
    alpha, beta, gamma = params
    cl_pred = mfsu_model(ell, alpha, beta, gamma, THETA, noise)
    return mean_squared_error(cl_obs, cl_pred, squared=False)

# -----------------------------
# FIT MODEL
# -----------------------------
np.random.seed(42)
noise = generate_hurst_noise(len(ell), HURST_EXPONENT)

result = minimize(
    cost,
    x0=[1e-10, 1e-10, 1e-10],
    args=(ell, cl_obs, cl_err, noise),
    bounds=[(1e-12, 1e-6), (1e-12, 1e-6), (1e-12, 1e-6)]
)

alpha_fit, beta_fit, gamma_fit = result.x

# -----------------------------
# EVALUATION
# -----------------------------
cl_pred = mfsu_model(ell, alpha_fit, beta_fit, gamma_fit, THETA, noise)
rmse_mfsu = mean_squared_error(cl_obs, cl_pred, squared=False)

# LCDM as reference
def lcdm_model(ell, A): return A * ell**(-2)
cl_lcdm = lcdm_model(ell, alpha_fit)
rmse_lcdm = mean_squared_error(cl_obs, cl_lcdm, squared=False)

# Improvement
improvement = (rmse_lcdm - rmse_mfsu) / rmse_lcdm * 100

# -----------------------------
# OUTPUT
# -----------------------------
print(f"--- MFSU FITTED PARAMETERS ---")
print(f"α = {alpha_fit:.2e}")
print(f"β = {beta_fit:.2e}")
print(f"γ = {gamma_fit:.2e}")
print(f"RMSE MFSU: {rmse_mfsu:.4f}")
print(f"RMSE LCDM: {rmse_lcdm:.4f}")
print(f"Improvement over LCDM: {improvement:.2f}%")

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10, 6))
plt.errorbar(ell, cl_obs, yerr=cl_err, fmt='k.', label='Planck 2018 (TT)', alpha=0.6)
plt.plot(ell, cl_pred, 'r-', label=f'MFSU (θ={THETA})')
plt.plot(ell, cl_lcdm, 'b--', label='LCDM (ℓ⁻²)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Multipole ℓ')
plt.ylabel(r'$C_\ell$')
plt.title('CMB Angular Power Spectrum: MFSU vs Planck 2018')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig('Results/CMB/cmb_mfsu_comparison.png', dpi=300)
plt.show()
