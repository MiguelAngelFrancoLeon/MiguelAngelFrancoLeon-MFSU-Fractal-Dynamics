# Cosmo_Code/CMB_Simulation_Low_Ell.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# Load Planck 2018 TT data
data = pd.read_csv('Data/CMB/Planck_TT_2018_low_ell.csv')
ell = data['ell'].values
cl_obs = data['Cl'].values
cl_err = data['Cl_err'].values

# MFSU model: C_l = A * l^-(d_f - 1)
def cl_mfsu(ell, A, d_f):
    return A * ell ** -(d_f - 1)

# Cost function
def cost(params, ell, cl_obs, cl_err):
    A, d_f = params
    cl_pred = cl_mfsu(ell, A, d_f)
    return mean_squared_error(cl_obs, cl_pred, squared=False)

# Fit d_f
result = minimize(cost, x0=[1e-10, 1.5], args=(ell, cl_obs, cl_err), 
                 bounds=[(1e-12, 1e-8), (1.2, 2.0)])
A_best, d_f_best = result.x

# Bootstrap uncertainty
n_boot = 1000
d_f_boots = []
for _ in range(n_boot):
    ell_boot, cl_boot, cl_err_boot = resample(ell, cl_obs, cl_err)
    result_boot = minimize(cost, x0=[1e-10, 1.5], args=(ell_boot, cl_boot, cl_err_boot))
    d_f_boots.append(result_boot.x[1])
d_f_err = np.std(d_f_boots) * 1.96  # 95% CI
print(f"d_f = {d_f_best:.2f} ± {d_f_err:.2f}, A = {A_best:.2e}")

# RMSE
cl_pred = cl_mfsu(ell, A_best, d_f_best)
rmse_mfsu = mean_squared_error(cl_obs, cl_pred, squared=False)
cl_lcdm = cl_mfsu(ell, A_best, 2.0)
rmse_lcdm = mean_squared_error(cl_obs, cl_lcdm, squared=False)
improvement = (rmse_lcdm - rmse_mfsu) / rmse_lcdm * 100
print(f"RMSE MFSU: {rmse_mfsu:.4f}, RMSE LCDM: {rmse_lcdm:.4f}, Improvement: {improvement:.2f}%")

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(ell, cl_obs, yerr=cl_err, fmt='k.', label='Planck 2018 (TT, l ≤ 30)')
plt.plot(ell, cl_pred, 'r-', label=f'MFSU ($d_f$ = {d_f_best:.2f} ± {d_f_err:.2f})')
plt.plot(ell, cl_lcdm, 'b--', label='LCDM ($d_f$ = 2)')
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Multipole moment ($\\ell$)')
plt.ylabel('$C_\\ell$ (μK$^2$)')
plt.title('CMB Angular Power Spectrum: MFSU vs Planck 2018 (Low-$\\ell$)')
plt.legend()
plt.grid(True)
plt.savefig('Results/CMB/cmb_mfsu_comparison_low_ell.png')
plt.show()
