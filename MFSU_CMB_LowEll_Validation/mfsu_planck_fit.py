import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample

# Cargar datos Planck TT
data = pd.read_csv("Planck_TT_low_ell.csv")
ell = data["ell"].values
cl_obs = data["Cl"].values
cl_err = data["Cl_err"].values

# Modelo MFSU
def cl_mfsu(ell, A, d_f):
    return A * ell ** -(d_f - 1)

# Función de costo
def cost(params, ell, cl_obs):
    A, d_f = params
    cl_pred = cl_mfsu(ell, A, d_f)
    return mean_squared_error(cl_obs, cl_pred, squared=False)

# Ajuste
result = minimize(cost, x0=[1e-10, 1.5], args=(ell, cl_obs), bounds=[(1e-12, 1e-8), (1.2, 2.0)])
A_best, d_f_best = result.x

# Bootstrap
d_f_boots = []
for _ in range(500):
    idx = np.random.choice(len(ell), len(ell), replace=True)
    ell_b, cl_b = ell[idx], cl_obs[idx]
    res = minimize(cost, x0=[A_best, d_f_best], args=(ell_b, cl_b))
    d_f_boots.append(res.x[1])
d_f_err = np.std(d_f_boots) * 1.96

# RMSE y mejora
cl_pred = cl_mfsu(ell, A_best, d_f_best)
rmse_mfsu = mean_squared_error(cl_obs, cl_pred, squared=False)
cl_lcdm = cl_mfsu(ell, A_best, 2.0)
rmse_lcdm = mean_squared_error(cl_obs, cl_lcdm, squared=False)
improvement = (rmse_lcdm - rmse_mfsu) / rmse_lcdm * 100

# Gráfico
plt.figure(figsize=(10, 6))
plt.errorbar(ell, cl_obs, yerr=cl_err, fmt='ko', label='Planck 2018 (TT)')
plt.plot(ell, cl_pred, 'r-', label=f'MFSU ($d_f$ = {d_f_best:.2f} ± {d_f_err:.2f})')
plt.plot(ell, cl_lcdm, 'b--', label='ΛCDM ($d_f$ = 2.0)')
plt.yscale("log")
plt.xlabel('Multipole moment $\\ell$')
plt.ylabel('$C_\\ell$ [$\\mu K^2$]')
plt.title('CMB Angular Power Spectrum: MFSU vs ΛCDM (Low-$\\ell$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mfsu_vs_lcdm_fit.png")
plt.show()

# Resultados por consola
print(f"d_f = {d_f_best:.2f} ± {d_f_err:.2f}")
print(f"RMSE MFSU: {rmse_mfsu:.4f}")
print(f"RMSE LCDM: {rmse_lcdm:.4f}")
print(f"Mejora porcentual: {improvement:.2f}%")
