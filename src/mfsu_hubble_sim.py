import numpy as np

# Parámetros basados en datos 2025 y MFSU (δ_F del PDF de Miguel Ángel Franco León)
delta_F = 0.921  # Constante fractal universal del MFSU
H0_CMB = 67.24  # Valor del universo temprano (e.g., SPT-3G + ACT, junio 2025)
H0_local = 73.0  # Valor del universo local (e.g., SH0ES)

# Cálculo del factor fractal (basado en scale-invariance del MFSU)
factor_fractal = np.log(H0_local / H0_CMB)  # Logaritmo para capturar auto-similaridad

# Resolución de H0 con MFSU
H0_resuelto = H0_CMB * (1 + delta_F * factor_fractal)

# Imprimir resultados
print(f"Factor fractal calculado: {factor_fractal:.3f}")
print(f"H0 resuelto con MFSU: {H0_resuelto:.2f} km/s/Mpc")

# Cálculo de discrepancia resuelta (para validación)
discrepancia_original = abs(H0_local - H0_CMB) / ((H0_local + H0_CMB) / 2) * 100
discrepancia_resuelta = abs(70.4 - H0_resuelto) / ((70.4 + H0_resuelto) / 2) * 100  # Comparado con JWST ~70.4
print(f"Discrepancia original: {discrepancia_original:.1f}%")
print(f"Discrepancia resuelta vs. JWST: {discrepancia_resuelta:.1f}%")
