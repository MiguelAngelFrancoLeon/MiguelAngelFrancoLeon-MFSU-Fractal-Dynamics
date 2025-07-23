import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Parámetros del MFSU (de tu PDF)
d_F = 2.079  # Dimensión fractal efectiva
delta_F = 3 - d_F  # Constante Franco ≈ 0.921
A = 1.0  # Normalización (puedes ajustar)

# Generar multipoles l (como en datos CMB, de 10 a 3000)
l = np.logspace(1, 3.5, 200)  # 200 puntos para simulación suave

# Power spectrum teórico (fractal puro)
C_l_theory = A * l ** (-d_F)

# Componente estocástica: Agregar ruido Gaussiano (simula fluctuaciones estocásticas)
noise_level = 0.05  # Ajusta esto (menor = menos ruido, más cerca de teórico)
noise = np.random.normal(0, noise_level * C_l_theory, len(l))
C_l_sim = C_l_theory + noise

# Función para ajustar (validar el modelo)
def power_func(l, A_fit, d_F_fit):
    return A_fit * l ** (-d_F_fit)

# Ajuste con curve_fit
popt, pcov = curve_fit(power_func, l, C_l_sim, p0=[1.0, 2.0])
A_fit = popt[0]
d_F_fit = popt[1]
delta_F_fit = 3 - d_F_fit

# Imprimir resultados
print(f"Parámetros teóricos del MFSU:")
print(f"  - d_F: {d_F:.3f}")
print(f"  - δ_F: {delta_F:.3f}")
print("\nResultados de la simulación:")
print(f"  - d_F ajustado: {d_F_fit:.3f}")
print(f"  - δ_F ajustado: {delta_F_fit:.3f}")

# Visualización
plt.figure(figsize=(8, 6))
plt.loglog(l, C_l_theory, label='Teórico (fractal puro)', color='blue')
plt.loglog(l, C_l_sim, 'o', label='Simulado con ruido estocástico', color='red', alpha=0.5)
plt.loglog(l, power_func(l, *popt), '--', label=f'Fit (δ_F ≈ {delta_F_fit:.3f})', color='green')
plt.xlabel('Multipole l')
plt.ylabel('C_l (Power Spectrum)')
plt.title('Simulación del MFSU en Power Spectrum CMB')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.savefig('mfsu_cmb_sim.png')  # Guarda el plot
plt.show()  # Muestra en pantalla si ejecutas localmente
print("¡Simulación completada! Plot guardado como 'mfsu_cmb_sim.png'.")
