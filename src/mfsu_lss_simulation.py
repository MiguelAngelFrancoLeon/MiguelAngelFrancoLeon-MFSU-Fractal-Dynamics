# MFSU Large-Scale Structure Simulation
# Author: Miguel Ángel Franco León

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del modelo MFSU
N_points = 100000
df = 1.52
r_max = 100

def sample_fractal_radius(df, size, r_max):
    u = np.random.rand(size)
    return (u * r_max**df)**(1 / df)

# Generar puntos esféricos
r = sample_fractal_radius(df, N_points, r_max)
theta = np.arccos(1 - 2 * np.random.rand(N_points))
phi = 2 * np.pi * np.random.rand(N_points)

# Convertir a coordenadas cartesianas
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

# Guardar datos
df_coords = pd.DataFrame({'x': x, 'y': y, 'z': z})
df_coords.to_csv("lss_fractal_points.csv", index=False)

# Graficar
plt.figure(figsize=(8, 8))
plt.hexbin(x, y, gridsize=250, cmap='viridis', bins='log')
plt.colorbar(label='log(density)')
plt.title('MFSU Large-Scale Structure Simulation (Fractal dimension $d_f=1.52$)')
plt.xlabel('x (Mpc)')
plt.ylabel('y (Mpc)')
plt.axis('equal')
plt.tight_layout()
plt.savefig("lss_fractal_hexbin.png")
plt.show()
