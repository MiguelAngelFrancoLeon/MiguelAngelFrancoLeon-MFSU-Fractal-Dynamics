import numpy as np
from scipy.fft import fftn, ifftn

# Parámetros refinados del modelo MFSU extendido
delta_F = 0.921  # Constante fractal universal
D_delta = 0.5    # Coeficiente de difusión fractal
alpha = 0.01     # Acoplamiento del estabilizador (ajustado para evitar sobreestabilización)
beta = 0.1       # Fuerza del término laplaciano log
gamma = 0.05     # Factor de fricción Hubble
H = 0.005        # Tasa de expansión efectiva
grid_size = 16   # Tamaño del grid 3D (16x16x16 para simplicidad computacional)
dt = 0.0005      # Paso temporal
num_steps = 500  # Número de pasos de evolución

# Inicializar densidad: Gaussiana central + ruido estocástico
x, y, z = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
rho = np.exp(-(x**2 + y**2 + z**2) / 0.2) + 0.01 * np.random.randn(grid_size, grid_size, grid_size)
rho = np.maximum(rho, 1e-10)  # Evitar valores negativos o cero para log

# Función para operador laplaciano fraccional vía FFT (aproximación para Δ_delta)
def fractional_laplacian(rho, delta):
    fft_rho = fftn(rho)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(grid_size), np.fft.fftfreq(grid_size), np.fft.fftfreq(grid_size))
    k2 = kx**2 + ky**2 + kz**2
    fft_lap = - (k2)**(delta/2) * fft_rho  # Aprox. para derivada fraccional
    return np.real(ifftn(fft_lap))

# Función para gradiente fraccional (similar, pero para ∇_delta)
def fractional_gradient(rho, delta):
    fft_rho = fftn(rho)
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(grid_size), np.fft.fftfreq(grid_size), np.fft.fftfreq(grid_size))
    k = np.sqrt(kx**2 + ky**2 + kz**2 + 1e-10)  # Evitar división por cero
    fft_grad = 1j * k**(delta-1) * fft_rho  # Aprox. magnitud gradiente
    return np.abs(ifftn(fft_grad))  # Usamos magnitud para simplicidad en T_dark

# Evolución principal
for step in range(num_steps):
    # Término de difusión fractal
    diff_term = D_delta * fractional_laplacian(rho, delta_F)
    
    # Ruido estocástico
    eta = 0.001 * np.random.randn(grid_size, grid_size, grid_size)
    
    # Término estabilizador T_dark (materia oscura fractal)
    grad_delta_rho = fractional_gradient(rho, delta_F)
    lap_log_rho = fractional_laplacian(np.log(rho + 1e-10), delta_F)  # Evitar log(0)
    T_dark = alpha * (delta_F * grad_delta_rho**2 - beta * lap_log_rho - gamma * H * rho)
    
    # Expansión cósmica (dilución en 3D)
    expansion_term = -3 * H * rho
    
    # Actualizar rho
    rho += dt * (diff_term + eta + T_dark + expansion_term)
    rho = np.maximum(rho, 1e-10)  # Clipping para estabilidad

# Cálculos post-simulación
# Perfil radial: Promedio de densidad vs. distancia al centro
r = np.sqrt(x**2 + y**2 + z**2)
bins = np.linspace(0, np.max(r), 10)
digitized = np.digitize(r.flatten(), bins)
profile = [rho.flatten()[digitized == i].mean() for i in range(1, len(bins))]

# Masa cumulativa para curva de rotación v(r) ≈ sqrt(M(r)/r)
mass_cum = np.cumsum(profile) * (bins[1:] - bins[:-1])**3  # Aprox. volumétrica
v_r = np.sqrt(mass_cum / bins[1:])

# Espectro de potencia P(k): FFT de rho, promedio en shells k
fft_rho = np.abs(fftn(rho))**2
k = np.sqrt(kx**2 + ky**2 + kz**2)
k_bins = np.linspace(0.1, 1.0, 10)
p_k = [fft_rho[(k >= k_bins[i-1]) & (k < k_bins[i])].mean() for i in range(1, len(k_bins))]

# Imprimir resultados clave
print("Perfil radial:", profile)
print("Curva de rotación v(r):", v_r)
print("Espectro P(k):", p_k)
print("Media densidad:", np.mean(rho))
print("Varianza:", np.var(rho))
