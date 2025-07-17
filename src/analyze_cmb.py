import healpy as hp
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Cargar FITS
fits_file = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
try:
    map_cmb = hp.read_map(fits_file, field=0)
except Exception as e:
    print(f"Error al cargar FITS: {e}")
    exit()

# Dimensión fractal
def fractal_dimension_cmb(map_data):
    threshold = np.mean(map_data)
    binary = (map_data > threshold).astype(int)
    nside = hp.get_nside(map_data)
    box_sizes = np.logspace(np.log10(4), np.log10(nside//2), num=10, dtype=int)
    counts = []
    for size in box_sizes:
        count = 0
        pixels_per_box = size * size
        for i in range(0, len(map_data), pixels_per_box):
            if np.any(binary[i:i+pixels_per_box]):
                count += 1
        counts.append(count)
    slope, _, _, _, _ = linregress(np.log(1/box_sizes), np.log(counts))
    return slope

# Espectro de potencia
def power_spectrum_cmb(map_data):
    cl = hp.anafast(map_data, lmax=1000)
    ell = np.arange(len(cl))
    slope, _ = np.polyfit(np.log(ell[1:]), np.log(cl[1:]), 1)
    return slope, cl, ell

# Resultados CMB
partial_cmb = fractal_dimension_cmb(map_cmb)
print(f"Dimensión fractal CMB: {partial_cmb:.3f}")
slope_cmb, cl, ell = power_spectrum_cmb(map_cmb)
print(f"Pendiente del espectro CMB: {slope_cmb:.3f}")

# Guardar gráfico
plt.figure(figsize=(8, 6))
plt.loglog(ell[1:], cl[1:], label="CMB Power Spectrum")
plt.xlabel("Multipole (ℓ)")
plt.ylabel("C_ℓ")
plt.title(f"CMB Power Spectrum (Slope ≈ {slope_cmb:.3f})")
plt.legend()
plt.grid(True)
plt.savefig("figures/cmb_power_spectrum.png", dpi=300)
plt.close()

# Simulación MFSU
def simulate_mfsu_1d(N=100, steps=1000, delta_t=0.01, dx=0.1, alpha=1.0, beta=0.1, gamma=0.1, partial=0.921):
    from scipy.special import gamma
    def gl_coefficients(order, n):
        coeffs = [(-1)**k * gamma(order + 1) / (gamma(k + 1) * gamma(order - k + 1)) for k in range(n + 1)]
        return np.array(coffs)
    def fractional_gradient(psi, dx, order, n=5):
        coeffs = gl_coefficients(order, n)
        grad = np.zeros_like(psi)
        for i in range(len(psi)):
            for k in range(n + 1):
                if i - k >= 0:
                    grad[i] += coeffs[k] * psi[i - k]
        return grad / (dx ** order)
    def generate_1f_noise(N, steps, alpha_noise=1.0):
        freq = np.linspace(1, N//2, N//2)
        power_spectrum = 1 / (freq ** alpha_noise)
        noise = np.zeros((steps, N))
        for t in range(steps):
            phase = np.random.uniform(0, 2*np.pi, N//2)
            amplitude = np.sqrt(power_spectrum)
            fourier = np.concatenate([amplitude * np.exp(1j * phase), (amplitude * np.exp(1j * phase))[::-1]])
            noise[t, :] = np.fft.ifft(fourier).real * np.sqrt(N)
        return noise
    psi = np.random.normal(0, 0.1, N)
    history = np.zeros((steps, N))
    xi = generate_1f_noise(N, steps)
    for t in range(steps):
        grad = fractional_gradient(psi, dx, partial)
        psi += delta_t * (alpha * grad + beta * xi[t] * psi - gamma * psi**3)
        history[t, :] = psi
    np.savetxt('data/soliton_1d.csv', np.vstack((np.linspace(0, N*dx, N), psi)).T, delimiter=',')
    return history, xi

def fractal_dimension_sim(psi):
    binary = (psi > np.mean(psi)).astype(int)
    box_sizes = np.logspace(1, np.log10(psi.shape[1]//2), num=10, dtype=int)
    counts = []
    for size in box_sizes:
        count = 0
        for i in range(0, psi.shape[1], size):
            if np.any(binary[:, i:i+size]):
                count += 1
        counts.append(count)
    slope, _, _, _, _ = linregress(np.log(1/box_sizes), np.log(counts))
    return slope

def plot_power_spectrum_sim(xi):
    power = np.abs(np.fft.fft(xi, axis=1))**2
    freq = np.fft.fftfreq(xi.shape[1])
    slope = np.polyfit(np.log(freq[1:xi.shape[1]//2]), np.log(power[:, 1:xi.shape[1]//2].mean(axis=0)), 1)[0]
    plt.figure(figsize=(8, 6))
    plt.loglog(freq[1:xi.shape[1]//2], power[:, 1:xi.shape[1]//2].mean(axis=0), label="Simulated Noise")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.title(f"Simulated Power Spectrum (Slope ≈ {slope:.3f})")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/noise_spectrum.png", dpi=300)
    plt.close()
    return slope

# Ejecutar simulación
history, xi = simulate_mfsu_1d()
dim_sim = fractal_dimension_sim(history)
slope_sim = plot_power_spectrum_sim(xi)
print(f"Dimensión fractal simulada: {dim_sim:.3f}")
print(f"Pendiente del espectro simulada: {slope_sim:.3f}")

# Comparación
print(f"Diferencia en ∂: {abs(partial_cmb - dim_sim):.3f}")
print(f"Diferencia en pendiente: {abs(slope_cmb - slope_sim):.3f}")
