import numpy as np
import matplotlib.pyplot as plt

def fractal_emission_spectrum(frequencies, df=1.01, gamma=0.05, f0=589e12, sigma=2e12):
    k = frequencies / f0
    intensity = ((k**2)**((df - 2)/2) + gamma * np.log(k**2)) * np.exp(-((frequencies - f0)**2) / (2 * sigma**2))
    return intensity / np.max(intensity)

if __name__ == "__main__":
    freqs = np.linspace(580e12, 600e12, 1000)
    spectrum = fractal_emission_spectrum(freqs)

    plt.plot(freqs / 1e12, spectrum, label="Fractal-Stochastic Emission")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Normalized Intensity")
    plt.title("Enhanced Sodium Emission Spectrum")
    plt.legend()
    plt.grid(True)
    plt.savefig("espectro_comparacion.png")
    plt.show()
