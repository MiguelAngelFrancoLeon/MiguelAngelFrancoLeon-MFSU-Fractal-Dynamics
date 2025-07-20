#!/usr/bin/env python3
"""
Pipeline de Análisis MFSU para Resolución de Tensión de Hubble
Implementación del modelo Fractal-Estocástico para datos observacionales
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import emcee
import corner

class MFSUCosmology:
    """
    Modelo Fractal-Estocástico Unificado para Cosmología
    """
    
    def __init__(self):
        # Parámetros fractales fundamentales
        self.delta_fractal = 0.921  # Constante universal de colapso fractal
        self.d_fractal = 3 - self.delta_fractal  # Dimensión fractal efectiva
        self.hurst_exp = 0.7  # Exponente de Hurst
        
        # Escalas críticas
        self.d_crit = 100.0  # Mpc - escala de transición fractal-euclidiana
        self.d_norm = 1.0    # Mpc - escala de normalización
        self.z_transition = 0.23  # Redshift de transición
        
    def fractal_correction(self, z, local=True):
        """
        Calcula corrección fractal dependiente de escala
        """
        if local and z < self.z_transition:
            # Régimen local: z < z_∂
            alpha_fractal = 0.074  # Para mediciones locales
            return alpha_fractal * (z + 1e-6)**(-self.delta_fractal)
        else:
            # Régimen cosmológico: z > z_∂  
            omega_fractal = 0.079
            omega_matter = 0.315
            return (omega_fractal/omega_matter) * (1 + z)**(-self.delta_fractal)
    
    def stochastic_correction(self, z_array, sigma_H=0.02):
        """
        Genera correcciones estocásticas correlacionadas
        """
        n_points = len(z_array)
        # Ruido fraccionario con exponente de Hurst
        noise = np.random.randn(n_points)
        
        # Aplicar correlaciones de largo alcance
        for i in range(1, n_points):
            correlation = (i)**(-self.hurst_exp)
            noise[i] += correlation * noise[i-1]
            
        # Normalizar y escalar
        noise = noise / np.std(noise) * sigma_H
        return noise * (1 + z_array)**(-self.hurst_exp)
    
    def H_effective(self, z, H0=70.0, Omega_m=0.315, Omega_Lambda=0.685):
        """
        Parámetro de Hubble efectivo con correcciones MFSU
        """
        # Hubble estándar
        H_std = H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_Lambda)
        
        # Corrección fractal
        delta_frac = self.fractal_correction(z, local=(z < 0.01))
        
        # Corrección estocástica (simplificada para array único)
        if isinstance(z, np.ndarray):
            eta_stoch = self.stochastic_correction(z)
        else:
            eta_stoch = 0.0
            
        return H_std * (1 + delta_frac + eta_stoch)
    
    def luminosity_distance(self, z, H0=70.0, Omega_m=0.315, Omega_Lambda=0.685):
        """
        Distancia de luminosidad modificada por efectos fractales
        """
        c = 299792.458  # km/s
        
        def integrand(zp):
            return 1.0 / self.H_effective(zp, H0, Omega_m, Omega_Lambda)
        
        if isinstance(z, np.ndarray):
            d_L = []
            for zi in z:
                if zi > 0:
                    integral_result, _ = quad(integrand, 0, zi)
                    d_L.append((1 + zi) * c * integral_result)
                else:
                    d_L.append(0.0)
            return np.array(d_L)
        else:
            if z > 0:
                integral_result, _ = quad(integrand, 0, z)
                return (1 + z) * c * integral_result
            else:
                return 0.0
    
    def distance_modulus(self, z, H0=70.0, Omega_m=0.315, Omega_Lambda=0.685):
        """
        Módulo de distancia con correcciones fractales
        """
        d_L = self.luminosity_distance(z, H0, Omega_m, Omega_Lambda)
        return 5 * np.log10(d_L) + 25

class MFSUAnalysis:
    """
    Análisis estadístico del modelo MFSU con datos observacionales
    """
    
    def __init__(self):
        self.model = MFSUCosmology()
        
    def load_pantheon_data(self, filename=None):
        """
        Carga datos de supernovas Pantheon+ (simulados para demostración)
        """
        if filename is None:
            # Datos simulados representativos
            n_sn = 100
            z_sim = np.logspace(-3, 1, n_sn)  # z de 0.001 a 10
            
            # Módulo de distancia "observado" con ruido
            mu_true = self.model.distance_modulus(z_sim, H0=73.0)
            noise = np.random.normal(0, 0.15, n_sn)  # Error típico 0.15 mag
            mu_obs = mu_true + noise
            error_obs = np.full(n_sn, 0.15)
            
            return pd.DataFrame({
                'z': z_sim,
                'mu_obs': mu_obs,
                'mu_err': error_obs
            })
        else:
            # Cargar datos reales desde archivo
            return pd.read_csv(filename)
    
    def chi_squared(self, params, data):
        """
        Calcula chi-cuadrado para ajuste de parámetros
        """
        H0, Omega_m = params
        Omega_Lambda = 1 - Omega_m  # Universo plano
        
        mu_theory = self.model.distance_modulus(
            data['z'].values, H0, Omega_m, Omega_Lambda
        )
        
        chi2 = np.sum(((data['mu_obs'] - mu_theory) / data['mu_err'])**2)
        return chi2
    
    def fit_parameters(self, data):
        """
        Ajuste de parámetros por máxima verosimilitud
        """
        # Valores iniciales
        initial_params = [70.0, 0.3]  # H0, Omega_m
        
        # Optimización
        result = minimize(
            self.chi_squared, 
            initial_params,
            args=(data,),
            bounds=[(60, 80), (0.2, 0.4)],
            method='L-BFGS-B'
        )
        
        return result
    
    def mcmc_analysis(self, data, n_walkers=50, n_steps=5000):
        """
        Análisis MCMC para estimación bayesiana
        """
        def log_likelihood(params, data):
            if params[0] < 60 or params[0] > 80 or params[1] < 0.2 or params[1] > 0.4:
                return -np.inf
            return -0.5 * self.chi_squared(params, data)
        
        def log_prior(params):
            H0, Omega_m = params
            # Priors uniformes en rangos físicos
            if 60 < H0 < 80 and 0.2 < Omega_m < 0.4:
                return 0.0
            return -np.inf
        
        def log_probability(params, data):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(params, data)
        
        # Configuración MCMC
        n_dim = 2
        initial_guess = [70.0, 0.3]
        pos = initial_guess + 1e-2 * np.random.randn(n_walkers, n_dim)
        
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, log_probability, args=(data,)
        )
        
        # Ejecución
        sampler.run_mcmc(pos, n_steps, progress=True)
        
        return sampler
    
    def compare_with_planck(self, H0_fit):
        """
        Compara resultados con mediciones de Planck
        """
        H0_planck = 67.4  # km/s/Mpc
        H0_local = 73.8   # km/s/Mpc (SH0ES 2025)
        
        tension_planck = abs(H0_fit - H0_planck) / H0_planck
        tension_local = abs(H0_fit - H0_local) / H0_local
        
        print(f"H0 ajustado: {H0_fit:.2f} km/s/Mpc")
        print(f"Tensión con Planck: {tension_planck:.1%}")
        print(f"Tensión con mediciones locales: {tension_local:.1%}")
        
        return {
            'H0_fit': H0_fit,
            'tension_planck': tension_planck,
            'tension_local': tension_local
        }

def main_analysis():
    """
    Función principal para ejecutar análisis completo
    """
    print("=== Análisis MFSU para Tensión de Hubble ===\n")
    
    # Inicializar análisis
    analysis = MFSUAnalysis()
    
    # 1. Cargar datos
    print("1. Cargando datos de supernovas...")
    sn_data = analysis.load_pantheon_data()
    print(f"   Cargadas {len(sn_data)} supernovas")
    
    # 2. Ajuste de máxima verosimilitud
    print("\n2. Ajuste por máxima verosimilitud...")
    ml_result = analysis.fit_parameters(sn_data)
    H0_ml, Omega_m_ml = ml_result.x
    print(f"   H0 = {H0_ml:.2f} km/s/Mpc")
    print(f"   Ω_m = {Omega_m_ml:.3f}")
    print(f"   χ²_red = {ml_result.fun / (len(sn_data) - 2):.2f}")
    
    # 3. Análisis MCMC
    print("\n3. Ejecutando análisis MCMC...")
    sampler = analysis.mcmc_analysis(sn_data, n_walkers=32, n_steps=1000)
    
    # Resultados MCMC
    samples = sampler.get_chain(discard=300, flat=True)
    H0_mcmc = np.percentile(samples[:, 0], [16, 50, 84])
    Omega_m_mcmc = np.percentile(samples[:, 1], [16, 50, 84])
    
    print(f"   H0 = {H0_mcmc[1]:.2f} +{H0_mcmc[2]-H0_mcmc[1]:.2f} -{H0_mcmc[1]-H0_mcmc[0]:.2f}")
    print(f"   Ω_m = {Omega_m_mcmc[1]:.3f} +{Omega_m_mcmc[2]-Omega_m_mcmc[1]:.3f} -{Omega_m_mcmc[1]-Omega_m_mcmc[0]:.3f}")
    
    # 4. Comparación con observaciones
    print("\n4. Comparación con mediciones independientes:")
    comparison = analysis.compare_with_planck(H0_mcmc[1])
    
    # 5. Visualización
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Diagrama de Hubble
    plt.subplot(2, 2, 1)
    plt.errorbar(sn_data['z'], sn_data['mu_obs'], yerr=sn_data['mu_err'], 
                fmt='o', alpha=0.6, label='Datos simulados')
    
    z_theory = np.logspace(-3, 1, 100)
    mu_theory = analysis.model.distance_modulus(z_theory, H0_mcmc[1], Omega_m_mcmc[1])
    plt.plot(z_theory, mu_theory, 'r-', label='Modelo MFSU')
    
    plt.xlabel('Redshift z')
    plt.ylabel('Módulo de distancia μ')
    plt.xscale('log')
    plt.legend()
    plt.title('Diagrama de Hubble MFSU')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Residuos
    plt.subplot(2, 2, 2)
    mu_model = analysis.model.distance_modulus(sn_data['z'], H0_mcmc[1], Omega_m_mcmc[1])
    residuals = sn_data['mu_obs'] - mu_model
    plt.scatter(sn_data['z'], residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Redshift z')
    plt.ylabel('Residuos (obs - model)')
    plt.xscale('log')
    plt.title('Residuos del ajuste')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Corner plot (simplified)
    plt.subplot(2, 2, 3)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
    plt.xlabel('H₀ [km/s/Mpc]')
    plt.ylabel('Ω_m')
    plt.title('Distribución posterior')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Comparación H0
    plt.subplot(2, 2, 4)
    measurements = ['Planck\n(CMB)', 'SH0ES\n(Local)', 'MFSU\n(Ajuste)']
    h0_values = [67.4, 73.8, H0_mcmc[1]]
    h0_errors = [0.5, 1.0, (H0_mcmc[2]-H0_mcmc[0])/2]
    
    colors = ['blue', 'red', 'green']
    plt.errorbar(range(len(measurements)), h0_values, yerr=h0_errors,
                fmt='o', capsize=5, markersize=8)
    for i, (val, err, color) in enumerate(zip(h0_values, h0_errors, colors)):
        plt.scatter(i, val, color=color, s=100, zorder=5)
    
    plt.xticks(range(len(measurements)), measurements)
    plt.ylabel('H₀ [km/s/Mpc]')
    plt.title('Comparación H₀')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mfsu_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== RESULTADOS FINALES ===")
    print(f"Constante fractal δ = {analysis.model.delta_fractal}")
    print(f"Dimensión efectiva d_f = {analysis.model.d_fractal:.3f}")
    print(f"H₀ reconciliado = {H0_mcmc[1]:.2f} ± {(H0_mcmc[2]-H0_mcmc[0])/2:.2f} km/s/Mpc")
    print(f"Reducción de tensión: {(1-max(comparison['tension_planck'], comparison['tension_local']))*100:.1f}%")

if __name__ == "__main__":
    main_analysis()
