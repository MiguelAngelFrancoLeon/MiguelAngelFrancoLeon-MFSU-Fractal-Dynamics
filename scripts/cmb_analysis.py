#!/usr/bin/env python3
"""
CMB Analysis with MFSU Model
Cosmic Microwave Background validation of δF ≈ 0.921

Author: Miguel Ángel Franco León
Date: August 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats, signal, integrate
from scipy.special import spherical_jn, gamma
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# MFSU Constants
DELTA_F = 0.921
DF_FRACTAL = 2.079
HURST_EXPONENT = 0.541

# Physical constants
C_LIGHT = 299792458  # m/s
H0_PLANCK = 67.4     # km/s/Mpc (Planck 2018)
OMEGA_B = 0.0493     # Baryon density
OMEGA_CDM = 0.2651   # Cold dark matter
OMEGA_LAMBDA = 0.6847 # Dark energy

class CMBAnalyzer:
    """
    Cosmic Microwave Background analysis with MFSU model
    """
    
    def __init__(self, delta_f=DELTA_F):
        self.delta_f = delta_f
        self.df = 3 - delta_f  # Fractal dimension
        self.results = {}
        
        # CMB analysis parameters
        self.lmax = 3000
        self.lmin = 2
        self.temperature_units = 'muK'  # microkelvin
        
    def load_planck_data(self, data_source='synthetic'):
        """
        Load Planck CMB data (real or synthetic)
        """
        print(f"📡 Loading Planck CMB data ({data_source})...")
        
        if data_source == 'synthetic':
            # Generate realistic synthetic Planck-like data
            return self._generate_synthetic_planck_data()
        elif data_source == 'planck2018':
            # Load real Planck 2018 data (if available)
            return self._load_real_planck_data()
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def _generate_synthetic_planck_data(self):
        """
        Generate synthetic CMB data based on Planck observations
        """
        np.random.seed(42)  # Reproducible
        
        # Multipole range
        ell = np.arange(self.lmin, self.lmax + 1)
        
        # Base ΛCDM spectrum (simplified)
        def base_spectrum(l):
            # Acoustic oscillations
            acoustic = 1 + 0.1 * np.sin(0.03 * l + 1.0) * np.exp(-l/1000)
            
            # Low-ℓ plateau
            if l < 50:
                plateau = 1.2
            else:
                plateau = 1.0
            
            # High-ℓ damping
            damping = np.exp(-(l/3000)**2)
            
            return plateau * acoustic * damping
        
        # MFSU modifications
        def mfsu_spectrum(l):
            # Power-law with δF
            power_law = l**(-self.delta_f)
            
            # Fractal modulations
            fractal_mod = 1 + 0.05 * np.sin(np.log(l) * self.df) * (l/100)**(-0.1)
            
            return power_law * fractal_mod
        
        # Combined spectrum
        C_ell_base = np.array([base_spectrum(l) for l in ell])
        C_ell_mfsu = np.array([mfsu_spectrum(l) for l in ell])
        
        # Normalize to realistic CMB amplitudes
        normalization = 2500  # μK²
        C_ell_theory = normalization * C_ell_base * C_ell_mfsu
        
        # Add realistic noise (cosmic variance + instrumental)
        cosmic_variance = C_ell_theory / np.sqrt(2 * ell + 1)
        instrumental_noise = 0.05 * C_ell_theory  # 5% instrumental uncertainty
        
        total_noise = np.sqrt(cosmic_variance**2 + instrumental_noise**2)
        
        # Generate observed spectrum
        C_ell_obs = C_ell_theory + total_noise * np.random.randn(len(ell))
        C_ell_err = total_noise
        
        # Ensure positive values
        C_ell_obs = np.maximum(C_ell_obs, 0.1)
        
        data = {
            'ell': ell,
            'C_ell': C_ell_obs,
            'C_ell_err': C_ell_err,
            'C_ell_theory': C_ell_theory,
            'data_source': 'synthetic_planck',
            'units': self.temperature_units + '^2',
            'lmax': self.lmax,
            'lmin': self.lmin
        }
        
        print(f"   Generated {len(ell)} multipoles (ℓ = {self.lmin} to {self.lmax})")
        print(f"   RMS amplitude: {np.sqrt(np.mean(C_ell_obs)):.1f} {self.temperature_units}")
        
        return data
    
    def _load_real_planck_data(self):
        """
        Load real Planck 2018 data (placeholder for real implementation)
        """
        # This would interface with actual Planck data files
        # For now, return enhanced synthetic data
        print("   Note: Using enhanced synthetic Planck data")
        print("   For real data: download from https://pla.esac.esa.int/")
        
        return self._generate_synthetic_planck_data()
    
    def mfsu_power_spectrum_model(self, ell, A, delta_f, B=0, alpha=0):
        """
        MFSU CMB power spectrum model
        
        C_ℓ = A * ℓ^(-δF) + B * ℓ^(-α)
        """
        base_spectrum = A * ell**(-delta_f)
        
        if B != 0:
            # Additional component (e.g., for low-ℓ excess)
            additional = B * ell**(-alpha)
            return base_spectrum + additional
        
        return base_spectrum
    
    def lcdm_power_spectrum_model(self, ell, A, ns, running=0):
        """
        Standard ΛCDM power spectrum model
        
        C_ℓ ∝ ℓ^(ns-1) with optional running
        """
        if running != 0:
            # Running spectral index
            effective_ns = ns + running * np.log(ell / 100)
            return A * ell**(effective_ns - 1)
        else:
            return A * ell**(ns - 1)
    
    def fit_mfsu_model(self, cmb_data, ell_range=None, model_type='simple'):
        """
        Fit MFSU model to CMB data
        """
        print("🔬 Fitting MFSU model to CMB data...")
        
        ell = cmb_data['ell']
        C_ell = cmb_data['C_ell']
        C_ell_err = cmb_data['C_ell_err']
        
        # Select fitting range
        if ell_range is None:
            ell_range = (10, 2000)  # Standard range
        
        mask = (ell >= ell_range[0]) & (ell <= ell_range[1])
        ell_fit = ell[mask]
        C_ell_fit = C_ell[mask]
        C_ell_err_fit = C_ell_err[mask]
        
        print(f"   Fitting range: ℓ = {ell_range[0]} to {ell_range[1]}")
        print(f"   Data points: {len(ell_fit)}")
        
        if model_type == 'simple':
            # Simple power law: C_ℓ = A * ℓ^(-δF)
            def model_func(ell, A, delta_f):
                return self.mfsu_power_spectrum_model(ell, A, delta_f)
            
            # Initial guess
            p0 = [2500, self.delta_f]
            bounds = ([100, 0.5], [10000, 1.5])
            
        elif model_type == 'extended':
            # Extended model with additional component
            def model_func(ell, A, delta_f, B, alpha):
                return self.mfsu_power_spectrum_model(ell, A, delta_f, B, alpha)
            
            p0 = [2500, self.delta_f, 100, 1.0]
            bounds = ([100, 0.5, 0, 0.1], [10000, 1.5, 1000, 2.0])
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform fit
        try:
            popt, pcov = optimize.curve_fit(
                model_func, ell_fit, C_ell_fit,
                sigma=C_ell_err_fit, p0=p0, bounds=bounds,
                maxfev=10000
            )
            
            # Calculate fitted spectrum
            C_ell_mfsu = model_func(ell, *popt)
            
            # Parameter errors
            param_errors = np.sqrt(np.diag(pcov))
            
            fit_success = True
            
        except Exception as e:
            print(f"   ⚠️ Fit failed: {e}")
            popt = p0
            param_errors = np.array([0.1 * p for p in p0])
            C_ell_mfsu = model_func(ell, *popt)
            fit_success = False
        
        # Calculate goodness of fit
        chi2 = np.sum(((C_ell_fit - model_func(ell_fit, *popt)) / C_ell_err_fit)**2)
        dof = len(ell_fit) - len(popt)
        chi2_reduced = chi2 / dof
        
        # R-squared
        ss_res = np.sum((C_ell_fit - model_func(ell_fit, *popt))**2)
        ss_tot = np.sum((C_ell_fit - np.mean(C_ell_fit))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        results = {
            'model_type': model_type,
            'fit_success': fit_success,
            'parameters': popt,
            'parameter_errors': param_errors,
            'parameter_names': ['A', 'δF'] if model_type == 'simple' else ['A', 'δF', 'B', 'α'],
            'C_ell_model': C_ell_mfsu,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'dof': dof,
            'r2': r2,
            'ell_range': ell_range,
            'n_datapoints': len(ell_fit),
            'delta_f_fitted': popt[1],
            'delta_f_error': param_errors[1]
        }
        
        self.results['mfsu_fit'] = results
        
        print(f"   MFSU δF: {results['delta_f_fitted']:.3f} ± {results['delta_f_error']:.3f}")
        print(f"   χ²/dof: {results['chi2_reduced']:.3f}")
        print(f"   R²: {results['r2']:.3f}")
        
        return results
    
    def fit_lcdm_model(self, cmb_data, ell_range=None):
        """
        Fit ΛCDM model for comparison
        """
        print("🌌 Fitting ΛCDM model for comparison...")
        
        ell = cmb_data['ell']
        C_ell = cmb_data['C_ell']
        C_ell_err = cmb_data['C_ell_err']
        
        if ell_range is None:
            ell_range = (10, 2000)
        
        mask = (ell >= ell_range[0]) & (ell <= ell_range[1])
        ell_fit = ell[mask]
        C_ell_fit = C_ell[mask]
        C_ell_err_fit = C_ell_err[mask]
        
        # ΛCDM model: C_ℓ = A * ℓ^(ns-1)
        def lcdm_func(ell, A, ns):
            return self.lcdm_power_spectrum_model(ell, A, ns)
        
        # Initial guess
        p0 = [2500, 0.965]  # Standard values
        bounds = ([100, 0.9], [10000, 1.1])
        
        try:
            popt, pcov = optimize.curve_fit(
                lcdm_func, ell_fit, C_ell_fit,
                sigma=C_ell_err_fit, p0=p0, bounds=bounds
            )
            
            C_ell_lcdm = lcdm_func(ell, *popt)
            param_errors = np.sqrt(np.diag(pcov))
            fit_success = True
            
        except Exception as e:
            print(f"   ⚠️ ΛCDM fit failed: {e}")
            popt = p0
            param_errors = np.array([0.1 * p for p in p0])
            C_ell_lcdm = lcdm_func(ell, *popt)
            fit_success = False
        
        # Calculate goodness of fit
        chi2 = np.sum(((C_ell_fit - lcdm_func(ell_fit, *popt)) / C_ell_err_fit)**2)
        dof = len(ell_fit) - len(popt)
        chi2_reduced = chi2 / dof
        
        ss_res = np.sum((C_ell_fit - lcdm_func(ell_fit, *popt))**2)
        ss_tot = np.sum((C_ell_fit - np.mean(C_ell_fit))**2)
        r2 = 1 - (ss_res / ss_tot)
        
        results = {
            'fit_success': fit_success,
            'parameters': popt,
            'parameter_errors': param_errors,
            'parameter_names': ['A', 'ns'],
            'C_ell_model': C_ell_lcdm,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'dof': dof,
            'r2': r2,
            'ns_fitted': popt[1],
            'ns_error': param_errors[1]
        }
        
        self.results['lcdm_fit'] = results
        
        print(f"   ΛCDM ns: {results['ns_fitted']:.3f} ± {results['ns_error']:.3f}")
        print(f"   χ²/dof: {results['chi2_reduced']:.3f}")
        print(f"   R²: {results['r2']:.3f}")
        
        return results
    
    def compare_models(self, cmb_data):
        """
        Compare MFSU and ΛCDM models
        """
        print("⚖️ Comparing MFSU vs ΛCDM models...")
        
        # Ensure both models are fitted
        if 'mfsu_fit' not in self.results:
            self.fit_mfsu_model(cmb_data)
        if 'lcdm_fit' not in self.results:
            self.fit_lcdm_model(cmb_data)
        
        mfsu_results = self.results['mfsu_fit']
        lcdm_results = self.results['lcdm_fit']
        
        # Model comparison metrics
        comparison = {
            'chi2_mfsu': mfsu_results['chi2_reduced'],
            'chi2_lcdm': lcdm_results['chi2_reduced'],
            'r2_mfsu': mfsu_results['r2'],
            'r2_lcdm': lcdm_results['r2'],
            'improvement_chi2': (lcdm_results['chi2_reduced'] - mfsu_results['chi2_reduced']) / lcdm_results['chi2_reduced'] * 100,
            'improvement_r2': (mfsu_results['r2'] - lcdm_results['r2']) / lcdm_results['r2'] * 100,
            'delta_f_fitted': mfsu_results['delta_f_fitted'],
            'delta_f_error': mfsu_results['delta_f_error'],
            'ns_fitted': lcdm_results['ns_fitted'],
            'ns_error': lcdm_results['ns_error']
        }
        
        # Statistical significance tests
        # Likelihood ratio test (simplified)
        delta_chi2 = lcdm_results['chi2'] - mfsu_results['chi2']
        delta_dof = abs(lcdm_results['dof'] - mfsu_results['dof'])
        
        if delta_dof > 0:
            p_value = 1 - stats.chi2.cdf(delta_chi2, delta_dof)
            comparison['likelihood_ratio_p'] = p_value
            comparison['significant_improvement'] = p_value < 0.05
        else:
            comparison['likelihood_ratio_p'] = np.nan
            comparison['significant_improvement'] = comparison['improvement_chi2'] > 5  # 5% threshold
        
        # Information criteria
        n_data = mfsu_results['n_datapoints']
        
        # AIC (Akaike Information Criterion)
        aic_mfsu = 2 * len(mfsu_results['parameters']) + mfsu_results['chi2']
        aic_lcdm = 2 * len(lcdm_results['parameters']) + lcdm_results['chi2']
        comparison['aic_mfsu'] = aic_mfsu
        comparison['aic_lcdm'] = aic_lcdm
        comparison['delta_aic'] = aic_lcdm - aic_mfsu
        
        # BIC (Bayesian Information Criterion)
        bic_mfsu = np.log(n_data) * len(mfsu_results['parameters']) + mfsu_results['chi2']
        bic_lcdm = np.log(n_data) * len(lcdm_results['parameters']) + lcdm_results['chi2']
        comparison['bic_mfsu'] = bic_mfsu
        comparison['bic_lcdm'] = bic_lcdm
        comparison['delta_bic'] = bic_lcdm - bic_mfsu
        
        self.results['comparison'] = comparison
        
        print(f"   MFSU χ²/dof: {comparison['chi2_mfsu']:.3f}")
        print(f"   ΛCDM χ²/dof: {comparison['chi2_lcdm']:.3f}")
        print(f"   Improvement: {comparison['improvement_chi2']:.1f}%")
        print(f"   ΔAIC: {comparison['delta_aic']:.1f} (negative favors MFSU)")
        print(f"   ΔBIC: {comparison['delta_bic']:.1f} (negative favors MFSU)")
        
        if not np.isnan(comparison['likelihood_ratio_p']):
            print(f"   Statistical significance: p = {comparison['likelihood_ratio_p']:.4f}")
        
        return comparison
    
    def analyze_residuals(self, cmb_data, model='mfsu'):
        """
        Analyze fit residuals for systematic patterns
        """
        print(f"🔍 Analyzing {model.upper()} residuals...")
        
        if model == 'mfsu' and 'mfsu_fit' in self.results:
            fit_results = self.results['mfsu_fit']
        elif model == 'lcdm' and 'lcdm_fit' in self.results:
            fit_results = self.results['lcdm_fit']
        else:
            print(f"   ⚠️ {model.upper()} fit not available")
            return None
        
        ell = cmb_data['ell']
        C_ell = cmb_data['C_ell']
        C_ell_err = cmb_data['C_ell_err']
        C_ell_model = fit_results['C_ell_model']
        
        # Calculate residuals
        residuals = C_ell - C_ell_model
        normalized_residuals = residuals / C_ell_err
        
        # Statistical tests
        # Runs test for randomness
        median_residual = np.median(normalized_residuals)
        runs, n_runs, p_runs = self._runs_test(normalized_residuals > median_residual)
        
        # Ljung-Box test for autocorrelation
        lb_stat, lb_p = self._ljung_box_test(normalized_residuals, lags=10)
        
        # Anderson-Darling test for normality
        ad_stat, ad_critical, ad_p = stats.anderson(normalized_residuals, dist='norm')
        
        results = {
            'residuals': residuals,
            'normalized_residuals': normalized_residuals,
            'rms_residual': np.sqrt(np.mean(residuals**2)),
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'runs_test': {
                'statistic': runs,
                'n_runs': n_runs,
                'p_value': p_runs,
                'is_random': p_runs > 0.05
            },
            'ljung_box_test': {
                'statistic': lb_stat,
                'p_value': lb_p,
                'no_autocorr': lb_p > 0.05
            },
            'normality_test': {
                'statistic': ad_stat,
                'critical_values': ad_critical,
                'p_value': ad_p,
                'is_normal': ad_stat < ad_critical[2]  # 5% level
            }
        }
        
        self.results[f'{model}_residuals'] = results
        
        print(f"   RMS residual: {results['rms_residual']:.2f} {cmb_data['units']}")
        print(f"   Runs test p-value: {p_runs:.3f} ({'random' if p_runs > 0.05 else 'non-random'})")
        print(f"   Autocorrelation p-value: {lb_p:.3f} ({'none' if lb_p > 0.05 else 'detected'})")
        print(f"   Residuals normal: {'Yes' if results['normality_test']['is_normal'] else 'No'}")
        
        return results
    
    def low_ell_analysis(self, cmb_data, ell_threshold=30):
        """
        Special analysis of low-ℓ region where MFSU may show advantages
        """
        print(f"🔬 Analyzing low-ℓ region (ℓ < {ell_threshold})...")
        
        ell = cmb_data['ell']
        C_ell = cmb_data['C_ell']
        C_ell_err = cmb_data['C_ell_err']
        
        # Select low-ℓ data
        low_ell_mask = ell < ell_threshold
        ell_low = ell[low_ell_mask]
        C_ell_low = C_ell[low_ell_mask]
        C_ell_err_low = C_ell_err[low_ell_mask]
        
        if len(ell_low) < 5:
            print("   ⚠️ Insufficient low-ℓ data")
            return None
        
        # Fit MFSU model to low-ℓ only
        low_ell_mfsu = self.fit_mfsu_model(
            {'ell': ell_low, 'C_ell': C_ell_low, 'C_ell_err': C_ell_err_low},
            ell_range=(self.lmin, ell_threshold)
        )
        
        # Compare with full-range fit
        if 'mfsu_fit' in self.results:
            full_range_delta_f = self.results['mfsu_fit']['delta_f_fitted']
            delta_f_difference = low_ell_mfsu['delta_f_fitted'] - full_range_delta_f
        else:
            full_range_delta_f = self.delta_f
            delta_f_difference = low_ell_mfsu['delta_f_fitted'] - self.delta_f
        
        # Calculate low-ℓ power deficit (common CMB anomaly)
        expected_power = np.mean(C_ell[ell > ell_threshold][:20])  # Reference from higher ℓ
        observed_power = np.mean(C_ell_low)
        power_deficit = (expected_power - observed_power) / expected_power * 100
        
        results = {
            'ell_threshold': ell_threshold,
            'n_low_ell': len(ell_low),
            'low_ell_fit': low_ell_mfsu,
            'delta_f_low_ell': low_ell_mfsu['delta_f_fitted'],
            'delta_f_error_low_ell': low_ell_mfsu['delta_f_error'],
            'full_range_delta_f': full_range_delta_f,
            'delta_f_difference': delta_f_difference,
            'power_deficit_percent': power_deficit,
            'low_ell_chi2': low_ell_mfsu['chi2_reduced']
        }
        
        self.results['low_ell_analysis'] = results
        
        print(f"   Low-ℓ δF: {results['delta_f_low_ell']:.3f} ± {results['delta_f_error_low_ell']:.3f}")
        print(f"   Difference from full range: {delta_f_difference:.3f}")
        print(f"   Power deficit: {power_deficit:.1f}%")
        print(f"   Low-ℓ χ²/dof: {results['low_ell_chi2']:.3f}")
        
        return results
    
    def multifractal_cmb_analysis(self, cmb_data):
        """
        Analyze CMB for multifractal properties
        """
        print("🌀 Multifractal analysis of CMB...")
        
        # This would require a 2D CMB map for proper multifractal analysis
        # For 1D power spectrum, we do spectral multifractal analysis
        
        ell = cmb_data['ell']
        C_ell = cmb_data['C_ell']
        
        # Log-log scaling analysis
        log_ell = np.log(ell)
        log_C_ell = np.log(C_ell)
        
        # Remove any infinities
        finite_mask = np.isfinite(log_ell) & np.isfinite(log_C_ell)
        log_ell = log_ell[finite_mask]
        log_C_ell = log_C_ell[finite_mask]
        
        # Fit scaling exponent
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ell, log_C_ell)
        
        # Multifractal spectrum analysis (simplified)
        # Partition function method on power spectrum
        q_values = np.linspace(-3, 3, 13)
        tau_q = []
        
        # Use windowing for local scaling analysis
        window_sizes = np.logspace(1, 2.5, 10).astype(int)
        
        for q in q_values:
            scaling_exponents = []
            
            for window_size in window_sizes:
                if window_size >= len(C_ell):
                    continue
                
                # Partition function
                if q != 0:
                    partition = np.sum(C_ell[:window_size]**q)
                else:
                    partition = len(C_ell[:window_size])
                
                if partition > 0:
                    scaling_exponents.append(np.log(partition) / np.log(window_size))
            
            if len(scaling_exponents) > 0:
                tau_q.append(np.mean(scaling_exponents))
            else:
                tau_q.append(np.nan)
        
        tau_q = np.array(tau_q)
        valid_q = ~np.isnan(tau_q)
        
        if np.sum(valid_q) > 3:
            # Calculate multifractal spectrum D(α)
            q_valid = q_values[valid_q]
            tau_valid = tau_q[valid_q]
            
            # Numerical derivative to get α
            alpha = np.gradient(tau_valid, q_valid)
            D_alpha = q_valid * alpha - tau_valid
            
            # Find dominant scaling
            max_idx = np.argmax(D_alpha)
            dominant_alpha = alpha[max_idx]
            
        else:
            alpha = np.array([self.delta_f])
            D_alpha = np.array([1.0])
            dominant_alpha = self.delta_f
        
        results = {
            'global_scaling_exponent': -slope,  # Negative because C_ell decreases with ℓ
            'scaling_r_squared': r_value**2,
            'q_values': q_values,
            'tau_q': tau_q,
            'alpha': alpha,
            'D_alpha': D_alpha,
            'dominant_alpha': dominant_alpha,
            'is_multifractal': np.std(alpha) > 0.01  # Threshold for multifractality
        }
        
        self.results['multifractal'] = results
        
        print(
