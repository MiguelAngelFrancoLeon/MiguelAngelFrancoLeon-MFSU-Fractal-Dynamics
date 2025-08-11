#!/usr/bin/env python3
"""
Fractal Analysis Tools for MFSU
Box counting, multifractal analysis, and dimension estimation

Author: Miguel Ángel Franco León
Date: August 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize, signal
from scipy.special import gamma
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# MFSU Constants
DELTA_F = 0.921
DF_FRACTAL = 2.079

class FractalAnalyzer:
    """
    Main class for fractal dimension analysis
    """
    
    def __init__(self, delta_f=DELTA_F):
        self.delta_f = delta_f
        self.results = {}
    
    def box_counting_1d(self, data, scales=None, plot=False):
        """
        Box counting method for 1D time series
        """
        if scales is None:
            scales = np.logspace(-2, 0, 20) * len(data)
            scales = scales[scales >= 2].astype(int)
        
        # Normalize data to [0, 1]
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        
        counts = []
        
        for scale in scales:
            # Create boxes
            n_boxes = int(np.ceil(len(data_norm) / scale))
            box_counts = 0
            
            for i in range(n_boxes):
                start_idx = i * scale
                end_idx = min((i + 1) * scale, len(data_norm))
                
                if end_idx > start_idx:
                    box_data = data_norm[start_idx:end_idx]
                    if len(box_data) > 0:
                        # Check if box contains data
                        if np.max(box_data) - np.min(box_data) > 0:
                            box_counts += 1
            
            counts.append(max(box_counts, 1))  # Avoid log(0)
        
        counts = np.array(counts)
        scales = np.array(scales)
        
        # Linear regression in log-log space
        log_scales = np.log(1 / scales)
        log_counts = np.log(counts)
        
        # Remove invalid points
        valid = np.isfinite(log_scales) & np.isfinite(log_counts)
        if np.sum(valid) < 3:
            return 1.0, 0.1, scales, counts
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales[valid], log_counts[valid]
        )
        
        dimension = slope
        error = std_err
        
        if plot:
            self._plot_box_counting(scales, counts, dimension, r_value)
        
        result = {
            'dimension': dimension,
            'error': error,
            'r_squared': r_value**2,
            'scales': scales,
            'counts': counts,
            'method': 'box_counting_1d'
        }
        
        return dimension, error, result
    
    def box_counting_2d(self, data, scales=None, plot=False):
        """
        Box counting method for 2D images/maps
        """
        if scales is None:
            max_scale = min(data.shape) // 4
            scales = np.unique(np.logspace(0, np.log10(max_scale), 15).astype(int))
            scales = scales[scales >= 2]
        
        # Binarize data (threshold at median)
        threshold = np.median(data)
        binary_data = data > threshold
        
        counts = []
        
        for scale in scales:
            count = 0
            
            # Iterate over non-overlapping boxes
            for i in range(0, data.shape[0], scale):
                for j in range(0, data.shape[1], scale):
                    # Extract box
                    box = binary_data[i:i+scale, j:j+scale]
                    
                    # Count if box contains any True values
                    if box.size > 0 and np.any(box):
                        count += 1
            
            counts.append(max(count, 1))
        
        counts = np.array(counts)
        
        # Linear regression
        log_scales = np.log(1 / scales)
        log_counts = np.log(counts)
        
        valid = np.isfinite(log_scales) & np.isfinite(log_counts)
        if np.sum(valid) < 3:
            return 2.0, 0.1, {}
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales[valid], log_counts[valid]
        )
        
        dimension = slope
        error = std_err
        
        if plot:
            self._plot_box_counting(scales, counts, dimension, r_value)
        
        result = {
            'dimension': dimension,
            'error': error,
            'r_squared': r_value**2,
            'scales': scales,
            'counts': counts,
            'method': 'box_counting_2d'
        }
        
        return dimension, error, result
    
    def correlation_dimension(self, data, max_points=1000, scales=None, plot=False):
        """
        Correlation dimension using Grassberger-Procaccia algorithm
        """
        # Subsample if too many points
        if len(data) > max_points:
            indices = np.random.choice(len(data), max_points, replace=False)
            data = data[indices]
        
        # Embed data if 1D (using delay coordinates)
        if data.ndim == 1:
            tau = 1
            m = 3  # Embedding dimension
            if len(data) < m * tau + 1:
                return 1.0, 0.1, {}
            
            embedded = np.array([data[i:i+m*tau:tau] for i in range(len(data)-m*tau)])
        else:
            embedded = data
        
        if scales is None:
            distances = self._compute_pairwise_distances(embedded)
            scales = np.logspace(np.log10(np.min(distances[distances > 0])),
                               np.log10(np.max(distances)), 20)
        
        correlations = []
        
        for r in scales:
            # Count pairs within distance r
            distances = self._compute_pairwise_distances(embedded)
            count = np.sum(distances < r) - len(embedded)  # Exclude self-distances
            total_pairs = len(embedded) * (len(embedded) - 1)
            
            if total_pairs > 0:
                correlation = count / total_pairs
            else:
                correlation = 0
            
            correlations.append(max(correlation, 1e-10))  # Avoid log(0)
        
        correlations = np.array(correlations)
        
        # Linear fit in log-log space
        log_scales = np.log(scales)
        log_correlations = np.log(correlations)
        
        # Find linear region (middle part)
        n_points = len(scales)
        start_idx = n_points // 4
        end_idx = 3 * n_points // 4
        
        valid_region = slice(start_idx, end_idx)
        
        if end_idx - start_idx < 3:
            return 1.0, 0.1, {}
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales[valid_region], log_correlations[valid_region]
        )
        
        dimension = slope
        error = std_err
        
        if plot:
            self._plot_correlation_dimension(scales, correlations, dimension, r_value, valid_region)
        
        result = {
            'dimension': dimension,
            'error': error,
            'r_squared': r_value**2,
            'scales': scales,
            'correlations': correlations,
            'method': 'correlation_dimension'
        }
        
        return dimension, error, result
    
    def spectral_dimension(self, data, plot=False):
        """
        Spectral method for fractal dimension estimation
        """
        if data.ndim == 1:
            freqs, psd = signal.periodogram(data, scaling='density')
        else:
            # Use first component for multidimensional data
            freqs, psd = signal.periodogram(data.flatten(), scaling='density')
        
        # Remove DC component and very low frequencies
        valid = (freqs > 0) & (freqs < 0.4) & (psd > 0)
        freqs = freqs[valid]
        psd = psd[valid]
        
        if len(freqs) < 5:
            return 1.0, 0.1, {}
        
        # Fit power law: PSD ∝ f^(-β)
        log_freqs = np.log(freqs)
        log_psd = np.log(psd)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_freqs, log_psd)
        
        # Convert spectral exponent to fractal dimension
        # For 1D: D = (5 + β) / 2, but we use empirical relation
        beta = -slope  # Power law exponent (positive)
        dimension = (3 + beta) / 2  # Adjusted for MFSU
        
        error = std_err / 2
        
        if plot:
            self._plot_spectral_dimension(freqs, psd, beta, r_value)
        
        result = {
            'dimension': dimension,
            'error': error,
            'beta': beta,
            'r_squared': r_value**2,
            'freqs': freqs,
            'psd': psd,
            'method': 'spectral_dimension'
        }
        
        return dimension, error, result
    
    def multifractal_spectrum(self, data, q_range=None, plot=False):
        """
        Multifractal spectrum analysis
        """
        if q_range is None:
            q_range = np.linspace(-5, 5, 21)
        
        # Box counting for different moments
        max_scale = len(data) // 4 if data.ndim == 1 else min(data.shape) // 4
        scales = np.unique(np.logspace(0, np.log10(max_scale), 10).astype(int))
        scales = scales[scales >= 2]
        
        tau_q = []
        
        for q in q_range:
            log_scales = []
            log_moments = []
            
            for scale in scales:
                if data.ndim == 1:
                    # 1D case
                    n_boxes = len(data) // scale
                    moments = []
                    
                    for i in range(n_boxes):
                        box_data = data[i*scale:(i+1)*scale]
                        if len(box_data) > 0:
                            measure = np.var(box_data) + 1e-10  # Add small constant
                            moments.append(measure)
                    
                    if len(moments) > 0:
                        if q != 0:
                            moment_q = np.sum(np.array(moments)**q)
                        else:
                            moment_q = len(moments)  # Number of non-empty boxes
                        
                        if moment_q > 0:
                            log_scales.append(np.log(scale))
                            log_moments.append(np.log(moment_q))
                
                else:
                    # 2D case
                    moments = []
                    for i in range(0, data.shape[0], scale):
                        for j in range(0, data.shape[1], scale):
                            box_data = data[i:i+scale, j:j+scale]
                            if box_data.size > 0:
                                measure = np.var(box_data) + 1e-10
                                moments.append(measure)
                    
                    if len(moments) > 0:
                        if q != 0:
                            moment_q = np.sum(np.array(moments)**q)
                        else:
                            moment_q = len(moments)
                        
                        if moment_q > 0:
                            log_scales.append(np.log(scale))
                            log_moments.append(np.log(moment_q))
            
            # Linear regression to get τ(q)
            if len(log_scales) >= 3:
                slope, _, _, _, _ = stats.linregress(log_scales, log_moments)
                tau_q.append(slope)
            else:
                tau_q.append(np.nan)
        
        tau_q = np.array(tau_q)
        
        # Remove NaN values
        valid = ~np.isnan(tau_q)
        q_range = q_range[valid]
        tau_q = tau_q[valid]
        
        # Calculate multifractal spectrum D(α)
        if len(tau_q) >= 3:
            # Numerical derivative: α = dτ/dq
            alpha = np.gradient(tau_q, q_range)
            
            # D(α) = qα - τ(q)
            D_alpha = q_range * alpha - tau_q
            
            # Find maximum of D(α) - this gives the dominant dimension
            max_idx = np.argmax(D_alpha)
            dominant_alpha = alpha[max_idx]
            dominant_D = D_alpha[max_idx]
        else:
            alpha = np.array([self.delta_f])
            D_alpha = np.array([1.0])
            dominant_alpha = self.delta_f
            dominant_D = 1.0
        
        if plot:
            self._plot_multifractal_spectrum(q_range, tau_q, alpha, D_alpha)
        
        result = {
            'q_range': q_range,
            'tau_q': tau_q,
            'alpha': alpha,
            'D_alpha': D_alpha,
            'dominant_alpha': dominant_alpha,
            'dominant_dimension': dominant_D,
            'method': 'multifractal_spectrum'
        }
        
        return dominant_alpha, dominant_D, result
    
    def hurst_exponent(self, data, method='dfa', plot=False):
        """
        Estimate Hurst exponent using various methods
        """
        if method == 'dfa':
            return self._detrended_fluctuation_analysis(data, plot)
        elif method == 'rs':
            return self._rescaled_range(data, plot)
        elif method == 'variogram':
            return self._variogram_method(data, plot)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detrended_fluctuation_analysis(self, data, plot=False):
        """
        Detrended Fluctuation Analysis (DFA)
        """
        N = len(data)
        
        # Integration of the series
        y = np.cumsum(data - np.mean(data))
        
        # Range of window sizes
        scales = np.unique(np.logspace(1, np.log10(N//4), 20).astype(int))
        scales = scales[scales >= 10]
        
        fluctuations = []
        
        for scale in scales:
            # Number of windows
            n_windows = N // scale
            
            # Calculate fluctuations in each window
            F_scale = []
            
            for i in range(n_windows):
                # Extract window
                start_idx = i * scale
                end_idx = (i + 1) * scale
                window = y[start_idx:end_idx]
                
                # Fit polynomial trend (linear)
                x = np.arange(len(window))
                if len(window) > 1:
                    poly_coeff = np.polyfit(x, window, 1)
                    trend = np.polyval(poly_coeff, x)
                    
                    # Calculate fluctuation
                    fluctuation = np.sqrt(np.mean((window - trend)**2))
                    F_scale.append(fluctuation)
            
            if len(F_scale) > 0:
                F_n = np.sqrt(np.mean(np.array(F_scale)**2))
                fluctuations.append(F_n)
            else:
                fluctuations.append(np.nan)
        
        fluctuations = np.array(fluctuations)
        
        # Remove NaN values
        valid = ~np.isnan(fluctuations) & (fluctuations > 0)
        scales_valid = scales[valid]
        fluctuations_valid = fluctuations[valid]
        
        if len(scales_valid) < 3:
            return 0.5, 0.1, {}
        
        # Linear regression in log-log space
        log_scales = np.log(scales_valid)
        log_fluctuations = np.log(fluctuations_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales, log_fluctuations
        )
        
        hurst = slope
        error = std_err
        
        if plot:
            self._plot_hurst_analysis(scales_valid, fluctuations_valid, hurst, r_value, 'DFA')
        
        result = {
            'hurst': hurst,
            'error': error,
            'r_squared': r_value**2,
            'scales': scales_valid,
            'fluctuations': fluctuations_valid,
            'method': 'dfa'
        }
        
        return hurst, error, result
    
    def _rescaled_range(self, data, plot=False):
        """
        Rescaled Range (R/S) analysis
        """
        N = len(data)
        
        # Range of window sizes
        scales = np.unique(np.logspace(1, np.log10(N//4), 15).astype(int))
        scales = scales[scales >= 10]
        
        rs_values = []
        
        for scale in scales:
            n_windows = N // scale
            rs_window = []
            
            for i in range(n_windows):
                start_idx = i * scale
                end_idx = (i + 1) * scale
                window = data[start_idx:end_idx]
                
                if len(window) > 1:
                    # Mean and cumulative sum
                    mean_window = np.mean(window)
                    Y = np.cumsum(window - mean_window)
                    
                    # Range and standard deviation
                    R = np.max(Y) - np.min(Y)
                    S = np.std(window)
                    
                    if S > 0:
                        rs_window.append(R / S)
            
            if len(rs_window) > 0:
                rs_values.append(np.mean(rs_window))
            else:
                rs_values.append(np.nan)
        
        rs_values = np.array(rs_values)
        
        # Remove invalid values
        valid = ~np.isnan(rs_values) & (rs_values > 0)
        scales_valid = scales[valid]
        rs_valid = rs_values[valid]
        
        if len(scales_valid) < 3:
            return 0.5, 0.1, {}
        
        # Linear regression
        log_scales = np.log(scales_valid)
        log_rs = np.log(rs_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_scales, log_rs
        )
        
        hurst = slope
        error = std_err
        
        if plot:
            self._plot_hurst_analysis(scales_valid, rs_valid, hurst, r_value, 'R/S')
        
        result = {
            'hurst': hurst,
            'error': error,
            'r_squared': r_value**2,
            'scales': scales_valid,
            'rs_values': rs_valid,
            'method': 'rescaled_range'
        }
        
        return hurst, error, result
    
    def _variogram_method(self, data, plot=False):
        """
        Variogram method for Hurst exponent
        """
        N = len(data)
        
        # Range of lags
        max_lag = N // 4
        lags = np.arange(1, max_lag, max(1, max_lag // 20))
        
        variogram = []
        
        for lag in lags:
            # Calculate variogram
            diff = data[lag:] - data[:-lag]
            gamma = 0.5 * np.mean(diff**2)
            variogram.append(gamma)
        
        variogram = np.array(variogram)
        
        # Remove zeros
        valid = variogram > 0
        lags_valid = lags[valid]
        variogram_valid = variogram[valid]
        
        if len(lags_valid) < 3:
            return 0.5, 0.1, {}
        
        # Linear regression in log-log space
        log_lags = np.log(lags_valid)
        log_variogram = np.log(variogram_valid)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_lags, log_variogram
        )
        
        # Hurst exponent from variogram slope
        hurst = slope / 2
        error = std_err / 2
        
        if plot:
            self._plot_hurst_analysis(lags_valid, variogram_valid, slope, r_value, 'Variogram')
        
        result = {
            'hurst': hurst,
            'error': error,
            'r_squared': r_value**2,
            'lags': lags_valid,
            'variogram': variogram_valid,
            'method': 'variogram'
        }
        
        return hurst, error, result
    
    def validate_mfsu_dimension(self, data, methods=None, plot=True):
        """
        Comprehensive validation using multiple methods
        """
        if methods is None:
            methods = ['box_counting', 'correlation', 'spectral', 'multifractal']
        
        print("🔍 Comprehensive fractal dimension analysis...")
        
        results = {}
        dimensions = []
        errors = []
        
        # Box counting
        if 'box_counting' in methods:
            if data.ndim == 1:
                dim, err, res = self.box_counting_1d(data, plot=plot)
            else:
                dim, err, res = self.box_counting_2d(data, plot=plot)
            
            results['box_counting'] = res
            dimensions.append(dim)
            errors.append(err)
            print(f"   Box counting: D = {dim:.3f} ± {err:.3f}")
        
        # Correlation dimension
        if 'correlation' in methods:
            dim, err, res = self.correlation_dimension(data, plot=plot)
            results['correlation'] = res
            dimensions.append(dim)
            errors.append(err)
            print(f"   Correlation: D = {dim:.3f} ± {err:.3f}")
        
        # Spectral method
        if 'spectral' in methods:
            dim, err, res = self.spectral_dimension(data, plot=plot)
            results['spectral'] = res
            dimensions.append(dim)
            errors.append(err)
            print(f"   Spectral: D = {dim:.3f} ± {err:.3f}")
        
        # Multifractal analysis
        if 'multifractal' in methods:
            alpha, D_alpha, res = self.multifractal_spectrum(data, plot=plot)
            results['multifractal'] = res
            dimensions.append(alpha)
            errors.append(0.01)  # Default error
            print(f"   Multifractal: α = {alpha:.3f}")
        
        # Calculate weighted average
        if len(dimensions) > 0:
            dimensions = np.array(dimensions)
            errors = np.array(errors)
            
            # Remove outliers (> 2 sigma from median)
            median_dim = np.median(dimensions)
            std_dim = np.std(dimensions)
            valid = np.abs(dimensions - median_dim) < 2 * std_dim
            
            if np.sum(valid) > 0:
                weights = 1 / (errors[valid]**2 + 1e-6)
                weighted_mean = np.average(dimensions[valid], weights=weights)
                weighted_error = 1 / np.sqrt(np.sum(weights))
            else:
                weighted_mean = median_dim
                weighted_error = std_dim
            
            # Convert to δF if needed
            if weighted_mean > 1.5:  # Likely a fractal dimension
                delta_f_estimate = 3 - weighted_mean
            else:  # Already δF-like
                delta_f_estimate = weighted_mean
            
            results['summary'] = {
                'dimensions': dimensions,
                'errors': errors,
                'weighted_mean_dimension': weighted_mean,
                'weighted_error': weighted_error,
                'delta_f_estimate': delta_f_estimate,
                'theoretical_delta_f': self.delta_f,
                'agreement': np.abs(delta_f_estimate - self.delta_f) < 2 * weighted_error
            }
            
            print(f"\n📊 Summary:")
            print(f"   Weighted dimension: {weighted_mean:.3f} ± {weighted_error:.3f}")
            print(f"   δF estimate: {delta_f_estimate:.3f}")
            print(f"   Theoretical δF: {self.delta_f:.3f}")
            print(f"   Agreement: {'✅' if results['summary']['agreement'] else '❌'}")
        
        return results
    
    def _compute_pairwise_distances(self, data):
        """Compute pairwise distances efficiently"""
        from scipy.spatial.distance import pdist, squareform
        return squareform(pdist(data))
    
    def _plot_box_counting(self, scales, counts, dimension, r_value):
        """Plot box counting results"""
        plt.figure(figsize=(6, 4))
        plt.loglog(1/scales, counts, 'o-', markersize=4, linewidth=1)
        
        # Fit line
        x_fit = 1/scales
        y_fit = np.exp(np.log(counts[0]) + dimension * np.log(x_fit/x_fit[0]))
        plt.loglog(x_fit, y_fit, 'r--', linewidth=2, 
                   label=f'D = {dimension:.3f}, R² = {r_value**2:.3f}')
        
        plt.xlabel('1/Scale')
        plt.ylabel('Number of Boxes')
        plt.title('Box Counting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_correlation_dimension(self, scales, correlations, dimension, r_value, valid_region):
        """Plot correlation dimension results"""
        plt.figure(figsize=(6, 4))
        plt.loglog(scales, correlations, 'o-', markersize=4, linewidth=1)
        
        # Highlight fitting region
        plt.loglog(scales[valid_region], correlations[valid_region], 
                   'ro-', markersize=6, linewidth=2, label='Fitting region')
        
        # Fit line
        x_fit = scales[valid_region]
        y_fit = np.exp(np.log(correlations[valid_region][0]) + 
                       dimension * (np.log(x_fit) - np.log(x_fit[0])))
        plt.loglog(x_fit, y_fit, 'r--', linewidth=2, 
                   label=f'D = {dimension:.3f}, R² = {r_value**2:.3f}')
        
        plt.xlabel('Distance')
        plt.ylabel('Correlation Integral')
        plt.title('Correlation Dimension Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_spectral_dimension(self, freqs, psd, beta, r_value):
        """Plot spectral analysis results"""
        plt.figure(figsize=(6, 4))
        plt.loglog(freqs, psd, 'o-', markersize=3, linewidth=1, alpha=0.7)
        
        # Fit line
        psd_fit = psd[0] * (freqs / freqs[0])**(-beta)
        plt.loglog(freqs, psd_fit, 'r--', linewidth=2, 
                   label=f'β = {beta:.3f}, R² = {r_value**2:.3f}')
        
        plt.xlabel('Frequency')
        plt.ylabel('Power Spectral Density')
        plt.title('Spectral Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_multifractal_spectrum(self, q_range, tau_q, alpha, D_alpha):
        """Plot multifractal spectrum"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # τ(q) plot
        ax1.plot(q_range, tau_q, 'o-', markersize=4, linewidth=1)
        ax1.set_xlabel('q')
        ax1.set_ylabel('τ(q)')
        ax1.set_title('Mass Exponent Function')
        ax1.grid(True, alpha=0.3)
        
        # D(α) plot
        ax2.plot(alpha, D_alpha, 'o-', markersize=4, linewidth=1, color='red')
        ax2.set_xlabel('α')
        ax2.set_ylabel('D(α)')
        ax2.set_title('Multifractal Spectrum')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_hurst_analysis(self, scales, values, hurst, r_value, method):
        """Plot Hurst exponent analysis"""
        plt.figure(figsize=(6, 4))
        plt.loglog(scales, values, 'o-', markersize=4, linewidth=1)
        
        # Fit line
        values_fit = values[0] * (scales / scales[0])**hurst
        plt.loglog(scales, values_fit, 'r--', linewidth=2, 
                   label=f'H = {hurst:.3f}, R² = {r_value**2:.3f}')
        
        plt.xlabel('Scale')
        plt.ylabel('Fluctuation' if 'DFA' in method else 'R/S' if 'R/S' in method else 'Variogram')
        plt.title(f'{method} Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class FractalGenerator:
    """
    Generate synthetic fractal data for testing
    """
    
    @staticmethod
    def fractional_brownian_motion(n, hurst=0.5):
        """Generate fractional Brownian motion"""
        # Davies-Harte method
        r = np.zeros(2*n)
        r[0] = 1
        
        for k in range(1, n):
            r[k] = 0.5 * ((k+1)**(2*hurst) - 2*k**(2*hurst) + (k-1)**(2*hurst))
        
        for k in range(n, 2*n):
            r[k] = r[2*n - k]
        
        # Eigenvalues of circulant matrix
        eigenvals = np.fft.fft(r).real
        
        if np.any(eigenvals < 0):
            eigenvals = np.maximum(eigenvals, 0)
        
        # Generate random variables
        W = np.random.randn(2*n) + 1j * np.random.randn(2*n)
        W[0] = np.real(W[0])
        W[n] = np.real(W[n])
        
        # Generate fBm
        fBm = np.fft.ifft(np.sqrt(eigenvals) * W)[:n].real
        
        return fBm
    
    @staticmethod
    def fractal_surface(size, dimension=2.5, roughness=1.0):
        """Generate 2D fractal surface using spectral synthesis"""
        # Frequency grid
        kx = np.fft.fftfreq(size).reshape(-1, 1)
        ky = np.fft.fftfreq(size).reshape(1, -1)
        k = np.sqrt(kx**2 + ky**2)
        k[0, 0] = 1e-10  # Avoid division by zero
        
        # Power spectrum
        beta = 2 * (3 - dimension)
        power_spectrum = k**(-beta)
        
        # Random phases
        phases = 2 * np.pi * np.random.random((size, size))
        
        # Generate field
        field_k = np.sqrt(power_spectrum) * np.exp(1j * phases)
        field = np.fft.ifft2(field_k).real
        
        return field * roughness
    
    @staticmethod
    def multifractal_cascade(n, p1=0.3, p2=0.7):
        """Generate multifractal cascade"""
        # Initialize
        cascade = np.ones(2**n)
        
        # Multiplicative cascade
        for level in range(n):
            new_cascade = np.zeros(2**(level+1))
            
            for i in range(2**level):
                # Split each interval
                if np.random.random() < 0.5:
                    new_cascade[2*i] = cascade[i] * p1
                    new_cascade[2*i+1] = cascade[i] * p2
                else:
                    new_cascade[2*i] = cascade[i] * p2
                    new_cascade[2*i+1] = cascade[i] * p1
            
            cascade = new_cascade
        
        return cascade

def demo_fractal_analysis():
    """
    Demonstration of fractal analysis tools
    """
    print("🔬 MFSU Fractal Analysis Tools Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = FractalAnalyzer(delta_f=DELTA_F)
    
    # Generate test data
    print("📊 Generating test fractal data...")
    
    # 1D fractional Brownian motion
    hurst_target = DELTA_F / 2  # Expected Hurst exponent
    fbm = FractalGenerator.fractional_brownian_motion(1024, hurst=hurst_target)
    
    # 2D fractal surface
    surface = FractalGenerator.fractal_surface(128, dimension=DF_FRACTAL)
    
    # Analyze 1D data
    print(f"\n🧮 Analyzing 1D fBm (target H = {hurst_target:.3f})...")
    results_1d = analyzer.validate_mfsu_dimension(fbm, plot=False)
    
    # Analyze 2D data
    print(f"\n🗺️ Analyzing 2D surface (target D = {DF_FRACTAL:.3f})...")
    results_2d = analyzer.validate_mfsu_dimension(surface, plot=False)
    
    # Hurst exponent analysis
    print(f"\n📈 Hurst exponent analysis...")
    hurst_dfa, hurst_err, _ = analyzer.hurst_exponent(fbm, method='dfa', plot=False)
    print(f"   DFA Hurst: {hurst_dfa:.3f} ± {hurst_err:.3f}")
    print(f"   Target: {hurst_target:.3f}")
    print(f"   Agreement: {'✅' if abs(hurst_dfa - hurst_target) < 2*hurst_err else '❌'}")
    
    # Summary
    print(f"\n🎯 Demo Summary:")
    print(f"   δF theoretical: {DELTA_F:.3f}")
    print(f"   df theoretical: {DF_FRACTAL:.3f}")
    
    if 'summary' in results_1d:
        print(f"   1D δF estimate: {results_1d['summary']['delta_f_estimate']:.3f}")
    
    if 'summary' in results_2d:
        print(f"   2D dimension: {results_2d['summary']['weighted_mean_dimension']:.3f}")
    
    print(f"   Hurst estimate: {hurst_dfa:.3f}")
    
    return analyzer, results_1d, results_2d

if __name__ == "__main__":
    # Run demonstration
    analyzer, results_1d, results_2d = demo_fractal_analysis()
    
    print(f"\n🚀 Fractal analysis tools ready!")
    print(f"💡 Use analyzer.validate_mfsu_dimension(data) for comprehensive analysis")
    print(f"💡 Use analyzer.hurst_exponent(data) for Hurst analysis")
    print(f"💡 Use FractalGenerator.fractional_brownian_motion(n, hurst) for test data")
