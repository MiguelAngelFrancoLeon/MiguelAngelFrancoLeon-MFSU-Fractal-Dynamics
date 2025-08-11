#!/usr/bin/env python3
"""
Statistical Analysis Tools for MFSU
Bootstrap, jackknife, cross-validation, and robust statistics

Author: Miguel Ángel Franco León
Date: August 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, optimize
from sklearn.model_selection import KFold, cross_val_score, bootstrap
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# MFSU Constants
DELTA_F = 0.921
DF_FRACTAL = 2.079

class MFSUStatistics:
    """
    Statistical analysis toolkit for MFSU validation
    """
    
    def __init__(self, delta_f=DELTA_F):
        self.delta_f = delta_f
        self.results = {}
    
    def bootstrap_analysis(self, data, statistic_func, n_bootstrap=1000, 
                          confidence_level=95, seed=42):
        """
        Bootstrap resampling analysis
        
        Parameters:
        -----------
        data : array-like
            Input data
        statistic_func : callable
            Function to compute statistic (e.g., np.mean, np.std)
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level for intervals
        """
        print(f"🔄 Running bootstrap analysis ({n_bootstrap} samples)...")
        
        np.random.seed(seed)
        n_samples = len(data)
        
        bootstrap_stats = []
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = data[indices]
            
            # Calculate statistic
            try:
                stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(stat)
            except:
                continue  # Skip failed calculations
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Remove outliers (beyond 3 sigma)
        mean_stat = np.mean(bootstrap_stats)
        std_stat = np.std(bootstrap_stats)
        valid_mask = np.abs(bootstrap_stats - mean_stat) < 3 * std_stat
        bootstrap_stats = bootstrap_stats[valid_mask]
        
        # Calculate statistics
        alpha = (100 - confidence_level) / 100
        
        results = {
            'bootstrap_samples': bootstrap_stats,
            'mean': np.mean(bootstrap_stats),
            'std': np.std(bootstrap_stats),
            'median': np.median(bootstrap_stats),
            'confidence_interval': np.percentile(bootstrap_stats, 
                                               [100*alpha/2, 100*(1-alpha/2)]),
            'n_valid_samples': len(bootstrap_stats),
            'n_total_samples': n_bootstrap,
            'success_rate': len(bootstrap_stats) / n_bootstrap,
            'original_statistic': statistic_func(data)
        }
        
        # Calculate bias and bias-corrected estimate
        results['bias'] = results['mean'] - results['original_statistic']
        results['bias_corrected'] = results['original_statistic'] - results['bias']
        
        print(f"   Bootstrap mean: {results['mean']:.4f} ± {results['std']:.4f}")
        print(f"   {confidence_level}% CI: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
        print(f"   Bias: {results['bias']:.4f}")
        print(f"   Success rate: {results['success_rate']*100:.1f}%")
        
        return results
    
    def jackknife_analysis(self, data, statistic_func):
        """
        Jackknife resampling analysis
        """
        print("🔪 Running jackknife analysis...")
        
        n_samples = len(data)
        jackknife_stats = []
        
        # Leave-one-out resampling
        for i in range(n_samples):
            # Create jackknife sample (all except i-th element)
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            
            try:
                stat = statistic_func(jackknife_sample)
                jackknife_stats.append(stat)
            except:
                continue
        
        jackknife_stats = np.array(jackknife_stats)
        
        # Original statistic
        original_stat = statistic_func(data)
        
        # Jackknife bias and variance
        mean_jackknife = np.mean(jackknife_stats)
        bias = (n_samples - 1) * (mean_jackknife - original_stat)
        variance = (n_samples - 1) / n_samples * np.sum((jackknife_stats - mean_jackknife)**2)
        
        results = {
            'jackknife_samples': jackknife_stats,
            'mean': mean_jackknife,
            'std': np.sqrt(variance),
            'bias': bias,
            'bias_corrected': original_stat - bias,
            'original_statistic': original_stat,
            'variance': variance,
            'n_samples': len(jackknife_stats)
        }
        
        print(f"   Jackknife mean: {results['mean']:.4f} ± {results['std']:.4f}")
        print(f"   Bias: {results['bias']:.4f}")
        print(f"   Bias-corrected: {results['bias_corrected']:.4f}")
        
        return results
    
    def cross_validation_analysis(self, X, y, model=None, cv_folds=5, 
                                 scoring='r2', seed=42):
        """
        Cross-validation analysis for model validation
        """
        print(f"✅ Running {cv_folds}-fold cross-validation...")
        
        if model is None:
            model = LinearRegression()
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        # Additional metrics
        train_scores = []
        val_scores = []
        
        for train_idx, val_idx in cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate scores
            train_score = r2_score(y_train, y_train_pred)
            val_score = r2_score(y_val, y_val_pred)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)
        
        results = {
            'cv_scores': cv_scores,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'mean_cv': np.mean(cv_scores),
            'std_cv': np.std(cv_scores),
            'mean_train': np.mean(train_scores),
            'std_train': np.std(train_scores),
            'mean_val': np.mean(val_scores),
            'std_val': np.std(val_scores),
            'overfitting': np.mean(train_scores) - np.mean(val_scores),
            'cv_folds': cv_folds
        }
        
        print(f"   CV {scoring}: {results['mean_cv']:.3f} ± {results['std_cv']:.3f}")
        print(f"   Train R²: {results['mean_train']:.3f} ± {results['std_train']:.3f}")
        print(f"   Val R²: {results['mean_val']:.3f} ± {results['std_val']:.3f}")
        print(f"   Overfitting: {results['overfitting']:.3f}")
        
        return results
    
    def sensitivity_analysis(self, base_params, param_ranges, model_func, 
                           output_func=None, n_samples=100):
        """
        Sensitivity analysis for model parameters
        """
        print("📊 Running sensitivity analysis...")
        
        if output_func is None:
            output_func = lambda x: x  # Identity function
        
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        # Generate parameter samples
        param_samples = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            param_samples[param_name] = np.linspace(min_val, max_val, n_samples)
        
        # One-at-a-time sensitivity analysis
        sensitivities = {}
        
        for param_name in param_names:
            print(f"   Analyzing {param_name}...")
            
            outputs = []
            param_values = param_samples[param_name]
            
            for param_val in param_values:
                # Create modified parameters
                modified_params = base_params.copy()
                modified_params[param_name] = param_val
                
                # Run model
                try:
                    model_output = model_func(**modified_params)
                    final_output = output_func(model_output)
                    outputs.append(final_output)
                except:
                    outputs.append(np.nan)
            
            outputs = np.array(outputs)
            
            # Calculate sensitivity metrics
            valid_mask = ~np.isnan(outputs)
            if np.sum(valid_mask) > 2:
                param_vals_valid = param_values[valid_mask]
                outputs_valid = outputs[valid_mask]
                
                # Linear sensitivity (derivative)
                slope, _, r_value, _, _ = stats.linregress(param_vals_valid, outputs_valid)
                
                # Normalized sensitivity
                base_output = output_func(model_func(**base_params))
                base_param = base_params[param_name]
                
                normalized_sensitivity = slope * base_param / base_output if base_output != 0 else 0
                
                sensitivities[param_name] = {
                    'param_values': param_values,
                    'outputs': outputs,
                    'slope': slope,
                    'r_squared': r_value**2,
                    'normalized_sensitivity': normalized_sensitivity,
                    'relative_change': (np.max(outputs_valid) - np.min(outputs_valid)) / base_output
                }
                
                print(f"     Sensitivity: {slope:.4f}")
                print(f"     Normalized: {normalized_sensitivity:.4f}")
                print(f"     R²: {r_value**2:.3f}")
        
        results = {
            'sensitivities': sensitivities,
            'base_params': base_params,
            'param_ranges': param_ranges,
            'most_sensitive': max(sensitivities.keys(), 
                                key=lambda k: abs(sensitivities[k]['normalized_sensitivity']))
        }
        
        print(f"   Most sensitive parameter: {results['most_sensitive']}")
        
        return results
    
    def robust_regression(self, X, y, method='huber', **kwargs):
        """
        Robust regression analysis
        """
        print(f"💪 Running robust regression ({method})...")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if method == 'huber':
            from sklearn.linear_model import HuberRegressor
            model = HuberRegressor(**kwargs)
        elif method == 'ransac':
            from sklearn.linear_model import RANSACRegressor
            model = RANSACRegressor(**kwargs)
        elif method == 'theil_sen':
            from sklearn.linear_model import TheilSenRegressor
            model = TheilSenRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown robust method: {method}")
        
        # Fit model
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        residuals = y - y_pred
        
        # Outlier detection
        residual_threshold = 2 * np.std(residuals)
        outliers = np.abs(residuals) > residual_threshold
        
        results = {
            'model': model,
            'y_pred': y_pred,
            'r2': r2,
            'mse': mse,
            'residuals': residuals,
            'outliers': outliers,
            'n_outliers': np.sum(outliers),
            'outlier_fraction': np.sum(outliers) / len(y),
            'method': method
        }
        
        print(f"   {method.upper()} R²: {r2:.3f}")
        print(f"   MSE: {mse:.4f}")
        print(f"   Outliers: {np.sum(outliers)}/{len(y)} ({np.sum(outliers)/len(y)*100:.1f}%)")
        
        return results
    
    def monte_carlo_uncertainty(self, model_func, param_distributions, 
                              n_simulations=1000, seed=42):
        """
        Monte Carlo uncertainty propagation
        """
        print(f"🎲 Running Monte Carlo uncertainty analysis ({n_simulations} simulations)...")
        
        np.random.seed(seed)
        
        outputs = []
        param_samples = []
        
        for i in range(n_simulations):
            # Sample parameters from distributions
            params = {}
            param_sample = {}
            
            for param_name, (dist_type, *dist_params) in param_distributions.items():
                if dist_type == 'normal':
                    mean, std = dist_params
                    value = np.random.normal(mean, std)
                elif dist_type == 'uniform':
                    low, high = dist_params
                    value = np.random.uniform(low, high)
                elif dist_type == 'lognormal':
                    mean, sigma = dist_params
                    value = np.random.lognormal(mean, sigma)
                else:
                    raise ValueError(f"Unknown distribution: {dist_type}")
                
                params[param_name] = value
                param_sample[param_name] = value
            
            # Run model
            try:
                output = model_func(**params)
                outputs.append(output)
                param_samples.append(param_sample)
            except:
                continue
        
        outputs = np.array(outputs)
        param_samples_array = {param: [sample[param] for sample in param_samples] 
                              for param in param_distributions.keys()}
        
        # Calculate output statistics
        output_stats = {
            'mean': np.mean(outputs),
            'std': np.std(outputs),
            'median': np.median(outputs),
            'percentiles': {
                '5': np.percentile(outputs, 5),
                '25': np.percentile(outputs, 25),
                '75': np.percentile(outputs, 75),
                '95': np.percentile(outputs, 95)
            },
            'min': np.min(outputs),
            'max': np.max(outputs)
        }
        
        # Sensitivity analysis via correlation
        correlations = {}
        for param_name, param_values in param_samples_array.items():
            corr, p_value = stats.pearsonr(param_values, outputs)
            correlations[param_name] = {
                'correlation': corr,
                'p_value': p_value,
                'significance': p_value < 0.05
            }
        
        results = {
            'outputs': outputs,
            'param_samples': param_samples_array,
            'output_stats': output_stats,
            'correlations': correlations,
            'n_successful': len(outputs),
            'success_rate': len(outputs) / n_simulations
        }
        
        print(f"   Output: {output_stats['mean']:.4f} ± {output_stats['std']:.4f}")
        print(f"   95% range: [{output_stats['percentiles']['5']:.4f}, {output_stats['percentiles']['95']:.4f}]")
        print(f"   Success rate: {results['success_rate']*100:.1f}%")
        
        return results
    
    def convergence_analysis(self, sequence, window_size=50, tolerance=1e-4):
        """
        Analyze convergence of iterative sequence
        """
        print("🎯 Analyzing convergence...")
        
        sequence = np.array(sequence)
        n_points = len(sequence)
        
        if n_points < window_size:
            print(f"⚠️ Sequence too short for window size {window_size}")
            return None
        
        # Moving averages
        moving_avg = np.convolve(sequence, np.ones(window_size)/window_size, mode='valid')
        
        # Moving standard deviations
        moving_std = []
        for i in range(len(moving_avg)):
            start_idx = i
            end_idx = i + window_size
            window_data = sequence[start_idx:end_idx
