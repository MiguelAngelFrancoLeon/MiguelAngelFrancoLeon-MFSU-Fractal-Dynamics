print("✅ Demo complete! Check 'figures/' directory for outputs")

def create_mfsu_presentation_slides():
    """Create presentation-ready slide figures"""
    print("📽️ Creating presentation slides...")
    
    # Slide 1: Title slide concept
    fig1, ax = plt.subplots(figsize=(16, 9))  # 16:9 aspect ratio
    ax.text(0.5, 0.7, 'MFSU Framework', ha='center', va='center', 
            fontsize=48, fontweight='bold', color=MFSU_COLORS['primary'])
    ax.text(0.5, 0.6, 'Universal Fractal Constant δF = 0.921', 
            ha='center', va='center', fontsize=24, color=MFSU_COLORS['neutral'])
    ax.text(0.5, 0.4, 'Unifying Physics from Quantum to Cosmic Scales', 
            ha='center', va='center', fontsize=18, style='italic')
    ax.text(0.5, 0.2, 'Miguel Ángel Franco León', 
            ha='center', va='center', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add decorative elements
    circle = plt.Circle((0.1, 0.8), 0.05, color=MFSU_COLORS['primary'], alpha=0.3)
    ax.add_patch(circle)
    circle2 = plt.Circle((0.9, 0.2), 0.03, color=MFSU_COLORS['secondary'], alpha=0.5)
    ax.add_patch(circle2)
    
    save_publication_figure(fig1, "slide_title", formats=['png', 'pdf'])
    plt.close(fig1)
    
    # Slide 2: Key results summary
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 9))
    
    # Universal validation
    systems = ['CMB', 'SC', 'Diff', 'LSS']
    values = [0.921, 0.920, 0.#!/usr/bin/env python3
"""
Plotting Utilities for MFSU
Nature journal style plots and visualization tools

Author: Miguel Ángel Franco León
Date: August 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Nature journal configuration
NATURE_CONFIG = {
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5
}

# Color schemes (colorblind-friendly)
MFSU_COLORS = {
    'primary': '#1f77b4',     # Blue
    'secondary': '#ff7f0e',   # Orange
    'success': '#2ca02c',     # Green
    'danger': '#d62728',      # Red
    'theory': '#9467bd',      # Purple
    'neutral': '#7f7f7f',     # Gray
    'highlight': '#e377c2',   # Pink
    'dark': '#2f2f2f'         # Dark gray
}

# Color palettes
PALETTE_QUALITATIVE = [MFSU_COLORS['primary'], MFSU_COLORS['secondary'], 
                      MFSU_COLORS['success'], MFSU_COLORS['danger'], 
                      MFSU_COLORS['theory'], MFSU_COLORS['highlight']]

# Custom colormap for heatmaps
MFSU_CMAP = LinearSegmentedColormap.from_list(
    'mfsu', ['#2c3e50', '#3498db', '#e74c3c', '#f39c12'], N=256)

def setup_nature_style():
    """Setup matplotlib with Nature journal styling"""
    plt.rcParams.update(NATURE_CONFIG)
    print("📊 Nature journal style activated")

def create_figure_grid(nrows=2, ncols=2, figsize=None, 
                      width_ratios=None, height_ratios=None):
    """Create publication-quality figure grid"""
    if figsize is None:
        figsize = (7, 6) if nrows <= 2 and ncols <= 2 else (10, 8)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols, 
                          width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          hspace=0.3, wspace=0.3)
    
    return fig, gs

def save_publication_figure(fig, filename, formats=['eps', 'png', 'pdf']):
    """Save figure in multiple publication formats"""
    # Create directories
    for fmt in formats:
        Path(f'figures/{fmt}').mkdir(parents=True, exist_ok=True)
    
    # Save in each format
    for fmt in formats:
        filepath = f'figures/{fmt}/{filename}.{fmt}'
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
    
    print(f"💾 Figure saved: {filename} ({', '.join(formats)})")

def plot_power_spectrum(frequencies, power, model_fit=None, 
                       title="Power Spectrum", labels=None):
    """Plot power spectrum with optional model fit"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data
    ax.loglog(frequencies, power, 'o', markersize=3, alpha=0.7, 
              color=MFSU_COLORS['primary'], 
              label=labels[0] if labels else 'Data')
    
    # Model fit
    if model_fit is not None:
        ax.loglog(frequencies, model_fit, '--', linewidth=2, 
                  color=MFSU_COLORS['danger'], 
                  label=labels[1] if labels and len(labels) > 1 else 'Model')
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

def plot_correlation_matrix(matrix, labels=None, title="Correlation Matrix", 
                           cmap=None, annot=True):
    """Plot correlation matrix with values"""
    if cmap is None:
        cmap = 'Blues'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', rotation=270, labelpad=15)
    
    # Set labels
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    # Add correlation values
    if annot and matrix.shape[0] <= 10:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if matrix[i, j] < 0.5 else "black",
                             fontsize=7)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig, ax

def plot_fractal_dimension_comparison(systems, dimensions, errors, 
                                    theoretical_value=0.921):
    """Plot fractal dimension estimates across systems"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x_pos = np.arange(len(systems))
    
    # Bar plot with error bars
    bars = ax.bar(x_pos, dimensions, yerr=errors, 
                  color=MFSU_COLORS['primary'], alpha=0.7,
                  capsize=5, capthick=1, ecolor='black')
    
    # Theoretical line
    ax.axhline(y=theoretical_value, color=MFSU_COLORS['danger'], 
               linestyle='--', linewidth=2, 
               label=f'Theoretical δF = {theoretical_value}')
    
    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(systems, rotation=45, ha='right')
    ax.set_ylabel('δF Value')
    ax.set_title('Fractal Dimension Validation Across Systems')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, dim, err) in enumerate(zip(bars, dimensions, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.001,
                f'{dim:.3f}±{err:.3f}',
                ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    return fig, ax

def plot_performance_comparison(metrics, mfsu_values, standard_values, 
                              metric_names=None):
    """Plot performance comparison between MFSU and standard models"""
    if metric_names is None:
        metric_names = [f'Metric {i+1}' for i in range(len(metrics))]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Bars
    bars1 = ax.bar(x - width/2, mfsu_values, width, 
                   label='MFSU', color=MFSU_COLORS['primary'], alpha=0.8)
    bars2 = ax.bar(x + width/2, standard_values, width,
                   label='Standard', color=MFSU_COLORS['secondary'], alpha=0.8)
    
    # Add improvement percentages
    for i, (mfsu, std) in enumerate(zip(mfsu_values, standard_values)):
        if std != 0:
            improvement = (std - mfsu) / std * 100
            ax.text(i, max(mfsu, std) + 0.05 * max(max(mfsu_values), max(standard_values)), 
                   f'+{improvement:.0f}%', 
                   ha='center', fontsize=7, color=MFSU_COLORS['success'],
                   weight='bold')
    
    ax.set_xlabel('Systems')
    ax.set_ylabel('Performance Metric')
    ax.set_title('MFSU vs Standard Models Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def plot_residuals_analysis(observed, predicted, title="Residuals Analysis"):
    """Plot residuals analysis with QQ plot"""
    residuals = observed - predicted
    standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    # Residuals vs Fitted
    ax1.scatter(predicted, residuals, alpha=0.6, color=MFSU_COLORS['primary'], s=20)
    ax1.axhline(y=0, color=MFSU_COLORS['danger'], linestyle='--')
    ax1.axhline(y=2*np.std(residuals), color=MFSU_COLORS['neutral'], 
                linestyle=':', alpha=0.7)
    ax1.axhline(y=-2*np.std(residuals), color=MFSU_COLORS['neutral'], 
                linestyle=':', alpha=0.7)
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('(a) Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)
    
    # QQ plot
    from scipy import stats
    stats.probplot(standardized_residuals, dist="norm", plot=ax2)
    ax2.set_title('(b) Normal Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax3.hist(standardized_residuals, bins=20, alpha=0.7, 
             color=MFSU_COLORS['primary'], edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0, color=MFSU_COLORS['danger'], linestyle='--')
    ax3.set_xlabel('Standardized Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('(c) Residuals Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Scale-Location plot
    sqrt_abs_residuals = np.sqrt(np.abs(standardized_residuals))
    ax4.scatter(predicted, sqrt_abs_residuals, alpha=0.6, 
                color=MFSU_COLORS['primary'], s=20)
    
    # Add trend line
    z = np.polyfit(predicted, sqrt_abs_residuals, 1)
    p = np.poly1d(z)
    ax4.plot(predicted, p(predicted), color=MFSU_COLORS['danger'], linestyle='-')
    
    ax4.set_xlabel('Fitted Values')
    ax4.set_ylabel('√|Standardized Residuals|')
    ax4.set_title('(d) Scale-Location')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout()
    
    return fig, (ax1, ax2, ax3, ax4)

def plot_bootstrap_distribution(samples, true_value=None, confidence_level=95):
    """Plot bootstrap distribution with confidence intervals"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Histogram
    ax1.hist(samples, bins=50, alpha=0.7, color=MFSU_COLORS['primary'], 
             edgecolor='black', linewidth=0.5, density=True)
    
    # Statistics
    mean_val = np.mean(samples)
    std_val = np.std(samples)
    ci_lower = np.percentile(samples, (100 - confidence_level) / 2)
    ci_upper = np.percentile(samples, 100 - (100 - confidence_level) / 2)
    
    # Add lines
    ax1.axvline(mean_val, color=MFSU_COLORS['danger'], linestyle='-', 
                linewidth=2, label=f'Mean = {mean_val:.3f}')
    ax1.axvline(ci_lower, color=MFSU_COLORS['neutral'], linestyle='--', 
                alpha=0.7, label=f'{confidence_level}% CI')
    ax1.axvline(ci_upper, color=MFSU_COLORS['neutral'], linestyle='--', alpha=0.7)
    
    if true_value is not None:
        ax1.axvline(true_value, color=MFSU_COLORS['success'], linestyle=':', 
                    linewidth=2, label=f'True = {true_value:.3f}')
    
    ax1.set_xlabel('Parameter Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Bootstrap Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    box_data = [samples]
    box = ax2.boxplot(box_data, patch_artist=True, notch=True, 
                      showmeans=True, meanline=True)
    
    # Customize box plot
    box['boxes'][0].set_facecolor(MFSU_COLORS['primary'])
    box['boxes'][0].set_alpha(0.7)
    box['medians'][0].set_color(MFSU_COLORS['danger'])
    box['means'][0].set_color(MFSU_COLORS['success'])
    box['means'][0].set_linewidth(2)
    
    if true_value is not None:
        ax2.axhline(true_value, color=MFSU_COLORS['success'], 
                    linestyle=':', linewidth=2, alpha=0.7)
    
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Bootstrap Statistics')
    ax2.set_xticklabels(['Samples'])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"📊 Bootstrap Statistics:")
    print(f"   Mean: {mean_val:.4f} ± {std_val:.4f}")
    print(f"   {confidence_level}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    if true_value is not None:
        bias = mean_val - true_value
        print(f"   Bias: {bias:.4f}")
        print(f"   Coverage: {'✅' if ci_lower <= true_value <= ci_upper else '❌'}")
    
    return fig, (ax1, ax2)

def plot_time_series_analysis(t, data, model_fit=None, title="Time Series"):
    """Plot time series with optional model fit"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    # Main time series
    ax1.plot(t, data, 'o-', markersize=2, linewidth=1, alpha=0.7,
             color=MFSU_COLORS['primary'], label='Data')
    
    if model_fit is not None:
        ax1.plot(t, model_fit, '--', linewidth=2, 
                 color=MFSU_COLORS['danger'], label='MFSU Model')
        
        # Residuals
        residuals = data - model_fit
        ax2.plot(t, residuals, 'o-', markersize=1, linewidth=0.5,
                 color=MFSU_COLORS['neutral'], alpha=0.7)
        ax2.axhline(0, color=MFSU_COLORS['danger'], linestyle='--')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
    
    ax1.set_ylabel('Amplitude')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time')
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def plot_fractal_surface(surface, title="Fractal Surface", cmap=None):
    """Plot 2D fractal surface with colorbar"""
    if cmap is None:
        cmap = MFSU_CMAP
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(surface, cmap=cmap, aspect='auto', 
                   origin='lower', interpolation='bilinear')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Amplitude', rotation=270, labelpad=15)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    
    # Add statistics text box
    stats_text = f"""Statistics:
Mean: {np.mean(surface):.3f}
Std: {np.std(surface):.3f}
Min: {np.min(surface):.3f}
Max: {np.max(surface):.3f}"""
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=8, verticalalignment='top', bbox=props)
    
    return fig, ax

def plot_galaxy_rotation_curve(radii, velocities, model_velocities=None, 
                              errors=None, title="Galaxy Rotation Curve"):
    """Plot galaxy rotation curve with model comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Observed data
    if errors is not None:
        ax.errorbar(radii, velocities, yerr=errors, fmt='o', 
                    markersize=4, capsize=3, capthick=1,
                    color=MFSU_COLORS['primary'], alpha=0.7,
                    label='Observations')
    else:
        ax.plot(radii, velocities, 'o', markersize=4, 
                color=MFSU_COLORS['primary'], alpha=0.7,
                label='Observations')
    
    # Model fit
    if model_velocities is not None:
        ax.plot(radii, model_velocities, '-', linewidth=2,
                color=MFSU_COLORS['danger'], label='MFSU Model')
    
    # Expected Newtonian decline
    v_max = np.max(velocities)
    r_max = radii[np.argmax(velocities)]
    v_newtonian = v_max * np.sqrt(r_max / radii)
    
    ax.plot(radii, v_newtonian, ':', linewidth=2, alpha=0.7,
            color=MFSU_COLORS['neutral'], label='Newtonian')
    
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel('Velocity [km/s]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add flat rotation annotation
    flat_region = radii > r_max
    if np.any(flat_region):
        flat_velocities = velocities[flat_region]
        flat_mean = np.mean(flat_velocities)
        flat_std = np.std(flat_velocities)
        
        ax.axhline(flat_mean, color=MFSU_COLORS['success'], 
                   linestyle='--', alpha=0.5)
        ax.fill_between(radii, flat_mean - flat_std, flat_mean + flat_std,
                        alpha=0.2, color=MFSU_COLORS['success'],
                        label=f'Flat region: {flat_mean:.0f}±{flat_std:.0f} km/s')
        ax.legend()
    
    return fig, ax

def plot_cmb_angular_spectrum(ell, cl_obs, cl_model=None, cl_errors=None,
                             title="CMB Angular Power Spectrum"):
    """Plot CMB angular power spectrum"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main spectrum
    if cl_errors is not None:
        ax1.errorbar(ell, cl_obs, yerr=cl_errors, fmt='o', 
                     markersize=2, capsize=1, alpha=0.7,
                     color=MFSU_COLORS['primary'], label='Planck 2018')
    else:
        ax1.loglog(ell, cl_obs, 'o', markersize=2, alpha=0.7,
                   color=MFSU_COLORS['primary'], label='Planck 2018')
    
    if cl_model is not None:
        ax1.loglog(ell, cl_model, '-', linewidth=2,
                   color=MFSU_COLORS['danger'], label='MFSU Model')
    
    ax1.set_ylabel('C_ℓ [μK²]')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Residuals
    if cl_model is not None and cl_errors is not None:
        residuals = (cl_obs - cl_model) / cl_errors
        ax2.semilogx(ell, residuals, 'o', markersize=2,
                     color=MFSU_COLORS['primary'], alpha=0.7)
        ax2.axhline(0, color=MFSU_COLORS['danger'], linestyle='-')
        ax2.axhline(1, color=MFSU_COLORS['neutral'], linestyle='--', alpha=0.7)
        ax2.axhline(-1, color=MFSU_COLORS['neutral'], linestyle='--', alpha=0.7)
        ax2.axhline(2, color=MFSU_COLORS['neutral'], linestyle=':', alpha=0.5)
        ax2.axhline(-2, color=MFSU_COLORS['neutral'], linestyle=':', alpha=0.5)
        
        ax2.set_ylabel('Residuals (σ)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-3, 3)
    
    ax2.set_xlabel('Multipole ℓ')
    plt.tight_layout()
    
    return fig, (ax1, ax2)

def plot_superconductor_data(materials, tc_exp, tc_model, tc_errors=None):
    """Plot superconductor critical temperature data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(materials))
    width = 0.35
    
    # Experimental data
    bars1 = ax.bar(x_pos - width/2, tc_exp, width, 
                   yerr=tc_errors if tc_errors is not None else None,
                   label='Experimental', color=MFSU_COLORS['primary'], 
                   alpha=0.8, capsize=5)
    
    # Model predictions
    bars2 = ax.bar(x_pos + width/2, tc_model, width,
                   label='MFSU Model', color=MFSU_COLORS['danger'], 
                   alpha=0.8)
    
    # Add error percentages
    for i, (exp, mod) in enumerate(zip(tc_exp, tc_model)):
        error_percent = abs(exp - mod) / exp * 100
        ax.text(i, max(exp, mod) + (max(tc_exp) * 0.02), 
                f'{error_percent:.1f}%',
                ha='center', fontsize=8, 
                color=MFSU_COLORS['success'] if error_percent < 2 else MFSU_COLORS['neutral'])
    
    ax.set_xlabel('Superconductor')
    ax.set_ylabel('Critical Temperature [K]')
    ax.set_title('Superconductor Critical Temperature Predictions')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(materials, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def create_publication_summary_figure():
    """Create comprehensive summary figure for publication"""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.4)
    
    # Placeholder data - replace with real data
    
    # Top row: Universal validation
    ax1 = fig.add_subplot(gs[0, :])
    systems = ['CMB', 'Supercond.', 'Diffusion', 'LSS', 'Quantum']
    delta_values = [0.921, 0.920, 0.922, 0.921, 0.921]
    delta_errors = [0.003, 0.002, 0.004, 0.003, 0.003]
    
    plot_fractal_dimension_comparison(systems, delta_values, delta_errors)
    ax1 = plt.gca()
    ax1.set_title('(a) Universal δF Validation Across Physical Systems')
    
    # Middle left: CMB spectrum
    ax2 = fig.add_subplot(gs[1, 0])
    ell = np.logspace(1, 3, 50)
    cl_data = 2500 * ell**(-0.921) * (1 + 0.1 * np.random.randn(len(ell)))
    cl_model = 2500 * ell**(-0.921)
    
    ax2.loglog(ell, cl_data, 'o', markersize=2, alpha=0.7, 
               color=MFSU_COLORS['primary'], label='Planck')
    ax2.loglog(ell, cl_model, '-', linewidth=2, 
               color=MFSU_COLORS['danger'], label='MFSU')
    ax2.set_xlabel('Multipole ℓ')
    ax2.set_ylabel('C_ℓ [μK²]')
    ax2.set_title('(b) CMB Power Spectrum')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # Middle center: Performance comparison
    ax3 = fig.add_subplot(gs[1, 1])
    metrics = ['CMB', 'SC', 'Diff']
    mfsu_vals = [0.77, 0.85, 0.92]
    std_vals = [1.00, 1.00, 1.00]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, mfsu_vals, width, label='MFSU', 
            color=MFSU_COLORS['primary'], alpha=0.8)
    ax3.bar(x + width/2, std_vals, width, label='Standard', 
            color=MFSU_COLORS['secondary'], alpha=0.8)
    ax3.set_ylabel('χ² (normalized)')
    ax3.set_title('(c) Model Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # Middle right: Correlation matrix
    ax4 = fig.add_subplot(gs[1, 2])
    corr_matrix = np.array([[1.00, 0.95, 0.93], 
                           [0.95, 1.00, 0.97], 
                           [0.93, 0.97, 1.00]])
    labels = ['CMB', 'SC', 'Diff']
    
    im = ax4.imshow(corr_matrix, cmap='Blues', vmin=0.8, vmax=1.0)
    ax4.set_xticks(range(len(labels)))
    ax4.set_yticks(range(len(labels)))
    ax4.set_xticklabels(labels)
    ax4.set_yticklabels(labels)
    ax4.set_title('(d) Method Correlations')
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax4.text(j, i, f'{corr_matrix[i,j]:.2f}', 
                    ha='center', va='center', fontsize=7)
    
    # Bottom row: Physical applications
    ax5 = fig.add_subplot(gs[2, 0])
    r = np.linspace(1, 20, 50)
    v_obs = 200 * np.ones_like(r) * (1 + 0.1 * np.random.randn(len(r)))
    v_model = 200 * (r/8)**(-0.921/4)
    
    ax5.plot(r, v_obs, 'o', markersize=2, alpha=0.7, 
             color=MFSU_COLORS['primary'], label='Observations')
    ax5.plot(r, v_model, '-', linewidth=2, 
             color=MFSU_COLORS['danger'], label='MFSU')
    ax5.set_xlabel('Radius [kpc]')
    ax5.set_ylabel('Velocity [km/s]')
    ax5.set_title('(e) Galaxy Rotation')
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    t = np.logspace(-1, 2, 30)
    msd_data = 0.5 * t**0.921 * (1 + 0.1 * np.random.randn(len(t)))
    msd_model = 0.5 * t**0.921
    
    ax6.loglog(t, msd_data, 'o', markersize=3, alpha=0.7, 
               color=MFSU_COLORS['primary'], label='Data')
    ax6.loglog(t, msd_model, '-', linewidth=2, 
               color=MFSU_COLORS['danger'], label='MFSU')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('MSD')
    ax6.set_title('(f) Anomalous Diffusion')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 2])
    materials = ['YBCO', 'BSCCO', 'Tl-2212']
    tc_exp = [93, 95, 108]
    tc_model = [92.3, 94.1, 107.2]
    
    x_pos = np.arange(len(materials))
    ax7.bar(x_pos, tc_exp, alpha=0.7, color=MFSU_COLORS['primary'], label='Exp.')
    ax7.scatter(x_pos, tc_model, color=MFSU_COLORS['danger'], 
                s=50, marker='x', linewidth=3, label='MFSU')
    ax7.set_ylabel('Tc [K]')
    ax7.set_title('(g) Superconductors')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(materials, fontsize=7)
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('MFSU: Universal Fractal Constant δF = 0.921', 
                 fontsize=14, fontweight='bold')
    
    return fig

def demo_plotting_utilities():
    """Demonstrate plotting utilities"""
    print("🎨 MFSU Plotting Utilities Demo")
    print("=" * 40)
    
    # Setup style
    setup_nature_style()
    
    # Generate sample data
    np.random.seed(42)
    
    # Power spectrum demo
    freqs = np.logspace(-2, 2, 50)
    power = 100 * freqs**(-0.921) * (1 + 0.1 * np.random.randn(len(freqs)))
    model = 100 * freqs**(-0.921)
    
    print("📊 Creating power spectrum plot...")
    fig1, _ = plot_power_spectrum(freqs, power, model, 
                                "MFSU Power Spectrum Example")
    save_publication_figure(fig1, "demo_power_spectrum")
    plt.close(fig1)
    
    # Correlation matrix demo
    print("🔗 Creating correlation matrix...")
    corr_data = np.random.rand(5, 5)
    corr_data = (corr_data + corr_data.T) / 2  # Make symmetric
    np.fill_diagonal(corr_data, 1.0)
    labels = ['CMB', 'SC', 'Diff', 'LSS', 'QM']
    
    fig2, _ = plot_correlation_matrix(corr_data, labels, 
                                    "Method Correlations")
    save_publication_figure(fig2, "demo_correlation_matrix")
    plt.close(fig2)
    
    # Bootstrap demo
    print("🔄 Creating bootstrap distribution...")
    samples = np.random.normal(0.921, 0.003, 1000)
    fig3, _ = plot_bootstrap_distribution(samples, true_value=0.921)
    save_publication_figure(fig3, "demo_bootstrap")
    plt.close(fig3)
    
    print("✅ Demo complete! Check 'figures/' directory for outputs")

if __name__ == "__main__":
    demo_plotting_utilities()
