"""
MFSU Fractal and Stochastic Operators
=====================================

This module implements the specialized mathematical operators used in the
Unified Fractal-Stochastic Model (MFSU):

1. Fractional Laplacian: (−Δ)^(δF/2) for anomalous diffusion
2. Fractal gradient: ∇^δF for non-integer derivatives  
3. Hurst noise generator: ξH(x,t) for long-range correlations
4. Box-counting operators: For fractal dimension analysis
5. Multifractal spectrum: For complex scaling analysis

These operators are the mathematical foundation that enables MFSU to model
scale-invariant phenomena across quantum, classical, and cosmological domains.

Author: Miguel Ángel Franco León
Mathematical Foundation: Fractional calculus and stochastic analysis
"""

import numpy as np
from typing import Union, Optional, Tuple, Dict, Callable
import warnings
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from scipy.special import gamma
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

from .constants import FRANCO_CONSTANT, FRACTAL_DIMENSION, HURST_EXPONENT

# ==============================================================================
# FRACTIONAL LAPLACIAN OPERATOR
# ==============================================================================

def fractional_laplacian(
    field: np.ndarray,
    order: float = FRANCO_CONSTANT,
    method: str = 'fourier',
    dx: float = 1.0
) -> np.ndarray:
    """
    Compute the fractional Laplacian (−Δ)^(α/2) of a field.
    
    This is the core operator in MFSU equations, representing anomalous
    diffusion in fractal media. The fractional order α ≈ 0.921 (Franco Constant)
    governs the non-local nature of the operator.
    
    Mathematical Definition:
    ----------------------
    (−Δ)^(α/2) u(x) = F^(-1){|k|^α F{u}(k)}
    
    where F is the Fourier transform and |k| is the wavenumber magnitude.
    
    Parameters:
    -----------
    field : np.ndarray
        Input field (2D or 3D)
    order : float, default=FRANCO_CONSTANT
        Fractional order α (typically ≈ 0.921)
    method : str, default='fourier'
        Computation method ('fourier', 'integral', 'spectral')
    dx : float, default=1.0
        Spatial discretization step
        
    Returns:
    --------
    np.ndarray
        Fractional Laplacian of the input field
        
    Notes:
    ------
    The Fourier method is most efficient and accurate for periodic boundary
    conditions. For other boundary conditions, use 'integral' method.
    
    Examples:
    ---------
    >>> import numpy as np
    >>> from mfsu.core.operators import fractional_laplacian
    >>> field = np.random.rand(64, 64)  
    >>> frac_lap = fractional_laplacian(field, order=0.921)
    >>> print(f"Field shape: {field.shape}, Output shape: {frac_lap.shape}")
    """
    if field.ndim not in [2, 3]:
        raise ValueError("Field must be 2D or 3D array")
    
    if not (0 < order <= 2):
        raise ValueError("Fractional order must be in (0, 2]")
    
    if method == 'fourier':
        return _fractional_laplacian_fourier(field, order, dx)
    elif method == 'integral':
        return _fractional_laplacian_integral(field, order, dx)
    elif method == 'spectral':
        return _fractional_laplacian_spectral(field, order)
    else:
        raise ValueError(f"Unknown method: {method}")

def _fractional_laplacian_fourier(field: np.ndarray, order: float, dx: float) -> np.ndarray:
    """Fourier-based implementation of fractional Laplacian"""
    # Get field dimensions
    shape = field.shape
    ndim = len(shape)
    
    # Compute Fourier transform
    field_ft = fft2(field) if ndim == 2 else np.fft.fftn(field)
    
    # Generate frequency grids
    if ndim == 2:
        kx = fftfreq(shape[0], dx)
        ky = fftfreq(shape[1], dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k_squared = KX**2 + KY**2
    else:  # 3D
        kx = fftfreq(shape[0], dx)
        ky = fftfreq(shape[1], dx)
        kz = fftfreq(shape[2], dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_squared = KX**2 + KY**2 + KZ**2
    
    # Avoid division by zero at k=0
    k_squared[0, 0] = 1.0  # Will be set to zero after computation
    
    # Apply fractional power: |k|^α
    k_magnitude = np.sqrt(k_squared)
    fractional_operator = np.power(k_magnitude, order)
    
    # Set zero frequency to zero (preserves mean)
    fractional_operator[0, 0] = 0.0
    
    # Apply operator in Fourier space
    result_ft = field_ft * fractional_operator
    
    # Transform back to real space
    if ndim == 2:
        result = np.real(ifft2(result_ft))
    else:
        result = np.real(np.fft.ifftn(result_ft))
    
    return result

def _fractional_laplacian_integral(field: np.ndarray, order: float, dx: float) -> np.ndarray:
    """Integral-based implementation (slower but handles arbitrary boundaries)"""
    # This is a simplified implementation for demonstration
    # Full implementation would use hypersingular integrals
    warnings.warn("Integral method is simplified - use Fourier method for accuracy")
    
    # Use finite differences as approximation
    if order <= 1:
        # First-order approximation
        if field.ndim == 2:
            grad_x = np.gradient(field, dx, axis=0)
            grad_y = np.gradient(field, dx, axis=1)
            return order * (grad_x + grad_y)
        else:
            return order * np.sum([np.gradient(field, dx, axis=i) 
                                 for i in range(field.ndim)], axis=0)
    else:
        # Second-order approximation with fractional scaling
        if field.ndim == 2:
            laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 
                        4 * field) / dx**2
            return np.power(abs(laplacian) + 1e-10, order/2) * np.sign(laplacian)
        else:
            # 3D case (simplified)
            laplacian = np.zeros_like(field)
            for axis in range(field.ndim):
                laplacian += (np.roll(field, 1, axis=axis) + 
                             np.roll(field, -1, axis=axis) - 2 * field) / dx**2
            return np.power(abs(laplacian) + 1e-10, order/2) * np.sign(laplacian)

def _fractional_laplacian_spectral(field: np.ndarray, order: float) -> np.ndarray:
    """Spectral method using eigenfunction expansion"""
    warnings.warn("Spectral method is experimental")
    # Simplified implementation - full version would use proper spectral basis
    return _fractional_laplacian_fourier(field, order, 1.0)

# ==============================================================================
# FRACTAL GRADIENT OPERATOR
# ==============================================================================

def fractal_gradient(
    field: np.ndarray,
    delta_f: float = FRANCO_CONSTANT,
    dx: float = 1.0,
    method: str = 'finite_difference'
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Compute the fractal gradient ∇^δF of a field.
    
    This operator generalizes the classical gradient to fractal dimensions,
    important for analyzing fields in non-integer dimensional spaces.
    
    Parameters:
    -----------
    field : np.ndarray
        Input field
    delta_f : float, default=FRANCO_CONSTANT
        Fractal dimension parameter
    dx : float, default=1.0
        Spatial step size
    method : str, default='finite_difference'
        Computation method
        
    Returns:
    --------
    np.ndarray or tuple
        Fractal gradient components
    """
    if method == 'finite_difference':
        # Modified finite differences with fractal scaling
        gradients = np.gradient(field, dx)
        
        # Apply fractal scaling: |∇u|^(δF-1) * ∇u
        grad_magnitude = np.sqrt(sum(g**2 for g in gradients))
        fractal_scaling = np.power(grad_magnitude + 1e-12, delta_f - 1)
        
        if isinstance(gradients, tuple):
            return tuple(fractal_scaling * g for g in gradients)
        else:
            return fractal_scaling * gradients
    
    elif method == 'spectral':
        # Spectral differentiation with fractal correction
        gradients = []
        for axis in range(field.ndim):
            # Standard spectral derivative
            field_ft = fft2(field) if field.ndim == 2 else np.fft.fftn(field)
            k = fftfreq(field.shape[axis], dx) * 2j * np.pi
            
            # Apply derivative
            if field.ndim == 2 and axis == 0:
                grad_ft = field_ft * k[:, np.newaxis]
            elif field.ndim == 2 and axis == 1:
                grad_ft = field_ft * k[np.newaxis, :]
            else:
                # 3D case (more complex indexing needed)
                grad_ft = field_ft  # Simplified
            
            # Transform back
            grad = np.real(ifft2(grad_ft)) if field.ndim == 2 else np.real(np.fft.ifftn(grad_ft))
            gradients.append(grad)
        
        # Apply fractal scaling
        grad_magnitude = np.sqrt(sum(g**2 for g in gradients))
        fractal_scaling = np.power(grad_magnitude + 1e-12, delta_f - 1)
        
        return tuple(fractal_scaling * g for g in gradients)
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ==============================================================================
# HURST NOISE GENERATOR
# ==============================================================================

def hurst_noise_generator(
    shape: Tuple[int, ...],
    hurst: float = HURST_EXPONENT,
    dt: float = 0.01,
    method: str = 'circulant_embedding'
) -> np.ndarray:
    """
    Generate fractional Gaussian noise with specified Hurst exponent.
    
    This creates the stochastic component ξH(x,t) in MFSU equations,
    providing long-range temporal and spatial correlations.
    
    Parameters:
    -----------
    shape : tuple
        Shape of output noise array
    hurst : float, default=HURST_EXPONENT
        Hurst exponent H ∈ (0,1), H ≈ 0.541 for MFSU
    dt : float, default=0.01
        Time step (affects temporal correlations)
    method : str, default='circulant_embedding'
        Generation method ('circulant_embedding', 'fft', 'simple')
        
    Returns:
    --------
    np.ndarray
        Correlated noise field ξH(x,t)
        
    Notes:
    ------
    - H = 0.5: Standard Brownian motion (uncorrelated)
    - H > 0.5: Persistent correlations (trending)
    - H < 0.5: Anti-persistent correlations (mean-reverting)
    - H ≈ 0.541: MFSU validated value from CMB analysis
    """
    if not (0 < hurst < 1):
        raise ValueError("Hurst exponent must be in (0, 1)")
    
    if method == 'circulant_embedding':
        return _hurst_noise_circulant(shape, hurst, dt)
    elif method == 'fft':
        return _hurst_noise_fft(shape, hurst, dt)  
    elif method == 'simple':
        return _hurst_noise_simple(shape, hurst)
    else:
        raise ValueError(f"Unknown method: {method}")

def _hurst_noise_circulant(shape: Tuple[int, ...], hurst: float, dt: float) -> np.ndarray:
    """Generate Hurst noise using circulant embedding method"""
    # For spatial correlations, use simplified approach
    if len(shape) == 2:
        # 2D spatial noise
        white_noise = np.random.normal(0, 1, shape)
        
        # Create correlation kernel
        n, m = shape
        x = np.arange(n)
        y = np.arange(m)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2) + dt
        
        # Hurst correlation: C(r) ∝ r^(2H-1)
        correlation = np.power(distance, 2*hurst - 1)
        correlation /= np.sum(correlation)  # Normalize
        
        # Apply correlation via convolution (simplified)
        # Full implementation would use circulant matrix embedding
        correlated_noise = gaussian_filter(white_noise, sigma=hurst)
        
        return correlated_noise
    
    else:
        # For other dimensions, use simple scaling
        return _hurst_noise_simple(shape, hurst)

def _hurst_noise_fft(shape: Tuple[int, ...], hurst: float, dt: float) -> np.ndarray:
    """Generate Hurst noise using FFT method"""
    # Generate white noise
    white_noise = np.random.normal(0, 1, shape)
    
    # Apply Hurst scaling in Fourier domain
    if len(shape) == 2:
        # 2D case
        noise_ft = fft2(white_noise)
        
        # Create frequency grid
        kx = fftfreq(shape[0])
        ky = fftfreq(shape[1])
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k_mag = np.sqrt(KX**2 + KY**2)
        k_mag[0, 0] = 1.0  # Avoid division by zero
        
        # Apply Hurst scaling: |k|^(-(H+1))
        hurst_filter = np.power(k_mag, -(hurst + 0.5))
        hurst_filter[0, 0] = 0.0  # Zero mean
        
        # Apply filter and transform back
        filtered_ft = noise_ft * hurst_filter
        correlated_noise = np.real(ifft2(filtered_ft))
        
        return correlated_noise
    
    else:
        # Fallback to simple method
        return _hurst_noise_simple(shape, hurst)

def _hurst_noise_simple(shape: Tuple[int, ...], hurst: float) -> np.ndarray:
    """Simple Hurst noise using scaling transformation"""
    # Generate white noise
    white_noise = np.random.normal(0, 1, shape)
    
    # Apply simple scaling (not exact but computationally fast)
    scaling_factor = np.power(np.prod(shape), hurst - 0.5)
    
    return white_noise * scaling_factor

# ==============================================================================
# BOX-COUNTING FRACTAL DIMENSION
# ==============================================================================

def box_counting_dimension(
    field: np.ndarray,
    threshold: Optional[float] = None,
    scales: Optional[np.ndarray] = None,
    method: str = 'binary'
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate fractal dimension using box-counting method.
    
    This is used to validate that MFSU solutions exhibit the expected
    fractal dimension df ≈ 2.079 (corresponding to δF ≈ 0.921).
    
    Parameters:
    -----------
    field : np.ndarray
        2D field to analyze
    threshold : float, optional
        Threshold for binarization (if None, use mean)
    scales : np.ndarray, optional
        Box sizes to test (if None, use powers of 2)
    method : str, default='binary'
        Analysis method ('binary', 'intensity')
        
    Returns:
    --------
    fractal_dim : float
        Estimated fractal dimension
    scales : np.ndarray
        Box sizes used
    counts : np.ndarray
        Number of boxes at each scale
        
    Mathematical Background:
    ----------------------
    Fractal dimension D is defined by:
    N(ε) ∝ ε^(-D)
    
    where N(ε) is the number of boxes of size ε needed to cover the fractal.
    Taking logarithms: log N(ε) = -D log ε + const
    
    So D is the negative slope of log N vs log ε.
    """
    if field.ndim != 2:
        raise ValueError("Box counting requires 2D field")
    
    # Set default scales (powers of 2)
    if scales is None:
        max_scale = min(field.shape) // 2
        scales = 2**np.arange(1, int(np.log2(max_scale)) + 1)
    
    # Binarize field if using binary method
    if method == 'binary':
        if threshold is None:
            threshold = np.mean(field)
        binary_field = (field > threshold).astype(int)
        analysis_field = binary_field
    else:
        analysis_field = field
    
    counts = []
    
    for scale in scales:
        if method == 'binary':
            count = _count_boxes_binary(analysis_field, scale)
        else:
            count = _count_boxes_intensity(analysis_field, scale)
        counts.append(count)
    
    counts = np.array(counts)
    
    # Fit log-log relationship: log(counts) vs log(1/scales)
    log_scales = np.log(1.0 / scales)
    log_counts = np.log(counts + 1e-10)  # Avoid log(0)
    
    # Linear regression
    fractal_dim = np.polyfit(log_scales, log_counts, 1)[0]
    
    return fractal_dim, scales, counts

def _count_boxes_binary(binary_field: np.ndarray, box_size: int) -> int:
    """Count non-empty boxes for binary field"""
    h, w = binary_field.shape
    count = 0
    
    for i in range(0, h, box_size):
        for j in range(0, w, box_size):
            box = binary_field[i:i+box_size, j:j+box_size]
            if np.any(box):  # Box contains at least one 1
                count += 1
    
    return count

def _count_boxes_intensity(field: np.ndarray, box_size: int) -> int:
    """Count boxes with significant intensity variation"""
    h, w = field.shape
    count = 0
    threshold = 0.1 * np.std(field)  # Arbitrary threshold
    
    for i in range(0, h, box_size):
        for j in range(0, w, box_size):
            box = field[i:i+box_size, j:j+box_size]
            if np.std(box) > threshold:  # Box has significant variation
                count += 1
    
    return count

# ==============================================================================
# OPERATOR CLASSES
# ==============================================================================

class FractalOperator:
    """
    Class for managing fractal differential operators
    """
    
    def __init__(self, delta_f: float = FRANCO_CONSTANT, dx: float = 1.0):
        """Initialize fractal operator with given parameters"""
        self.delta_f = delta_f
        self.dx = dx
        self.fractal_dimension = 3.0 - delta_f
    
    def laplacian(self, field: np.ndarray, method: str = 'fourier') -> np.ndarray:
        """Apply fractional Laplacian"""
        return fractional_laplacian(field, self.delta_f, method, self.dx)
    
    def gradient(self, field: np.ndarray, method: str = 'finite_difference') -> np.ndarray:
        """Apply fractal gradient"""
        return fractal_gradient(field, self.delta_f, self.dx, method)
    
    def divergence(self, vector_field: Tuple[np.ndarray, ...]) -> np.ndarray:
        """Compute fractal divergence of vector field"""
        # Standard divergence with fractal scaling
        div = sum(np.gradient(component, self.dx, axis=i) 
                 for i, component in enumerate(vector_field))
        
        # Apply fractal scaling
        div_magnitude = np.abs(div)
        fractal_scaling = np.power(div_magnitude + 1e-12, self.delta_f - 1)
        
        return fractal_scaling * div
    
    def analyze_fractal_dimension(self, field: np.ndarray) -> Dict[str, float]:
        """Analyze fractal properties of a field"""
        if field.ndim == 2:
            dim, scales, counts = box_counting_dimension(field)
            return {
                'fractal_dimension': dim,
                'expected_dimension': self.fractal_dimension,
                'delta_f_estimate': 3.0 - dim,
                'match_quality': abs(dim - self.fractal_dimension)
            }
        else:
            # For higher dimensions, use simplified analysis
            power_spectrum = np.abs(np.fft.fftn(field))**2
            k = fftfreq(field.shape[0])
            k_positive = k[k > 0]
            ps_positive = power_spectrum.flatten()[:len(k_positive)]
            
            # Fit power law
            log_k = np.log(k_positive)
            log_ps = np.log(ps_positive[ps_positive > 0])
            if len(log_ps) > 5:
                slope = np.polyfit(log_k[:len(log_ps)], log_ps, 1)[0]
                estimated_delta_f = -slope - 2
                return {
                    'power_law_slope': slope,
                    'delta_f_estimate': estimated_delta_f,
                    'expected_delta_f': self.delta_f,
                    'match_quality': abs(estimated_delta_f - self.delta_f)
                }
            else:
                return {'error': 'Insufficient data for analysis'}

class StochasticOperator:
    """
    Class for managing stochastic processes in MFSU
    """
    
    def __init__(self, hurst: float = HURST_EXPONENT, dt: float = 0.01):
        """Initialize stochastic operator"""
        self.hurst = hurst
        self.dt = dt
    
    def generate_noise(self, shape: Tuple[int, ...], method: str = 'fft') -> np.ndarray:
        """Generate correlated noise"""
        return hurst_noise_generator(shape, self.hurst, self.dt, method)
    
    def analyze_correlations(self, time_series: np.ndarray) -> Dict[str, float]:
        """Analyze temporal correlations in a time series"""
        if time_series.ndim == 1:
            # 1D time series analysis
            n = len(time_series)
            
            # Detrended Fluctuation Analysis (simplified)
            scales = np.logspace(1, np.log10(n//4), 10, dtype=int)
            fluctuations = []
            
            for scale in scales:
                # Divide into non-overlapping windows
                n_windows = n // scale
                if n_windows < 2:
                    continue
                
                window_flucts = []
                for i in range(n_windows):
                    window = time_series[i*scale:(i+1)*scale]
                    # Detrend (remove linear trend)
                    t = np.arange(len(window))
                    if len(window) > 1:
                        detrended = window - np.polyval(np.polyfit(t, window, 1), t)
                        window_flucts.append(np.std(detrended))
                
                if window_flucts:
                    fluctuations.append(np.mean(window_flucts))
            
            # Fit power law to get Hurst exponent
            if len(fluctuations) > 3:
                log_scales = np.log(scales[:len(fluctuations)])
                log_flucts = np.log(fluctuations)
                hurst_estimate = np.polyfit(log_scales, log_flucts, 1)[0]
                
                return {
                    'hurst_estimate': hurst_estimate,
                    'expected_hurst': self.hurst,
                    'match_quality': abs(hurst_estimate - self.hurst)
                }
            else:
                return {'error': 'Insufficient data for Hurst analysis'}
        
        else:
            return {'error': 'Multi-dimensional correlation analysis not implemented'}

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def validate_fractal_field(
    field: np.ndarray,
    expected_delta_f: float = FRANCO_CONSTANT,
    tolerance: float = 0.1
) -> bool:
    """
    Validate that a field exhibits expected fractal properties
    
    Parameters:
    -----------
    field : np.ndarray
        Field to validate
    expected_delta_f : float
        Expected δF value
    tolerance : float
        Tolerance for validation
        
    Returns:
    --------
    bool
        True if field passes validation
    """
    operator = FractalOperator(expected_delta_f)
    analysis = operator.analyze_fractal_dimension(field)
    
    if 'match_quality' in analysis:
        return analysis['match_quality'] < tolerance
    else:
        return False

def quick_fractal_analysis(field: np.ndarray) -> Dict[str, float]:
    """Quick analysis of fractal properties"""
    if field.ndim == 2:
        dim, _, _ = box_counting_dimension(field)
        return {
            'fractal_dimension': dim,
            'delta_f_estimate': 3.0 - dim,
            'deviation_from_mfsu': abs((3.0 - dim) - FRANCO_CONSTANT)
        }
    else:
        return {'error': 'Only 2D fields supported for quick analysis'}

# Create convenience instances
default_fractal_operator = FractalOperator()
default_stochastic_operator = StochasticOperator()
