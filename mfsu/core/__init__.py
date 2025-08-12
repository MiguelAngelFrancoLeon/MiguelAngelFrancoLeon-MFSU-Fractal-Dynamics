"""
MFSU Core Module
===============

The fundamental mathematical and physical foundations of the 
Unified Fractal-Stochastic Model (MFSU).

This module contains:
- Universal constants (Franco Constant δF ≈ 0.921)
- Core MFSU equations and mathematical framework
- Fractal and stochastic operators
- Validation and consistency checks

Author: Miguel Ángel Franco León
"""

# Import all core components
from .constants import *
from .equations import *
from .operators import *

# Core validation function
def validate_core():
    """Validate that all core components are consistent"""
    from .constants import FRANCO_CONSTANT, FRACTAL_DIMENSION
    from .equations import mfsu_equation
    from .operators import FractalOperator
    
    # Test constant relationships
    expected_df = 3.0 - FRANCO_CONSTANT
    assert abs(FRACTAL_DIMENSION - expected_df) < 1e-6, \
        f"Inconsistent fractal dimension: {FRACTAL_DIMENSION} vs {expected_df}"
    
    # Test equation functionality
    import numpy as np
    test_field = np.random.rand(5, 5)
    result = mfsu_equation(test_field, t=0.1)
    assert result is not None, "MFSU equation failed basic test"
    
    # Test operators
    op = FractalOperator(delta_f=FRANCO_CONSTANT)
    assert op.delta_f == FRANCO_CONSTANT, "FractalOperator initialization failed"
    
    return True

# Expose key components at core level
__all__ = [
    # From constants
    'FRANCO_CONSTANT', 'DELTA_F', 'FRACTAL_DIMENSION', 'HURST_EXPONENT',
    'PLANCK_SCALE_FRACTAL', 'COSMIC_SCALE_FACTOR', 'get_universal_constants',
    
    # From equations  
    'mfsu_equation', 'gauss_fractal_law', 'fractal_power_spectrum',
    'MFSUEquation', 'validate_mfsu_solution',
    
    # From operators
    'fractional_laplacian', 'fractal_gradient', 'hurst_noise_generator',
    'FractalOperator', 'StochasticOperator',
    
    # Validation
    'validate_core'
]
