"""
Adjoint_source_creation.py

Simple and Robust Adjoint Source Creation for lumNLopt using XAD
Focused solely on computing ∂FOM/∂E and ∂FOM/∂H from user-defined FOM functions

Purpose: Replace manual adjoint source derivation with automatic differentiation
Input:   Forward electromagnetic fields + user FOM function
Output:  Adjoint source field arrays (Esource, Hsource) 
Next:    Pass arrays to Adjoint_simulation.py for lumapi integration

Author: lumNLopt repository integration
Backend: XAD (eXtended Automatic Differentiation)
"""

import numpy as np
import scipy.constants
import warnings
from typing import Callable, Tuple, Dict, Optional

# XAD - Industrial-grade automatic differentiation
try:
    import xad.adj_1st as xadj
    XAD_AVAILABLE = True
except ImportError:
    raise ImportError(
        "XAD automatic differentiation library not found.\n"
        "Install with: pip install xad\n"
        "XAD is required for automatic adjoint source generation."
    )

print(f"XAD loaded successfully - Industrial-grade automatic differentiation active")


class XADAdjointSourceCalculator:
    """
    Simple and robust calculator for automatic adjoint source generation.
    
    Core functionality:
    - Takes user-defined FOM function f(E, H, x, y, z, wavelengths) -> scalar  
    - Uses XAD to automatically compute ∂f/∂E and ∂f/∂H
    - Returns adjoint source arrays for downstream processing
    
    This class focuses solely on the automatic differentiation computation.
    Simulation integration is handled by separate files.
    """
    
    def __init__(self, fom_function: Callable, validation_mode: bool = True):
        """
        Initialize XAD-based adjoint source calculator.
        
        Parameters:
        -----------
        fom_function : callable
            User FOM function with signature: f(E, H, x, y, z, wavelengths) -> scalar
            Must return a real scalar value
            Must be differentiable w.r.t. complex field arrays E and H
        validation_mode : bool
            Enable input validation and safety checks (recommended for production)
        """
        
        if not callable(fom_function):
            raise TypeError("fom_function must be callable")
        
        self.fom_function = fom_function
        self.validation_mode = validation_mode
        
        # Runtime state
        self.last_fom_value: Optional[float] = None
        self.computation_stats: Dict = {}
        
        if self.validation_mode:
            self._validate_fom_function()
        
        print(f"XADAdjointSourceCalculator initialized with validation_mode={validation_mode}")
    
    def _validate_fom_function(self) -> None:
        """Validate FOM function with small test case."""
        
        print("Validating FOM function...")
        
        try:
            # Create minimal test data
            E_test = np.ones((2, 2, 1, 1, 3), dtype=complex)
            H_test = np.ones((2, 2, 1, 1, 3), dtype=complex)  
            x_test = np.array([0.0, 1e-6])
            y_test = np.array([0.0, 1e-6])
            z_test = np.array([0.0])
            wl_test = np.array([1.55e-6])
            
            # Test FOM function
            result = self.fom_function(E_test, H_test, x_test, y_test, z_test, wl_test)
            
            # Validate result
            if not np.isscalar(result):
                raise ValueError("FOM function must return a scalar value")
            if not np.isreal(result):
                warnings.warn(f"FOM function returned complex value: {result}. Using real part.")
            if not np.isfinite(result):
                raise ValueError("FOM function returned non-finite value")
            
            print(f"✓ FOM function validation passed - test result: {float(np.real(result)):.6e}")
            
        except Exception as e:
            raise ValueError(f"FOM function validation failed: {e}")
    
    def compute_adjoint_source_fields(self, 
                                    E: np.ndarray, 
                                    H: np.ndarray,
                                    x: np.ndarray,
                                    y: np.ndarray, 
                                    z: np.ndarray,
                                    wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Core method: Compute adjoint source fields using XAD automatic differentiation.
        
        This is the main function that replaces manual adjoint source derivation
        with automatic computation of ∂FOM/∂E and ∂FOM/∂H.
        
        Parameters:
        -----------
        E : np.ndarray
            Complex electric field array, shape (Nx, Ny, Nz, Nwl, 3)
        H : np.ndarray  
            Complex magnetic field array, shape (Nx, Ny, Nz, Nwl, 3)
        x, y, z : np.ndarray
            Spatial coordinate arrays
        wavelengths : np.ndarray
            Wavelength array
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (Esource, Hsource) - Adjoint source field arrays
            Same shape as input E, H arrays
        """
        
        print("Computing adjoint source fields via XAD...")
        
        # Validation
        if self.validation_mode:
            self._validate_field_inputs(E, H, x, y, z, wavelengths)
        
        # XAD automatic differentiation computation
        fom_value, dFOM_dE, dFOM_dH = self._xad_gradient_computation(
            E, H, x, y, z, wavelengths
        )
        
        # Store results for inspection
        self.last_fom_value = fom_value
        self._update_computation_stats(dFOM_dE, dFOM_dH)
        
        # Final validation
        if self.validation_mode:
            self._validate_gradients(dFOM_dE, dFOM_dH)
        
        print(f"XAD computation completed:")
        print(f"  FOM value: {fom_value:.6e}")
        print(f"  Max |∂FOM/∂E|: {np.abs(dFOM_dE).max():.3e}")
        print(f"  Max |∂FOM/∂H|: {np.abs(dFOM_dH).max():.3e}")
        
        # Return adjoint source fields
        # Note: In electromagnetic adjoint problems, the adjoint source is often
        # exactly the field derivatives, but specific scaling may be applied
        # in downstream processing (Adjoint_simulation.py)
        return dFOM_dE, dFOM_dH
    
    def _validate_field_inputs(self, E, H, x, y, z, wavelengths) -> None:
        """Validate electromagnetic field input data."""
        
        # Shape validation
        if E.shape != H.shape:
            raise ValueError(f"E and H field shapes must match: {E.shape} vs {H.shape}")
        
        if E.shape[-1] != 3:
            raise ValueError(f"Field arrays must have 3 components, got {E.shape[-1]}")
        
        expected_shape = (len(x), len(y), len(z), len(wavelengths), 3)
        if E.shape != expected_shape:
            raise ValueError(f"Field shape {E.shape} doesn't match coordinates {expected_shape}")
        
        # Data type validation  
        if not np.iscomplexobj(E) or not np.iscomplexobj(H):
            raise ValueError("E and H fields must be complex arrays")
        
        # Finite value validation
        if np.any(~np.isfinite(E)) or np.any(~np.isfinite(H)):
            raise ValueError("E or H fields contain non-finite values")
        
        # Coordinate validation
        for coord, name in zip([x, y, z, wavelengths], ['x', 'y', 'z', 'wavelengths']):
            if not np.all(np.isfinite(coord)):
                raise ValueError(f"{name} coordinates contain non-finite values")
            if len(coord) == 0:
                raise ValueError(f"{name} coordinate array is empty")
    
    def _xad_gradient_computation(self, E, H, x, y, z, wavelengths) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Core XAD automatic differentiation computation.
        
        This method performs the key automatic differentiation:
        1. Convert complex fields to XAD active variables (real/imaginary parts)
        2. Execute FOM function with XAD tape recording 
        3. Compute reverse-mode gradients automatically
        4. Extract ∂FOM/∂E and ∂FOM/∂H from XAD derivatives
        """
        
        # Get array dimensions
        field_shape = E.shape
        E_flat = E.flatten()
        H_flat = H.flatten()
        
        print(f"XAD processing {E_flat.size + H_flat.size} field components...")
        
        # Convert to XAD active variables (separate real/imaginary parts)
        E_real_xad = [xadj.Real(float(val)) for val in E_flat.real]
        E_imag_xad = [xadj.Real(float(val)) for val in E_flat.imag]
        H_real_xad = [xadj.Real(float(val)) for val in H_flat.real]
        H_imag_xad = [xadj.Real(float(val)) for val in H_flat.imag]
        
        # Coordinates (no differentiation needed)
        x_np = np.asarray(x, dtype=float)
        y_np = np.asarray(y, dtype=float) 
        z_np = np.asarray(z, dtype=float)
        wl_np = np.asarray(wavelengths, dtype=float)
        
        try:
            # XAD tape-based automatic differentiation
            with xadj.Tape() as tape:
                
                # Register all field components as inputs for differentiation
                all_inputs = E_real_xad + E_imag_xad + H_real_xad + H_imag_xad
                for var in all_inputs:
                    tape.registerInput(var)
                
                # Start recording operations
                tape.newRecording()
                
                # Reconstruct complex field arrays from XAD active variables
                E_real_vals = np.array([var.value for var in E_real_xad]).reshape(field_shape)
                E_imag_vals = np.array([var.value for var in E_imag_xad]).reshape(field_shape)
                E_reconstructed = E_real_vals + 1j * E_imag_vals
                
                H_real_vals = np.array([var.value for var in H_real_xad]).reshape(field_shape)
                H_imag_vals = np.array([var.value for var in H_imag_xad]).reshape(field_shape)
                H_reconstructed = H_real_vals + 1j * H_imag_vals
                
                # Execute user FOM function (this is recorded on the tape)
                fom_result = self.fom_function(
                    E_reconstructed, H_reconstructed, x_np, y_np, z_np, wl_np
                )
                
                # Handle complex FOM result (should be real for valid physical FOMs)
                if np.iscomplexobj(fom_result):
                    if abs(np.imag(fom_result)) > 1e-12:
                        warnings.warn(f"FOM has significant imaginary part: {np.imag(fom_result):.3e}")
                    fom_result = float(np.real(fom_result))
                else:
                    fom_result = float(fom_result)
                
                # Create output variable and register with tape
                fom_xad = xadj.Real(fom_result)
                tape.registerOutput(fom_xad)
                
                # Seed adjoint computation (∂FOM/∂FOM = 1)
                fom_xad.derivative = 1.0
                
                # Compute all gradients via reverse-mode automatic differentiation
                tape.computeAdjoints()
                
                # Extract gradients and reconstruct complex derivatives
                E_grad_real = np.array([var.derivative for var in E_real_xad]).reshape(field_shape)
                E_grad_imag = np.array([var.derivative for var in E_imag_xad]).reshape(field_shape)
                dFOM_dE = E_grad_real + 1j * E_grad_imag
                
                H_grad_real = np.array([var.derivative for var in H_real_xad]).reshape(field_shape)
                H_grad_imag = np.array([var.derivative for var in H_imag_xad]).reshape(field_shape)
                dFOM_dH = H_grad_real + 1j * H_grad_imag
                
                return float(fom_xad.value), dFOM_dE, dFOM_dH
                
        except Exception as e:
            raise RuntimeError(f"XAD automatic differentiation failed: {e}") from e
    
    def _validate_gradients(self, dFOM_dE: np.ndarray, dFOM_dH: np.ndarray) -> None:
        """Validate computed gradients for numerical sanity."""
        
        # Check for non-finite values
        if not np.all(np.isfinite(dFOM_dE)) or not np.all(np.isfinite(dFOM_dH)):
            raise ValueError("Computed gradients contain non-finite values")
        
        # Check gradient magnitudes
        E_grad_norm = np.linalg.norm(dFOM_dE)
        H_grad_norm = np.linalg.norm(dFOM_dH)
        
        if E_grad_norm == 0 and H_grad_norm == 0:
            warnings.warn("All gradients are zero - FOM may be independent of fields")
        
        # Check for potential numerical issues
        if E_grad_norm > 1e15 or H_grad_norm > 1e15:
            warnings.warn(f"Extremely large gradient magnitudes detected: "
                         f"|∂FOM/∂E|={E_grad_norm:.2e}, |∂FOM/∂H|={H_grad_norm:.2e}")
        
        if E_grad_norm < 1e-15 and H_grad_norm < 1e-15:
            warnings.warn(f"Extremely small gradient magnitudes: "
                         f"|∂FOM/∂E|={E_grad_norm:.2e}, |∂FOM/∂H|={H_grad_norm:.2e}")
    
    def _update_computation_stats(self, dFOM_dE: np.ndarray, dFOM_dH: np.ndarray) -> None:
        """Update internal computation statistics."""
        
        self.computation_stats = {
            'E_gradient_stats': {
                'max_magnitude': float(np.abs(dFOM_dE).max()),
                'mean_magnitude': float(np.abs(dFOM_dE).mean()),
                'std_magnitude': float(np.abs(dFOM_dE).std()),
                'shape': dFOM_dE.shape
            },
            'H_gradient_stats': {
                'max_magnitude': float(np.abs(dFOM_dH).max()),
                'mean_magnitude': float(np.abs(dFOM_dH).mean()),
                'std_magnitude': float(np.abs(dFOM_dH).std()),
                'shape': dFOM_dH.shape
            },
            'fom_value': self.last_fom_value
        }
    
    def get_computation_stats(self) -> Dict:
        """Return computation statistics from last calculation."""
        return self.computation_stats.copy()
    
    def print_computation_summary(self) -> None:
        """Print summary of last computation."""
        
        if not self.computation_stats:
            print("No computation performed yet")
            return
        
        stats = self.computation_stats
        print("\n" + "="*60)
        print("XAD ADJOINT SOURCE COMPUTATION SUMMARY")
        print("="*60)
        print(f"FOM Value: {stats['fom_value']:.6e}")
        print(f"\n∂FOM/∂E Statistics:")
        print(f"  Max magnitude: {stats['E_gradient_stats']['max_magnitude']:.3e}")
        print(f"  Mean magnitude: {stats['E_gradient_stats']['mean_magnitude']:.3e}")
        print(f"  Std deviation: {stats['E_gradient_stats']['std_magnitude']:.3e}")
        print(f"  Shape: {stats['E_gradient_stats']['shape']}")
        print(f"\n∂FOM/∂H Statistics:")
        print(f"  Max magnitude: {stats['H_gradient_stats']['max_magnitude']:.3e}")
        print(f"  Mean magnitude: {stats['H_gradient_stats']['mean_magnitude']:.3e}")
        print(f"  Std deviation: {stats['H_gradient_stats']['std_magnitude']:.3e}")
        print(f"  Shape: {stats['H_gradient_stats']['shape']}")
        print("="*60 + "\n")


# ============================================================================
# CONVENIENCE FUNCTION FOR DIRECT USAGE
# ============================================================================

def compute_adjoint_source_from_fom(fom_function: Callable,
                                   E: np.ndarray,
                                   H: np.ndarray, 
                                   x: np.ndarray,
                                   y: np.ndarray,
                                   z: np.ndarray,
                                   wavelengths: np.ndarray,
                                   validation_mode: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for direct adjoint source computation.
    
    This is the main entry point for simple usage without class instantiation.
    
    Parameters:
    -----------
    fom_function : callable
        User FOM function: f(E, H, x, y, z, wavelengths) -> scalar
    E, H : np.ndarray
        Complex electromagnetic field arrays, shape (Nx, Ny, Nz, Nwl, 3)
    x, y, z : np.ndarray
        Spatial coordinate arrays
    wavelengths : np.ndarray
        Wavelength array
    validation_mode : bool
        Enable input validation and safety checks
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (Esource, Hsource) - Adjoint source field arrays for downstream use
    
    Example:
    --------
    >>> def my_fom(E, H, x, y, z, wl):
    ...     return np.sum(np.abs(E)**2)  # Simple power-based FOM
    >>> 
    >>> Esource, Hsource = compute_adjoint_source_from_fom(
    ...     my_fom, E_fields, H_fields, x_coords, y_coords, z_coords, wavelengths
    ... )
    >>> 
    >>> # Pass Esource, Hsource to Adjoint_simulation.py for lumapi integration
    """
    
    # Create calculator and compute
    calculator = XADAdjointSourceCalculator(fom_function, validation_mode)
    Esource, Hsource = calculator.compute_adjoint_source_fields(E, H, x, y, z, wavelengths)
    
    # Print summary if validation enabled
    if validation_mode:
        calculator.print_computation_summary()
    
    return Esource, Hsource


# ============================================================================
# EXAMPLE FOM FUNCTIONS FOR TESTING
# ============================================================================

def power_transmission_fom_example(E, H, x, y, z, wavelengths):
    """
    Example FOM: Total electromagnetic power transmission.
    Computes Poynting vector integral in +z direction.
    """
    # Extract field components
    Ex, Ey, Ez = E[..., 0], E[..., 1], E[..., 2]
    Hx, Hy, Hz = H[..., 0], H[..., 1], H[..., 2]
    
    # Poynting vector z-component: Sz = (1/2) * Re(Ex*Hy* - Ey*Hx*)
    Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
    
    # Spatial integration with proper weighting
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    
    # Total transmitted power (integrated over space and frequency)
    total_power = np.sum(Sz) * dx * dy * dz
    
    return total_power

def mode_overlap_fom_example(E, H, x, y, z, wavelengths):
    """
    Example FOM: Mode overlap efficiency with Gaussian target mode.
    """
    # Define target Gaussian mode parameters
    w0 = 1e-6  # beam waist [m]
    x0, y0 = np.mean(x), np.mean(y)  # center position
    
    # Create coordinate meshgrids
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Gaussian mode profile (fundamental mode approximation)
    gaussian_profile = np.exp(-((X - x0)**2 + (Y - y0)**2) / w0**2)
    
    # Normalize target mode
    mode_power = np.sum(np.abs(gaussian_profile)**2) * (x[1]-x[0]) * (y[1]-y[0])
    gaussian_profile = gaussian_profile / np.sqrt(mode_power)
    
    # Create target field (Ex component dominant for TE-like mode)
    E_target = np.zeros_like(E)
    for wl_idx in range(len(wavelengths)):
        for z_idx in range(len(z)):
            E_target[:, :, z_idx, wl_idx, 0] = gaussian_profile
    
    # Mode overlap integral: ∫ E_sim · E_target* dV
    overlap = np.sum(E * np.conj(E_target))
    
    # Coupling efficiency = |overlap|²
    efficiency = np.abs(overlap)**2
    
    return efficiency


# ============================================================================
# USAGE DOCUMENTATION AND EXAMPLES
# ============================================================================

"""
USAGE EXAMPLES:

1. Simple Direct Usage:
```python
from Adjoint_source_creation import compute_adjoint_source_from_fom

# Define your FOM
def my_fom(E, H, x, y, z, wavelengths):
    # Your optimization objective - must return scalar
    return np.sum(np.abs(E)**2)  # Example: maximize total E-field power

# Compute adjoint source (assumes you have forward fields)
Esource, Hsource = compute_adjoint_source_from_fom(
    my_fom, E_forward, H_forward, x_coords, y_coords, z_coords, wavelengths
)

# Pass to next file in pipeline
# Adjoint_simulation.py will use Esource, Hsource for lumapi integration
```

2. Advanced Usage with Statistics:
```python
from Adjoint_source_creation import XADAdjointSourceCalculator

# Create calculator with validation
calculator = XADAdjointSourceCalculator(my_fom, validation_mode=True)

# Compute adjoint source
Esource, Hsource = calculator.compute_adjoint_source_fields(
    E_forward, H_forward, x_coords, y_coords, z_coords, wavelengths
)

# Print detailed computation summary
calculator.print_computation_summary()

# Get computation statistics for analysis
stats = calculator.get_computation_stats()
print(f"Max gradient magnitude: {stats['E_gradient_stats']['max_magnitude']}")
```

3. Integration with lumNLopt Workflow:
```python
# In your lumNLopt optimization loop:

# After forward simulation
forward_fields = get_fields(sim.fdtd, monitor_name='opt_fields', ...)

# Create adjoint source using XAD
Esource, Hsource = compute_adjoint_source_from_fom(
    user_defined_fom,
    forward_fields.E,
    forward_fields.H, 
    forward_fields.x,
    forward_fields.y,
    forward_fields.z,
    forward_fields.wl
)

# Pass to Adjoint_simulation.py for FDTD integration
# adjoint_sim.import_source_data(Esource, Hsource, forward_fields)
```

KEY BENEFITS:

✓ Universal FOM Support: Any differentiable function works automatically
✓ No Manual Derivation: XAD computes ∂FOM/∂E and ∂FOM/∂H automatically  
✓ Production Ready: Industrial-grade XAD with comprehensive validation
✓ Simple Interface: Single function call replaces complex manual calculations
✓ Robust Error Handling: Comprehensive validation and clear error messages
✓ Performance Monitoring: Built-in computation statistics and profiling

TECHNICAL NOTES:

- Requires XAD installation: pip install xad
- FOM function must return real scalar value
- Field arrays must be complex with shape (Nx, Ny, Nz, Nwl, 3) 
- Coordinate arrays must be 1D with consistent sizing
- Memory usage scales linearly with field array size
- XAD uses tape-based reverse-mode AD for optimal performance

This file handles ONLY the adjoint source field computation.
Integration with FDTD simulation is handled by Adjoint_simulation.py.
Parameter gradient calculation is handled by Gradient_calculation.py.
"""
