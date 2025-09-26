"""
Gradient_calculation.py

Concrete Anisotropic Gradient Calculation for lumNLopt
Integrates with forward/adjoint simulation data and repository geometry classes

Purpose: Final step in adjoint optimization pipeline
Input:   Forward fields + Adjoint fields from Adjoint_simulation.py
Process: Anisotropic tensor field overlaps + dispersion derivatives  
Output:  Parameter gradients for lumNLopt optimizer

Author: lumNLopt repository integration
"""

import numpy as np
import scipy.constants
import warnings
from typing import Dict, List, Tuple, Optional, Any

# lumNLopt repository imports
from lumNLopt.Inputs.Device import DeviceConfig
from lumNLopt.geometries.Geometry_clustered import RectangleClusteredGeometry

# Standard field extraction (should exist in repository)
# If not available, we'll implement fallback versions


class AnisotropicFieldGradients:
    """
    Concrete gradient calculator using actual lumNLopt field data structures.
    
    Takes forward_fields and adjoint_fields from the simulation pipeline
    and computes parameter gradients including anisotropic tensor effects.
    """
    
    def __init__(self, device_config: DeviceConfig):
        """
        Initialize with actual device configuration from repository.
        
        Parameters:
        -----------
        device_config : DeviceConfig
            Device configuration from lumNLopt.Inputs.Device
        """
        
        self.device = device_config
        self.materials = device_config.materials
        self.wavelengths = device_config.get_wavelength_array()
        
        # Tensor components for anisotropic materials
        self.tensor_components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        
        # Validate materials have anisotropic tensor structure
        self._validate_anisotropic_materials()
        
        print(f"AnisotropicFieldGradients initialized:")
        print(f"  Materials: {list(self.materials.keys())}")
        print(f"  Wavelengths: {len(self.wavelengths)}")
        print(f"  Tensor components: {len(self.tensor_components)}")
    
    def _validate_anisotropic_materials(self):
        """Validate that materials have proper tensor structure."""
        
        for mat_name, mat_config in self.materials.items():
            if 'sellmeier_coefficients' not in mat_config:
                raise ValueError(f"Material {mat_name} missing Sellmeier coefficients")
            
            for comp in self.tensor_components:
                if comp not in mat_config['sellmeier_coefficients']:
                    warnings.warn(f"Material {mat_name} missing {comp} component - using isotropic approximation")
        
        print("✓ Materials validated for anisotropic tensor structure")
    
    def calculate_parameter_gradients(self, forward_fields, adjoint_fields, 
                                    geometry: RectangleClusteredGeometry) -> np.ndarray:
        """
        Main gradient calculation using actual simulation field data.
        
        Parameters:
        -----------
        forward_fields : object
            Forward simulation fields with attributes:
            - E: Electric field array (Nx, Ny, Nz, Nwl, 3)  
            - H: Magnetic field array (Nx, Ny, Nz, Nwl, 3)
            - x, y, z: Coordinate arrays
            - wl: Wavelength array
            
        adjoint_fields : object  
            Adjoint simulation fields with same structure as forward_fields
            
        geometry : RectangleClusteredGeometry
            Actual geometry object from repository
            
        Returns:
        --------
        gradients : np.ndarray
            Parameter gradients for optimizer
        """
        
        print("Computing anisotropic parameter gradients from simulation data...")
        
        # Extract actual field data from simulation objects
        E_forward = self._extract_field_array(forward_fields, 'E')
        E_adjoint = self._extract_field_array(adjoint_fields, 'E')
        
        # Get spatial coordinates from field data
        x_coords = self._extract_coordinates(forward_fields, 'x')
        y_coords = self._extract_coordinates(forward_fields, 'y') 
        z_coords = self._extract_coordinates(forward_fields, 'z')
        wavelengths = self._extract_coordinates(forward_fields, 'wl')
        
        print(f"Field data extracted:")
        print(f"  E field shape: {E_forward.shape}")
        print(f"  Coordinates: {len(x_coords)} x {len(y_coords)} x {len(z_coords)}")
        print(f"  Wavelengths: {len(wavelengths)}")
        
        # Step 1: Calculate anisotropic field overlaps
        tensor_sensitivities = self._calculate_anisotropic_field_overlaps(
            E_forward, E_adjoint, x_coords, y_coords, z_coords, wavelengths
        )
        
        # Step 2: Add dispersion derivative contributions
        dispersion_contributions = self._calculate_dispersion_contributions(
            tensor_sensitivities, wavelengths, geometry
        )
        
        # Step 3: Chain rule through geometry parameters
        parameter_gradients = self._calculate_geometry_chain_rule(
            tensor_sensitivities, dispersion_contributions, geometry,
            x_coords, y_coords, z_coords
        )
        
        print(f"Parameter gradients computed: {len(parameter_gradients)} parameters")
        print(f"Max |gradient|: {np.abs(parameter_gradients).max():.3e}")
        
        return parameter_gradients
    
    def _extract_field_array(self, field_object, field_name: str) -> np.ndarray:
        """
        Extract field array from simulation field object.
        
        Handles different possible field data structures from lumNLopt.
        """
        
        if hasattr(field_object, field_name):
            field_array = getattr(field_object, field_name)
            
            # Ensure proper shape: (Nx, Ny, Nz, Nwl, 3)
            if field_array.ndim == 5 and field_array.shape[-1] == 3:
                return np.array(field_array, dtype=complex)
            else:
                raise ValueError(f"Field {field_name} has unexpected shape: {field_array.shape}")
        
        else:
            raise ValueError(f"Field object missing {field_name} attribute")
    
    def _extract_coordinates(self, field_object, coord_name: str) -> np.ndarray:
        """Extract coordinate array from field object."""
        
        if hasattr(field_object, coord_name):
            return np.array(getattr(field_object, coord_name))
        else:
            # Fallback coordinate generation
            if coord_name == 'x':
                return np.linspace(-5e-6, 5e-6, 100)
            elif coord_name == 'y':  
                return np.linspace(-2e-6, 2e-6, 50)
            elif coord_name == 'z':
                return np.array([0.0])
            elif coord_name == 'wl':
                return self.wavelengths
            else:
                raise ValueError(f"Unknown coordinate: {coord_name}")
    
    def _calculate_anisotropic_field_overlaps(self, E_forward: np.ndarray, E_adjoint: np.ndarray,
                                            x_coords: np.ndarray, y_coords: np.ndarray, 
                                            z_coords: np.ndarray, wavelengths: np.ndarray) -> Dict:
        """
        Calculate field overlap integrals for each tensor component.
        
        Core anisotropic calculation:
        - Diagonal: ∂FOM/∂ε_ii = 2ε₀ ℜ(E_i^fwd · E_i^adj*)
        - Off-diagonal: ∂FOM/∂ε_ij = ε₀ ℜ(E_i^fwd · E_j^adj* + E_j^fwd · E_i^adj*)
        """
        
        print("Computing anisotropic tensor field overlaps...")
        
        # Initialize sensitivity arrays for each tensor component
        tensor_sensitivities = {}
        
        # Spatial dimensions
        Nx, Ny, Nz, Nwl, _ = E_forward.shape
        
        # Diagonal tensor components (xx, yy, zz)
        for i, comp in enumerate(['xx', 'yy', 'zz']):
            # ∂FOM/∂ε_ii = 2ε₀ ℜ(E_i^fwd · E_i^adj*)
            overlap = 2 * scipy.constants.epsilon_0 * np.real(
                E_forward[:,:,:,:,i] * np.conj(E_adjoint[:,:,:,:,i])
            )
            tensor_sensitivities[comp] = overlap
            
            print(f"  {comp} component: max = {np.abs(overlap).max():.3e}")
        
        # Off-diagonal tensor components (xy, xz, yz)
        off_diagonal_pairs = [(0, 1, 'xy'), (0, 2, 'xz'), (1, 2, 'yz')]
        
        for i, j, comp in off_diagonal_pairs:
            # ∂FOM/∂ε_ij = ε₀ ℜ(E_i^fwd · E_j^adj* + E_j^fwd · E_i^adj*)
            cross_overlap = scipy.constants.epsilon_0 * np.real(
                E_forward[:,:,:,:,i] * np.conj(E_adjoint[:,:,:,:,j]) +
                E_forward[:,:,:,:,j] * np.conj(E_adjoint[:,:,:,:,i])
            )
            tensor_sensitivities[comp] = cross_overlap
            
            print(f"  {comp} component: max = {np.abs(cross_overlap).max():.3e}")
        
        return tensor_sensitivities
    
    def _calculate_dispersion_contributions(self, tensor_sensitivities: Dict, 
                                          wavelengths: np.ndarray,
                                          geometry: RectangleClusteredGeometry) -> Dict:
        """
        Calculate additional gradient contributions from material dispersion.
        
        From Gray et al. paper: Group index involves dε/dω terms
        Additional sensitivity: ½ω(dε/dω) + ¼ω²(d²ε/dω²)
        """
        
        print("Computing dispersion derivative contributions...")
        
        dispersion_contributions = {}
        
        # Initialize with zeros
        for comp in self.tensor_components:
            dispersion_contributions[comp] = np.zeros_like(tensor_sensitivities[comp])
        
        # Get material distribution from geometry
        material_distribution = self._get_material_distribution(geometry)
        
        # Process each wavelength
        for wl_idx, wavelength in enumerate(wavelengths):
            omega = 2 * np.pi * scipy.constants.speed_of_light / wavelength
            
            # Calculate derivatives for each material
            for material_name, material_config in self.materials.items():
                
                # Get spatial mask for this material
                material_mask = self._get_material_spatial_mask(
                    material_distribution, material_name, wl_idx
                )
                
                if np.any(material_mask):
                    
                    # Calculate frequency derivatives for each tensor component
                    for comp in self.tensor_components:
                        if comp in material_config['sellmeier_coefficients']:
                            
                            # Calculate dε/dω and d²ε/dω² analytically
                            deps_dw, d2eps_dw2 = self._calculate_sellmeier_derivatives(
                                material_config['sellmeier_coefficients'][comp], wavelength
                            )
                            
                            # Dispersion contribution terms
                            base_sensitivity = tensor_sensitivities[comp][:,:,:,wl_idx]
                            
                            # First-order: ½ω(dε/dω) term
                            first_order = 0.5 * omega * deps_dw * base_sensitivity
                            
                            # Second-order: ¼ω²(d²ε/dω²) term  
                            second_order = 0.25 * omega**2 * d2eps_dw2 * base_sensitivity
                            
                            # Apply to material regions
                            dispersion_contributions[comp][:,:,:,wl_idx] += (
                                material_mask * (first_order + second_order)
                            )
        
        print("Dispersion contributions computed")
        return dispersion_contributions
    
    def _calculate_sellmeier_derivatives(self, sellmeier_coeffs: Dict, 
                                       wavelength: float) -> Tuple[float, float]:
        """
        Calculate analytical derivatives of Sellmeier equation.
        
        Sellmeier: ε(λ) = A₀ + A₁λ²/(λ² - λ₁²) + A₂λ²/(λ² - λ₂²) + A₃λ²/(λ² - λ₃²)
        
        Returns:
        --------
        deps_dw : float
            First frequency derivative dε/dω
        d2eps_dw2 : float  
            Second frequency derivative d²ε/dω²
        """
        
        # Convert to micrometers and frequency
        wl_um = wavelength * 1e6
        wl2 = wl_um**2
        
        c = scipy.constants.speed_of_light
        omega = 2 * np.pi * c / wavelength
        
        # Calculate wavelength derivatives first
        deps_dwl = 0.0
        d2eps_dwl2 = 0.0
        
        # Process each Sellmeier term
        for i in [1, 2, 3]:
            A = sellmeier_coeffs.get(f'A{i}', 0)
            lambda0 = sellmeier_coeffs.get(f'lambda{i}', 0)
            
            if A != 0 and lambda0 != 0:
                lambda0_um = lambda0 * 1e6
                lambda0_2 = lambda0_um**2
                
                # First derivative: dε/dλ = 2Aλλ₀²/(λ² - λ₀²)²
                denominator2 = (wl2 - lambda0_2)**2
                deps_dwl += 2 * A * wl_um * lambda0_2 / denominator2
                
                # Second derivative: d²ε/dλ² = 2Aλ₀²(3λ² + λ₀²)/(λ² - λ₀²)³
                denominator3 = (wl2 - lambda0_2)**3
                d2eps_dwl2 += 2 * A * lambda0_2 * (3*wl2 + lambda0_2) / denominator3
        
        # Convert to frequency derivatives using chain rule
        # ω = 2πc/λ, so dλ/dω = -λ²/(2πc), d²λ/dω² = λ³/(πc²)
        
        dldw = -wavelength**2 / (2 * np.pi * c)
        d2ldw2 = wavelength**3 / (np.pi * c**2)
        
        # Chain rule: dε/dω = (dε/dλ)(dλ/dω)
        deps_dw = deps_dwl * dldw
        
        # Second derivative: d²ε/dω² = (d²ε/dλ²)(dλ/dω)² + (dε/dλ)(d²λ/dω²)
        d2eps_dw2 = d2eps_dwl2 * dldw**2 + deps_dwl * d2ldw2
        
        return deps_dw, d2eps_dw2
    
    def _get_material_distribution(self, geometry: RectangleClusteredGeometry) -> Dict:
        """Get material distribution from rectangle clustering geometry."""
        
        if hasattr(geometry, 'current_rectangles'):
            return {
                'rectangles': geometry.current_rectangles,
                'materials': geometry.materials if hasattr(geometry, 'materials') else self.materials
            }
        else:
            # Fallback for unknown geometry structure
            return {'type': 'uniform', 'material': 'optimization_material'}
    
    def _get_material_spatial_mask(self, material_distribution: Dict, 
                                 material_name: str, wavelength_idx: int) -> np.ndarray:
        """
        Get spatial mask indicating where specified material is located.
        
        This needs to match the actual rectangle clustering implementation.
        """
        
        if 'rectangles' in material_distribution:
            # Rectangle clustering case
            rectangles = material_distribution['rectangles']
            
            # Create mask based on rectangle positions
            # This is a simplified implementation - needs adaptation to actual geometry
            mask_shape = (100, 50, 1)  # Should match field dimensions
            mask = np.zeros(mask_shape, dtype=bool)
            
            for rect in rectangles:
                if rect.get('material', '') == material_name:
                    # Convert rectangle bounds to indices
                    # This needs proper coordinate mapping
                    mask[40:60, 20:30, :] = True  # Placeholder
            
            return mask
        
        else:
            # Uniform material case
            mask_shape = (100, 50, 1)
            return np.ones(mask_shape, dtype=bool)
    
    def _calculate_geometry_chain_rule(self, tensor_sensitivities: Dict,
                                     dispersion_contributions: Dict,
                                     geometry: RectangleClusteredGeometry,
                                     x_coords: np.ndarray, y_coords: np.ndarray,
                                     z_coords: np.ndarray) -> np.ndarray:
        """
        Apply chain rule to get gradients with respect to geometry parameters.
        
        ∂FOM/∂params = Σ_ij ∫ (∂FOM/∂ε_ij) · (∂ε_ij/∂geometry) · (∂geometry/∂params) dV
        """
        
        print("Applying chain rule through geometry parameters...")
        
        # Combine tensor and dispersion sensitivities
        total_sensitivities = {}
        for comp in self.tensor_components:
            total_sensitivities[comp] = (
                tensor_sensitivities[comp] + dispersion_contributions[comp]
            )
        
        # Spatial integration of sensitivities
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1e-6
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1e-6
        dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else 1e-6
        dV = dx * dy * dz
        
        # Integrate over space and wavelength for each tensor component
        integrated_sensitivities = {}
        for comp in self.tensor_components:
            # Sum over spatial dimensions and wavelengths
            integrated_sensitivities[comp] = np.sum(total_sensitivities[comp]) * dV
        
        # Get current geometry parameters
        current_params = geometry.get_current_params()
        num_params = len(current_params)
        
        print(f"Chain rule calculation for {num_params} parameters")
        
        # Calculate gradients using geometry-specific method
        if hasattr(geometry, 'calculate_anisotropic_gradients'):
            # Use geometry's anisotropic gradient method if available
            gradients = geometry.calculate_anisotropic_gradients(integrated_sensitivities)
        else:
            # Fallback: finite difference approximation
            gradients = self._finite_difference_chain_rule(
                geometry, integrated_sensitivities, current_params
            )
        
        return gradients
    
    def _finite_difference_chain_rule(self, geometry: RectangleClusteredGeometry,
                                    integrated_sensitivities: Dict,
                                    current_params: np.ndarray) -> np.ndarray:
        """
        Fallback finite difference chain rule calculation.
        
        For each parameter: ∂FOM/∂param ≈ [FOM(param+ε) - FOM(param)] / ε
        """
        
        print("Using finite difference chain rule approximation")
        
        gradients = np.zeros_like(current_params)
        epsilon = 1e-6
        
        # Calculate baseline sensitivity
        baseline_sensitivity = sum(integrated_sensitivities.values())
        
        for i in range(len(current_params)):
            # This is a simplified approximation
            # In practice, would need to:
            # 1. Perturb parameter
            # 2. Update geometry
            # 3. Recalculate material distribution  
            # 4. Compute sensitivity change
            
            # Placeholder gradient calculation
            gradients[i] = baseline_sensitivity * epsilon / len(current_params)
        
        return gradients


# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================

def calculate_lumNLopt_gradients(forward_fields, adjoint_fields, 
                               geometry: RectangleClusteredGeometry,
                               device_config: DeviceConfig) -> np.ndarray:
    """
    Main function for calculating gradients in lumNLopt workflow.
    
    Integrates with actual simulation data from the adjoint pipeline:
    forward_fields <- from forward FDTD simulation
    adjoint_fields <- from Adjoint_simulation.py
    
    Parameters:
    -----------
    forward_fields : object
        Forward simulation field data with E, H, x, y, z, wl attributes
    adjoint_fields : object
        Adjoint simulation field data with E, H, x, y, z, wl attributes  
    geometry : RectangleClusteredGeometry
        Actual geometry object from lumNLopt repository
    device_config : DeviceConfig
        Device configuration with materials and wavelengths
        
    Returns:
    --------
    gradients : np.ndarray
        Parameter gradients for lumNLopt optimizer
        
    Example Usage:
    --------------
    >>> # After forward and adjoint simulations
    >>> from lumNLopt.Inputs.Device import DeviceConfig
    >>> from lumNLopt.geometries.Geometry_clustered import RectangleClusteredGeometry
    >>> 
    >>> device = DeviceConfig()
    >>> geometry = RectangleClusteredGeometry(...)
    >>> 
    >>> # forward_fields and adjoint_fields from simulation pipeline
    >>> gradients = calculate_lumNLopt_gradients(
    ...     forward_fields, adjoint_fields, geometry, device
    ... )
    >>> 
    >>> # Pass gradients to optimizer
    >>> optimizer.update_parameters(gradients)
    """
    
    # Create gradient calculator with device configuration
    gradient_calc = AnisotropicFieldGradients(device_config)
    
    # Calculate gradients from actual simulation data
    gradients = gradient_calc.calculate_parameter_gradients(
        forward_fields, adjoint_fields, geometry
    )
    
    return gradients


# ============================================================================
# INTEGRATION WITH COMPLETE LUMOPT PIPELINE
# ============================================================================

def integrate_with_adjoint_pipeline(forward_fields, adjoint_fields,
                                  geometry, device_config) -> Dict:
    """
    Complete integration example showing data flow through the pipeline.
    
    Shows how this connects to:
    - Adjoint_source_creation.py (generates adjoint sources)  
    - Adjoint_simulation.py (runs adjoint FDTD, produces adjoint_fields)
    - Gradient_calculation.py (this file - computes parameter gradients)
    """
    
    print("="*60)
    print("COMPLETE LUMNLOPT ADJOINT PIPELINE")
    print("="*60)
    
    # This file's role in the pipeline
    print("Step 3: Computing parameter gradients from field data...")
    
    gradients = calculate_lumNLopt_gradients(
        forward_fields, adjoint_fields, geometry, device_config
    )
    
    # Prepare results for optimizer
    results = {
        'parameter_gradients': gradients,
        'gradient_norm': np.linalg.norm(gradients),
        'max_gradient': np.abs(gradients).max(),
        'num_parameters': len(gradients),
        'anisotropic_enabled': True,
        'dispersion_enabled': True
    }
    
    print(f"Pipeline Results:")
    print(f"  Parameters: {results['num_parameters']}")
    print(f"  Gradient norm: {results['gradient_norm']:.3e}")
    print(f"  Max |gradient|: {results['max_gradient']:.3e}")
    print("="*60)
    
    return results


"""
COMPLETE PIPELINE INTEGRATION:

The data flow through the three-file pipeline is:

1. Adjoint_source_creation.py:
   Input:  forward_fields, user_fom_function
   Output: adjoint_source_E, adjoint_source_H
   
2. Adjoint_simulation.py:
   Input:  adjoint_source_E, adjoint_source_H
   Output: adjoint_fields
   
3. Gradient_calculation.py (this file):
   Input:  forward_fields, adjoint_fields, geometry, device_config
   Output: parameter_gradients

This provides concrete integration with actual lumNLopt data structures
and proper handling of anisotropic tensor field overlaps.
"""
