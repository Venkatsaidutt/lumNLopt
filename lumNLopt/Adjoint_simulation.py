"""
Adjoint_simulation.py

FDTD Adjoint Simulation Management for lumNLopt
Handles lumapi integration, source import, and adjoint simulation execution

Purpose: Bridge between adjoint source fields and FDTD simulation
Input:   Adjoint source field arrays (Esource, Hsource) from Adjoint_source_creation.py
Process: lumapi integration, FDTD simulation setup and execution
Output:  Adjoint electromagnetic fields for Gradient_calculation.py

Author: lumNLopt repository integration
Based on: lumopt_mbso adjoint simulation patterns
"""

import numpy as np
import scipy.constants
import lumapi
import warnings
import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# lumopt imports
from lumopt.lumerical_methods.lumerical_scripts import get_fields
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.utilities.wavelengths import Wavelengths

# lumopt_mbso imports (if available)
try:
    from lumopt_mbso.utils.spatial_integral import spatial_integral
    LUMOPT_MBSO_AVAILABLE = True
except ImportError:
    LUMOPT_MBSO_AVAILABLE = False
    def spatial_integral(data, x, y, z):
        """Fallback spatial integration"""
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0  
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        return np.sum(data) * dx * dy * dz


@dataclass
class AdjointSourceData:
    """Data structure for adjoint source field data"""
    Esource: np.ndarray          # Electric field adjoint source (Nx, Ny, Nz, Nwl, 3)
    Hsource: np.ndarray          # Magnetic field adjoint source (Nx, Ny, Nz, Nwl, 3)
    x: np.ndarray                # x coordinates
    y: np.ndarray                # y coordinates  
    z: np.ndarray                # z coordinates
    wavelengths: np.ndarray      # wavelengths
    fom_value: Optional[float] = None
    monitor_name: Optional[str] = None


@dataclass
class AdjointSimConfig:
    """Configuration for adjoint simulation"""
    adjoint_source_name: str = 'adjoint_source'
    monitor_name: str = 'opt_fields' 
    direction: str = 'Forward'           # Direction of original source
    multi_freq_src: bool = False         # Use multi-frequency source
    scaling_factor: float = 1.0          # Global scaling factor
    enable_validation: bool = True       # Input validation
    store_simulation: bool = True        # Store simulation files
    simulation_name: str = 'adjoint'     # Name for adjoint simulation


class AdjointSimulationManager:
    """
    Manages FDTD adjoint simulation workflow for lumNLopt optimization.
    
    Core responsibilities:
    1. Import adjoint source fields into FDTD via lumapi
    2. Configure adjoint simulation (disable forward, enable adjoint sources)  
    3. Execute adjoint FDTD simulation
    4. Extract adjoint electromagnetic fields
    5. Apply proper scaling and normalization
    
    Integrates with lumNLopt optimization workflow following lumopt_mbso patterns.
    """
    
    def __init__(self, config: AdjointSimConfig):
        """
        Initialize adjoint simulation manager.
        
        Parameters:
        -----------
        config : AdjointSimConfig
            Configuration object with simulation parameters
        """
        
        self.config = config
        
        # Runtime state
        self.adjoint_source_created: bool = False
        self.adjoint_fields: Optional[Any] = None
        self.simulation_stats: Dict = {}
        
        print(f"AdjointSimulationManager initialized:")
        print(f"  Adjoint source: {config.adjoint_source_name}")
        print(f"  Monitor: {config.monitor_name}")
        print(f"  Direction: {config.direction}")
        print(f"  Multi-frequency: {config.multi_freq_src}")
    
    def create_and_import_adjoint_source(self, sim, adjoint_data: AdjointSourceData) -> None:
        """
        Complete adjoint source setup: create object and import field data.
        
        This is the main method that handles the full adjoint source workflow:
        1. Create adjoint source object in FDTD (copies monitor geometry)
        2. Import adjoint source field data via lumapi
        3. Configure source properties
        
        Parameters:
        -----------
        sim : FDTD simulation object
            lumopt simulation object with fdtd handle
        adjoint_data : AdjointSourceData
            Adjoint source field data from Adjoint_source_creation.py
        """
        
        print("Setting up adjoint source in FDTD simulation...")
        
        if self.config.enable_validation:
            self._validate_adjoint_data(adjoint_data)
        
        # Step 1: Create adjoint source object (empty, with geometry from monitor)
        monitor_name = adjoint_data.monitor_name or self.config.monitor_name
        self._create_adjoint_source_object(sim, monitor_name)
        
        # Step 2: Import adjoint source field data
        self._import_adjoint_source_fields(sim, adjoint_data)
        
        # Step 3: Configure source properties
        self._configure_adjoint_source_properties(sim, adjoint_data)
        
        self.adjoint_source_created = True
        print(f"Adjoint source successfully created: {self.config.adjoint_source_name}")
    
    def _validate_adjoint_data(self, adjoint_data: AdjointSourceData) -> None:
        """Validate adjoint source data before import."""
        
        # Shape validation
        if adjoint_data.Esource.shape != adjoint_data.Hsource.shape:
            raise ValueError("Esource and Hsource must have the same shape")
        
        expected_shape = (len(adjoint_data.x), len(adjoint_data.y), 
                         len(adjoint_data.z), len(adjoint_data.wavelengths), 3)
        if adjoint_data.Esource.shape != expected_shape:
            raise ValueError(f"Source field shape {adjoint_data.Esource.shape} "
                           f"doesn't match coordinates {expected_shape}")
        
        # Data validation
        if not np.all(np.isfinite(adjoint_data.Esource)) or not np.all(np.isfinite(adjoint_data.Hsource)):
            raise ValueError("Adjoint source fields contain non-finite values")
        
        if len(adjoint_data.wavelengths) == 0:
            raise ValueError("Wavelength array is empty")
        
        # Physical validation
        E_max = np.abs(adjoint_data.Esource).max()
        H_max = np.abs(adjoint_data.Hsource).max()
        if E_max == 0 and H_max == 0:
            warnings.warn("All adjoint source fields are zero")
        
        print(f"Adjoint source validation passed:")
        print(f"  Field shape: {adjoint_data.Esource.shape}")
        print(f"  Max |Esource|: {E_max:.3e}")
        print(f"  Max |Hsource|: {H_max:.3e}")
    
    def _create_adjoint_source_object(self, sim, monitor_name: str) -> None:
        """
        Create adjoint source object in FDTD simulation.
        Based on lumopt_mbso.add_adjoint_source pattern.
        """
        
        print(f"Creating adjoint source object from monitor: {monitor_name}")
        
        # Verify FDTD solver exists
        if sim.fdtd.getnamednumber('FDTD') != 1:
            raise RuntimeError('No FDTD solver object found in simulation')
        
        # Add imported source object
        sim.fdtd.addimportedsource()
        sim.fdtd.set('name', self.config.adjoint_source_name)
        sim.fdtd.select(self.config.adjoint_source_name)
        
        # Copy monitor geometry to adjoint source
        self._copy_monitor_geometry(sim, monitor_name)
        
        # Initially disable (will be enabled during adjoint simulation)
        sim.fdtd.setnamed(self.config.adjoint_source_name, 'enabled', False)
        
        print(f"Adjoint source object created: {self.config.adjoint_source_name}")
    
    def _copy_monitor_geometry(self, sim, monitor_name: str) -> None:
        """Copy geometric properties from monitor to adjoint source."""
        
        # Validate monitor exists
        if sim.fdtd.getnamednumber(monitor_name) != 1:
            raise ValueError(f'Monitor "{monitor_name}" not found or name not unique')
        
        # Get monitor type and determine geometric properties
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
        
        # Set injection axis based on monitor normal
        adjoint_direction = 'Backward' if self.config.direction == 'Forward' else 'Forward'
        sim.fdtd.setnamed(self.config.adjoint_source_name, 'injection axis', 
                         normal.lower() + '-axis')
        if adjoint_direction == 'Backward':
            sim.fdtd.setnamed(self.config.adjoint_source_name, 'direction', 'Backward')
        
        # Copy geometric properties from monitor
        copied_props = []
        for prop_name in geo_props:
            try:
                prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
                sim.fdtd.setnamed(self.config.adjoint_source_name, prop_name, prop_val)
                copied_props.append(prop_name)
            except:
                continue  # Skip properties that don't exist
        
        print(f"Copied geometry properties: {copied_props}")
        print(f"Injection direction: {adjoint_direction} along {normal}-axis")
    
    def _import_adjoint_source_fields(self, sim, adjoint_data: AdjointSourceData) -> None:
        """
        Import adjoint source field data into FDTD via lumapi.
        Based on lumopt_mbso.import_adjoint_source pattern.
        """
        
        print("Importing adjoint source field data via lumapi...")
        
        # Extract coordinates and field data
        x = adjoint_data.x
        y = adjoint_data.y
        z = adjoint_data.z
        wavelengths = adjoint_data.wavelengths
        
        # Handle multi-frequency vs single frequency
        if self.config.multi_freq_src:
            used_wavelengths = wavelengths
            Esource = adjoint_data.Esource * self.config.scaling_factor
            Hsource = adjoint_data.Hsource * self.config.scaling_factor
            print(f"Using multi-frequency source: {len(wavelengths)} wavelengths")
        else:
            # Use center wavelength only (lumopt_mbso pattern)
            center_idx = len(wavelengths) // 2
            used_wavelengths = np.array([wavelengths[center_idx]])
            Esource = adjoint_data.Esource[:,:,:,center_idx:center_idx+1,:] * self.config.scaling_factor
            Hsource = adjoint_data.Hsource[:,:,:,center_idx:center_idx+1,:] * self.config.scaling_factor
            print(f"Using single-frequency source: λ = {used_wavelengths[0]*1e9:.1f} nm")
        
        # Convert wavelengths to frequencies for FDTD
        frequencies = scipy.constants.speed_of_light / used_wavelengths
        
        try:
            # Push coordinate arrays to lumapi
            lumapi.putMatrix(sim.fdtd.handle, 'x', x)
            lumapi.putMatrix(sim.fdtd.handle, 'y', y)
            lumapi.putMatrix(sim.fdtd.handle, 'z', z)
            lumapi.putMatrix(sim.fdtd.handle, 'f', frequencies)
            
            # Push field components (exactly following lumopt_mbso pattern)
            lumapi.putMatrix(sim.fdtd.handle, 'Ex', Esource[:,:,:,:,0])
            lumapi.putMatrix(sim.fdtd.handle, 'Ey', Esource[:,:,:,:,1]) 
            lumapi.putMatrix(sim.fdtd.handle, 'Ez', Esource[:,:,:,:,2])
            lumapi.putMatrix(sim.fdtd.handle, 'Hx', Hsource[:,:,:,:,0])
            lumapi.putMatrix(sim.fdtd.handle, 'Hy', Hsource[:,:,:,:,1])
            lumapi.putMatrix(sim.fdtd.handle, 'Hz', Hsource[:,:,:,:,2])
            
            # Create electromagnetic dataset (lumopt_mbso pattern)
            sim.fdtd.eval("EM = rectilineardataset('EM fields', x, y, z);")
            sim.fdtd.eval("EM.addparameter('lambda', c/f, 'f', f);") 
            sim.fdtd.eval("EM.addattribute('E', Ex, Ey, Ez);")
            sim.fdtd.eval("EM.addattribute('H', Hx, Hy, Hz);")
            
            # Import dataset into adjoint source
            sim.fdtd.select(self.config.adjoint_source_name)
            dataset = sim.fdtd.getv("EM")
            sim.fdtd.importdataset(dataset)
            
            print(f"Field data imported successfully:")
            print(f"  Grid size: {len(x)} × {len(y)} × {len(z)}")
            print(f"  Frequencies: {len(frequencies)}")
            print(f"  Source magnitude: E={np.abs(Esource).max():.3e}, H={np.abs(Hsource).max():.3e}")
            
        except Exception as e:
            raise RuntimeError(f"lumapi data import failed: {e}") from e
    
    def _configure_adjoint_source_properties(self, sim, adjoint_data: AdjointSourceData) -> None:
        """Configure adjoint source properties in FDTD."""
        
        try:
            sim.fdtd.select(self.config.adjoint_source_name)
            
            # Configure frequency settings (lumopt_mbso pattern)
            if not self.config.multi_freq_src:
                sim.fdtd.setnamed(self.config.adjoint_source_name, 
                                 'override global source settings', False)
            
            print("Adjoint source properties configured")
            
        except Exception as e:
            warnings.warn(f"Adjoint source configuration issue: {e}")
    
    def setup_adjoint_simulation(self, sim, params=None) -> None:
        """
        Configure FDTD simulation for adjoint solve.
        Based on lumNLopt.optimization.make_adjoint_sim pattern.
        
        Parameters:
        -----------
        sim : FDTD simulation object
        params : array-like, optional
            Geometry parameters for current optimization iteration
        """
        
        print("Configuring simulation for adjoint solve...")
        
        if not self.adjoint_source_created:
            raise RuntimeError("Adjoint source must be created before simulation setup")
        
        # Switch to layout mode (lumNLopt pattern)
        sim.fdtd.switchtolayout()
        
        # Update geometry if parameters provided
        if params is not None:
            # This would typically call geometry.add_geo() in lumNLopt
            # Implementation depends on specific geometry class being used
            pass
        
        # Disable forward source (lumNLopt pattern) 
        if sim.fdtd.getnamednumber('source') >= 1:
            sim.fdtd.setnamed('source', 'enabled', False)
            print("Forward source disabled")
        
        # Enable adjoint source
        sim.fdtd.setnamed(self.config.adjoint_source_name, 'enabled', True)
        print(f"Adjoint source enabled: {self.config.adjoint_source_name}")
    
    def run_adjoint_simulation(self, sim, iteration: int = 0) -> None:
        """
        Execute FDTD adjoint simulation.
        Based on lumNLopt.optimization.run_adjoint_solves pattern.
        
        Parameters:
        -----------
        sim : FDTD simulation object
        iteration : int
            Optimization iteration number for file naming
        """
        
        print("Running FDTD adjoint simulation...")
        
        if not self.adjoint_source_created:
            raise RuntimeError("Adjoint source must be created before simulation")
        
        # Run adjoint simulation (lumNLopt pattern)
        simulation_name = f"{self.config.simulation_name}"
        if self.config.store_simulation:
            sim.run(name=simulation_name, iter=iteration)
        else:
            sim.run(name=simulation_name)
        
        print(f"Adjoint simulation completed: {simulation_name}")
    
    def extract_adjoint_fields(self, sim, get_H: bool = True, get_eps: bool = True, 
                             get_D: bool = False, nointerpolation: bool = False,
                             unfold_symmetry: bool = True) -> Any:
        """
        Extract adjoint electromagnetic fields from simulation.
        Based on lumNLopt.optimization.run_adjoint_solves pattern.
        
        Parameters:
        -----------
        sim : FDTD simulation object
        get_H : bool
            Extract magnetic field components  
        get_eps : bool
            Extract permittivity data
        get_D : bool
            Extract electric displacement field
        nointerpolation : bool
            Disable field interpolation
        unfold_symmetry : bool
            Unfold symmetric simulation regions
            
        Returns:
        --------
        adjoint_fields : Field object
            Adjoint electromagnetic field data with scaling applied
        """
        
        print("Extracting adjoint electromagnetic fields...")
        
        try:
            # Extract adjoint fields (lumNLopt pattern)
            adjoint_fields = get_fields(sim.fdtd,
                                      monitor_name=self.config.monitor_name,
                                      field_result_name='adjoint_fields',
                                      get_eps=get_eps,
                                      get_D=get_D,
                                      get_H=get_H,
                                      nointerpolation=nointerpolation,
                                      unfold_symmetry=unfold_symmetry)
            
            # Calculate adjoint field scaling (lumNLopt pattern)
            scaling_factor = self._calculate_adjoint_field_scaling(sim, adjoint_fields.wl)
            adjoint_fields.scaling_factor = scaling_factor
            
            # Apply scaling to fields (lumNLopt pattern - scale axis 3)
            adjoint_fields.scale(3, scaling_factor)
            
            # Store results
            self.adjoint_fields = adjoint_fields
            self._update_simulation_stats(adjoint_fields)
            
            print(f"Adjoint fields extracted successfully:")
            print(f"  Field shape: {adjoint_fields.E.shape}")
            print(f"  Wavelengths: {len(adjoint_fields.wl)}")
            print(f"  Scaling factor magnitude: {np.abs(scaling_factor).max():.3e}")
            
            return adjoint_fields
            
        except Exception as e:
            raise RuntimeError(f"Adjoint field extraction failed: {e}") from e
    
    def _calculate_adjoint_field_scaling(self, sim, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculate adjoint field scaling factors.
        Based on lumopt_mbso.get_adjoint_field_scaling patterns.
        """
        
        omega = 2.0 * np.pi * scipy.constants.speed_of_light / wavelengths
        
        try:
            # Get adjoint source power for normalization (lumopt_mbso pattern)
            adjoint_source_power = ModeMatch.get_source_power(sim, wavelengths)
            scaling = 1j * omega * self.config.scaling_factor / (4 * adjoint_source_power)
        except:
            # Fallback scaling if source power calculation fails
            scaling = 1j * omega * self.config.scaling_factor / 4
            warnings.warn("Using fallback adjoint field scaling")
        
        return scaling
    
    def _update_simulation_stats(self, adjoint_fields) -> None:
        """Update simulation statistics."""
        
        self.simulation_stats = {
            'field_shape': adjoint_fields.E.shape,
            'wavelengths': len(adjoint_fields.wl),
            'max_E_magnitude': float(np.abs(adjoint_fields.E).max()),
            'max_H_magnitude': float(np.abs(adjoint_fields.H).max()) if hasattr(adjoint_fields, 'H') else 0.0,
            'scaling_magnitude': float(np.abs(adjoint_fields.scaling_factor).max())
        }
    
    def get_simulation_stats(self) -> Dict:
        """Return simulation statistics."""
        return self.simulation_stats.copy()
    
    def cleanup_simulation_files(self, sim) -> None:
        """Clean up temporary simulation files if needed."""
        
        if self.config.store_simulation:
            try:
                sim.remove_data_and_save()
                print("Simulation data cleaned up")
            except:
                pass  # Ignore cleanup errors


# ============================================================================
# CONVENIENCE FUNCTIONS FOR DIRECT USAGE
# ============================================================================

def run_adjoint_simulation_workflow(sim, adjoint_data: AdjointSourceData, 
                                  config: Optional[AdjointSimConfig] = None,
                                  params=None, iteration: int = 0) -> Any:
    """
    Complete adjoint simulation workflow in a single function call.
    
    This is the main entry point for running the full adjoint simulation:
    1. Create and import adjoint source
    2. Setup adjoint simulation  
    3. Run FDTD adjoint simulation
    4. Extract adjoint fields
    
    Parameters:
    -----------
    sim : FDTD simulation object
    adjoint_data : AdjointSourceData
        Adjoint source field data from Adjoint_source_creation.py
    config : AdjointSimConfig, optional
        Simulation configuration (uses defaults if None)
    params : array-like, optional
        Geometry parameters for optimization iteration
    iteration : int
        Optimization iteration number
        
    Returns:
    --------
    adjoint_fields : Field object
        Adjoint electromagnetic field data for Gradient_calculation.py
        
    Example:
    --------
    >>> # From previous step
    >>> from Adjoint_source_creation import compute_adjoint_source_from_fom
    >>> Esource, Hsource = compute_adjoint_source_from_fom(my_fom, ...)
    >>> 
    >>> # Create adjoint source data
    >>> adjoint_data = AdjointSourceData(
    ...     Esource=Esource, Hsource=Hsource,
    ...     x=forward_fields.x, y=forward_fields.y, z=forward_fields.z,
    ...     wavelengths=forward_fields.wl, monitor_name='opt_fields'
    ... )
    >>> 
    >>> # Run complete adjoint simulation
    >>> adjoint_fields = run_adjoint_simulation_workflow(sim, adjoint_data)
    >>> 
    >>> # Pass to next step in pipeline
    >>> # gradient_calc.compute_parameter_gradients(forward_fields, adjoint_fields, ...)
    """
    
    # Use default configuration if none provided
    if config is None:
        config = AdjointSimConfig()
    
    # Create simulation manager
    manager = AdjointSimulationManager(config)
    
    # Execute complete workflow
    print("="*60)
    print("ADJOINT SIMULATION WORKFLOW")
    print("="*60)
    
    # Step 1: Create and import adjoint source
    manager.create_and_import_adjoint_source(sim, adjoint_data)
    
    # Step 2: Setup adjoint simulation
    manager.setup_adjoint_simulation(sim, params)
    
    # Step 3: Run adjoint simulation
    manager.run_adjoint_simulation(sim, iteration)
    
    # Step 4: Extract adjoint fields
    adjoint_fields = manager.extract_adjoint_fields(sim)
    
    # Print summary
    stats = manager.get_simulation_stats()
    print("\nADJOINT SIMULATION SUMMARY:")
    print(f"  Field shape: {stats['field_shape']}")
    print(f"  Max |E|: {stats['max_E_magnitude']:.3e}")
    print(f"  Max |H|: {stats['max_H_magnitude']:.3e}")
    print("="*60 + "\n")
    
    return adjoint_fields


def create_adjoint_source_from_arrays(sim, Esource: np.ndarray, Hsource: np.ndarray,
                                     x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                     wavelengths: np.ndarray, monitor_name: str,
                                     config: Optional[AdjointSimConfig] = None) -> None:
    """
    Convenience function to create adjoint source from field arrays.
    
    Parameters:
    -----------
    sim : FDTD simulation object
    Esource, Hsource : np.ndarray
        Adjoint source field arrays from Adjoint_source_creation.py
    x, y, z : np.ndarray
        Spatial coordinate arrays
    wavelengths : np.ndarray
        Wavelength array
    monitor_name : str
        Name of monitor for geometry copying
    config : AdjointSimConfig, optional
        Configuration (uses defaults if None)
    """
    
    # Create adjoint source data object
    adjoint_data = AdjointSourceData(
        Esource=Esource, Hsource=Hsource,
        x=x, y=y, z=z, wavelengths=wavelengths,
        monitor_name=monitor_name
    )
    
    # Use default config if none provided
    if config is None:
        config = AdjointSimConfig()
    
    # Create and import adjoint source
    manager = AdjointSimulationManager(config)
    manager.create_and_import_adjoint_source(sim, adjoint_data)


# ============================================================================
# INTEGRATION EXAMPLES FOR LUMNLOPT
# ============================================================================

"""
USAGE EXAMPLES:

1. Complete Adjoint Simulation Workflow:
```python
from Adjoint_source_creation import compute_adjoint_source_from_fom
from Adjoint_simulation import run_adjoint_simulation_workflow, AdjointSourceData

# Step 1: Get adjoint source fields (from previous file)
Esource, Hsource = compute_adjoint_source_from_fom(
    my_fom_function, forward_fields.E, forward_fields.H,
    forward_fields.x, forward_fields.y, forward_fields.z, forward_fields.wl
)

# Step 2: Create adjoint data object
adjoint_data = AdjointSourceData(
    Esource=Esource, Hsource=Hsource,
    x=forward_fields.x, y=forward_fields.y, z=forward_fields.z,
    wavelengths=forward_fields.wl, monitor_name='opt_fields'
)

# Step 3: Run complete adjoint simulation
adjoint_fields = run_adjoint_simulation_workflow(sim, adjoint_data)

# Step 4: Pass to gradient calculation
# (handled by Gradient_calculation.py)
```

2. Advanced Configuration:
```python
from Adjoint_simulation import AdjointSimConfig, AdjointSimulationManager

# Custom configuration
config = AdjointSimConfig(
    adjoint_source_name='my_adjoint_source',
    monitor_name='custom_monitor',
    multi_freq_src=True,
    scaling_factor=1e-6,
    store_simulation=True
)

# Manual control over simulation steps
manager = AdjointSimulationManager(config)
manager.create_and_import_adjoint_source(sim, adjoint_data)
manager.setup_adjoint_simulation(sim, geometry_params)
manager.run_adjoint_simulation(sim, iteration=5)
adjoint_fields = manager.extract_adjoint_fields(sim)
```

3. Integration with lumNLopt Optimization Loop:
```python
# In your lumNLopt optimization workflow:

class MyOptimization(Optimization):
    
    def run_adjoint_solves(self, params):
        # Get forward fields (already computed)
        forward_fields = self.forward_fields
        
        # Generate adjoint source using XAD
        Esource, Hsource = compute_adjoint_source_from_fom(
            self.user_fom_function, forward_fields.E, forward_fields.H,
            forward_fields.x, forward_fields.y, forward_fields.z, forward_fields.wl
        )
        
        # Create adjoint data
        adjoint_data = AdjointSourceData(
            Esource=Esource, Hsource=Hsource,
            x=forward_fields.x, y=forward_fields.y, z=forward_fields.z,
            wavelengths=forward_fields.wl, monitor_name='opt_fields'
        )
        
        # Run adjoint simulation
        self.adjoint_fields = run_adjoint_simulation_workflow(
            self.sim, adjoint_data, params=params, 
            iteration=self.optimizer.iteration
        )
```

TECHNICAL NOTES:

- Follows lumopt_mbso patterns for lumapi integration
- Compatible with lumNLopt optimization workflow
- Handles both single and multi-frequency adjoint sources
- Proper field scaling and normalization applied
- Comprehensive error handling and validation
- Memory efficient - cleans up temporary data
- Integrates seamlessly with Adjoint_source_creation.py and Gradient_calculation.py

The file handles all FDTD simulation aspects of the adjoint workflow.
Field computation is done by Adjoint_source_creation.py.
Parameter gradient calculation is done by Gradient_calculation.py.
"""
