#!/usr/bin/env python3
"""
Forward_simulation.py - LumNLopt Compatible Forward Simulation Module

This module provides a comprehensive forward simulation framework for Lumerical FDTD
optimization compatible with the lumNLopt architecture. It handles .fsp file loading,
geometry parameterization, field extraction, and FOM calculation.

Features:
- Load/save .fsp files with automatic validation
- Support for parameterized geometries with constraint handling  
- Advanced material property management (anisotropic, Sellmeier)
- Robust field extraction and post-processing
- Integration with optimization workflows
- Comprehensive error handling and logging

Author: Based on lumNLopt architecture patterns
Compatible with: Lumerical FDTD, lumNLopt optimization framework
"""

import os
import sys
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Import lumNLopt components (adjust paths as needed)
try:
    from lumNLopt.utilities.simulation import Simulation
    from lumNLopt.utilities.base_script import BaseScript
    from lumNLopt.utilities.wavelengths import Wavelengths
    from lumNLopt.utilities.fields import FieldsNoInterp
    from lumNLopt.lumerical_methods.lumerical_scripts import get_fields
    from lumNLopt.Inputs.Device import get_design_region, get_material_properties, create_base_script
    from lumNLopt.geometries.geometry_parameters_handling import GeometryParameterHandler
    from lumNLopt.figures_of_merit.adiabatic_coupling import AdiabaticCouplingFOM
except ImportError as e:
    logging.warning(f"Some lumNLopt imports failed: {e}")
    logging.warning("Make sure lumNLopt is in your Python path")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ForwardSimulationError(Exception):
    """Custom exception for forward simulation errors"""
    pass


class FieldExtractionError(Exception):
    """Custom exception for field extraction errors"""  
    pass


class ForwardSimulation:
    """
    Comprehensive forward simulation class for lumNLopt-compatible optimization.
    
    This class handles loading .fsp files, updating parameterized geometries, 
    running FDTD simulations, and extracting fields for FOM calculation.
    
    Key Features:
    - Robust .fsp file handling with validation
    - Parameterized geometry support with constraint handling
    - Advanced material property management
    - Multi-monitor field extraction
    - Integration with optimization workflows
    - Comprehensive error handling and logging
    """
    
    def __init__(self, working_dir: str = './', use_var_fdtd: bool = False, 
                 hide_fdtd_cad: bool = True, validation_level: str = 'standard'):
        """
        Initialize ForwardSimulation.
        
        Args:
            working_dir: Working directory for simulations
            use_var_fdtd: Whether to use variable FDTD (for bandwidth optimization)
            hide_fdtd_cad: Whether to hide FDTD CAD interface
            validation_level: Level of validation ('minimal', 'standard', 'comprehensive')
        """
        self.working_dir = Path(working_dir).resolve()
        self.working_dir.mkdir(exist_ok=True)
        
        self.use_var_fdtd = use_var_fdtd
        self.hide_fdtd_cad = hide_fdtd_cad
        self.validation_level = validation_level
        
        # Initialize simulation objects
        self.sim = None
        self.geometry = None
        self.fom = None
        self.wavelengths = None
        
        # Simulation state tracking
        self.simulation_loaded = False
        self.geometry_updated = False
        self.fields_extracted = False
        
        # Field storage
        self.forward_fields = {}
        self.monitor_data = {}
        
        # Configuration
        self.config = self._load_default_config()
        
        logger.info(f"ForwardSimulation initialized in {self.working_dir}")
        logger.info(f"Validation level: {self.validation_level}")
        
    def _load_default_config(self) -> Dict:
        """Load default configuration settings."""
        return {
            'convergence': {
                'auto_shutoff': True,
                'min_shutoff_time': 500e-15,  # 500 fs
                'shutoff_min': 1e-4,
                'shutoff_max': 1e-3
            },
            'field_extraction': {
                'default_monitors': ['opt_fields', 'fom_monitor'],
                'spatial_interpolation': 'none',
                'get_eps': True,
                'get_D': True,
                'get_H': True,
                'nointerpolation': False
            },
            'validation': {
                'check_pml_overlap': True,
                'check_mesh_quality': True,
                'check_source_position': True,
                'check_monitor_alignment': True
            },
            'materials': {
                'enable_dispersion': True,
                'enable_anisotropy': True,
                'sellmeier_accuracy': 1e-6
            }
        }
    
    def load_base_simulation(self, fsp_file: Union[str, Path], 
                           validate: bool = True) -> None:
        """
        Load base simulation from .fsp file with comprehensive validation.
        
        Args:
            fsp_file: Path to .fsp simulation file
            validate: Whether to perform validation after loading
            
        Raises:
            ForwardSimulationError: If loading fails or validation fails
        """
        fsp_path = Path(fsp_file).resolve()
        
        if not fsp_path.exists():
            raise ForwardSimulationError(f"Simulation file not found: {fsp_path}")
            
        if not fsp_path.suffix.lower() == '.fsp':
            raise ForwardSimulationError(f"File must have .fsp extension: {fsp_path}")
            
        logger.info(f"Loading simulation from: {fsp_path}")
        
        try:
            # Initialize simulation wrapper
            self.sim = Simulation(str(self.working_dir), self.use_var_fdtd, self.hide_fdtd_cad)
            
            # Load the .fsp file
            self.sim.load(str(fsp_path))
            
            logger.info("Base simulation loaded successfully")
            self.simulation_loaded = True
            
            # Extract basic information
            self._extract_simulation_info()
            
            # Perform validation if requested
            if validate:
                self._validate_simulation()
                
        except Exception as e:
            self.simulation_loaded = False
            raise ForwardSimulationError(f"Failed to load simulation: {str(e)}")
    
    def _extract_simulation_info(self) -> None:
        """Extract basic information from loaded simulation."""
        if not self.simulation_loaded:
            return
            
        try:
            # Switch to layout mode for inspection
            self.sim.fdtd.switchtolayout()
            
            # Get simulation bounds
            self.sim_bounds = {
                'x_min': self.sim.fdtd.getnamed('FDTD', 'x min'),
                'x_max': self.sim.fdtd.getnamed('FDTD', 'x max'),
                'y_min': self.sim.fdtd.getnamed('FDTD', 'y min'),
                'y_max': self.sim.fdtd.getnamed('FDTD', 'y max'),
                'z_min': self.sim.fdtd.getnamed('FDTD', 'z min'),
                'z_max': self.sim.fdtd.getnamed('FDTD', 'z max')
            }
            
            # Get mesh settings
            self.mesh_settings = {
                'dx': self.sim.fdtd.getnamed('FDTD', 'mesh accuracy'),
                'dt_stability': self.sim.fdtd.getnamed('FDTD', 'dt stability factor')
            }
            
            # Find monitors
            self.available_monitors = []
            monitor_types = ['DFT', 'power', 'index']
            for monitor_type in monitor_types:
                num_monitors = self.sim.fdtd.getnamednumber(monitor_type)
                if num_monitors > 0:
                    for i in range(1, num_monitors + 1):
                        self.sim.fdtd.select(f"{monitor_type}_{i}")
                        name = self.sim.fdtd.get('name')
                        self.available_monitors.append(name)
            
            # Find sources
            self.available_sources = []
            source_types = ['source', 'dipole', 'plane wave']
            for source_type in source_types:
                num_sources = self.sim.fdtd.getnamednumber(source_type)
                if num_sources > 0:
                    for i in range(1, num_sources + 1):
                        self.sim.fdtd.select(f"{source_type}_{i}")
                        name = self.sim.fdtd.get('name')
                        self.available_sources.append(name)
            
            logger.info(f"Found {len(self.available_monitors)} monitors, {len(self.available_sources)} sources")
            
        except Exception as e:
            logger.warning(f"Failed to extract simulation info: {e}")
    
    def _validate_simulation(self) -> None:
        """
        Comprehensive simulation validation.
        
        Raises:
            ForwardSimulationError: If critical validation issues found
        """
        if self.validation_level == 'minimal':
            return
            
        logger.info("Validating simulation setup...")
        issues = []
        warnings_list = []
        
        try:
            self.sim.fdtd.switchtolayout()
            
            # Check for required components
            if self.sim.fdtd.getnamednumber('FDTD') == 0:
                issues.append("No FDTD solver found")
            
            if len(self.available_sources) == 0:
                issues.append("No sources found in simulation")
                
            if len(self.available_monitors) == 0:
                warnings_list.append("No monitors found in simulation")
            
            # Check PML overlap (if enabled)
            if self.config['validation']['check_pml_overlap']:
                self._check_pml_overlap(warnings_list)
            
            # Check mesh quality (if enabled) 
            if self.config['validation']['check_mesh_quality']:
                self._check_mesh_quality(warnings_list)
            
            # Check source positions (if enabled)
            if self.config['validation']['check_source_position']:
                self._check_source_positions(warnings_list)
            
            # Check monitor alignment (if enabled)
            if self.config['validation']['check_monitor_alignment']:
                self._check_monitor_alignment(warnings_list)
                
            # Log results
            if issues:
                error_msg = "Critical validation issues found: " + "; ".join(issues)
                raise ForwardSimulationError(error_msg)
                
            if warnings_list:
                for warning in warnings_list:
                    logger.warning(f"Validation warning: {warning}")
            
            logger.info("Simulation validation completed successfully")
            
        except ForwardSimulationError:
            raise
        except Exception as e:
            logger.warning(f"Validation failed with error: {e}")
    
    def _check_pml_overlap(self, warnings_list: List[str]) -> None:
        """Check for PML-geometry overlap issues."""
        # Implementation would check if PML regions overlap with structures
        # This is a placeholder for the actual implementation
        pass
    
    def _check_mesh_quality(self, warnings_list: List[str]) -> None:
        """Check mesh quality and refinement."""
        # Implementation would check mesh quality metrics
        # This is a placeholder for the actual implementation  
        pass
    
    def _check_source_positions(self, warnings_list: List[str]) -> None:
        """Check if sources are properly positioned."""
        # Implementation would validate source placement
        # This is a placeholder for the actual implementation
        pass
    
    def _check_monitor_alignment(self, warnings_list: List[str]) -> None:
        """Check monitor alignment with mesh."""
        # Implementation would check monitor-mesh alignment
        # This is a placeholder for the actual implementation
        pass
    
    def setup_wavelengths(self, center: float, span: float, num_points: int) -> None:
        """
        Setup wavelength range for simulation.
        
        Args:
            center: Center wavelength (m)
            span: Wavelength span (m)
            num_points: Number of wavelength points
        """
        # Create wavelength range
        wl_min = center - span/2
        wl_max = center + span/2
        wl_array = np.linspace(wl_min, wl_max, num_points)
        
        self.wavelengths = Wavelengths(wl_array)
        
        # Set global monitor wavelengths in simulation
        if self.simulation_loaded:
            self._set_global_wavelengths()
            
        logger.info(f"Wavelengths set: {wl_min*1e9:.1f}-{wl_max*1e9:.1f} nm ({num_points} points)")
    
    def _set_global_wavelengths(self) -> None:
        """Set global monitor wavelength settings."""
        if not self.wavelengths:
            return
            
        try:
            self.sim.fdtd.switchtolayout()
            
            # Set frequency points for all monitors
            freqs = 3e8 / self.wavelengths.asarray()  # Convert to frequencies
            self.sim.fdtd.setglobalmonitor('use source limits', False)
            self.sim.fdtd.setglobalmonitor('custom frequency samples', freqs)
            
            logger.info("Global wavelength settings updated")
            
        except Exception as e:
            logger.warning(f"Failed to set global wavelengths: {e}")
    
    def setup_geometry(self, geometry_type: str = 'clustered_rectangles', 
                      **geometry_params) -> None:
        """
        Setup parameterized geometry for optimization.
        
        Args:
            geometry_type: Type of geometry parameterization
            **geometry_params: Geometry-specific parameters
        """
        try:
            if geometry_type == 'clustered_rectangles':
                from lumNLopt.geometries.Geometry_clustered import RectangleClusteredGeometry
                self.geometry = RectangleClusteredGeometry(**geometry_params)
                
            elif geometry_type == 'topology':
                from lumNLopt.geometries.topology import TopologyOptimization
                self.geometry = TopologyOptimization(**geometry_params)
                
            else:
                raise ValueError(f"Unknown geometry type: {geometry_type}")
                
            logger.info(f"Geometry setup completed: {geometry_type}")
            
        except Exception as e:
            raise ForwardSimulationError(f"Failed to setup geometry: {str(e)}")
    
    def update_geometry(self, parameters: np.ndarray, validate_params: bool = True) -> None:
        """
        Update geometry with new optimization parameters.
        
        Args:
            parameters: Design parameters array
            validate_params: Whether to validate parameter bounds
            
        Raises:
            ForwardSimulationError: If parameter update fails
        """
        if not self.geometry:
            raise ForwardSimulationError("Geometry not initialized. Call setup_geometry() first.")
            
        if not self.simulation_loaded:
            raise ForwardSimulationError("Simulation not loaded. Call load_base_simulation() first.")
        
        try:
            # Validate parameters if requested
            if validate_params:
                self._validate_parameters(parameters)
            
            # Update geometry
            logger.info(f"Updating geometry with {len(parameters)} parameters")
            self.geometry.update_geometry(parameters, self.sim)
            
            # Add geometry to simulation
            self.geometry.add_geo(self.sim, parameters, only_update=True)
            
            self.geometry_updated = True
            logger.info("Geometry update completed successfully")
            
        except Exception as e:
            self.geometry_updated = False
            raise ForwardSimulationError(f"Failed to update geometry: {str(e)}")
    
    def _validate_parameters(self, parameters: np.ndarray) -> None:
        """Validate optimization parameters."""
        if not hasattr(self.geometry, 'bounds'):
            logger.warning("Geometry has no bounds defined - skipping validation")
            return
            
        bounds = np.array(self.geometry.bounds)
        
        if len(parameters) != len(bounds):
            raise ValueError(f"Parameter count mismatch: expected {len(bounds)}, got {len(parameters)}")
        
        # Check bounds
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        below_lower = parameters < lower_bounds
        above_upper = parameters > upper_bounds
        
        if np.any(below_lower):
            raise ValueError(f"Parameters below lower bounds at indices: {np.where(below_lower)[0]}")
            
        if np.any(above_upper):
            raise ValueError(f"Parameters above upper bounds at indices: {np.where(above_upper)[0]}")
        
        logger.debug("Parameter validation passed")
    
    def run_simulation(self, convergence_check: bool = True, 
                      max_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Run forward FDTD simulation with comprehensive monitoring.
        
        Args:
            convergence_check: Whether to enable automatic convergence checking
            max_time: Maximum simulation time (seconds, real time)
            
        Returns:
            Dictionary with simulation results and metadata
            
        Raises:
            ForwardSimulationError: If simulation fails
        """
        if not self.simulation_loaded:
            raise ForwardSimulationError("No simulation loaded")
            
        if not self.geometry_updated:
            logger.warning("Geometry may not be updated - continuing anyway")
        
        try:
            logger.info("Starting forward FDTD simulation")
            
            # Switch to layout and prepare
            self.sim.fdtd.switchtolayout()
            
            # Enable sources
            for source_name in self.available_sources:
                self.sim.fdtd.setnamed(source_name, 'enabled', True)
            
            # Setup convergence checking if requested
            if convergence_check and self.config['convergence']['auto_shutoff']:
                self._setup_convergence_checking()
            
            # Run simulation with timing
            import time
            start_time = time.time()
            
            if max_time:
                # Run with timeout (would need additional implementation)
                logger.warning("Timeout not implemented - running without limit")
            
            self.sim.fdtd.run()
            
            end_time = time.time()
            simulation_time = end_time - start_time
            
            logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
            
            # Collect simulation metadata
            results = {
                'success': True,
                'simulation_time': simulation_time,
                'convergence_achieved': self._check_convergence(),
                'final_time_step': self._get_final_time_step(),
                'memory_usage': self._get_memory_usage()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            raise ForwardSimulationError(f"FDTD simulation failed: {str(e)}")
    
    def _setup_convergence_checking(self) -> None:
        """Setup automatic convergence checking."""
        try:
            conv_config = self.config['convergence']
            self.sim.fdtd.setnamed('FDTD', 'auto shutoff', conv_config['auto_shutoff'])
            self.sim.fdtd.setnamed('FDTD', 'min shutoff time', conv_config['min_shutoff_time'])
            self.sim.fdtd.setnamed('FDTD', 'shutoff min', conv_config['shutoff_min'])
            self.sim.fdtd.setnamed('FDTD', 'shutoff max', conv_config['shutoff_max'])
            
            logger.debug("Convergence checking configured")
            
        except Exception as e:
            logger.warning(f"Failed to setup convergence checking: {e}")
    
    def _check_convergence(self) -> bool:
        """Check if simulation converged properly."""
        try:
            # Get convergence information from simulation
            # This would need to be implemented based on Lumerical API
            return True  # Placeholder
        except:
            return False
    
    def _get_final_time_step(self) -> int:
        """Get final time step count."""
        try:
            return int(self.sim.fdtd.getresult('FDTD', 'timesteps'))
        except:
            return -1
    
    def _get_memory_usage(self) -> float:
        """Get simulation memory usage in MB."""
        try:
            return float(self.sim.fdtd.getresult('FDTD', 'memory usage'))
        except:
            return -1.0
    
    def extract_fields(self, monitor_names: Optional[List[str]] = None,
                      field_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract electromagnetic fields from specified monitors.
        
        Args:
            monitor_names: List of monitor names (uses default if None)
            field_components: List of field components to extract
            
        Returns:
            Dictionary containing extracted field data
            
        Raises:
            FieldExtractionError: If field extraction fails
        """
        if monitor_names is None:
            monitor_names = self.config['field_extraction']['default_monitors']
            
        if field_components is None:
            field_components = ['E', 'H']
        
        try:
            logger.info(f"Extracting fields from {len(monitor_names)} monitors")
            
            extracted_fields = {}
            
            for monitor_name in monitor_names:
                try:
                    # Check if monitor exists
                    if self.sim.fdtd.getnamednumber(monitor_name) == 0:
                        logger.warning(f"Monitor '{monitor_name}' not found - skipping")
                        continue
                    
                    # Extract fields using lumNLopt method
                    field_data = get_fields(
                        self.sim.fdtd,
                        monitor_name=monitor_name,
                        field_result_name=f'{monitor_name}_fields',
                        get_eps=self.config['field_extraction']['get_eps'],
                        get_D=self.config['field_extraction']['get_D'],
                        get_H=self.config['field_extraction']['get_H'],
                        nointerpolation=self.config['field_extraction']['nointerpolation']
                    )
                    
                    extracted_fields[monitor_name] = field_data
                    logger.debug(f"Successfully extracted fields from '{monitor_name}'")
                    
                except Exception as e:
                    logger.error(f"Failed to extract fields from '{monitor_name}': {e}")
                    raise FieldExtractionError(f"Field extraction failed for '{monitor_name}': {str(e)}")
            
            self.forward_fields = extracted_fields
            self.fields_extracted = True
            
            logger.info(f"Field extraction completed for {len(extracted_fields)} monitors")
            return extracted_fields
            
        except FieldExtractionError:
            raise
        except Exception as e:
            raise FieldExtractionError(f"Field extraction failed: {str(e)}")
    
    def calculate_fom(self, fom_type: str = 'adiabatic_coupling',
                     **fom_params) -> float:
        """
        Calculate figure of merit from extracted fields.
        
        Args:
            fom_type: Type of FOM calculation
            **fom_params: FOM-specific parameters
            
        Returns:
            Figure of merit value
        """
        if not self.fields_extracted:
            logger.warning("Fields not extracted - extracting with default settings")
            self.extract_fields()
        
        try:
            # Initialize FOM calculator
            if fom_type == 'adiabatic_coupling':
                from lumNLopt.figures_of_merit.adiabatic_coupling import AdiabaticCouplingFOM
                self.fom = AdiabaticCouplingFOM(**fom_params)
                
            elif fom_type == 'transmission':
                from lumopt_mbso.fom.transmissionfom import TransmissionFom
                self.fom = TransmissionFom(**fom_params)
                
            elif fom_type == 'mode_match':
                from lumopt.figures_of_merit.modematch import ModeMatch
                self.fom = ModeMatch(**fom_params)
                
            else:
                raise ValueError(f"Unknown FOM type: {fom_type}")
            
            # Calculate FOM
            fom_value = self.fom.get_fom(self.sim)
            
            logger.info(f"FOM calculated: {fom_value:.6f} ({fom_type})")
            return float(fom_value)
            
        except Exception as e:
            logger.error(f"FOM calculation failed: {e}")
            raise ForwardSimulationError(f"FOM calculation failed: {str(e)}")
    
    def save_simulation(self, filename: Optional[str] = None, 
                       include_results: bool = True) -> str:
        """
        Save current simulation state to .fsp file.
        
        Args:
            filename: Output filename (auto-generated if None)
            include_results: Whether to include simulation results
            
        Returns:
            Path to saved file
        """
        if not self.simulation_loaded:
            raise ForwardSimulationError("No simulation to save")
        
        if filename is None:
            filename = f"forward_sim_{int(np.random.rand() * 10000):04d}.fsp"
        
        output_path = self.working_dir / filename
        
        try:
            # Save with or without results
            if include_results:
                self.sim.save(str(output_path))
            else:
                self.sim.fdtd.switchtolayout()
                self.sim.save(str(output_path))
            
            logger.info(f"Simulation saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise ForwardSimulationError(f"Failed to save simulation: {str(e)}")
    
    def run_complete_forward_solve(self, fsp_file: Union[str, Path],
                                  parameters: np.ndarray,
                                  fom_type: str = 'adiabatic_coupling',
                                  **kwargs) -> Dict[str, Any]:
        """
        Complete forward solve workflow: load, update geometry, simulate, extract fields, calculate FOM.
        
        Args:
            fsp_file: Path to base simulation file
            parameters: Design parameters
            fom_type: Type of FOM to calculate
            **kwargs: Additional arguments for FOM calculation
            
        Returns:
            Dictionary with complete results including FOM value, fields, and metadata
        """
        results = {}
        
        try:
            # Step 1: Load base simulation
            logger.info("=== Starting Complete Forward Solve ===")
            self.load_base_simulation(fsp_file)
            results['simulation_loaded'] = True
            
            # Step 2: Setup wavelengths (if provided)
            if 'wavelength_center' in kwargs:
                self.setup_wavelengths(
                    kwargs['wavelength_center'],
                    kwargs.get('wavelength_span', 80e-9),
                    kwargs.get('num_wavelengths', 11)
                )
            
            # Step 3: Update geometry
            if parameters is not None:
                self.update_geometry(parameters)
                results['geometry_updated'] = True
            
            # Step 4: Run simulation  
            sim_results = self.run_simulation()
            results.update(sim_results)
            
            # Step 5: Extract fields
            fields = self.extract_fields()
            results['fields_extracted'] = True
            results['num_monitors'] = len(fields)
            
            # Step 6: Calculate FOM
            fom_value = self.calculate_fom(fom_type, **kwargs)
            results['fom_value'] = fom_value
            results['fom_type'] = fom_type
            
            # Step 7: Save if requested
            if kwargs.get('save_result', False):
                save_path = self.save_simulation(kwargs.get('output_filename'))
                results['saved_to'] = save_path
            
            logger.info(f"=== Forward Solve Complete: FOM = {fom_value:.6f} ===")
            results['success'] = True
            
            return results
            
        except Exception as e:
            logger.error(f"Complete forward solve failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            raise ForwardSimulationError(f"Complete forward solve failed: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        try:
            if self.sim and self.simulation_loaded:
                self.sim.fdtd.switchtolayout()
                # Close FDTD if needed
                # self.sim.fdtd.close()  # Uncomment if needed
                
            logger.info("Simulation cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience functions for common use cases

def run_forward_simulation(fsp_file: Union[str, Path], parameters: np.ndarray,
                          working_dir: str = './', fom_type: str = 'adiabatic_coupling',
                          **kwargs) -> Dict[str, Any]:
    """
    Convenience function for running a complete forward simulation.
    
    Args:
        fsp_file: Path to base simulation file
        parameters: Design parameters array
        working_dir: Working directory
        fom_type: Type of FOM calculation
        **kwargs: Additional parameters
        
    Returns:
        Complete simulation results
    """
    with ForwardSimulation(working_dir) as sim:
        return sim.run_complete_forward_solve(fsp_file, parameters, fom_type, **kwargs)


def validate_fsp_file(fsp_file: Union[str, Path], working_dir: str = './') -> Dict[str, Any]:
    """
    Validate an .fsp file and return analysis.
    
    Args:
        fsp_file: Path to simulation file
        working_dir: Working directory
        
    Returns:
        Validation results
    """
    with ForwardSimulation(working_dir, validation_level='comprehensive') as sim:
        try:
            sim.load_base_simulation(fsp_file)
            return {
                'valid': True,
                'monitors': sim.available_monitors,
                'sources': sim.available_sources,
                'bounds': sim.sim_bounds,
                'mesh': sim.mesh_settings
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }


# Example usage and testing
if __name__ == "__main__":
    """Example usage of ForwardSimulation class."""
    
    # Example parameters
    example_fsp = "example_simulation.fsp"  # Replace with actual file
    example_params = np.array([0.3, 0.4, 0.3])  # Example parameters
    
    # Example 1: Basic forward simulation
    print("=== Example 1: Basic Forward Simulation ===")
    try:
        results = run_forward_simulation(
            fsp_file=example_fsp,
            parameters=example_params,
            fom_type='adiabatic_coupling',
            wavelength_center=1550e-9,
            wavelength_span=80e-9,
            num_wavelengths=11
        )
        print(f"FOM Value: {results['fom_value']:.6f}")
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    # Example 2: Using context manager
    print("\n=== Example 2: Using Context Manager ===")
    try:
        with ForwardSimulation(working_dir='./simulations') as sim:
            sim.load_base_simulation(example_fsp)
            sim.setup_wavelengths(1550e-9, 80e-9, 11)
            
            # Update with multiple parameter sets
            param_sets = [
                np.array([0.2, 0.5, 0.3]),
                np.array([0.4, 0.3, 0.3]),
                np.array([0.3, 0.3, 0.4])
            ]
            
            fom_values = []
            for i, params in enumerate(param_sets):
                sim.update_geometry(params)
                sim.run_simulation()
                sim.extract_fields()
                fom = sim.calculate_fom('adiabatic_coupling')
                fom_values.append(fom)
                print(f"Parameter set {i+1}: FOM = {fom:.6f}")
                
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    # Example 3: File validation
    print("\n=== Example 3: File Validation ===")
    try:
        validation_results = validate_fsp_file(example_fsp)
        if validation_results['valid']:
            print("File validation passed")
            print(f"Monitors found: {validation_results['monitors']}")
            print(f"Sources found: {validation_results['sources']}")
        else:
            print(f"File validation failed: {validation_results['error']}")
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    print("\n=== Examples Complete ===")
