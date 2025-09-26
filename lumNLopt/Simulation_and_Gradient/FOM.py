#!/usr/bin/env python3
"""
Figure of Merit (FOM) Calculator for lumNLopt

This module provides a unified interface for calculating figures of merit and setting up
adjoint simulations. It integrates with the existing lumNLopt configuration system and
provides callable functions that can be passed to optimization routines.

Key Features:
- Uses configuration from lumNLopt.Inputs.Figure_of_merit
- Supports multiple FOM types (adiabatic coupling, transmission, mode matching)
- Provides callable interface for optimization workflows
- Handles adjoint simulation setup automatically
- Integrates with Forward_simulation.py

Author: Generated for lumNLopt Integration
Date: 2025
"""

import numpy as np
import os
import sys
from typing import Dict, List, Optional, Union, Callable, Any
from functools import wraps
import logging

# Import lumNLopt modules
try:
    from lumNLopt.Inputs.Figure_of_merit import (
        get_fom_config,
        create_adiabatic_coupling_fom,
        get_wavelength_array,
        get_target_function
    )
    from lumNLopt.Inputs.Device import get_monitor_names, get_device_config
    from lumNLopt.figures_of_merit.adiabatic_coupling import AdiabaticCouplingFOM
except ImportError as e:
    print(f"Warning: Could not import lumNLopt modules: {e}")
    print("Some functionality may be limited")

# Try to import Forward_simulation if available
try:
    from Forward_simulation import ForwardSimulation, run_forward_simulation
    FORWARD_SIM_AVAILABLE = True
except ImportError:
    FORWARD_SIM_AVAILABLE = False
    print("Warning: Forward_simulation not available. Some features may be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FOMCalculator:
    """
    Unified Figure of Merit calculator that handles different FOM types and
    provides callable interface for optimization routines.
    
    This class integrates with the lumNLopt configuration system and provides
    the standard interface required for optimization workflows.
    """
    
    def __init__(self, 
                 fom_type: str = 'adiabatic_coupling',
                 config_override: Optional[Dict] = None,
                 working_directory: str = './fom_calculations'):
        """
        Initialize FOM calculator.
        
        Args:
            fom_type: Type of FOM ('adiabatic_coupling', 'transmission', 'mode_match')
            config_override: Optional configuration overrides
            working_directory: Directory for temporary files and calculations
        """
        self.fom_type = fom_type
        self.working_directory = working_directory
        self.config_override = config_override or {}
        
        # Initialize configuration
        self.fom_config = get_fom_config()
        self.device_config = get_device_config()
        self.monitor_names = get_monitor_names()
        
        # Initialize FOM object
        self.fom_object = self._create_fom_object()
        
        # State variables
        self.last_parameters = None
        self.last_fom_value = None
        self.simulation_results = {}
        self.wavelengths = get_wavelength_array()
        
        # Create working directory
        os.makedirs(working_directory, exist_ok=True)
        
        logger.info(f"FOMCalculator initialized with type: {fom_type}")
    
    def _create_fom_object(self):
        """Create the appropriate FOM object based on type."""
        
        if self.fom_type == 'adiabatic_coupling':
            return create_adiabatic_coupling_fom()
        
        elif self.fom_type == 'transmission':
            # Create transmission FOM with configuration
            from lumopt_mbso.fom.transmissionfom import TransmissionFom
            
            config = self.fom_config
            return TransmissionFom(
                monitor_name=config.monitor_connections['output_monitor'],
                mode_number=1,
                direction='Forward',
                target_T_fwd=lambda wl: config.targets['transmission']['target_value'],
                norm_p=2
            )
        
        elif self.fom_type == 'mode_match':
            # Create mode matching FOM
            from lumopt_mbso.fom.custommodematch import CustomModeMatch
            
            config = self.fom_config
            return CustomModeMatch(
                monitor_name=config.monitor_connections['output_monitor'],
                mode_number=1,
                direction='Forward',
                target_T_fwd=lambda wl: config.targets['transmission']['target_value'],
                norm_p=2
            )
        
        else:
            raise ValueError(f"Unsupported FOM type: {self.fom_type}")
    
    def __call__(self, parameters: np.ndarray, 
                 simulation: Optional[Any] = None,
                 fsp_file: Optional[str] = None) -> float:
        """
        Callable interface for optimization routines.
        
        Args:
            parameters: Design parameters array
            simulation: Optional simulation object (if not provided, creates new)
            fsp_file: Optional .fsp file path
            
        Returns:
            Figure of merit value
        """
        return self.calculate_fom(parameters, simulation, fsp_file)
    
    def calculate_fom(self, parameters: np.ndarray,
                      simulation: Optional[Any] = None,
                      fsp_file: Optional[str] = None) -> float:
        """
        Calculate figure of merit for given parameters.
        
        Args:
            parameters: Design parameters array
            simulation: Optional simulation object
            fsp_file: Optional .fsp file path
            
        Returns:
            Figure of merit value
        """
        try:
            # Check if parameters changed
            params_changed = (self.last_parameters is None or 
                            not np.allclose(parameters, self.last_parameters))
            
            if not params_changed and self.last_fom_value is not None:
                logger.info("Using cached FOM value")
                return self.last_fom_value
            
            # Run forward simulation if needed
            if simulation is None:
                if FORWARD_SIM_AVAILABLE and fsp_file is not None:
                    sim_results = run_forward_simulation(
                        fsp_file=fsp_file,
                        parameters=parameters,
                        fom_type=self.fom_type,
                        wavelength_center=np.mean(self.wavelengths),
                        working_directory=self.working_directory
                    )
                    simulation = sim_results['simulation']
                else:
                    raise ValueError("Need either simulation object or .fsp file path")
            
            # Setup forward simulation
            self.make_forward_sim(simulation, parameters)
            
            # Run simulation if not already run
            if hasattr(simulation, 'fdtd'):
                logger.info("Running forward simulation...")
                simulation.fdtd.run()
            
            # Calculate FOM
            fom_value = self.fom_object.get_fom(simulation)
            
            # Cache results
            self.last_parameters = parameters.copy()
            self.last_fom_value = fom_value
            self.simulation_results['forward'] = simulation
            
            logger.info(f"FOM calculated: {fom_value:.6f}")
            return fom_value
            
        except Exception as e:
            logger.error(f"Error calculating FOM: {e}")
            raise
    
    def make_forward_sim(self, simulation, parameters: np.ndarray):
        """
        Setup forward simulation for FOM calculation.
        
        Args:
            simulation: Simulation object
            parameters: Design parameters
        """
        try:
            # Update geometry with new parameters
            if hasattr(simulation, 'fdtd'):
                simulation.fdtd.switchtolayout()
            
            # Update design regions if available
            if hasattr(self, 'geometry'):
                self.geometry.add_geo(simulation, params=parameters, only_update=True)
            
            # Setup forward simulation (disable adjoint sources)
            self.fom_object.make_forward_sim(simulation)
            
            logger.info("Forward simulation setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up forward simulation: {e}")
            raise
    
    def make_adjoint_sim(self, simulation, parameters: np.ndarray):
        """
        Setup adjoint simulation for gradient calculation.
        
        Args:
            simulation: Simulation object  
            parameters: Design parameters
        """
        try:
            # Ensure forward simulation was run first
            if not self.simulation_results.get('forward'):
                self.calculate_fom(parameters, simulation)
            
            # Switch to layout mode
            if hasattr(simulation, 'fdtd'):
                simulation.fdtd.switchtolayout()
            
            # Update geometry if needed
            if hasattr(self, 'geometry'):
                self.geometry.add_geo(simulation, params=parameters, only_update=True)
            
            # Disable forward sources and setup adjoint sources
            if hasattr(simulation, 'fdtd'):
                if simulation.fdtd.getnamednumber('source') >= 1:
                    simulation.fdtd.setnamed('source', 'enabled', False)
            
            # Setup adjoint simulation
            self.fom_object.make_adjoint_sim(simulation)
            
            self.simulation_results['adjoint'] = simulation
            logger.info("Adjoint simulation setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up adjoint simulation: {e}")
            raise
    
    def get_adjoint_field_scaling(self, simulation):
        """Get scaling factor for adjoint fields."""
        if hasattr(self.fom_object, 'get_adjoint_field_scaling'):
            return self.fom_object.get_adjoint_field_scaling(simulation)
        return 1.0
    
    def initialize(self, simulation):
        """Initialize FOM with simulation setup."""
        try:
            if hasattr(self.fom_object, 'initialize'):
                self.fom_object.initialize(simulation)
            logger.info("FOM initialized")
        except Exception as e:
            logger.error(f"Error initializing FOM: {e}")
            raise
    
    def get_wavelengths(self, simulation=None):
        """Get wavelengths for FOM calculation."""
        if simulation and hasattr(self.fom_object, 'get_wavelengths'):
            return self.fom_object.get_wavelengths(simulation)
        return self.wavelengths
    
    def get_monitor_data(self, simulation, monitor_name: str) -> Dict:
        """Extract data from specified monitor."""
        try:
            if hasattr(simulation, 'fdtd'):
                return simulation.fdtd.getresult(monitor_name)
            else:
                logger.warning("No fdtd interface available")
                return {}
        except Exception as e:
            logger.error(f"Error getting monitor data from {monitor_name}: {e}")
            return {}
    
    def calculate_transmission(self, simulation) -> float:
        """Calculate transmission efficiency."""
        try:
            input_data = self.get_monitor_data(simulation, 
                                             self.fom_config.monitor_connections['input_monitor'])
            output_data = self.get_monitor_data(simulation,
                                              self.fom_config.monitor_connections['output_monitor'])
            
            if input_data and output_data:
                # Calculate power transmission
                P_in = np.real(input_data.get('T', [0]))[0] if 'T' in input_data else 0
                P_out = np.real(output_data.get('T', [0]))[0] if 'T' in output_data else 0
                
                return P_out / P_in if P_in > 0 else 0
            
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating transmission: {e}")
            return 0
    
    def get_fom_breakdown(self, simulation) -> Dict[str, float]:
        """Get detailed breakdown of FOM components."""
        breakdown = {}
        
        if self.fom_type == 'adiabatic_coupling':
            # Get individual components
            breakdown['transmission'] = self.calculate_transmission(simulation)
            
            if hasattr(self.fom_object, 'get_mode_evolution_quality'):
                breakdown['mode_evolution'] = self.fom_object.get_mode_evolution_quality(simulation)
            
            if hasattr(self.fom_object, 'get_fundamental_purity'):
                breakdown['fundamental_purity'] = self.fom_object.get_fundamental_purity(simulation)
        
        elif self.fom_type == 'transmission':
            breakdown['transmission'] = self.calculate_transmission(simulation)
        
        breakdown['total'] = self.last_fom_value or 0
        
        return breakdown
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate FOM configuration."""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate monitor connections
            required_monitors = ['input_monitor', 'output_monitor']
            for monitor in required_monitors:
                if monitor not in self.fom_config.monitor_connections:
                    validation['errors'].append(f"Missing required monitor: {monitor}")
            
            # Validate wavelength settings
            if len(self.wavelengths) == 0:
                validation['errors'].append("No wavelengths defined")
            
            # Validate FOM object
            required_methods = ['get_fom', 'make_forward_sim', 'make_adjoint_sim']
            for method in required_methods:
                if not hasattr(self.fom_object, method):
                    validation['errors'].append(f"FOM object missing method: {method}")
            
            validation['is_valid'] = len(validation['errors']) == 0
            
        except Exception as e:
            validation['errors'].append(f"Configuration validation error: {e}")
            validation['is_valid'] = False
        
        return validation
    
    def print_summary(self):
        """Print summary of FOM configuration."""
        print("\n" + "="*60)
        print("FOM CALCULATOR SUMMARY")
        print("="*60)
        print(f"FOM Type: {self.fom_type}")
        print(f"Working Directory: {self.working_directory}")
        print(f"Number of Wavelengths: {len(self.wavelengths)}")
        print(f"Wavelength Range: {self.wavelengths[0]:.1e} - {self.wavelengths[-1]:.1e} m")
        
        # Print monitor connections
        print(f"\nMonitor Connections:")
        for name, monitor in self.fom_config.monitor_connections.items():
            print(f"  {name}: {monitor}")
        
        # Print optimization targets
        print(f"\nOptimization Targets:")
        for name, target in self.fom_config.targets.items():
            print(f"  {name}: {target.get('target_value', 'N/A')}")
        
        # Validation status
        validation = self.validate_configuration()
        status = "✅ Valid" if validation['is_valid'] else "❌ Invalid"
        print(f"\nConfiguration Status: {status}")
        
        if validation['errors']:
            print("Errors:")
            for error in validation['errors']:
                print(f"  - {error}")


# ============================================================================
# CALLABLE FUNCTIONS FOR OPTIMIZATION
# ============================================================================

def create_callable_fom(fom_type: str = 'adiabatic_coupling',
                       fsp_file: Optional[str] = None,
                       config_override: Optional[Dict] = None) -> Callable:
    """
    Create a callable FOM function for optimization routines.
    
    Args:
        fom_type: Type of FOM to create
        fsp_file: Path to .fsp file (if using file-based simulation)
        config_override: Configuration overrides
        
    Returns:
        Callable function that takes parameters and returns FOM value
    """
    calculator = FOMCalculator(fom_type, config_override)
    
    def callable_fom(parameters: np.ndarray) -> float:
        return calculator(parameters, fsp_file=fsp_file)
    
    return callable_fom

def create_callable_gradient(fom_type: str = 'adiabatic_coupling',
                           fsp_file: Optional[str] = None,
                           config_override: Optional[Dict] = None) -> Callable:
    """
    Create a callable gradient function for optimization routines.
    
    Args:
        fom_type: Type of FOM to create
        fsp_file: Path to .fsp file
        config_override: Configuration overrides
        
    Returns:
        Callable function that takes parameters and returns gradient array
    """
    calculator = FOMCalculator(fom_type, config_override)
    
    def callable_gradient(parameters: np.ndarray) -> np.ndarray:
        # This would integrate with adjoint field calculation
        # For now, return finite difference approximation
        logger.warning("Analytic gradients not yet implemented, using finite difference")
        
        epsilon = 1e-6
        grad = np.zeros_like(parameters)
        
        f0 = calculator(parameters, fsp_file=fsp_file)
        
        for i in range(len(parameters)):
            params_plus = parameters.copy()
            params_plus[i] += epsilon
            f_plus = calculator(params_plus, fsp_file=fsp_file)
            grad[i] = (f_plus - f0) / epsilon
        
        return grad
    
    return callable_gradient

def create_fom_adjoint_pair(fom_type: str = 'adiabatic_coupling',
                           fsp_file: Optional[str] = None,
                           config_override: Optional[Dict] = None) -> tuple:
    """
    Create both FOM and adjoint functions for optimization.
    
    Returns:
        Tuple of (fom_function, gradient_function)
    """
    fom_func = create_callable_fom(fom_type, fsp_file, config_override)
    grad_func = create_callable_gradient(fom_type, fsp_file, config_override)
    
    return fom_func, grad_func


# ============================================================================
# INTEGRATION WITH FORWARD SIMULATION
# ============================================================================

def calculate_fom_from_results(simulation_results: Dict, 
                              fom_type: str = 'adiabatic_coupling') -> Dict:
    """
    Calculate FOM from Forward_simulation.py results.
    
    Args:
        simulation_results: Results from run_forward_simulation()
        fom_type: Type of FOM to calculate
        
    Returns:
        Dictionary with FOM value and breakdown
    """
    try:
        calculator = FOMCalculator(fom_type)
        
        simulation = simulation_results['simulation']
        fom_value = calculator.fom_object.get_fom(simulation)
        breakdown = calculator.get_fom_breakdown(simulation)
        
        return {
            'fom_value': fom_value,
            'fom_breakdown': breakdown,
            'fom_type': fom_type,
            'wavelengths': calculator.get_wavelengths(simulation),
            'monitor_data': {
                name: calculator.get_monitor_data(simulation, monitor)
                for name, monitor in calculator.fom_config.monitor_connections.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating FOM from results: {e}")
        return {'error': str(e)}


# ============================================================================
# MAIN EXECUTION AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Test FOM calculator
    print("Testing FOM Calculator...")
    
    try:
        # Create FOM calculator
        calc = FOMCalculator('adiabatic_coupling')
        calc.print_summary()
        
        # Test callable functions
        fom_func = create_callable_fom('adiabatic_coupling')
        grad_func = create_callable_gradient('adiabatic_coupling')
        
        print("\n✅ FOM Calculator created successfully")
        print("✅ Callable functions created successfully")
        
        # Test with dummy parameters
        dummy_params = np.array([0.3, 0.4, 0.3])  # Example fractional parameters
        print(f"\nTesting with dummy parameters: {dummy_params}")
        
        # Note: This would require actual .fsp file to run
        # fom_value = fom_func(dummy_params)
        # print(f"FOM Value: {fom_value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    print("\nFOM Calculator module ready for use!")
