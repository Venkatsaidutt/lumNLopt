"""
Figure of Merit Configuration for lumNLopt Edge Coupler Optimization

This file defines the optimization objectives, target values, weights, and 
monitor connections for the adiabatic edge coupler optimization.

The configuration here will be used to initialize the AdiabaticCouplingFOM class.
"""

import numpy as np
import lumapi

# Import from the merged Device.py file (replaces old Structure.py and Monitors.py imports)
from lumNLopt.Inputs.Device import (
    get_device_config, 
    get_monitor_names, 
    get_wavelength_settings,
    get_performance_targets,
    get_fabrication_constraints
)

# ============================================================================
# OPTIMIZATION OBJECTIVES CONFIGURATION
# ============================================================================

class FigureOfMeritConfig:
    """
    Configuration class for edge coupler figure of merit optimization.
    Defines all objectives, targets, and optimization parameters.
    Now properly integrated with Device.py configuration.
    """
    
    def __init__(self):
        # Get device configuration for integration
        self.device_config = get_device_config()
        self.device_monitor_names = get_monitor_names()
        self.device_wavelength_settings = get_wavelength_settings()
        self.device_targets = get_performance_targets()
        
        # Initialize all FOM parameters
        self.setup_optimization_objectives()
        self.setup_monitor_connections()
        self.setup_target_values()
        self.setup_weights()
        self.setup_arithmetic_progression()
        self.setup_optimization_settings()
        
        # Validate configuration
        self.validate_configuration()
    
    def setup_optimization_objectives(self):
        """Define the three main optimization objectives"""
        
        self.objectives = {
            'transmission': {
                'name': 'power_coupling_efficiency',
                'description': 'Power transmission from input waveguide to output',
                'type': 'maximization',
                'priority': 'primary',
                'fom_component': 'transmission_efficiency'
            },
            
            'fundamental_purity': {
                'name': 'mode_purity',
                'description': 'Fundamental mode content at output',
                'type': 'maximization', 
                'priority': 'secondary',
                'fom_component': 'mode_purity'
            },
            
            'mode_evolution': {
                'name': 'adiabatic_transition_quality',
                'description': 'Smooth mode evolution through device',
                'type': 'arithmetic_progression',
                'priority': 'secondary',
                'fom_component': 'mode_evolution'
            },
            
            'reflection_minimization': {
                'name': 'back_reflection_suppression',
                'description': 'Minimize back-reflection at input',
                'type': 'minimization',
                'priority': 'tertiary',
                'fom_component': 'reflection_loss'
            }
        }
    
    def setup_monitor_connections(self):
        """
        Define which monitors are used for each objective.
        Now uses monitor names from Device.py to ensure consistency.
        """
        
        # Get monitor names from Device.py (replaces hardcoded names)
        device_monitors = self.device_monitor_names
        
        self.monitor_connections = {
            # Primary monitors for power coupling
            'input_monitor': device_monitors['input_monitor'],           # 'input_mode_expansion'
            'output_monitor': device_monitors['output_monitor'],         # 'output_mode_expansion'
            'fiber_reference': device_monitors['fiber_reference'],       # 'fiber_mode_reference'
            
            # Slice monitors for mode evolution (use Device.py naming: mode_slice_01, mode_slice_02, etc.)
            'slice_monitors': device_monitors['slice_monitors'],         # ['mode_slice_01', 'mode_slice_02', ...]
            
            # Power and field monitors
            'reflection_monitor': device_monitors['reflection_monitor'], # 'input_reflection'
            'transmission_monitor': device_monitors['transmission_monitor'], # 'output_transmission'
            'field_monitor': device_monitors['field_monitor'],           # 'design_region_fields'
            'gradient_monitor': device_monitors['gradient_monitor'],     # 'design_region_fields'
            
            # Index monitor for geometry verification
            'index_monitor': device_monitors['index_monitor']            # 'geometry_verification'
        }
        
        # Number of slices for mode evolution analysis
        self.num_slices = len(self.monitor_connections['slice_monitors'])
        
        print(f"FOM configured with {self.num_slices} slice monitors for mode evolution analysis")
    
    def setup_target_values(self):
        """
        Define target values for each objective.
        Integrates with Device.py performance targets where available.
        """
        
        # Get performance targets from Device.py
        device_targets = self.device_targets
        
        self.targets = {
            'transmission': {
                'target_value': device_targets.get('coupling_efficiency', 0.90),  # 90% coupling efficiency
                'acceptable_min': 0.80,         # Minimum acceptable value
                'acceptable_max': 1.0,          # Maximum possible value
                'units': 'fraction',
                'weight_in_fom': 0.50          # 50% weight in multi-objective FOM
            },
            
            'fundamental_purity': {
                'target_value': 0.95,           # 95% fundamental mode content
                'acceptable_min': 0.90,         # Minimum acceptable purity
                'acceptable_max': 1.0,          # Pure fundamental mode
                'units': 'fraction',
                'weight_in_fom': 0.25           # 25% weight
            },
            
            'mode_evolution': {
                'start_overlap': 0.999,         # 99.9% overlap at input (pure waveguide mode)
                'end_overlap': 0.950,           # 95% overlap at output (some expansion allowed)
                'progression_type': 'arithmetic', # Linear arithmetic progression
                'deviation_tolerance': 0.05,     # ±5% tolerance from arithmetic progression
                'units': 'overlap_fraction',
                'weight_in_fom': 0.20           # 20% weight
            },
            
            'reflection': {
                'max_allowed': device_targets.get('return_loss', -20),    # -20 dB return loss
                'target_value': -30,            # Target -30 dB return loss (0.1% reflection)
                'target_fraction': 0.001,       # 0.1% reflection as fraction
                'acceptable_max_fraction': 0.01, # 1% max acceptable reflection
                'units': 'dB',
                'weight_in_fom': 0.05           # 5% weight
            }
        }
        
        # Validate weights sum to 1
        total_weight = sum([target.get('weight_in_fom', 0) for target in self.targets.values()])
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Target weights sum to {total_weight:.6f}, normalizing to 1.0")
    
    def setup_weights(self):
        """Define relative importance weights for multi-objective optimization"""
        
        self.weights = {
            'transmission': 0.50,               # 50% weight - most important
            'fundamental_purity': 0.25,         # 25% weight - secondary
            'mode_evolution': 0.20,             # 20% weight - secondary
            'reflection_minimization': 0.05     # 5% weight - tertiary
        }
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {total_weight:.6f}, normalizing...")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total_weight
            print("Weights normalized to sum to 1.0")
    
    def setup_arithmetic_progression(self):
        """Define parameters for arithmetic progression in mode evolution"""
        
        self.arithmetic_progression = {
            # Mode overlap progression parameters
            'start_overlap': self.targets['mode_evolution']['start_overlap'],  # 0.999
            'end_overlap': self.targets['mode_evolution']['end_overlap'],      # 0.950
            'num_steps': self.num_slices,                                      # 12 slices
            
            # Calculate progression parameters
            'step_size': None,              # Will be calculated
            'target_overlaps': None,        # Will be calculated
            
            # Quality assessment
            'deviation_penalty': 2.0,       # Penalty factor for deviations
            'smoothness_weight': 0.3,       # Weight for smoothness vs target matching
            'monotonicity_requirement': True, # Require monotonic progression
        }
        
        # Calculate arithmetic progression
        self._calculate_arithmetic_progression()
    
    def _calculate_arithmetic_progression(self):
        """Calculate the target arithmetic progression for mode overlaps"""
        
        start = self.arithmetic_progression['start_overlap']
        end = self.arithmetic_progression['end_overlap']
        n_steps = self.arithmetic_progression['num_steps']
        
        # Calculate step size for arithmetic progression
        step_size = (end - start) / (n_steps - 1)
        self.arithmetic_progression['step_size'] = step_size
        
        # Generate target overlap array
        target_overlaps = np.linspace(start, end, n_steps)
        self.arithmetic_progression['target_overlaps'] = target_overlaps
        
        print(f"Arithmetic progression: {start:.3f} → {end:.3f} in {n_steps} steps")
        print(f"Step size: {step_size:.4f}")
        print(f"Target overlaps: [{target_overlaps[0]:.3f}, {target_overlaps[1]:.3f}, ..., {target_overlaps[-1]:.3f}]")
    
    def setup_optimization_settings(self):
        """Define optimization algorithm settings and wavelength range"""
        
        # Get wavelength settings from Device.py
        wl_settings = self.device_wavelength_settings
        
        self.optimization_settings = {
            # Wavelength range (from Device.py)
            'wavelength_range': {
                'center': wl_settings['center'],            # 1550 nm
                'span': wl_settings['span'],                # 80 nm
                'num_points': wl_settings['num_points'],    # 11 points
                'target_bandwidth': wl_settings.get('target_bandwidth', 0.06e-6)  # 60 nm
            },
            
            # Multi-objective optimization settings
            'multi_objective': {
                'method': 'weighted_sum',       # Weighted sum approach
                'normalization': 'target_based', # Normalize by target values
                'combination_rule': 'product',   # Product rule for combining objectives
                'adaptive_weights': False       # Fixed weights (no adaptation)
            },
            
            # Algorithm-specific settings
            'algorithm': {
                'type': 'nlopt',               # Use NLopt optimizers
                'primary_algorithm': 'LD_MMA', # Method of Moving Asymptotes
                'secondary_algorithm': 'LD_CCSAQ', # Conservative Convex Separable Approximation
                'max_iterations': 100,         # Maximum optimization iterations
                'tolerance': 1e-6,             # Convergence tolerance
                'constraint_tolerance': 1e-4   # Constraint violation tolerance
            },
            
            # Robustness and uncertainty analysis
            'robustness': {
                'enable_monte_carlo': False,   # Disable by default for speed
                'fabrication_variation': 0.02, # ±2% fabrication variation
                'wavelength_variation': 0.001, # ±1 nm wavelength variation
                'num_samples': 25              # Monte Carlo samples for robustness
            }
        }
    
    def get_fom_parameters(self):
        """
        Return parameters for AdiabaticCouplingFOM initialization.
        This is the main interface for creating the FOM object.
        """
        
        return {
            # Monitor names from Device.py
            'input_monitor_name': self.monitor_connections['input_monitor'],
            'output_monitor_name': self.monitor_connections['output_monitor'],
            'slice_monitors': self.monitor_connections['slice_monitors'],
            
            # Basic parameters
            'num_slices': self.num_slices,
            'target_transmission': self.targets['transmission']['target_value'],
            
            # Arithmetic progression parameters
            'overlap_progression_params': {
                'start_wg_overlap': self.targets['mode_evolution']['start_overlap'],
                'end_wg_overlap': self.targets['mode_evolution']['end_overlap'],
                'start_fiber_overlap': self.targets['mode_evolution']['end_overlap'],    # Reversed
                'end_fiber_overlap': self.targets['mode_evolution']['start_overlap'],    # Reversed
                'progression_type': 'arithmetic',
                'target_overlaps': self.arithmetic_progression['target_overlaps']
            },
            
            # Weights and combination
            'weights': self.weights,
            'norm_p': 2,                       # Use L2 norm for error calculations
            'combination_method': 'weighted_product'  # Product of weighted objectives
        }
    
    def create_adiabatic_coupling_fom(self):
        """
        Create and return an AdiabaticCouplingFOM instance with configured parameters.
        This replaces manual FOM instantiation.
        """
        
        from lumNLopt.figures_of_merit.adiabatic_coupling import AdiabaticCouplingFOM
        
        fom_params = self.get_fom_parameters()
        
        return AdiabaticCouplingFOM(**fom_params)
    
    def get_wavelength_array(self):
        """Generate wavelength array for optimization"""
        
        wl_settings = self.optimization_settings['wavelength_range']
        center = wl_settings['center']
        span = wl_settings['span']
        num_points = wl_settings['num_points']
        
        min_wl = center - span/2
        max_wl = center + span/2
        
        return np.linspace(min_wl, max_wl, num_points)
    
    def get_target_function(self, wavelengths):
        """Generate target transmission function for the given wavelengths"""
        
        target_transmission = self.targets['transmission']['target_value']
        
        # For now, return constant target transmission
        # Can be modified for wavelength-dependent targets
        return np.ones_like(wavelengths) * target_transmission
    
    def validate_configuration(self):
        """Validate the FOM configuration for consistency"""
        
        validation_errors = []
        validation_warnings = []
        
        # Check weight normalization
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-3:
            validation_errors.append(f"Weights sum to {total_weight:.3f}, should be 1.0")
        
        # Check target values are reasonable
        for obj_name, targets in self.targets.items():
            if 'target_value' in targets:
                target = targets['target_value']
                if obj_name != 'reflection' and (target < 0 or target > 1):
                    validation_errors.append(f"Target value for {obj_name} ({target:.3f}) outside [0,1] range")
        
        # Check monitor connections exist in Device.py
        device_monitors = self.device_monitor_names
        for connection_name, monitor_name in self.monitor_connections.items():
            if connection_name == 'slice_monitors':
                # Check slice monitors
                if not isinstance(monitor_name, list):
                    validation_errors.append(f"slice_monitors should be a list, got {type(monitor_name)}")
                elif len(monitor_name) != self.num_slices:
                    validation_errors.append(f"Expected {self.num_slices} slice monitors, got {len(monitor_name)}")
            else:
                # Check single monitors
                if monitor_name not in device_monitors.values():
                    found_in_device = False
                    for key, value in device_monitors.items():
                        if isinstance(value, list) and monitor_name in value:
                            found_in_device = True
                            break
                        elif value == monitor_name:
                            found_in_device = True
                            break
                    
                    if not found_in_device:
                        validation_warnings.append(f"Monitor '{monitor_name}' not found in Device.py monitor names")
        
        # Check arithmetic progression parameters
        start_overlap = self.arithmetic_progression['start_overlap']
        end_overlap = self.arithmetic_progression['end_overlap']
        if start_overlap <= end_overlap:
            validation_warnings.append(f"Start overlap ({start_overlap:.3f}) should be > end overlap ({end_overlap:.3f}) for waveguide-to-fiber transition")
        
        # Check wavelength settings consistency with Device.py
        device_wl = self.device_wavelength_settings
        fom_wl = self.optimization_settings['wavelength_range']
        
        if abs(device_wl['center'] - fom_wl['center']) > 1e-9:
            validation_warnings.append("Wavelength center mismatch between Device.py and FOM config")
        
        # Store validation results
        self.validation_results = {
            'errors': validation_errors,
            'warnings': validation_warnings,
            'is_valid': len(validation_errors) == 0
        }
        
        # Print validation summary
        if validation_errors:
            print("❌ FOM CONFIGURATION ERRORS:")
            for error in validation_errors:
                print(f"  - {error}")
        
        if validation_warnings:
            print("⚠️  FOM CONFIGURATION WARNINGS:")
            for warning in validation_warnings:
                print(f"  - {warning}")
        
        if not validation_errors and not validation_warnings:
            print("✅ FOM configuration is valid!")
        
        return self.validation_results['is_valid']
    
    def print_configuration_summary(self):
        """Print comprehensive FOM configuration summary"""
        
        print("\n" + "="*70)
        print("FIGURE OF MERIT CONFIGURATION SUMMARY")
        print("="*70)
        
        # Objectives
        print(f"\nOptimization Objectives ({len(self.objectives)}):")
        for obj_name, obj_config in self.objectives.items():
            weight = self.weights.get(obj_name, 0)
            print(f"  {obj_name}: {obj_config['description']} (weight: {weight:.2f})")
        
        # Target values
        print(f"\nTarget Values:")
        for target_name, target_config in self.targets.items():
            if 'target_value' in target_config:
                print(f"  {target_name}: {target_config['target_value']:.3f} {target_config.get('units', '')}")
        
        # Mode evolution
        progression = self.arithmetic_progression
        print(f"\nMode Evolution (Arithmetic Progression):")
        print(f"  Start overlap: {progression['start_overlap']:.3f}")
        print(f"  End overlap: {progression['end_overlap']:.3f}")
        print(f"  Step size: {progression['step_size']:.4f}")
        print(f"  Number of slices: {progression['num_steps']}")
        
        # Monitor connections
        print(f"\nMonitor Connections:")
        for connection_name, monitor_name in self.monitor_connections.items():
            if isinstance(monitor_name, list):
                print(f"  {connection_name}: {len(monitor_name)} monitors ({monitor_name[0]}, ..., {monitor_name[-1]})")
            else:
                print(f"  {connection_name}: {monitor_name}")
        
        # Wavelength settings
        wl = self.optimization_settings['wavelength_range']
        print(f"\nWavelength Settings:")
        print(f"  Center: {wl['center']*1e9:.0f} nm")
        print(f"  Span: {wl['span']*1e9:.0f} nm")
        print(f"  Points: {wl['num_points']}")
        
        # Validation
        if hasattr(self, 'validation_results'):
            results = self.validation_results
            print(f"\nValidation:")
            print(f"  Status: {'✅ Valid' if results['is_valid'] else '❌ Invalid'}")
            print(f"  Errors: {len(results['errors'])}")
            print(f"  Warnings: {len(results['warnings'])}")
        
        print("="*70)
    
    def get_monitor_requirements(self):
        """Return monitor requirements for integration with Device.py"""
        
        return {
            'required_monitors': list(self.monitor_connections.keys()),
            'slice_monitor_count': self.num_slices,
            'monitor_types': {
                'mode_expansion': ['input_monitor', 'output_monitor', 'fiber_reference'],
                'field_time': ['slice_monitors', 'field_monitor', 'gradient_monitor'],
                'power': ['reflection_monitor', 'transmission_monitor'],
                'index': ['index_monitor']
            }
        }


# ============================================================================
# INSTANTIATE CONFIGURATION
# ============================================================================

# Create the global FOM configuration instance
fom_config = FigureOfMeritConfig()

# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL ACCESS
# ============================================================================

def get_fom_config():
    """Return the global FOM configuration instance"""
    return fom_config

def create_adiabatic_coupling_fom():
    """Create and return an AdiabaticCouplingFOM instance with the configured parameters"""
    return fom_config.create_adiabatic_coupling_fom()

def get_target_function(wavelengths):
    """Generate target transmission function for the given wavelengths"""
    return fom_config.get_target_function(wavelengths)

def get_wavelength_array():
    """Get wavelength array for optimization"""
    return fom_config.get_wavelength_array()

def print_optimization_objectives():
    """Print detailed information about optimization objectives"""
    
    print("\n" + "="*50)
    print("OPTIMIZATION OBJECTIVES DETAILS")
    print("="*50)
    
    for obj_name, obj_config in fom_config.objectives.items():
        print(f"\n{obj_name.upper()}:")
        print(f"  Description: {obj_config['description']}")
        print(f"  Type: {obj_config['type']}")
        print(f"  Priority: {obj_config['priority']}")
        print(f"  Weight: {fom_config.weights[obj_name]:.2f}")
        
        if obj_name in fom_config.targets:
            targets = fom_config.targets[obj_name]
            if 'target_value' in targets:
                print(f"  Target: {targets['target_value']:.3f}")
                if 'acceptable_min' in targets and 'acceptable_max' in targets:
                    print(f"  Range: [{targets['acceptable_min']:.3f}, {targets['acceptable_max']:.3f}]")

def get_monitor_names_for_device():
    """Return monitor names and requirements for Device.py integration"""
    return fom_config.monitor_connections

def get_monitor_requirements():
    """Return monitor requirements for Device.py validation"""
    return fom_config.get_monitor_requirements()

def validate_fom_device_integration():
    """Validate that FOM configuration is compatible with Device.py"""
    return fom_config.validate_configuration()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary when run directly
    fom_config.print_configuration_summary()
    
    # Print detailed objective information
    print_optimization_objectives()
    
    # Test FOM creation
    print("\nTesting FOM Creation:")
    try:
        test_fom = create_adiabatic_coupling_fom()
        print("✅ AdiabaticCouplingFOM created successfully")
    except Exception as e:
        print(f"❌ FOM creation failed: {e}")
    
    # Print integration status
    print(f"\nDevice.py Integration:")
    print(f"  Monitor connections: {len(fom_config.monitor_connections)} defined")
    print(f"  Slice monitors: {fom_config.num_slices}")
    print(f"  Wavelength sync: {'✅ Synced' if fom_config.validation_results['is_valid'] else '❌ Issues'}")
