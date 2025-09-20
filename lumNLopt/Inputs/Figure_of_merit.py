"""
Figure of Merit Configuration for lumNLopt Edge Coupler Optimization

This file defines the optimization objectives, target values, weights, and 
monitor connections for the adiabatic edge coupler optimization.

The configuration here will be used to initialize the AdiabaticCouplingFOM class.
"""

import numpy as np
import lumapi

# ============================================================================
# OPTIMIZATION OBJECTIVES CONFIGURATION
# ============================================================================

class FigureOfMeritConfig:
    """
    Configuration class for edge coupler figure of merit optimization.
    Defines all objectives, targets, and optimization parameters.
    """
    
    def __init__(self):
        # Initialize all FOM parameters
        self.setup_optimization_objectives()
        self.setup_monitor_connections()
        self.setup_target_values()
        self.setup_weights()
        self.setup_arithmetic_progression()
        self.setup_optimization_settings()
    
    def setup_optimization_objectives(self):
        """Define the three main optimization objectives"""
        
        self.objectives = {
            'transmission': {
                'name': 'power_coupling_efficiency',
                'description': 'Power transmission from input waveguide to output',
                'type': 'maximization',
                'priority': 'primary'
            },
            
            'fundamental_purity': {
                'name': 'mode_purity',
                'description': 'Fundamental mode content at output',
                'type': 'maximization', 
                'priority': 'secondary'
            },
            
            'mode_evolution': {
                'name': 'adiabatic_transition_quality',
                'description': 'Smooth mode evolution through device',
                'type': 'arithmetic_progression',
                'priority': 'secondary'
            }
        }
    
    def setup_monitor_connections(self):
        """Define which monitors are used for each objective"""
        
        # Monitor names (must match those defined in Monitors.py)
        self.monitor_connections = {
            'input_monitor': 'input_mode_expansion',
            'output_monitor': 'output_mode_expansion', 
            'slice_monitors': [
                'mode_slice_1', 'mode_slice_2', 'mode_slice_3', 'mode_slice_4',
                'mode_slice_5', 'mode_slice_6', 'mode_slice_7', 'mode_slice_8',
                'mode_slice_9', 'mode_slice_10', 'mode_slice_11', 'mode_slice_12'
            ],
            'reflection_monitor': 'input_reflection',
            'field_monitor': 'design_region_fields'
        }
        
        # Number of slices for mode evolution analysis
        self.num_slices = len(self.monitor_connections['slice_monitors'])
    
    def setup_target_values(self):
        """Define target values for each objective"""
        
        self.targets = {
            'transmission': {
                'target_value': 0.90,           # 90% coupling efficiency (-0.46 dB)
                'acceptable_min': 0.80,         # Minimum acceptable value
                'acceptable_max': 1.0,          # Maximum possible value
                'units': 'fraction'
            },
            
            'fundamental_purity': {
                'target_value': 0.95,           # 95% fundamental mode content
                'acceptable_min': 0.90,         # Minimum acceptable purity
                'acceptable_max': 1.0,          # Pure fundamental mode
                'units': 'fraction'
            },
            
            'mode_evolution': {
                'start_overlap': 0.999,         # 99.9% overlap at input (pure waveguide mode)
                'end_overlap': 0.950,           # 95% overlap at output (some expansion allowed)
                'progression_type': 'arithmetic', # Linear progression
                'deviation_tolerance': 0.05,     # ±5% tolerance from arithmetic progression
                'units': 'overlap_fraction'
            },
            
            'reflection': {
                'max_allowed': 0.02,            # Maximum 2% reflection (-17 dB)
                'target_value': 0.001,          # Target 0.1% reflection (-30 dB)
                'units': 'fraction'
            }
        }
    
    def setup_weights(self):
        """Define relative importance weights for multi-objective optimization"""
        
        self.weights = {
            'transmission': 0.50,               # 50% weight - most important
            'fundamental_purity': 0.25,         # 25% weight - secondary
            'mode_evolution': 0.25,             # 25% weight - secondary
        }
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"Warning: Weights sum to {total_weight:.6f}, should be 1.0")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total_weight
            print("Weights have been normalized")
    
    def setup_arithmetic_progression(self):
        """Configure arithmetic progression for mode evolution"""
        
        self.arithmetic_progression = {
            'enabled': True,
            'start_overlap': self.targets['mode_evolution']['start_overlap'],
            'end_overlap': self.targets['mode_evolution']['end_overlap'],
            'num_points': self.num_slices,
            'tolerance': self.targets['mode_evolution']['deviation_tolerance']
        }
        
        # Calculate target overlaps for each slice
        self.target_overlaps = self.calculate_target_overlap_progression()
    
    def calculate_target_overlap_progression(self):
        """Calculate target overlap values for arithmetic progression"""
        
        start = self.arithmetic_progression['start_overlap']
        end = self.arithmetic_progression['end_overlap']
        num_points = self.arithmetic_progression['num_points']
        
        # Linear progression from start to end
        target_overlaps = np.linspace(start, end, num_points)
        
        return target_overlaps.tolist()
    
    def setup_optimization_settings(self):
        """Configure optimization algorithm settings"""
        
        self.optimization_settings = {
            'algorithm': 'product_weighted',    # Combine objectives via weighted product
            'convergence_criterion': 1e-6,     # FOM change threshold for convergence
            'max_iterations': 100,             # Maximum optimization iterations
            'gradient_method': 'adjoint',       # Use adjoint method for gradients
            'constraint_handling': 'penalty',   # How to handle constraint violations
            
            # Wavelength settings
            'wavelength_range': {
                'center': 1.55e-6,             # C-band center wavelength
                'span': 0.08e-6,               # 80 nm bandwidth
                'num_points': 11                # Number of wavelength points
            },
            
            # Robustness settings
            'robustness_analysis': {
                'enabled': False,               # Enable for robust optimization
                'fabrication_tolerance': 10e-9, # ±10 nm fabrication variation
                'num_samples': 25               # Monte Carlo samples for robustness
            }
        }
    
    def get_fom_parameters(self):
        """Return parameters for AdiabaticCouplingFOM initialization"""
        
        return {
            'input_monitor_name': self.monitor_connections['input_monitor'],
            'output_monitor_name': self.monitor_connections['output_monitor'],
            'slice_monitors': self.monitor_connections['slice_monitors'],
            'num_slices': self.num_slices,
            'target_transmission': self.targets['transmission']['target_value'],
            'overlap_progression_params': {
                'start_wg_overlap': self.targets['mode_evolution']['start_overlap'],
                'end_wg_overlap': self.targets['mode_evolution']['end_overlap'],
                'start_fiber_overlap': self.targets['mode_evolution']['end_overlap'],
                'end_fiber_overlap': self.targets['mode_evolution']['start_overlap']
            },
            'weights': self.weights,
            'norm_p': 2  # Use L2 norm for error calculations
        }
    
    def get_wavelength_array(self):
        """Generate wavelength array for optimization"""
        
        wl_settings = self.optimization_settings['wavelength_range']
        center = wl_settings['center']
        span = wl_settings['span']
        num_points = wl_settings['num_points']
        
        min_wl = center - span/2
        max_wl = center + span/2
        
        return np.linspace(min_wl, max_wl, num_points)
    
    def validate_configuration(self):
        """Validate the FOM configuration for consistency"""
        
        validation_errors = []
        
        # Check weight normalization
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-3:
            validation_errors.append(f"Weights sum to {total_weight:.3f}, should be 1.0")
        
        # Check target values are reasonable
        for obj_name, targets in self.targets.items():
            if 'target_value' in targets:
                target = targets['target_value']
                if target < 0 or target > 1:
                    validation_errors.append(f"{obj_name} target {target} outside [0,1] range")
        
        # Check monitor consistency
        required_monitors = ['input_monitor', 'output_monitor', 'slice_monitors']
        for monitor_type in required_monitors:
            if monitor_type not in self.monitor_connections:
                validation_errors.append(f"Missing {monitor_type} in monitor connections")
        
        # Check arithmetic progression
        if self.arithmetic_progression['start_overlap'] <= self.arithmetic_progression['end_overlap']:
            print("Warning: Start overlap should typically be higher than end overlap for edge couplers")
        
        return validation_errors
    
    def print_configuration_summary(self):
        """Print a summary of the FOM configuration"""
        
        print("\n" + "="*60)
        print("FIGURE OF MERIT CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"\nOptimization Objectives:")
        for obj_name, obj_config in self.objectives.items():
            weight = self.weights.get(obj_name, 0)
            print(f"  {obj_name}: {obj_config['description']} (weight: {weight:.2f})")
        
        print(f"\nTarget Values:")
        for obj_name, targets in self.targets.items():
            if 'target_value' in targets:
                print(f"  {obj_name}: {targets['target_value']:.3f}")
        
        print(f"\nMode Evolution:")
        print(f"  Start overlap: {self.targets['mode_evolution']['start_overlap']:.3f}")
        print(f"  End overlap: {self.targets['mode_evolution']['end_overlap']:.3f}")
        print(f"  Number of slices: {self.num_slices}")
        
        print(f"\nWavelength Range:")
        wl_range = self.optimization_settings['wavelength_range']
        print(f"  Center: {wl_range['center']*1e9:.0f} nm")
        print(f"  Span: {wl_range['span']*1e9:.0f} nm")
        print(f"  Points: {wl_range['num_points']}")
        
        # Validation
        errors = self.validate_configuration()
        if errors:
            print(f"\nValidation Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"\n✅ Configuration is valid!")
        
        print("="*60)


# ============================================================================
# INSTANTIATE CONFIGURATION
# ============================================================================

# Create the global FOM configuration instance
fom_config = FigureOfMeritConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_fom_config():
    """Return the global FOM configuration instance"""
    return fom_config

def create_adiabatic_coupling_fom():
    """Create and return an AdiabaticCouplingFOM instance with the configured parameters"""
    
    from lumNLopt.figures_of_merit.adiabatic_coupling import AdiabaticCouplingFOM
    
    fom_params = fom_config.get_fom_parameters()
    
    return AdiabaticCouplingFOM(**fom_params)

def get_target_function(wavelengths):
    """Generate target transmission function for the given wavelengths"""
    
    target_transmission = fom_config.targets['transmission']['target_value']
    
    # For now, return constant target transmission
    # Can be modified for wavelength-dependent targets
    return np.ones_like(wavelengths) * target_transmission

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
                print(f"  Range: [{targets['acceptable_min']:.3f}, {targets['acceptable_max']:.3f}]")

def get_monitor_names():
    """Return dictionary of monitor names for use by other modules"""
    return fom_config.monitor_connections

def get_wavelength_settings():
    """Return wavelength configuration"""
    return fom_config.optimization_settings['wavelength_range']

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary when run directly
    fom_config.print_configuration_summary()
    print_optimization_objectives()
