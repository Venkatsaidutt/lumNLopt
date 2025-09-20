"""
Monitor Configuration for lumNLopt Edge Coupler Optimization

This file defines all field monitors, mode expansion monitors, and data collection
points required for the adiabatic edge coupler optimization.

Monitors include:
1. Mode expansion monitors for power coupling analysis
2. Slice monitors for mode evolution tracking  
3. Field monitors for gradient calculations
4. Reflection monitors for loss analysis
5. Index monitors for geometry verification
"""

import numpy as np
import lumapi

# Import structure configuration to get geometry info
try:
    from lumNLopt.Inputs.Structure import get_structure_config
    from lumNLopt.Inputs.Figure_of_merit import get_monitor_names, get_wavelength_settings
except ImportError:
    # Fallback if imports not available
    def get_structure_config():
        return None
    def get_monitor_names():
        return {}
    def get_wavelength_settings():
        return {}

# ============================================================================
# MONITOR CONFIGURATION CLASS
# ============================================================================

class MonitorConfig:
    """
    Configuration class for all monitors in the edge coupler optimization.
    Defines positions, settings, and data collection parameters.
    """
    
    def __init__(self):
        # Get configuration from other input files
        self.structure_config = get_structure_config()
        self.monitor_names = get_monitor_names()
        self.wavelength_settings = get_wavelength_settings()
        
        # Initialize monitor configurations
        self.setup_monitor_parameters()
        self.setup_mode_expansion_monitors()
        self.setup_slice_monitors()
        self.setup_field_monitors()
        self.setup_specialized_monitors()
        self.setup_data_collection()
    
    def setup_monitor_parameters(self):
        """Setup global monitor parameters"""
        
        self.global_params = {
            'total_monitors': 20,               # Total number of monitors
            'data_storage_format': 'HDF5',     # Data storage format
            'spatial_interpolation': 'specified', # Spatial interpolation method
            'frequency_points': 11,             # Number of frequency points
            
            # Wavelength settings (from FOM config or defaults)
            'wavelength': {
                'center': self.wavelength_settings.get('center', 1.55e-6),
                'span': self.wavelength_settings.get('span', 0.08e-6),
                'points': self.wavelength_settings.get('num_points', 11)
            },
            
            # Time domain settings
            'time_domain': {
                'auto_shutoff': True,
                'shutoff_level': 1e-5,
                'max_time_steps': 2000
            }
        }
    
    def setup_mode_expansion_monitors(self):
        """Configure mode expansion monitors for power coupling analysis"""
        
        self.mode_expansion_monitors = {}
        
        # ==== INPUT MODE EXPANSION MONITOR ====
        self.mode_expansion_monitors['input_mode_expansion'] = {
            'name': 'input_mode_expansion',
            'type': 'mode_expansion',
            'description': 'Input waveguide mode expansion for power analysis',
            
            'geometry': {
                'orientation': 'x_normal',
                'position': {
                    'x': -5.0e-6,               # In input waveguide
                    'y': 0.0,                   # Centered
                    'z': 110e-9                 # Center of silicon layer
                },
                'span': {
                    'y': 2.5e-6,                # Wide enough to capture mode
                    'z': 1.5e-6                 # Include substrate/cladding overlap
                }
            },
            
            'mode_settings': {
                'mode_calculation': 'user_select',
                'selected_mode_numbers': [1],   # Fundamental TE mode
                'number_of_modes': 5,           # Calculate 5 modes for analysis
                'search_index': 3.2,            # Around silicon effective index
                'mode_selection': 'fundamental',
                'bent_waveguide': False,        # Straight waveguide
                'n_coefficient': 2.8            # Effective index estimate
            },
            
            'data_collection': {
                'record_power': True,
                'record_mode_profiles': True,
                'record_effective_index': True,
                'record_group_velocity': True,
                'record_mode_area': True
            }
        }
        
        # ==== OUTPUT MODE EXPANSION MONITOR ====
        self.mode_expansion_monitors['output_mode_expansion'] = {
            'name': 'output_mode_expansion',
            'type': 'mode_expansion', 
            'description': 'Output waveguide mode expansion for coupling efficiency',
            
            'geometry': {
                'orientation': 'x_normal',
                'position': {
                    'x': 17.5e-6,               # In output waveguide
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'y': 4.0e-6,                # Wider output waveguide
                    'z': 2.0e-6                 # Include mode expansion
                }
            },
            
            'mode_settings': {
                'mode_calculation': 'user_select',
                'selected_mode_numbers': [1, 2], # Include first two modes
                'number_of_modes': 8,           # More modes in wider guide
                'search_index': 2.5,            # Lower effective index
                'mode_selection': 'fundamental',
                'bent_waveguide': False,
                'n_coefficient': 2.3
            },
            
            'data_collection': {
                'record_power': True,
                'record_mode_profiles': True,
                'record_effective_index': True,
                'record_group_velocity': True,
                'record_mode_area': True,
                'record_mode_overlap': True     # For fiber coupling analysis
            }
        }
        
        # ==== FIBER MODE REFERENCE MONITOR ====
        self.mode_expansion_monitors['fiber_reference'] = {
            'name': 'fiber_reference',
            'type': 'mode_expansion',
            'description': 'Reference monitor for fiber mode calculation',
            
            'geometry': {
                'orientation': 'x_normal',
                'position': {
                    'x': 22.0e-6,               # Beyond output waveguide
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'y': 12.0e-6,               # Large area for fiber mode
                    'z': 12.0e-6
                }
            },
            
            'mode_settings': {
                'mode_calculation': 'user_import',  # Import fiber mode profile
                'selected_mode_numbers': [1],
                'number_of_modes': 3,
                'search_index': 1.46,           # Fiber effective index
                'mode_selection': 'user_select',
                'fiber_mode': {
                    'type': 'SMF28',            # Standard single-mode fiber
                    'mode_field_diameter': 10.4e-6,
                    'numerical_aperture': 0.14
                }
            },
            
            'data_collection': {
                'record_power': True,
                'record_mode_profiles': True,
                'record_effective_index': True
            }
        }
    
    def setup_slice_monitors(self):
        """Configure slice monitors for mode evolution analysis"""
        
        self.slice_monitors = {}
        
        # Get design region info from structure config
        if self.structure_config:
            design_geom = self.structure_config.design_region['geometry']
            design_start = design_geom['center_position']['x'] - design_geom['length']/2
            design_end = design_geom['center_position']['x'] + design_geom['length']/2
        else:
            # Fallback values
            design_start = -3.0e-6
            design_end = 15.0e-6
        
        # Number of slice monitors (from FOM config)
        num_slices = 12
        slice_positions = np.linspace(design_start, design_end, num_slices)
        
        # Create slice monitors
        for i, x_pos in enumerate(slice_positions):
            monitor_name = f'mode_slice_{i+1}'
            
            self.slice_monitors[monitor_name] = {
                'name': monitor_name,
                'type': 'field_time',
                'description': f'Slice monitor {i+1} for mode evolution analysis',
                
                'geometry': {
                    'orientation': 'x_normal',
                    'position': {
                        'x': float(x_pos),
                        'y': 0.0,
                        'z': 110e-9
                    },
                    'span': {
                        'y': 6.0e-6,            # Wide enough for mode expansion
                        'z': 2.5e-6            # Include substrate/cladding
                    }
                },
                
                'field_settings': {
                    'record_E': True,
                    'record_H': True,
                    'record_power': True,
                    'spatial_interpolation': 'specified',
                    'override_global_settings': False
                },
                
                'analysis': {
                    'mode_expansion': True,      # Expand fields into modes
                    'fundamental_overlap': True, # Calculate overlap with fundamental
                    'power_flow_direction': True, # Monitor forward/backward power
                    'mode_evolution_tracking': True
                }
            }
    
    def setup_field_monitors(self):
        """Configure field monitors for gradient calculations"""
        
        self.field_monitors = {}
        
        # ==== DESIGN REGION FIELD MONITOR ====
        self.field_monitors['design_region_fields'] = {
            'name': 'design_region_fields',
            'type': 'field_time',
            'description': 'High-resolution field monitor for gradient calculations',
            
            'geometry': {
                'position': {
                    'x': 6.0e-6,                # Center of design region
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'x': 18.0e-6,               # Full design region length
                    'y': 3.2e-6,                # Slightly larger than design width
                    'z': 240e-9                 # Silicon + small margin
                }
            },
            
            'resolution': {
                'mesh_override': True,
                'dx': 15e-9,                    # Fine mesh for gradients
                'dy': 15e-9,
                'dz': 20e-9,
                'spatial_interpolation': 'none' # No interpolation for accuracy
            },
            
            'field_settings': {
                'record_E': True,
                'record_H': True,
                'record_eps': True,             # Permittivity for adjoint method
                'record_power': True,
                'record_conformal_mesh': True   # For interface detection
            },
            
            'optimization': {
                'gradient_calculation': True,
                'adjoint_fields': True,
                'interface_detection': True
            }
        }
        
        # ==== INTERFACE DETECTION MONITOR ====
        self.field_monitors['interface_fields'] = {
            'name': 'interface_fields',
            'type': 'field_time',
            'description': 'Monitor for interface detection and field validation',
            
            'geometry': {
                'position': {
                    'x': 6.0e-6,
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'x': 18.5e-6,               # Slightly larger than design
                    'y': 3.5e-6,
                    'z': 300e-9                 # Include interfaces
                }
            },
            
            'resolution': {
                'mesh_override': True,
                'dx': 25e-9,                    # Medium resolution
                'dy': 25e-9,
                'dz': 25e-9
            },
            
            'field_settings': {
                'record_E': True,
                'record_eps': True,
                'record_conformal_mesh': True,
                'record_material_id': True      # Material assignment verification
            }
        }
    
    def setup_specialized_monitors(self):
        """Configure specialized monitors for specific measurements"""
        
        self.specialized_monitors = {}
        
        # ==== REFLECTION MONITOR ====
        self.specialized_monitors['input_reflection'] = {
            'name': 'input_reflection',
            'type': 'power',
            'description': 'Monitor back-reflection at input',
            
            'geometry': {
                'orientation': 'x_normal',
                'position': {
                    'x': -6.5e-6,               # In input waveguide
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'y': 2.0e-6,
                    'z': 1.0e-6
                }
            },
            
            'measurement': {
                'power_direction': 'backward',   # Monitor backward propagating power
                'reference_direction': 'forward'
            }
        }
        
        # ==== TRANSMISSION MONITOR ====
        self.specialized_monitors['output_transmission'] = {
            'name': 'output_transmission',
            'type': 'power',
            'description': 'Simple transmission measurement',
            
            'geometry': {
                'orientation': 'x_normal',
                'position': {
                    'x': 19.0e-6,               # In output region
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'y': 3.5e-6,
                    'z': 1.5e-6
                }
            },
            
            'measurement': {
                'power_direction': 'forward',
                'reference_direction': 'forward'
            }
        }
        
        # ==== MODE CONTENT ANALYSIS MONITORS ====
        positions = [2.0e-6, 6.0e-6, 10.0e-6, 14.0e-6]  # Key positions in device
        for i, x_pos in enumerate(positions):
            monitor_name = f'mode_content_{i+1}'
            
            self.specialized_monitors[monitor_name] = {
                'name': monitor_name,
                'type': 'mode_expansion',
                'description': f'Mode content analysis at position {i+1}',
                
                'geometry': {
                    'orientation': 'x_normal',
                    'position': {
                        'x': x_pos,
                        'y': 0.0,
                        'z': 110e-9
                    },
                    'span': {
                        'y': 5.0e-6,            # Wide for mode capture
                        'z': 2.0e-6
                    }
                },
                
                'mode_settings': {
                    'mode_calculation': 'user_select',
                    'selected_mode_numbers': [1, 2, 3, 4], # First 4 modes
                    'number_of_modes': 10,
                    'search_index': 2.8
                },
                
                'analysis': {
                    'mode_power_fraction': True,
                    'higher_order_content': True,
                    'radiation_loss': True
                }
            }
        
        # ==== INDEX MONITOR ====
        self.specialized_monitors['geometry_verification'] = {
            'name': 'geometry_verification',
            'type': 'index',
            'description': 'Verify geometry and material assignment',
            
            'geometry': {
                'position': {
                    'x': 6.0e-6,
                    'y': 0.0,
                    'z': 110e-9
                },
                'span': {
                    'x': 19.0e-6,               # Full device
                    'y': 4.0e-6,
                    'z': 400e-9                 # Include all layers
                }
            },
            
            'settings': {
                'record_conformal_mesh': True,
                'record_material_id': True,
                'record_index_profile': True
            }
        }
    
    def setup_data_collection(self):
        """Configure data collection and storage settings"""
        
        self.data_collection = {
            'storage': {
                'base_filename': 'edge_coupler_monitor_data',
                'compression': 'gzip',
                'precision': 'double',
                'format': 'HDF5'
            },
            
            'frequency_domain': {
                'use_wavelength_sweep': True,
                'wavelength_points': self.global_params['wavelength']['points'],
                'center_wavelength': self.global_params['wavelength']['center'],
                'wavelength_span': self.global_params['wavelength']['span']
            },
            
            'time_domain': {
                'auto_shutoff': self.global_params['time_domain']['auto_shutoff'],
                'shutoff_level': self.global_params['time_domain']['shutoff_level'],
                'max_time': self.global_params['time_domain']['max_time_steps']
            },
            
            'post_processing': {
                'mode_overlap_calculation': True,
                'power_flow_analysis': True,
                'field_enhancement_mapping': True,
                'loss_analysis': True
            }
        }
    
    def get_all_monitors(self):
        """Return dictionary of all monitor configurations"""
        
        all_monitors = {}
        
        # Combine all monitor types
        all_monitors.update(self.mode_expansion_monitors)
        all_monitors.update(self.slice_monitors)
        all_monitors.update(self.field_monitors)
        all_monitors.update(self.specialized_monitors)
        
        return all_monitors
    
    def add_monitors_to_simulation(self, fdtd):
        """Add all monitors to Lumerical FDTD simulation"""
        
        print("Adding monitors to Lumerical simulation...")
        
        # Add mode expansion monitors
        for monitor_name, monitor_config in self.mode_expansion_monitors.items():
            self._add_mode_expansion_monitor(fdtd, monitor_config)
        
        # Add slice monitors
        for monitor_name, monitor_config in self.slice_monitors.items():
            self._add_field_time_monitor(fdtd, monitor_config)
        
        # Add field monitors
        for monitor_name, monitor_config in self.field_monitors.items():
            self._add_field_time_monitor(fdtd, monitor_config)
        
        # Add specialized monitors
        for monitor_name, monitor_config in self.specialized_monitors.items():
            if monitor_config['type'] == 'power':
                self._add_power_monitor(fdtd, monitor_config)
            elif monitor_config['type'] == 'mode_expansion':
                self._add_mode_expansion_monitor(fdtd, monitor_config)
            elif monitor_config['type'] == 'index':
                self._add_index_monitor(fdtd, monitor_config)
        
        # Set global simulation settings
        self._set_global_settings(fdtd)
        
        print(f"Added {len(self.get_all_monitors())} monitors to simulation")
    
    def _add_mode_expansion_monitor(self, fdtd, config):
        """Add mode expansion monitor to Lumerical simulation"""
        
        fdtd.addmodeexpansion()
        fdtd.set('name', config['name'])
        
        # Set position and geometry
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        
        # Set span based on orientation
        if geometry['orientation'] == 'x_normal':
            fdtd.set('monitor type', '2D X-normal')
            fdtd.set('y span', span['y'])
            fdtd.set('z span', span['z'])
        elif geometry['orientation'] == 'y_normal':
            fdtd.set('monitor type', '2D Y-normal')
            fdtd.set('x span', span['x'])
            fdtd.set('z span', span['z'])
        elif geometry['orientation'] == 'z_normal':
            fdtd.set('monitor type', '2D Z-normal')
            fdtd.set('x span', span['x'])
            fdtd.set('y span', span['y'])
        
        # Set mode calculation settings
        mode_settings = config['mode_settings']
        fdtd.set('mode calculation', mode_settings['mode_calculation'])
        fdtd.set('selected mode numbers', mode_settings['selected_mode_numbers'])
        fdtd.set('number of modes for expansion', mode_settings['number_of_modes'])
        
        # Set mode search parameters
        fdtd.set('search', 'max index')
        fdtd.set('index', mode_settings['search_index'])
        
        # Set wavelength range
        wl_params = self.global_params['wavelength']
        fdtd.set('center wavelength', wl_params['center'])
        fdtd.set('wavelength span', wl_params['span'])
    
    def _add_field_time_monitor(self, fdtd, config):
        """Add field time monitor to Lumerical simulation"""
        
        fdtd.addtime()
        fdtd.set('name', config['name'])
        
        # Set position and geometry
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        
        # Set monitor type and span
        if geometry.get('orientation') == 'x_normal':
            fdtd.set('monitor type', '2D X-normal')
            fdtd.set('y span', span['y'])
            fdtd.set('z span', span['z'])
        else:
            # 3D monitor
            fdtd.set('monitor type', '3D')
            fdtd.set('x span', span['x'])
            fdtd.set('y span', span['y'])
            fdtd.set('z span', span['z'])
        
        # Set field recording options
        field_settings = config.get('field_settings', {})
        fdtd.set('record E', field_settings.get('record_E', True))
        fdtd.set('record H', field_settings.get('record_H', True))
        fdtd.set('record power', field_settings.get('record_power', True))
        
        # Set spatial interpolation
        spatial_interp = field_settings.get('spatial_interpolation', 'specified')
        fdtd.set('spatial interpolation', spatial_interp)
        
        # Set mesh override if specified
        if 'resolution' in config and config['resolution'].get('mesh_override', False):
            resolution = config['resolution']
            fdtd.set('override global monitor settings', True)
            if 'dx' in resolution:
                fdtd.set('dx', resolution['dx'])
            if 'dy' in resolution:
                fdtd.set('dy', resolution['dy'])
            if 'dz' in resolution:
                fdtd.set('dz', resolution['dz'])
    
    def _add_power_monitor(self, fdtd, config):
        """Add power monitor to Lumerical simulation"""
        
        fdtd.addpower()
        fdtd.set('name', config['name'])
        
        # Set position and geometry
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        
        # Set monitor type and span based on orientation
        if geometry['orientation'] == 'x_normal':
            fdtd.set('monitor type', '2D X-normal')
            fdtd.set('y span', span['y'])
            fdtd.set('z span', span['z'])
        elif geometry['orientation'] == 'y_normal':
            fdtd.set('monitor type', '2D Y-normal')
            fdtd.set('x span', span['x'])
            fdtd.set('z span', span['z'])
        elif geometry['orientation'] == 'z_normal':
            fdtd.set('monitor type', '2D Z-normal')
            fdtd.set('x span', span['x'])
            fdtd.set('y span', span['y'])
    
    def _add_index_monitor(self, fdtd, config):
        """Add index monitor to Lumerical simulation"""
        
        fdtd.addindex()
        fdtd.set('name', config['name'])
        
        # Set position and geometry
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        fdtd.set('x span', span['x'])
        fdtd.set('y span', span['y'])
        fdtd.set('z span', span['z'])
    
    def _set_global_settings(self, fdtd):
        """Set global simulation settings based on monitor configuration"""
        
        time_settings = self.data_collection['time_domain']
        
        # Set auto shutoff settings
        if time_settings['auto_shutoff']:
            fdtd.set('auto shutoff level', time_settings['shutoff_level'])
            fdtd.set('use early shutoff', True)
        
        # Set maximum simulation time
        fdtd.set('simulation time', time_settings['max_time'])
        
        # Set wavelength range for all monitors
        wl_settings = self.data_collection['frequency_domain']
        fdtd.set('center wavelength', wl_settings['center_wavelength'])
        fdtd.set('wavelength span', wl_settings['wavelength_span'])
    
    def validate_monitors(self):
        """Validate monitor configuration for consistency"""
        
        validation_errors = []
        
        # Check that all required monitors exist
        required_monitors = [
            'input_mode_expansion',
            'output_mode_expansion', 
            'design_region_fields'
        ]
        
        all_monitors = self.get_all_monitors()
        for monitor_name in required_monitors:
            if monitor_name not in all_monitors:
                validation_errors.append(f"Missing required monitor: {monitor_name}")
        
        # Check monitor positions are reasonable
        for monitor_name, monitor_config in all_monitors.items():
            if 'geometry' in monitor_config:
                position = monitor_config['geometry'].get('position', {})
                x = position.get('x', 0)
                
                # Check if position is reasonable (within expected device range)
                if abs(x) > 30e-6:  # 30 μm seems unreasonable
                    validation_errors.append(f"Monitor {monitor_name} position seems unreasonable: x={x*1e6:.1f} μm")
        
        # Check slice monitor count matches FOM expectations
        slice_count = len(self.slice_monitors)
        expected_slices = 12  # From FOM config
        if slice_count != expected_slices:
            validation_errors.append(f"Expected {expected_slices} slice monitors, found {slice_count}")
        
        return validation_errors
    
    def print_monitor_summary(self):
        """Print summary of monitor configuration"""
        
        print("\n" + "="*60)
        print("MONITOR CONFIGURATION SUMMARY")
        print("="*60)
        
        # Count monitors by type
        mode_exp_count = len(self.mode_expansion_monitors)
        slice_count = len(self.slice_monitors)
        field_count = len(self.field_monitors)
        specialized_count = len(self.specialized_monitors)
        total_count = mode_exp_count + slice_count + field_count + specialized_count
        
        print(f"\nMonitor Count:")
        print(f"  Mode Expansion: {mode_exp_count}")
        print(f"  Slice Monitors: {slice_count}")
        print(f"  Field Monitors: {field_count}")
        print(f"  Specialized: {specialized_count}")
        print(f"  Total: {total_count}")
        
        # Wavelength settings
        wl_params = self.global_params['wavelength']
        print(f"\nWavelength Settings:")
        print(f"  Center: {wl_params['center']*1e9:.0f} nm")
        print(f"  Span: {wl_params['span']*1e9:.0f} nm")
        print(f"  Points: {wl_params['points']}")
        
        # Validation
        errors = self.validate_monitors()
        if errors:
            print(f"\nValidation Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"\n✅ Monitor configuration is valid!")
        
        print("="*60)


# ============================================================================
# INSTANTIATE CONFIGURATION
# ============================================================================

# Create the global monitor configuration instance
monitor_config = MonitorConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_monitor_config():
    """Return the global monitor configuration instance"""
    return monitor_config

def get_all_monitor_definitions():
    """Return dictionary of all monitor configurations"""
    return monitor_config.get_all_monitors()

def add_monitors_to_simulation(fdtd):
    """Add all monitors to Lumerical FDTD simulation"""
    monitor_config.add_monitors_to_simulation(fdtd)

def get_monitor_names_list():
    """Return list of all monitor names"""
    return list(monitor_config.get_all_monitors().keys())

def get_slice_monitor_positions():
    """Return positions of slice monitors for mode evolution analysis"""
    positions = []
    for monitor_name, monitor_def in monitor_config.slice_monitors.items():
        x_pos = monitor_def['geometry']['position']['x']
        positions.append((monitor_name, x_pos))
    
    return sorted(positions, key=lambda x: x[1])  # Sort by x position

def get_data_collection_settings():
    """Return data collection configuration"""
    return monitor_config.data_collection

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary when run directly
    monitor_config.print_monitor_summary()
    
    # Print detailed monitor information
    print("\nDetailed Monitor Information:")
    print("-" * 40)
    
    all_monitors = get_all_monitor_definitions()
    for monitor_name, monitor_def in all_monitors.items():
        print(f"\n{monitor_name}:")
        print(f"  Type: {monitor_def['type']}")
        print(f"  Description: {monitor_def.get('description', 'N/A')}")
        if 'geometry' in monitor_def:
            pos = monitor_def['geometry']['position']
            print(f"  Position: ({pos['x']*1e6:.1f}, {pos['y']*1e6:.1f}, {pos['z']*1e9:.0f}) μm")
