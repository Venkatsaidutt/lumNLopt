"""
Device Configuration for lumNLopt Edge Coupler Optimization

This file combines structure and monitor configuration into a single device definition.
Contains:
1. Fixed geometry (substrate, cladding, waveguides)
2. Optimizable design region (materials, dimensions, topology)
3. All monitors (mode expansion, field, slice, specialized)
4. Simulation domain and mesh settings
5. FSP file management and output settings
6. Material definitions using Lumerical's most generalized format

"""

import numpy as np
import lumapi

# ============================================================================
# DEVICE CONFIGURATION CLASS
# ============================================================================

class DeviceConfig:
    """
    Unified device configuration combining structure and monitor definitions.
    Central configuration point for all device parameters, simulation settings,
    and data collection requirements.
    """
    
    def __init__(self):
        # Initialize all device parameters
        self.setup_device_specifications()
        self.setup_material_definitions()
        self.setup_fixed_geometry()
        self.setup_design_region()
        self.setup_simulation_domain()
        self.setup_mesh_settings()
        self.setup_monitor_configuration()
        self.setup_source_configuration()
        self.setup_output_settings()
        self.setup_monitor_names()
        
        # Validate configuration consistency
        self.validate_device_configuration()
    
    def setup_device_specifications(self):
        """Define basic device parameters and operating conditions"""
        
        self.device_specs = {
            # ==== DEVICE TYPE AND PLATFORM ====
            'device_type': 'adiabatic_edge_coupler',
            'platform': 'SOI',  # Silicon-on-Insulator
            'description': 'Adiabatic edge coupler for fiber-chip coupling',
            
            # ==== OPERATING WAVELENGTH ====
            'wavelength': {
                'center': 1.55e-6,              # 1550 nm center wavelength
                'span': 0.08e-6,                # 80 nm span (1510-1590 nm)
                'num_points': 11,               # Frequency points for simulation
                'target_bandwidth': 0.06e-6     # 60 nm target bandwidth
            },
            
            # ==== FABRICATION CONSTRAINTS ====
            'fabrication': {
                'min_feature_size': 120e-9,     # 120 nm minimum feature size
                'min_gap_size': 100e-9,         # 100 nm minimum gap
                'etch_depth_tolerance': 5e-9,   # ±5 nm etch depth variation
                'sidewall_angle': 85,           # 85° sidewall angle (degrees)
                'line_edge_roughness': 3e-9     # 3 nm LER
            },
            
            # ==== PERFORMANCE TARGETS ====
            'targets': {
                'coupling_efficiency': 0.90,    # 90% coupling efficiency target
                'return_loss': -20,             # -20 dB return loss (1% reflection)
                'bandwidth_3dB': 0.06e-6,       # 60 nm 3dB bandwidth
                'polarization_dependence': 0.5  # <0.5 dB PDL
            }
        }
    
    def setup_material_definitions(self):
        """
        Define materials using Lumerical's most generalized format.
        Supports wavelength-dependent, anisotropic, and dispersive materials.
        """
        
        self.materials = {
            # ==== SILICON (HIGH INDEX MATERIAL) ====
            'silicon_anisotropic': {
                'name': 'Silicon_optimized',
                'type': 'dispersive',           # Full dispersive model
                'model': 'user_defined',        # Custom material model
                
                # Anisotropic properties (most general form)
                'anisotropy': 1,                # Enable anisotropic mode
                'anisotropy_type': 'diagonal',  # Diagonal tensor
                
                # Wavelength-dependent refractive indices (functions)
                'index_functions': {
                    'n_xx': lambda wl: 3.476 + 0.012 * (wl*1e6 - 1.55),  # In-plane x
                    'n_yy': lambda wl: 3.476 + 0.012 * (wl*1e6 - 1.55),  # In-plane y  
                    'n_zz': lambda wl: 3.488 + 0.015 * (wl*1e6 - 1.55),  # Out-of-plane z
                },
                
                # Constant values for initialization (will be overridden by functions)
                'index_xx': 3.476,
                'index_yy': 3.476,
                'index_zz': 3.488,
                
                # Permittivity tensor (calculated from indices)
                'permittivity_tensor': {
                    'eps_xx': 12.11,            # n_xx^2 
                    'eps_yy': 12.11,            # n_yy^2
                    'eps_zz': 12.25,            # n_zz^2
                    'eps_xy': 0.0,              # Off-diagonal terms
                    'eps_xz': 0.0,
                    'eps_yz': 0.0
                },
                
                # Mesh and simulation properties
                'mesh_order': 2,                # High priority for meshing
                'color': [0.7, 0.1, 0.1, 0.8],  # Dark red for visualization
                'thermal_conductivity': 148,    # W/m/K (for thermal simulations)
                'density': 2329                 # kg/m³
            },
            
            # ==== SILICON DIOXIDE (LOW INDEX MATERIAL) ====
            'silica_isotropic': {
                'name': 'SiO2_cladding',
                'type': 'standard',             # Standard Lumerical material
                'model': 'sellmeier',           # Sellmeier dispersion model
                
                # Isotropic properties
                'anisotropy': 0,                # Disable anisotropic mode
                
                # Sellmeier coefficients for SiO2
                'sellmeier_coefficients': {
                    'B1': 0.696166300,
                    'B2': 0.407942600, 
                    'B3': 0.897479400,
                    'C1': 4.67914826e-3,
                    'C2': 1.35120631e-2,
                    'C3': 97.9340025
                },
                
                # Nominal refractive index
                'index': 1.444,                 # At 1550 nm
                'permittivity': 2.085,          # eps = n^2
                
                # Material properties
                'mesh_order': 3,                # Lower priority than silicon
                'color': [0.7, 0.7, 0.9, 0.3],  # Light blue, transparent
                'thermal_conductivity': 1.4,    # W/m/K
                'density': 2203                 # kg/m³
            },
            
            # ==== SUBSTRATE MATERIAL ====
            'silicon_substrate': {
                'name': 'Si_substrate',
                'type': 'standard',
                'model': 'constant',            # Constant index (substrate)
                
                'anisotropy': 0,
                'index': 3.476,                 # Silicon index
                'permittivity': 12.11,
                
                'mesh_order': 4,                # Lowest priority
                'color': [0.3, 0.3, 0.3, 0.8],  # Dark gray
            },
            
            # ==== AIR (BACKGROUND) ====
            'air': {
                'name': 'Air',
                'type': 'standard',
                'model': 'constant',
                
                'anisotropy': 0,
                'index': 1.0,
                'permittivity': 1.0,
                
                'mesh_order': 5,                # Background priority
                'color': [1.0, 1.0, 1.0, 0.0]   # Transparent
            }
        }
    
    def setup_fixed_geometry(self):
        """Define fixed geometry components that don't change during optimization"""
        
        self.fixed_geometry = {
            # ==== SUBSTRATE LAYER ====
            'substrate': {
                'material': 'silicon_substrate',
                'geometry': {
                    'type': 'rectangle',
                    'x_span': 35e-6,            # Full simulation width
                    'y_span': 12e-6,            # Full simulation height  
                    'z_span': 2e-6,             # 2 μm thick substrate
                    'center_position': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': -1.11e-6              # Below buried oxide
                    }
                },
                'description': 'Silicon substrate'
            },
            
            # ==== BURIED OXIDE LAYER ====
            'buried_oxide': {
                'material': 'silica_isotropic',
                'geometry': {
                    'type': 'rectangle',
                    'x_span': 35e-6,
                    'y_span': 12e-6,
                    'z_span': 2e-6,             # 2 μm BOX layer
                    'center_position': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': -0.11e-6              # Below silicon layer
                    }
                },
                'description': 'Buried oxide (BOX) layer'
            },
            
            # ==== INPUT WAVEGUIDE ====
            'input_waveguide': {
                'material': 'silicon_anisotropic',
                'geometry': {
                    'type': 'rectangle',
                    'x_span': 8e-6,             # 8 μm long input section
                    'y_span': 450e-9,           # 450 nm width (single-mode)
                    'z_span': 220e-9,           # 220 nm thickness
                    'center_position': {
                        'x': -13.5e-6,            # Left side of simulation
                        'y': 0.0,
                        'z': 110e-9               # Center of silicon layer
                    }
                },
                'description': 'Single-mode input waveguide'
            },
            
            # ==== OUTPUT WAVEGUIDE ====
            'output_waveguide': {
                'material': 'silicon_anisotropic',  
                'geometry': {
                    'type': 'rectangle',
                    'x_span': 5e-6,             # 5 μm long output section
                    'y_span': 2e-6,             # 2 μm width (multimode/tapered)
                    'z_span': 220e-9,           # 220 nm thickness
                    'center_position': {
                        'x': 14.5e-6,             # Right side of simulation
                        'y': 0.0,
                        'z': 110e-9
                    }
                },
                'description': 'Tapered output waveguide for fiber coupling'
            },
            
            # ==== TOP CLADDING ====
            'top_cladding': {
                'material': 'silica_isotropic',
                'geometry': {
                    'type': 'rectangle',
                    'x_span': 35e-6,
                    'y_span': 12e-6,
                    'z_span': 2e-6,             # 2 μm top cladding
                    'center_position': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': 1.22e-6               # Above silicon layer
                    }
                },
                'description': 'Silicon dioxide top cladding'
            },
            
            # ==== SIDE CLADDING (LEFT) ====
            'side_cladding_left': {
                'material': 'silica_isotropic',
                'geometry': {
                    'type': 'rectangle',
                    'x_span': 35e-6,
                    'y_span': 4.775e-6,         # Fill to left boundary
                    'z_span': 300e-9,           # Cover waveguide height
                    'center_position': {
                        'x': 0.0,
                        'y': -3.6125e-6,          # Left side
                        'z': 110e-9
                    }
                },
                'description': 'Left side oxide cladding'
            },
            
            # ==== SIDE CLADDING (RIGHT) ====
            'side_cladding_right': {
                'material': 'silica_isotropic',
                'geometry': {
                    'type': 'rectangle', 
                    'x_span': 35e-6,
                    'y_span': 4.775e-6,         # Fill to right boundary
                    'z_span': 300e-9,
                    'center_position': {
                        'x': 0.0,
                        'y': 3.6125e-6,           # Right side
                        'z': 110e-9
                    }
                },
                'description': 'Right side oxide cladding'
            }
        }
    
    def setup_design_region(self):
        """Define the optimizable design region and its parameters"""
        
        self.design_region = {
            # ==== DESIGN REGION GEOMETRY ====
            'geometry': {
                'type': 'rectangle',
                'length': 18e-6,                # 18 μm long design region
                'width': 3e-6,                  # 3 μm wide design region  
                'thickness': 220e-9,            # 220 nm thick (SOI)
                'center_position': {
                    'x': 0.0,                   # Center of simulation
                    'y': 0.0,                   # Centered vertically
                    'z': 110e-9                 # Center of silicon layer
                }
            },
            
            # ==== RECTANGLE CLUSTERING PARAMETERS ====
            'topology': {
                'optimization_type': 'rectangle_clustering',
                'min_feature_size': 120e-9,     # From fabrication constraints
                'max_rectangles': 50,           # Maximum number of rectangles
                'clustering_direction': 'x',     # Primary clustering direction
                'material_order': 'alternating', # Alternating high/low index
                
                # Parameter bounds for optimization
                'bounds': {
                    'min_fraction': 0.008,      # Minimum width fraction (0.8%)
                    'max_fraction': 0.20,       # Maximum width fraction (20%)
                    'sum_constraint': 1.0       # Fractions must sum to 1
                },
                
                # Initial parameter distribution
                'initialization': {
                    'method': 'smooth_taper',   # Smooth transition from narrow to wide
                    'taper_ratio': 3.0,         # End width / start width ratio
                    'smoothing_factor': 0.8,    # Smoothness of transition
                    'randomization': 0.05       # 5% random variation
                }
            },
            
            # ==== DESIGN REGION MATERIALS ====
            'materials': {
                'high_index': 'silicon_anisotropic',  # Use defined materials
                'low_index': 'silica_isotropic',
                'background': 'air'
            }
        }
    
    def setup_simulation_domain(self):
        """Define simulation domain, boundaries, and solver settings"""
        
        self.simulation_domain = {
            # ==== SIMULATION DIMENSIONS ====
            'dimensions': {
                'x_span': 35e-6,                # 35 μm x-direction
                'y_span': 12e-6,                # 12 μm y-direction
                'z_span': 6e-6,                 # 6 μm z-direction (include substrate/cladding)
                'center': {
                    'x': 0.0,                   # Centered on design region
                    'y': 0.0,                   # Centered vertically
                    'z': 0.0                    # Centered on silicon layer
                }
            },
            
            # ==== BOUNDARY CONDITIONS ====
            'boundaries': {
                'x_min': 'PML',                 # Perfectly Matched Layer
                'x_max': 'PML',
                'y_min': 'PML',
                'y_max': 'PML', 
                'z_min': 'PML',
                'z_max': 'PML'
            },
            
            # ==== PML SETTINGS ====
            'pml': {
                'layers': 12,                   # 12 PML layers
                'conductivity': {
                    'kappa': 2.0,              # PML conductivity scaling
                    'sigma': 0.8,              # PML conductivity factor
                    'polynomial': 3,           # Polynomial grading
                    'alpha': 0.0               # CFS alpha factor
                }
            },
            
            # ==== SOLVER SETTINGS ====
            'solver': {
                'type': 'FDTD',                # Finite-difference time-domain
                'auto_shutoff': True,          # Enable auto shutoff
                'shutoff_level': 1e-5,         # Auto shutoff level
                'max_time_steps': 2000,        # Maximum time steps
                'dt_stability_factor': 0.99,   # Time step stability factor
                'pml_same_on_all_boundaries': True
            }
        }
    
    def setup_mesh_settings(self):
        """Define mesh configuration for accurate simulations"""
        
        self.mesh = {
            # ==== GLOBAL MESH SETTINGS ====
            'global': {
                'accuracy': 4,                  # Mesh accuracy level (1-8)
                'max_mesh_step': 50e-9,         # 50 nm maximum mesh step
                'min_mesh_step': 10e-9,         # 10 nm minimum mesh step
                'mesh_type': 'auto',            # Automatic mesh generation
                'mesh_refinement': 'conformal'  # Conformal mesh for interfaces
            },
            
            # ==== REGION-SPECIFIC MESH OVERRIDES ====
            'overrides': {
                'design_region': {
                    'enabled': True,
                    'max_mesh_step': 15e-9,     # Fine mesh in design region
                    'min_mesh_step': 8e-9,      # Very fine minimum
                    'mesh_type': 'uniform',     # Uniform mesh for consistency
                    'override_x': True,
                    'override_y': True,
                    'override_z': True
                },
                
                'waveguides': {
                    'enabled': True,
                    'max_mesh_step': 25e-9,     # Medium mesh in waveguides
                    'min_mesh_step': 12e-9,
                    'mesh_type': 'auto'         # Auto mesh for efficiency
                },
                
                'cladding': {
                    'enabled': False,           # Use global mesh in cladding
                    'max_mesh_step': 60e-9,     # Coarser mesh allowed
                    'min_mesh_step': 30e-9
                }
            },
            
            # ==== MESH QUALITY SETTINGS ====
            'quality': {
                'aspect_ratio_limit': 5.0,      # Maximum element aspect ratio
                'smoothing_iterations': 3,       # Mesh smoothing passes
                'quality_threshold': 0.3,        # Minimum mesh quality
                'adaptive_refinement': True      # Enable adaptive refinement
            }
        }
    
    def setup_monitor_configuration(self):
        """Define all monitors for data collection and analysis"""
        
        # ==== MODE EXPANSION MONITORS ====
        self.mode_expansion_monitors = {
            'input_mode_expansion': {
                'type': 'mode_expansion',
                'description': 'Input waveguide mode expansion for power analysis',
                'geometry': {
                    'orientation': 'x_normal',
                    'position': {
                        'x': -5.0e-6,               # In input waveguide
                        'y': 0.0,
                        'z': 110e-9
                    },
                    'span': {
                        'y': 2.5e-6,                # Wide enough to capture mode
                        'z': 1.5e-6
                    }
                },
                'mode_settings': {
                    'mode_calculation': 'user_select',
                    'selected_mode_numbers': [1],   # Fundamental TE mode
                    'number_of_modes': 5,           # Calculate 5 modes
                    'search_index': 3.2,            # Around silicon effective index
                    'bent_waveguide': False
                }
            },
            
            'output_mode_expansion': {
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
                        'z': 2.0e-6
                    }
                },
                'mode_settings': {
                    'mode_calculation': 'user_select',
                    'selected_mode_numbers': [1, 2], # Include first two modes
                    'number_of_modes': 8,           # More modes in wider guide
                    'search_index': 2.5,            # Lower effective index
                    'bent_waveguide': False
                }
            },
            
            'fiber_mode_reference': {
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
                    'fiber_mode': {
                        'type': 'SMF28',            # Standard single-mode fiber
                        'mode_field_diameter': 10.4e-6,
                        'numerical_aperture': 0.14
                    }
                }
            }
        }
        
        # ==== SLICE MONITORS FOR MODE EVOLUTION ====
        self.slice_monitors = {}
        design_start = self.design_region['geometry']['center_position']['x'] - \
                      self.design_region['geometry']['length']/2
        design_end = self.design_region['geometry']['center_position']['x'] + \
                    self.design_region['geometry']['length']/2
        
        num_slices = 12
        slice_positions = np.linspace(design_start, design_end, num_slices)
        
        for i, x_pos in enumerate(slice_positions):
            monitor_name = f'mode_slice_{i+1:02d}'  # e.g., mode_slice_01, mode_slice_02
            
            self.slice_monitors[monitor_name] = {
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
                        'y': 6.0e-6,                # Wide enough for mode expansion
                        'z': 2.5e-6                 # Include substrate/cladding
                    }
                },
                'field_settings': {
                    'record_E': True,
                    'record_H': True,
                    'record_power': True,
                    'spatial_interpolation': 'specified'
                }
            }
        
        # ==== FIELD MONITORS ====
        self.field_monitors = {
            'design_region_fields': {
                'type': 'field_time',
                'description': 'High-resolution field monitor for gradient calculations',
                'geometry': {
                    'position': {
                        'x': 0.0,                   # Center of design region
                        'y': 0.0,
                        'z': 110e-9
                    },
                    'span': {
                        'x': 18.2e-6,               # Slightly larger than design region
                        'y': 3.2e-6,
                        'z': 240e-9
                    }
                },
                'resolution': {
                    'mesh_override': True,
                    'dx': 15e-9,                    # Fine mesh for gradients
                    'dy': 15e-9,
                    'dz': 20e-9,
                    'spatial_interpolation': 'none'
                },
                'field_settings': {
                    'record_E': True,
                    'record_H': True,
                    'record_eps': True,             # For adjoint method
                    'record_power': True,
                    'record_conformal_mesh': True
                }
            },
            
            'interface_detection_fields': {
                'type': 'field_time',
                'description': 'Monitor for interface detection and validation',
                'geometry': {
                    'position': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': 110e-9
                    },
                    'span': {
                        'x': 18.5e-6,
                        'y': 3.5e-6,
                        'z': 300e-9
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
                    'record_material_id': True
                }
            }
        }
        
        # ==== SPECIALIZED MONITORS ====
        self.specialized_monitors = {
            'input_reflection': {
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
                    'power_direction': 'backward',   # Monitor backward power
                    'reference_direction': 'forward'
                }
            },
            
            'output_transmission': {
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
            },
            
            'geometry_verification': {
                'type': 'index',
                'description': 'Verify geometry and material assignment',
                'geometry': {
                    'position': {
                        'x': 0.0,
                        'y': 0.0,
                        'z': 110e-9
                    },
                    'span': {
                        'x': 19.0e-6,               # Full device
                        'y': 4.0e-6,
                        'z': 400e-9
                    }
                },
                'settings': {
                    'record_conformal_mesh': True,
                    'record_material_id': True,
                    'record_index_profile': True
                }
            }
        }
    
    def setup_source_configuration(self):
        """Define comprehensive source configuration for FDTD simulation"""
        
        self.source_config = {
            # ==== SOURCE TYPE AND BASIC SETTINGS ====
            'source_type': 'mode',              # 'mode', 'plane_wave', 'gaussian', 'dipole'
            'name': 'source',                   # Source object name
            'enabled': True,                    # Enable source by default
            
            # ==== SOURCE POSITION AND ORIENTATION ====
            'geometry': {
                'position': {
                    'x': -7.0e-6,               # Position in input waveguide
                    'y': 0.0,                   # Centered
                    'z': 110e-9                 # Center of silicon layer
                },
                'span': {
                    'y': 2.0e-6,                # Wide enough to capture mode
                    'z': 1.0e-6                 # Include waveguide height
                },
                'orientation': 'x_normal',       # Normal to x-plane (propagation in +x)
                'injection_axis': 'x'           # Injection direction
            },
            
            # ==== WAVELENGTH AND FREQUENCY SETTINGS ====
            'wavelength': {
                'center': self.device_specs['wavelength']['center'],    # From device specs
                'span': self.device_specs['wavelength']['span'],        # From device specs
                'frequency_points': self.device_specs['wavelength']['num_points'],
                'use_wavelength_sweep': True,   # Enable wavelength sweep
                'optimization_range': True      # Use for optimization
            },
            
            # ==== MODE SOURCE SPECIFIC SETTINGS ====
            'mode_settings': {
                'mode_selection': 'fundamental TE mode',  # Select fundamental TE mode
                'mode_calculation': 'auto',     # Auto calculate modes
                'number_of_modes': 5,           # Calculate first 5 modes
                'search_index': 3.2,            # Search around Si effective index
                'bent_waveguide': False,        # Straight waveguide
                'mode_number': 1,               # Use mode #1 (fundamental)
                'effective_index_calculation': True,
                'direction': 'forward'          # Forward propagation (+x direction)
            },
            
            # ==== POLARIZATION AND AMPLITUDE ====
            'excitation': {
                'polarization': 'TE',           # TE polarization (Ey dominant)
                'amplitude': 1.0,               # Source amplitude
                'phase': 0.0,                   # Phase in degrees
                'power': 1.0,                   # Normalized power
                'field_component': 'Ey'         # Primary field component
            },
            
            # ==== TIME DOMAIN SETTINGS ====
            'time_domain': {
                'pulse_type': 'continuous',     # 'continuous' or 'pulsed'
                'offset': 0.0,                  # Time offset
                'pulse_length': None,           # For pulsed sources
                'bandwidth_limit': False        # Bandwidth limiting
            },
            
            # ==== ADVANCED SETTINGS ====
            'advanced': {
                'override_global_source_settings': False,
                'optimize_for_short_pulse': False,
                'rotations': {
                    'theta': 0.0,               # Rotation angles
                    'phi': 0.0,
                    'psi': 0.0
                },
                'multifrequency_beam_calculation': True,
                'modal_properties_calculation': True
            }
        }
    
    def setup_monitor_names(self):
        """
        Define configurable monitor names to replace hardcoded values.
        This enables flexible integration with Figure of Merit calculations.
        """
        
        self.monitor_names = {
            # Primary monitors for FOM calculation
            'field_monitor': 'design_region_fields',
            'index_monitor': 'geometry_verification',
            
            # Mode expansion monitors
            'input_monitor': 'input_mode_expansion',
            'output_monitor': 'output_mode_expansion',
            'fiber_reference': 'fiber_mode_reference',
            
            # Power monitors
            'reflection_monitor': 'input_reflection',
            'transmission_monitor': 'output_transmission',
            
            # Slice monitors (list)
            'slice_monitors': [f'mode_slice_{i+1:02d}' for i in range(12)],
            
            # Field monitors
            'gradient_monitor': 'design_region_fields',
            'interface_monitor': 'interface_detection_fields'
        }
    
    def setup_output_settings(self):
        """Define FSP file management and output configuration"""
        
        self.output_settings = {
            # ==== FSP FILE MANAGEMENT ====
            'fsp_file': {
                'filename': 'edge_coupler_optimization.fsp',
                'directory': './',              # Current working directory
                'update_each_iteration': True,  # Save after each iteration
                'backup_frequency': 10,         # Backup every 10 iterations
                'final_filename': 'edge_coupler_optimized_final.fsp'
            },
            
            # ==== DATA EXPORT SETTINGS ====
            'data_export': {
                'export_monitors': True,        # Export monitor data
                'export_geometry': True,        # Export geometry parameters
                'export_materials': True,       # Export material properties
                'export_format': 'mat',         # MATLAB format for post-processing
                'export_fields': False          # Don't export large field arrays (by default)
            },
            
            # ==== GEOMETRY EXPORT ====
            'geometry_export': {
                'gds_filename': 'edge_coupler_optimized.gds',  # GDS layout file
                'export_layers': {
                    'silicon': 1,               # Layer 1 for silicon
                    'etch': 2,                  # Layer 2 for etch regions
                    'markers': 10               # Layer 10 for alignment markers
                },
                'coordinate_system': 'center_origin',  # Center at (0,0)
                'unit_scale': 1e-6              # Micron units
            },
            
            # ==== DASHBOARD SETTINGS ====
            'dashboard': {
                'update_frequency': 1,          # Update every iteration
                'save_plots': True,             # Save plots as images
                'plot_format': 'png',           # PNG format for plots
                'show_realtime': True,          # Show real-time updates
                'plot_directory': './plots/',   # Directory for saved plots
                'history_length': 100           # Keep last 100 iterations in memory
            }
        }
    
    def validate_device_configuration(self):
        """Validate configuration consistency and catch common errors"""
        
        validation_errors = []
        warnings = []
        
        # Check design region fits within simulation domain
        design_x = self.design_region['geometry']['length']
        design_y = self.design_region['geometry']['width']
        sim_x = self.simulation_domain['dimensions']['x_span']
        sim_y = self.simulation_domain['dimensions']['y_span']
        
        if design_x >= sim_x * 0.8:  # Design region too large (80% threshold)
            warnings.append(f"Design region x-span ({design_x*1e6:.1f} μm) is large compared to simulation domain ({sim_x*1e6:.1f} μm)")
        
        if design_y >= sim_y * 0.6:  # Design region too wide
            warnings.append(f"Design region y-span ({design_y*1e6:.1f} μm) is large compared to simulation domain ({sim_y*1e6:.1f} μm)")
        
        # Check monitor positions are within simulation domain
        for monitor_group in [self.mode_expansion_monitors, self.slice_monitors, 
                            self.field_monitors, self.specialized_monitors]:
            for monitor_name, monitor_config in monitor_group.items():
                if 'geometry' in monitor_config:
                    pos = monitor_config['geometry'].get('position', {})
                    x = pos.get('x', 0)
                    
                    if abs(x) > sim_x/2:
                        validation_errors.append(f"Monitor {monitor_name} position x={x*1e6:.1f} μm outside simulation domain")
        
        # Check material consistency
        for material_name in self.design_region['materials'].values():
            if material_name not in self.materials:
                validation_errors.append(f"Material '{material_name}' referenced but not defined")
        
        # Check fabrication constraints
        min_feature = self.design_region['topology']['min_feature_size']
        mesh_size = self.mesh['overrides']['design_region']['max_mesh_step'] 
        
        if mesh_size > min_feature / 3:  # Mesh too coarse for features
            warnings.append(f"Mesh size ({mesh_size*1e9:.0f} nm) may be too coarse for min feature size ({min_feature*1e9:.0f} nm)")
        
        # Check wavelength settings
        center_wl = self.device_specs['wavelength']['center']
        wl_span = self.device_specs['wavelength']['span']
        
        if wl_span / center_wl > 0.1:  # >10% fractional bandwidth
            warnings.append(f"Large fractional bandwidth ({wl_span/center_wl*100:.1f}%) may require careful dispersion modeling")
        
        # Store validation results
        self.validation_results = {
            'errors': validation_errors,
            'warnings': warnings,
            'is_valid': len(validation_errors) == 0
        }
        
        # Print validation summary
        if validation_errors:
            print("❌ DEVICE CONFIGURATION ERRORS:")
            for error in validation_errors:
                print(f"  - {error}")
        
        if warnings:
            print("⚠️  DEVICE CONFIGURATION WARNINGS:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if not validation_errors and not warnings:
            print("✅ Device configuration is valid!")
    
    # ======================================================================
    # UTILITY FUNCTIONS FOR INTEGRATION
    # ======================================================================
    
    def get_geometry_parameters(self):
        """Return parameters for RectangleClusteringTopology initialization"""
        
        design = self.design_region
        geom = design['geometry']
        
        # Create coordinate arrays for design region
        x_coords = np.linspace(-geom['length']/2, geom['length']/2, 120)
        y_coords = np.linspace(-geom['width']/2, geom['width']/2, 60)
        z_coord = geom['thickness']
        
        # Get material properties
        high_index_material = self.materials[design['materials']['high_index']]
        low_index_material = self.materials[design['materials']['low_index']]
        
        return {
            'min_feature_size': design['topology']['min_feature_size'],
            'eps_min': low_index_material['permittivity_tensor'],
            'eps_max': high_index_material['permittivity_tensor'],
            'x': x_coords,
            'y': y_coords,
            'z': z_coord,
            'material_order': design['topology']['material_order'],
            'anisotropic_materials': {
                'eps_min': low_index_material['permittivity_tensor'],
                'eps_max': high_index_material['permittivity_tensor']
            }
        }
    
    def get_initial_parameters(self):
        """Generate initial parameters for rectangle clustering optimization"""
        
        topology = self.design_region['topology']
        num_rectangles = topology['max_rectangles']
        init_method = topology['initialization']['method']
        
        if init_method == 'uniform':
            initial_params = np.ones(num_rectangles) / num_rectangles
            
        elif init_method == 'smooth_taper':
            taper_ratio = topology['initialization']['taper_ratio']
            smoothing = topology['initialization']['smoothing_factor']
            randomization = topology['initialization'].get('randomization', 0.0)
            
            # Create smooth taper profile
            x = np.linspace(0, 1, num_rectangles)
            weights = 1 + (taper_ratio - 1) * (1 / (1 + np.exp(-10 * (x - 0.5))))
            
            # Apply smoothing
            weights = smoothing * weights + (1 - smoothing) * np.mean(weights)
            
            # Add randomization if specified
            if randomization > 0:
                noise = np.random.normal(0, randomization, num_rectangles)
                weights = weights * (1 + noise)
                weights = np.maximum(weights, 0.1)  # Ensure positive
            
            # Normalize to sum to 1
            initial_params = weights / np.sum(weights)
            
        else:
            initial_params = np.ones(num_rectangles) / num_rectangles
        
        return initial_params
    
    def create_base_script_function(self):
        """Create function for Lumerical base script generation"""
        
        def base_script(fdtd):
            """Function to create base simulation in Lumerical FDTD"""
            
            # Clear existing simulation
            fdtd.clear()
            
            # Add FDTD solver
            fdtd.addfdtd()
            
            # Set simulation domain
            domain = self.simulation_domain['dimensions']
            fdtd.set('x span', domain['x_span'])
            fdtd.set('y span', domain['y_span'])
            fdtd.set('z span', domain['z_span'])
            fdtd.set('x', domain['center']['x'])
            fdtd.set('y', domain['center']['y'])
            fdtd.set('z', domain['center']['z'])
            
            # Set mesh settings
            mesh = self.mesh['global']
            fdtd.set('mesh accuracy', mesh['accuracy'])
            fdtd.set('max mesh step', mesh['max_mesh_step'])
            fdtd.set('min mesh step', mesh['min_mesh_step'])
            
            # Set boundary conditions
            boundaries = self.simulation_domain['boundaries']
            fdtd.set('x min bc', boundaries['x_min'])
            fdtd.set('x max bc', boundaries['x_max'])
            fdtd.set('y min bc', boundaries['y_min'])
            fdtd.set('y max bc', boundaries['y_max'])
            fdtd.set('z min bc', boundaries['z_min'])
            fdtd.set('z max bc', boundaries['z_max'])
            
            # Set PML settings
            pml = self.simulation_domain['pml']
            fdtd.set('pml layers', pml['layers'])
            fdtd.set('pml kappa', pml['conductivity']['kappa'])
            fdtd.set('pml sigma', pml['conductivity']['sigma'])
            
            # Set solver settings
            solver = self.simulation_domain['solver']
            fdtd.set('auto shutoff level', solver['shutoff_level'])
            fdtd.set('use early shutoff', solver['auto_shutoff'])
            fdtd.set('simulation time', solver['max_time_steps'] * 1e-15)  # Convert to time
            
            # Add fixed geometry structures
            self._add_fixed_geometry_to_fdtd(fdtd)
            
            # Add monitors
            self._add_monitors_to_fdtd(fdtd)
            
            # Add source (will be configured by optimization system)
            self._add_source_to_fdtd(fdtd)
            
            print("Base simulation created successfully")
            
        return base_script
    
    def _add_fixed_geometry_to_fdtd(self, fdtd):
        """Add all fixed geometry structures to FDTD simulation"""
        
        for struct_name, struct_config in self.fixed_geometry.items():
            material_name = struct_config['material']
            geometry = struct_config['geometry']
            
            # Add rectangle structure
            fdtd.addrect()
            fdtd.set('name', struct_name)
            
            # Set geometry
            fdtd.set('x span', geometry['x_span'])
            fdtd.set('y span', geometry['y_span'])
            fdtd.set('z span', geometry['z_span'])
            fdtd.set('x', geometry['center_position']['x'])
            fdtd.set('y', geometry['center_position']['y'])
            fdtd.set('z', geometry['center_position']['z'])
            
            # Set material
            material_def = self.materials[material_name]
            self._set_material_properties_fdtd(fdtd, struct_name, material_def)
    
    def _add_monitors_to_fdtd(self, fdtd):
        """Add all monitors to FDTD simulation"""
        
        # Add mode expansion monitors
        for monitor_name, monitor_config in self.mode_expansion_monitors.items():
            self._add_mode_expansion_monitor_fdtd(fdtd, monitor_name, monitor_config)
        
        # Add field monitors
        for monitor_name, monitor_config in self.field_monitors.items():
            self._add_field_monitor_fdtd(fdtd, monitor_name, monitor_config)
        
        # Add slice monitors
        for monitor_name, monitor_config in self.slice_monitors.items():
            self._add_field_monitor_fdtd(fdtd, monitor_name, monitor_config)
        
        # Add specialized monitors
        for monitor_name, monitor_config in self.specialized_monitors.items():
            if monitor_config['type'] == 'power':
                self._add_power_monitor_fdtd(fdtd, monitor_name, monitor_config)
            elif monitor_config['type'] == 'index':
                self._add_index_monitor_fdtd(fdtd, monitor_name, monitor_config)
    
    def _add_source_to_fdtd(self, fdtd):
        """Add comprehensive source configuration to FDTD simulation"""
        
        source_config = self.source_config
        geometry = source_config['geometry']
        wavelength = source_config['wavelength']
        mode_settings = source_config['mode_settings']
        excitation = source_config['excitation']
        
        # Add mode source
        fdtd.addmode()
        fdtd.set('name', source_config['name'])
        fdtd.set('enabled', source_config['enabled'])
        
        # Set position and geometry
        fdtd.set('x', geometry['position']['x'])
        fdtd.set('y', geometry['position']['y'])
        fdtd.set('z', geometry['position']['z'])
        fdtd.set('y span', geometry['span']['y'])
        fdtd.set('z span', geometry['span']['z'])
        
        # Set injection axis and direction
        fdtd.set('injection axis', geometry['injection_axis'])
        fdtd.set('direction', mode_settings['direction'])
        
        # Set wavelength/frequency settings
        fdtd.set('center wavelength', wavelength['center'])
        fdtd.set('wavelength span', wavelength['span'])
        if wavelength['use_wavelength_sweep']:
            fdtd.set('number of frequency points', wavelength['frequency_points'])
        
        # Set mode calculation settings
        fdtd.set('mode selection', mode_settings['mode_selection'])
        fdtd.set('mode calculation', mode_settings['mode_calculation'])
        fdtd.set('number of modes for expansion', mode_settings['number_of_modes'])
        fdtd.set('search', 'max index')
        fdtd.set('index', mode_settings['search_index'])
        fdtd.set('bent waveguide', mode_settings['bent_waveguide'])
        fdtd.set('selected mode numbers', [mode_settings['mode_number']])
        
        # Set excitation properties
        fdtd.set('amplitude', excitation['amplitude'])
        fdtd.set('phase', excitation['phase'])
        
        # Set advanced properties
        advanced = source_config['advanced']
        if advanced['override_global_source_settings']:
            fdtd.set('override global source settings', True)
        
        print(f"Source '{source_config['name']}' configured: {mode_settings['mode_selection']} at {wavelength['center']*1e9:.0f} nm")
    
    def _set_material_properties_fdtd(self, fdtd, object_name, material_def):
        """Set material properties in FDTD for an object"""
        
        fdtd.select(object_name)
        
        if material_def.get('anisotropy', 0) == 1:
            # Anisotropic material
            fdtd.set('material', '<Object defined dielectric>')
            fdtd.set('anisotropy', 1)
            fdtd.set('index x', material_def['index_xx'])
            fdtd.set('index y', material_def['index_yy'])
            fdtd.set('index z', material_def['index_zz'])
        else:
            # Isotropic material
            fdtd.set('material', '<Object defined dielectric>')
            fdtd.set('index', material_def['index'])
        
        # Set mesh order
        fdtd.set('mesh order', material_def.get('mesh_order', 2))
    
    def _add_mode_expansion_monitor_fdtd(self, fdtd, monitor_name, config):
        """Add mode expansion monitor to FDTD"""
        
        fdtd.addmodeexpansion()
        fdtd.set('name', monitor_name)
        
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        fdtd.set('monitor type', '2D X-normal')
        fdtd.set('y span', span['y'])
        fdtd.set('z span', span['z'])
        
        # Mode settings
        mode_settings = config['mode_settings']
        fdtd.set('mode calculation', mode_settings['mode_calculation'])
        fdtd.set('selected mode numbers', mode_settings['selected_mode_numbers'])
        fdtd.set('number of modes for expansion', mode_settings['number_of_modes'])
    
    def _add_field_monitor_fdtd(self, fdtd, monitor_name, config):
        """Add field monitor to FDTD"""
        
        fdtd.addtime()
        fdtd.set('name', monitor_name)
        
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        
        if geometry.get('orientation') == 'x_normal':
            fdtd.set('monitor type', '2D X-normal')
            fdtd.set('y span', span['y'])
            fdtd.set('z span', span['z'])
        else:
            fdtd.set('monitor type', '3D')
            fdtd.set('x span', span['x'])
            fdtd.set('y span', span['y'])
            fdtd.set('z span', span['z'])
        
        # Field settings
        field_settings = config.get('field_settings', {})
        fdtd.set('record E', field_settings.get('record_E', True))
        fdtd.set('record H', field_settings.get('record_H', True))
        fdtd.set('record power', field_settings.get('record_power', True))
        
        # Mesh override
        if 'resolution' in config and config['resolution'].get('mesh_override', False):
            resolution = config['resolution']
            fdtd.set('override global monitor settings', True)
            if 'dx' in resolution:
                fdtd.set('dx', resolution['dx'])
            if 'dy' in resolution:
                fdtd.set('dy', resolution['dy'])
            if 'dz' in resolution:
                fdtd.set('dz', resolution['dz'])
    
    def _add_power_monitor_fdtd(self, fdtd, monitor_name, config):
        """Add power monitor to FDTD"""
        
        fdtd.addpower()
        fdtd.set('name', monitor_name)
        
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        fdtd.set('monitor type', '2D X-normal')
        fdtd.set('y span', span['y'])
        fdtd.set('z span', span['z'])
    
    def _add_index_monitor_fdtd(self, fdtd, monitor_name, config):
        """Add index monitor to FDTD"""
        
        fdtd.addindex()
        fdtd.set('name', monitor_name)
        
        geometry = config['geometry']
        position = geometry['position']
        span = geometry['span']
        
        fdtd.set('x', position['x'])
        fdtd.set('y', position['y'])
        fdtd.set('z', position['z'])
        fdtd.set('x span', span['x'])
        fdtd.set('y span', span['y'])
        fdtd.set('z span', span['z'])
    
    def get_all_monitors(self):
        """Return dictionary of all monitor configurations"""
        
        all_monitors = {}
        all_monitors.update(self.mode_expansion_monitors)
        all_monitors.update(self.slice_monitors)
        all_monitors.update(self.field_monitors) 
        all_monitors.update(self.specialized_monitors)
        
        return all_monitors
    
    def print_device_summary(self):
        """Print comprehensive device configuration summary"""
        
        print("\n" + "="*80)
        print("DEVICE CONFIGURATION SUMMARY")
        print("="*80)
        
        # Device specifications
        specs = self.device_specs
        print(f"\nDevice Type: {specs['device_type']}")
        print(f"Platform: {specs['platform']}")
        print(f"Description: {specs['description']}")
        
        # Wavelength settings
        wl = specs['wavelength']
        print(f"\nWavelength Settings:")
        print(f"  Center: {wl['center']*1e9:.0f} nm")
        print(f"  Span: {wl['span']*1e9:.0f} nm")
        print(f"  Points: {wl['num_points']}")
        
        # Design region
        design = self.design_region
        geom = design['geometry']
        print(f"\nDesign Region:")
        print(f"  Dimensions: {geom['length']*1e6:.1f} × {geom['width']*1e6:.1f} × {geom['thickness']*1e9:.0f} μm³")
        print(f"  Min feature size: {design['topology']['min_feature_size']*1e9:.0f} nm")
        print(f"  Max rectangles: {design['topology']['max_rectangles']}")
        
        # Simulation domain
        sim = self.simulation_domain['dimensions']
        print(f"\nSimulation Domain:")
        print(f"  Dimensions: {sim['x_span']*1e6:.1f} × {sim['y_span']*1e6:.1f} × {sim['z_span']*1e6:.1f} μm³")
        
        # Source settings
        source = self.source_config
        print(f"\nSource Configuration:")
        print(f"  Type: {source['source_type']}")
        print(f"  Position: ({source['geometry']['position']['x']*1e6:.1f}, {source['geometry']['position']['y']*1e6:.1f}) μm")
        print(f"  Mode: {source['mode_settings']['mode_selection']}")
        print(f"  Direction: {source['mode_settings']['direction']}")
        print(f"  Wavelength: {source['wavelength']['center']*1e9:.0f} ± {source['wavelength']['span']*1e9/2:.0f} nm")
        
        # Monitor count
        monitor_counts = {
            'Mode Expansion': len(self.mode_expansion_monitors),
            'Slice Monitors': len(self.slice_monitors),
            'Field Monitors': len(self.field_monitors),
            'Specialized': len(self.specialized_monitors)
        }
        total_monitors = sum(monitor_counts.values())
        
        print(f"\nMonitor Configuration:")
        for monitor_type, count in monitor_counts.items():
            print(f"  {monitor_type}: {count}")
        print(f"  Total: {total_monitors}")
        
        # Materials
        print(f"\nMaterials Defined:")
        for material_name, material_def in self.materials.items():
            print(f"  {material_name}: {material_def['type']} ({material_def.get('index', 'variable')})")
        
        # Output settings
        fsp = self.output_settings['fsp_file']
        print(f"\nOutput Settings:")
        print(f"  FSP file: {fsp['filename']}")
        print(f"  Directory: {fsp['directory']}")
        print(f"  Update each iteration: {fsp['update_each_iteration']}")
        
        # Validation results
        if hasattr(self, 'validation_results'):
            results = self.validation_results
            print(f"\nValidation:")
            print(f"  Status: {'✅ Valid' if results['is_valid'] else '❌ Invalid'}")
            print(f"  Errors: {len(results['errors'])}")
            print(f"  Warnings: {len(results['warnings'])}")
        
        print("="*80)


# ============================================================================
# INSTANTIATE CONFIGURATION
# ============================================================================

# Create the global device configuration instance
device_config = DeviceConfig()

# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL ACCESS
# ============================================================================

def get_device_config():
    """Return the global device configuration instance"""
    return device_config

def get_geometry_parameters():
    """Return parameters for RectangleClusteringTopology initialization"""
    return device_config.get_geometry_parameters()

def get_initial_parameters():
    """Return initial parameters for optimization"""
    return device_config.get_initial_parameters()

def create_base_script():
    """Create BaseScript object for Lumerical simulation"""
    from lumNLopt.utilities.base_script import BaseScript
    
    base_script_function = device_config.create_base_script_function()
    return BaseScript(base_script_function)

def get_monitor_names():
    """Return dictionary of configurable monitor names"""
    return device_config.monitor_names

def get_all_monitors():
    """Return dictionary of all monitor configurations"""
    return device_config.get_all_monitors()

def get_material_definitions():
    """Return dictionary of material definitions"""
    return device_config.materials

def get_simulation_settings():
    """Return simulation domain and solver settings"""
    return device_config.simulation_domain

def get_mesh_settings():
    """Return mesh configuration"""
    return device_config.mesh

def get_output_settings():
    """Return FSP file and export settings"""
    return device_config.output_settings

def get_wavelength_settings():
    """Return wavelength configuration for use by Figure_of_merit.py"""
    return device_config.device_specs['wavelength']

def get_fabrication_constraints():
    """Return fabrication constraints"""
    return device_config.device_specs['fabrication']

def get_performance_targets():
    """Return performance targets"""
    return device_config.device_specs['targets']

def get_source_config():
    """Return source configuration"""
    return device_config.source_config

def get_source_wavelength_settings():
    """Return source wavelength settings"""
    return device_config.source_config['wavelength']

def get_source_mode_settings():
    """Return source mode settings"""
    return device_config.source_config['mode_settings']

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary when run directly
    device_config.print_device_summary()
    
    # Print monitor names for reference
    print("\nMonitor Names for FOM Integration:")
    print("-" * 40)
    for key, value in device_config.monitor_names.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} monitors")
            for i, monitor in enumerate(value[:3]):  # Show first 3
                print(f"  {monitor}")
            if len(value) > 3:
                print(f"  ... and {len(value)-3} more")
        else:
            print(f"{key}: {value}")
    
    # Test validation
    if device_config.validation_results['is_valid']:
        print("\n✅ Device configuration ready for optimization!")
    else:
        print("\n❌ Device configuration has errors that need to be fixed.")
