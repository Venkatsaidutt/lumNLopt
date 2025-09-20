"""
Structure Configuration for lumNLopt Edge Coupler Optimization

This file defines:
1. Fixed structure components (substrate, cladding, waveguides, etc.)
2. Optimizable design region parameters for rectangle clustering topology
3. Material properties and simulation domain settings

The configuration separates fixed geometry from the optimizable region,
allowing the optimization to focus on the design region while maintaining
proper context with input/output waveguides and substrate.
"""

import numpy as np
import lumapi
from lumNLopt.utilities.materials import Material

# ============================================================================
# DEVICE AND TECHNOLOGY PARAMETERS
# ============================================================================

class StructureConfig:
    """
    Configuration class for edge coupler structure definition.
    Separates fixed components from optimizable design region.
    """
    
    def __init__(self):
        # Initialize all structure parameters
        self.setup_device_parameters()
        self.setup_fixed_geometry()
        self.setup_design_region()
        self.setup_materials()
        self.setup_simulation_domain()
        self.setup_mesh_settings()
    
    def setup_device_parameters(self):
        """Define basic device and technology parameters"""
        
        self.device_params = {
            'name': 'adiabatic_edge_coupler',
            'description': 'Silicon photonic edge coupler with rectangle clustering optimization',
            'technology': 'SOI_220nm',          # Silicon-on-insulator platform
            'design_wavelength': 1.55e-6,      # Primary design wavelength [m]
            'operating_band': 'C_band',        # Telecommunications C-band
            'polarization': 'TE',              # TE polarization optimization
            
            # Platform specifications
            'silicon_thickness': 220e-9,        # Standard SOI thickness [m]
            'box_thickness': 2.0e-6,           # Buried oxide thickness [m] 
            'substrate_material': 'SiO2',       # Substrate material
            'cladding_material': 'SiO2'        # Top cladding material
        }
    
    def setup_fixed_geometry(self):
        """Define all fixed (non-optimizable) structure components"""
        
        # ==== SUBSTRATE LAYER ====
        self.substrate = {
            'material': 'SiO2_substrate',
            'thickness': self.device_params['box_thickness'],
            'refractive_index': 1.444,          # At 1550 nm
            'position': {
                'z_min': -self.device_params['box_thickness'],
                'z_max': 0.0
            },
            'extend_to_boundaries': True        # Extends to simulation boundaries
        }
        
        # ==== TOP CLADDING ====
        self.top_cladding = {
            'material': 'SiO2_cladding',
            'thickness': 2.0e-6,
            'refractive_index': 1.444,
            'position': {
                'z_min': self.device_params['silicon_thickness'],
                'z_max': self.device_params['silicon_thickness'] + 2.0e-6
            },
            'extend_to_boundaries': True
        }
        
        # ==== INPUT WAVEGUIDE ====
        self.input_waveguide = {
            'material': 'Silicon',
            'refractive_index': 3.48,           # Silicon at 1550 nm
            'geometry': {
                'width': 450e-9,                # Single mode width for TE00
                'thickness': self.device_params['silicon_thickness'],
                'length': 5.0e-6                # Length before design region
            },
            'position': {
                'x_start': -8.0e-6,             # Start position relative to design region
                'x_end': -3.0e-6,               # End position (start of design region)
                'y_center': 0.0,                # Centered on design region
                'z_center': self.device_params['silicon_thickness']/2
            },
            'taper': {
                'enabled': False,               # No taper on input side
                'start_width': 450e-9,
                'end_width': 450e-9
            }
        }
        
        # ==== OUTPUT WAVEGUIDE ====
        self.output_waveguide = {
            'material': 'Silicon',
            'refractive_index': 3.48,
            'geometry': {
                'width': 2.0e-6,                # Wider for better fiber coupling
                'thickness': self.device_params['silicon_thickness'],
                'length': 5.0e-6                # Length after design region
            },
            'position': {
                'x_start': 15.0e-6,             # Start after design region
                'x_end': 20.0e-6,               # End position
                'y_center': 0.0,
                'z_center': self.device_params['silicon_thickness']/2
            },
            'taper': {
                'enabled': True,                # Taper for mode expansion
                'start_width': 1.2e-6,          # Narrower at design region interface
                'end_width': 2.0e-6             # Wider at fiber coupling end
            }
        }
        
        # ==== SIDE CLADDING REGIONS ====
        self.side_cladding = {
            'material': 'SiO2_side',
            'refractive_index': 1.444,
            'geometry': {
                'width': 3.0e-6,                # Width on each side of device
                'thickness': self.device_params['silicon_thickness']
            },
            'extend_to_boundaries': True        # Extends to y-boundaries
        }
    
    def setup_design_region(self):
        """Define the optimizable design region for rectangle clustering"""
        
        self.design_region = {
            # ==== GEOMETRIC PARAMETERS ====
            'geometry': {
                'length': 18.0e-6,              # Total design region length [m]
                'width': 3.0e-6,                # Design region width [m]
                'thickness': self.device_params['silicon_thickness'],
                'center_position': {
                    'x': 6.0e-6,                # Center x-coordinate
                    'y': 0.0,                   # Center y-coordinate  
                    'z': self.device_params['silicon_thickness']/2
                }
            },
            
            # ==== RECTANGLE CLUSTERING PARAMETERS ====
            'topology': {
                'optimization_type': 'rectangle_clustering',
                'min_feature_size': 120e-9,     # Minimum feature size [m] (120 nm)
                'max_rectangles': 50,           # Maximum number of rectangles
                'initial_distribution': 'tapered', # 'uniform', 'tapered', or 'custom'
                'clustering_direction': 'x',     # Primary clustering direction
                
                # Parameter bounds for optimization
                'bounds': {
                    'min_fraction': 0.008,      # Minimum width fraction (0.8%)
                    'max_fraction': 0.15,       # Maximum width fraction (15%)
                    'sum_constraint': 1.0       # Fractions must sum to 1
                },
                
                # Initial parameter distribution
                'initialization': {
                    'method': 'smooth_taper',   # Smooth transition from narrow to wide
                    'taper_ratio': 3.0,         # End width / start width ratio
                    'smoothing_factor': 0.8     # Smoothness of transition
                }
            },
            
            # ==== DESIGN REGION MATERIALS ====
            'materials': {
                'high_index': {
                    'name': 'Silicon_anisotropic',
                    'base_material': 'Silicon',
                    'refractive_index': 3.48,
                    
                    # Anisotropic permittivity tensor (for accurate modeling)
                    'permittivity_tensor': {
                        'eps_xx': 12.11,       # In-plane (x) permittivity
                        'eps_yy': 12.11,       # In-plane (y) permittivity
                        'eps_zz': 12.25,       # Out-of-plane (z) permittivity
                        'eps_xy': 0.0,         # Cross terms
                        'eps_xz': 0.0,
                        'eps_yz': 0.0
                    },
                    'mesh_order': 2             # High priority for meshing
                },
                
                'low_index': {
                    'name': 'SiO2_design',
                    'base_material': 'SiO2',
                    'refractive_index': 1.444,
                    
                    # Isotropic cladding material
                    'permittivity_tensor': {
                        'eps_xx': 2.085,
                        'eps_yy': 2.085,
                        'eps_zz': 2.085,
                        'eps_xy': 0.0,
                        'eps_xz': 0.0,
                        'eps_yz': 0.0
                    },
                    'mesh_order': 3             # Lower priority for meshing
                }
            },
            
            # ==== FABRICATION CONSTRAINTS ====
            'fabrication': {
                'min_gap_size': 80e-9,          # Minimum gap between features
                'max_aspect_ratio': 8.0,        # Maximum length/width ratio
                'edge_roughness_tolerance': 5e-9, # Expected edge roughness
                'etch_bias': 10e-9,             # Systematic etch bias
                'overlay_tolerance': 20e-9       # Lithography overlay tolerance
            }
        }
    
    def setup_materials(self):
        """Create Material objects for all structure components"""
        
        self.materials = {}
        
        # ==== FIXED STRUCTURE MATERIALS ====
        
        # Substrate material
        self.materials['substrate'] = Material(
            base_epsilon=self.substrate['refractive_index']**2,
            name=self.substrate['material']
        )
        
        # Top cladding material  
        self.materials['cladding'] = Material(
            base_epsilon=self.top_cladding['refractive_index']**2,
            name=self.top_cladding['material']
        )
        
        # Input waveguide material
        self.materials['input_waveguide'] = Material(
            base_epsilon=self.input_waveguide['refractive_index']**2,
            name=self.input_waveguide['material']
        )
        
        # Output waveguide material
        self.materials['output_waveguide'] = Material(
            base_epsilon=self.output_waveguide['refractive_index']**2,
            name=self.output_waveguide['material']
        )
        
        # ==== DESIGN REGION MATERIALS ====
        
        # High index material (Silicon with anisotropy)
        high_index_config = self.design_region['materials']['high_index']
        self.materials['high_index'] = Material(
            base_epsilon=high_index_config['permittivity_tensor']['eps_xx'],
            name=high_index_config['name'],
            mesh_order=high_index_config['mesh_order'],
            anisotropic_params=high_index_config['permittivity_tensor']
        )
        
        # Low index material (SiO2)
        low_index_config = self.design_region['materials']['low_index']
        self.materials['low_index'] = Material(
            base_epsilon=low_index_config['permittivity_tensor']['eps_xx'],
            name=low_index_config['name'],
            mesh_order=low_index_config['mesh_order'],
            anisotropic_params=low_index_config['permittivity_tensor']
        )
    
    def setup_simulation_domain(self):
        """Define simulation domain and boundary conditions"""
        
        self.simulation_domain = {
            # ==== DOMAIN SIZE ====
            'dimensions': {
                'x_span': 35.0e-6,              # Total simulation length
                'y_span': 12.0e-6,              # Total simulation width
                'z_span': 6.0e-6,               # Total simulation height
                'center': {
                    'x': 6.0e-6,                # Center on design region
                    'y': 0.0,
                    'z': self.device_params['silicon_thickness']/2
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
                'layers': 12,                   # Number of PML layers
                'conductivity': {
                    'kappa': 2.5,               # PML kappa parameter
                    'sigma': 1.0,               # PML sigma parameter
                    'polynomial_order': 3        # PML polynomial grading
                },
                'thickness': 0.8e-6            # PML thickness
            },
            
            # ==== SIMULATION SETTINGS ====
            'solver': {
                'type': 'FDTD',                # Use FDTD solver
                'auto_shutoff': True,          # Enable auto shutoff
                'shutoff_level': 1e-5,         # Shutoff threshold
                'max_simulation_time': 2000,   # Maximum time steps
                'dt_stability_factor': 0.99    # Time step stability factor
            }
        }
    
    def setup_mesh_settings(self):
        """Configure mesh settings for accurate simulation"""
        
        self.mesh = {
            # ==== GLOBAL MESH SETTINGS ====
            'global': {
                'accuracy': 5,                  # Mesh accuracy (1-8 scale)
                'max_mesh_step': 50e-9,         # Maximum mesh size
                'min_mesh_step': 15e-9,         # Minimum mesh size
                'mesh_refinement': 'automatic'   # Auto mesh refinement
            },
            
            # ==== REGION-SPECIFIC MESH OVERRIDES ====
            'overrides': {
                'design_region': {
                    'enabled': True,
                    'max_mesh_step': 15e-9,     # Fine mesh in design region
                    'min_mesh_step': 8e-9,      # Very fine mesh for gradients
                    'mesh_type': 'uniform'      # Uniform mesh for consistency
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
    
    def get_geometry_parameters(self):
        """Return parameters for RectangleClusteringTopology initialization"""
        
        design = self.design_region
        
        # Create coordinate arrays
        geom = design['geometry']
        x_coords = np.linspace(-geom['length']/2, geom['length']/2, 120)
        y_coords = np.linspace(-geom['width']/2, geom['width']/2, 60)
        z_coord = geom['thickness']
        
        return {
            'min_feature_size': design['topology']['min_feature_size'],
            'eps_min': design['materials']['low_index']['permittivity_tensor'],
            'eps_max': design['materials']['high_index']['permittivity_tensor'],
            'x': x_coords,
            'y': y_coords,
            'z': z_coord,
            'material_order': 'alternating',
            'anisotropic_materials': {
                'eps_min': design['materials']['low_index']['permittivity_tensor'],
                'eps_max': design['materials']['high_index']['permittivity_tensor']
            }
        }
    
    def get_initial_parameters(self):
        """Generate initial parameters for rectangle clustering optimization"""
        
        topology = self.design_region['topology']
        num_rectangles = topology['max_rectangles']
        
        init_method = topology['initialization']['method']
        
        if init_method == 'uniform':
            # Uniform distribution
            initial_params = np.ones(num_rectangles) / num_rectangles
            
        elif init_method == 'smooth_taper':
            # Smooth taper from narrow to wide
            taper_ratio = topology['initialization']['taper_ratio']
            smoothing = topology['initialization']['smoothing_factor']
            
            # Create smooth taper profile
            x = np.linspace(0, 1, num_rectangles)
            # Sigmoid-like function for smooth transition
            weights = 1 + (taper_ratio - 1) * (1 / (1 + np.exp(-10 * (x - 0.5))))
            
            # Apply smoothing
            weights = smoothing * weights + (1 - smoothing) * np.mean(weights)
            
            # Normalize to sum to 1
            initial_params = weights / np.sum(weights)
            
        else:
            # Default to uniform
            initial_params = np.ones(num_rectangles) / num_rectangles
        
        return initial_params
    
    def create_base_script_function(self):
        """Create function for Lumerical base script generation"""
        
        def base_script(fdtd):
            """Function to create base simulation in Lumerical FDTD"""
            
            # Clear existing simulation
            fdtd.clear()
            
            # ==== ADD FDTD SOLVER ====
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
            fdtd.set('simulation time', solver['max_simulation_time'])
            
            # ==== ADD FIXED GEOMETRY ====
            self._add_substrate(fdtd)
            self._add_cladding(fdtd)
            self._add_input_waveguide(fdtd)
            self._add_output_waveguide(fdtd)
            
            # ==== ADD SOURCE ====
            self._add_source(fdtd)
            
            # Note: Design region geometry will be added by RectangleClusteringTopology
            # Note: Monitors will be added by Monitors.py
            
            return fdtd
        
        return base_script
    
    def _add_substrate(self, fdtd):
        """Add substrate to Lumerical simulation"""
        fdtd.addrect()
        fdtd.set('name', 'substrate')
        fdtd.set('material', self.substrate['material'])
        fdtd.set('index', self.substrate['refractive_index'])
        
        # Set geometry
        domain = self.simulation_domain['dimensions']
        fdtd.set('x span', domain['x_span'])
        fdtd.set('y span', domain['y_span'])
        fdtd.set('z min', self.substrate['position']['z_min'])
        fdtd.set('z max', self.substrate['position']['z_max'])
        fdtd.set('x', domain['center']['x'])
        fdtd.set('y', domain['center']['y'])
    
    def _add_cladding(self, fdtd):
        """Add top cladding to Lumerical simulation"""
        fdtd.addrect()
        fdtd.set('name', 'top_cladding')
        fdtd.set('material', self.top_cladding['material'])
        fdtd.set('index', self.top_cladding['refractive_index'])
        
        # Set geometry
        domain = self.simulation_domain['dimensions']
        fdtd.set('x span', domain['x_span'])
        fdtd.set('y span', domain['y_span'])
        fdtd.set('z min', self.top_cladding['position']['z_min'])
        fdtd.set('z max', self.top_cladding['position']['z_max'])
        fdtd.set('x', domain['center']['x'])
        fdtd.set('y', domain['center']['y'])
    
    def _add_input_waveguide(self, fdtd):
        """Add input waveguide to Lumerical simulation"""
        wg = self.input_waveguide
        
        fdtd.addrect()
        fdtd.set('name', 'input_waveguide')
        fdtd.set('material', wg['material'])
        fdtd.set('index', wg['refractive_index'])
        
        # Set geometry
        fdtd.set('x min', wg['position']['x_start'])
        fdtd.set('x max', wg['position']['x_end'])
        fdtd.set('y span', wg['geometry']['width'])
        fdtd.set('z span', wg['geometry']['thickness'])
        fdtd.set('y', wg['position']['y_center'])
        fdtd.set('z', wg['position']['z_center'])
    
    def _add_output_waveguide(self, fdtd):
        """Add output waveguide to Lumerical simulation"""
        wg = self.output_waveguide
        
        if wg['taper']['enabled']:
            # Add tapered output waveguide
            fdtd.addwaveguide()
            fdtd.set('name', 'output_waveguide')
            fdtd.set('base material', wg['material'])
            fdtd.set('base index', wg['refractive_index'])
            
            # Define taper points
            x_coords = [wg['position']['x_start'], wg['position']['x_end']]
            width_coords = [wg['taper']['start_width'], wg['taper']['end_width']]
            
            # Set waveguide path and widths
            fdtd.set('x', x_coords)
            fdtd.set('y', [wg['position']['y_center']] * len(x_coords))
            fdtd.set('z', wg['position']['z_center'])
            fdtd.set('base width', width_coords)
            fdtd.set('base height', wg['geometry']['thickness'])
        else:
            # Add rectangular output waveguide
            fdtd.addrect()
            fdtd.set('name', 'output_waveguide')
            fdtd.set('material', wg['material'])
            fdtd.set('index', wg['refractive_index'])
            
            fdtd.set('x min', wg['position']['x_start'])
            fdtd.set('x max', wg['position']['x_end'])
            fdtd.set('y span', wg['geometry']['width'])
            fdtd.set('z span', wg['geometry']['thickness'])
            fdtd.set('y', wg['position']['y_center'])
            fdtd.set('z', wg['position']['z_center'])
    
    def _add_source(self, fdtd):
        """Add mode source to Lumerical simulation"""
        wg = self.input_waveguide
        
        fdtd.addmode()
        fdtd.set('name', 'mode_source')
        fdtd.set('injection axis', 'x-axis')
        fdtd.set('direction', 'forward')
        
        # Position source in input waveguide
        source_x = wg['position']['x_start'] + 1.0e-6  # 1 μm into input waveguide
        fdtd.set('x', source_x)
        fdtd.set('y', wg['position']['y_center'])
        fdtd.set('z', wg['position']['z_center'])
        
        # Set source size (larger than waveguide for mode calculation)
        fdtd.set('y span', wg['geometry']['width'] * 3)
        fdtd.set('z span', wg['geometry']['thickness'] * 3)
        
        # Set wavelength from device parameters
        fdtd.set('center wavelength', self.device_params['design_wavelength'])
        fdtd.set('wavelength span', 0.1e-6)  # 100 nm span for broadband
    
    def validate_structure(self):
        """Validate structure configuration for consistency"""
        
        validation_errors = []
        
        # Check design region fits within simulation domain
        design_length = self.design_region['geometry']['length']
        domain_length = self.simulation_domain['dimensions']['x_span']
        
        if design_length > domain_length * 0.8:  # Leave 20% margin
            validation_errors.append(f"Design region too large for simulation domain")
        
        # Check waveguide alignment
        input_end = self.input_waveguide['position']['x_end']
        design_start = self.design_region['geometry']['center_position']['x'] - design_length/2
        
        if abs(input_end - design_start) > 0.1e-6:  # 100 nm tolerance
            validation_errors.append(f"Input waveguide not aligned with design region")
        
        # Check material indices
        for mat_name, material in self.materials.items():
            if hasattr(material, 'base_epsilon') and material.base_epsilon <= 0:
                validation_errors.append(f"Invalid material index for {mat_name}")
        
        # Check mesh settings
        min_mesh = self.mesh['global']['min_mesh_step']
        min_feature = self.design_region['topology']['min_feature_size']
        
        if min_mesh > min_feature / 5:  # Need at least 5 mesh points per feature
            validation_errors.append(f"Mesh too coarse for minimum feature size")
        
        return validation_errors
    
    def print_structure_summary(self):
        """Print summary of structure configuration"""
        
        print("\n" + "="*60)
        print("STRUCTURE CONFIGURATION SUMMARY")
        print("="*60)
        
        # Device info
        device = self.device_params
        print(f"\nDevice: {device['name']}")
        print(f"Technology: {device['technology']}")
        print(f"Design wavelength: {device['design_wavelength']*1e9:.0f} nm")
        
        # Design region
        design = self.design_region['geometry']
        print(f"\nDesign Region:")
        print(f"  Size: {design['length']*1e6:.1f} × {design['width']*1e6:.1f} × {design['thickness']*1e9:.0f} μm³")
        
        topology = self.design_region['topology']
        print(f"  Min feature: {topology['min_feature_size']*1e9:.0f} nm")
        print(f"  Max rectangles: {topology['max_rectangles']}")
        
        # Simulation domain
        domain = self.simulation_domain['dimensions']
        print(f"\nSimulation Domain:")
        print(f"  Size: {domain['x_span']*1e6:.1f} × {domain['y_span']*1e6:.1f} × {domain['z_span']*1e6:.1f} μm³")
        
        # Validation
        errors = self.validate_structure()
        if errors:
            print(f"\nValidation Errors:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"\n✅ Structure configuration is valid!")
        
        print("="*60)


# ============================================================================
# INSTANTIATE CONFIGURATION
# ============================================================================

# Create the global structure configuration instance
structure_config = StructureConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_structure_config():
    """Return the global structure configuration instance"""
    return structure_config

def get_geometry_parameters():
    """Return parameters for RectangleClusteringTopology initialization"""
    return structure_config.get_geometry_parameters()

def get_initial_parameters():
    """Return initial parameters for optimization"""
    return structure_config.get_initial_parameters()

def create_base_script():
    """Create BaseScript object for Lumerical simulation"""
    from lumNLopt.utilities.base_script import BaseScript
    
    base_script_function = structure_config.create_base_script_function()
    return BaseScript(base_script_function)

def get_material_objects():
    """Return dictionary of Material objects"""
    return structure_config.materials

def get_simulation_settings():
    """Return simulation domain and solver settings"""
    return structure_config.simulation_domain

def get_mesh_settings():
    """Return mesh configuration"""
    return structure_config.mesh

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Print configuration summary when run directly
    structure_config.print_structure_summary()
