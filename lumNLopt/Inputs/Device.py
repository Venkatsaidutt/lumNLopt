# Device.py - Shape Optimization with Full Anisotropic + Dispersive Material Support
# Uses Sellmeier coefficients for complete wavelength-dependent anisotropic materials

import numpy as np
import lumapi

class DeviceConfig:
    """
    Device configuration for shape optimization with:
    1. Design region volume (for rectangle clustering)
    2. Complete anisotropic + dispersive materials (Sellmeier coefficients)
    3. Shape optimization (discrete materials, not density-based)
    4. Full tensor support for optimization_material and background_material
    """
    
    def __init__(self):
        self.setup_device_specifications()
        self.setup_design_region()
        self.setup_anisotropic_materials()
        self.setup_simulation_domain()
        self.setup_monitors()
        self.setup_source()
        self.setup_output_settings()
        
    def setup_device_specifications(self):
        """Basic device specifications"""
        self.device_specs = {
            'type': 'edge_coupler',
            'platform': 'SOI', 
            'technology_node': '220nm_SOI',
            'wavelength': {
                'center': 1.55e-6,          # 1550 nm
                'span': 0.08e-6,            # ±40 nm
                'num_points': 21,
                'min_wavelength': 1.51e-6,  # For Sellmeier evaluation
                'max_wavelength': 1.59e-6
            }
        }
    
    def setup_design_region(self):
        """
        Design region volume for rectangle clustering.
        topology.py will divide this volume into rectangles.
        """
        self.design_region = {
            # ==== DESIGN REGION VOLUME ====
            'volume': {
                'x_center': 0.0,            # Design region center
                'y_center': 0.0,
                'z_center': 110e-9,         # Center of silicon layer
                
                'length': 8.0e-6,           # Total length (x-direction)
                'width': 3.0e-6,            # Total width (y-direction)
                'thickness': 220e-9,        # Silicon layer thickness
                
                # Bounds for rectangle clustering
                'x_min': -4.0e-6,
                'x_max': 4.0e-6,
                'y_min': -1.5e-6,
                'y_max': 1.5e-6,
                'z_min': 0.0,
                'z_max': 220e-9
            },
            
            # ==== OPTIMIZATION CONSTRAINTS ====
            'constraints': {
                'min_feature_size': 150e-9,     # Minimum manufacturable feature
                'num_layers': 1,                # Single layer optimization  
                'optimization_type': 'shape'    # Shape optimization (not density)
            }
        }
    
    def setup_anisotropic_materials(self):
        """
        Complete anisotropic + dispersive materials using Sellmeier coefficients.
        Both optimization_material and background_material have full tensor support.
        """
        self.materials = {
            
            # ==== OPTIMIZATION MATERIAL: Anisotropic Silicon ====
            'optimization_material': {
                'name': 'silicon_anisotropic',
                'type': 'sellmeier_anisotropic',
                'lumerical_material': 'Si (Silicon) - Anisotropic',
                
                # Sellmeier coefficients for each tensor component
                'sellmeier_coefficients': {
                    # Diagonal tensor components (primary)
                    'xx': {
                        'A0': 11.6858,
                        'A1': 0.939816,  'lambda1': 1.10795e-6,
                        'A2': 8.10461e-3, 'lambda2': 1.54482e-6,
                        'A3': 2.54438e-4, 'lambda3': 1.13488e-3
                    },
                    'yy': {
                        'A0': 11.6858,
                        'A1': 0.939816,  'lambda1': 1.10795e-6,
                        'A2': 8.10461e-3, 'lambda2': 1.54482e-6, 
                        'A3': 2.54438e-4, 'lambda3': 1.13488e-3
                    },
                    'zz': {
                        'A0': 11.9242,   # Slightly different for z-direction
                        'A1': 1.007825, 'lambda1': 1.10795e-6,
                        'A2': 8.50615e-3, 'lambda2': 1.54482e-6,
                        'A3': 2.67166e-4, 'lambda3': 1.13488e-3
                    },
                    
                    # Off-diagonal tensor components (birefringence/dichroism)
                    'xy': {
                        'A0': 0.0,       # Small off-diagonal terms
                        'A1': 0.001,    'lambda1': 1.55e-6,
                        'A2': 0.0,      'lambda2': 0.0,
                        'A3': 0.0,      'lambda3': 0.0
                    },
                    'xz': {
                        'A0': 0.0,
                        'A1': 0.002,    'lambda1': 1.55e-6,
                        'A2': 0.0,      'lambda2': 0.0, 
                        'A3': 0.0,      'lambda3': 0.0
                    },
                    'yz': {
                        'A0': 0.0,
                        'A1': 0.001,    'lambda1': 1.55e-6,
                        'A2': 0.0,      'lambda2': 0.0,
                        'A3': 0.0,      'lambda3': 0.0
                    }
                },
                
                # Material properties
                'mesh_order': 2,
                'color': [0.7, 0.1, 0.1, 0.8]  # Red for silicon
            },
            
            # ==== BACKGROUND MATERIAL: Anisotropic SiO2 ====
            'background_material': {
                'name': 'sio2_anisotropic', 
                'type': 'sellmeier_anisotropic',
                'lumerical_material': 'SiO2 (Glass) - Anisotropic',
                
                # Sellmeier coefficients for SiO2 tensor components
                'sellmeier_coefficients': {
                    # Diagonal components (nearly isotropic for SiO2)
                    'xx': {
                        'A0': 1.0,
                        'A1': 0.6961663, 'lambda1': 0.0684043e-6,
                        'A2': 0.4079426, 'lambda2': 0.1162414e-6,
                        'A3': 0.8974794, 'lambda3': 9.896161e-6
                    },
                    'yy': {
                        'A0': 1.0,
                        'A1': 0.6961663, 'lambda1': 0.0684043e-6,
                        'A2': 0.4079426, 'lambda2': 0.1162414e-6,
                        'A3': 0.8974794, 'lambda3': 9.896161e-6
                    },
                    'zz': {
                        'A0': 1.0,
                        'A1': 0.6965210, 'lambda1': 0.0684043e-6,  # Slightly different
                        'A2': 0.4079426, 'lambda2': 0.1162414e-6,
                        'A3': 0.8974794, 'lambda3': 9.896161e-6
                    },
                    
                    # Off-diagonal components (minimal for SiO2)
                    'xy': {
                        'A0': 0.0,
                        'A1': 0.0,      'lambda1': 0.0,
                        'A2': 0.0,      'lambda2': 0.0,
                        'A3': 0.0,      'lambda3': 0.0
                    },
                    'xz': {
                        'A0': 0.0,
                        'A1': 0.0,      'lambda1': 0.0,
                        'A2': 0.0,      'lambda2': 0.0,
                        'A3': 0.0,      'lambda3': 0.0
                    },
                    'yz': {
                        'A0': 0.0,
                        'A1': 0.0,      'lambda1': 0.0,
                        'A2': 0.0,      'lambda2': 0.0,
                        'A3': 0.0,      'lambda3': 0.0
                    }
                },
                
                # Material properties
                'mesh_order': 3,
                'color': [0.1, 0.1, 0.7, 0.6]  # Blue for SiO2
            },
            
            # ==== FIXED STRUCTURE MATERIALS (Standard Lumerical) ====
            'substrate': {
                'name': 'substrate_silicon',
                'lumerical_material': 'Si (Silicon) - Palik',
                'mesh_order': 4
            },
            'cladding': {
                'name': 'cladding_sio2',
                'lumerical_material': 'SiO2 (Glass) - Palik', 
                'mesh_order': 5
            },
            'waveguide': {
                'name': 'waveguide_silicon',
                'lumerical_material': 'Si (Silicon) - Palik',
                'mesh_order': 2
            }
        }
    
    def calculate_sellmeier_permittivity(self, material_name, component, wavelength):
        """
        Calculate permittivity tensor component using Sellmeier equation.
        
        ε(λ) = A₀ + A₁λ²/(λ² - λ₁²) + A₂λ²/(λ² - λ₂²) + A₃λ²/(λ² - λ₃²)
        
        Parameters:
        -----------
        material_name : str
            'optimization_material' or 'background_material'
        component : str  
            'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
        wavelength : float
            Wavelength in meters
        """
        coeffs = self.materials[material_name]['sellmeier_coefficients'][component]
        
        wl_um = wavelength * 1e6  # Convert to micrometers
        wl2 = wl_um**2
        
        eps = coeffs['A0']
        
        if coeffs['A1'] != 0 and coeffs['lambda1'] != 0:
            lambda1_um = coeffs['lambda1'] * 1e6
            eps += coeffs['A1'] * wl2 / (wl2 - lambda1_um**2)
            
        if coeffs['A2'] != 0 and coeffs['lambda2'] != 0:
            lambda2_um = coeffs['lambda2'] * 1e6
            eps += coeffs['A2'] * wl2 / (wl2 - lambda2_um**2)
            
        if coeffs['A3'] != 0 and coeffs['lambda3'] != 0:
            lambda3_um = coeffs['lambda3'] * 1e6
            eps += coeffs['A3'] * wl2 / (wl2 - lambda3_um**2)
            
        return eps
    
    def get_material_tensor_at_wavelength(self, material_name, wavelength):
        """
        Get complete 3x3 permittivity tensor for a material at specific wavelength.
        
        Returns:
        --------
        tensor : dict
            Complete permittivity tensor with all components
        """
        tensor = {}
        components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        
        for comp in components:
            tensor[f'eps_{comp}'] = self.calculate_sellmeier_permittivity(
                material_name, comp, wavelength
            )
        
        return tensor
    
    def setup_simulation_domain(self):
        """FDTD simulation domain settings"""
        self.simulation = {
            'domain': {
                'x_min': -10.0e-6,
                'x_max': 10.0e-6,
                'y_min': -5.0e-6,
                'y_max': 5.0e-6,
                'z_min': -2.0e-6,
                'z_max': 2.0e-6
            },
            'mesh': {
                'global_dx': 50e-9,
                'global_dy': 50e-9,
                'global_dz': 25e-9,
                'design_region_dx': 25e-9,  # Finer mesh in design region
                'design_region_dy': 25e-9,
                'design_region_dz': 10e-9
            },
            'boundaries': {
                'x_min': 'PML',
                'x_max': 'PML', 
                'y_min': 'PML',
                'y_max': 'PML',
                'z_min': 'PML',
                'z_max': 'PML'
            },
            'pml_layers': 12
        }
    
    def setup_monitors(self):
        """Monitor configuration"""
        self.monitors = {
            'input_mode_expansion': {
                'name': 'input_monitor',
                'x': -7.0e-6,
                'y_span': 2.0e-6,
                'z_span': 1.0e-6
            },
            'output_mode_expansion': {
                'name': 'output_monitor',
                'x': 7.0e-6,
                'y_span': 2.0e-6,
                'z_span': 1.0e-6
            },
            'field_monitor': {
                'name': 'field_monitor',
                'x_span': 10.0e-6,
                'y_span': 4.0e-6,
                'z': 110e-9
            }
        }
    
    def setup_source(self):
        """Source configuration"""
        self.source = {
            'type': 'mode',
            'name': 'source',
            'x': -7.0e-6,
            'y': 0.0,
            'z': 110e-9,
            'y_span': 2.0e-6,
            'z_span': 1.0e-6,
            'wavelength_center': self.device_specs['wavelength']['center'],
            'wavelength_span': self.device_specs['wavelength']['span']
        }
    
    def setup_output_settings(self):
        """Output file settings"""
        self.output = {
            'fsp_file': {
                'filename': 'edge_coupler_shape_optimization.fsp',
                'directory': './',
                'update_each_iteration': True
            },
            'results': {
                'save_geometry': True,
                'save_parameters': True,
                'save_fom_history': True
            }
        }
    
    def create_lumerical_anisotropic_material(self, fdtd, material_name):
        """
        Create anisotropic material in Lumerical using Sellmeier coefficients.
        This creates a proper dispersive anisotropic material.
        """
        material_config = self.materials[material_name]
        lum_name = material_config['name']
        
        # Create anisotropic material
        fdtd.eval(f"""
        # Create anisotropic material: {lum_name}
        addmaterial("Anisotropic");
        set("name", "{lum_name}");
        """)
        
        # Add Sellmeier coefficients for each tensor component
        components = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
        
        for comp in components:
            coeffs = material_config['sellmeier_coefficients'][comp]
            
            # Set Sellmeier coefficients
            fdtd.eval(f"""
            select("{lum_name}");
            set("permittivity {comp}", 1);  # Enable tensor component
            set("sellmeier model {comp}", 1);  # Enable Sellmeier
            
            # Set coefficients
            set("A0 {comp}", {coeffs['A0']});
            set("A1 {comp}", {coeffs['A1']});
            set("lambda1 {comp}", {coeffs['lambda1']});
            set("A2 {comp}", {coeffs['A2']});
            set("lambda2 {comp}", {coeffs['lambda2']});
            set("A3 {comp}", {coeffs['A3']});
            set("lambda3 {comp}", {coeffs['lambda3']});
            """)
        
        # Set mesh order and other properties
        fdtd.eval(f"""
        select("{lum_name}");
        set("mesh order", {material_config['mesh_order']});
        """)
        
        print(f"Created anisotropic Sellmeier material: {lum_name}")
    
    def create_base_script(self):
        """
        Create base Lumerical script with fixed geometry and materials.
        Design region will be filled by topology.py with rectangle clustering.
        """
        def base_script_function(simulation_obj):
            fdtd = simulation_obj.fdtd
            
            # Clear existing objects
            fdtd.switchtolayout()
            fdtd.selectall()
            fdtd.delete()
            
            # Create custom anisotropic materials
            self.create_lumerical_anisotropic_material(fdtd, 'optimization_material')
            self.create_lumerical_anisotropic_material(fdtd, 'background_material')
            
            # Add FDTD solver
            fdtd.addfdtd()
            fdtd.set('x min', self.simulation['domain']['x_min'])
            fdtd.set('x max', self.simulation['domain']['x_max'])
            fdtd.set('y min', self.simulation['domain']['y_min'])
            fdtd.set('y max', self.simulation['domain']['y_max'])
            fdtd.set('z min', self.simulation['domain']['z_min'])
            fdtd.set('z max', self.simulation['domain']['z_max'])
            
            # Set boundaries
            for direction, bc in self.simulation['boundaries'].items():
                fdtd.set(f'{direction} bc', bc)
            
            # Set mesh
            fdtd.set('mesh accuracy', 4)
            fdtd.set('dx', self.simulation['mesh']['global_dx'])
            fdtd.set('dy', self.simulation['mesh']['global_dy'])
            fdtd.set('dz', self.simulation['mesh']['global_dz'])
            
            # Add substrate
            fdtd.addrect()
            fdtd.set('name', 'substrate')
            fdtd.set('x', 0)
            fdtd.set('x span', 20e-6)
            fdtd.set('y', 0)
            fdtd.set('y span', 12e-6)
            fdtd.set('z', -1e-6)
            fdtd.set('z span', 2e-6)
            fdtd.set('material', self.materials['substrate']['lumerical_material'])
            
            # Add BOX layer
            fdtd.addrect()
            fdtd.set('name', 'box')
            fdtd.set('x', 0)
            fdtd.set('x span', 20e-6)
            fdtd.set('y', 0)
            fdtd.set('y span', 12e-6)
            fdtd.set('z', 110e-9)
            fdtd.set('z span', 2e-6)
            fdtd.set('material', self.materials['cladding']['lumerical_material'])
            
            # Add input waveguide
            fdtd.addrect()
            fdtd.set('name', 'input_waveguide')
            fdtd.set('x', -6e-6)
            fdtd.set('x span', 6e-6)
            fdtd.set('y', 0)
            fdtd.set('y span', 500e-9)
            fdtd.set('z', 110e-9)
            fdtd.set('z span', 220e-9)
            fdtd.set('material', self.materials['waveguide']['lumerical_material'])
            
            # Add output waveguide
            fdtd.addrect()
            fdtd.set('name', 'output_waveguide')
            fdtd.set('x', 6e-6)
            fdtd.set('x span', 6e-6)
            fdtd.set('y', 0)
            fdtd.set('y span', 500e-9)
            fdtd.set('z', 110e-9)
            fdtd.set('z span', 220e-9)
            fdtd.set('material', self.materials['waveguide']['lumerical_material'])
            
            # Add cladding
            fdtd.addrect()
            fdtd.set('name', 'cladding')
            fdtd.set('x', 0)
            fdtd.set('x span', 20e-6)
            fdtd.set('y', 0)
            fdtd.set('y span', 12e-6)
            fdtd.set('z', 1.11e-6)
            fdtd.set('z span', 2e-6)
            fdtd.set('material', self.materials['cladding']['lumerical_material'])
            
            # Add source
            fdtd.addmode()
            fdtd.set('name', self.source['name'])
            fdtd.set('x', self.source['x'])
            fdtd.set('y', self.source['y'])
            fdtd.set('z', self.source['z'])
            fdtd.set('y span', self.source['y_span'])
            fdtd.set('z span', self.source['z_span'])
            fdtd.set('center wavelength', self.source['wavelength_center'])
            fdtd.set('wavelength span', self.source['wavelength_span'])
            
            # Add monitors
            for monitor_type, config in self.monitors.items():
                if monitor_type in ['input_mode_expansion', 'output_mode_expansion']:
                    fdtd.addpower()
                    fdtd.set('name', config['name'])
                    fdtd.set('monitor type', 'Point')
                    fdtd.set('x', config['x'])
                    fdtd.set('y', 0)
                    fdtd.set('z', 110e-9)
                    fdtd.set('y span', config['y_span'])
                    fdtd.set('z span', config['z_span'])
                    
                elif monitor_type == 'field_monitor':
                    fdtd.addpower()
                    fdtd.set('name', config['name'])
                    fdtd.set('monitor type', 'Point')
                    fdtd.set('x', 0)
                    fdtd.set('y', 0)
                    fdtd.set('z', config['z'])
                    fdtd.set('x span', config['x_span'])
                    fdtd.set('y span', config['y_span'])
            
            # Add fine mesh in design region
            fdtd.addmesh()
            fdtd.set('name', 'design_region_mesh')
            fdtd.set('x', self.design_region['volume']['x_center'])
            fdtd.set('y', self.design_region['volume']['y_center'])
            fdtd.set('z', self.design_region['volume']['z_center'])
            fdtd.set('x span', self.design_region['volume']['length'])
            fdtd.set('y span', self.design_region['volume']['width'])
            fdtd.set('z span', self.design_region['volume']['thickness'])
            fdtd.set('dx', self.simulation['mesh']['design_region_dx'])
            fdtd.set('dy', self.simulation['mesh']['design_region_dy'])
            fdtd.set('dz', self.simulation['mesh']['design_region_dz'])
            
            print("Base simulation created with anisotropic Sellmeier materials")
            print("Design region ready for rectangle clustering")
            
        return base_script_function
    
    def print_design_summary(self):
        """Print summary of design region and material configuration"""
        vol = self.design_region['volume']
        constraints = self.design_region['constraints']
        
        print("\n" + "="*70)
        print("SHAPE OPTIMIZATION CONFIGURATION")
        print("="*70)
        print(f"Design region: {vol['length']*1e6:.1f} × {vol['width']*1e6:.1f} × {vol['thickness']*1e9:.0f} μm³")
        print(f"Bounds: x=[{vol['x_min']*1e6:.1f}, {vol['x_max']*1e6:.1f}] μm")
        print(f"        y=[{vol['y_min']*1e6:.1f}, {vol['y_max']*1e6:.1f}] μm")
        print(f"        z=[{vol['z_min']*1e9:.0f}, {vol['z_max']*1e9:.0f}] nm")
        print(f"Min feature size: {constraints['min_feature_size']*1e9:.0f} nm")
        print(f"Optimization type: {constraints['optimization_type']}")
        
        print(f"\nMaterials:")
        print(f"  Optimization: {self.materials['optimization_material']['name']} (anisotropic Sellmeier)")
        print(f"  Background:   {self.materials['background_material']['name']} (anisotropic Sellmeier)")
        
        # Show material properties at center wavelength
        wl_center = self.device_specs['wavelength']['center']
        print(f"\nMaterial properties at {wl_center*1e9:.0f} nm:")
        
        for mat_name in ['optimization_material', 'background_material']:
            tensor = self.get_material_tensor_at_wavelength(mat_name, wl_center)
            name = self.materials[mat_name]['name']
            print(f"  {name}:")
            print(f"    εxx = {tensor['eps_xx']:.3f}")
            print(f"    εyy = {tensor['eps_yy']:.3f}")
            print(f"    εzz = {tensor['eps_zz']:.3f}")
            print(f"    εxy = {tensor['eps_xy']:.6f}")
        
        print("="*70)

# ======================================================================
# GLOBAL CONFIGURATION ACCESS
# ======================================================================

# Create global device configuration instance
device_config = DeviceConfig()

def get_device_config():
    """Return the global device configuration"""
    return device_config

def get_design_region():
    """Return design region specification for topology.py"""
    return device_config.design_region

def get_material_properties():
    """Return complete material definitions for topology.py"""
    return device_config.materials

def get_sellmeier_functions():
    """Return functions to calculate material properties at any wavelength"""
    return {
        'calculate_tensor': device_config.get_material_tensor_at_wavelength,
        'calculate_component': device_config.calculate_sellmeier_permittivity
    }

def create_base_script():
    """Return base script function for simulation setup"""
    return device_config.create_base_script()

def get_monitor_names():
    """Return monitor names for FOM integration"""
    return {
        'input_monitor': device_config.monitors['input_mode_expansion']['name'],
        'output_monitor': device_config.monitors['output_mode_expansion']['name'],
        'field_monitor': device_config.monitors['field_monitor']['name']
    }

def get_output_settings():
    """Return output file settings"""
    return device_config.output

def get_wavelength_settings():
    """Return wavelength configuration"""
    return device_config.device_specs['wavelength']

# Print configuration on import
if __name__ == "__main__":
    device_config.print_design_summary()
