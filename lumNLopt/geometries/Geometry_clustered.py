# Geometry_clustered.py - Rectangle Clustering Implementation
# Handles actual rectangle creation, material assignment, and Lumerical simulation updates

import numpy as np
import lumapi
import matplotlib.pyplot as plt
from lumNLopt.Inputs.Device import get_design_region, get_material_properties, get_sellmeier_functions

class RectangleClusteredGeometry:
    """
    Rectangle clustering geometry that creates discrete rectangles from fractional width parameters.
    Handles all anisotropic material assignments and Lumerical simulation updates.
    """
    
    def __init__(self):
        """Initialize from Device.py configuration"""
        self.design_region = get_design_region()
        self.materials = get_material_properties()
        self.sellmeier_functions = get_sellmeier_functions()
        
        self.setup_design_region()
        self.setup_rectangle_framework()
        self.current_rectangles = []
        self.rectangle_objects = []  # Lumerical object names
        
        print(f"Rectangle clustering initialized:")
        print(f"  Design region: {self.length*1e6:.1f} × {self.width*1e6:.1f} × {self.thickness*1e9:.0f} μm³")
        print(f"  Min feature: {self.min_feature_size*1e9:.0f} nm")
        print(f"  Max rectangles: {self.max_rectangles}")
    
    def setup_design_region(self):
        """Extract design region parameters from Device.py"""
        volume = self.design_region['volume']
        constraints = self.design_region['constraints']
        
        # Design region bounds
        self.x_center = volume['x_center']
        self.y_center = volume['y_center']
        self.z_center = volume['z_center']
        
        self.length = volume['length']      # Total x-span
        self.width = volume['width']        # Total y-span  
        self.thickness = volume['thickness'] # Total z-span
        
        self.x_min = volume['x_min']
        self.x_max = volume['x_max']
        self.y_min = volume['y_min']
        self.y_max = volume['y_max']
        self.z_min = volume['z_min']
        self.z_max = volume['z_max']
        
        # Constraints
        self.min_feature_size = constraints['min_feature_size']
        self.num_layers = constraints['num_layers']
        
        # Material names
        self.optimization_material_name = self.materials['optimization_material']['name']
        self.background_material_name = self.materials['background_material']['name']
    
    def setup_rectangle_framework(self):
        """Calculate maximum rectangles and setup clustering framework"""
        # Calculate maximum number of rectangles based on minimum feature size
        self.max_rectangles = max(2, int(np.floor(self.length / self.min_feature_size)))
        
        # Parameter bounds for fractional widths
        # Each fraction must be >= min_fraction and sum must = 1
        min_fraction = max(1e-4, self.min_feature_size / self.length)
        max_fraction = 1.0 - (self.max_rectangles - 1) * min_fraction
        
        self.bounds = [(min_fraction, max_fraction)] * self.max_rectangles
        
        # Initial equal-width rectangles
        initial_fractions = np.ones(self.max_rectangles) / self.max_rectangles
        self.current_fractions = initial_fractions
        
        print(f"  Parameter bounds: [{min_fraction:.4f}, {max_fraction:.4f}]")
        print(f"  Initial fractions: {initial_fractions}")
    
    def update_rectangles_from_fractions(self, fractional_widths):
        """
        Main method: Convert fractional widths to rectangle geometries.
        Called by geometry_parameters_handling.
        """
        # Validate and normalize fractions
        fractions = self.validate_and_normalize_fractions(fractional_widths)
        self.current_fractions = fractions
        
        # Generate rectangle list
        self.current_rectangles = self._generate_rectangles(fractions)
        
        print(f"Updated {len(self.current_rectangles)} rectangles from fractions")
        return self.current_rectangles
    
    def validate_and_normalize_fractions(self, fractions):
        """Ensure fractions are valid and sum to 1"""
        fractions_array = np.array(fractions)
        
        # Check bounds
        for i, (frac, (min_val, max_val)) in enumerate(zip(fractions_array, self.bounds)):
            if frac < min_val or frac > max_val:
                print(f"Warning: Fraction {i} = {frac:.4f} outside bounds [{min_val:.4f}, {max_val:.4f}]")
        
        # Normalize to ensure sum = 1
        frac_sum = np.sum(fractions_array)
        if abs(frac_sum - 1.0) > 1e-6:
            print(f"Normalizing fractions: sum = {frac_sum:.6f} → 1.0")
            fractions_normalized = fractions_array / frac_sum
        else:
            fractions_normalized = fractions_array
            
        return fractions_normalized
    
    def _generate_rectangles(self, fractions):
        """Generate rectangle coordinates and materials from fractional widths"""
        rectangles = []
        
        # Calculate rectangle widths
        rectangle_widths = fractions * self.length
        
        # Generate rectangles sequentially along x-direction
        current_x = self.x_min
        
        for i, rect_width in enumerate(rectangle_widths):
            # Rectangle bounds
            x_start = current_x
            x_end = current_x + rect_width
            
            # Determine material (alternating pattern)
            if i % 2 == 0:
                material_name = self.optimization_material_name
                material_config = self.materials['optimization_material']
            else:
                material_name = self.background_material_name
                material_config = self.materials['background_material']
            
            # Create rectangle dictionary
            rectangle = {
                'index': i,
                'x_min': x_start,
                'x_max': x_end,
                'y_min': self.y_min,
                'y_max': self.y_max,
                'z_min': self.z_min,
                'z_max': self.z_max,
                'width': rect_width,
                'material_name': material_name,
                'material_config': material_config,
                'lumerical_name': f'rectangle_cluster_{i}'
            }
            
            rectangles.append(rectangle)
            current_x += rect_width
        
        return rectangles
    
    def update_lumerical_simulation(self, fdtd):
        """
        Update Lumerical simulation with current rectangle configuration.
        This is where anisotropic materials and geometry are applied.
        """
        # Switch to layout mode
        fdtd.switchtolayout()
        
        # Remove existing rectangle cluster objects
        self._remove_existing_rectangles(fdtd)
        
        # Add current rectangles
        for rect in self.current_rectangles:
            self._add_rectangle_to_lumerical(fdtd, rect)
        
        print(f"Updated Lumerical simulation with {len(self.current_rectangles)} rectangles")
    
    def _remove_existing_rectangles(self, fdtd):
        """Remove existing rectangle cluster objects from simulation"""
        for obj_name in self.rectangle_objects:
            try:
                fdtd.select(obj_name)
                fdtd.delete()
            except:
                pass  # Object may not exist
        
        self.rectangle_objects = []
    
    def _add_rectangle_to_lumerical(self, fdtd, rectangle):
        """Add a single rectangle with anisotropic material to Lumerical"""
        rect_name = rectangle['lumerical_name']
        
        # Add rectangle geometry
        fdtd.addrect()
        fdtd.set('name', rect_name)
        
        # Set position and size
        fdtd.set('x', (rectangle['x_min'] + rectangle['x_max']) / 2)
        fdtd.set('x span', rectangle['x_max'] - rectangle['x_min'])
        fdtd.set('y', (rectangle['y_min'] + rectangle['y_max']) / 2)
        fdtd.set('y span', rectangle['y_max'] - rectangle['y_min'])
        fdtd.set('z', (rectangle['z_min'] + rectangle['z_max']) / 2)
        fdtd.set('z span', rectangle['z_max'] - rectangle['z_min'])
        
        # Set anisotropic material
        material_name = rectangle['material_name']
        fdtd.set('material', material_name)
        
        # Set mesh order
        mesh_order = rectangle['material_config']['mesh_order']
        fdtd.set('mesh order', mesh_order)
        
        # Store object name for future removal
        self.rectangle_objects.append(rect_name)
        
        print(f"Added rectangle {rectangle['index']}: {material_name} ({rectangle['width']*1e9:.0f} nm)")
    
    def get_current_fractions(self):
        """Return current fractional widths"""
        return self.current_fractions.copy()
    
    def get_bounds(self):
        """Return parameter bounds for optimizer"""
        return self.bounds
    
    def get_rectangle_count(self):
        """Return number of rectangles (= number of parameters)"""
        return self.max_rectangles
    
    def calculate_material_properties_at_wavelength(self, wavelength):
        """Calculate material properties for all rectangles at specific wavelength"""
        material_props = {}
        
        for material_type in ['optimization_material', 'background_material']:
            tensor = self.sellmeier_functions['calculate_tensor'](material_type, wavelength)
            material_props[material_type] = tensor
            
        return material_props
    
    def plot_rectangle_configuration(self, ax=None, show_materials=True):
        """Plot current rectangle configuration"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.clear()
        
        # Plot each rectangle
        for rect in self.current_rectangles:
            # Color based on material
            if rect['material_name'] == self.optimization_material_name:
                color = 'red'
                alpha = 0.7
                label = 'Optimization'
            else:
                color = 'blue' 
                alpha = 0.5
                label = 'Background'
            
            # Draw rectangle
            width_um = rect['width'] * 1e6
            height_um = self.width * 1e6
            
            rect_patch = plt.Rectangle(
                (rect['x_min'] * 1e6, rect['y_min'] * 1e6),
                width_um, height_um,
                facecolor=color, alpha=alpha, edgecolor='black', linewidth=1,
                label=label if rect['index'] == 0 or rect['index'] == 1 else ""
            )
            ax.add_patch(rect_patch)
            
            # Add rectangle index
            x_center = (rect['x_min'] + rect['x_max']) * 0.5 * 1e6
            y_center = (rect['y_min'] + rect['y_max']) * 0.5 * 1e6
            ax.text(x_center, y_center, f"{rect['index']}", 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_xlim(self.x_min * 1e6, self.x_max * 1e6)
        ax.set_ylim(self.y_min * 1e6, self.y_max * 1e6)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_title(f'Rectangle Clustering: {self.max_rectangles} rectangles')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if show_materials:
            ax.legend()
        
        # Add parameter info
        param_text = f"Fractions: {np.array2string(self.current_fractions, precision=3)}"
        param_text += f"\nSum: {np.sum(self.current_fractions):.6f}"
        ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return True
    
    def save_configuration(self, filename):
        """Save current rectangle configuration"""
        save_data = {
            'fractions': self.current_fractions,
            'rectangles': self.current_rectangles,
            'design_region': self.design_region,
            'materials': self.materials,
            'bounds': self.bounds
        }
        
        np.savez(filename, **save_data)
        print(f"Rectangle configuration saved to {filename}")
    
    def print_summary(self):
        """Print summary of current configuration"""
        print("\n" + "="*60)
        print("RECTANGLE CLUSTERING SUMMARY")
        print("="*60)
        print(f"Design region: {self.length*1e6:.1f} × {self.width*1e6:.1f} × {self.thickness*1e9:.0f} μm")
        print(f"Rectangle count: {len(self.current_rectangles)}")
        print(f"Min feature size: {self.min_feature_size*1e9:.0f} nm")
        print(f"Current fractions: {self.current_fractions}")
        print(f"Fraction sum: {np.sum(self.current_fractions):.6f}")
        
        print("\nRectangle details:")
        for i, rect in enumerate(self.current_rectangles):
            width_nm = rect['width'] * 1e9
            print(f"  [{i}] {width_nm:.0f} nm - {rect['material_name']}")
        
        print("="*60)
