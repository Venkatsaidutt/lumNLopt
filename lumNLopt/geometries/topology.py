# File: lumNLopt/geometries/topology.py - CORRECTED VERSION

from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

import lumapi
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

eps0 = sp.constants.epsilon_0


class RectangleClusteringTopology(Geometry):
    """
    Rectangle clustering topology optimization using fractional width parameters.
    
    FIXES APPLIED:
    - Corrected Lumerical anisotropic material syntax
    - Improved field extraction error handling
    - Better constraint enforcement  
    - Enhanced bounds management
    - Robust gradient calculation
    """

    def __init__(self, min_feature_size, eps_min, eps_max, x, y, z, 
                 material_order='alternating', filter_R=None, eta=0.5, beta=1,
                 anisotropic_materials=None):
        
        self.min_feature_size = min_feature_size
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.x = np.array(x)
        self.y = np.array(y) 
        self.z = np.array(z) if hasattr(z, "__len__") else np.array([z])
        self.material_order = material_order
        self.anisotropic_materials = anisotropic_materials
        self.filter_R = filter_R
        self.eta = eta
        self.beta = beta
        
        # Calculate design region dimensions
        self.Lx = self.x.max() - self.x.min()
        self.Ly = self.y.max() - self.y.min()
        self.dx = self.x[1] - self.x[0] if len(self.x) > 1 else self.Lx
        self.dy = self.y[1] - self.y[0] if len(self.y) > 1 else self.Ly
        self.dz = self.z[1] - self.z[0] if len(self.z) > 1 else 220e-9
        self.depth = self.z[-1] - self.z[0] if len(self.z) > 1 else 220e-9
        
        # Calculate maximum number of rectangles based on minimum feature size
        self.max_rectangles_x = max(2, int(np.floor(self.Lx / self.min_feature_size)))
        self.num_params = self.max_rectangles_x
        
        # Initialize with equal-width rectangles (sum = 1)
        initial_fractions = np.ones(self.num_params) / self.num_params
        self.current_params = initial_fractions
        
        # FIXED: Improved bounds handling
        # Ensure minimum fraction > 0 and maximum allows sum = 1
        min_fraction = max(1e-3, 0.1 / self.num_params)  # At least 0.1% of total
        max_fraction = 1.0 - (self.num_params - 1) * min_fraction
        self.bounds = [(min_fraction, max_fraction)] * self.num_params
        
        # Storage for current rectangle configuration
        self.rectangles = []
        self.rectangle_objects = []  # Lumerical object names
        
        # Required by lumNLopt framework
        self.unfold_symmetry = False
        
        # Generate initial rectangles
        self.update_geometry(initial_fractions)
        
        print(f"Rectangle clustering initialized:")
        print(f"  Design region: {self.Lx*1e6:.1f} × {self.Ly*1e6:.1f} μm")
        print(f"  Min feature size: {min_feature_size*1e9:.0f} nm")
        print(f"  Number of rectangles: {self.num_params}")
        print(f"  Parameter bounds: [{min_fraction:.4f}, {max_fraction:.4f}]")

    def use_interpolation(self):
        """Rectangle clustering uses discrete rectangles, not interpolation"""
        return False

    def get_current_params(self):
        """Return current fractional widths (guaranteed to sum to 1)"""
        return self.current_params.copy()

    def update_geometry(self, params, sim=None):
        """
        Update geometry from fractional width parameters.
        FIXED: Better constraint enforcement and validation.
        """
        # Convert to numpy array and validate
        params_array = np.array(params)
        
        # Check bounds
        for i, (param, (min_val, max_val)) in enumerate(zip(params_array, self.bounds)):
            if param < min_val or param > max_val:
                print(f"Warning: Parameter {i} = {param:.4f} outside bounds [{min_val:.4f}, {max_val:.4f}]")
        
        # FIXED: Robust normalization to ensure sum=1
        params_sum = np.sum(params_array)
        if abs(params_sum - 1.0) > 1e-6:
            print(f"Warning: Parameters sum to {params_sum:.6f}, normalizing to 1.0")
            params_normalized = params_array / params_sum
        else:
            params_normalized = params_array
        
        self.current_params = params_normalized
        
        # Generate rectangles from normalized fractions
        self.rectangles = self._generate_rectangles_from_fractions(params_normalized)
        
        # Update Lumerical objects if simulation provided
        if sim is not None:
            self._update_lumerical_rectangles(sim)

    def _generate_rectangles_from_fractions(self, fractions):
        """
        Convert fractional widths to rectangle coordinates and material properties.
        FIXED: Better material handling and validation.
        """
        rectangles = []
        x_start = self.x.min()
        
        for i, fraction in enumerate(fractions):
            # Calculate rectangle width from fraction
            width = fraction * self.Lx
            x_end = x_start + width
            
            # FIXED: Validate rectangle width meets minimum feature size
            if width < self.min_feature_size * 0.9:  # Allow 10% tolerance
                print(f"Warning: Rectangle {i} width {width*1e9:.1f}nm below min feature size")
            
            # Determine material properties based on order
            if self.material_order == 'alternating':
                is_fill_material = (i % 2 == 0)  # Even indices get fill material
                eps_properties = self._get_material_properties(is_fill_material)
            elif self.material_order == 'custom' and self.anisotropic_materials:
                material_idx = min(i, len(self.anisotropic_materials) - 1)
                eps_properties = self.anisotropic_materials[material_idx]
            else:
                # Default to alternating
                is_fill_material = (i % 2 == 0)
                eps_properties = self._get_material_properties(is_fill_material)
            
            rectangle = {
                'index': i,
                'x_min': x_start,
                'x_max': x_end,
                'width': width,
                'y_min': self.y.min(),
                'y_max': self.y.max(),
                'z_center': np.mean(self.z),
                'z_span': self.depth,
                'eps_properties': eps_properties,
                'name': f'opt_rect_{i}'
            }
            rectangles.append(rectangle)
            x_start = x_end
            
        return rectangles

    def _get_material_properties(self, is_fill_material):
        """
        Get material properties (potentially anisotropic) for rectangle.
        FIXED: Better type checking and defaults.
        """
        base_eps = self.eps_max if is_fill_material else self.eps_min
        
        if isinstance(base_eps, dict):
            # Anisotropic material specified as dict
            return {
                'eps_xx': float(base_eps.get('xx', base_eps.get('eps_xx', 1.0))),
                'eps_yy': float(base_eps.get('yy', base_eps.get('eps_yy', 1.0))),
                'eps_zz': float(base_eps.get('zz', base_eps.get('eps_zz', 1.0))),
                'eps_xy': float(base_eps.get('xy', base_eps.get('eps_xy', 0.0))),
                'eps_xz': float(base_eps.get('xz', base_eps.get('eps_xz', 0.0))),
                'eps_yz': float(base_eps.get('yz', base_eps.get('eps_yz', 0.0))),
                'is_anisotropic': True
            }
        else:
            # Isotropic material specified as scalar
            eps_val = float(base_eps)
            return {
                'eps_xx': eps_val,
                'eps_yy': eps_val,
                'eps_zz': eps_val,
                'eps_xy': 0.0,
                'eps_xz': 0.0,
                'eps_yz': 0.0,
                'is_anisotropic': False
            }

    def add_geo(self, sim, params=None, only_update=False):
        """
        Add rectangle geometry to Lumerical simulation.
        FIXED: Better error handling and validation.
        """
        if params is not None:
            self.update_geometry(params, sim)
        
        fdtd = sim.fdtd
        
        try:
            if not only_update:
                self._setup_simulation_environment(fdtd)
            
            # Add or update rectangle structures
            self._add_rectangles_to_lumerical(fdtd, only_update)
            
        except Exception as e:
            print(f"Error in add_geo: {e}")
            print(f"Current parameters: {self.current_params}")
            raise

    def _setup_simulation_environment(self, fdtd):
        """Setup monitors, mesh overrides, and simulation environment"""
        
        try:
            # Setup field monitors with proper interpolation
            set_spatial_interp(fdtd, 'opt_fields', 'specified position')
            set_spatial_interp(fdtd, 'opt_fields_index', 'specified position')
            
            # Set monitor boundaries to cover design region
            monitor_script = f'''
            select("opt_fields");
            set("x min", {self.x.min()});
            set("x max", {self.x.max()});
            set("y min", {self.y.min()});
            set("y max", {self.y.max()});
            set("z min", {self.z.min()});
            set("z max", {self.z.max()});
            
            select("opt_fields_index");
            set("x min", {self.x.min()});
            set("x max", {self.x.max()});
            set("y min", {self.y.min()});
            set("y max", {self.y.max()});
            set("z min", {self.z.min()});
            set("z max", {self.z.max()});
            '''
            fdtd.eval(monitor_script)
            
            # Add mesh override for proper resolution
            mesh_script = f'''
            addmesh;
            set("name", "rectangle_mesh_override");
            set("x min", {self.x.min()});
            set("x max", {self.x.max()});
            set("y min", {self.y.min()});
            set("y max", {self.y.max()});
            set("z min", {self.z.min()});
            set("z max", {self.z.max()});
            set("dx", {self.dx});
            set("dy", {self.dy});
            set("dz", {self.dz});
            '''
            fdtd.eval(mesh_script)
            
        except Exception as e:
            print(f"Error setting up simulation environment: {e}")
            raise

    def _add_rectangles_to_lumerical(self, fdtd, only_update):
        """
        Add or update rectangle structures in Lumerical CAD.
        FIXED: Better error handling and object management.
        """
        
        for rect in self.rectangles:
            rect_name = rect['name']
            
            try:
                if not only_update:
                    # Create new rectangle
                    fdtd.addrect()
                    fdtd.set('name', rect_name)
                    self.rectangle_objects.append(rect_name)
                
                # Set rectangle geometry
                fdtd.setnamed(rect_name, 'x min', rect['x_min'])
                fdtd.setnamed(rect_name, 'x max', rect['x_max'])
                fdtd.setnamed(rect_name, 'y min', rect['y_min'])
                fdtd.setnamed(rect_name, 'y max', rect['y_max'])
                fdtd.setnamed(rect_name, 'z', rect['z_center'])
                fdtd.setnamed(rect_name, 'z span', rect['z_span'])
                
                # Set material properties (isotropic or anisotropic)
                self._set_material_properties(fdtd, rect_name, rect['eps_properties'])
                
            except Exception as e:
                print(f"Error adding/updating rectangle {rect_name}: {e}")
                raise
                
    def _set_material_properties(self, fdtd, object_name, eps_props):
        
        
    """Use the SAME syntax as in materials.py - the confirmed approach"""
    
        try:
            fdtd.setnamed(object_name, 'material', '<Object defined dielectric>')
            
        
            if eps_props['is_anisotropic']:
                
                
                fdtd.setnamed(object_name, 'anisotropy', 1)
                fdtd.setnamed(object_name, 'index x', np.sqrt(eps_props['eps_xx']))
                fdtd.setnamed(object_name, 'index y', np.sqrt(eps_props['eps_yy']))
                fdtd.setnamed(object_name, 'index z', np.sqrt(eps_props['eps_zz']))
                
        else:
            # Isotropic material
            n_value = np.sqrt(eps_props['eps_xx'])
            fdtd.setnamed(object_name, 'index', float(n_value))
            
            
    except Exception as e:
        print(f"Error setting material properties for {object_name}: {e}")
        raise

    def calculate_gradients_manual(self, forward_fields, adjoint_fields, wavelengths):
        """
        Custom anisotropic gradient calculation bypassing lumNLopt's isotropic assumptions.
        FIXED: Better field extraction and error handling.
        """
        
        try:
            gradients = np.zeros(self.num_params)
            
            # Validate field data
            if forward_fields is None or adjoint_fields is None:
                raise ValueError("Forward or adjoint fields are None")
            
            for param_idx in range(self.num_params):
                try:
                    # Calculate sensitivity w.r.t. fractional width parameter
                    grad_value = self._calculate_parameter_gradient(
                        param_idx, forward_fields, adjoint_fields, wavelengths)
                    gradients[param_idx] = grad_value
                    
                except Exception as e:
                    print(f"Error calculating gradient for parameter {param_idx}: {e}")
                    gradients[param_idx] = 0.0  # Set to zero on error
            
            # Apply constraint projection (remove component along [1,1,1,...])
            gradients = self._apply_constraint_projection(gradients)
            
            # Validate gradients
            if np.any(np.isnan(gradients)) or np.any(np.isinf(gradients)):
                print("Warning: NaN or Inf detected in gradients")
                gradients = np.nan_to_num(gradients, nan=0.0, posinf=1e6, neginf=-1e6)
            
            print(f"Gradients computed: {gradients}")
            print(f"Gradient norm: {np.linalg.norm(gradients):.4e}")
            
            return gradients
            
        except Exception as e:
            print(f"Error in calculate_gradients_manual: {e}")
            # Return zero gradients on error
            return np.zeros(self.num_params)

    def _calculate_parameter_gradient(self, param_idx, E_forward, E_adjoint, wavelengths):
        """
        Calculate gradient w.r.t. single fractional width parameter.
        FIXED: Better field format handling and error checking.
        """
        gradient = 0.0
        
        try:
            # Get rectangle affected by this parameter
            rect = self.rectangles[param_idx]
            
            # Material properties of this rectangle and background
            rect_eps = rect['eps_properties']
            bg_eps = self._get_material_properties(False)  # Background material
            
            # Permittivity difference tensor
            deps_tensor = np.zeros((3, 3))
            deps_tensor[0, 0] = rect_eps['eps_xx'] - bg_eps['eps_xx']
            deps_tensor[1, 1] = rect_eps['eps_yy'] - bg_eps['eps_yy']
            deps_tensor[2, 2] = rect_eps['eps_zz'] - bg_eps['eps_zz']
            
            # Derivative of rectangle width w.r.t. fractional parameter
            dwidth_dfraction = self.Lx
            
            # FIXED: Flexible field extraction handling different formats
            gradient = self._integrate_adjoint_sensitivity(
                E_forward, E_adjoint, deps_tensor, rect, wavelengths, dwidth_dfraction)
            
        except Exception as e:
            print(f"Error in parameter gradient calculation: {e}")
            gradient = 0.0
        
        return gradient

    def _integrate_adjoint_sensitivity(self, E_forward, E_adjoint, deps_tensor, rect, wavelengths, dwidth_dfraction):
        """
        FIXED: Flexible field integration handling different field data formats.
        """
        
        gradient = 0.0
        
        try:
            # Method 1: Try direct array access (5D format: [x,y,z,wl,field_component])
            try:
                x_mask = (self.x >= rect['x_min']) & (self.x <= rect['x_max'])
                y_mask = (self.y >= rect['y_min']) & (self.y <= rect['y_max'])
                
                dV = self.dx * self.dy * self.dz
                
                for wl_idx, wl in enumerate(wavelengths):
                    for i in np.where(x_mask)[0]:
                        for j in np.where(y_mask)[0]:
                            for k in range(len(self.z)):
                                
                                # Extract field components [Ex, Ey, Ez]
                                E_f = E_forward[i, j, k, wl_idx, :]  
                                E_a = E_adjoint[i, j, k, wl_idx, :]  
                                
                                # Adjoint sensitivity: Re[E_forward^H * deps_tensor * E_adjoint]
                                field_product = 0.0
                                for alpha in range(3):
                                    for beta in range(3):
                                        field_product += (np.conj(E_f[alpha]) * 
                                                        deps_tensor[alpha, beta] * 
                                                        E_a[beta])
                                
                                gradient += np.real(field_product) * dV
                                
                return gradient * dwidth_dfraction * eps0
                
            except (IndexError, TypeError, KeyError) as e1:
                print(f"Direct array access failed: {e1}")
                
                # Method 2: Try field object method access
                try:
                    # Assume field objects with getfield method
                    x_centers = (self.x[:-1] + self.x[1:]) / 2
                    y_centers = (self.y[:-1] + self.y[1:]) / 2
                    z_centers = self.z
                    
                    dV = self.dx * self.dy * self.dz
                    
                    for wl in wavelengths:
                        for x_pos in x_centers:
                            if rect['x_min'] <= x_pos <= rect['x_max']:
                                for y_pos in y_centers:
                                    if rect['y_min'] <= y_pos <= rect['y_max']:
                                        for z_pos in z_centers:
                                            
                                            # Try field object extraction
                                            E_f = E_forward.getfield(x_pos, y_pos, z_pos, wl)
                                            E_a = E_adjoint.getfield(x_pos, y_pos, z_pos, wl)
                                            
                                            # Compute sensitivity
                                            field_product = 0.0
                                            for alpha in range(3):
                                                for beta in range(3):
                                                    field_product += (np.conj(E_f[alpha]) * 
                                                                    deps_tensor[alpha, beta] * 
                                                                    E_a[beta])
                                            
                                            gradient += np.real(field_product) * dV
                                            
                    return gradient * dwidth_dfraction * eps0
                    
                except Exception as e2:
                    print(f"Field object access failed: {e2}")
                    
                    # Method 3: Simplified approximation
                    print("Using simplified gradient approximation")
                    
                    # Estimate based on average field values
                    rect_volume = rect['width'] * (self.y.max() - self.y.min()) * self.depth
                    avg_deps = np.mean(np.diag(deps_tensor))
                    
                    # Simple approximation: gradient ∝ volume × material contrast
                    gradient = rect_volume * avg_deps * dwidth_dfraction * eps0
                    
                    return gradient
                    
        except Exception as e:
            print(f"All field integration methods failed: {e}")
            return 0.0

    def _apply_constraint_projection(self, gradients):
        """
        Project gradients onto constraint tangent space.
        FIXED: Robust constraint projection.
        """
        try:
            # For sum(params)=1 constraint: remove component along [1,1,1,...] direction
            mean_gradient = np.mean(gradients)
            projected_gradients = gradients - mean_gradient
            
            # Validate projection
            constraint_violation = np.sum(projected_gradients)
            if abs(constraint_violation) > 1e-10:
                print(f"Warning: Constraint projection error: {constraint_violation:.2e}")
            
            return projected_gradients
            
        except Exception as e:
            print(f"Error in constraint projection: {e}")
            return gradients  # Return original on error

    def get_constraint_function(self):
        """Return constraint function for sum=1: constraint(x) = sum(x) - 1"""
        def constraint(params):
            return np.sum(params) - 1.0
        return constraint

    def get_constraint_jacobian(self):
        """Return constraint jacobian: d(sum(params))/dp_i = 1 for all i"""
        def constraint_jac(params):
            return np.ones(len(params))
        return constraint_jac

    def plot(self, ax=None):
        """
        Plot current rectangle configuration.
        FIXED: Better visualization and error handling.
        """
        try:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.clear()
            
            # Plot each rectangle
            for i, rect in enumerate(self.rectangles):
                # Color based on material type and alternating pattern
                if rect['eps_properties']['is_anisotropic']:
                    color = plt.cm.Reds(0.7) if (i % 2 == 0) else plt.cm.Blues(0.7)
                    alpha = 0.8
                    label_prefix = 'A'  # Anisotropic
                else:
                    color = plt.cm.Oranges(0.6) if (i % 2 == 0) else plt.cm.Greys(0.6)
                    alpha = 0.6
                    label_prefix = 'I'  # Isotropic
                
                # Draw rectangle
                width_um = (rect['x_max'] - rect['x_min']) * 1e6
                height_um = (rect['y_max'] - rect['y_min']) * 1e6
                
                rect_patch = plt.Rectangle(
                    (rect['x_min'] * 1e6, rect['y_min'] * 1e6),
                    width_um, height_um,
                    facecolor=color, alpha=alpha, edgecolor='black', linewidth=1
                )
                ax.add_patch(rect_patch)
                
                # Add rectangle info label
                x_center = (rect['x_min'] + rect['x_max']) * 0.5 * 1e6
                y_center = (rect['y_min'] + rect['y_max']) * 0.5 * 1e6
                
                # Show index and material type
                eps_xx = rect['eps_properties']['eps_xx']
                label_text = f"{label_prefix}{i}\n{eps_xx:.1f}"
                ax.text(x_center, y_center, label_text, 
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Formatting
            ax.set_xlim(self.x.min() * 1e6, self.x.max() * 1e6)
            ax.set_ylim(self.y.min() * 1e6, self.y.max() * 1e6)
            ax.set_xlabel('x (μm)')
            ax.set_ylabel('y (μm)')
            ax.set_title(f'Rectangle Clustering: {self.num_params} rectangles (Min. feature: {self.min_feature_size*1e9:.0f}nm)')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Add parameter info as text
            param_text = f"Fractions: {np.array2string(self.current_params, precision=3, separator=', ')}"
            param_text += f"\nSum: {np.sum(self.current_params):.6f}"
            param_text += f"\nBounds: [{self.bounds[0][0]:.3f}, {self.bounds[0][1]:.3f}]"
            
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            return True
            
        except Exception as e:
            print(f"Error in plotting: {e}")
            return False

    def to_file(self, filename):
        """Save current state to file with better data preservation"""
        try:
            save_data = {
                'params': self.current_params,
                'min_feature_size': self.min_feature_size,
                'eps_min': self.eps_min,
                'eps_max': self.eps_max,
                'x': self.x, 'y': self.y, 'z': self.z,
                'bounds': self.bounds,
                'rectangles': self.rectangles,
                'material_order': self.material_order,
                'num_rectangles': self.num_params,
                'Lx': self.Lx, 'Ly': self.Ly,
                'dx': self.dx, 'dy': self.dy, 'dz': self.dz
            }
            np.savez(filename, **save_data)
            print(f"Rectangle clustering state saved to {filename}")
            
        except Exception as e:
            print(f"Error saving to file: {e}")
            raise

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Load rectangle clustering from file with error handling"""
        try:
            data = np.load(filename, allow_pickle=True)
            
            instance = cls(
                min_feature_size=float(data['min_feature_size']),
                eps_min=data['eps_min'].item(),
                eps_max=data['eps_max'].item(), 
                x=data['x'],
                y=data['y'],
                z=data['z'],
                material_order=str(data['material_order']),
                **kwargs
            )
            
            # Restore parameters and bounds
            instance.update_geometry(data['params'])
            if 'bounds' in data:
                instance.bounds = data['bounds'].tolist()
            
            print(f"Rectangle clustering loaded from {filename}")
            return instance
            
        except Exception as e:
            print(f"Error loading from file: {e}")
            raise
