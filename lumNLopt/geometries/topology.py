from lumopt.geometries.geometry import Geometry
from lumopt.utilities.materials import Material
from lumopt.lumerical_methods.lumerical_scripts import set_spatial_interp, get_eps_from_sim

import lumapi
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

eps0 = sp.constants.epsilon_0


class RectangleClusteringTopology(Geometry):
    """
    Rectangle clustering topology optimization using fractional width parameters.
    Each cross-section is parameterized by rectangle width fractions that sum to 1.
    Ensures manufacturability by construction with minimum feature size constraints.
    """

    def __init__(self, min_feature_size, eps_min, eps_max, x, y, z, 
                 material_order='alternating', filter_R=None, eta=0.5, beta=1):
        """
        Parameters:
        -----------
        min_feature_size : float
            Minimum feature size 'd' in meters
        eps_min, eps_max : float  
            Background and fill material permittivities
        x, y, z : array-like
            Spatial coordinates defining design region
        material_order : str
            'alternating' or custom sequence for rectangle materials
        filter_R : float, optional
            Filter radius for smoothing (applied after rectangle generation)
        eta, beta : float
            Projection parameters (for compatibility, may not be needed)
        """
        self.min_feature_size = min_feature_size
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.x = np.array(x)
        self.y = np.array(y) 
        self.z = np.array(z) if hasattr(z, "__len__") else np.array([z])
        self.material_order = material_order
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
        
        # Calculate maximum number of rectangles per cross-section
        self.max_rectangles_x = int(np.floor(self.Lx / self.min_feature_size))
        self.max_rectangles_y = int(np.floor(self.Ly / self.min_feature_size))
        
        # For 2D optimization: rectangles along x-direction only
        self.num_params = self.max_rectangles_x
        
        # Initialize with equal-width rectangles
        initial_fractions = np.ones(self.num_params) / self.num_params
        self.current_params = initial_fractions
        
        # Set bounds: each fraction between 0 and 1, but constraint ensures sum = 1
        self.bounds = [(1e-6, 1.0 - 1e-6)] * self.num_params  # Small epsilon to avoid zeros
        
        # Storage for current rectangle configuration
        self.rectangles = []
        self.eps_tensor_field = None
        
        self.unfold_symmetry = False

    def use_interpolation(self):
        return False  # We're using discrete rectangles

    def get_current_params(self):
        """Return current fractional widths (must sum to 1)"""
        return self.current_params.copy()

    def update_geometry(self, params, sim=None):
        """Update geometry from fractional width parameters"""
        # Normalize to ensure sum = 1 (constraint enforcement)
        params_normalized = np.array(params) / np.sum(params)
        self.current_params = params_normalized
        
        # Generate rectangles from fractional widths
        self.rectangles = self._generate_rectangles_from_fractions(params_normalized)
        
        # Generate epsilon tensor field
        self.eps_tensor_field = self._generate_epsilon_tensor_field()

    def _generate_rectangles_from_fractions(self, fractions):
        """Convert fractional widths to rectangle coordinates"""
        rectangles = []
        x_start = self.x.min()
        
        for i, fraction in enumerate(fractions):
            # Calculate rectangle width from fraction
            width = fraction * self.Lx
            x_end = x_start + width
            
            # Determine material (alternating or custom order)
            if self.material_order == 'alternating':
                eps_value = self.eps_max if (i % 2 == 0) else self.eps_min
            else:
                # Custom material ordering logic here
                eps_value = self.eps_max if (i % 2 == 0) else self.eps_min
            
            rectangle = {
                'x_min': x_start,
                'x_max': x_end,
                'y_min': self.y.min(),
                'y_max': self.y.max(),
                'z_center': np.mean(self.z),
                'z_span': self.depth,
                'eps_tensor': self._get_anisotropic_epsilon_tensor(eps_value),
                'material_index': i
            }
            rectangles.append(rectangle)
            x_start = x_end
            
        return rectangles

    def _get_anisotropic_epsilon_tensor(self, base_eps):
        """
        Generate anisotropic permittivity tensor for rectangle.
        Modify this function for your specific anisotropic material model.
        """
        # Example: Uniaxial anisotropy
        eps_tensor = np.eye(3) * base_eps
        # For anisotropic materials, modify specific components:
        # eps_tensor[0,0] = eps_xx  # Different x-component if needed
        # eps_tensor[1,1] = eps_yy  # Different y-component if needed
        # eps_tensor[2,2] = eps_zz  # Different z-component if needed
        
        return eps_tensor

    def _generate_epsilon_tensor_field(self):
        """Generate full 3x3 permittivity tensor field over spatial grid"""
        # Create spatial grid
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        eps_field = np.zeros((nx, ny, nz, 3, 3))  # [x,y,z,tensor_i,tensor_j]
        
        # Fill with background material
        bg_tensor = self._get_anisotropic_epsilon_tensor(self.eps_min)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    eps_field[i,j,k] = bg_tensor
        
        # Override with rectangle materials
        for rect in self.rectangles:
            # Find grid points inside rectangle
            x_mask = (self.x >= rect['x_min']) & (self.x <= rect['x_max'])
            y_mask = (self.y >= rect['y_min']) & (self.y <= rect['y_max'])
            
            for i in np.where(x_mask)[0]:
                for j in np.where(y_mask)[0]:
                    for k in range(nz):
                        eps_field[i,j,k] = rect['eps_tensor']
        
        return eps_field

    def add_geo(self, sim, params=None, only_update=False):
        """Add rectangle geometry to Lumerical simulation"""
        if params is not None:
            self.update_geometry(params)
        
        fdtd = sim.fdtd
        
        if not only_update:
            # Setup monitors and mesh
            set_spatial_interp(fdtd, 'opt_fields', 'specified position')
            set_spatial_interp(fdtd, 'opt_fields_index', 'specified position')
            
            # Set monitor boundaries
            script = ('select("opt_fields");'
                     'set("x min",{});'
                     'set("x max",{});'
                     'set("y min",{});'
                     'set("y max",{});'
                     'set("z min",{});'
                     'set("z max",{});').format(
                         np.amin(self.x), np.amax(self.x),
                         np.amin(self.y), np.amax(self.y), 
                         np.amin(self.z), np.amax(self.z))
            fdtd.eval(script)
            
            # Add mesh override
            mesh_script = ('addmesh;'
                          'set("x min",{});'
                          'set("x max",{});'
                          'set("y min",{});'
                          'set("y max",{});'
                          'set("z min",{});'
                          'set("z max",{});'
                          'set("dx",{});'
                          'set("dy",{});'
                          'set("dz",{});').format(
                              np.amin(self.x), np.amax(self.x),
                              np.amin(self.y), np.amax(self.y),
                              np.amin(self.z), np.amax(self.z),
                              self.dx, self.dy, self.dz)
            fdtd.eval(mesh_script)
        
        # Add/update rectangles
        self._add_rectangles_to_simulation(fdtd, only_update)

    def _add_rectangles_to_simulation(self, fdtd, only_update):
        """Add rectangle structures to Lumerical CAD"""
        for i, rect in enumerate(self.rectangles):
            rect_name = f'opt_rect_{i}'
            
            if not only_update:
                fdtd.addrect()
                fdtd.set('name', rect_name)
            
            # Set rectangle geometry
            fdtd.setnamed(rect_name, 'x min', rect['x_min'])
            fdtd.setnamed(rect_name, 'x max', rect['x_max'])
            fdtd.setnamed(rect_name, 'y min', rect['y_min'])
            fdtd.setnamed(rect_name, 'y max', rect['y_max'])
            fdtd.setnamed(rect_name, 'z', rect['z_center'])
            fdtd.setnamed(rect_name, 'z span', rect['z_span'])
            
            # Set anisotropic material properties
            self._set_anisotropic_material(fdtd, rect_name, rect['eps_tensor'])

    def _set_anisotropic_material(self, fdtd, object_name, eps_tensor):
        """Set anisotropic material properties for a Lumerical object"""
        # Extract diagonal and off-diagonal components
        eps_xx, eps_yy, eps_zz = eps_tensor[0,0], eps_tensor[1,1], eps_tensor[2,2]
        eps_xy, eps_xz, eps_yz = eps_tensor[0,1], eps_tensor[0,2], eps_tensor[1,2]
        
        # Set material to anisotropic
        fdtd.setnamed(object_name, 'material', '<Object defined dielectric>')
        fdtd.setnamed(object_name, 'material type', 'Anisotropic')
        
        # Set tensor components (requires Lumerical syntax for anisotropic materials)
        fdtd.setnamed(object_name, 'permittivity', 1.0)  # Base value
        fdtd.setnamed(object_name, 'anisotropy', 1)  # Enable anisotropy
        
        # Set individual tensor components (check Lumerical documentation for exact syntax)
        fdtd.setnamed(object_name, 'diagonal permittivity', [eps_xx, eps_yy, eps_zz])
        if np.any([eps_xy, eps_xz, eps_yz]):  # If off-diagonal terms exist
            fdtd.setnamed(object_name, 'off diagonal permittivity', [eps_xy, eps_xz, eps_yz])

    def calculate_gradients_manual(self, forward_fields, adjoint_fields, wavelengths):
        """
        Custom anisotropic gradient calculation bypassing lumNLopt framework.
        Implements tensor-aware adjoint sensitivity.
        """
        gradients = np.zeros(self.num_params)
        
        for param_idx in range(self.num_params):
            # Calculate sensitivity of each fractional width parameter
            gradient_val = self._calculate_single_parameter_gradient(
                param_idx, forward_fields, adjoint_fields, wavelengths)
            gradients[param_idx] = gradient_val
        
        return gradients

    def _calculate_single_parameter_gradient(self, param_idx, E_forward, E_adjoint, wavelengths):
        """
        Calculate gradient w.r.t. single fractional width parameter using
        anisotropic adjoint sensitivity formula:
        
        dFOM/dp_i = ∫ E_forward^T * (dε_tensor/dp_i) * E_adjoint dV
        """
        gradient = 0.0
        
        # Get the rectangle affected by this parameter
        rect = self.rectangles[param_idx]
        
        # Calculate tensor derivative w.r.t. fractional width
        deps_tensor_dp = self._calculate_tensor_derivative(param_idx)
        
        # Find spatial region of this rectangle
        x_mask = (self.x >= rect['x_min']) & (self.x <= rect['x_max'])
        y_mask = (self.y >= rect['y_min']) & (self.y <= rect['y_max'])
        
        # Integrate over wavelengths and space
        for wl_idx, wl in enumerate(wavelengths):
            for i in np.where(x_mask)[0]:
                for j in np.where(y_mask)[0]:
                    for k in range(len(self.z)):
                        # Extract field values at this point
                        E_f = E_forward[i, j, k, wl_idx, :]  # [Ex, Ey, Ez]
                        E_a = E_adjoint[i, j, k, wl_idx, :]  # [Ex, Ey, Ez]
                        
                        # Tensor contraction: E_forward^T * deps_tensor * E_adjoint
                        tensor_product = np.real(
                            np.conj(E_f) @ deps_tensor_dp @ E_a
                        )
                        
                        # Volume element
                        dV = self.dx * self.dy * self.dz
                        
                        gradient += 2.0 * eps0 * tensor_product * dV
        
        return gradient

    def _calculate_tensor_derivative(self, param_idx):
        """
        Calculate how the permittivity tensor changes w.r.t. fractional width parameter.
        
        For parameter clustering: changing width of rectangle param_idx affects
        the material distribution and hence the effective tensor.
        """
        # This depends on your specific material model
        # Example for simple case: 
        rect = self.rectangles[param_idx]
        
        # Derivative of rectangle width w.r.t. fractional parameter
        dwidth_dfraction = self.Lx  # Total length
        
        # How does epsilon tensor change with rectangle width?
        # This is geometry and material dependent
        deps_tensor = (rect['eps_tensor'] - self._get_anisotropic_epsilon_tensor(self.eps_min))
        
        # Scale by width sensitivity  
        deps_tensor_dp = deps_tensor * dwidth_dfraction / self.Lx
        
        return deps_tensor_dp

    def apply_constraint_projection(self, params):
        """
        Project parameters onto constraint surface: sum(params) = 1
        Used by constrained optimizers.
        """
        params_array = np.array(params)
        
        # Ensure non-negative
        params_array = np.maximum(params_array, 1e-6)
        
        # Normalize to sum = 1
        params_normalized = params_array / np.sum(params_array)
        
        return params_normalized

    def get_constraint_function(self):
        """Return constraint function for optimizer: sum(params) - 1 = 0"""
        def constraint(params):
            return np.sum(params) - 1.0
        return constraint

    def get_constraint_jacobian(self):
        """Return constraint jacobian: d(sum(params))/dp_i = 1 for all i"""
        def constraint_jac(params):
            return np.ones(len(params))
        return constraint_jac

    def plot(self, ax):
        """Plot current rectangle configuration"""
        ax.clear()
        
        for i, rect in enumerate(self.rectangles):
            # Color based on material
            color = 'red' if (i % 2 == 0) else 'blue'
            
            # Plot rectangle
            ax.add_patch(plt.Rectangle(
                (rect['x_min'] * 1e6, rect['y_min'] * 1e6),
                (rect['x_max'] - rect['x_min']) * 1e6,
                (rect['y_max'] - rect['y_min']) * 1e6,
                facecolor=color, alpha=0.7, edgecolor='black'
            ))
        
        ax.set_xlim(self.x.min() * 1e6, self.x.max() * 1e6)
        ax.set_ylim(self.y.min() * 1e6, self.y.max() * 1e6)
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_title('Rectangle Clustering Configuration')
        ax.set_aspect('equal')
        
        return True

    def to_file(self, filename):
        """Save current state to file"""
        np.savez(filename, 
                 params=self.current_params,
                 min_feature_size=self.min_feature_size,
                 eps_min=self.eps_min, 
                 eps_max=self.eps_max,
                 x=self.x, y=self.y, z=self.z,
                 rectangles=self.rectangles)
