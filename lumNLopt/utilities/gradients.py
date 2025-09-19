import lumapi
import numpy as np
import scipy as sp

class GradientFields(object):
    """
    ENHANCED GradientFields with tensor-aware gradient calculations
    for rectangle clustering with anisotropic materials.
    """

    def __init__(self, forward_fields, adjoint_fields):
        self.forward_fields = forward_fields
        self.adjoint_fields = adjoint_fields

    def boundary_perturbation_integrand(self):
        """Standard boundary perturbation method (for polygon optimization)"""
        
        def project(a, b):
            b_norm = b / np.linalg.norm(b)
            return np.dot(a, b_norm) * b_norm

        def gradient_field(x, y, z, wl, normal, eps_in, eps_out):
            E_forward = self.forward_fields.getfield(x, y, z, wl)
            D_forward = self.forward_fields.getDfield(x, y, z, wl)
            E_adjoint = self.adjoint_fields.getfield(x, y, z, wl)
            D_adjoint = self.adjoint_fields.getDfield(x, y, z, wl)
            E_parallel_forward = E_forward - project(E_forward, normal)
            E_parallel_adjoint = E_adjoint - project(E_adjoint, normal)
            D_perp_forward = project(D_forward, normal)
            D_perp_adjoint = project(D_adjoint, normal)
            result = ( 2.0 * sp.constants.epsilon_0 * (eps_in - eps_out) * np.sum(E_parallel_forward * E_parallel_adjoint) 
                      + (1.0/eps_out - 1.0/eps_in) / sp.constants.epsilon_0 * np.sum(D_perp_forward * D_perp_adjoint) )
            return np.real(result)
        
        return gradient_field

    def tensor_volume_integrand(self, eps_tensor_derivative):
        """
        NEW: Tensor-aware volume integrand for rectangle clustering
        
        For anisotropic materials, the gradient is:
        dFOM/dp = Re[∫ E_forward^H * (dε_tensor/dp) * E_adjoint dV]
        
        Parameters:
        -----------
        eps_tensor_derivative : np.ndarray, shape (3, 3)
            Derivative of permittivity tensor w.r.t. parameter
        """
        
        def tensor_gradient_field(x, y, z, wl):
            """Calculate tensor-aware gradient at a point"""
            
            try:
                # Get field vectors at point
                E_forward = self.forward_fields.getfield(x, y, z, wl)  # [Ex, Ey, Ez]
                E_adjoint = self.adjoint_fields.getfield(x, y, z, wl)  # [Ex, Ey, Ez]
                
                # Tensor contraction: E_forward^H * deps_tensor * E_adjoint
                # This is the correct adjoint sensitivity for anisotropic materials
                sensitivity = 0.0
                for i in range(3):
                    for j in range(3):
                        sensitivity += (np.conj(E_forward[i]) * 
                                      eps_tensor_derivative[i, j] * 
                                      E_adjoint[j])
                
                # Return real part (gradient is real)
                return np.real(sensitivity) * sp.constants.epsilon_0
                
            except Exception as e:
                print(f"Error in tensor gradient calculation: {e}")
                return 0.0
        
        return tensor_gradient_field

    @staticmethod
    def spatial_gradient_integral_on_cad_tensor(sim, forward_fields, adjoint_fields, 
                                               eps_tensor_derivatives, wl_scaling_factor):
        """
        NEW: Enhanced spatial gradient calculation for anisotropic rectangle clustering.
        
        This bypasses the standard lumNLopt isotropic assumption and calculates
        proper tensor-aware gradients using Lumerical's field data.
        
        Parameters:
        -----------
        eps_tensor_derivatives : list of np.ndarray
            List of 3x3 permittivity tensor derivatives for each parameter
        """
        
        lumapi.putMatrix(sim.fdtd.handle, "wl_scaling_factor", wl_scaling_factor)
        
        # Get field data from Lumerical
        sim.fdtd.eval(f"""
        E_forward = {forward_fields}.E.E;
        E_adjoint = {adjoint_fields}.E.E;
        x_coords = {forward_fields}.E.x;
        y_coords = {forward_fields}.E.y; 
        z_coords = {forward_fields}.E.z;
        wavelengths = {forward_fields}.E.lambda;
        
        num_params = {len(eps_tensor_derivatives)};
        num_wl = length(wavelengths);
        gradient_results = zeros(num_wl, num_params);
        """)
        
        # Upload tensor derivatives to Lumerical
        for param_idx, deps_tensor in enumerate(eps_tensor_derivatives):
            # Upload 3x3 tensor derivative
            lumapi.putMatrix(sim.fdtd.handle, f"deps_tensor_{param_idx}", deps_tensor)
        
        # Calculate tensor gradients in Lumerical script
        sim.fdtd.eval(f"""
        for param_idx = 1:num_params {{
            deps_tensor = getnamed("deps_tensor_" + num2str(param_idx-1));
            
            for wl_idx = 1:num_wl {{
                % Extract fields at this wavelength
                E_f = E_forward(:,:,:,wl_idx,:);
                E_a = E_adjoint(:,:,:,wl_idx,:);
                
                % Initialize gradient accumulator
                grad_integrand = zeros(size(E_f,1), size(E_f,2), size(E_f,3));
                
                % Tensor contraction: E_forward^H * deps_tensor * E_adjoint
                for i = 1:3 {{
                    for j = 1:3 {{
                        grad_integrand = grad_integrand + ...
                            real(conj(E_f(:,:,:,1,i)) .* deps_tensor(i,j) .* E_a(:,:,:,1,j));
                    }}
                }}
                
                % Spatial integration
                gradient_results(wl_idx, param_idx) = eps0 * wl_scaling_factor(wl_idx) * ...
                    integrate2(grad_integrand, [1,2,3], x_coords, y_coords, z_coords);
            }}
        }}
        """)
        
        # Get results back from Lumerical
        gradient_results = lumapi.getVar(sim.fdtd.handle, 'gradient_results')
        
        # Clean up Lumerical workspace
        cleanup_vars = ['E_forward', 'E_adjoint', 'x_coords', 'y_coords', 'z_coords',
                       'wavelengths', 'gradient_results', 'num_params', 'num_wl']
        cleanup_vars.extend([f'deps_tensor_{i}' for i in range(len(eps_tensor_derivatives))])
        
        cleanup_script = "clear(" + ", ".join(cleanup_vars) + ");"
        sim.fdtd.eval(cleanup_script)
        
        return gradient_results

    @staticmethod  
    def calculate_rectangle_gradients(geometry, forward_fields, adjoint_fields, wavelengths):
        """
        NEW: Calculate gradients for rectangle clustering parameters.
        
        This is the main interface that rectangle clustering geometry will use
        instead of the standard lumNLopt gradient calculation.
        """
        
        gradients = np.zeros(geometry.num_params)
        
        try:
            for param_idx in range(geometry.num_params):
                # Get rectangle affected by this parameter
                rect = geometry.rectangles[param_idx]
                
                # Material properties 
                rect_material = rect['eps_properties']
                bg_material = geometry._get_material_properties(False)
                
                # Tensor derivative: how permittivity tensor changes w.r.t. parameter
                deps_tensor = np.zeros((3, 3))
                deps_tensor[0, 0] = rect_material['eps_xx'] - bg_material['eps_xx']
                deps_tensor[1, 1] = rect_material['eps_yy'] - bg_material['eps_yy']  
                deps_tensor[2, 2] = rect_material['eps_zz'] - bg_material['eps_zz']
                
                # Create gradient field integrator
                grad_fields = GradientFields(forward_fields, adjoint_fields)
                tensor_integrand = grad_fields.tensor_volume_integrand(deps_tensor)
                
                # Spatial integration over rectangle region
                grad_value = geometry._integrate_over_rectangle(
                    rect, tensor_integrand, wavelengths)
                
                gradients[param_idx] = grad_value
                
        except Exception as e:
            print(f"Error in rectangle gradient calculation: {e}")
            return np.zeros(geometry.num_params)
        
        return gradients
