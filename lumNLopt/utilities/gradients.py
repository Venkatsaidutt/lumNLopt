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

    def mode_overlap_integrand(self, mode1_fields, mode2_fields):
        """
        Create integrand for mode overlap gradient calculation.
        
        For adiabatic coupling optimization, we need gradients of mode overlaps
        between adjacent slices w.r.t. geometry parameters.
        
        Parameters:
        -----------
        mode1_fields : field object
            Mode fields from first monitor
        mode2_fields : field object
            Mode fields from second monitor
            
        Returns:
        --------
        overlap_gradient_field : function
            Function that calculates overlap gradient at a point
        """
        
        def overlap_gradient_field(x, y, z, wl):
            """Calculate mode overlap gradient contribution at a point"""
            
            try:
                # Get mode fields at this spatial point
                E1 = mode1_fields.getfield(x, y, z, wl)  # [Ex, Ey, Ez]
                E2 = mode2_fields.getfield(x, y, z, wl)  # [Ex, Ey, Ez]
                
                # Mode overlap integrand: ∫ E1*(r) · E2(r) dr
                # The gradient contribution is the local field overlap
                overlap_contrib = np.sum(np.conj(E1) * E2)
                
                # Return real part (overlap gradient is real)
                return np.real(overlap_contrib) * sp.constants.epsilon_0
                
            except Exception as e:
                print(f"Error in mode overlap gradient calculation: {e}")
                return 0.0
        
        return overlap_gradient_field

    def adiabatic_evolution_integrand(self, slice_fields_list, target_overlaps):
        """
        Create integrand for adiabatic evolution quality gradient.
        
        This handles gradients of the mode evolution quality metric w.r.t.
        geometry parameters for the entire adiabatic coupler.
        
        Parameters:
        -----------
        slice_fields_list : list
            List of field objects for each slice
        target_overlaps : np.ndarray
            Target overlap values from arithmetic progression
            
        Returns:
        --------
        evolution_gradient_field : function
            Function that calculates evolution gradient at a point
        """
        
        def evolution_gradient_field(x, y, z, wl, slice_index):
            """Calculate adiabatic evolution gradient contribution"""
            
            try:
                gradient_contrib = 0.0
                
                # Consider overlaps with adjacent slices
                if slice_index > 0:
                    # Overlap with previous slice
                    E_current = slice_fields_list[slice_index].getfield(x, y, z, wl)
                    E_prev = slice_fields_list[slice_index-1].getfield(x, y, z, wl)
                    
                    overlap_prev = np.sum(np.conj(E_prev) * E_current)
                    target_prev = target_overlaps[slice_index-1]
                    
                    # Gradient weight based on deviation from target
                    weight_prev = 2.0 * (np.real(overlap_prev) - target_prev)
                    gradient_contrib += weight_prev * np.real(overlap_prev)
                
                if slice_index < len(slice_fields_list) - 1:
                    # Overlap with next slice
                    E_current = slice_fields_list[slice_index].getfield(x, y, z, wl)
                    E_next = slice_fields_list[slice_index+1].getfield(x, y, z, wl)
                    
                    overlap_next = np.sum(np.conj(E_current) * E_next)
                    target_next = target_overlaps[slice_index]
                    
                    # Gradient weight based on deviation from target
                    weight_next = 2.0 * (np.real(overlap_next) - target_next)
                    gradient_contrib += weight_next * np.real(overlap_next)
                
                return gradient_contrib * sp.constants.epsilon_0
                
            except Exception as e:
                print(f"Error in adiabatic evolution gradient: {e}")
                return 0.0
        
        return evolution_gradient_field







        
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

@staticmethod
    def calculate_adiabatic_coupling_gradients(geometry, forward_fields, adjoint_fields, 
                                               slice_fields_list, target_overlaps, wavelengths):
        """
        Calculate gradients for adiabatic coupling optimization.
        
        This combines:
        1. Standard power transfer gradients
        2. Mode overlap gradients between adjacent slices
        3. Adiabatic evolution quality gradients
        
        Parameters:
        -----------
        geometry : RectangleClusteringTopology
            Geometry object with parameter information
        forward_fields : field object
            Forward simulation fields
        adjoint_fields : field object
            Adjoint simulation fields
        slice_fields_list : list
            List of field objects for each slice
        target_overlaps : np.ndarray
            Target overlap progression from arithmetic sequence
        wavelengths : np.ndarray
            Wavelength array
            
        Returns:
        --------
        gradients : np.ndarray
            Combined gradients for all geometry parameters
        """
        
        gradients = np.zeros(geometry.num_params)
        
        try:
            # 1. Standard power transfer gradients (main transmission)
            power_gradients = GradientFields.calculate_rectangle_gradients(
                geometry, forward_fields, adjoint_fields, wavelengths)
            
            # 2. Mode overlap gradients (adiabatic evolution)
            overlap_gradients = np.zeros(geometry.num_params)
            
            if len(slice_fields_list) > 1:
                grad_fields = GradientFields(forward_fields, adjoint_fields)
                
                for i in range(len(slice_fields_list) - 1):
                    # Mode overlap between adjacent slices
                    overlap_integrand = grad_fields.mode_overlap_integrand(
                        slice_fields_list[i], slice_fields_list[i+1])
                    
                    # Target overlap for this slice pair
                    target_overlap = target_overlaps[i] if i < len(target_overlaps) else 0.8
                    
                    # Calculate overlap gradient for each parameter
                    for param_idx in range(geometry.num_params):
                        rect = geometry.rectangles[param_idx]
                        
                        # Spatial integration over rectangle
                        overlap_grad = geometry._integrate_over_rectangle(
                            rect, overlap_integrand, wavelengths)
                        
                        # Weight by deviation from target overlap
                        weight = 2.0 * (target_overlap - 0.8)  # Simplified weighting
                        overlap_gradients[param_idx] += weight * overlap_grad
            
            # 3. Combine gradients with appropriate weights
            transmission_weight = 2.0  # Primary objective
            evolution_weight = 1.5     # Secondary objective
            
            gradients = (transmission_weight * power_gradients + 
                        evolution_weight * overlap_gradients)
            
            # Apply constraint projection
            if hasattr(geometry, '_apply_constraint_projection'):
                gradients = geometry._apply_constraint_projection(gradients)
            
            return gradients
            
        except Exception as e:
            print(f"Error in adiabatic coupling gradient calculation: {e}")
            return np.zeros(geometry.num_params)
