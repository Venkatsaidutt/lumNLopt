# geometry_parameters_handling.py - Parameter Marshaling Interface
# Simple parameter organization between optimizer and Geometry_clustered
# NO material handling, NO Lumerical interaction - pure parameter management

import numpy as np
from lumNLopt.geometries.Geometry_clustered import RectangleClusteredGeometry

class GeometryParameterHandler:
    """
    Simple parameter marshaling interface between optimizer and rectangle clustering.
    Handles parameter organization, validation, and format conversion.
    """
    
    def __init__(self):
        """Initialize with geometry cluster manager"""
        self.geometry_cluster = RectangleClusteredGeometry()
        self.num_params = self.geometry_cluster.get_rectangle_count()
        self.bounds = self.geometry_cluster.get_bounds()
        self.current_params = self.geometry_cluster.get_current_fractions()
        
        print(f"Parameter handler initialized:")
        print(f"  Parameter count: {self.num_params}")
        print(f"  Parameter bounds: {self.bounds[0]}")
        print(f"  Sum constraint: Σ(params) = 1.0")
    
    def get_current_params(self):
        """
        Interface method for optimizer: Return current fractional widths.
        Required by lumNLopt optimization framework.
        """
        return self.current_params.copy()
    
    def update_geometry(self, params, sim=None):
        """
        Interface method for optimizer: Update geometry from new parameters.
        Required by lumNLopt optimization framework.
        
        Parameters:
        -----------
        params : array_like
            Fractional width parameters from optimizer
        sim : object, optional
            Lumerical simulation object (passed to geometry_cluster)
        """
        # Organize parameters in correct format
        organized_params = self.organize_parameters(params)
        
        # Update geometry cluster
        rectangles = self.geometry_cluster.update_rectangles_from_fractions(organized_params)
        
        # Update Lumerical simulation if provided
        if sim is not None:
            self.geometry_cluster.update_lumerical_simulation(sim.fdtd)
        
        # Store current parameters
        self.current_params = organized_params
        
        return rectangles
    
    def organize_parameters(self, params):
        """
        Organize parameters in correct format for geometry clustering.
        Handles validation and normalization.
        
        Parameters:
        -----------
        params : array_like
            Raw parameters from optimizer
            
        Returns:
        --------
        organized_params : np.ndarray
            Properly formatted fractional widths
        """
        # Convert to numpy array
        params_array = np.array(params, dtype=float)
        
        # Validate parameter count
        if len(params_array) != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameters, got {len(params_array)}")
        
        # Validate parameter bounds
        for i, (param, (min_val, max_val)) in enumerate(zip(params_array, self.bounds)):
            if param < min_val:
                print(f"Warning: Parameter {i} = {param:.6f} below minimum {min_val:.6f}")
                params_array[i] = min_val
            elif param > max_val:
                print(f"Warning: Parameter {i} = {param:.6f} above maximum {max_val:.6f}")
                params_array[i] = max_val
        
        # Enforce sum = 1 constraint
        param_sum = np.sum(params_array)
        if abs(param_sum - 1.0) > 1e-8:
            print(f"Normalizing parameters: sum = {param_sum:.8f} → 1.0")
            params_array = params_array / param_sum
        
        return params_array
    
    def reorganize_gradients(self, gradient_array):
        """
        Reorganize gradients from field/FOM calculation for optimizer.
        
        Parameters:
        -----------
        gradient_array : array_like
            Raw gradients from gradient calculation
            
        Returns:
        --------
        organized_gradients : np.ndarray
            Properly formatted gradients for optimizer
        """
        # Convert to numpy array
        gradients = np.array(gradient_array, dtype=float)
        
        # Validate gradient count
        if len(gradients) != self.num_params:
            raise ValueError(f"Expected {self.num_params} gradients, got {len(gradients)}")
        
        # Apply sum=1 constraint to gradients (project onto constraint manifold)
        # For constraint Σ(x_i) = 1, projected gradient is: g_proj = g - (1/n)Σ(g_i)
        gradient_mean = np.mean(gradients)
        projected_gradients = gradients - gradient_mean
        
        print(f"Gradient projection applied: mean = {gradient_mean:.8f}")
        
        return projected_gradients
    
    def calculate_gradients(self, gradient_fields):
        """
        Interface method for optimization: Calculate gradients w.r.t. parameters.
        Required by lumNLopt optimization framework.
        
        This method receives gradient fields and should return gradients w.r.t.
        the fractional width parameters.
        
        Parameters:
        -----------
        gradient_fields : object
            Gradient field object from lumNLopt
            
        Returns:
        --------
        param_gradients : np.ndarray
            Gradients w.r.t. fractional width parameters
        """
        # This is where the actual gradient calculation would happen
        # For now, return placeholder gradients
        # In full implementation, this would:
        # 1. Extract field gradients from gradient_fields
        # 2. Calculate gradients w.r.t. rectangle boundaries
        # 3. Transform to gradients w.r.t. fractional widths
        
        print("Calculating gradients w.r.t. fractional width parameters...")
        
        # Placeholder: finite difference approximation
        gradients = self._calculate_finite_difference_gradients(gradient_fields)
        
        # Reorganize gradients for optimizer
        organized_gradients = self.reorganize_gradients(gradients)
        
        return organized_gradients
    
    def _calculate_finite_difference_gradients(self, gradient_fields):
        """
        Placeholder gradient calculation using finite differences.
        In full implementation, this would use adjoint field gradients.
        """
        # Placeholder implementation
        # Real implementation would interface with lumNLopt gradient calculation
        
        gradients = np.zeros(self.num_params)
        dx = 1e-6  # Small perturbation
        
        # This is just a placeholder - real gradients would come from
        # the adjoint field calculation in the optimization framework
        for i in range(self.num_params):
            # Placeholder gradient calculation
            gradients[i] = np.random.normal(0, 0.1)  # Random gradient for testing
        
        print(f"Calculated {len(gradients)} parameter gradients")
        return gradients
    
    def get_bounds(self):
        """Return parameter bounds for optimizer"""
        return self.bounds
    
    def get_constraint_function(self):
        """
        Return constraint function for sum=1 constraint.
        For use with constrained optimizers (e.g., NLopt equality constraints).
        """
        def sum_constraint(params, grad):
            """Constraint function: sum(params) - 1 = 0"""
            if grad.size > 0:
                grad[:] = 1.0  # Gradient of sum w.r.t. each parameter is 1
            return np.sum(params) - 1.0
        
        return sum_constraint
    
    def add_geo(self, sim, params=None, only_update=False):
        """
        Interface method for lumNLopt: Add/update geometry in simulation.
        Required by lumNLopt optimization framework.
        
        Parameters:
        -----------
        sim : object
            Lumerical simulation object
        params : array_like, optional
            Parameters to use (if None, use current)
        only_update : bool
            Whether to only update existing geometry
        """
        if params is not None:
            # Update geometry with new parameters
            self.update_geometry(params, sim)
        else:
            # Use current geometry
            self.geometry_cluster.update_lumerical_simulation(sim.fdtd)
        
        print(f"Geometry {'updated' if only_update else 'added'} to simulation")
    
    def plot(self, ax=None):
        """Plot current rectangle configuration"""
        return self.geometry_cluster.plot_rectangle_configuration(ax)
    
    def use_interpolation(self):
        """Rectangle clustering doesn't use interpolation"""
        return False
    
    def print_parameter_summary(self):
        """Print summary of current parameters and configuration"""
        print("\n" + "="*50)
        print("PARAMETER HANDLER SUMMARY")
        print("="*50)
        print(f"Parameter count: {self.num_params}")
        print(f"Current parameters: {self.current_params}")
        print(f"Parameter sum: {np.sum(self.current_params):.8f}")
        print(f"Bounds: [{self.bounds[0][0]:.6f}, {self.bounds[0][1]:.6f}]")
        
        # Check constraint satisfaction
        sum_violation = abs(np.sum(self.current_params) - 1.0)
        print(f"Sum constraint violation: {sum_violation:.2e}")
        
        # Check bounds violations
        bounds_violations = 0
        for i, (param, (min_val, max_val)) in enumerate(zip(self.current_params, self.bounds)):
            if param < min_val or param > max_val:
                bounds_violations += 1
        
        print(f"Bounds violations: {bounds_violations}")
        print("="*50)
        
        # Print geometry summary
        self.geometry_cluster.print_summary()

# Convenience functions for integration with lumNLopt optimization framework
def create_rectangle_clustering_geometry():
    """Factory function to create rectangle clustering geometry for optimization"""
    return GeometryParameterHandler()

def get_parameter_bounds():
    """Get parameter bounds for rectangle clustering"""
    handler = GeometryParameterHandler()
    return handler.get_bounds()

def get_parameter_count():
    """Get number of parameters for rectangle clustering"""
    handler = GeometryParameterHandler()
    return handler.num_params
