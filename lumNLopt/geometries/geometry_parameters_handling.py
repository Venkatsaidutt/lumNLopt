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
        
        # Initialize parameter storage for filtering
        self.current_params_raw = self.current_params.copy()      # Raw parameters (for gradients)
        self.current_params_filtered = self.current_params.copy() # Filtered parameters (for simulation)
        
        # Setup filter placeholder
        self.setup_filter_placeholder()
        
        print(f"Parameter handler initialized:")
        print(f"  Parameter count: {self.num_params}")
        print(f"  Parameter bounds: {self.bounds[0]}")
        print(f"  Sum constraint: Σ(params) = 1.0")
        print(f"  Filtering: {self.filter_enabled} (placeholder)")
        
    def setup_filter_placeholder(self):
        """Setup placeholder for minimum feature size filtering"""
        
        # Get design region info for future filter implementation
        design_region = self.geometry_cluster.design_region
        self.total_length = design_region['volume']['length']
        self.min_feature_size = design_region['constraints']['min_feature_size']
        
        # Placeholder filter configuration
        self.filter_enabled = False  # Currently disabled - passes raw parameters
        self.filter_config = {
            'min_feature_size': self.min_feature_size,
            'total_length': self.total_length,
            'method': 'placeholder'  # To be implemented later
        }
        
        print(f"  Min feature size: {self.min_feature_size*1e9:.0f} nm (filter not active)")
        print(f"  Total length: {self.total_length*1e6:.1f} μm")
        
    def apply_minimum_feature_size_filter(self, raw_params):
        """
        Apply minimum feature size filtering to fractional width parameters.
        
        Parameters:
        -----------
        raw_params : np.ndarray
            Raw fractional width parameters from optimizer
            
        Returns:
        --------
        filtered_params : np.ndarray
            Filtered parameters that satisfy minimum feature size constraints
        """
        
        if not self.filter_config['enabled']:
            return raw_params.copy()
        
        method = self.filter_config['method']
        
        if method == 'hard_constraint_projection':
            return self._hard_constraint_projection_filter(raw_params)
        elif method == 'soft_penalty':
            return self._soft_penalty_filter(raw_params)
        elif method == 'merging':
            return self._merging_filter(raw_params)
        else:
            print(f"Warning: Unknown filter method '{method}', using raw parameters")
            return raw_params.copy()
    
    def _hard_constraint_projection_filter(self, raw_params):
        """
        Hard constraint projection: Force all rectangles >= min_feature_size.
        Redistribute excess/deficit to maintain sum=1 constraint.
        """
        
        min_fraction = self.filter_config['min_fraction']
        total_length = self.filter_config['total_length']
        
        # Convert to actual widths
        raw_widths = raw_params * total_length
        
        # Identify rectangles that violate minimum feature size
        violations = raw_widths < self.filter_config['min_feature_size']
        
        if not np.any(violations):
            # No violations, return raw parameters
            return raw_params.copy()
        
        # Force minimum feature size
        filtered_widths = np.maximum(raw_widths, self.filter_config['min_feature_size'])
        
        # Calculate excess width that needs to be redistributed
        excess_width = np.sum(filtered_widths) - total_length
        
        if excess_width > 0:
            # Need to reduce some widths to maintain total length
            non_violating_indices = ~violations
            
            if np.any(non_violating_indices):
                # Redistribute excess proportionally among non-violating rectangles
                excess_per_rect = excess_width / np.sum(non_violating_indices)
                
                for i in range(len(filtered_widths)):
                    if non_violating_indices[i]:
                        # Reduce width but not below minimum
                        max_reduction = filtered_widths[i] - self.filter_config['min_feature_size']
                        reduction = min(excess_per_rect, max_reduction)
                        filtered_widths[i] -= reduction
                        excess_width -= reduction
                        
                        if excess_width <= 1e-12:  # Numerical precision
                            break
            
            # If still excess, need more aggressive redistribution
            if excess_width > 1e-12:
                print(f"Warning: Cannot satisfy min feature size constraint. Excess: {excess_width*1e9:.1f} nm")
                # Normalize to maintain total length
                filtered_widths = filtered_widths * total_length / np.sum(filtered_widths)
        
        # Convert back to fractions
        filtered_params = filtered_widths / total_length
        
        # Ensure sum = 1 (handle numerical precision)
        filtered_params = filtered_params / np.sum(filtered_params)
        
        # Report filtering results
        num_violations = np.sum(violations)
        if num_violations > 0:
            print(f"Min feature filter: Fixed {num_violations} violations")
            print(f"  Raw range: [{np.min(raw_widths)*1e9:.0f}, {np.max(raw_widths)*1e9:.0f}] nm")
            print(f"  Filtered range: [{np.min(filtered_widths)*1e9:.0f}, {np.max(filtered_widths)*1e9:.0f}] nm")
        
        return filtered_params
    
    def _soft_penalty_filter(self, raw_params):
        """
        Soft penalty approach: Apply penalty for rectangles below minimum size.
        This is more like a smooth constraint rather than hard projection.
        """
        # This would typically be handled in the FOM calculation
        # For now, just return raw parameters
        print("Soft penalty filter not implemented yet - using raw parameters")
        return raw_params.copy()
    
    def _merging_filter(self, raw_params):
        """
        Rectangle merging approach: Merge adjacent rectangles that are too small.
        This changes the parameter count dynamically.
        """
        # This is complex as it changes the optimization parameter space
        print("Merging filter not implemented yet - using hard constraint projection")
        return self._hard_constraint_projection_filter(raw_params)
    
    def get_current_params_raw(self):
        """Return current raw parameters (for gradient calculation)"""
        return self.current_params_raw.copy()
    
    def get_current_params_filtered(self):
        """Return current filtered parameters (for field calculation)"""
        return self.current_params_filtered.copy()
    
    def get_current_params(self):
        """
        Interface method for optimizer: Return current fractional widths.
        Returns filtered parameters by default for compatibility.
        """
        return self.current_params_filtered.copy()
    
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
        raw_params = self.organize_parameters(params)
        
        # Apply filtering (currently just passes raw parameters through)
        filtered_params = self.apply_filtering_placeholder(raw_params)
        
        # Update geometry cluster with filtered parameters (for simulation)
        rectangles = self.geometry_cluster.update_rectangles_from_fractions(filtered_params)
        
        # Update Lumerical simulation if provided
        if sim is not None:
            self.geometry_cluster.update_lumerical_simulation(sim.fdtd)
        
        # Store both parameter sets
        self.current_params_raw = raw_params           # For gradient calculation
        self.current_params_filtered = filtered_params # For field calculation
        self.current_params = filtered_params          # Maintain compatibility
        
        return rectangles
        
    def apply_filtering_placeholder(self, raw_params):
        """
        Placeholder for minimum feature size filtering.
        Currently just returns raw parameters unchanged.
        
        Parameters:
        -----------
        raw_params : np.ndarray
            Raw fractional width parameters from optimizer
            
        Returns:
        --------
        filtered_params : np.ndarray
            Filtered parameters (currently same as raw)
        """
        
        if self.filter_enabled:
            # TODO: Implement actual filtering here
            print("Filter enabled but not implemented - using raw parameters")
            return raw_params.copy()
        else:
            # No filtering - pass through raw parameters
            return raw_params.copy()
    
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
        print(f"Raw parameters: {self.current_params_raw}")
        print(f"Filtered parameters: {self.current_params_filtered}")
        print(f"Raw sum: {np.sum(self.current_params_raw):.8f}")
        print(f"Filtered sum: {np.sum(self.current_params_filtered):.8f}")
        print(f"Bounds: [{self.bounds[0][0]:.6f}, {self.bounds[0][1]:.6f}]")
        
        # Check constraint satisfaction
        raw_sum_violation = abs(np.sum(self.current_params_raw) - 1.0)
        filtered_sum_violation = abs(np.sum(self.current_params_filtered) - 1.0)
        print(f"Raw sum constraint violation: {raw_sum_violation:.2e}")
        print(f"Filtered sum constraint violation: {filtered_sum_violation:.2e}")
        
        # Filter status
        print(f"Filtering enabled: {self.filter_enabled}")
        if self.filter_enabled:
            print(f"Filter method: {self.filter_config['method']}")
        
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
