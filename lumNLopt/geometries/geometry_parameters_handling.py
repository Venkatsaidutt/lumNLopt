# geometry_parameters_handling.py - Parameter Marshaling with Morphological Filtering
# Handles parameter organization and 2D morphological filtering for minimum feature size constraints

import numpy as np
from lumNLopt.geometries.Geometry_clustered import RectangleClusteredGeometry

class GeometryParameterHandler:
    """
    Parameter marshaling interface between optimizer and rectangle clustering.
    
    Implements morphology-based 2D filtering to enforce minimum feature size constraints
    using binary opening operations on pixelized design representations.
    
    References:
    -----------
    [1] O. Sigmund, "Morphology-based black and white filters for topology optimization",
        Structural and Multidisciplinary Optimization, 33(4-5), pp. 401-424 (2007).
    [2] M. Schevenels & O. Sigmund, "On the implementation and effectiveness of 
        morphological close-open and open-close filters for topology optimization",
        Structural and Multidisciplinary Optimization, 54(1), pp. 15-21 (2016).
    """
    
    def __init__(self):
        """Initialize parameter handler with morphological filtering"""
        
        # Initialize geometry cluster manager
        self.geometry_cluster = RectangleClusteredGeometry()
        self.num_params = self.geometry_cluster.get_rectangle_count()
        self.bounds = self.geometry_cluster.get_bounds()
        self.current_params = self.geometry_cluster.get_current_fractions()
        
        # Initialize dual parameter storage
        self.current_params_raw = self.current_params.copy()      # For gradient calculation
        self.current_params_filtered = self.current_params.copy() # For field calculation
        
        # Setup morphological filter
        self.setup_morphological_filter()
        
        print(f"Geometry Parameter Handler initialized:")
        print(f"  Parameters: {self.num_params}")
        print(f"  Bounds: {self.bounds[0]}")
        print(f"  Constraint: Σ(params) = 1.0")
        print(f"  Morphological filtering: {self.filter_enabled}")
        
    def setup_morphological_filter(self):
        """Setup 2D morphological filtering for minimum feature size enforcement"""
        
        # Get design region from geometry cluster
        design_region = self.geometry_cluster.design_region
        self.total_length = design_region['volume']['length']      # 8μm
        self.total_width = design_region['volume']['width']        # 3μm  
        self.min_feature_size = design_region['constraints']['min_feature_size']  # 150nm
        
        # Morphological filter configuration
        self.filter_enabled = True
        self.pixels_per_micron = 100  # 100 pixels/μm resolution
        
        # Calculate pixel array dimensions
        self.array_width_pixels = int(self.total_length * self.pixels_per_micron)    # 800 pixels
        self.array_height_pixels = int(self.total_width * self.pixels_per_micron)    # 300 pixels
        
        # Morphological structuring element size (circular)
        self.min_feature_radius_pixels = (self.min_feature_size / 2) * self.pixels_per_micron  # ~7.5 pixels
        
        # Create morphological structuring element
        self.structuring_element = self._create_circular_structuring_element(
            self.min_feature_radius_pixels
        )
        
        # Store filter configuration
        self.filter_config = {
            'method': 'morphological_opening',
            'min_feature_size': self.min_feature_size,
            'total_length': self.total_length,
            'total_width': self.total_width, 
            'pixels_per_micron': self.pixels_per_micron,
            'array_width_pixels': self.array_width_pixels,
            'array_height_pixels': self.array_height_pixels,
            'min_feature_radius_pixels': self.min_feature_radius_pixels
        }
        
        print(f"  Morphological Filter Setup:")
        print(f"    Design region: {self.total_length*1e6:.1f} × {self.total_width*1e6:.1f} μm")
        print(f"    Pixel array: {self.array_width_pixels} × {self.array_height_pixels}")
        print(f"    Resolution: {self.pixels_per_micron} pixels/μm")
        print(f"    Min feature: {self.min_feature_size*1e9:.0f} nm")
        print(f"    Structuring element radius: {self.min_feature_radius_pixels:.1f} pixels")
        
    def _create_circular_structuring_element(self, radius):
        """
        Create circular structuring element for morphological operations.
        
        Parameters:
        -----------
        radius : float
            Radius of structuring element in pixels
            
        Returns:
        --------
        element : np.ndarray
            Binary circular structuring element
        """
        
        size = int(2 * radius + 1)
        center = radius
        
        # Create coordinate grids
        y, x = np.ogrid[:size, :size]
        
        # Create circular mask
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        
        return mask.astype(bool)
    
    def apply_morphological_filter(self, raw_params):
        """
        Apply morphological opening filter to enforce minimum feature size.
        
        This implements the complete workflow:
        1. Convert fractional parameters to 2D binary pixel array
        2. Apply morphological opening (erosion followed by dilation)
        3. Convert filtered binary array back to fractional parameters
        
        Parameters:
        -----------
        raw_params : np.ndarray
            Raw fractional width parameters from optimizer
            
        Returns:
        --------
        filtered_params : np.ndarray
            Filtered parameters satisfying minimum feature size constraints
        """
        
        if not self.filter_enabled:
            print("Morphological filter disabled - returning raw parameters")
            return raw_params.copy()
        
        try:
            # Step 1: Convert to 2D binary array
            binary_array = self._fractions_to_binary_array(raw_params)
            
            # Step 2: Apply morphological opening
            filtered_array = self._apply_morphological_opening(binary_array)
            
            # Step 3: Convert back to fractions
            filtered_params = self._binary_array_to_fractions(filtered_array)
            
            # Report filtering statistics
            original_pixels = np.sum(binary_array)
            filtered_pixels = np.sum(filtered_array)
            removed_pixels = original_pixels - filtered_pixels
            
            print(f"Morphological filter applied:")
            print(f"  Original pixels: {original_pixels}")
            print(f"  Filtered pixels: {filtered_pixels}")
            print(f"  Removed pixels: {removed_pixels}")
            
            return filtered_params
            
        except Exception as e:
            print(f"Error in morphological filtering: {e}")
            print("Falling back to raw parameters")
            return raw_params.copy()
    
    def _fractions_to_binary_array(self, raw_params):
        """
        Convert fractional width parameters to 2D binary pixel array.
        
        Creates a binary representation where 1 = optimization material,
        0 = background material, using alternating pattern.
        
        Parameters:
        -----------
        raw_params : np.ndarray
            Fractional width parameters [f1, f2, f3, ...] where Σfi = 1
            
        Returns:
        --------
        binary_array : np.ndarray
            2D binary array (height_pixels × width_pixels)
        """
        
        # Initialize binary array
        binary_array = np.zeros((self.array_height_pixels, self.array_width_pixels), dtype=bool)
        
        # Convert fractional widths to pixel widths
        pixel_widths = raw_params * self.array_width_pixels
        
        # Fill binary array with alternating material pattern
        current_x = 0
        
        for rect_idx, width_pixels in enumerate(pixel_widths):
            width_int = int(np.round(width_pixels))
            
            if width_int > 0:
                end_x = min(current_x + width_int, self.array_width_pixels)
                
                # Alternating material pattern (even = optimization material)
                is_optimization_material = (rect_idx % 2 == 0)
                
                if is_optimization_material:
                    # Fill with True for optimization material
                    binary_array[:, current_x:end_x] = True
                # else: already False for background material
                
                current_x = end_x
        
        return binary_array
    
    def _apply_morphological_opening(self, binary_array):
        """
        Apply morphological opening operation using scipy.ndimage.
        
        Opening = Erosion followed by Dilation
        Removes features smaller than the structuring element.
        
        Parameters:
        -----------
        binary_array : np.ndarray
            Binary array to filter
            
        Returns:
        --------
        opened_array : np.ndarray
            Filtered binary array after morphological opening
        """
        
        try:
            from scipy.ndimage import binary_opening
            
            # Apply morphological opening with circular structuring element
            opened_array = binary_opening(
                binary_array, 
                structure=self.structuring_element,
                iterations=1
            )
            
            return opened_array.astype(bool)
            
        except ImportError:
            print("Warning: scipy.ndimage not available")
            print("Using fallback morphological operations")
            return self._fallback_morphological_opening(binary_array)
    
    def _fallback_morphological_opening(self, binary_array):
        """
        Fallback morphological opening when scipy is not available.
        Implements basic erosion + dilation manually.
        """
        
        # Simple fallback - just return original for now
        # In production, could implement manual erosion/dilation
        print("Using identity filter (no morphological operations)")
        return binary_array.copy()
    
    def _binary_array_to_fractions(self, filtered_array):
        """
        Convert filtered 2D binary array back to fractional width parameters.
        
        Extracts rectangle pattern and converts to fractional widths
        while maintaining sum=1 constraint.
        
        Parameters:
        -----------
        filtered_array : np.ndarray
            2D binary array after morphological filtering
            
        Returns:
        --------
        filtered_params : np.ndarray
            Fractional width parameters that sum to 1
        """
        
        # Extract pattern from middle row (representative slice)
        middle_row = filtered_array.shape[0] // 2
        pattern_row = filtered_array[middle_row, :]
        
        # Find material transitions (0→1 and 1→0)
        # Add padding to detect edge transitions
        padded_pattern = np.concatenate(([False], pattern_row, [False]))
        transitions = np.diff(padded_pattern.astype(int))
        
        region_starts = np.where(transitions == 1)[0]   # Start of material regions
        region_ends = np.where(transitions == -1)[0]    # End of material regions
        
        # Initialize fractional widths
        filtered_fractions = np.zeros(self.num_params)
        
        # Convert regions to fractional parameters
        region_idx = 0
        for start, end in zip(region_starts, region_ends):
            region_width_pixels = end - start
            region_width_fraction = region_width_pixels / self.array_width_pixels
            
            if region_idx < self.num_params and region_width_fraction > 0:
                filtered_fractions[region_idx] = region_width_fraction
                region_idx += 1
        
        # Handle edge cases
        if region_idx == 0:
            # No regions survived - create minimal valid configuration
            print("Warning: All regions eliminated by morphological filter")
            print("Creating minimal single-region configuration")
            filtered_fractions[0] = 1.0
        else:
            # Normalize to ensure sum = 1 (handle numerical precision)
            total_fraction = np.sum(filtered_fractions)
            if total_fraction > 0:
                filtered_fractions = filtered_fractions / total_fraction
            else:
                # All zeros case - uniform distribution
                filtered_fractions = np.ones(self.num_params) / self.num_params
        
        return filtered_fractions
    
    def organize_parameters(self, params):
        """
        Organize and validate parameters from optimizer.
        
        Parameters:
        -----------
        params : array_like
            Raw parameters from optimizer
            
        Returns:
        --------
        organized_params : np.ndarray
            Validated fractional widths
        """
        
        # Convert to numpy array
        params_array = np.array(params, dtype=float)
        
        # Validate parameter count
        if len(params_array) != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameters, got {len(params_array)}")
        
        # Validate bounds
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
        Reorganize gradients for optimizer with sum=1 constraint projection.
        
        Parameters:
        -----------
        gradient_array : array_like
            Raw gradients from field/FOM calculation
            
        Returns:
        --------
        projected_gradients : np.ndarray
            Gradients projected onto constraint manifold
        """
        
        # Convert to numpy array
        gradients = np.array(gradient_array, dtype=float)
        
        # Validate gradient count
        if len(gradients) != self.num_params:
            raise ValueError(f"Expected {self.num_params} gradients, got {len(gradients)}")
        
        # Project onto sum=1 constraint manifold
        # For constraint Σ(xi) = 1, projected gradient: g_proj = g - mean(g)
        gradient_mean = np.mean(gradients)
        projected_gradients = gradients - gradient_mean
        
        print(f"Gradient constraint projection: mean = {gradient_mean:.8f}")
        
        return projected_gradients
    
    # Interface methods for lumNLopt optimization framework
    
    def get_current_params(self):
        """Return current filtered parameters (for optimizer interface)"""
        return self.current_params_filtered.copy()
    
    def get_current_params_raw(self):
        """Return raw parameters (for gradient calculation)"""
        return self.current_params_raw.copy()
    
    def get_current_params_filtered(self):
        """Return filtered parameters (for field calculation)"""
        return self.current_params_filtered.copy()
    
    def update_geometry(self, params, sim=None):
        """
        Update geometry from new parameters (main interface method).
        
        Parameters:
        -----------
        params : array_like
            Fractional width parameters from optimizer
        sim : object, optional
            Lumerical simulation object
        """
        
        # Organize and validate parameters
        raw_params = self.organize_parameters(params)
        
        # Apply morphological filtering
        filtered_params = self.apply_morphological_filter(raw_params)
        
        # Update geometry cluster with filtered parameters
        rectangles = self.geometry_cluster.update_rectangles_from_fractions(filtered_params)
        
        # Update Lumerical simulation if provided
        if sim is not None:
            self.geometry_cluster.update_lumerical_simulation(sim.fdtd)
        
        # Store both parameter sets
        self.current_params_raw = raw_params
        self.current_params_filtered = filtered_params
        self.current_params = filtered_params  # For compatibility
        
        return rectangles
    
    def calculate_gradients(self, gradient_fields):
        """
        Calculate gradients w.r.t. fractional width parameters.
        
        Parameters:
        -----------
        gradient_fields : object
            Gradient field data from lumNLopt
            
        Returns:
        --------
        projected_gradients : np.ndarray
            Gradients w.r.t. raw parameters (for optimizer)
        """
        
        print("Calculating morphology-aware gradients...")
        
        # Placeholder gradient calculation
        # In full implementation, this would:
        # 1. Extract field gradients from gradient_fields
        # 2. Calculate rectangle boundary gradients
        # 3. Apply chain rule through morphological filter
        # 4. Transform to raw parameter gradients
        
        gradients = self._calculate_placeholder_gradients(gradient_fields)
        
        # Apply constraint projection
        projected_gradients = self.reorganize_gradients(gradients)
        
        return projected_gradients
    
    def _calculate_placeholder_gradients(self, gradient_fields):
        """Placeholder gradient calculation (to be replaced with actual implementation)"""
        
        # Random gradients for testing
        gradients = np.random.normal(0, 0.1, self.num_params)
        
        print(f"Generated {len(gradients)} placeholder gradients")
        return gradients
    
    def get_bounds(self):
        """Return parameter bounds for optimizer"""
        return self.bounds
    
    def get_constraint_function(self):
        """
        Return constraint function for sum=1 equality constraint.
        For use with NLopt or other constrained optimizers.
        """
        
        def sum_constraint(params, grad):
            """Sum constraint: sum(params) - 1 = 0"""
            if grad.size > 0:
                grad[:] = 1.0  # Gradient is 1 for all parameters
            return np.sum(params) - 1.0
        
        return sum_constraint
    
    def add_geo(self, sim, params=None, only_update=False):
        """
        Add/update geometry in Lumerical simulation.
        Required interface method for lumNLopt.
        """
        
        if params is not None:
            self.update_geometry(params, sim)
        else:
            self.geometry_cluster.update_lumerical_simulation(sim.fdtd)
        
        action = "updated" if only_update else "added"
        print(f"Geometry {action} in simulation with morphological constraints")
    
    def plot(self, ax=None):
        """Plot current rectangle configuration"""
        return self.geometry_cluster.plot_rectangle_configuration(ax)
    
    def use_interpolation(self):
        """Rectangle clustering doesn't use material interpolation"""
        return False
    
    def validate_morphological_filter(self):
        """Validate morphological filter setup and results"""
        
        if not self.filter_enabled:
            print("Morphological filter is disabled")
            return True
        
        config = self.filter_config
        
        # Check pixel resolution
        min_pixels_per_feature = config['min_feature_radius_pixels'] * 2
        if min_pixels_per_feature < 4:
            print(f"Warning: Low resolution for minimum feature size")
            print(f"  Feature diameter: {min_pixels_per_feature:.1f} pixels")
            print(f"  Recommended: ≥ 4 pixels for accurate morphological operations")
        
        # Check array size
        total_pixels = config['array_width_pixels'] * config['array_height_pixels']
        if total_pixels > 1e6:
            print(f"Warning: Large pixel array ({total_pixels:.0f} pixels)")
            print("  This may slow down morphological operations")
        
        # Validate sum constraint preservation
        if hasattr(self, 'current_params_filtered'):
            sum_violation = abs(np.sum(self.current_params_filtered) - 1.0)
            if sum_violation > 1e-6:
                print(f"ERROR: Morphological filter violated sum constraint: {sum_violation:.2e}")
                return False
        
        print("✓ Morphological filter validation passed")
        return True
    
    def test_morphological_filter(self, test_params=None):
        """
        Test morphological filter on sample parameters.
        
        Parameters:
        -----------
        test_params : np.ndarray, optional
            Parameters to test. If None, uses current raw parameters.
        """
        
        if test_params is None:
            test_params = self.current_params_raw.copy()
        
        print("\nTesting morphological filter:")
        print(f"Input params: {test_params}")
        print(f"Input sum: {np.sum(test_params):.8f}")
        
        # Apply filter
        filtered_params = self.apply_morphological_filter(test_params)
        
        print(f"Filtered params: {filtered_params}")
        print(f"Filtered sum: {np.sum(filtered_params):.8f}")
        
        # Check feature size compliance
        total_length = self.filter_config['total_length']
        raw_widths = test_params * total_length
        filtered_widths = filtered_params * total_length
        
        raw_violations = np.sum(raw_widths < self.filter_config['min_feature_size'])
        filtered_violations = np.sum(filtered_widths < self.filter_config['min_feature_size'])
        
        print(f"Feature size violations:")
        print(f"  Raw: {raw_violations}")
        print(f"  Filtered: {filtered_violations}")
        print(f"  Removed: {raw_violations - filtered_violations}")
        
        if raw_violations > 0:
            print(f"Width ranges:")
            print(f"  Raw: [{np.min(raw_widths)*1e9:.0f}, {np.max(raw_widths)*1e9:.0f}] nm")
            print(f"  Filtered: [{np.min(filtered_widths)*1e9:.0f}, {np.max(filtered_widths)*1e9:.0f}] nm")
        
        return filtered_params
    
    def print_parameter_summary(self):
        """Print comprehensive summary of parameters and morphological filter"""
        
        print("\n" + "="*70)
        print("GEOMETRY PARAMETER HANDLER - MORPHOLOGICAL FILTERING")
        print("="*70)
        
        # Basic parameter info
        print(f"Parameter count: {self.num_params}")
        print(f"Raw parameters: {self.current_params_raw}")
        print(f"Filtered parameters: {self.current_params_filtered}")
        print(f"Raw sum: {np.sum(self.current_params_raw):.8f}")
        print(f"Filtered sum: {np.sum(self.current_params_filtered):.8f}")
        print(f"Parameter bounds: [{self.bounds[0][0]:.6f}, {self.bounds[0][1]:.6f}]")
        
        # Constraint validation
        raw_violation = abs(np.sum(self.current_params_raw) - 1.0)
        filtered_violation = abs(np.sum(self.current_params_filtered) - 1.0)
        print(f"Raw sum constraint violation: {raw_violation:.2e}")
        print(f"Filtered sum constraint violation: {filtered_violation:.2e}")
        
        # Morphological filter details
        print(f"\nMorphological Filter Configuration:")
        print(f"  Status: {'Enabled' if self.filter_enabled else 'Disabled'}")
        
        if self.filter_enabled:
            config = self.filter_config
            print(f"  Method: Binary morphological opening")
            print(f"  Design region: {config['total_length']*1e6:.1f} × {config['total_width']*1e6:.1f} μm")
            print(f"  Pixel array: {config['array_width_pixels']} × {config['array_height_pixels']}")
            print(f"  Resolution: {config['pixels_per_micron']} pixels/μm")
            print(f"  Min feature size: {config['min_feature_size']*1e9:.0f} nm")
            print(f"  Structuring element radius: {config['min_feature_radius_pixels']:.1f} pixels")
            
            # Feature size compliance
            if hasattr(self, 'current_params_raw') and hasattr(self, 'current_params_filtered'):
                total_length = config['total_length']
                raw_widths = self.current_params_raw * total_length
                filtered_widths = self.current_params_filtered * total_length
                
                raw_violations = np.sum(raw_widths < config['min_feature_size'])
                filtered_violations = np.sum(filtered_widths < config['min_feature_size'])
                
                print(f"  Feature size compliance:")
                print(f"    Raw violations: {raw_violations}")
                print(f"    Filtered violations: {filtered_violations}")
                print(f"    Violations removed: {raw_violations - filtered_violations}")
                
                if raw_violations > 0 or filtered_violations > 0:
                    print(f"    Raw range: [{np.min(raw_widths)*1e9:.0f}, {np.max(raw_widths)*1e9:.0f}] nm")
                    print(f"    Filtered range: [{np.min(filtered_widths)*1e9:.0f}, {np.max(filtered_widths)*1e9:.0f}] nm")
        
        print("="*70)
        
        # Print geometry cluster summary
        self.geometry_cluster.print_summary()

# Convenience functions for lumNLopt framework integration

def create_rectangle_clustering_geometry():
    """Factory function for creating morphology-filtered rectangle clustering"""
    return GeometryParameterHandler()

def get_parameter_bounds():
    """Get parameter bounds for rectangle clustering optimization"""
    handler = GeometryParameterHandler()
    return handler.get_bounds()

def get_parameter_count():
    """Get number of optimization parameters"""
    handler = GeometryParameterHandler()
    return handler.num_params
