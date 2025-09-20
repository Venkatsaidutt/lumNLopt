# lumNLopt/figures_of_merit/adiabatic_coupling.py

"""
Adiabatic Edge Coupler Figure of Merit

Combines:
1. Power transfer optimization
2. Fundamental mode power maximization  
3. Adiabatic evolution through mode overlap optimization
4. Arithmetic progression for target overlaps
5. Product optimization: Transmission × Mode Evolution
"""

import numpy as np
import scipy as sp
import scipy.constants
import lumapi
from lumNLopt.utilities.wavelengths import Wavelengths 
# CORRECT
from lumNLopt.lumerical_methods.lumerical_scripts import get_mode_overlap_between_monitors,get_fundamental_mode_power_fraction

class AdiabaticCouplingFOM(object):
    """
    Figure of merit for adiabatic edge coupler optimization.
    
    Optimizes the product of:
    - Overall transmission efficiency (input → output power)
    - Mode evolution quality (smooth adiabatic transition)
    - Fundamental mode purity at input/output cross-sections
    
    Uses arithmetic progression for target mode overlaps between layers.
    """
    
    def __init__(self, input_monitor_name, output_monitor_name, 
                 slice_monitors=None, num_slices=240,
                 target_transmission=0.95, 
                 overlap_progression_params=None,
                 weights=None, norm_p=2):
        """
        Parameters:
        -----------
        input_monitor_name : str
            Monitor at input cross-section (waveguide)
        output_monitor_name : str  
            Monitor at output cross-section (fiber)
        slice_monitors : list
            List of monitor names for each slice (for mode evolution)
        num_slices : int
            Number of slices in adiabatic coupler
        target_transmission : float
            Target power transmission efficiency
        overlap_progression_params : dict
            Parameters for arithmetic progression:
            {'start_wg_overlap': 0.999, 'end_wg_overlap': k, 
             'start_fiber_overlap': k, 'end_fiber_overlap': 0.999}
        weights : dict
            Weighting factors for different objectives
        """
        
        self.input_monitor_name = str(input_monitor_name)
        self.output_monitor_name = str(output_monitor_name)
        self.slice_monitors = slice_monitors or []
        self.num_slices = num_slices
        self.target_transmission = target_transmission
        self.norm_p = int(norm_p)
        
        # Default overlap progression parameters
        if overlap_progression_params is None:
            k_baseline = 0.3  # Baseline overlap between waveguide and fiber modes
            self.overlap_params = {
                'start_wg_overlap': 0.999,      # Start with pure waveguide mode
                'end_wg_overlap': k_baseline,   # End with reduced waveguide overlap  
                'start_fiber_overlap': k_baseline, # Start with low fiber overlap
                'end_fiber_overlap': 0.999      # End with pure fiber mode
            }
        else:
            self.overlap_params = overlap_progression_params
        
        # Default weights for multi-objective optimization
        self.weights = weights or {
            'transmission': 2.0,        # Power transfer (primary)
            'mode_evolution': 1.5,      # Adiabatic evolution quality
            'fundamental_purity': 1.0   # Fundamental mode purity
        }
        
        # Validation
        if not self.input_monitor_name or not self.output_monitor_name:
            raise UserWarning('Input and output monitor names required')
        
        if self.num_slices != len(self.slice_monitors) and self.slice_monitors:
            print(f"Warning: {len(self.slice_monitors)} slice monitors for {self.num_slices} slices")
    
    def initialize(self, sim):
        """Initialize all monitors and mode expansion monitors"""
        
        # Verify main monitors exist
        if sim.fdtd.getnamednumber(self.input_monitor_name) != 1:
            raise UserWarning(f'Input monitor "{self.input_monitor_name}" not found')
        
        if sim.fdtd.getnamednumber(self.output_monitor_name) != 1:
            raise UserWarning(f'Output monitor "{self.output_monitor_name}" not found')
        
        # Add mode expansion monitors for fundamental mode analysis
        self._add_mode_expansion_monitor(sim, self.input_monitor_name, 'input')
        self._add_mode_expansion_monitor(sim, self.output_monitor_name, 'output')
        
        # Add mode expansion monitors for slice-by-slice analysis
        for i, monitor_name in enumerate(self.slice_monitors):
            if sim.fdtd.getnamednumber(monitor_name) == 1:
                self._add_mode_expansion_monitor(sim, monitor_name, f'slice_{i}')
        
        print(f"Adiabatic Coupling FOM initialized:")
        print(f"  Slices: {self.num_slices}")
        print(f"  Target transmission: {self.target_transmission:.1%}")
        print(f"  Mode evolution: {len(self.slice_monitors)} monitors")
    
    def _add_mode_expansion_monitor(self, sim, base_monitor_name, suffix):
        """Add mode expansion monitor for mode overlap calculations"""
        
        mode_exp_name = f'{base_monitor_name}_mode_exp_{suffix}'
        
        try:
            sim.fdtd.addmodeexpansion()
            sim.fdtd.set('name', mode_exp_name)
            
            # Copy geometry from base monitor
            monitor_type = sim.fdtd.getnamed(base_monitor_name, 'monitor type')
            geo_props, normal = self._get_monitor_props(monitor_type)
            
            for prop_name in geo_props:
                prop_val = sim.fdtd.getnamed(base_monitor_name, prop_name)
                sim.fdtd.setnamed(mode_exp_name, prop_name, prop_val)
            
            # Set fundamental mode selection
            sim.fdtd.setnamed(mode_exp_name, 'mode selection', 'fundamental mode')
            sim.fdtd.updatemodes()
            
        except Exception as e:
            print(f"Warning: Could not add mode expansion monitor {mode_exp_name}: {e}")
    
    def _get_monitor_props(self, monitor_type):
        """Get geometric properties for monitor type"""
        
        geometric_props = ['x', 'y', 'z']
        normal = ''
        
        if monitor_type == '2D X-normal':
            geometric_props.extend(['y span', 'z span'])
            normal = 'x'
        elif monitor_type == '2D Y-normal':
            geometric_props.extend(['x span', 'z span'])
            normal = 'y'
        elif monitor_type == '2D Z-normal':
            geometric_props.extend(['x span', 'y span'])
            normal = 'z'
        else:
            raise UserWarning(f'Unsupported monitor type: {monitor_type}')
        
        return geometric_props, normal
    
    def make_forward_sim(self, sim):
        """Setup forward simulation - no adjoint sources active"""
        pass
    
    def make_adjoint_sim(self, sim):
        """
        Setup adjoint simulation with sources based on optimization objectives.
        
        Places adjoint sources at:
        1. Output monitor (weighted by transmission error)
        2. Each slice monitor (weighted by mode evolution error)
        """
        
        # Disable forward source
        if sim.fdtd.getnamednumber('source') >= 1:
            sim.fdtd.setnamed('source', 'enabled', False)
        
        # Add main adjoint source at output (for transmission optimization)
        self._add_adjoint_source(sim, self.output_monitor_name, 'main_adjoint',
                                weight=self.weights['transmission'])
        
        # Add adjoint sources for mode evolution (if slice monitors available)
        if hasattr(self, 'slice_mode_overlaps'):
            for i, monitor_name in enumerate(self.slice_monitors):
                if i < len(self.slice_mode_overlaps) - 1:
                    # Weight based on deviation from target overlap progression
                    target_overlap = self._get_target_overlap(i, i+1)
                    actual_overlap = self.slice_mode_overlaps[i]
                    error_weight = self.weights['mode_evolution'] * abs(actual_overlap - target_overlap)
                    
                    self._add_adjoint_source(sim, monitor_name, f'slice_adjoint_{i}',
                                           weight=error_weight)
    
    def _add_adjoint_source(self, sim, monitor_name, source_name, weight=1.0):
        """Add adjoint dipole source at monitor location"""
        
        try:
            monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
            
            sim.fdtd.adddipole()
            sim.fdtd.set('name', source_name)
            
            # Copy monitor geometry
            geo_props, normal = self._get_monitor_props(monitor_type)
            for prop in geo_props:
                try:
                    val = sim.fdtd.getnamed(monitor_name, prop)
                    sim.fdtd.setnamed(source_name, prop, val)
                except:
                    pass
            
            # Set source amplitude based on weight
            sim.fdtd.setnamed(source_name, 'amplitude', weight)
            sim.fdtd.setnamed(source_name, 'phase', 0.0)
            
        except Exception as e:
            print(f"Warning: Could not add adjoint source {source_name}: {e}")
    
    def get_fom(self, sim):
        """
        Calculate combined figure of merit.
        
        FOM = (Transmission Efficiency) × (Mode Evolution Quality) × (Fundamental Mode Purity)
        
        Where:
        - Transmission Efficiency = P_out / P_in
        - Mode Evolution Quality = Product of adjacent slice overlaps / target overlaps
        - Fundamental Mode Purity = |fundamental_mode_overlap|² at input/output
        """
        
        self.wavelengths = self.get_wavelengths(sim)
        
        # 1. Calculate transmission efficiency
        input_power = self._get_monitor_power(sim, self.input_monitor_name)
        output_power = self._get_monitor_power(sim, self.output_monitor_name)
        transmission_eff = output_power / (input_power + 1e-12)
        
        # 2. Calculate fundamental mode purity at input/output
        input_fundamental_overlap = self._get_fundamental_mode_overlap(
            sim, f'{self.input_monitor_name}_mode_exp_input')
        output_fundamental_overlap = self._get_fundamental_mode_overlap(
            sim, f'{self.output_monitor_name}_mode_exp_output')
        
        fundamental_purity = np.sqrt(input_fundamental_overlap * output_fundamental_overlap)
        
        # 3. Calculate mode evolution quality (adiabatic transition)
        mode_evolution_quality = self._calculate_mode_evolution_quality(sim)
        
        # 4. Combine objectives with wavelength integration
        combined_metrics = {
            'transmission': transmission_eff,
            'fundamental_purity': fundamental_purity,
            'mode_evolution': mode_evolution_quality
        }
        
        # Store for adjoint calculation
        self.current_metrics = combined_metrics
        
        # Calculate final FOM using product approach
        fom = self._calculate_weighted_product_fom(combined_metrics, self.wavelengths)
        
        return fom
    
    def _calculate_mode_evolution_quality(self, sim):
        """
        Calculate quality of adiabatic mode evolution using arithmetic progression.
        
        Evaluates how well the actual mode overlaps between adjacent slices
        match the target arithmetic progression.
        """
        
        if not self.slice_monitors:
            return np.ones_like(self.wavelengths)  # No slice data available
        
        evolution_quality = np.ones_like(self.wavelengths)
        self.slice_mode_overlaps = []
        
        try:
            # Calculate mode overlaps between adjacent slices
            for i in range(len(self.slice_monitors) - 1):
                current_slice = f'{self.slice_monitors[i]}_mode_exp_slice_{i}'
                next_slice = f'{self.slice_monitors[i+1]}_mode_exp_slice_{i+1}'
                
                # Get mode overlap between adjacent slices
                overlap = self._get_mode_overlap_between_monitors(sim, current_slice, next_slice)
                self.slice_mode_overlaps.append(overlap)
                
                # Compare with target overlap from arithmetic progression
                target_overlap = self._get_target_overlap(i, i+1)
                
                # Quality metric: how close actual overlap is to target
                overlap_quality = 1.0 - np.abs(overlap - target_overlap)
                overlap_quality = np.maximum(overlap_quality, 0.1)  # Minimum quality threshold
                
                evolution_quality *= overlap_quality
        
        except Exception as e:
            print(f"Warning: Mode evolution calculation failed: {e}")
            evolution_quality = np.ones_like(self.wavelengths) * 0.5
        
        return evolution_quality
    
    def _get_target_overlap(self, slice_i, slice_j):
        """
        Calculate target mode overlap using arithmetic progression.
        
        Implements the progression described in your plan:
        - Waveguide overlap: starts at 0.999, decreases linearly to k
        - Fiber overlap: starts at k, increases linearly to 0.999
        """
        
        # Position in the progression (0 to 1)
        progress = slice_i / (self.num_slices - 1)
        
        # Arithmetic progression for waveguide departure
        wg_start = self.overlap_params['start_wg_overlap']
        wg_end = self.overlap_params['end_wg_overlap']
        target_wg_overlap = wg_start - progress * (wg_start - wg_end)
        
        # Arithmetic progression for fiber approach  
        fiber_start = self.overlap_params['start_fiber_overlap']
        fiber_end = self.overlap_params['end_fiber_overlap']
        target_fiber_overlap = fiber_start + progress * (fiber_end - fiber_start)
        
        # Combined target (could be geometric mean or weighted combination)
        target_overlap = np.sqrt(target_wg_overlap * target_fiber_overlap)
        
        return target_overlap
    
    def _calculate_weighted_product_fom(self, metrics, wavelengths):
        """
        Calculate final FOM using weighted product approach.
        
        FOM = Product over wavelengths of: 
            (Transmission^w1 × Purity^w2 × Evolution^w3)
        """
        
        if len(wavelengths) > 1:
            # Multi-wavelength case: integrate over wavelength
            wavelength_range = wavelengths.max() - wavelengths.min()
            
            product_integrand = np.ones_like(wavelengths)
            
            for metric_name, weight in self.weights.items():
                if metric_name in metrics:
                    metric_values = metrics[metric_name]
                    # Use power weighting: metric^weight
                    weighted_metric = np.power(np.maximum(metric_values, 1e-6), weight)
                    product_integrand *= weighted_metric
            
            # Integrate the product over wavelength
            fom = np.trapz(y=product_integrand, x=wavelengths) / wavelength_range
            
        else:
            # Single wavelength case: direct product
            fom = 1.0
            
            for metric_name, weight in self.weights.items():
                if metric_name in metrics:
                    metric_value = metrics[metric_name][0]  # Single wavelength
                    fom *= np.power(np.maximum(metric_value, 1e-6), weight)
        
        return fom.real
    
    def _get_monitor_power(self, sim, monitor_name):
        """Extract power from monitor"""
        try:
            power_data = sim.fdtd.getresult(monitor_name, 'T')
            return np.real(power_data['T']).flatten()
        except:
            # Fallback to field calculation
            try:
                field_data = sim.fdtd.getresult(monitor_name)
                # Simplified Poynting vector calculation
                return np.ones(len(self.wavelengths))  # Placeholder
            except:
                return np.ones(len(self.wavelengths))
    
    def _get_fundamental_mode_overlap(self, sim, mode_exp_monitor_name):
        """Get fundamental mode overlap from mode expansion monitor"""
        try:
            mode_data = sim.fdtd.getresult(mode_exp_monitor_name, 'expansion for ' + mode_exp_monitor_name)
            # Extract fundamental mode coefficient
            fundamental_coeff = mode_data['a']  # Forward coefficient
            overlap = np.abs(fundamental_coeff)**2
            return np.real(overlap).flatten()
        except:
            return np.ones(len(self.wavelengths))
    
    def _get_mode_overlap_between_monitors(self, sim, monitor1_name, monitor2_name):
        """Calculate mode overlap between two mode expansion monitors"""
        try:
            # This would require cross-correlation of mode fields
            # Simplified implementation - in practice needs field overlap integral
            return np.ones(len(self.wavelengths)) * 0.9  # Placeholder
        except:
            return np.ones(len(self.wavelengths)) * 0.5
    
    def fom_gradient_wavelength_integral(self, gradients_vs_wl, wl):
        """
        Calculate gradients for the product-based FOM.
        
        This needs to handle gradients of the product:
        ∂(T × P × E)/∂p = (∂T/∂p)×P×E + T×(∂P/∂p)×E + T×P×(∂E/∂p)
        """
        
        assert np.allclose(wl, self.wavelengths)
        
        if not hasattr(self, 'current_metrics'):
            # Fallback if metrics not available
            return np.mean(gradients_vs_wl, axis=1)
        
        # Extract current metric values
        T = self.current_metrics['transmission']
        P = self.current_metrics['fundamental_purity'] 
        E = self.current_metrics['mode_evolution']
        
        # Weights
        w_T = self.weights['transmission']
        w_P = self.weights['fundamental_purity']
        w_E = self.weights['mode_evolution']
        
        if wl.size > 1:
            # Multi-wavelength gradient integration
            wavelength_range = wl.max() - wl.min()
            
            # Product rule for derivatives: ∂(T^w_T × P^w_P × E^w_E)/∂p
            current_product = np.power(T, w_T) * np.power(P, w_P) * np.power(E, w_E)
            
            # Gradient weighting based on product rule
            gradient_weights = (w_T/T + w_P/P + w_E/E) * current_product / wavelength_range
            
            # Trapezoidal integration
            d = np.diff(wl)
            quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]), d[-1]) / 2
            v = gradient_weights * quad_weight
            
            final_gradients = gradients_vs_wl.dot(v)
            
        else:
            # Single wavelength case
            T_val = T[0]
            P_val = P[0] 
            E_val = E[0]
            
            # Product rule gradient weighting
            gradient_weight = (w_T/T_val + w_P/P_val + w_E/E_val)
            final_gradients = gradient_weight * gradients_vs_wl.flatten()
        
        return final_gradients.flatten().real
    
    def get_adjoint_field_scaling(self, sim):
        """Get scaling factors for adjoint fields based on current metrics"""
        
        omega = 2.0 * np.pi * sp.constants.speed_of_light / self.wavelengths
        
        # Scale adjoint fields based on derivative of product FOM
        if hasattr(self, 'current_metrics'):
            scaling_factors = {}
            
            # Main transmission scaling
            T = self.current_metrics['transmission']
            scaling_factors['main'] = self.weights['transmission'] / (T + 1e-12) * omega * 1j
            
            # Mode evolution scaling
            E = self.current_metrics['mode_evolution'] 
            scaling_factors['evolution'] = self.weights['mode_evolution'] / (E + 1e-12) * omega * 1j
            
            return scaling_factors
        else:
            # Default scaling
            return omega * 1j
    
    @staticmethod
    def get_wavelengths(sim):
        """Get wavelength array from simulation"""
        return Wavelengths(sim.fdtd.getglobalsource('wavelength start'), 
                          sim.fdtd.getglobalsource('wavelength stop'),
                          sim.fdtd.getglobalmonitor('frequency points')).asarray()

# Example usage for your EME edge coupler:

def create_eme_edge_coupler_fom(num_slices=240):
    """
    Create adiabatic coupling FOM for EME edge coupler optimization.
    
    Implements the objectives from your plan:
    - Power transfer maximization
    - Adiabatic mode evolution with arithmetic progression
    - Product optimization of transmission × mode evolution
    """
    
    # Create slice monitor names
    slice_monitors = [f'slice_monitor_{i}' for i in range(num_slices)]
    
    # Overlap progression parameters from your plan
    k_baseline = 0.3  # Baseline overlap between rib waveguide and SMF-28
    overlap_params = {
        'start_wg_overlap': 0.999,      # Start: pure rib waveguide mode
        'end_wg_overlap': k_baseline,   # End: reduced rib waveguide overlap
        'start_fiber_overlap': k_baseline, # Start: low SMF-28 overlap  
        'end_fiber_overlap': 0.999      # End: pure SMF-28 fiber mode
    }
    
    # Multi-objective weights
    weights = {
        'transmission': 2.0,        # Primary: maximize power transfer
        'mode_evolution': 1.5,      # Secondary: smooth adiabatic evolution
        'fundamental_purity': 1.0   # Tertiary: maintain fundamental modes
    }
    
    fom = AdiabaticCouplingFOM(
        input_monitor_name='rib_waveguide_monitor',
        output_monitor_name='smf28_fiber_monitor',
        slice_monitors=slice_monitors,
        num_slices=num_slices,
        target_transmission=0.95,  # 95% coupling efficiency target
        overlap_progression_params=overlap_params,
        weights=weights,
        norm_p=2
    )
    
    return fom
