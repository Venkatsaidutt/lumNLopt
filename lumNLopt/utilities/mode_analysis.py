"""
Mode analysis utilities for adiabatic coupling optimization
"""

import numpy as np
import scipy as sp

class ModeAnalyzer:
    """Utility class for mode overlap and evolution analysis"""
    
    @staticmethod
    def calculate_mode_overlap_matrix(sim, monitor_names):
        """Calculate overlap matrix between all monitor pairs"""
        n_monitors = len(monitor_names)
        overlap_matrix = np.zeros((n_monitors, n_monitors))
        
        for i, mon1 in enumerate(monitor_names):
            for j, mon2 in enumerate(monitor_names):
                if i != j:
                    overlap = get_mode_overlap_between_monitors(sim.fdtd, mon1, mon2)
                    overlap_matrix[i, j] = np.mean(overlap)
                else:
                    overlap_matrix[i, j] = 1.0
        
        return overlap_matrix
    
    @staticmethod
    def validate_adiabatic_progression(overlaps, target_progression):
        """Validate how well actual overlaps match target progression"""
        errors = np.abs(overlaps - target_progression)
        return 1.0 - np.mean(errors)  # Quality metric
