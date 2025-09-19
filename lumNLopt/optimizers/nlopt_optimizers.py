
import nlopt
import numpy as np
from lumopt.optimizers.optimizer import Optimizer

class NLoptOptimizer(Optimizer):
    """
    NLopt wrapper optimized for rectangle clustering with anisotropic materials.
    Default algorithm: LD_MMA (Method of Moving Asymptotes)
    """
    
    def __init__(self, algorithm='LD_MMA', max_iter=100, ftol_rel=1e-6, 
                 xtol_rel=1e-4, verbose=True):
        
        self.algorithm_name = algorithm  
        self.max_iter = max_iter
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.verbose = verbose
        
        # Algorithm mapping
        self.algorithm_map = {
            'LD_MMA': nlopt.LD_MMA,           # DEFAULT - best for topology/material optimization
            'LD_CCSAQ': nlopt.LD_CCSAQ,      # More stable alternative
            'LD_SLSQP': nlopt.LD_SLSQP,      # Handles equality constraints (sum=1)
            'LD_LBFGS': nlopt.LD_LBFGS,      # Fast unconstrained
        }
        
        self.iteration = 0
        self.fom_history = []
        self.param_history = []
        
        print(f"NLopt optimizer initialized with {algorithm} (default for rectangle clustering)")

    def optimize(self, optimization_problem):
        """Run optimization with enhanced support for rectangle clustering"""
        
        geometry = optimization_problem.geometry
        initial_params = geometry.get_current_params()
        n_params = len(initial_params)
        
        # Initialize NLopt
        algorithm = self.algorithm_map[self.algorithm_name]
        opt = nlopt.opt(algorithm, n_params)
        
        # Set bounds
        bounds = geometry.bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        
        # Set objective (maximization for FOM)
        opt.set_max_objective(self._objective_function)
        
        # Add sum=1 constraint for rectangle clustering
        if hasattr(geometry, 'get_constraint_function'):
            constraint_func = geometry.get_constraint_function()
            opt.add_equality_constraint(
                lambda x, grad: self._constraint_wrapper(x, grad, constraint_func, geometry), 
                1e-8
            )
            print("Added rectangle clustering constraint: sum(fractions) = 1")
        
        # Set convergence criteria
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_xtol_rel(self.xtol_rel) 
        opt.set_maxeval(self.max_iter)
        
        # Store reference
        self.opt_problem = optimization_problem
        self.iteration = 0
        
        try:
            print(f"\nStarting {self.algorithm_name} optimization for anisotropic rectangle clustering")
            print(f"Parameters: {n_params}, Max iterations: {self.max_iter}")
            print(f"Initial sum: {np.sum(initial_params):.6f}")
            
            optimal_params = opt.optimize(initial_params)
            optimal_fom = opt.last_optimum_value()
            
            print(f"\nOptimization completed successfully!")
            print(f"Final FOM: {optimal_fom:.6f}")
            print(f"Final parameter sum: {np.sum(optimal_params):.6f}")
            print(f"Total iterations: {self.iteration}")
            
            return optimal_params, optimal_fom
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            raise

    def _objective_function(self, params, grad):
        """Enhanced objective function with anisotropic gradient support"""
        
        self.iteration += 1
        params_normalized = params / np.sum(params)  # Ensure constraint
        
        try:
            # Update geometry
            geometry = self.opt_problem.geometry  
            geometry.update_geometry(params_normalized)
            
            # Run simulations
            print(f"\nIteration {self.iteration}")
            print(f"Parameters: {params_normalized}")
            
            self.opt_problem.run_forward_solves()
            current_fom = self.opt_problem.calculate_fom()
            
            print(f"FOM: {current_fom:.6f}")
            
            # Calculate gradients if needed
            if grad.size > 0:
                print("Calculating anisotropic gradients...")
                
                self.opt_problem.run_adjoint_solves()
                
                # Use custom anisotropic gradient calculation
                if hasattr(geometry, 'calculate_gradients_manual'):
                    gradients = geometry.calculate_gradients_manual(
                        self.opt_problem.forward_fields,
                        self.opt_problem.adjoint_fields,
                        self.opt_problem.wavelengths
                    )
                else:
                    # Fallback
                    gradients = self.opt_problem.calculate_gradients()
                
                grad[:] = gradients  # CRITICAL: In-place modification for NLopt
                print(f"Gradients: {gradients}")
                print(f"Gradient norm: {np.linalg.norm(gradients):.4e}")
            
            # Store history
            self.fom_history.append(current_fom)
            self.param_history.append(params_normalized.copy())
            
            return current_fom
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            if grad.size > 0:
                grad[:] = 0.0
            return -1e6  # Poor FOM for errors

    def _constraint_wrapper(self, params, grad, constraint_func, geometry):
        """Handle sum=1 constraint for rectangle clustering"""
        
        constraint_value = constraint_func(params) 
        
        if grad.size > 0:
            if hasattr(geometry, 'get_constraint_jacobian'):
                constraint_jac = geometry.get_constraint_jacobian()
                grad[:] = constraint_jac(params)
            else:
                grad[:] = 1.0  # d(sum)/dp_i = 1
        
        return constraint_value
