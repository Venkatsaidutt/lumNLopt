# File: lumNLopt/optimizers/nlopt_optimizers.py

import nlopt
import numpy as np
from lumopt.optimizers.optimizer import Optimizer

class NLoptOptimizer(Optimizer):
    """
    NLopt wrapper for rectangle clustering optimization.
    Default algorithm: LD_MMA (Method of Moving Asymptotes)
    """
    
    def __init__(self, algorithm='LD_MMA', max_iter=100, ftol_rel=1e-6, 
                 xtol_rel=1e-4, verbose=True):
        """
        Parameters:
        -----------
        algorithm : str, default 'LD_MMA'
            NLopt algorithm name. Options:
            - 'LD_MMA': Method of Moving Asymptotes (best for topology opt)
            - 'LD_CCSAQ': Conservative Convex Separable Approximations
            - 'LD_SLSQP': Sequential Quadratic Programming (handles constraints)
            - 'LD_LBFGS': L-BFGS (fast, no constraints)
        """
        self.algorithm_name = algorithm
        self.max_iter = max_iter
        self.ftol_rel = ftol_rel
        self.xtol_rel = xtol_rel
        self.verbose = verbose
        
        # Map string names to NLopt constants
        self.algorithm_map = {
            'LD_MMA': nlopt.LD_MMA,
            'LD_CCSAQ': nlopt.LD_CCSAQ,
            'LD_SLSQP': nlopt.LD_SLSQP,
            'LD_LBFGS': nlopt.LD_LBFGS,
            'LD_TNEWTON': nlopt.LD_TNEWTON,
            'LD_VAR1': nlopt.LD_VAR1,
            'LD_VAR2': nlopt.LD_VAR2
        }
        
        # Storage for optimization data
        self.iteration = 0
        self.fom_history = []
        self.param_history = []
        
        print(f"NLopt optimizer initialized with {algorithm}")

    def optimize(self, optimization_problem):
        """
        Run optimization using NLopt with rectangle clustering geometry.
        
        Parameters:
        -----------
        optimization_problem : OptimizationProblem
            Contains geometry, figure of merit, and simulation setup
        """
        
        # Extract initial parameters from geometry
        geometry = optimization_problem.geometry
        initial_params = geometry.get_current_params()
        n_params = len(initial_params)
        
        print(f"Starting optimization with {n_params} parameters")
        print(f"Initial parameters: {initial_params}")
        print(f"Parameter sum: {np.sum(initial_params):.6f}")
        
        # Initialize NLopt optimizer
        if self.algorithm_name not in self.algorithm_map:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
            
        algorithm = self.algorithm_map[self.algorithm_name]
        opt = nlopt.opt(algorithm, n_params)
        
        # Set bounds from geometry
        bounds = geometry.bounds
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        
        # Set optimization objective
        opt.set_max_objective(self._objective_function)
        
        # Handle sum=1 constraint for rectangle clustering
        if hasattr(geometry, 'get_constraint_function'):
            constraint_func = geometry.get_constraint_function()
            opt.add_equality_constraint(
                lambda x, grad: self._constraint_wrapper(x, grad, constraint_func, geometry), 
                1e-8
            )
            print("Added sum=1 equality constraint")
        
        # Set convergence criteria
        opt.set_ftol_rel(self.ftol_rel)
        opt.set_xtol_rel(self.xtol_rel)
        opt.set_maxeval(self.max_iter)
        
        # Store optimization problem reference
        self.opt_problem = optimization_problem
        self.iteration = 0
        
        # Run optimization
        try:
            print("\n" + "="*60)
            print("STARTING NLOPT OPTIMIZATION")
            print("="*60)
            
            optimal_params = opt.optimize(initial_params)
            optimal_fom = opt.last_optimum_value()
            result_code = opt.last_optimize_result()
            
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETED")
            print("="*60)
            print(f"Result: {result_code}")
            print(f"Optimal FOM: {optimal_fom:.6f}")
            print(f"Optimal parameters: {optimal_params}")
            print(f"Parameter sum: {np.sum(optimal_params):.6f}")
            print(f"Total iterations: {self.iteration}")
            
            return optimal_params, optimal_fom
            
        except Exception as e:
            print(f"\nOptimization failed: {e}")
            print(f"Last parameters: {self.param_history[-1] if self.param_history else 'None'}")
            print(f"Last FOM: {self.fom_history[-1] if self.fom_history else 'None'}")
            raise

    def _objective_function(self, params, grad):
        """
        NLopt objective function wrapper.
        
        Parameters:
        -----------
        params : np.ndarray
            Current parameter values (fractional widths)
        grad : np.ndarray
            Gradient array to be filled (modified in-place)
        
        Returns:
        --------
        fom : float
            Figure of merit value
        """
        self.iteration += 1
        
        # Ensure parameters satisfy constraint (normalize)
        params_normalized = params / np.sum(params)
        
        try:
            # Update geometry with current parameters
            geometry = self.opt_problem.geometry
            geometry.update_geometry(params_normalized)
            
            # Run forward simulation
            print(f"\nIteration {self.iteration}")
            print(f"Parameters: {params_normalized}")
            print(f"Running forward simulation...")
            
            self.opt_problem.run_forward_solves()
            
            # Calculate figure of merit
            current_fom = self.opt_problem.calculate_fom()
            print(f"FOM: {current_fom:.6f}")
            
            # Calculate gradients if requested
            if grad.size > 0:
                print("Calculating gradients...")
                
                # Run adjoint simulation
                self.opt_problem.run_adjoint_solves()
                
                # Use custom anisotropic gradient calculation
                if hasattr(geometry, 'calculate_gradients_manual'):
                    gradients = geometry.calculate_gradients_manual(
                        self.opt_problem.forward_fields,
                        self.opt_problem.adjoint_fields, 
                        self.opt_problem.wavelengths
                    )
                else:
                    # Fallback to standard gradient calculation
                    gradients = self.opt_problem.calculate_gradients()
                
                # Critical: Modify grad array in-place (NLopt requirement)
                grad[:] = gradients
                
                print(f"Gradients: {gradients}")
                print(f"Gradient norm: {np.linalg.norm(gradients):.4e}")
            
            # Store history
            self.fom_history.append(current_fom)
            self.param_history.append(params_normalized.copy())
            
            return current_fom
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            print(f"Parameters causing error: {params}")
            
            # Return poor FOM to continue optimization
            if grad.size > 0:
                grad[:] = 0.0
            return -1e6  # Large negative value for maximization

    def _constraint_wrapper(self, params, grad, constraint_func, geometry):
        """
        Wrapper for constraint function compatible with NLopt.
        
        For sum=1 constraint: constraint(params) = sum(params) - 1 = 0
        """
        constraint_value = constraint_func(params)
        
        if grad.size > 0:
            # Constraint jacobian: d(sum(params))/dp_i = 1
            if hasattr(geometry, 'get_constraint_jacobian'):
                constraint_jac = geometry.get_constraint_jacobian()
                grad[:] = constraint_jac(params)
            else:
                # Default: derivative of sum is 1 for all parameters
                grad[:] = 1.0
        
        return constraint_value

    def plot_convergence(self, ax=None):
        """Plot optimization convergence history"""
        if ax is None:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        else:
            ax1, ax2 = ax, None
        
        # Plot FOM history
        iterations = range(1, len(self.fom_history) + 1)
        ax1.plot(iterations, self.fom_history, 'b-', marker='o', markersize=4)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Figure of Merit')
        ax1.set_title(f'Optimization Convergence ({self.algorithm_name})')
        ax1.grid(True, alpha=0.3)
        
        if ax2 is not None and self.param_history:
            # Plot parameter evolution
            param_array = np.array(self.param_history)
            for i in range(param_array.shape[1]):
                ax2.plot(iterations, param_array[:, i], label=f'Param {i}')
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Parameter Value')
            ax2.set_title('Parameter Evolution')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        return ax1 if ax2 is None else (ax1, ax2)

# Modified optimization.py integration
class OptimizationProblem:
    """Modified to use NLopt by default"""
    
    def __init__(self, geometry, fom, base_script, optimizer=None):
        self.geometry = geometry
        self.fom = fom
        self.base_script = base_script
        
        # Use NLopt with LD_MMA as default
        if optimizer is None:
            self.optimizer = NLoptOptimizer(algorithm='LD_MMA')
        else:
            self.optimizer = optimizer
    
    def run(self):
        """Run the optimization using NLopt"""
        return self.optimizer.optimize(self)

# Usage example for edge coupler with anisotropic materials
def create_edge_coupler_optimization():
    """
    Example: Edge coupler optimization with anisotropic materials
    """
    
    # Define anisotropic materials for edge coupler
    silicon_aniso = {
        'xx': 12.0,    # In-plane silicon permittivity
        'yy': 12.0,    # In-plane silicon permittivity  
        'zz': 11.8     # Out-of-plane silicon permittivity
    }
    
    air = {'xx': 1.0, 'yy': 1.0, 'zz': 1.0}
    
    # Create rectangle clustering geometry
    from lumopt.geometries.topology import RectangleClusteringTopology
    
    geometry = RectangleClusteringTopology(
        min_feature_size=100e-9,    # 100nm minimum feature
        eps_min=air,                # Background material
        eps_max=silicon_aniso,      # Fill material (anisotropic)
        x=np.linspace(-2e-6, 8e-6, 100),   # Edge coupler length
        y=np.linspace(-1e-6, 1e-6, 50),    # Width
        z=np.array([0])             # Single layer for now
    )
    
    print(f"Edge coupler geometry created:")
    print(f"  Number of rectangles: {geometry.num_params}")
    print(f"  Design region: 10×2 μm") 
    print(f"  Min feature size: 100nm")
    
    # Create optimization problem
    # (base_script and fom would be defined separately)
    # problem = OptimizationProblem(geometry, fom, base_script)
    # optimal_params, optimal_fom = problem.run()
    
    return geometry

if __name__ == "__main__":
    # Test the optimization framework
    geometry = create_edge_coupler_optimization()
    geometry.plot()
