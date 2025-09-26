# lumNLopt/optimization_loop.py
"""
Main Optimization Loop using NLopt.

This module orchestrates the entire inverse design optimization process. It
initializes all necessary components—simulation, geometry, FOM, and gradient
calculation—and uses the NLopt library to find the optimal device parameters.
"""

import nlopt
import numpy as np
import time

# Import all necessary components from the lumNLopt package
from lumNLopt.Forward_simulation import ForwardSimulation
from lumNLopt.FOM import FOMCalculator
from lumNLopt.Adjoint_source_creation import compute_adjoint_source_from_fom
from lumNLopt.Adjoint_simulation import run_adjoint_simulation_workflow, AdjointSourceData
from lumNLopt.Gradient_Calculation import calculate_lumNLopt_gradients
from lumNLopt.geometries.geometry_parameters_handling import GeometryParameterHandler
from lumNLopt.Inputs.Device import get_device_config

# Import the new, modular optimization components
from lumNLopt.Gradient_wrapper import project_gradient_for_sum_constraint
from lumNLopt.Parameter_Update import update_geometry_parameters
from lumNLopt.Dashboard_and_plots import OptimizationDashboard

class NloptOptimizationProblem:
    """A class to define and run an NLopt-based optimization problem."""

    def __init__(self, base_fsp_file: str, max_iter: int = 100):
        self.base_fsp_file = base_fsp_file
        self.max_iter = max_iter

        # Initialize core components
        self.sim = ForwardSimulation(hide_fdtd_cad=True)
        self.sim.load_base_simulation(self.base_fsp_file)
        
        self.device_config = get_device_config()
        self.geometry_handler = GeometryParameterHandler()
        self.fom_calculator = FOMCalculator()
        self.dashboard = OptimizationDashboard()

        # State tracking
        self.iteration = 0
        self.start_time = time.time()

    def _objective_function(self, params: np.ndarray, grad: np.ndarray) -> float:
        """
        The objective function that NLopt will minimize (or maximize).
        This function orchestrates a single iteration of the optimization.
        """
        # --- 1. Update Geometry ---
        # Note: We use the unfiltered `params` from the optimizer here.
        # The handler will apply filtering before updating the simulation.
        update_geometry_parameters(params, self.geometry_handler, self.sim)

        # --- 2. Run Forward Simulation and Calculate FOM ---
        print(f"\n--- Iteration {self.iteration} | Time: {time.time() - self.start_time:.2f}s ---")
        self.sim.run_simulation()
        self.sim.extract_fields()
        fom_value = self.fom_calculator.calculate_fom(self.sim)

        # --- 3. Calculate Gradient (if required by optimizer) ---
        if grad.size > 0:
            forward_fields = self.sim.forward_fields['opt_fields']

            # a. Create Adjoint Source
            E_source, H_source = compute_adjoint_source_from_fom(
                self.fom_calculator.fom_object.get_fom, # Pass the callable fom
                E=forward_fields.E, H=forward_fields.H,
                x=forward_fields.x, y=forward_fields.y, z=forward_fields.z,
                wavelengths=forward_fields.wl
            )
            adjoint_data = AdjointSourceData(Esource=E_source, Hsource=H_source,
                                             x=forward_fields.x, y=forward_fields.y,
                                             z=forward_fields.z, wavelengths=forward_fields.wl,
                                             monitor_name='opt_fields')

            # b. Run Adjoint Simulation
            adjoint_fields = run_adjoint_simulation_workflow(self.sim, adjoint_data)

            # c. Calculate Raw Gradient
            raw_gradient = calculate_lumNLopt_gradients(forward_fields, adjoint_fields,
                                                        self.geometry_handler.geometry_cluster,
                                                        self.device_config)

            # d. Project Gradient for Constraint
            projected_grad = project_gradient_for_sum_constraint(raw_gradient, params)
            grad[:] = projected_grad # Update gradient in-place for NLopt

        # --- 4. Update Dashboard and Increment Iteration Counter ---
        self.dashboard.update(self.iteration, fom_value, self.geometry_handler)
        self.iteration += 1
        
        # NLopt maximizers return the value. We are maximizing the FOM.
        return fom_value

    def run(self):
        """Initializes and runs the NLopt optimization."""
        initial_params = self.geometry_handler.get_current_params()
        num_params = len(initial_params)
        bounds = self.geometry_handler.get_bounds()
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])

        # Initialize NLopt optimizer (LD_MMA is excellent for this)
        opt = nlopt.opt(nlopt.LD_MMA, num_params)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_max_objective(self._objective_function)
        opt.set_maxeval(self.max_iter)

        # Add the sum-to-one equality constraint
        def sum_constraint(p, grad):
            if grad.size > 0:
                grad[:] = 1.0
            return np.sum(p) - 1.0
        
        opt.add_equality_constraint(sum_constraint, 1e-8)

        print("=== Starting NLopt Optimization ===")
        print(f"Algorithm: LD_MMA, Max Iterations: {self.max_iter}")
        
        final_params = opt.optimize(initial_params)
        final_fom = opt.last_optimum_value()
        
        print("\n=== Optimization Finished ===")
        print(f"Final FOM: {final_fom:.6f}")
        print(f"Optimal Parameters saved in '{self.dashboard.output_path}'")
        
        return final_fom, final_params

if __name__ == '__main__':
    # This is how you would run the optimization
    # You would need a base .fsp file (e.g., 'base_coupler.fsp')
    
    # problem = NloptOptimizationProblem(base_fsp_file='base_coupler.fsp', max_iter=50)
    # final_fom, final_params = problem.run()
    pass
