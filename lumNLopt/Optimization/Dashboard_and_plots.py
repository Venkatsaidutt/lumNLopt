# lumNLopt/Dashboard_and_plots.py
"""
Optimization Dashboard and Progress Visualization.

This module handles the saving of optimization progress and the generation
of visualizations. It is designed to run with a suppressed graphical interface,
making it suitable for background or server-based execution.

Responsibilities:
- Create and manage a directory for optimization results.
- Save the raw (pre-filtering) and filtered (post-filtering) parameters
  at each iteration.
- Generate and save a plot of the Figure of Merit (FOM) vs. iteration.
- Generate and save an image of the device structure at each iteration.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from lumNLopt.geometries.geometry_parameters_handling import GeometryParameterHandler

class OptimizationDashboard:
    """Manages the output and visualization of the optimization process."""

    def __init__(self, output_dir: str = 'optimization_results'):
        """
        Initializes the dashboard and creates the necessary output directory.

        Args:
            output_dir: The name of the directory to save results to.
        """
        self.output_path = Path(output_dir).resolve()
        self.params_raw_path = self.output_path / "parameters_raw"
        self.params_filtered_path = self.output_path / "parameters_filtered"
        self.structures_path = self.output_path / "structures"

        # Create directories
        self.output_path.mkdir(exist_ok=True)
        self.params_raw_path.mkdir(exist_ok=True)
        self.params_filtered_path.mkdir(exist_ok=True)
        self.structures_path.mkdir(exist_ok=True)

        self.fom_history = []
        print(f"Dashboard initialized. Results will be saved in '{self.output_path}'")

    def update(
        self,
        iteration: int,
        fom_value: float,
        geometry_handler: GeometryParameterHandler
    ) -> None:
        """
        Updates the dashboard at the end of an optimization iteration.

        Args:
            iteration: The current iteration number (starting from 0).
            fom_value: The Figure of Merit calculated in this iteration.
            geometry_handler: The geometry handler containing the current
                              parameter and structure information.
        """
        print(f"Dashboard update for iteration {iteration}...")
        self.fom_history.append(fom_value)

        # 1. Save raw and filtered parameters as .npy files
        raw_params = geometry_handler.get_current_params_raw()
        filtered_params = geometry_handler.get_current_params_filtered()
        np.save(self.params_raw_path / f"params_raw_{iteration:04d}.npy", raw_params)
        np.save(self.params_filtered_path / f"params_filtered_{iteration:04d}.npy", filtered_params)

        # 2. Update and save the FOM plot
        self._plot_fom()

        # 3. Update and save the structure image
        self._plot_structure(iteration, geometry_handler)
        
        print("Dashboard update complete.")

    def _plot_fom(self) -> None:
        """Generates and saves the FOM vs. Iteration plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.fom_history)), self.fom_history, 'o-', color='b')
        plt.title('Figure of Merit (FOM) vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Figure of Merit')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_path / "fom_vs_iteration.png")
        plt.close()

    def _plot_structure(self, iteration: int, geometry_handler: GeometryParameterHandler) -> None:
        """Generates and saves an image of the current device structure."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # The plot_rectangle_configuration method is in Geometry_clustered.py,
        # accessed via the geometry_handler
        geometry_handler.plot(ax=ax)
        
        ax.set_title(f'Device Structure - Iteration {iteration}')
        plt.tight_layout()
        plt.savefig(self.structures_path / f"structure_{iteration:04d}.png")
        plt.close(fig)
