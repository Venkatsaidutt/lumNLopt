# Inverse Design using Lumerical with NLopt

This repository provides a Python-based wrapper for performing inverse design of photonic components using Lumerical's FDTD solver. The Core intent is to enable Dispersion and anisotropy into inverse design lumerical. There is a decent progress so far Current looking into syntax issues and lots of variable naming issues.

I have written it partly by myself and partly using AI, Some references that helped while building this 

https://github.com/chriskeraly/lumopt

https://github.com/penn-qel/lumopt-mbso

https://github.com/doddgray/OptiMode.jl

.
## Optimization Workflow

The optimization process is managed by a modular, four-part system that clearly separates concerns, making the workflow robust and easy to understand. The core logic is orchestrated by `optimization_loop.py`.

Here is a step-by-step breakdown of a single optimization iteration:

1.  **Parameter and Geometry Update (`Parameter_Update.py`)**
    The optimizer proposes a new set of raw design parameters. These are passed to the `GeometryParameterHandler`, which applies morphological filtering to enforce minimum feature sizes (ensuring manufacturability) and then updates the device structure in the Lumerical simulation.

2.  **Forward Simulation (`Forward_simulation.py`)**
    A standard FDTD simulation is run with the updated geometry to solve for the electromagnetic fields ($E, H$) produced by the source.

3.  **Figure of Merit Calculation (`FOM.py`)**
    The performance of the current design is evaluated by calculating a scalar Figure of Merit (FOM) from the forward fields. This value quantifies how well the device meets the design objectives (e.g., maximizing transmission).

4.  **Adjoint Pass for Gradient Calculation**
    This is the core of the gradient-based method and involves three sub-steps:
    * **Adjoint Source Creation (`Adjoint_source_creation.py`)**: The user-defined FOM function is automatically differentiated with respect to the forward fields ($∂FOM/∂E$) using XAD to generate the adjoint source.
    * **Adjoint Simulation (`Adjoint_simulation.py`)**: A second FDTD simulation is run, but this time using the adjoint source. This simulation calculates the adjoint fields ($E_{adj}, H_{adj}$).
    * **Gradient Calculation (`Gradient_Calculation.py`)**: The forward fields and adjoint fields are combined in an overlap integral to efficiently calculate the raw gradient of the FOM with respect to all design parameters. This method supports complex anisotropic materials.

5.  **Gradient Processing (`Gradient_wrapper.py`)**
    The raw gradient is projected to ensure it complies with any optimization constraints. For the rectangle clustering geometry, this involves ensuring the gradient step respects the `sum(parameters) = 1` constraint.

6.  **Optimization Step (`optimization_loop.py`)**
    The processed gradient and the current FOM are passed to the NLopt optimizer, which calculates the next set of design parameters.

7.  **Dashboard Update (`Dashboard_and_plots.py`)**
    At the end of the iteration, key metrics are saved. The FOM vs. iteration plot is updated, and an image of the new device structure is saved to disk, providing a complete record of the optimization progress.

This loop repeats until the FOM converges or the maximum number of iterations is reached.


## Acknowledgements

This work is built upon the excellent foundations provided by several open-source projects and commercial software packages. I gratefully acknowledge the developers and communities **lumerical**, **NLopt**, **XAD**, **numpy**, **matplotlib**, and others. **Thank you Sweet People**
 
