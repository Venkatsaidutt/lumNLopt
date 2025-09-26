# lumNLopt/Gradient_wrapper.py
"""
Gradient Wrapper for NLopt Constraint Compliance.

This module prepares the raw gradient calculated from the adjoint simulation
for use with the NLopt optimizer. For parameter sets with a sum-to-one
constraint (like the fractional widths in rectangle clustering), the raw
gradient must be projected to ensure that the optimization step does not
violate the constraint.

This is achieved by subtracting the mean of the gradient from each of its
components, effectively projecting it onto the hyperplane defined by sum(p) = 1.
"""

import numpy as np
from typing import Union

def project_gradient_for_sum_constraint(
    gradient: np.ndarray,
    parameters: np.ndarray
) -> np.ndarray:
    """
    Projects the gradient onto the constraint manifold where sum(parameters) = 1.

    When the optimization parameters (p_i) must sum to a constant (e.g., 1),
    any valid update step (dp_i) must also sum to zero. The gradient descent
    step is proportional to the gradient (g_i), so the projected gradient (g_proj)
    must be used to ensure the constraint is respected.

    The projection is calculated as: g_proj = g - mean(g).

    Args:
        gradient: The raw gradient array from the adjoint calculation.
        parameters: The current parameter array (used for validation).

    Returns:
        The projected gradient array, ready for use by the optimizer.
    """
    if not isinstance(gradient, np.ndarray) or not isinstance(parameters, np.ndarray):
        raise TypeError("Gradient and parameters must be NumPy arrays.")

    if gradient.size != parameters.size:
        raise ValueError(
            f"Gradient size ({gradient.size}) must match parameter size ({parameters.size})."
        )

    # Project the gradient by subtracting its mean. This ensures that the sum
    # of the components of the resulting vector is zero, making it a valid
    # direction for an update step on the sum-to-one constraint surface.
    gradient_mean = np.mean(gradient)
    projected_gradient = gradient - gradient_mean

    print(f"Gradient projected for sum-to-one constraint (mean subtracted: {gradient_mean:.6e})")

    return projected_gradient
