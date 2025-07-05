"""JAX-based ALTRO trajectory optimization package.

This package provides a JAX implementation of the ALTRO (Augmented Lagrangian
Trajectory Optimizer) algorithm that maintains identical computational behavior
to the original C++ implementation while leveraging JAX for GPU acceleration
and automatic differentiation.
"""

from __future__ import annotations

import jax

# Core solver interface
from .altro_solver import ALTROSolver, ConstraintIndex

# Utility functions
from .cones import (
    conic_projection,
    conic_projection_hessian,
    conic_projection_is_linear,
    conic_projection_jacobian,
    dual_cone,
)
from .cubic_spline import (
    CubicSpline,
    CubicSplineReturnCode,
    cubic_spline_argmin,
    cubic_spline_from_2_points,
    cubic_spline_from_3_points,
    quadratic_spline_from_2_points,
)

# Exception hierarchy
from .exceptions import (
    AltroException,
    ConstraintError,
    DimensionError,
    InitializationError,
    OptimizationError,
)
from .line_search import CubicLineSearch, LineSearchReturnCode

# Configuration classes
from .solver_options import AltroOptions
from .solver_stats import AltroStats
from .tvlqr import TVLQR_SUCCESS, tvlqr_backward_pass, tvlqr_forward_pass, tvlqr_total_mem_size

# Type definitions
from .types import (
    ALL_INDICES,
    # Constants
    LAST_INDEX,
    CallbackFunction,
    ConstraintFunction,
    ConstraintJacobian,
    ConstraintType,
    ControlInput,
    CostFunction,
    CostGradient,
    CostHessian,
    DualVariable,
    ErrorCode,
    # Function types
    ExplicitDynamicsFunction,
    ExplicitDynamicsJacobian,
    # Scalar types
    Float,
    HessianMatrix,
    ImplicitDynamicsFunction,
    ImplicitDynamicsJacobian,
    JacobianMatrix,
    # Enums
    SolveStatus,
    # Array types
    StateVector,
    Time,
    Verbosity,
)


# Version information
__version__ = "1.0.0"
__author__ = "JAX ALTRO Development Team"
__email__ = "altro@example.com"
__license__ = "MIT"

# Public API
__all__ = [
    "ALL_INDICES",
    "LAST_INDEX",
    "TVLQR_SUCCESS",
    # Main solver interface
    "ALTROSolver",
    # Exceptions
    "AltroException",
    # Configuration
    "AltroOptions",
    "AltroStats",
    "CallbackFunction",
    "ConstraintError",
    "ConstraintFunction",
    "ConstraintIndex",
    "ConstraintJacobian",
    "ConstraintType",
    "ControlInput",
    "CostFunction",
    "CostGradient",
    "CostGradient",
    "CostHessian",
    "CubicLineSearch",
    "CubicSpline",
    "CubicSplineReturnCode",
    "DimensionError",
    "DualVariable",
    "ErrorCode",
    "ExplicitDynamicsFunction",
    "ExplicitDynamicsJacobian",
    "Float",
    "HessianMatrix",
    "ImplicitDynamicsFunction",
    "ImplicitDynamicsJacobian",
    "InitializationError",
    "JacobianMatrix",
    "LineSearchReturnCode",
    "OptimizationError",
    "SolveStatus",
    # Type definitions
    "StateVector",
    "Time",
    "Verbosity",
    "__author__",
    "__email__",
    "__license__",
    # Version info
    "__version__",
    # Utilities
    "conic_projection",
    "conic_projection_hessian",
    "conic_projection_is_linear",
    "conic_projection_jacobian",
    "cubic_spline_argmin",
    "cubic_spline_from_2_points",
    "cubic_spline_from_3_points",
    "dual_cone",
    "quadratic_spline_from_2_points",
    "tvlqr_backward_pass",
    "tvlqr_forward_pass",
    "tvlqr_total_mem_size",
]

# Package documentation
__doc__ = """
JAX-based ALTRO (Augmented Lagrangian Trajectory Optimizer)

This package provides a high-performance implementation of the ALTRO algorithm
using JAX for automatic differentiation and GPU acceleration. The implementation
maintains identical computational behavior to the original C++ version while
leveraging JAX's functional programming paradigm and JIT compilation.

Key Features:
- JAX-native implementation with GPU acceleration
- Automatic differentiation for gradients and Jacobians
- Functional programming paradigm with immutable data structures
- JIT compilation for maximum performance
- Thread-safe and vectorizable operations
- Identical numerical behavior to C++ implementation

Basic Usage:
    import jax.numpy as jnp
    import altro

    # Create solver
    solver = altro.ALTROSolver(horizon_length=50)

    # Set problem dimensions
    solver.set_dimension(num_states=4, num_inputs=2)

    # Set dynamics and cost functions
    solver.set_explicit_dynamics(dynamics_func, dynamics_jac)
    solver.set_lqr_cost(num_states=4, num_inputs=2, Q_diag, R_diag, x_ref, u_ref)

    # Initialize and solve
    solver.set_initial_state(x0, n=4)
    solver.initialize()
    status = solver.solve()

    # Extract results
    x_final = solver.get_state(solver.get_horizon_length())
    cost = solver.calc_cost()

For more detailed examples and documentation, see the examples directory
and the API documentation.
"""


def _check_jax_installation() -> None:
    """Check that JAX is properly installed and accessible."""
    try:
        import jax
        import jax.numpy as jnp

        # Test basic JAX functionality
        _ = jnp.array([1.0, 2.0, 3.0])
        _ = jax.grad(lambda x: x**2)(1.0)
    except ImportError as e:
        raise ImportError(
            "JAX is required for ALTRO but not found. "
            "Please install JAX with: pip install jax jaxlib"
        ) from e
    except Exception as e:
        raise RuntimeError(
            "JAX installation appears to be broken. "
            "Please reinstall JAX with: pip install --upgrade jax jaxlib"
        ) from e


def _check_dependencies() -> None:
    """Check that all required dependencies are available."""
    missing_deps = []

    # Check required packages
    try:
        import jax
    except ImportError:
        missing_deps.append("jax")

    try:
        import jax.numpy
    except ImportError:
        missing_deps.append("jax[numpy]")

    try:
        import jax.scipy
    except ImportError:
        missing_deps.append("jax[scipy]")

    if missing_deps:
        raise ImportError(
            f"Missing required dependencies: {', '.join(missing_deps)}. "
            "Please install with: pip install jax jaxlib"
        )


# Perform dependency checks on import
_check_dependencies()
_check_jax_installation()

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)
# Enable JIT compilation
jax.config.update("jax_disable_jit", False)
# Enable JIT compilation
jax.config.update("jax_disable_jit", False)
