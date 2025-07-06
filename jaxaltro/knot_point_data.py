"""Knot point data structure for JAX-based ALTRO trajectory optimization.

This module provides the knot point data structure that directly corresponds to the C++
KnotPointData class, maintaining identical data organization and computational methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import Array

from .cones import (
    conic_projection,
    conic_projection_hessian,
    conic_projection_is_linear,
    conic_projection_jacobian,
    dual_cone,
)
from .exceptions import ErrorCode, _altro_throw
from .types import (
    ConstraintFunction,
    ConstraintType,
    CostFunction,
    ExplicitDynamicsFunction,
    Float,
    _ConstraintJacobian,
    _CostGradient,
    _CostHessian,
    _ExplicitDynamicsJacobian,
)


# Maximum number of constraints matching C++ kMaxConstraints
MAX_CONSTRAINTS = 2**31 - 1

# Global compiled functions cache to avoid recompilation
_compiled_functions_cache: dict[str, Any] = {}


@jax.jit
def _constraint_cost_jit(projected_dual: Array, penalty_param: Float) -> Array:
    """JIT-compiled constraint cost contribution."""
    return jnp.sum(projected_dual**2) / (2 * penalty_param)


@jax.jit
def _dual_estimation_jit(dual_var: Array, penalty_param: Float, constraint_val: Array) -> Array:
    """JIT-compiled dual variable estimation."""
    return dual_var - penalty_param * constraint_val


@jax.jit
def _violation_computation_jit(projected: Array, constraint_val: Array) -> Array:
    """JIT-compiled violation computation."""
    violation = projected - constraint_val
    return jnp.max(jnp.abs(violation))


@jax.jit
def _transpose_matrix_vector_product_jit(matrix: Array, vector: Array) -> Array:
    """JIT-compiled transpose matrix-vector product."""
    return matrix.T @ vector


def _get_or_create_dynamics_jacobian(
    dynamics_function: ExplicitDynamicsFunction,
) -> _ExplicitDynamicsJacobian:
    """Get or create cached dynamics Jacobian function."""
    func_id = id(dynamics_function)
    cache_key = f"dynamics_jac_{func_id}"

    if cache_key not in _compiled_functions_cache:

        @jax.jit
        def auto_dynamics_jacobian(x: Array, u: Array, h: Float) -> Array:
            n, _m = len(x), len(u)

            # Combined input for Jacobian computation
            def dynamics_combined(xu):
                x_part, u_part = xu[:n], xu[n:]
                return dynamics_function(x_part, u_part, h)

            # Automatic Jacobian computation
            jac_combined = jax.jacobian(dynamics_combined)
            xu_combined = jnp.concatenate([x, u])
            return cast(Array, jac_combined(xu_combined))

        _compiled_functions_cache[cache_key] = auto_dynamics_jacobian

    return cast(_ExplicitDynamicsJacobian, _compiled_functions_cache[cache_key])


def _get_or_create_cost_derivatives(
    cost_function: CostFunction,
) -> tuple[_CostGradient, _CostHessian]:
    """Get or create cached cost derivative functions."""
    func_id = id(cost_function)
    cache_key = f"cost_derivs_{func_id}"

    if cache_key not in _compiled_functions_cache:
        # Create automatic gradient function
        @jax.jit
        def auto_cost_gradient(x: Array, u: Array) -> tuple[Array, Array]:
            grad_x = jax.grad(cost_function, argnums=0)(x, u)
            if len(u) > 0:  # Non-terminal
                grad_u = jax.grad(cost_function, argnums=1)(x, u)
            else:  # Terminal
                grad_u = jnp.array([])
            return grad_x, grad_u

        # Create automatic Hessian function
        @jax.jit
        def auto_cost_hessian(x: Array, u: Array) -> tuple[Array, Array, Array]:
            hess_xx = jax.hessian(cost_function, argnums=0)(x, u)
            if len(u) > 0:  # Non-terminal
                hess_uu = jax.hessian(cost_function, argnums=1)(x, u)
                # Mixed Hessian: ∂²J/∂u∂x - Jacobian of gradient w.r.t. u taken w.r.t. x
                grad_u_func = jax.grad(cost_function, argnums=1)
                hess_ux = jax.jacobian(grad_u_func, argnums=0)(x, u)
            else:  # Terminal
                hess_uu = jnp.array([]).reshape(0, 0)
                hess_ux = jnp.array([]).reshape(0, len(x))
            return hess_xx, hess_uu, hess_ux

        _compiled_functions_cache[cache_key] = (auto_cost_gradient, auto_cost_hessian)

    return cast(tuple[_CostGradient, _CostHessian], _compiled_functions_cache[cache_key])


def _get_or_create_constraint_jacobian(
    constraint_function: ConstraintFunction,
) -> _ConstraintJacobian:
    """Get or create cached constraint Jacobian function."""
    func_id = id(constraint_function)
    cache_key = f"constraint_jac_{func_id}"

    if cache_key not in _compiled_functions_cache:

        @jax.jit
        def auto_constraint_jacobian(x: Array, u: Array) -> Array:
            n, _m = len(x), len(u)

            def constraint_combined(xu):
                x_part, u_part = xu[:n], xu[n:]
                return constraint_function(x_part, u_part)

            jac_fn = jax.jacobian(constraint_combined)
            xu_combined = jnp.concatenate([x, u])
            return cast(Array, jac_fn(xu_combined))

        _compiled_functions_cache[cache_key] = auto_constraint_jacobian

    return cast(_ConstraintJacobian, _compiled_functions_cache[cache_key])


class CostFunctionType(Enum):
    """Cost function types matching C++ CostFunType enum."""

    GENERIC = "Generic"
    QUADRATIC = "Quadratic"
    DIAGONAL = "Diagonal"


@dataclass
class KnotPointData:
    """Knot point data structure matching C++ KnotPointData class.

    Contains all data associated with a single time step in the trajectory optimization problem.
    """

    # Basic properties
    knot_point_index: int
    is_terminal: bool

    # Dimensions
    num_states: int = 0
    num_inputs: int = 0
    num_next_states: int = 0
    timestep: Float = 0.0

    # Cost function data
    cost_function_type: CostFunctionType = CostFunctionType.GENERIC
    cost_function: CostFunction | None = None
    cost_gradient: _CostGradient | None = None
    cost_hessian: _CostHessian | None = None

    # Quadratic cost matrices
    Q: Array | None = None  # State cost matrix (vectorized for diagonal)
    R: Array | None = None  # Control cost matrix (vectorized for diagonal)
    H: Array | None = None  # Cross-term cost matrix
    q: Array | None = None  # Linear state cost
    r: Array | None = None  # Linear control cost
    c: Float = 0.0  # Constant cost term

    # Dynamics data
    dynamics_are_linear: bool = False
    dynamics_function: ExplicitDynamicsFunction | None = None
    dynamics_jacobian: _ExplicitDynamicsJacobian | None = None
    affine_term: Array | None = None

    # Constraint data
    constraint_functions: list[ConstraintFunction] = field(default_factory=list)
    constraint_jacobians: list[_ConstraintJacobian] = field(default_factory=list)
    constraint_dims: list[int] = field(default_factory=list)
    constraint_types: list[ConstraintType] = field(default_factory=list)
    constraint_labels: list[str] = field(default_factory=list)

    # Bound constraints
    x_upper_bounds: Array | None = None
    x_lower_bounds: Array | None = None
    u_upper_bounds: Array | None = None
    u_lower_bounds: Array | None = None

    # State flags
    is_initialized: bool = False
    cost_function_is_set: bool = False
    dynamics_is_set: bool = False

    # Trajectory data (public access matching C++ implementation)
    x: Array | None = None  # Reference state
    u: Array | None = None  # Reference control
    y: Array | None = None  # Reference dual

    x_: Array | None = None  # Current state
    u_: Array | None = None  # Current control
    y_: Array | None = None  # Current dual

    # Dynamics expansion data
    A_: Array | None = None  # State Jacobian
    B_: Array | None = None  # Control Jacobian
    f_: Array | None = None  # Affine term

    # Cost expansion data
    lxx_: Array | None = None  # State Hessian
    luu_: Array | None = None  # Control Hessian
    lux_: Array | None = None  # Cross Hessian
    lx_: Array | None = None  # State gradient
    lu_: Array | None = None  # Control gradient

    # Backward pass data
    Qxx_: Array | None = None
    Quu_: Array | None = None
    Qux_: Array | None = None
    Qx_: Array | None = None
    Qu_: Array | None = None

    K_: Array | None = None  # Feedback gain
    d_: Array | None = None  # Feedforward term
    P_: Array | None = None  # Cost-to-go matrix
    p_: Array | None = None  # Cost-to-go vector

    # Forward pass data
    dx_da_: Array | None = None  # State gradient wrt step size
    du_da_: Array | None = None  # Control gradient wrt step size

    # Constraint data
    constraint_vals: list[Array] = field(default_factory=list)
    constraint_jacs: list[Array] = field(default_factory=list)
    constraint_hessians: list[Array] = field(default_factory=list)
    dual_variables: list[Array] = field(default_factory=list)
    projected_duals: list[Array] = field(default_factory=list)
    penalty_parameters: list[Float] = field(default_factory=list)

    def _assert_initialized(self) -> None:
        """Assert knot point is initialized for computational operations."""
        if not self.is_initialized:
            _altro_throw(
                f"Knot point {self.knot_point_index} not initialized",
                ErrorCode.SOLVER_NOT_INITIALIZED,
            )

    def _assert_arrays_non_none(self) -> None:
        """Assert computational arrays are non-None after initialization."""
        self._assert_initialized()

        # These assertions inform the type checker that arrays are non-None
        assert self.x_ is not None
        assert self.u_ is not None
        assert self.y_ is not None
        assert self.lx_ is not None
        assert self.lxx_ is not None

        if not self.is_terminal:
            assert self.lu_ is not None
            assert self.luu_ is not None
            assert self.lux_ is not None

    def set_dimension(self, num_states: int, num_inputs: int) -> None:
        """Set state and control dimensions matching C++ SetDimension."""
        if num_states <= 0:
            _altro_throw(
                f"State dimension must be specified at index {self.knot_point_index}",
                ErrorCode.STATE_DIM_UNKNOWN,
            )

        if num_inputs <= 0:
            msg = "Input dimension must be specified"
            if self.is_terminal:
                msg += " at the terminal knot point"
            else:
                msg += f" at index {self.knot_point_index}"
            _altro_throw(msg, ErrorCode.INPUT_DIM_UNKNOWN)

        self.num_states = num_states
        self.num_inputs = num_inputs

    def set_next_state_dimension(self, num_next_states: int) -> None:
        """Set next state dimension matching C++ SetNextStateDimension."""
        if self.is_terminal:
            _altro_throw(
                "Cannot set next state dimension at terminal knot point",
                ErrorCode.INVALID_OPT_AT_TERMINAL_KNOT_POINT,
            )

        if num_next_states <= 0:
            _altro_throw("Next state dimension must be positive", ErrorCode.NEXT_STATE_DIM_UNKNOWN)

        self.num_next_states = num_next_states

    def set_timestep(self, timestep: Float) -> None:
        """Set timestep matching C++ SetTimestep."""
        if self.is_terminal:
            _altro_throw(
                "Cannot set timestep at terminal knot point",
                ErrorCode.INVALID_OPT_AT_TERMINAL_KNOT_POINT,
            )

        if timestep <= 0.0:
            _altro_throw("Timestep must be positive", ErrorCode.TIMESTEP_NOT_POSITIVE)

        self.timestep = timestep

    def set_quadratic_cost(
        self,
        n: int,
        m: int,
        Q_mat: Array,
        R_mat: Array,
        H_mat: Array,
        q_vec: Array,
        r_vec: Array,
        c: Float,
    ) -> None:
        """Set quadratic cost function matching C++ SetQuadraticCost."""
        if self.num_states > 0 and self.num_states != n:
            _altro_throw("State dimension mismatch", ErrorCode.DIMENSION_MISMATCH)
        if self.num_inputs > 0 and self.num_inputs != m:
            _altro_throw("Input dimension mismatch", ErrorCode.DIMENSION_MISMATCH)

        self.Q = Q_mat.flatten()  # Store as vector for consistency with C++
        self.R = R_mat.flatten()
        self.H = H_mat
        self.q = q_vec
        self.r = r_vec
        self.c = c

        self.cost_function_type = CostFunctionType.QUADRATIC
        self.cost_function_is_set = True

    def set_diagonal_cost(
        self, n: int, m: int, Q_diag: Array, R_diag: Array, q_vec: Array, r_vec: Array, c: Float
    ) -> None:
        """Set diagonal cost function matching C++ SetDiagonalCost."""
        if self.num_states > 0 and self.num_states != n:
            _altro_throw("State dimension mismatch", ErrorCode.DIMENSION_MISMATCH)
        if self.num_inputs > 0 and self.num_inputs != m:
            _altro_throw("Input dimension mismatch", ErrorCode.DIMENSION_MISMATCH)

        # Store diagonal elements in first n elements (matching C++ implementation)
        self.Q = jnp.zeros(n * n)
        self.Q = self.Q.at[:n].set(Q_diag)

        self.H = jnp.zeros((m, n))
        self.q = q_vec
        self.c = c

        if not self.is_terminal:
            self.R = jnp.zeros(m * m)
            self.R = self.R.at[:m].set(R_diag)
            self.r = r_vec

        self.cost_function_type = CostFunctionType.DIAGONAL
        self.cost_function_is_set = True

    def set_cost_function(self, cost_function: CostFunction) -> None:
        """Set generic cost function with cached automatic gradient/Hessian computation."""
        self.cost_function = cost_function

        # Get cached derivative functions (compiled once, reused across knot points)
        self.cost_gradient, self.cost_hessian = _get_or_create_cost_derivatives(cost_function)

        self.cost_function_type = CostFunctionType.GENERIC
        self.cost_function_is_set = True

    def set_linear_dynamics(
        self, n2: int, n: int, m: int, A: Array, B: Array, f: Array | None = None
    ) -> None:
        """Set linear dynamics matching C++ SetLinearDynamics."""
        if self.is_terminal:
            _altro_throw(
                "Cannot set dynamics at terminal knot point",
                ErrorCode.INVALID_OPT_AT_TERMINAL_KNOT_POINT,
            )

        if self.num_states > 0 and n != self.num_states:
            _altro_throw("State dimension mismatch", ErrorCode.DIMENSION_MISMATCH)
        if self.num_inputs > 0 and m != self.num_inputs:
            _altro_throw("Input dimension mismatch", ErrorCode.DIMENSION_MISMATCH)
        if self.num_next_states > 0 and n2 != self.num_next_states:
            _altro_throw("Next state dimension mismatch", ErrorCode.DIMENSION_MISMATCH)

        self.num_states = n
        self.num_inputs = m
        self.num_next_states = n2

        self.A_ = A
        self.B_ = B
        if f is not None:
            self.affine_term = f
        else:
            self.affine_term = jnp.zeros(n2)

        self.dynamics_are_linear = True
        self.dynamics_is_set = True

    def set_dynamics(self, dynamics_function: ExplicitDynamicsFunction) -> None:
        """Set nonlinear dynamics with cached automatic Jacobian computation."""
        if self.is_terminal:
            _altro_throw(
                "Cannot set dynamics at terminal knot point",
                ErrorCode.INVALID_OPT_AT_TERMINAL_KNOT_POINT,
            )

        self.dynamics_function = dynamics_function

        # Get cached Jacobian function (compiled once, reused across knot points)
        self.dynamics_jacobian = _get_or_create_dynamics_jacobian(dynamics_function)

        self.dynamics_are_linear = False
        self.dynamics_is_set = True

    def set_constraint(
        self,
        constraint_function: ConstraintFunction,
        dim: int,
        constraint_type: ConstraintType,
        label: str,
    ) -> None:
        """Set constraint with cached automatic Jacobian computation."""
        if len(self.constraint_functions) >= MAX_CONSTRAINTS:
            _altro_throw(
                f"Maximum number of constraints exceeded at knot point {self.knot_point_index}",
                ErrorCode.MAX_CONSTRAINTS_EXCEEDED,
            )

        if dim <= 0:
            _altro_throw(
                f"Invalid constraint dimension {dim} at knot point {self.knot_point_index}",
                ErrorCode.INVALID_CONSTRAINT_DIM,
            )

        # Get cached constraint Jacobian (compiled once, reused across knot points)
        auto_constraint_jacobian = _get_or_create_constraint_jacobian(constraint_function)

        self.constraint_functions.append(constraint_function)
        self.constraint_jacobians.append(auto_constraint_jacobian)
        self.constraint_dims.append(dim)
        self.constraint_types.append(constraint_type)
        self.constraint_labels.append(label)

    def set_penalty(self, rho: Float) -> None:
        """Set penalty parameter matching C++ SetPenalty."""
        if rho <= 0:
            _altro_throw(
                f"Non-positive penalty {rho} at index {self.knot_point_index}",
                ErrorCode.NON_POSITIVE_PENALTY,
            )

        self.penalty_parameters = [rho] * len(self.constraint_functions)

    def initialize(self) -> None:
        """Initialize knot point data matching C++ Initialize."""
        n2 = self.get_next_state_dim()
        n = self.get_state_dim()
        m = self.get_input_dim()
        h = self.get_time_step()

        # Validation
        if n <= 0:
            _altro_throw(
                f"Failed to initialize knot point {self.knot_point_index}: State dimension unknown",
                ErrorCode.STATE_DIM_UNKNOWN,
            )

        if not self.is_terminal:
            if m <= 0:
                _altro_throw(
                    f"Failed to initialize knot point {self.knot_point_index}: Input dimension unknown",
                    ErrorCode.INPUT_DIM_UNKNOWN,
                )
            if n2 <= 0:
                _altro_throw(
                    f"Failed to initialize knot point {self.knot_point_index}: Next state dimension unknown",
                    ErrorCode.NEXT_STATE_DIM_UNKNOWN,
                )
            if h <= 0:
                _altro_throw(
                    f"Failed to initialize knot point {self.knot_point_index}: Time step not set",
                    ErrorCode.TIMESTEP_NOT_POSITIVE,
                )
            if not self.dynamics_is_set:
                _altro_throw(
                    f"Failed to initialize knot point {self.knot_point_index}: Dynamics function not set",
                    ErrorCode.DYNAMICS_FUN_NOT_SET,
                )

        if not self.cost_function_is_set:
            _altro_throw(
                f"Failed to initialize knot point {self.knot_point_index}: Cost function not set",
                ErrorCode.COST_FUN_NOT_SET,
            )

        # Initialize trajectory data
        self.x = jnp.zeros(n)
        self.u = jnp.zeros(m)
        self.y = jnp.zeros(n)

        self.x_ = jnp.zeros(n)
        self.u_ = jnp.zeros(m)
        self.y_ = jnp.zeros(n)

        if not self.is_terminal:
            self.f_ = jnp.zeros(n2)
            if not self.dynamics_are_linear:
                self.A_ = jnp.zeros((n2, n))
                self.B_ = jnp.zeros((n2, m))
                self.f_ = jnp.zeros(n2)

        # Initialize bound constraints
        inf = jnp.inf
        self.x_upper_bounds = jnp.full(n, inf)
        self.x_lower_bounds = jnp.full(n, -inf)
        self.u_upper_bounds = jnp.full(m, inf)
        self.u_lower_bounds = jnp.full(m, -inf)

        # Initialize constraint data
        num_constraints = len(self.constraint_functions)
        self.constraint_vals = [jnp.zeros(dim) for dim in self.constraint_dims]
        self.constraint_jacs = [jnp.zeros((dim, n + m)) for dim in self.constraint_dims]
        self.constraint_hessians = [jnp.zeros((n + m, n + m)) for _ in range(num_constraints)]
        self.dual_variables = [jnp.zeros(dim) for dim in self.constraint_dims]
        self.projected_duals = [jnp.zeros(dim) for dim in self.constraint_dims]
        self.penalty_parameters = [1.0] * num_constraints

        # Initialize cost expansion data
        self.lxx_ = jnp.zeros((n, n))
        self.luu_ = jnp.zeros((m, m))
        self.lux_ = jnp.zeros((m, n))
        self.lx_ = jnp.zeros(n)
        self.lu_ = jnp.zeros(m)

        # Initialize backward pass data
        self.Qxx_ = jnp.zeros((n, n))
        self.Quu_ = jnp.zeros((m, m))
        self.Qux_ = jnp.zeros((m, n))
        self.Qx_ = jnp.zeros(n)
        self.Qu_ = jnp.zeros(m)

        self.K_ = jnp.zeros((m, n))
        self.d_ = jnp.zeros(m)
        self.P_ = jnp.zeros((n, n))
        self.p_ = jnp.zeros(n)

        # Initialize forward pass data
        self.dx_da_ = jnp.zeros(n)
        self.du_da_ = jnp.zeros(m)

        # Calculate constant Hessian if quadratic
        if self.cost_function_type != CostFunctionType.GENERIC:
            self._calc_original_cost_hessian()

        # Set linear cost gradients for LQ problems
        if self.is_terminal and self.cost_function_type != CostFunctionType.GENERIC:
            assert self.q is not None
            self.lx_ = self.q

        if (
            not self.is_terminal
            and self.cost_function_type != CostFunctionType.GENERIC
            and self.dynamics_are_linear
        ):
            assert self.q is not None
            assert self.r is not None
            assert self.affine_term is not None
            self.lx_ = self.q
            self.lu_ = self.r
            self.f_ = self.affine_term

        self.is_initialized = True

    def get_state_dim(self) -> int:
        """Get state dimension."""
        return self.num_states

    def get_input_dim(self) -> int:
        """Get input dimension."""
        return self.num_inputs

    def get_next_state_dim(self) -> int:
        """Get next state dimension."""
        return self.num_next_states

    def get_time_step(self) -> Float:
        """Get time step."""
        return self.timestep

    def is_terminal_knot_point(self) -> bool:
        """Check if this is the terminal knot point."""
        return self.is_terminal

    def num_constraints(self) -> int:
        """Get number of constraints."""
        return len(self.constraint_functions)

    def dynamics_are_linear_fn(self) -> bool:
        """Check if dynamics are linear."""
        return self.dynamics_are_linear

    def cost_function_is_quadratic(self) -> bool:
        """Check if cost function is quadratic."""
        return self.cost_function_type != CostFunctionType.GENERIC

    def calc_dynamics(self, x_next: Array) -> Array:
        """Calculate dynamics matching C++ CalcDynamics."""
        if self.is_terminal:
            _altro_throw(
                "Cannot calculate dynamics at terminal knot point",
                ErrorCode.INVALID_OPT_AT_TERMINAL_KNOT_POINT,
            )

        self._assert_initialized()
        assert self.x_ is not None
        assert self.u_ is not None

        if self.dynamics_are_linear:
            assert self.A_ is not None
            assert self.B_ is not None
            assert self.affine_term is not None
            return self.A_ @ self.x_ + self.B_ @ self.u_ + self.affine_term
        else:
            assert self.dynamics_function is not None
            return self.dynamics_function(self.x_, self.u_, self.timestep)

    def calc_dynamics_expansion(self) -> None:
        """Calculate dynamics expansion matching C++ CalcDynamicsExpansion."""
        if self.is_terminal:
            return

        self._assert_initialized()

        if not self.dynamics_are_linear:
            h = self.get_time_step()

            assert self.dynamics_jacobian is not None
            assert self.x_ is not None
            assert self.u_ is not None

            # Compute Jacobian using automatic differentiation
            jac = self.dynamics_jacobian(self.x_, self.u_, h)
            n = self.get_state_dim()
            m = self.get_input_dim()
            self.A_ = jac[:, :n]
            self.B_ = jac[:, n : n + m]
        else:
            # For linear dynamics, f_ is set to zero for linearization
            assert self.f_ is not None
            self.f_ = jnp.zeros_like(self.f_)

    def calc_cost(self) -> Float:
        """Calculate cost including Augmented Lagrangian terms matching C++ CalcCost."""
        # Original cost
        cost = self._calc_original_cost()

        # Add Augmented Lagrangian constraint terms
        self.calc_constraints()
        al_cost = self._calc_constraint_costs()

        return cost + al_cost

    def calc_cost_gradient(self) -> None:
        """Calculate cost gradient matching C++ CalcCostGradient."""
        # Original cost gradient
        self._calc_original_cost_gradient()

        # Add Augmented Lagrangian constraint gradient terms
        self._calc_constraint_cost_gradients()

    def calc_cost_hessian(self) -> None:
        """Calculate cost Hessian matching C++ CalcCostHessian."""
        # Original cost Hessian
        self._calc_original_cost_hessian()

        # Add Augmented Lagrangian constraint Hessian terms
        self._calc_constraint_cost_hessians()

    def calc_constraints(self) -> None:
        """Calculate constraints - functions already JIT-compiled individually."""
        if not self.constraint_functions:
            return

        assert self.x_ is not None
        assert self.u_ is not None

        # Direct evaluation - constraint functions are already JIT-compiled
        for j, constraint_func in enumerate(self.constraint_functions):
            self.constraint_vals[j] = constraint_func(self.x_, self.u_)

    def calc_constraint_jacobians(self) -> None:
        """Calculate constraint Jacobians using cached automatic differentiation."""
        if not self.constraint_jacobians:
            return

        assert self.x_ is not None
        assert self.u_ is not None

        # Cached Jacobian evaluation
        for j, constraint_jac in enumerate(self.constraint_jacobians):
            self.constraint_jacs[j] = constraint_jac(self.x_, self.u_)

    def calc_violations(self) -> Float:
        """Calculate constraint violations with JIT-compiled math operations."""
        viol = 0.0
        for j in range(len(self.constraint_functions)):
            cone = self.constraint_types[j]
            # Project constraint value onto cone
            projected = conic_projection(cone, self.constraint_vals[j])
            # JIT-compiled violation computation
            viol_j = _violation_computation_jit(projected, self.constraint_vals[j])
            viol = max(viol, float(viol_j))
        return viol

    def dual_update(self) -> None:
        """Update dual variables matching C++ DualUpdate."""
        for j in range(len(self.constraint_functions)):
            # Set dual to projected dual (computed when calculating cost)
            self.dual_variables[j] = self.projected_duals[j]

    def penalty_update(self, scaling: Float, penalty_max: Float) -> None:
        """Update penalty parameters matching C++ PenaltyUpdate."""
        for j in range(len(self.constraint_functions)):
            self.penalty_parameters[j] = min(self.penalty_parameters[j] * scaling, penalty_max)

    def _calc_projected_duals(self) -> None:
        if not self.constraint_functions:
            return

        for j in range(len(self.constraint_functions)):
            dual_cone_type = dual_cone(self.constraint_types[j])

            # JIT-compiled dual estimation
            z_est = _dual_estimation_jit(
                self.dual_variables[j], self.penalty_parameters[j], self.constraint_vals[j]
            )

            # Project onto dual cone
            self.projected_duals[j] = conic_projection(dual_cone_type, z_est)

    def _calc_constraint_costs(self) -> Float:
        if not self.constraint_functions:
            return 0.0

        self._calc_projected_duals()

        cost = 0.0
        for j in range(len(self.constraint_functions)):
            # JIT-compiled cost contribution
            cost_contrib = _constraint_cost_jit(self.projected_duals[j], self.penalty_parameters[j])
            cost += float(cost_contrib)

        return cost

    def _calc_constraint_cost_gradients(self) -> None:
        if not self.constraint_functions:
            return

        self._calc_conic_jacobians()

        n = self.get_state_dim()
        m = self.get_input_dim()

        self._assert_arrays_non_none()
        assert self.lx_ is not None
        if not self.is_terminal:
            assert self.lu_ is not None

        for j in range(len(self.constraint_functions)):
            constraint_jac = self.constraint_jacs[j]
            projected_dual = self.projected_duals[j]

            # Extract state and input parts
            jac_x = constraint_jac[:, :n]  # State jacobian part
            jac_u = constraint_jac[:, n : n + m]  # Input jacobian part

            # JIT-compiled matrix operations
            grad_x = _transpose_matrix_vector_product_jit(jac_x, projected_dual)

            # Accumulate gradients
            self.lx_ = self.lx_ - grad_x

            if not self.is_terminal:
                assert self.lu_ is not None
                grad_u = _transpose_matrix_vector_product_jit(jac_u, projected_dual)
                self.lu_ = self.lu_ - grad_u

    def _calc_constraint_cost_hessians(self) -> None:
        self._calc_conic_hessians()

        n = self.get_state_dim()
        m = self.get_input_dim()

        self._assert_arrays_non_none()
        assert self.lxx_ is not None
        if not self.is_terminal:
            assert self.luu_ is not None
            assert self.lux_ is not None

        for j in range(len(self.constraint_functions)):
            # Add constraint Hessian contribution
            self.lxx_ = self.lxx_ + self.constraint_hessians[j][:n, :n]

            if not self.is_terminal:
                assert self.luu_ is not None
                assert self.lux_ is not None
                self.luu_ = self.luu_ + self.constraint_hessians[j][n : n + m, n : n + m]
                self.lux_ = self.lux_ + self.constraint_hessians[j][n : n + m, :n]

    def _calc_conic_jacobians(self) -> None:
        for j in range(len(self.constraint_functions)):
            dual_cone_type = dual_cone(self.constraint_types[j])
            z_est = self.dual_variables[j] - self.penalty_parameters[j] * self.constraint_vals[j]

            # Compute projection Jacobian
            proj_jac = conic_projection_jacobian(dual_cone_type, z_est)

            # Compute Jacobian-transpose vector product
            self.projected_duals[j] = proj_jac.T @ self.projected_duals[j]

    def _calc_conic_hessians(self) -> None:
        for j in range(len(self.constraint_functions)):
            dual_cone_type = dual_cone(self.constraint_types[j])
            z_est = self.dual_variables[j] - self.penalty_parameters[j] * self.constraint_vals[j]

            # Gauss-Newton approximation
            proj_jac = conic_projection_jacobian(dual_cone_type, z_est)
            jac_proj = proj_jac @ self.constraint_jacs[j]

            self.constraint_hessians[j] = self.penalty_parameters[j] * jac_proj.T @ jac_proj

            # Add second-order term for non-linear cones
            if not conic_projection_is_linear(dual_cone_type):
                proj_hess = conic_projection_hessian(dual_cone_type, z_est, self.projected_duals[j])
                hess_contrib = proj_hess @ self.constraint_jacs[j]
                self.constraint_hessians[j] = (
                    self.constraint_hessians[j]
                    + self.penalty_parameters[j] * self.constraint_jacs[j].T @ hess_contrib
                )

    def _calc_original_cost(self) -> Float:
        n = self.get_state_dim()
        m = self.get_input_dim()

        assert self.x_ is not None
        assert self.u_ is not None

        if self.cost_function_type == CostFunctionType.GENERIC:
            assert self.cost_function is not None
            return float(self.cost_function(self.x_, self.u_))
        elif self.cost_function_type == CostFunctionType.QUADRATIC:
            assert self.Q is not None
            assert self.q is not None
            Q_mat = self.Q.reshape(n, n)
            cost = 0.5 * self.x_.T @ Q_mat @ self.x_ + self.q.T @ self.x_

            if not self.is_terminal:
                assert self.R is not None
                assert self.r is not None
                assert self.H is not None
                R_mat = self.R.reshape(m, m)
                cost += 0.5 * self.u_.T @ R_mat @ self.u_ + self.r.T @ self.u_
                cost += self.u_.T @ self.H @ self.x_

            cost += self.c
            return float(cost)
        elif self.cost_function_type == CostFunctionType.DIAGONAL:
            assert self.Q is not None
            assert self.q is not None
            cost = 0.5 * self.x_.T @ jnp.diag(self.Q[:n]) @ self.x_ + self.q.T @ self.x_

            if not self.is_terminal:
                assert self.R is not None
                assert self.r is not None
                cost += 0.5 * self.u_.T @ jnp.diag(self.R[:m]) @ self.u_ + self.r.T @ self.u_

            cost += self.c
            return float(cost)

        # This should never be reached due to all enum cases being covered
        _altro_throw("Invalid cost function type", ErrorCode.COST_FUN_NOT_SET)

    def _calc_original_cost_gradient(self) -> None:
        n = self.get_state_dim()
        m = self.get_input_dim()

        assert self.x_ is not None
        assert self.u_ is not None

        if self.cost_function_type == CostFunctionType.GENERIC:
            assert self.cost_gradient is not None
            self.lx_, self.lu_ = self.cost_gradient(self.x_, self.u_)
        elif self.cost_function_type == CostFunctionType.QUADRATIC:
            assert self.Q is not None
            assert self.q is not None
            Q_mat = self.Q.reshape(n, n)
            self.lx_ = Q_mat @ self.x_ + self.q

            if not self.is_terminal:
                assert self.R is not None
                assert self.r is not None
                assert self.H is not None
                R_mat = self.R.reshape(m, m)
                self.lu_ = R_mat @ self.u_ + self.r + self.H @ self.x_
                self.lx_ = self.lx_ + self.H.T @ self.u_
        elif self.cost_function_type == CostFunctionType.DIAGONAL:
            assert self.Q is not None
            assert self.q is not None
            self.lx_ = self.Q[:n] * self.x_ + self.q

            if not self.is_terminal:
                assert self.R is not None
                assert self.r is not None
                self.lu_ = self.R[:m] * self.u_ + self.r

    def _calc_original_cost_hessian(self) -> None:
        n = self.get_state_dim()
        m = self.get_input_dim()

        if self.cost_function_type == CostFunctionType.GENERIC:
            assert self.cost_hessian is not None
            assert self.x_ is not None
            assert self.u_ is not None
            self.lxx_, self.luu_, self.lux_ = self.cost_hessian(self.x_, self.u_)
        elif self.cost_function_type == CostFunctionType.QUADRATIC:
            assert self.Q is not None
            self.lxx_ = self.Q.reshape(n, n)
            if not self.is_terminal:
                assert self.R is not None
                assert self.H is not None
                self.luu_ = self.R.reshape(m, m)
                self.lux_ = self.H
        elif self.cost_function_type == CostFunctionType.DIAGONAL:
            assert self.Q is not None
            self.lxx_ = jnp.diag(self.Q[:n])
            if not self.is_terminal:
                assert self.R is not None
                self.luu_ = jnp.diag(self.R[:m])
                self.lux_ = jnp.zeros((m, n))
