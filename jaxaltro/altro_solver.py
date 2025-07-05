"""User-facing ALTRO solver API for JAX-based trajectory optimization.

This module provides the main ALTROSolver class that directly corresponds to the C++
altro_solver.hpp/cpp implementation, maintaining identical interface and functionality.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .exceptions import ErrorCode, _altro_throw
from .solver_impl import SolverImpl
from .solver_options import AltroOptions
from .types import (
    ALL_INDICES,
    LAST_INDEX,
    CallbackFunction,
    ConstraintFunction,
    ConstraintJacobian,
    ConstraintType,
    CostFunction,
    CostGradient,
    CostHessian,
    ExplicitDynamicsFunction,
    ExplicitDynamicsJacobian,
    Float,
    SolveStatus,
)


class ConstraintIndex:
    def __init__(self, knot_point_index: int, constraint_index: int):
        self.knot_point_index = knot_point_index
        self.constraint_index = constraint_index

    def get_knot_point_index(self) -> int:
        return self.knot_point_index


class ALTROSolver:
    def __init__(self, horizon_length: int):
        self.solver = SolverImpl(horizon_length)
        self._callback_function: CallbackFunction | None = None

    def set_dimension(
        self, num_states: int, num_inputs: int, k_start: int = ALL_INDICES, k_stop: int = 0
    ) -> None:
        if self.is_initialized():
            _altro_throw(
                "Cannot change dimension once solver has been initialized",
                ErrorCode.SOLVER_ALREADY_INITIALIZED,
            )

        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        if num_states <= 0:
            _altro_throw("Number of states must be positive", ErrorCode.STATE_DIM_UNKNOWN)

        for k in range(k_start, k_stop):
            self.solver.nx[k] = num_states
            self.solver.nu[k] = num_inputs
            self.solver.data[k].set_dimension(num_states, num_inputs)

            # Set next state dimension for previous knot point
            if k > 0:
                self.solver.data[k - 1].set_next_state_dimension(num_states)

    def set_time_step(self, h: Float, k_start: int = ALL_INDICES, k_stop: int = 0) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, False)

        if h <= 0.0:
            _altro_throw("Time step must be positive", ErrorCode.TIMESTEP_NOT_POSITIVE)

        for k in range(k_start, k_stop):
            self.solver.data[k].set_timestep(h)
            self.solver.h[k] = h

    def set_explicit_dynamics(
        self,
        dynamics_function: ExplicitDynamicsFunction,
        dynamics_jacobian: ExplicitDynamicsJacobian,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, False)
        self._assert_dimensions_are_set(k_start, k_stop, "Cannot set dynamics")

        for k in range(k_start, k_stop):
            self.solver.data[k].set_dynamics(dynamics_function, dynamics_jacobian)

    def set_cost_function(
        self,
        cost_function: CostFunction,
        cost_gradient: CostGradient,
        cost_hessian: CostHessian,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        for k in range(k_start, k_stop):
            self.solver.data[k].set_cost_function(cost_function, cost_gradient, cost_hessian)

    def set_diagonal_cost(
        self,
        num_states: int,
        num_inputs: int,
        Q_diag: Array,
        R_diag: Array,
        q: Array,
        r: Array,
        c: Float,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)
        self._assert_dimensions_are_set(k_start, k_stop, "Cannot set cost function")

        for k in range(k_start, k_stop):
            n = self.get_state_dim(k)
            m = self.get_input_dim(k)

            if n != num_states:
                _altro_throw("State dimension mismatch", ErrorCode.DIMENSION_MISMATCH)
            if k != self.get_horizon_length() and m != num_inputs:
                _altro_throw("Input dimension mismatch", ErrorCode.DIMENSION_MISMATCH)

            self.solver.data[k].set_diagonal_cost(n, m, Q_diag, R_diag, q, r, float(c))

    def set_quadratic_cost(
        self,
        num_states: int,
        num_inputs: int,
        Q: Array,
        R: Array,
        H: Array,
        q: Array,
        r: Array,
        c: Float,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)
        self._assert_dimensions_are_set(k_start, k_stop, "Cannot set cost function")

        for k in range(k_start, k_stop):
            n = self.get_state_dim(k)
            m = self.get_input_dim(k)

            if n != num_states:
                _altro_throw("State dimension mismatch", ErrorCode.DIMENSION_MISMATCH)
            if k != self.get_horizon_length() and m != num_inputs:
                _altro_throw("Input dimension mismatch", ErrorCode.DIMENSION_MISMATCH)

            self.solver.data[k].set_quadratic_cost(n, m, Q, R, H, q, r, c)

    def set_lqr_cost(
        self,
        num_states: int,
        num_inputs: int,
        Q_diag: Array,
        R_diag: Array,
        x_ref: Array,
        u_ref: Array,
        k_start: int,
        k_stop: int = 0,
    ) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)
        self._assert_dimensions_are_set(k_start, k_stop, "Cannot set cost function")

        N = self.get_horizon_length()

        for k in range(k_start, k_stop):
            n = self.get_state_dim(k)
            m = self.get_input_dim(k)

            if n != num_states:
                _altro_throw(f"State dimension mismatch at index {k}", ErrorCode.DIMENSION_MISMATCH)
            if k != N and m != num_inputs:
                _altro_throw(f"Input dimension mismatch at index {k}", ErrorCode.DIMENSION_MISMATCH)

            # Convert to standard quadratic form: (x-xref)^T Q (x-xref) + (u-uref)^T R (u-uref)
            q = -Q_diag * x_ref
            r = -R_diag * u_ref
            c = 0.5 * jnp.dot(x_ref, Q_diag * x_ref)

            if k != N:
                c += 0.5 * jnp.dot(u_ref, R_diag * u_ref)

            self.solver.data[k].set_diagonal_cost(n, m, Q_diag, R_diag, q, r, float(c))

    def set_constraint(
        self,
        constraint_function: ConstraintFunction,
        constraint_jacobian: ConstraintJacobian,
        dim: int,
        constraint_type: ConstraintType,
        label: str,
        k_start: int,
        k_stop: int = 0,
        con_inds: list[ConstraintIndex] | None = None,
    ) -> None:
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        if self.is_initialized():
            _altro_throw(
                "Cannot set constraints: Solver already initialized",
                ErrorCode.SOLVER_ALREADY_INITIALIZED,
            )

        self._assert_dimensions_are_set(k_start, k_stop, "Cannot set constraint")

        num_indices = k_stop - k_start
        if con_inds is not None:
            con_inds.clear()

        for k in range(k_start, k_stop):
            label_k = label
            if num_indices > 1:
                label_k += f"_{k}"

            # Add constraint to knot point
            ncon = self.solver.data[k].num_constraints()
            self.solver.data[k].set_constraint(
                constraint_function, constraint_jacobian, dim, constraint_type, label_k
            )

            # Store constraint index
            if con_inds is not None:
                con_inds.append(ConstraintIndex(k, ncon))

    def set_initial_state(self, x0: Array, n: int) -> None:
        n0 = self.get_state_dim(0)

        if n0 > 0 and n != n0:
            _altro_throw(
                f"Dimension mismatch: Got {n}, expected {n0}", ErrorCode.DIMENSION_MISMATCH
            )

        if n0 <= 0:
            self.solver.nx[0] = n

        self.solver.initial_state = x0

    def initialize(self) -> None:
        self._assert_dimensions_are_set(0, self.get_horizon_length(), "Cannot initialize solver")
        self._assert_timesteps_are_positive("Cannot initialize solver")
        self.solver.initialize()

    def set_state(self, x: Array, n: int, k_start: int = ALL_INDICES, k_stop: int = 0) -> None:
        self._assert_initialized()
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        for k in range(k_start, k_stop):
            self._assert_state_dim(k, n)
            self.solver.data[k].x_ = x

    def set_input(self, u: Array, m: int, k_start: int = ALL_INDICES, k_stop: int = 0) -> None:
        self._assert_initialized()
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, False)

        for k in range(k_start, k_stop):
            self._assert_input_dim(k, m)
            self.solver.data[k].u_ = u

    def open_loop_rollout(self) -> None:
        self.solver.open_loop_rollout()

    def set_options(self, opts: AltroOptions) -> None:
        self.solver.opts = opts

    def get_options(self) -> AltroOptions:
        return self.solver.opts

    def set_callback(self, callback: CallbackFunction) -> None:
        self._callback_function = callback

    def solve(self) -> SolveStatus:
        self.solver.solve()
        return self.solver.stats.status

    def get_status(self) -> SolveStatus:
        return self.solver.stats.status

    def get_iterations(self) -> int:
        return self.solver.stats.iterations

    def get_solve_time_ms(self) -> Float:
        return self.solver.stats.solve_time

    def get_primal_feasibility(self) -> Float:
        return self.solver.stats.primal_feasibility

    def get_final_objective(self) -> Float:
        return self.solver.stats.objective_value

    def calc_cost(self) -> Float:
        return self.solver.calc_cost()

    def get_horizon_length(self) -> int:
        return self.solver.horizon_length

    def get_state_dim(self, k: int) -> int:
        return self.solver.data[k].get_state_dim()

    def get_input_dim(self, k: int) -> int:
        return self.solver.data[k].get_input_dim()

    def get_time_step(self, k: int) -> Float:
        return self.solver.h[k]

    def is_initialized(self) -> bool:
        return self.solver.is_initialized_fn()

    def get_state(self, k: int) -> Array:
        k_stop = k + 1
        self._check_knot_point_indices(k, k_stop, True)
        self._assert_dimensions_are_set(k, k_stop)

        state = self.solver.data[k].x_
        if state is None:
            _altro_throw(
                f"State not initialized at knot point {k}", ErrorCode.SOLVER_NOT_INITIALIZED
            )
        return state

    def get_input(self, k: int) -> Array:
        k_stop = k + 1
        self._check_knot_point_indices(k, k_stop, False)
        self._assert_dimensions_are_set(k, k_stop)

        input_val = self.solver.data[k].u_
        if input_val is None:
            _altro_throw(
                f"Input not initialized at knot point {k}", ErrorCode.SOLVER_NOT_INITIALIZED
            )
        return input_val

    def get_dual_dynamics(self, k: int) -> Array:
        k_stop = k + 1
        self._check_knot_point_indices(k, k_stop, True)
        self._assert_dimensions_are_set(k, k_stop)

        dual = self.solver.data[k].y_
        if dual is None:
            _altro_throw(
                f"Dual not initialized at knot point {k}", ErrorCode.SOLVER_NOT_INITIALIZED
            )
        return dual

    def update_linear_costs(
        self,
        q: Array | None,
        r: Array | None,
        c: Float,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        self._assert_initialized()
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        for k in range(k_start, k_stop):
            # Update costs - this would need to be implemented in KnotPointData
            # For now, just validate the operation is possible
            if not self.solver.data[k].cost_function_is_quadratic():
                _altro_throw(
                    f"Cannot update linear costs at index {k}. Cost function not quadratic",
                    ErrorCode.COST_NOT_QUADRATIC,
                )

    def shift_trajectory(self) -> None:
        N = self.get_horizon_length()

        for k in range(N):
            self.solver.data[k].x_ = self.solver.data[k + 1].x_
            if k < N - 1:
                self.solver.data[k].u_ = self.solver.data[k + 1].u_

    def print_state_trajectory(self) -> None:
        print("STATE TRAJECTORY:")
        for k in range(self.get_horizon_length() + 1):
            print(f" x[{k:03d}]: {self.solver.data[k].x_}")

    def print_input_trajectory(self) -> None:
        """Print input trajectory matching C++ PrintInputTrajectory."""
        print("INPUT TRAJECTORY:")
        for k in range(self.get_horizon_length()):
            print(f" u[{k:03d}]: {self.solver.data[k].u_}")

    def _check_knot_point_indices(
        self, k_start: int, k_stop: int, inclusive: bool
    ) -> tuple[int, int]:
        # Determine terminal index
        terminal_index = self.get_horizon_length()
        if not inclusive:
            terminal_index -= 1

        # Handle default cases
        if k_start == ALL_INDICES and k_stop == 0:
            k_start = 0
            k_stop = LAST_INDEX

        if k_start == 0 and k_stop == LAST_INDEX:
            k_start = 0
            k_stop = terminal_index + 1

        # Default k_stop if not set
        if k_stop <= 0:
            k_stop = k_start + 1

        # Validate indices
        if k_start < 0 or k_start > terminal_index:
            _altro_throw(f"Knot point index out of range: {k_start}", ErrorCode.BAD_INDEX)
        if k_stop < 0 or k_stop > terminal_index + 1:
            _altro_throw(f"Terminal knot point index out of range: {k_stop}", ErrorCode.BAD_INDEX)
        if k_stop > 0 and k_stop <= k_start:
            print(f"WARNING: Stopping index {k_stop} not greater than starting index {k_start}")

        return k_start, k_stop

    def _assert_initialized(self) -> None:
        """Assert solver is initialized."""
        if not self.is_initialized():
            _altro_throw("Solver must be initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

    def _assert_dimensions_are_set(self, k_start: int, k_stop: int, msg: str = "") -> None:
        """Assert dimensions are set for knot point range."""
        N = self.get_horizon_length()

        for k in range(k_start, k_stop):
            n = self.get_state_dim(k)
            m = self.get_input_dim(k)

            if n <= 0 or (k < N and m <= 0):
                _altro_throw(
                    f"{msg}. Dimensions at knot point {k} haven't been set",
                    ErrorCode.DIMENSION_UNKNOWN,
                )

    def _assert_state_dim(self, k: int, n: int) -> None:
        """Assert state dimension matches expected."""
        state_dim = self.get_state_dim(k)
        if state_dim != n:
            _altro_throw(
                f"State dimension mismatch. Got {n}, expected {state_dim}",
                ErrorCode.DIMENSION_MISMATCH,
            )

    def _assert_input_dim(self, k: int, m: int) -> None:
        """Assert input dimension matches expected."""
        input_dim = self.get_input_dim(k)
        if input_dim != m:
            _altro_throw(
                f"Input dimension mismatch. Got {m}, expected {input_dim}",
                ErrorCode.DIMENSION_MISMATCH,
            )

    def _assert_timesteps_are_positive(self, msg: str = "") -> None:
        """Assert all timesteps are positive."""
        for k in range(self.get_horizon_length()):
            h = self.get_time_step(k)
            if h <= 0.0:
                _altro_throw(
                    f"{msg}. Timestep is nonpositive at timestep {k}", ErrorCode.NON_POSITIVE
                )
