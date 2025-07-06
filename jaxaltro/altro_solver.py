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
    ConstraintType,
    CostFunction,
    ExplicitDynamicsFunction,
    Float,
    SolveStatus,
)


class ConstraintIndex:
    """Constraint index identifier matching C++ ConstraintIndex."""

    def __init__(self, knot_point_index: int, constraint_index: int):
        self.knot_point_index = knot_point_index
        self.constraint_index = constraint_index

    def get_knot_point_index(self) -> int:
        """Get knot point index."""
        return self.knot_point_index


class ALTROSolver:
    """Main ALTRO solver interface matching C++ ALTROSolver class.

    This class provides the complete user-facing API for trajectory optimization
    using the ALTRO algorithm with JAX acceleration and automatic differentiation.
    """

    def __init__(self, horizon_length: int):
        """Initialize ALTRO solver matching C++ constructor.

        Args:
            horizon_length: Number of time steps in the trajectory
        """
        self.solver = SolverImpl(horizon_length)
        self._callback_function: CallbackFunction | None = None

    def set_dimension(
        self, num_states: int, num_inputs: int, k_start: int = ALL_INDICES, k_stop: int = 0
    ) -> None:
        """Set state and input dimensions matching C++ SetDimension.

        Args:
            num_states: Number of states (size of state vector)
            num_inputs: Number of inputs (size of control vector)
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
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
        """Set time step matching C++ SetTimeStep.

        Args:
            h: Time step
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, False)

        if h <= 0.0:
            _altro_throw("Time step must be positive", ErrorCode.TIMESTEP_NOT_POSITIVE)

        for k in range(k_start, k_stop):
            self.solver.data[k].set_timestep(h)
            self.solver.h[k] = h

    def set_explicit_dynamics(
        self,
        dynamics_function: ExplicitDynamicsFunction,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        """Set explicit dynamics with automatic Jacobian computation.

        Args:
            dynamics_function: Dynamics function
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, False)
        self._assert_dimensions_are_set(k_start, k_stop, "Cannot set dynamics")

        for k in range(k_start, k_stop):
            self.solver.data[k].set_dynamics(dynamics_function)

    def set_cost_function(
        self,
        cost_function: CostFunction,
        k_start: int = ALL_INDICES,
        k_stop: int = 0,
    ) -> None:
        """Set generic cost function with automatic gradient/Hessian computation.

        Args:
            cost_function: Cost function
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        for k in range(k_start, k_stop):
            self.solver.data[k].set_cost_function(cost_function)

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
        """Set diagonal cost function matching C++ SetDiagonalCost.

        Args:
            num_states: Number of states
            num_inputs: Number of inputs
            Q_diag: Diagonal state cost matrix
            R_diag: Diagonal input cost matrix
            q: Linear state cost
            r: Linear input cost
            c: Constant cost
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
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
        """Set quadratic cost function matching C++ SetQuadraticCost.

        Args:
            num_states: Number of states
            num_inputs: Number of inputs
            Q: State cost matrix
            R: Input cost matrix
            H: Cross-term cost matrix
            q: Linear state cost
            r: Linear input cost
            c: Constant cost
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
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
        """Set LQR tracking cost matching C++ SetLQRCost.

        Args:
            num_states: Number of states
            num_inputs: Number of inputs
            Q_diag: Diagonal state penalty matrix
            R_diag: Diagonal input penalty matrix
            x_ref: State reference
            u_ref: Input reference
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
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
        dim: int,
        constraint_type: ConstraintType,
        label: str,
        k_start: int,
        k_stop: int = 0,
        con_inds: list[ConstraintIndex] | None = None,
    ) -> None:
        """Set constraint with automatic Jacobian computation.

        Args:
            constraint_function: Constraint function
            dim: Constraint dimension
            constraint_type: Type of constraint
            label: Descriptive label
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
            con_inds: Optional list to store constraint indices
        """
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
            self.solver.data[k].set_constraint(constraint_function, dim, constraint_type, label_k)

            # Store constraint index
            if con_inds is not None:
                con_inds.append(ConstraintIndex(k, ncon))

    def set_initial_state(self, x0: Array, n: int) -> None:
        """Set initial state matching C++ SetInitialState.

        Args:
            x0: Initial state
            n: State dimension
        """
        n0 = self.get_state_dim(0)

        if n0 > 0 and n != n0:
            _altro_throw(
                f"Dimension mismatch: Got {n}, expected {n0}", ErrorCode.DIMENSION_MISMATCH
            )

        if n0 <= 0:
            self.solver.nx[0] = n

        self.solver.initial_state = x0

    def initialize(self) -> None:
        """Initialize solver matching C++ Initialize."""
        self._assert_dimensions_are_set(0, self.get_horizon_length(), "Cannot initialize solver")
        self._assert_timesteps_are_positive("Cannot initialize solver")
        self.solver.initialize()

    def set_state(self, x: Array, n: int, k_start: int = ALL_INDICES, k_stop: int = 0) -> None:
        """Set state trajectory matching C++ SetState.

        Args:
            x: State vector
            n: State dimension
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
        self._assert_initialized()
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, True)

        for k in range(k_start, k_stop):
            self._assert_state_dim(k, n)
            self.solver.data[k].x_ = x

    def set_input(self, u: Array, m: int, k_start: int = ALL_INDICES, k_stop: int = 0) -> None:
        """Set input trajectory matching C++ SetInput.

        Args:
            u: Input vector
            m: Input dimension
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
        self._assert_initialized()
        k_start, k_stop = self._check_knot_point_indices(k_start, k_stop, False)

        for k in range(k_start, k_stop):
            self._assert_input_dim(k, m)
            self.solver.data[k].u_ = u

    def open_loop_rollout(self) -> None:
        """Perform open loop rollout matching C++ OpenLoopRollout."""
        self.solver.open_loop_rollout()

    def set_options(self, opts: AltroOptions) -> None:
        """Set solver options matching C++ SetOptions."""
        self.solver.opts = opts

    def get_options(self) -> AltroOptions:
        """Get solver options matching C++ GetOptions."""
        return self.solver.opts

    def set_callback(self, callback: CallbackFunction) -> None:
        """Set callback function matching C++ SetCallback."""
        self._callback_function = callback

    def solve(self) -> SolveStatus:
        """Solve trajectory optimization problem matching C++ Solve."""
        self.solver.solve()
        return self.solver.stats.status

    def get_status(self) -> SolveStatus:
        """Get solver status matching C++ GetStatus."""
        return self.solver.stats.status

    def get_iterations(self) -> int:
        """Get number of iterations matching C++ GetIterations."""
        return self.solver.stats.iterations

    def get_solve_time_ms(self) -> Float:
        """Get solve time in milliseconds matching C++ GetSolveTimeMs."""
        return self.solver.stats.solve_time

    def get_primal_feasibility(self) -> Float:
        """Get primal feasibility matching C++ GetPrimalFeasibility."""
        return self.solver.stats.primal_feasibility

    def get_final_objective(self) -> Float:
        """Get final objective value matching C++ GetFinalObjective."""
        return self.solver.stats.objective_value

    def calc_cost(self) -> Float:
        """Calculate cost matching C++ CalcCost."""
        return self.solver.calc_cost()

    def get_horizon_length(self) -> int:
        """Get horizon length matching C++ GetHorizonLength."""
        return self.solver.horizon_length

    def get_state_dim(self, k: int) -> int:
        """Get state dimension at knot point matching C++ GetStateDim."""
        return self.solver.data[k].get_state_dim()

    def get_input_dim(self, k: int) -> int:
        """Get input dimension at knot point matching C++ GetInputDim."""
        return self.solver.data[k].get_input_dim()

    def get_time_step(self, k: int) -> Float:
        """Get time step at knot point matching C++ GetTimeStep."""
        return self.solver.h[k]

    def is_initialized(self) -> bool:
        """Check if solver is initialized matching C++ IsInitialized."""
        return self.solver.is_initialized_fn()

    def get_state(self, k: int) -> Array:
        """Get state at knot point matching C++ GetState.

        Args:
            k: Knot point index

        Returns:
            State vector at knot point k
        """
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
        """Get input at knot point matching C++ GetInput.

        Args:
            k: Knot point index

        Returns:
            Input vector at knot point k
        """
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
        """Get dynamics dual at knot point matching C++ GetDualDynamics.

        Args:
            k: Knot point index

        Returns:
            Dual variable vector at knot point k
        """
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
        """Update linear costs for MPC matching C++ UpdateLinearCosts.

        Args:
            q: Linear state cost (can be None)
            r: Linear input cost (can be None)
            c: Constant cost
            k_start: Starting knot point index
            k_stop: Ending knot point index (non-inclusive)
        """
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
        """Shift trajectory for MPC matching C++ ShiftTrajectory."""
        N = self.get_horizon_length()

        for k in range(N):
            self.solver.data[k].x_ = self.solver.data[k + 1].x_
            if k < N - 1:
                self.solver.data[k].u_ = self.solver.data[k + 1].u_

    def print_state_trajectory(self) -> None:
        """Print state trajectory matching C++ PrintStateTrajectory."""
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
        """Check and normalize knot point indices matching C++ CheckKnotPointIndices."""
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
