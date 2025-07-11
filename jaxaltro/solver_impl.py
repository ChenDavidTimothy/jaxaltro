"""Core ALTRO solver implementation for JAX-based trajectory optimization.

This module provides the main solver implementation that directly corresponds to the C++
solver.cpp file, maintaining identical algorithmic behavior and convergence properties.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import Array

from .exceptions import ErrorCode, _altro_throw
from .knot_point_data import KnotPointData
from .line_search import CubicLineSearch, LineSearchReturnCode
from .solver_options import AltroOptions
from .solver_stats import AltroStats
from .tvlqr import TVLQR_SUCCESS, tvlqr_backward_pass, tvlqr_forward_pass
from .types import Float, SolveStatus


def _require_initialized_array(arr: Array | None, context: str) -> Array:
    if arr is None:
        _altro_throw(f"Array not initialized: {context}", ErrorCode.SOLVER_NOT_INITIALIZED)
    return arr


@jax.jit
def _compute_stationarity_step(lx_k, A_k, y_next, y_k, lu_k, B_k):
    stationarity_x = jnp.max(jnp.abs(lx_k + A_k.T @ y_next - y_k))
    stationarity_u = jnp.max(jnp.abs(lu_k + B_k.T @ y_next))
    return stationarity_x, stationarity_u


class SolverImpl:
    def __init__(self, horizon_length: int):
        self.horizon_length = horizon_length
        self.nx = [0] * (horizon_length + 1)  # State dimensions
        self.nu = [0] * (horizon_length + 1)  # Input dimensions
        self.h = [0.0] * horizon_length  # Time steps
        self.initial_state = jnp.zeros(0)

        # Options and statistics
        self.opts = AltroOptions()
        self.stats = AltroStats()

        # Knot point data
        self.data = []
        for k in range(horizon_length + 1):
            is_terminal = k == horizon_length
            self.data.append(KnotPointData(k, is_terminal))

        # Line search
        self.line_search = CubicLineSearch()

        # Internal state flags
        self.is_initialized = False
        self.constraint_vals_up_to_date = False
        self.constraint_jacs_up_to_date = False
        self.projected_duals_up_to_date = False
        self.conic_jacs_up_to_date = False
        self.conic_hessians_up_to_date = False
        self.cost_gradients_up_to_date = False
        self.cost_hessians_up_to_date = False
        self.dynamics_jacs_up_to_date = False

        # Merit function values
        self.phi0 = 0.0
        self.dphi0 = 0.0
        self.phi = 0.0
        self.dphi = 0.0
        self.rho = 0.0
        self.ls_iters = 0

        # Expected cost reduction
        self.delta_V = jnp.zeros(2)

    def initialize(self) -> None:
        # Initialize each knot point
        for data in self.data:
            data.initialize()

        # Initialize pointer arrays for TVLQR (conceptually - we use lists in Python)
        N = self.horizon_length
        for k in range(N + 1):
            self.nx[k] = self.data[k].get_state_dim()
            if k < N:
                self.nu[k] = self.data[k].get_input_dim()

        self.is_initialized = True

    def is_initialized_fn(self) -> bool:
        return self.is_initialized

    def solve(self) -> None:
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        # Configure line search
        self.line_search.use_backtracking_linesearch = self.opts.use_backtracking_linesearch
        self.rho = self.opts.penalty_initial

        # Initial rollout
        self.open_loop_rollout()
        self.copy_trajectory()
        cost_initial = self.calc_cost()

        # Initialize expansions
        for k in range(self.horizon_length + 1):
            self.data[k].calc_dynamics_expansion()
            self.data[k].calc_constraint_jacobians()
            self.data[k].calc_cost_gradient()
            self.data[k].set_penalty(self.opts.penalty_initial)

        self.dynamics_jacs_up_to_date = True
        self.constraint_jacs_up_to_date = True
        self.conic_jacs_up_to_date = True
        self.cost_gradients_up_to_date = True

        # Start timing
        start_time = time.time()

        if self.opts.verbose.value != "Silent":
            print("STARTING ALTRO iLQR SOLVE....")
            print(f"  Initial Cost: {cost_initial}")

        # Main iteration loop
        is_converged = False
        stop_iterating = False

        self.stats.status = SolveStatus.UNSOLVED

        for iteration in range(self.opts.iterations_max):
            # Calculate expansions
            self.calc_expansions()

            # Backward pass
            backward_pass_result = self.backward_pass()
            if backward_pass_result != ErrorCode.NO_ERROR:
                print(f"Backward pass failed: {backward_pass_result}")
                # ADD THESE LINES to compute stationarity before breaking:
                stationarity = self.calc_stationarity()
                feasibility = self.calc_feasibility()
                self.stats.stationarity = stationarity
                self.stats.primal_feasibility = feasibility
                stop_iterating = True
                break

            # Forward pass
            alpha, forward_pass_result = self.forward_pass()
            if forward_pass_result not in [
                ErrorCode.NO_ERROR,
                ErrorCode.MERIT_FUNCTION_GRADIENT_TOO_SMALL,
            ]:
                print(f"Forward pass failed: {forward_pass_result}")
                # ADD THESE LINES to compute stationarity before breaking:
                stationarity = self.calc_stationarity()
                feasibility = self.calc_feasibility()
                self.stats.stationarity = stationarity
                self.stats.primal_feasibility = feasibility
                stop_iterating = True
                break

            # Calculate convergence criteria
            stationarity = self.calc_stationarity()
            feasibility = self.calc_feasibility()
            self.copy_trajectory()

            cost_decrease = self.phi0 - self.phi

            # Check convergence
            if (
                abs(stationarity) < self.opts.tol_stationarity
                and feasibility < self.opts.tol_primal_feasibility
            ):
                is_converged = True
                stop_iterating = True
                self.stats.status = SolveStatus.SUCCESS

            # Dual and penalty updates
            dual_update = False
            penalty = self.rho

            if stationarity < jnp.sqrt(self.opts.tol_stationarity):
                self.dual_update()

                if feasibility > self.opts.tol_primal_feasibility:
                    self.penalty_update()

                # Update projected duals
                for k in range(self.horizon_length + 1):
                    self.data[k]._calc_projected_duals()
                    self.data[k].calc_cost_gradient()

                self.projected_duals_up_to_date = True
                dual_update = True

            # Print progress
            if self.opts.verbose.value != "Silent":
                print(
                    f"  iter = {iteration:3d}, phi = {self.phi0:8.4g} -> {self.phi:8.4g} "
                    f"({cost_decrease:10.3g}), dphi = {self.dphi0:10.3g} -> {self.dphi:10.3g}, "
                    f"alpha = {alpha:8.3g}, ls_iter = {self.ls_iters:2d}, "
                    f"stat = {stationarity:8.3e}, feas = {feasibility:8.3e}, "
                    f"rho = {penalty:7.2g}, dual update? {dual_update}"
                )

            if stop_iterating:
                break

        # Finalize statistics
        if not is_converged and iteration == self.opts.iterations_max - 1:
            self.stats.status = SolveStatus.MAX_ITERATIONS

        self.stats.iterations = iteration + 1
        self.stats.solve_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.stats.objective_value = self.phi
        self.stats.stationarity = stationarity
        self.stats.primal_feasibility = feasibility

        if self.opts.verbose.value != "Silent":
            print("ALTRO SOLVE FINISHED!")

    def backward_pass(self) -> ErrorCode:
        if not self.is_initialized:
            return ErrorCode.SOLVER_NOT_INITIALIZED

        N = self.horizon_length

        # Prepare data for TVLQR with proper validation - matching C++ initialization pattern
        nx_list = [self.data[k].get_state_dim() for k in range(N + 1)]
        nu_list = [self.data[k].get_input_dim() for k in range(N)]

        # Validate and extract arrays for backward pass
        A_list = [
            _require_initialized_array(self.data[k].A_, f"dynamics A matrix at knot point {k}")
            for k in range(N)
        ]
        B_list = [
            _require_initialized_array(self.data[k].B_, f"dynamics B matrix at knot point {k}")
            for k in range(N)
        ]
        f_list = [
            _require_initialized_array(self.data[k].f_, f"dynamics f vector at knot point {k}")
            for k in range(N)
        ]

        Q_list = [
            _require_initialized_array(self.data[k].lxx_, f"cost Hessian lxx at knot point {k}")
            for k in range(N + 1)
        ]
        q_list = [
            _require_initialized_array(self.data[k].lx_, f"cost gradient lx at knot point {k}")
            for k in range(N + 1)
        ]

        R_list = [
            _require_initialized_array(self.data[k].luu_, f"cost Hessian luu at knot point {k}")
            for k in range(N)
        ]
        H_list = [
            _require_initialized_array(self.data[k].lux_, f"cost Hessian lux at knot point {k}")
            for k in range(N)
        ]
        r_list = [
            _require_initialized_array(self.data[k].lu_, f"cost gradient lu at knot point {k}")
            for k in range(N)
        ]

        # Perform backward pass
        K_list, d_list, P_list, p_list, delta_V, status = tvlqr_backward_pass(
            nx_list,
            nu_list,
            N,
            A_list,
            B_list,
            f_list,
            Q_list,
            R_list,
            H_list,
            q_list,
            r_list,
            reg=0.0,
            is_diag=False,
        )

        if status != TVLQR_SUCCESS:
            return ErrorCode.BACKWARD_PASS_FAILED

        # Store results back in knot point data
        for k in range(N):
            self.data[k].K_ = K_list[k]
            self.data[k].d_ = d_list[k]

        for k in range(N + 1):
            self.data[k].P_ = P_list[k]
            self.data[k].p_ = p_list[k]

        self.delta_V = delta_V

        return ErrorCode.NO_ERROR

    def forward_pass(self) -> tuple[Float, ErrorCode]:
        # Define merit function for line search
        def merit_function(alpha: Float) -> tuple[Float, Float]:
            phi, dphi = self.merit_function(alpha)
            return phi, dphi

        # Initial merit function evaluation
        self.phi0, self.dphi0 = self.merit_function(0.0)

        if abs(self.dphi0) < self.opts.tol_meritfun_gradient:
            return 0.0, ErrorCode.MERIT_FUNCTION_GRADIENT_TOO_SMALL

        # Configure line search
        self.line_search.set_verbose(self.opts.verbose.value == "LineSearch")
        self.line_search.try_cubic_first = True

        # Run line search
        alpha = self.line_search.run(merit_function, 1.0, self.phi0, self.dphi0)
        self.phi, self.dphi = self.line_search.get_final_merit_values()

        ls_status = self.line_search.get_status()
        self.ls_iters = self.line_search.iterations()

        # Handle backtracking line search case
        if self.opts.use_backtracking_linesearch and abs(alpha - 1.0) > 0:
            for k in range(self.horizon_length + 1):
                self.data[k].calc_dynamics_expansion()
                self.data[k].calc_constraint_jacobians()
                self.data[k].calc_cost_gradient()

        # Check line search success
        if jnp.isnan(alpha) or ls_status not in [
            LineSearchReturnCode.MINIMUM_FOUND,
            LineSearchReturnCode.HIT_MAX_STEPSIZE,
        ]:
            return alpha, ErrorCode.LINE_SEARCH_FAILED

        return alpha, ErrorCode.NO_ERROR

    def merit_function(self, alpha: Float) -> tuple[Float, Float]:
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        N = self.horizon_length
        phi = 0.0
        dphi = 0.0

        # Validate and cache arrays ONCE at start
        initial_state = _require_initialized_array(self.data[0].x_, "initial state x_[0]")
        assert initial_state is not None  # Type hint for checker
        self.data[0].x_ = self.initial_state
        self.data[0].dx_da_ = jnp.zeros_like(self.initial_state)

        # Pre-validate all arrays to avoid repeated validation overhead
        cached_arrays = {}
        for k in range(N + 1):
            knot_point = self.data[k]

            # Cache reference arrays (read-only in loop)
            if k < N:
                ref_state = _require_initialized_array(knot_point.x, f"reference state x[{k}]")
                ref_input = _require_initialized_array(knot_point.u, f"reference input u[{k}]")
                feedback_gain = _require_initialized_array(knot_point.K_, f"feedback gain K_[{k}]")
                feedforward = _require_initialized_array(knot_point.d_, f"feedforward term d_[{k}]")
                cost_to_go_matrix = _require_initialized_array(
                    knot_point.P_, f"cost-to-go matrix P_[{k}]"
                )
                cost_to_go_vector = _require_initialized_array(
                    knot_point.p_, f"cost-to-go vector p_[{k}]"
                )

                # Type assertions for checker
                assert ref_state is not None
                assert ref_input is not None
                assert feedback_gain is not None
                assert feedforward is not None
                assert cost_to_go_matrix is not None
                assert cost_to_go_vector is not None

                cached_arrays[f"ref_state_{k}"] = ref_state
                cached_arrays[f"ref_input_{k}"] = ref_input
                cached_arrays[f"feedback_gain_{k}"] = feedback_gain
                cached_arrays[f"feedforward_{k}"] = feedforward
                cached_arrays[f"cost_to_go_matrix_{k}"] = cost_to_go_matrix
                cached_arrays[f"cost_to_go_vector_{k}"] = cost_to_go_vector

        # Forward simulate with cached arrays
        for k in range(N):
            knot_point = self.data[k]
            next_knot_point = self.data[k + 1]

            # Use cached arrays (no validation overhead)
            current_state = knot_point.x_  # Already validated above
            assert current_state is not None  # Type hint for checker
            ref_state = cached_arrays[f"ref_state_{k}"]
            ref_input = cached_arrays[f"ref_input_{k}"]
            feedback_gain = cached_arrays[f"feedback_gain_{k}"]
            feedforward_term = cached_arrays[f"feedforward_{k}"]
            cost_to_go_matrix = cached_arrays[f"cost_to_go_matrix_{k}"]
            cost_to_go_vector = cached_arrays[f"cost_to_go_vector_{k}"]

            # Compute control (preserve exact logic)
            dx = current_state - ref_state
            du = -feedback_gain @ dx + alpha * feedforward_term
            knot_point.u_ = ref_input + du
            knot_point.y_ = cost_to_go_matrix @ dx + cost_to_go_vector

            # Simulate forward (preserve method call - may have side effects)
            next_state = _require_initialized_array(next_knot_point.x_, f"next state x_[{k + 1}]")
            assert next_state is not None  # Type hint for checker
            next_knot_point.x_ = knot_point.calc_dynamics(next_state)

            # Calculate cost (preserve method calls)
            knot_point.calc_constraints()
            cost = knot_point.calc_cost()
            phi += cost

            # Calculate gradient wrt alpha
            knot_point.calc_dynamics_expansion()

            # Use validated arrays for gradient computation
            state_sensitivity = knot_point.dx_da_  # Already initialized above
            assert state_sensitivity is not None  # Type hint for checker
            knot_point.du_da_ = -feedback_gain @ state_sensitivity + feedforward_term

            # These arrays are set by calc_dynamics_expansion() above
            dynamics_A = _require_initialized_array(knot_point.A_, f"dynamics A_[{k}]")
            dynamics_B = _require_initialized_array(knot_point.B_, f"dynamics B_[{k}]")
            assert dynamics_A is not None
            assert dynamics_B is not None
            next_knot_point.dx_da_ = dynamics_A @ state_sensitivity + dynamics_B @ knot_point.du_da_

            # Calculate cost gradient
            knot_point.calc_constraint_jacobians()
            knot_point.calc_cost_gradient()

            # Use arrays without validation (set by calc_cost_gradient())
            cost_grad_state = _require_initialized_array(knot_point.lx_, f"cost gradient lx_[{k}]")
            cost_grad_input = _require_initialized_array(knot_point.lu_, f"cost gradient lu_[{k}]")
            assert cost_grad_state is not None
            assert cost_grad_input is not None

            dphi += float(cost_grad_state.T @ state_sensitivity)
            dphi += float(cost_grad_input.T @ knot_point.du_da_)

        # Terminal knot point (preserve existing logic exactly)
        terminal_knot_point = self.data[N]
        terminal_knot_point.calc_constraints()
        cost = terminal_knot_point.calc_cost()
        phi += cost

        terminal_state = _require_initialized_array(
            terminal_knot_point.x_, f"terminal state x_[{N}]"
        )
        terminal_ref_state = _require_initialized_array(
            terminal_knot_point.x, f"terminal reference state x[{N}]"
        )
        assert terminal_state is not None
        assert terminal_ref_state is not None
        dx = terminal_state - terminal_ref_state

        terminal_cost_to_go_matrix = _require_initialized_array(
            terminal_knot_point.P_, f"terminal cost-to-go matrix P_[{N}]"
        )
        terminal_cost_to_go_vector = _require_initialized_array(
            terminal_knot_point.p_, f"terminal cost-to-go vector p_[{N}]"
        )
        assert terminal_cost_to_go_matrix is not None
        assert terminal_cost_to_go_vector is not None
        terminal_knot_point.y_ = terminal_cost_to_go_matrix @ dx + terminal_cost_to_go_vector

        terminal_knot_point.calc_constraint_jacobians()
        terminal_knot_point.calc_cost_gradient()

        terminal_cost_grad = _require_initialized_array(
            terminal_knot_point.lx_, f"terminal cost gradient lx_[{N}]"
        )
        terminal_state_sens = _require_initialized_array(
            terminal_knot_point.dx_da_, f"terminal state sensitivity dx_da_[{N}]"
        )
        assert terminal_cost_grad is not None
        assert terminal_state_sens is not None
        dphi += float(terminal_cost_grad.T @ terminal_state_sens)

        # Preserve cache invalidation logic
        self._invalidate_cache()
        self.constraint_vals_up_to_date = True
        self.projected_duals_up_to_date = True
        self.dynamics_jacs_up_to_date = True
        self.cost_gradients_up_to_date = True
        self.constraint_jacs_up_to_date = True
        self.conic_jacs_up_to_date = True

        return phi, dphi

    def calc_cost(self) -> Float:
        """Calculate total cost matching C++ CalcCost."""
        cost = 0.0
        for k in range(self.horizon_length + 1):
            self.data[k].calc_constraints()
            cost_k = self.data[k].calc_cost()
            cost += cost_k

        self.constraint_vals_up_to_date = True
        self.projected_duals_up_to_date = True
        return cost

    def calc_stationarity(self) -> Float:
        N = self.horizon_length

        # Pre-validate all required arrays once
        validated_data = []
        for k in range(N):
            z = self.data[k]
            zn = self.data[k + 1]

            lx_k = _require_initialized_array(z.lx_, f"state gradient lx_[{k}]")
            A_k = _require_initialized_array(z.A_, f"dynamics A_[{k}]")
            y_next = _require_initialized_array(zn.y_, f"next dual y_[{k + 1}]")
            y_k = _require_initialized_array(z.y_, f"current dual y_[{k}]")
            lu_k = _require_initialized_array(z.lu_, f"input gradient lu_[{k}]")
            B_k = _require_initialized_array(z.B_, f"dynamics B_[{k}]")

            # Type assertions for checker
            assert lx_k is not None
            assert A_k is not None
            assert y_next is not None
            assert y_k is not None
            assert lu_k is not None
            assert B_k is not None

            validated_data.append((lx_k, A_k, y_next, y_k, lu_k, B_k))

        # Compute stationarity using JIT-compiled kernel
        res_x = 0.0
        res_u = 0.0

        for lx_k, A_k, y_next, y_k, lu_k, B_k in validated_data:
            stat_x, stat_u = _compute_stationarity_step(lx_k, A_k, y_next, y_k, lu_k, B_k)
            res_x = max(res_x, float(stat_x))
            res_u = max(res_u, float(stat_u))

        # Terminal stationarity (preserve exact logic)
        terminal_z = self.data[N]
        terminal_lx = _require_initialized_array(
            terminal_z.lx_, f"terminal state gradient lx_[{N}]"
        )
        terminal_y = _require_initialized_array(terminal_z.y_, f"terminal dual y_[{N}]")
        assert terminal_lx is not None
        assert terminal_y is not None
        terminal_stat = float(jnp.max(jnp.abs(terminal_lx - terminal_y)))

        res_x = max(res_x, terminal_stat)
        return max(res_x, res_u)

    def calc_feasibility(self) -> Float:
        viol = 0.0
        for k in range(self.horizon_length + 1):
            viol_k = self.data[k].calc_violations()
            viol = max(viol, viol_k)
        return viol

    def open_loop_rollout(self) -> None:
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        self.data[0].x_ = self.initial_state

        for k in range(self.horizon_length):
            # Validate current state and initialize next state if needed
            current_state = _require_initialized_array(
                self.data[k].x_, f"current state x_[{k}] for rollout"
            )
            assert current_state is not None

            # Initialize next state if not set
            if self.data[k + 1].x_ is None:
                self.data[k + 1].x_ = jnp.zeros(self.data[k + 1].get_state_dim())

            next_state_ref = _require_initialized_array(
                self.data[k + 1].x_, f"next state reference x_[{k + 1}] for rollout"
            )
            assert next_state_ref is not None

            next_x = self.data[k].calc_dynamics(next_state_ref)
            self.data[k + 1].x_ = next_x

        self._invalidate_cache()

    def linear_rollout(self) -> None:
        """Linear rollout using TVLQR matching C++ LinearRollout."""
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        N = self.horizon_length

        # Prepare data for TVLQR forward pass with validation
        nx_list = [self.data[k].get_state_dim() for k in range(N + 1)]
        nu_list = [self.data[k].get_input_dim() for k in range(N)]

        # Validate and extract arrays for TVLQR forward pass
        A_list = [
            _require_initialized_array(self.data[k].A_, f"dynamics A_[{k}] for linear rollout")
            for k in range(N)
        ]
        B_list = [
            _require_initialized_array(self.data[k].B_, f"dynamics B_[{k}] for linear rollout")
            for k in range(N)
        ]
        f_list = [
            _require_initialized_array(self.data[k].f_, f"dynamics f_[{k}] for linear rollout")
            for k in range(N)
        ]
        K_list = [
            _require_initialized_array(self.data[k].K_, f"feedback gain K_[{k}] for linear rollout")
            for k in range(N)
        ]
        d_list = [
            _require_initialized_array(self.data[k].d_, f"feedforward d_[{k}] for linear rollout")
            for k in range(N)
        ]
        P_list = [
            _require_initialized_array(self.data[k].P_, f"cost-to-go P_[{k}] for linear rollout")
            for k in range(N + 1)
        ]
        p_list = [
            _require_initialized_array(self.data[k].p_, f"cost-to-go p_[{k}] for linear rollout")
            for k in range(N + 1)
        ]

        # Perform forward pass
        x_list, u_list, y_list = tvlqr_forward_pass(
            nx_list,
            nu_list,
            N,
            A_list,
            B_list,
            f_list,
            K_list,
            d_list,
            P_list,
            p_list,
            self.initial_state,
        )

        # Store results
        for k in range(N + 1):
            self.data[k].x_ = x_list[k]
            self.data[k].y_ = y_list[k]
            if k < N:
                self.data[k].u_ = u_list[k]

        self._invalidate_cache()

    def copy_trajectory(self) -> None:
        """Copy trajectory matching C++ CopyTrajectory."""
        for k in range(self.horizon_length + 1):
            self.data[k].x = self.data[k].x_
            self.data[k].y = self.data[k].y_
            if k < self.horizon_length:
                self.data[k].u = self.data[k].u_

    def dual_update(self) -> None:
        """Update dual variables matching C++ DualUpdate."""
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        for k in range(self.horizon_length + 1):
            self.data[k].dual_update()

        self._invalidate_dual_cache()

    def penalty_update(self) -> None:
        """Update penalty parameters matching C++ PenaltyUpdate."""
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        for k in range(self.horizon_length + 1):
            self.data[k].penalty_update(self.opts.penalty_scaling, self.opts.penalty_max)

        self.rho = min(self.rho * self.opts.penalty_scaling, self.opts.penalty_max)
        self._invalidate_dual_cache()

    def calc_expansions(self) -> None:
        """Calculate all expansions matching C++ CalcExpansions."""
        for k in range(self.horizon_length + 1):
            self.data[k].calc_cost_hessian()

        self.conic_hessians_up_to_date = True
        self.cost_hessians_up_to_date = True

    def _invalidate_cache(self) -> None:
        """Invalidate all cached computation flags."""
        self.constraint_vals_up_to_date = False
        self.constraint_jacs_up_to_date = False
        self.projected_duals_up_to_date = False
        self.conic_jacs_up_to_date = False
        self.conic_hessians_up_to_date = False
        self.cost_gradients_up_to_date = False
        self.cost_hessians_up_to_date = False
        self.dynamics_jacs_up_to_date = False

    def _invalidate_dual_cache(self) -> None:
        """Invalidate dual-related cached computation flags."""
        self.projected_duals_up_to_date = False
        self.conic_jacs_up_to_date = False
        self.conic_hessians_up_to_date = False
        self.cost_gradients_up_to_date = False
        self.cost_hessians_up_to_date = False
