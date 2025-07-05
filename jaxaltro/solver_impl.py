"""Core ALTRO solver implementation for JAX-based trajectory optimization.

This module provides the main solver implementation that directly corresponds to the C++
solver.cpp file, maintaining identical algorithmic behavior and convergence properties.
"""

from __future__ import annotations

import time

import jax.numpy as jnp

from .exceptions import ErrorCode, _altro_throw
from .knot_point_data import KnotPointData
from .line_search import CubicLineSearch, LineSearchReturnCode
from .solver_options import AltroOptions
from .solver_stats import AltroStats
from .tvlqr import TVLQR_SUCCESS, tvlqr_backward_pass, tvlqr_forward_pass
from .types import Float, SolveStatus


class SolverImpl:
    """Core ALTRO solver implementation matching C++ SolverImpl class."""

    def __init__(self, horizon_length: int):
        """Initialize solver with given horizon length matching C++ constructor."""
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
        """Initialize solver matching C++ Initialize."""
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
        """Check if solver is initialized."""
        return self.is_initialized

    def solve(self) -> None:
        """Main solve loop matching C++ Solve."""
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
                stop_iterating = True
                break

            # Forward pass
            alpha, forward_pass_result = self.forward_pass()
            if forward_pass_result not in [
                ErrorCode.NO_ERROR,
                ErrorCode.MERIT_FUNCTION_GRADIENT_TOO_SMALL,
            ]:
                print(f"Forward pass failed: {forward_pass_result}")
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
        """Backward pass using TVLQR matching C++ BackwardPass."""
        N = self.horizon_length

        # Prepare data for TVLQR
        nx_list = [self.data[k].get_state_dim() for k in range(N + 1)]
        nu_list = [self.data[k].get_input_dim() for k in range(N)]

        A_list = [self.data[k].A_ for k in range(N)]
        B_list = [self.data[k].B_ for k in range(N)]
        f_list = [self.data[k].f_ for k in range(N)]

        Q_list = [self.data[k].lxx_ for k in range(N + 1)]
        R_list = [self.data[k].luu_ for k in range(N)]
        H_list = [self.data[k].lux_ for k in range(N)]

        q_list = [self.data[k].lx_ for k in range(N + 1)]
        r_list = [self.data[k].lu_ for k in range(N)]

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
        """Forward pass with line search matching C++ ForwardPass."""

        # Define merit function for line search
        def merit_function(alpha: Float) -> tuple[Float, Float | None]:
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
        """Merit function evaluation matching C++ MeritFunction."""
        N = self.horizon_length

        phi = 0.0
        dphi = 0.0

        # Set initial state
        self.data[0].x_ = self.initial_state
        self.data[0].dx_da_ = jnp.zeros_like(self.initial_state)

        # Forward simulate
        for k in range(N):
            knot_point = self.data[k]
            next_knot_point = self.data[k + 1]

            # Compute control
            dx = knot_point.x_ - knot_point.x
            du = -knot_point.K_ @ dx + alpha * knot_point.d_
            knot_point.u_ = knot_point.u + du
            knot_point.y_ = knot_point.P_ @ dx + knot_point.p_

            # Simulate forward
            next_knot_point.x_ = knot_point.calc_dynamics(next_knot_point.x_)

            # Calculate cost
            knot_point.calc_constraints()
            cost = knot_point.calc_cost()
            phi += cost

            # Calculate gradient wrt alpha
            knot_point.calc_dynamics_expansion()
            knot_point.du_da_ = -knot_point.K_ @ knot_point.dx_da_ + knot_point.d_
            next_knot_point.dx_da_ = (
                knot_point.A_ @ knot_point.dx_da_ + knot_point.B_ @ knot_point.du_da_
            )

            # Calculate cost gradient
            knot_point.calc_constraint_jacobians()
            knot_point.calc_cost_gradient()
            dphi += knot_point.lx_.T @ knot_point.dx_da_
            dphi += knot_point.lu_.T @ knot_point.du_da_

        # Terminal knot point
        knot_point = self.data[N]
        knot_point.calc_constraints()
        cost = knot_point.calc_cost()
        phi += cost

        dx = knot_point.x_ - knot_point.x
        knot_point.y_ = knot_point.P_ @ dx + knot_point.p_

        knot_point.calc_constraint_jacobians()
        knot_point.calc_cost_gradient()
        dphi += knot_point.lx_.T @ knot_point.dx_da_

        # Invalidate cached values
        self._invalidate_cache()

        # Mark what was updated
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
        """Calculate stationarity measure matching C++ Stationarity."""
        N = self.horizon_length
        res_x = 0.0
        res_u = 0.0

        for k in range(N):
            z = self.data[k]
            zn = self.data[k + 1]

            # Stationarity condition: lx + A^T * y_next - y = 0
            res_x = max(res_x, jnp.max(jnp.abs(z.lx_ + z.A_.T @ zn.y_ - z.y_)))

            # Stationarity condition: lu + B^T * y_next = 0
            res_u = max(res_u, jnp.max(jnp.abs(z.lu_ + z.B_.T @ zn.y_)))

        # Terminal stationarity: lx - y = 0
        z = self.data[N]
        res_x = max(res_x, jnp.max(jnp.abs(z.lx_ - z.y_)))

        return max(res_x, res_u)

    def calc_feasibility(self) -> Float:
        """Calculate feasibility measure matching C++ Feasibility."""
        viol = 0.0
        for k in range(self.horizon_length + 1):
            viol = max(viol, self.data[k].calc_violations())
        return viol

    def open_loop_rollout(self) -> None:
        """Open loop rollout matching C++ OpenLoopRollout."""
        if not self.is_initialized:
            _altro_throw("Solver not initialized", ErrorCode.SOLVER_NOT_INITIALIZED)

        self.data[0].x_ = self.initial_state

        for k in range(self.horizon_length):
            next_x = self.data[k].calc_dynamics(self.data[k + 1].x_)
            self.data[k + 1].x_ = next_x

        self._invalidate_cache()

    def linear_rollout(self) -> None:
        """Linear rollout using TVLQR matching C++ LinearRollout."""
        N = self.horizon_length

        # Prepare data for TVLQR forward pass
        nx_list = [self.data[k].get_state_dim() for k in range(N + 1)]
        nu_list = [self.data[k].get_input_dim() for k in range(N)]

        A_list = [self.data[k].A_ for k in range(N)]
        B_list = [self.data[k].B_ for k in range(N)]
        f_list = [self.data[k].f_ for k in range(N)]

        K_list = [self.data[k].K_ for k in range(N)]
        d_list = [self.data[k].d_ for k in range(N)]
        P_list = [self.data[k].P_ for k in range(N + 1)]
        p_list = [self.data[k].p_ for k in range(N + 1)]

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
