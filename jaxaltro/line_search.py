"""Cubic line search algorithm for JAX-based ALTRO trajectory optimization.

This module provides line search functionality that directly corresponds to the C++
linesearch.hpp/cpp implementation, maintaining identical optimization behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp

from .cubic_spline import (
    CubicSplineReturnCode,
    cubic_spline_argmin,
    cubic_spline_from_2_points,
)
from .types import Float


# Type alias for merit function - always returns both phi and dphi
MeritFunction = Callable[[Float], tuple[Float, Float]]


class LineSearchReturnCode(Enum):
    """Return codes for line search matching C++ enum."""

    NO_ERROR = "CLS_NOERROR"
    MINIMUM_FOUND = "CLS_MINIMUM_FOUND"
    INVALID_POINTER = "CLS_INVALID_POINTER"
    NOT_DESCENT_DIRECTION = "CLS_NOT_DESCENT_DIRECTION"
    WINDOW_TOO_SMALL = "CLS_WINDOW_TOO_SMALL"
    GOT_NONFINITE_STEP_SIZE = "CLS_GOT_NONFINITE_STEP_SIZE"
    MAX_ITERATIONS = "CLS_MAX_ITERS"
    HIT_MAX_STEPSIZE = "CLS_HIT_MAX_STEPSIZE"


@dataclass
class CubicLineSearch:
    """Cubic line search implementation matching C++ CubicLineSearch class."""

    # Options
    max_iters: int = 25
    alpha_max: Float = 2.0
    c1: Float = 1e-4  # Armijo condition parameter
    c2: Float = 0.9  # Wolfe condition parameter
    beta_increase: Float = 1.5
    beta_decrease: Float = 0.5
    min_interval_size: Float = 1e-6
    try_cubic_first: bool = False
    use_backtracking_linesearch: bool = False

    # State variables
    n_iters: int = 0
    phi0: Float = 0.0
    phi: Float = 0.0
    phi_lo: Float = 0.0
    phi_hi: Float = 0.0
    dphi0: Float = 0.0
    dphi: Float = 0.0
    dphi_lo: Float = 0.0
    dphi_hi: Float = 0.0
    sufficient_decrease: bool = False
    curvature: bool = False
    return_code: LineSearchReturnCode = LineSearchReturnCode.NO_ERROR
    verbose: bool = False

    def set_optimality_tolerances(self, c1: Float, c2: Float) -> bool:
        """Set Wolfe condition parameters."""
        if c2 <= c1:
            return False
        self.c1 = c1
        self.c2 = c2
        return True

    def set_verbose(self, verbose: bool) -> bool:
        """Set verbosity level."""
        old_verbose = self.verbose
        self.verbose = verbose
        return old_verbose

    def get_final_merit_values(self) -> tuple[Float, Float]:
        """Get final merit function values."""
        return self.phi, self.dphi

    def get_status(self) -> LineSearchReturnCode:
        """Get line search status."""
        return self.return_code

    def iterations(self) -> int:
        """Get number of iterations performed."""
        return self.n_iters

    def run(self, merit_fun: MeritFunction, alpha0: Float, phi0: Float, dphi0: Float) -> Float:
        """Run cubic line search matching C++ CubicLineSearch::Run."""
        # Store initial values
        self.phi0 = phi0
        self.dphi0 = dphi0

        # Reset state
        self.n_iters = 0
        self.sufficient_decrease = False
        self.curvature = False
        self.return_code = LineSearchReturnCode.NO_ERROR

        # Check descent direction
        if dphi0 >= 0.0:
            self.return_code = LineSearchReturnCode.NOT_DESCENT_DIRECTION
            return 0.0

        if self.verbose:
            print(f"Starting Cubic Line Search with phi0 = {phi0}, dphi0 = {dphi0}")

        # Stage 1: Expand interval until conditions are met
        alpha_prev = 0.0
        phi_prev = phi0
        dphi_prev = dphi0
        alpha = alpha0
        hit_max_alpha = False

        for iter_count in range(self.max_iters):
            self.n_iters += 1

            # Evaluate merit function - always returns both values
            phi, dphi = merit_fun(alpha)
            self.phi = phi
            self.dphi = dphi

            sufficient_decrease_satisfied = phi <= phi0 + self.c1 * alpha * dphi0
            function_not_decreasing = phi >= phi_prev
            strong_wolfe_satisfied = bool(jnp.abs(dphi) <= -self.c2 * dphi0)

            if self.verbose:
                print(f"  iter = {iter_count}: alpha = {alpha}, phi = {phi}, dphi = {dphi}")
                print(
                    f"    Armijo? {sufficient_decrease_satisfied}, Wolfe? {strong_wolfe_satisfied}"
                )

            # Check convergence
            if sufficient_decrease_satisfied and strong_wolfe_satisfied:
                if self.verbose:
                    print("  Optimal Step Found!")
                self.sufficient_decrease = True
                self.curvature = True
                self.return_code = LineSearchReturnCode.MINIMUM_FOUND
                return alpha

            # Try cubic interpolation on first iteration if enabled
            if iter_count == 0 and self.try_cubic_first:
                alpha_cubic = self._try_cubic_interpolation(alpha, phi, dphi)
                if alpha_cubic is not None:
                    phi_cubic, dphi_cubic = merit_fun(alpha_cubic)
                    self.n_iters += 1

                    sufficient_decrease_cubic = phi_cubic <= phi0 + self.c1 * alpha_cubic * dphi0
                    strong_wolfe_cubic = bool(jnp.abs(dphi_cubic) <= -self.c2 * dphi0)

                    if sufficient_decrease_cubic and strong_wolfe_cubic:
                        if self.verbose:
                            print("  Optimal Step Found via cubic interpolation!")
                        self.phi = phi_cubic
                        self.dphi = dphi_cubic
                        self.sufficient_decrease = True
                        self.curvature = True
                        self.return_code = LineSearchReturnCode.MINIMUM_FOUND
                        return alpha_cubic

            # Fall back to backtracking if enabled
            if self.use_backtracking_linesearch:
                return self._simple_backtracking(merit_fun, alpha0 * self.beta_decrease)

            # Check if we need to zoom
            if not sufficient_decrease_satisfied or (iter_count > 0 and function_not_decreasing):
                if self.verbose:
                    print("    Zooming with alo < ahi")

                self.phi_lo = phi_prev
                self.dphi_lo = dphi_prev
                self.phi_hi = phi
                self.dphi_hi = dphi
                return self._zoom(merit_fun, alpha_prev, alpha)

            # Check if gradient is non-negative
            if dphi >= 0:
                if self.verbose:
                    print(f"    Zooming with ahi < alo ({alpha_prev}, {alpha})")

                self.phi_lo = phi
                self.dphi_lo = dphi
                self.phi_hi = phi_prev
                self.dphi_hi = dphi_prev
                return self._zoom(merit_fun, alpha, alpha_prev)

            # Expand interval
            alpha_prev = alpha
            phi_prev = phi
            dphi_prev = dphi
            alpha = alpha * self.beta_increase

            if alpha > self.alpha_max:
                alpha = self.alpha_max
                if hit_max_alpha:
                    self.return_code = LineSearchReturnCode.HIT_MAX_STEPSIZE
                    self.sufficient_decrease = sufficient_decrease_satisfied
                    self.curvature = strong_wolfe_satisfied
                    return alpha
                else:
                    hit_max_alpha = True

            if self.verbose:
                print(f"    Expanding interval to alpha = {alpha}")

        self.return_code = LineSearchReturnCode.MAX_ITERATIONS
        return alpha

    def _try_cubic_interpolation(self, alpha: Float, phi: Float, dphi: Float) -> Float | None:
        """Try cubic interpolation on initial interval."""
        spline, err_code = cubic_spline_from_2_points(0.0, self.phi0, self.dphi0, alpha, phi, dphi)

        if err_code != CubicSplineReturnCode.NO_ERROR:
            return None

        alpha_cubic, argmin_code = cubic_spline_argmin(spline)

        if argmin_code == CubicSplineReturnCode.FOUND_MINIMUM and jnp.isfinite(alpha_cubic):
            if self.verbose:
                print(
                    f"    Used cubic interpolation on interval (0, {alpha}) -> alpha = {alpha_cubic}"
                )
            return alpha_cubic

        return None

    def _zoom(self, merit_fun: MeritFunction, alo: Float, ahi: Float) -> Float:
        """Zoom phase of line search matching C++ implementation."""
        if not jnp.isfinite(alo) or not jnp.isfinite(ahi):
            self.return_code = LineSearchReturnCode.GOT_NONFINITE_STEP_SIZE
            return 0.0

        for zoom_iter in range(self.n_iters + 1, self.max_iters):
            # Check if interval is too small
            if jnp.abs(alo - ahi) < self.min_interval_size:
                if self.verbose:
                    print(f"    Window size too small with alo = {alo}, ahi = {ahi}")

                alpha = (alo + ahi) / 2.0
                self.n_iters += 1
                phi, dphi = merit_fun(alpha)
                self.phi = phi
                self.dphi = dphi

                self.sufficient_decrease = phi <= self.phi0 + self.c1 * alpha * self.dphi0
                self.curvature = bool(jnp.abs(dphi) <= -self.c2 * self.dphi0)

                if self.sufficient_decrease and self.curvature:
                    self.return_code = LineSearchReturnCode.MINIMUM_FOUND
                else:
                    self.return_code = LineSearchReturnCode.WINDOW_TOO_SMALL

                return alpha

            # Try cubic interpolation
            spline, err_code = cubic_spline_from_2_points(
                alo, self.phi_lo, self.dphi_lo, ahi, self.phi_hi, self.dphi_hi
            )

            cubic_spline_failed = True
            if err_code == CubicSplineReturnCode.NO_ERROR:
                alpha, argmin_code = cubic_spline_argmin(spline)
                if argmin_code == CubicSplineReturnCode.FOUND_MINIMUM and jnp.isfinite(alpha):
                    cubic_spline_failed = False
                    if self.verbose:
                        a_min = min(alo, ahi)
                        a_max = max(alo, ahi)
                        print(
                            f"    Used cubic interpolation on interval ({a_min}, {a_max}) -> alpha = {alpha}"
                        )

            # Fall back to midpoint if cubic fails
            if cubic_spline_failed:
                if self.verbose:
                    print("    Cubic interpolation failed. Using midpoint.")
                alpha = (alo + ahi) / 2.0

            # Evaluate merit function
            self.n_iters += 1
            phi, dphi = merit_fun(alpha)
            self.phi = phi
            self.dphi = dphi

            sufficient_decrease = phi <= self.phi0 + self.c1 * alpha * self.dphi0
            higher_than_lo = phi > self.phi_lo
            curvature = bool(jnp.abs(dphi) <= -self.c2 * self.dphi0)

            if self.verbose:
                print(f"  zoom iter = {zoom_iter}: alpha = {alpha}, phi = {phi}, dphi = {dphi}")
                print(f"    Armijo? {sufficient_decrease}, Wolfe? {curvature}")

            if sufficient_decrease and curvature:
                if self.verbose:
                    print("  Optimal Step Found!")
                self.sufficient_decrease = True
                self.curvature = True
                self.return_code = LineSearchReturnCode.MINIMUM_FOUND
                return alpha

            if not sufficient_decrease or higher_than_lo:
                # Adjust ahi
                if self.verbose:
                    print("    Adjusting ahi")
                ahi = alpha
                self.phi_hi = phi
                self.dphi_hi = dphi
            else:
                # Adjust alo
                reset_ahi = dphi * (ahi - alo) <= 0
                if reset_ahi:
                    ahi = alo
                    self.phi_hi = self.phi_lo
                    self.dphi_hi = self.dphi_lo
                    if self.verbose:
                        print("    Setting ahi = alo")

                if self.verbose:
                    print("    Adjusting alo")
                alo = alpha
                self.phi_lo = phi
                self.dphi_lo = dphi

        self.return_code = LineSearchReturnCode.MAX_ITERATIONS
        return alpha

    def _simple_backtracking(self, merit_fun: MeritFunction, alpha0: Float) -> Float:
        """Simple backtracking line search."""
        alpha = alpha0

        for _ in range(1, self.max_iters):
            self.n_iters += 1
            phi, _ = merit_fun(alpha)
            self.phi = phi

            sufficient_decrease_satisfied = phi <= self.phi0 + self.c1 * alpha * self.dphi0

            if sufficient_decrease_satisfied:
                if self.verbose:
                    print("  Optimal Step Found!")
                self.sufficient_decrease = True
                self.curvature = True
                self.return_code = LineSearchReturnCode.MINIMUM_FOUND
                return alpha
            else:
                alpha *= self.beta_decrease

        return alpha

    def status_to_string(self) -> str:
        """Convert status to string description."""
        status_strings = {
            LineSearchReturnCode.NO_ERROR: "No error",
            LineSearchReturnCode.MINIMUM_FOUND: "Minimum found",
            LineSearchReturnCode.INVALID_POINTER: "Invalid pointer",
            LineSearchReturnCode.NOT_DESCENT_DIRECTION: "Not a descent direction",
            LineSearchReturnCode.WINDOW_TOO_SMALL: "Window too small",
            LineSearchReturnCode.GOT_NONFINITE_STEP_SIZE: "Got non-finite step size",
            LineSearchReturnCode.MAX_ITERATIONS: "Hit max iterations",
            LineSearchReturnCode.HIT_MAX_STEPSIZE: "Hit max stepsize. Try increasing alpha_max",
        }
        return status_strings.get(self.return_code, "Unknown status")
