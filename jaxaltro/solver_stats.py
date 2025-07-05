"""Solver performance statistics for JAX-based ALTRO trajectory optimization.

This module provides the solver statistics that directly correspond to the C++
AltroStats struct, maintaining identical metrics and tracking capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Float, SolveStatus


@dataclass
class AltroStats:
    """Performance statistics for ALTRO solver matching C++ AltroStats struct.

    Tracks solver performance metrics including timing, convergence, and iteration counts.
    """

    # Solver termination status
    status: SolveStatus = SolveStatus.UNSOLVED

    # Timing information (in milliseconds)
    solve_time: Float = 0.0

    # Iteration counts
    iterations: int = 0
    outer_iterations: int = 0

    # Convergence metrics
    objective_value: Float = 0.0
    stationarity: Float = 0.0
    primal_feasibility: Float = 0.0
    complimentarity: Float = 0.0

    def reset(self) -> None:
        """Reset all statistics to initial values."""
        self.status = SolveStatus.UNSOLVED
        self.solve_time = 0.0
        self.iterations = 0
        self.outer_iterations = 0
        self.objective_value = 0.0
        self.stationarity = 0.0
        self.primal_feasibility = 0.0
        self.complimentarity = 0.0

    def is_converged(self) -> bool:
        """Check if solver has converged successfully."""
        return self.status == SolveStatus.SUCCESS

    def get_solve_time_ms(self) -> Float:
        """Get solve time in milliseconds."""
        return self.solve_time

    def get_iterations(self) -> int:
        """Get total number of iterations."""
        return self.iterations

    def get_final_objective(self) -> Float:
        """Get final objective value."""
        return self.objective_value

    def get_primal_feasibility(self) -> Float:
        """Get final primal feasibility measure."""
        return self.primal_feasibility

    def get_stationarity(self) -> Float:
        """Get final stationarity measure."""
        return self.stationarity
