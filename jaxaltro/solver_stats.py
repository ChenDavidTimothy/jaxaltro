from __future__ import annotations

from dataclasses import dataclass

from .types import Float, SolveStatus


@dataclass
class AltroStats:
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
        self.status = SolveStatus.UNSOLVED
        self.solve_time = 0.0
        self.iterations = 0
        self.outer_iterations = 0
        self.objective_value = 0.0
        self.stationarity = 0.0
        self.primal_feasibility = 0.0
        self.complimentarity = 0.0

    def is_converged(self) -> bool:
        return self.status == SolveStatus.SUCCESS

    def get_solve_time_ms(self) -> Float:
        return self.solve_time

    def get_iterations(self) -> int:
        return self.iterations

    def get_final_objective(self) -> Float:
        return self.objective_value

    def get_primal_feasibility(self) -> Float:
        return self.primal_feasibility

    def get_stationarity(self) -> Float:
        return self.stationarity
