from __future__ import annotations

from dataclasses import dataclass

from .types import Float, Verbosity


@dataclass(frozen=True)
class AltroOptions:
    # Maximum number of iterations
    iterations_max: int = 200

    # Convergence tolerances
    tol_cost: Float = 1e-4
    tol_cost_intermediate: Float = 1e-4
    tol_primal_feasibility: Float = 1e-4
    tol_stationarity: Float = 1e-4
    tol_meritfun_gradient: Float = 1e-8

    # State and input bounds (optional)
    max_state_value: Float | None = None
    max_input_value: Float | None = None

    # Penalty method parameters
    penalty_initial: Float = 1.0
    penalty_scaling: Float = 10.0
    penalty_max: Float = 1e8

    # Verbosity level
    verbose: Verbosity = Verbosity.SILENT

    # Time limits
    max_solve_time: Float = float("inf")

    # Line search options
    use_backtracking_linesearch: bool = False

    # Exception handling
    throw_errors: bool = True

    def __post_init__(self) -> None:
        """Validate option values after initialization."""
        if self.iterations_max <= 0:
            raise ValueError("iterations_max must be positive")
        if self.tol_cost <= 0:
            raise ValueError("tol_cost must be positive")
        if self.tol_primal_feasibility <= 0:
            raise ValueError("tol_primal_feasibility must be positive")
        if self.tol_stationarity <= 0:
            raise ValueError("tol_stationarity must be positive")
        if self.penalty_initial <= 0:
            raise ValueError("penalty_initial must be positive")
        if self.penalty_scaling <= 1:
            raise ValueError("penalty_scaling must be greater than 1")
        if self.penalty_max <= self.penalty_initial:
            raise ValueError("penalty_max must be greater than penalty_initial")
