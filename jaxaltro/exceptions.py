"""Exception hierarchy for JAX-based ALTRO trajectory optimization.

This module provides the exception classes that directly correspond to the C++
error handling system, maintaining identical error categorization and messaging.
"""

from __future__ import annotations

from .types import ErrorCode


class AltroException(Exception):
    """Base exception class for ALTRO trajectory optimization errors.

    Corresponds to C++ AltroErrorException base class.
    """

    def __init__(self, message: str, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def __str__(self) -> str:
        return f"ALTRO Error {self.error_code.value}: {self.message}"


class DimensionError(AltroException):
    """Exception for dimension-related errors."""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DIMENSION_MISMATCH) -> None:
        super().__init__(message, error_code)


class InitializationError(AltroException):
    """Exception for solver initialization errors."""

    def __init__(
        self, message: str, error_code: ErrorCode = ErrorCode.SOLVER_NOT_INITIALIZED
    ) -> None:
        super().__init__(message, error_code)


class OptimizationError(AltroException):
    """Exception for optimization algorithm errors."""

    def __init__(self, message: str, error_code: ErrorCode) -> None:
        super().__init__(message, error_code)


class ConstraintError(AltroException):
    """Exception for constraint-related errors."""

    def __init__(
        self, message: str, error_code: ErrorCode = ErrorCode.INVALID_CONSTRAINT_DIM
    ) -> None:
        super().__init__(message, error_code)


def _error_code_to_string(error_code: ErrorCode) -> str:
    """Convert error code to descriptive string matching C++ ErrorCodeToString."""
    error_messages = {
        ErrorCode.NO_ERROR: "no error",
        ErrorCode.STATE_DIM_UNKNOWN: "state dimension unknown",
        ErrorCode.INPUT_DIM_UNKNOWN: "input dimension unknown",
        ErrorCode.NEXT_STATE_DIM_UNKNOWN: "next state dimension unknown",
        ErrorCode.DIMENSION_UNKNOWN: "dimension unknown",
        ErrorCode.BAD_INDEX: "bad index",
        ErrorCode.DIMENSION_MISMATCH: "dimension mismatch",
        ErrorCode.SOLVER_NOT_INITIALIZED: "solver not initialized",
        ErrorCode.SOLVER_ALREADY_INITIALIZED: "solver already initialized",
        ErrorCode.NON_POSITIVE: "expected a positive value",
        ErrorCode.TIMESTEP_NOT_POSITIVE: "timestep not positive",
        ErrorCode.COST_FUN_NOT_SET: "cost function not set",
        ErrorCode.DYNAMICS_FUN_NOT_SET: "dynamics function not set",
        ErrorCode.INVALID_OPT_AT_TERMINAL_KNOT_POINT: "invalid operation at terminal knot point index",
        ErrorCode.MAX_CONSTRAINTS_EXCEEDED: "max number of constraints at a knot point exceeded",
        ErrorCode.INVALID_CONSTRAINT_DIM: "invalid constraint dimension",
        ErrorCode.CHOLESKY_FAILED: "Cholesky factorization failed",
        ErrorCode.OP_ONLY_VALID_AT_TERMINAL_KNOT_POINT: "operation only valid at terminal knot point",
        ErrorCode.INVALID_POINTER: "Invalid pointer",
        ErrorCode.BACKWARD_PASS_FAILED: "Backward pass failed. Try increasing regularization",
        ErrorCode.LINE_SEARCH_FAILED: "Line search failed to find a point satisfying the Strong Wolfe Conditions",
        ErrorCode.MERIT_FUNCTION_GRADIENT_TOO_SMALL: "Merit function gradient under tol_meritfun_gradient. Aborting line search",
        ErrorCode.INVALID_BOUND_CONSTRAINT: "Invalid bound constraint. Make sure all upper bounds are greater than or equal to the lower bounds",
        ErrorCode.NON_POSITIVE_PENALTY: "Penalty must be strictly positive",
        ErrorCode.COST_NOT_QUADRATIC: "Invalid operation. Cost function not quadratic",
        ErrorCode.FILE_ERROR: "file error",
    }
    return error_messages.get(error_code, "unknown error")


def _print_error_code(error_code: ErrorCode) -> None:
    """Print error code information matching C++ PrintErrorCode."""
    print(f"Got error code {error_code.value}: {_error_code_to_string(error_code)}")


def _altro_throw(message: str, error_code: ErrorCode) -> None:
    """Throw ALTRO exception matching C++ ALTRO_THROW macro behavior."""
    raise AltroException(message, error_code)
