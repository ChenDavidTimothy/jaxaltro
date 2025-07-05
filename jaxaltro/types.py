"""Core type definitions for JAX-based ALTRO trajectory optimization.

This module provides JAX-compatible type aliases and enums that directly correspond
to the C++ implementation's type system.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

from jax import Array


# Core JAX array types matching C++ a_float usage
StateVector: TypeAlias = Array  # JAX array for state vectors
ControlInput: TypeAlias = Array  # JAX array for control inputs
DualVariable: TypeAlias = Array  # JAX array for dual variables
GradientArray: TypeAlias = Array  # JAX array for gradients
JacobianMatrix: TypeAlias = Array  # JAX array for Jacobian matrices
HessianMatrix: TypeAlias = Array  # JAX array for Hessian matrices

# Scalar types
Float: TypeAlias = float
Time: TypeAlias = float

# Function type aliases matching C++ function signatures
ExplicitDynamicsFunction: TypeAlias = Callable[[StateVector, ControlInput, Float], StateVector]
ExplicitDynamicsJacobian: TypeAlias = Callable[[StateVector, ControlInput, Float], JacobianMatrix]
ImplicitDynamicsFunction: TypeAlias = Callable[
    [StateVector, ControlInput, StateVector, ControlInput, Float], StateVector
]
ImplicitDynamicsJacobian: TypeAlias = Callable[
    [StateVector, ControlInput, StateVector, ControlInput, Float],
    tuple[JacobianMatrix, JacobianMatrix],
]

CostFunction: TypeAlias = Callable[[StateVector, ControlInput], Float]
CostGradient: TypeAlias = Callable[[StateVector, ControlInput], tuple[GradientArray, GradientArray]]
CostHessian: TypeAlias = Callable[
    [StateVector, ControlInput], tuple[HessianMatrix, HessianMatrix, HessianMatrix]
]

ConstraintFunction: TypeAlias = Callable[[StateVector, ControlInput], Array]
ConstraintJacobian: TypeAlias = Callable[[StateVector, ControlInput], JacobianMatrix]
CallbackFunction: TypeAlias = Callable[[], None]

# Constants matching C++ definitions
LAST_INDEX = -1
ALL_INDICES = -2


class SolveStatus(Enum):
    """Solver termination status matching C++ SolveStatus enum."""

    SUCCESS = "Success"
    UNSOLVED = "Unsolved"
    MAX_ITERATIONS = "MaxIterations"
    MAX_OBJECTIVE_EXCEEDED = "MaxObjectiveExceeded"
    STATE_OUT_OF_BOUNDS = "StateOutOfBounds"
    INPUT_OUT_OF_BOUNDS = "InputOutOfBounds"
    MERIT_FUN_GRADIENT_TOO_SMALL = "MeritFunGradientTooSmall"


class ConstraintType(Enum):
    """Constraint types matching C++ ConstraintType enum."""

    EQUALITY = "EQUALITY"
    IDENTITY = "IDENTITY"
    INEQUALITY = "INEQUALITY"
    SECOND_ORDER_CONE = "SECOND_ORDER_CONE"


class Verbosity(Enum):
    """Verbosity levels matching C++ Verbosity enum."""

    SILENT = "Silent"
    OUTER = "Outer"
    INNER = "Inner"
    LINE_SEARCH = "LineSearch"


class ErrorCode(Enum):
    """Error codes matching C++ ErrorCodes enum."""

    NO_ERROR = "NoError"
    STATE_DIM_UNKNOWN = "StateDimUnknown"
    INPUT_DIM_UNKNOWN = "InputDimUnknown"
    NEXT_STATE_DIM_UNKNOWN = "NextStateDimUnknown"
    DIMENSION_UNKNOWN = "DimensionUnknown"
    BAD_INDEX = "BadIndex"
    DIMENSION_MISMATCH = "DimensionMismatch"
    SOLVER_NOT_INITIALIZED = "SolverNotInitialized"
    SOLVER_ALREADY_INITIALIZED = "SolverAlreadyInitialized"
    NON_POSITIVE = "NonPositive"
    TIMESTEP_NOT_POSITIVE = "TimestepNotPositive"
    COST_FUN_NOT_SET = "CostFunNotSet"
    DYNAMICS_FUN_NOT_SET = "DynamicsFunNotSet"
    INVALID_OPT_AT_TERMINAL_KNOT_POINT = "InvalidOptAtTerminalKnotPoint"
    MAX_CONSTRAINTS_EXCEEDED = "MaxConstraintsExceeded"
    INVALID_CONSTRAINT_DIM = "InvalidConstraintDim"
    CHOLESKY_FAILED = "CholeskyFailed"
    OP_ONLY_VALID_AT_TERMINAL_KNOT_POINT = "OpOnlyValidAtTerminalKnotPoint"
    INVALID_POINTER = "InvalidPointer"
    BACKWARD_PASS_FAILED = "BackwardPassFailed"
    LINE_SEARCH_FAILED = "LineSearchFailed"
    MERIT_FUNCTION_GRADIENT_TOO_SMALL = "MeritFunctionGradientTooSmall"
    INVALID_BOUND_CONSTRAINT = "InvalidBoundConstraint"
    NON_POSITIVE_PENALTY = "NonPositivePenalty"
    COST_NOT_QUADRATIC = "CostNotQuadratic"
    FILE_ERROR = "FileError"
