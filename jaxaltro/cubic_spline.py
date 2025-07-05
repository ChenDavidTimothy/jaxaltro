"""Cubic spline interpolation for line search in JAX-based ALTRO.

This module provides cubic spline functionality that directly corresponds to the C++
cubicspline.h/c implementation, maintaining identical mathematical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp

from .types import Float


# Tolerance matching C++ LINESEARCH_TOL
LINESEARCH_TOL = 1e-6


class CubicSplineReturnCode(Enum):
    """Return codes for cubic spline operations matching C++ enum."""

    NO_ERROR = "CS_NOERROR"
    FOUND_MINIMUM = "CS_FOUND_MINIMUM"
    INVALID_POINTER = "CS_INVALIDPOINTER"
    SADDLE_POINT = "CS_SADDLEPOINT"
    NO_MINIMUM = "CS_NOMINIMUM"
    IS_POSITIVE_QUADRATIC = "CS_IS_POSITIVE_QUADRATIC"
    IS_LINEAR = "CS_IS_LINEAR"
    IS_CONSTANT = "CS_IS_CONSTANT"
    UNEXPECTED_ERROR = "CS_UNEXPECTED_ERROR"
    SAME_POINT = "CS_SAME_POINT"


@dataclass(frozen=True)
class CubicSpline:
    """Cubic spline representation matching C++ CubicSpline struct.

    Represents polynomial: a + b*(x-x0) + c*(x-x0)^2 + d*(x-x0)^3
    """

    x0: Float
    a: Float
    b: Float
    c: Float
    d: Float

    def is_valid(self) -> bool:
        """Check if spline coefficients are valid (finite)."""
        return bool(
            jnp.all(
                jnp.array(
                    [
                        jnp.isfinite(self.x0),
                        jnp.isfinite(self.a),
                        jnp.isfinite(self.b),
                        jnp.isfinite(self.c),
                        jnp.isfinite(self.d),
                    ]
                )
            )
        )

    def evaluate(self, x: Float) -> Float:
        """Evaluate spline at given point matching C++ CubicSpline_Eval."""
        if not self.is_valid():
            return float(jnp.nan)

        delta = x - self.x0
        return float(self.a + self.b * delta + self.c * delta**2 + self.d * delta**3)

    def is_quadratic(self) -> bool:
        """Check if spline is effectively quadratic matching C++ CubicSpline_IsQuadratic."""
        return bool(jnp.abs(self.d) < LINESEARCH_TOL)


def _quadratic_formula(a: Float, b: Float, c: Float) -> tuple[Float, Float, CubicSplineReturnCode]:
    """Solve quadratic equation matching C++ QuadraticFormula."""
    if jnp.abs(a) < LINESEARCH_TOL:
        return float(jnp.nan), float(jnp.nan), CubicSplineReturnCode.IS_LINEAR

    discriminant = b**2 - 4 * a * c

    if jnp.abs(discriminant) < LINESEARCH_TOL:
        discriminant = 0.0
    elif discriminant < 0:
        return float(jnp.nan), float(jnp.nan), CubicSplineReturnCode.NO_MINIMUM

    sqrt_discriminant = jnp.sqrt(discriminant)
    x1 = (-b + sqrt_discriminant) / (2 * a)
    x2 = (-b - sqrt_discriminant) / (2 * a)

    return float(x1), float(x2), CubicSplineReturnCode.NO_ERROR


def cubic_spline_from_2_points(
    x1: Float, y1: Float, d1: Float, x2: Float, y2: Float, d2: Float
) -> tuple[CubicSpline, CubicSplineReturnCode]:
    """Create cubic spline from two points and derivatives matching C++ CubicSpline_From2Points."""
    delta = x2 - x1

    if jnp.abs(delta) < LINESEARCH_TOL:
        return CubicSpline(
            float(jnp.nan), float(jnp.nan), float(jnp.nan), float(jnp.nan), float(jnp.nan)
        ), CubicSplineReturnCode.SAME_POINT

    a = y1
    b = d1
    c = 3 * (y2 - y1) / (delta**2) - (d2 + 2 * d1) / delta
    d = (d2 + d1) / (delta**2) - 2 * (y2 - y1) / (delta**3)

    return CubicSpline(
        float(x1), float(a), float(b), float(c), float(d)
    ), CubicSplineReturnCode.NO_ERROR


def cubic_spline_from_3_points(
    x0: Float, y0: Float, d0: Float, x1: Float, y1: Float, x2: Float, y2: Float
) -> tuple[CubicSpline, CubicSplineReturnCode]:
    """Create cubic spline from three points matching C++ CubicSpline_From3Points."""
    delta1 = x1 - x0
    delta2 = x2 - x0

    if jnp.abs(delta1) < LINESEARCH_TOL or jnp.abs(delta2) < LINESEARCH_TOL:
        return CubicSpline(
            float(jnp.nan), float(jnp.nan), float(jnp.nan), float(jnp.nan), float(jnp.nan)
        ), CubicSplineReturnCode.SAME_POINT

    dy1 = (y1 - y0) / (delta1**2) - d0 / delta1
    dy2 = (y2 - y0) / (delta2**2) - d0 / delta2
    s = 1 / (delta2 - delta1)

    a = y0
    b = d0
    c = dy1 * (1 + delta1 * s) - dy2 * delta1 * s
    d = -dy1 * s + dy2 * s

    return CubicSpline(
        float(x0), float(a), float(b), float(c), float(d)
    ), CubicSplineReturnCode.NO_ERROR


def quadratic_spline_from_2_points(
    x0: Float, y0: Float, d0: Float, x1: Float, y1: Float
) -> tuple[CubicSpline, CubicSplineReturnCode]:
    """Create quadratic spline from two points matching C++ QuadraticSpline_From2Points."""
    delta = x1 - x0
    dy = (y1 - y0) / (delta**2) - d0 / delta

    return CubicSpline(
        float(x0), float(y0), float(d0), float(dy), 0.0
    ), CubicSplineReturnCode.NO_ERROR


def cubic_spline_argmin(spline: CubicSpline) -> tuple[Float, CubicSplineReturnCode]:
    """Find minimum of cubic spline matching C++ CubicSpline_ArgMin."""
    if not spline.is_valid():
        return float(jnp.nan), CubicSplineReturnCode.INVALID_POINTER

    b, c, d = spline.b, spline.c, spline.d

    # Check if quadratic
    is_quadratic = jnp.abs(d) < LINESEARCH_TOL
    is_linear = is_quadratic and jnp.abs(c) < LINESEARCH_TOL
    is_constant = is_linear and jnp.abs(b) < LINESEARCH_TOL

    if is_constant:
        return float(jnp.nan), CubicSplineReturnCode.IS_CONSTANT
    elif is_linear:
        return float(jnp.nan), CubicSplineReturnCode.IS_LINEAR
    elif is_quadratic:
        if c <= 0:
            return float(jnp.nan), CubicSplineReturnCode.IS_POSITIVE_QUADRATIC
        else:
            return float(-b / (2 * c) + spline.x0), CubicSplineReturnCode.FOUND_MINIMUM

    # Find stationary points for cubic
    d1, d2, err_code = _quadratic_formula(3 * d, 2 * c, b)

    if err_code != CubicSplineReturnCode.NO_ERROR:
        return float(jnp.nan), err_code

    # Check curvature at stationary points
    curv1 = 2 * c + 6 * d * d1
    curv2 = 2 * c + 6 * d * d2

    x1 = d1 + spline.x0
    x2 = d2 + spline.x0

    if jnp.abs(curv1) < LINESEARCH_TOL and jnp.abs(curv2) < LINESEARCH_TOL:
        return float(jnp.nan), CubicSplineReturnCode.SADDLE_POINT
    elif curv1 > 0 and curv2 < 0:
        return float(x1), CubicSplineReturnCode.FOUND_MINIMUM
    elif curv1 < 0 and curv2 > 0:
        return float(x2), CubicSplineReturnCode.FOUND_MINIMUM
    else:
        return float(jnp.nan), CubicSplineReturnCode.UNEXPECTED_ERROR
