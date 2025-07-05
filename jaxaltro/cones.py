"""Constraint cone projections for JAX-based ALTRO trajectory optimization.

This module provides cone projection algorithms that directly correspond to the C++
cones.hpp/cpp implementation, maintaining identical mathematical behavior.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .exceptions import AltroException, ErrorCode
from .types import ConstraintType


def dual_cone(cone: ConstraintType) -> ConstraintType:
    """Get dual cone type matching C++ DualCone function."""
    dual_cone_map = {
        ConstraintType.EQUALITY: ConstraintType.IDENTITY,
        ConstraintType.INEQUALITY: ConstraintType.INEQUALITY,
        ConstraintType.SECOND_ORDER_CONE: ConstraintType.SECOND_ORDER_CONE,
        ConstraintType.IDENTITY: ConstraintType.EQUALITY,
    }
    return dual_cone_map[cone]


def conic_projection_is_linear(cone: ConstraintType) -> bool:
    """Check if conic projection is linear matching C++ ConicProjectionIsLinear."""
    linear_cones = {
        ConstraintType.EQUALITY,
        ConstraintType.IDENTITY,
        ConstraintType.INEQUALITY,
    }
    return cone in linear_cones


def _second_order_cone_projection(x: Array) -> Array:
    """Project onto second-order cone matching C++ SecondOrderConeProjection.

    Assumes x is stacked as [v; s] where ||v|| <= s defines the cone.
    """
    n = x.shape[0] - 1  # dimension of v
    v = x[:n]
    s = x[n]

    # Compute ||v||
    v_norm = jnp.linalg.norm(v)

    # Three cases matching C++ implementation
    def below_cone():
        # a <= -s: below the cone
        return jnp.zeros_like(x)

    def in_cone():
        # a <= s: in the cone
        return x

    def outside_cone():
        # a > s: outside the cone
        c = 0.5 * (1 + s / v_norm)
        v_proj = c * v
        s_proj = c * v_norm
        return jnp.concatenate([v_proj, jnp.array([s_proj])])

    # Use JAX conditional logic
    return jnp.where(v_norm <= -s, below_cone(), jnp.where(v_norm <= s, in_cone(), outside_cone()))


def _second_order_cone_jacobian(x: Array) -> Array:
    """Compute Jacobian of second-order cone projection matching C++ SecondOrderConeJacobian."""
    n = x.shape[0] - 1
    v = x[:n]
    s = x[n]
    v_norm = jnp.linalg.norm(v)

    def below_cone_jac():
        return jnp.zeros((x.shape[0], x.shape[0]))

    def in_cone_jac():
        return jnp.eye(x.shape[0])

    def outside_cone_jac():
        c = 0.5 * (1 + s / v_norm)

        # dvdv block
        dvdv = jnp.outer(v, v) * (-0.5 * s / (v_norm**3))
        dvdv = dvdv + c * jnp.eye(n)

        # dvds block
        dvds = 0.5 * v / v_norm

        # dsdv block
        dsdv = ((-0.5 * s / (v_norm**2)) + c / v_norm) * v

        # dsds block
        dsds = jnp.array([0.5])

        # Assemble Jacobian
        top_row = jnp.hstack([dvdv, dvds.reshape(-1, 1)])
        bottom_row = jnp.hstack([dsdv.reshape(1, -1), dsds.reshape(1, 1)])
        return jnp.vstack([top_row, bottom_row])

    return jnp.where(
        v_norm <= -s, below_cone_jac(), jnp.where(v_norm <= s, in_cone_jac(), outside_cone_jac())
    )


def _second_order_cone_hessian(x: Array, b: Array) -> Array:
    """Compute Hessian of second-order cone projection matching C++ SecondOrderConeHessian."""
    n = x.shape[0] - 1
    v = x[:n]
    s = x[n]
    bv = b[:n]
    bs = b[n]

    v_norm = jnp.linalg.norm(v)
    vbv = jnp.dot(v, bv)

    def below_cone_hess():
        return jnp.zeros((x.shape[0], x.shape[0]))

    def in_cone_hess():
        return jnp.zeros((x.shape[0], x.shape[0]))

    def outside_cone_hess():
        H = jnp.zeros((x.shape[0], x.shape[0]))

        for i in range(n):
            hi = 0.0
            for j in range(n):
                Hij = -v[i] * v[j] / (v_norm**2)
                if i == j:
                    Hij += 1.0
                hi += Hij * bv[j]

            # H[i, n] and H[n, i]
            H = H.at[i, n].set(hi / (2 * v_norm))
            H = H.at[n, i].set(hi / (2 * v_norm))

            for j in range(i + 1):
                vij = v[i] * v[j]
                H1 = hi * v[j] * (-s / (v_norm**3))
                H2 = vij * (2 * vbv) / (v_norm**4) - v[i] * bv[j] / (v_norm**2)
                H3 = -vij / (v_norm**2)

                if i == j:
                    H2 -= vbv / (v_norm**2)
                    H3 += 1.0

                H2 *= s / v_norm
                H3 *= bs / v_norm

                Hij_val = (H1 + H2 + H3) / 2.0
                H = H.at[i, j].set(Hij_val)
                H = H.at[j, i].set(Hij_val)

        # H[n, n] = 0
        H = H.at[n, n].set(0.0)
        return H

    return jnp.where(
        v_norm <= -s, below_cone_hess(), jnp.where(v_norm <= s, in_cone_hess(), outside_cone_hess())
    )


def conic_projection(cone: ConstraintType, x: Array) -> Array:
    """Project vector onto specified cone matching C++ ConicProjection."""
    if cone == ConstraintType.EQUALITY:
        # Zero cone
        return jnp.zeros_like(x)
    elif cone == ConstraintType.IDENTITY:
        # Identity (no projection)
        return x
    elif cone == ConstraintType.INEQUALITY:
        # Negative orthant
        return jnp.minimum(0.0, x)
    elif cone == ConstraintType.SECOND_ORDER_CONE:
        return _second_order_cone_projection(x)
    else:
        raise AltroException(f"Unknown cone type: {cone}", ErrorCode.INVALID_CONSTRAINT_DIM)


def conic_projection_jacobian(cone: ConstraintType, x: Array) -> Array:
    """Compute Jacobian of conic projection matching C++ ConicProjectionJacobian."""
    if cone == ConstraintType.EQUALITY:
        return jnp.zeros((x.shape[0], x.shape[0]))
    elif cone == ConstraintType.IDENTITY:
        return jnp.eye(x.shape[0])
    elif cone == ConstraintType.INEQUALITY:
        return jnp.diag(jnp.where(x <= 0, 1.0, 0.0))
    elif cone == ConstraintType.SECOND_ORDER_CONE:
        return _second_order_cone_jacobian(x)
    else:
        raise AltroException(f"Unknown cone type: {cone}", ErrorCode.INVALID_CONSTRAINT_DIM)


def conic_projection_hessian(cone: ConstraintType, x: Array, b: Array) -> Array:
    """Compute Hessian of conic projection matching C++ ConicProjectionHessian."""
    if cone in [ConstraintType.EQUALITY, ConstraintType.IDENTITY, ConstraintType.INEQUALITY]:
        return jnp.zeros((x.shape[0], x.shape[0]))
    elif cone == ConstraintType.SECOND_ORDER_CONE:
        return _second_order_cone_hessian(x, b)
    else:
        raise AltroException(f"Unknown cone type: {cone}", ErrorCode.INVALID_CONSTRAINT_DIM)
