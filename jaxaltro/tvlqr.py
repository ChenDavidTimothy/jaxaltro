"""Time-Varying Linear Quadratic Regulator for JAX-based ALTRO.

This module provides TVLQR functionality that directly corresponds to the C++
tvlqr.h/cpp implementation, maintaining identical mathematical behavior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from .types import Float


# Success code matching C++ TVLQR_SUCCESS
TVLQR_SUCCESS = -1


# REMOVE @jax.jit from the main function, add it to computational kernels

# ADD these JIT-compiled computational kernels at the top of tvlqr.py:


# REPLACE the JIT-compiled computational kernel with this corrected version:


@jax.jit
def _lqr_step_backward(
    A_k: Array,
    B_k: Array,
    f_k: Array,
    Q_k: Array,
    R_k: Array,
    H_k: Array,
    q_k: Array,
    r_k: Array,
    P_next: Array,
    p_next: Array,
    reg: Float,
) -> tuple[Array, Array, Array, Array, Array, Array, bool]:
    """Single backward LQR step - JIT compiled with error handling."""
    n, m = A_k.shape[0], B_k.shape[1]

    # Action-value function expansion
    AtP = A_k.T @ P_next
    Qxx = Q_k + AtP @ A_k

    BtP = B_k.T @ P_next
    Quu = R_k + BtP @ B_k

    Qux = H_k + BtP @ A_k

    Ppf = P_next @ f_k + p_next
    Qx = q_k + A_k.T @ Ppf
    Qu = r_k + B_k.T @ Ppf

    # Compute gains with regularization
    Quu_reg = Quu + reg * jnp.eye(m)

    # Check condition number before Cholesky (JAX-compatible)
    min_eigenval = jnp.linalg.eigvals(Quu_reg).min()
    cholesky_success = min_eigenval > 1e-12

    # Use conditional to handle potential failure
    def successful_decomp():
        Quu_chol = jsp.linalg.cholesky(Quu_reg, lower=True)

        # Solve for gains using triangular solves
        K_k = jsp.linalg.solve_triangular(Quu_chol, Qux, lower=True)
        K_k = jsp.linalg.solve_triangular(Quu_chol.T, K_k, lower=False)

        d_k = jsp.linalg.solve_triangular(Quu_chol, -Qu, lower=True)
        d_k = jsp.linalg.solve_triangular(Quu_chol.T, d_k, lower=False)

        # Compute cost-to-go
        KtQuu = K_k.T @ Quu_reg
        KtQux = K_k.T @ Qux

        P_k = Qxx + KtQuu @ K_k - KtQux - KtQux.T
        p_k = Qx - KtQuu @ d_k - K_k.T @ Qu + Qux.T @ d_k

        return K_k, d_k, P_k, p_k, Qu, Quu_reg @ d_k

    def failed_decomp():
        # Return zeros if decomposition fails
        K_k = jnp.zeros((m, n))
        d_k = jnp.zeros(m)
        P_k = jnp.zeros((n, n))
        p_k = jnp.zeros(n)
        return K_k, d_k, P_k, p_k, Qu, jnp.zeros(m)

    K_k, d_k, P_k, p_k, Qu_out, Quu_d = jax.lax.cond(
        cholesky_success, successful_decomp, failed_decomp
    )

    return K_k, d_k, P_k, p_k, Qu_out, Quu_d, cholesky_success


# REPLACE the _tvlqr_backward_pass function with this corrected version:


def _tvlqr_backward_pass(
    nx: list[int],
    nu: list[int],
    num_horizon: int,
    A: list[Array],
    B: list[Array],
    f: list[Array],
    Q: list[Array],
    R: list[Array],
    H: list[Array],
    q: list[Array],
    r: list[Array],
    reg: Float,
    is_diag: bool = False,
) -> tuple[list[Array], list[Array], list[Array], list[Array], Array, int]:
    """Backward pass for TVLQR with JIT-compiled computational kernels."""
    N = num_horizon

    # Initialize storage
    K = [jnp.zeros((nu[k], nx[k])) for k in range(N)]
    d = [jnp.zeros(nu[k]) for k in range(N)]
    P = [jnp.zeros((nx[k], nx[k])) for k in range(N + 1)]
    p = [jnp.zeros(nx[k]) for k in range(N + 1)]
    delta_V = jnp.zeros(2)

    # Terminal cost-to-go
    if is_diag:
        P[N] = jnp.diag(Q[N])
    else:
        P[N] = Q[N]
    p[N] = q[N]

    # Backward pass
    for k in range(N - 1, -1, -1):
        n = nx[k]
        m = nu[k]

        # Get current matrices
        A_k = A[k]
        B_k = B[k]
        f_k = f[k]

        if is_diag:
            Q_k = jnp.diag(Q[k])
            R_k = jnp.diag(R[k])
            H_k = jnp.zeros((n, m))
        else:
            Q_k = Q[k]
            R_k = R[k]
            H_k = H[k]

        q_k = q[k]
        r_k = r[k]
        P_next = P[k + 1]
        p_next = p[k + 1]

        # JIT-compiled computational step with error checking
        K_k, d_k, P_k, p_k, Qu_final, Quu_d, success = _lqr_step_backward(
            A_k, B_k, f_k, Q_k, R_k, H_k, q_k, r_k, P_next, p_next, reg
        )

        # Check if step succeeded
        if not success:
            # Return failure code (knot point index where failure occurred)
            return K, d, P, p, delta_V, k

        K[k] = K_k
        d[k] = d_k
        P[k] = P_k
        p[k] = p_k

        # Expected cost reduction (corrected computation)
        delta_V = delta_V.at[0].add(d_k.T @ Qu_final)
        delta_V = delta_V.at[1].add(0.5 * d_k.T @ Quu_d)

    return K, d, P, p, delta_V, TVLQR_SUCCESS


@jax.jit
def _lqr_step_forward(
    x_k: Array, K_k: Array, d_k: Array, A_k: Array, B_k: Array, f_k: Array, P_k: Array, p_k: Array
) -> tuple[Array, Array, Array]:
    """Single forward LQR step - JIT compiled."""
    # Compute control
    u_k = d_k - K_k @ x_k

    # Simulate dynamics
    x_next = f_k + A_k @ x_k + B_k @ u_k

    # Compute dual variables
    y_k = P_k @ x_k + p_k

    return u_k, x_next, y_k


# REPLACE _tvlqr_forward_pass with this (NO @jax.jit decorator):


def _tvlqr_forward_pass(
    nx: list[int],
    nu: list[int],
    num_horizon: int,
    A: list[Array],
    B: list[Array],
    f: list[Array],
    K: list[Array],
    d: list[Array],
    P: list[Array],
    p: list[Array],
    x0: Array,
) -> tuple[list[Array], list[Array], list[Array]]:
    """Forward pass for TVLQR with JIT-compiled computational kernels."""
    N = num_horizon

    # Initialize trajectories (Python control flow - not JIT compiled)
    x = [jnp.zeros(nx[k]) for k in range(N + 1)]
    u = [jnp.zeros(nu[k]) for k in range(N)]
    y = [jnp.zeros(nx[k]) for k in range(N + 1)]

    # Set initial state
    x[0] = x0

    # Forward simulate (Python loop with JIT-compiled steps)
    for k in range(N):
        # JIT-compiled computational step
        u[k], x[k + 1], y[k] = _lqr_step_forward(x[k], K[k], d[k], A[k], B[k], f[k], P[k], p[k])

    # Terminal dual
    y[N] = P[N] @ x[N] + p[N]

    return x, u, y


def tvlqr_backward_pass(
    nx: list[int],
    nu: list[int],
    num_horizon: int,
    A: list[Array],
    B: list[Array],
    f: list[Array],
    Q: list[Array],
    R: list[Array],
    H: list[Array],
    q: list[Array],
    r: list[Array],
    reg: Float,
    is_diag: bool = False,
) -> tuple[list[Array], list[Array], list[Array], list[Array], Array, int]:
    """Perform backward pass for time-varying LQR problem.

    Args:
        nx: State dimensions at each time step
        nu: Control dimensions at each time step
        num_horizon: Number of time steps
        A: Dynamics Jacobian matrices (wrt state)
        B: Dynamics Jacobian matrices (wrt control)
        f: Affine dynamics terms
        Q: State cost matrices
        R: Control cost matrices
        H: Cross-term cost matrices
        q: Linear state cost terms
        r: Linear control cost terms
        reg: Regularization parameter
        is_diag: Whether cost matrices are diagonal

    Returns:
        Tuple of (K, d, P, p, delta_V, status_code)
        K: Feedback gain matrices
        d: Feedforward terms
        P: Cost-to-go matrices
        p: Cost-to-go vectors
        delta_V: Expected cost reduction
        status_code: Success/failure indicator (TVLQR_SUCCESS or knot point index)
    """
    return _tvlqr_backward_pass(nx, nu, num_horizon, A, B, f, Q, R, H, q, r, reg, is_diag)


def tvlqr_forward_pass(
    nx: list[int],
    nu: list[int],
    num_horizon: int,
    A: list[Array],
    B: list[Array],
    f: list[Array],
    K: list[Array],
    d: list[Array],
    P: list[Array],
    p: list[Array],
    x0: Array,
) -> tuple[list[Array], list[Array], list[Array]]:
    """Perform forward pass for time-varying LQR problem.

    Args:
        nx: State dimensions at each time step
        nu: Control dimensions at each time step
        num_horizon: Number of time steps
        A: Dynamics Jacobian matrices (wrt state)
        B: Dynamics Jacobian matrices (wrt control)
        f: Affine dynamics terms
        K: Feedback gain matrices
        d: Feedforward terms
        P: Cost-to-go matrices
        p: Cost-to-go vectors
        x0: Initial state

    Returns:
        Tuple of (x, u, y)
        x: State trajectory
        u: Control trajectory
        y: Dual variable trajectory
    """
    return _tvlqr_forward_pass(nx, nu, num_horizon, A, B, f, K, d, P, p, x0)


def tvlqr_total_mem_size(
    nx: list[int], nu: list[int], num_horizon: int, is_diag: bool = False
) -> int:
    """Calculate total memory size needed for TVLQR data structures.

    Args:
        nx: State dimensions at each time step
        nu: Control dimensions at each time step
        num_horizon: Number of time steps
        is_diag: Whether cost matrices are diagonal

    Returns:
        Total memory size in bytes
    """
    mem_size = 0

    for k in range(num_horizon + 1):
        n = nx[k]
        mem_size += n if is_diag else n * n  # Q
        mem_size += n  # q
        mem_size += n * n  # P
        mem_size += n  # p
        mem_size += n  # x
        mem_size += n  # y

        if k < num_horizon:
            m = nu[k]
            mem_size += n * n  # A
            mem_size += n * m  # B
            mem_size += n  # f

            mem_size += m if is_diag else m * m  # R
            mem_size += 0 if is_diag else m * n  # H
            mem_size += m  # r

            mem_size += m * n  # K
            mem_size += m  # d

            # Temporary matrices for backward pass
            mem_size += n * n  # Qxx
            mem_size += m * m  # Quu
            mem_size += m * n  # Qux
            mem_size += n  # Qx
            mem_size += m  # Qu

            mem_size += n * n  # Qxx_tmp
            mem_size += m * m  # Quu_tmp
            mem_size += m * n  # Qux_tmp
            mem_size += n  # Qx_tmp
            mem_size += m  # Qu_tmp

            mem_size += m  # u

    mem_size += 2  # delta_V

    return mem_size * 8  # 8 bytes per float64
