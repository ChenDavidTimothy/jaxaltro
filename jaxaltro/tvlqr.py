from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from .types import Float


# Success code matching C++ TVLQR_SUCCESS
TVLQR_SUCCESS = -1


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

        # Action-value function expansion
        # Qxx = Q + A^T P A
        AtP = A_k.T @ P_next
        Qxx = Q_k + AtP @ A_k

        # Quu = R + B^T P B
        BtP = B_k.T @ P_next
        Quu = R_k + BtP @ B_k

        # Qux = H + B^T P A
        Qux = H_k + BtP @ A_k

        # Qx = q + A^T (P f + p)
        Ppf = P_next @ f_k + p_next
        Qx = q_k + A_k.T @ Ppf

        # Qu = r + B^T (P f + p)
        Qu = r_k + B_k.T @ Ppf

        # Compute gains with regularization
        Quu_reg = Quu + reg * jnp.eye(m)

        # Check for positive definiteness via Cholesky
        try:
            Quu_chol = jsp.linalg.cholesky(Quu_reg, lower=True)
        except Exception:
            # Return failure code (knot point index where failure occurred)
            return K, d, P, p, delta_V, k

        # Solve for gains
        K_k = jsp.linalg.solve_triangular(Quu_chol, Qux, lower=True)
        K_k = jsp.linalg.solve_triangular(Quu_chol.T, K_k, lower=False)

        d_k = jsp.linalg.solve_triangular(Quu_chol, -Qu, lower=True)
        d_k = jsp.linalg.solve_triangular(Quu_chol.T, d_k, lower=False)

        K[k] = K_k
        d[k] = d_k

        # Compute cost-to-go
        # P = Qxx + K^T Quu K - K^T Qux - Qux^T K
        KtQuu = K_k.T @ Quu
        KtQux = K_k.T @ Qux

        P[k] = Qxx + KtQuu @ K_k - KtQux - KtQux.T

        # p = Qx - K^T Quu d - K^T Qu + Qux^T d
        p[k] = Qx - KtQuu @ d_k - K_k.T @ Qu + Qux.T @ d_k

        # Expected cost reduction
        Quu_d = Quu @ d_k
        delta_V = delta_V.at[0].add(d_k.T @ Qu)
        delta_V = delta_V.at[1].add(0.5 * d_k.T @ Quu_d)

    return K, d, P, p, delta_V, TVLQR_SUCCESS


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
    N = num_horizon

    # Initialize trajectories
    x = [jnp.zeros(nx[k]) for k in range(N + 1)]
    u = [jnp.zeros(nu[k]) for k in range(N)]
    y = [jnp.zeros(nx[k]) for k in range(N + 1)]

    # Set initial state
    x[0] = x0

    # Forward simulate
    for k in range(N):
        # Compute control
        u[k] = d[k] - K[k] @ x[k]

        # Simulate dynamics
        x[k + 1] = f[k] + A[k] @ x[k] + B[k] @ u[k]

        # Compute dual variables
        y[k] = P[k] @ x[k] + p[k]

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
