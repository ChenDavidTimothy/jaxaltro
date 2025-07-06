"""Double integrator example demonstrating JAX-based ALTRO trajectory optimization.

This example replicates the C++ double integrator test case, showing how to:
1. Set up a trajectory optimization problem
2. Define dynamics functions (Jacobians computed automatically)
3. Add constraints (Jacobians computed automatically)
4. Solve the optimization problem
5. Extract and analyze results

The problem is to control a 2D double integrator from an initial state to the origin
while minimizing control effort.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import Array

from jaxaltro import AltroOptions, ALTROSolver, ConstraintType, Verbosity
from jaxaltro.types import Float


def create_double_integrator_dynamics(dim: int = 2) -> callable:
    """Create double integrator dynamics function.

    State: [position, velocity] (2*dim dimensional)
    Input: [acceleration] (dim dimensional)

    Dynamics: x_{k+1} = A*x_k + B*u_k
    where A = [[I, h*I], [0, I]], B = [[0.5*h^2*I], [h*I]]

    Args:
        dim: Spatial dimension (2 for 2D double integrator)

    Returns:
        Dynamics function (Jacobian computed automatically by JAX)
    """

    @jax.jit
    def dynamics_function(x: Array, u: Array, h: Float) -> Array:
        """Double integrator dynamics: x_{k+1} = f(x_k, u_k, h)."""

        # Extract position and velocity
        pos = x[:dim]
        vel = x[dim:]

        # Integrate dynamics
        pos_next = pos + vel * h + 0.5 * u * h**2
        vel_next = vel + u * h

        return jnp.concatenate([pos_next, vel_next])

    # Note: No manual Jacobian needed! JAX computes it automatically
    return dynamics_function


def create_goal_constraint(x_goal: Array) -> callable:
    """Create goal constraint function.

    Args:
        x_goal: Target state

    Returns:
        Constraint function (Jacobian computed automatically by JAX)
    """

    @jax.jit
    def constraint_function(x: Array, u: Array) -> Array:
        """Goal constraint: x - x_goal = 0."""
        return x - x_goal

    # Note: No manual Jacobian needed! JAX computes it automatically
    return constraint_function


def create_lqr_cost_function(
    Q_diag: Array, R_diag: Array, x_ref: Array, u_ref: Array, is_terminal: bool = False
) -> callable:
    """Create LQR cost function.

    Args:
        Q_diag: Diagonal state penalty matrix
        R_diag: Diagonal input penalty matrix
        x_ref: State reference
        u_ref: Input reference
        is_terminal: Whether this is a terminal cost

    Returns:
        Cost function (gradients/Hessians computed automatically by JAX)
    """

    @jax.jit
    def cost_function(x: Array, u: Array) -> Float:
        """LQR cost: (x-xref)^T Q (x-xref) + (u-uref)^T R (u-uref)."""
        dx = x - x_ref
        cost = 0.5 * jnp.dot(dx, Q_diag * dx)

        if not is_terminal and len(u) > 0:
            du = u - u_ref
            cost += 0.5 * jnp.dot(du, R_diag * du)

        return cost

    return cost_function


def solve_double_integrator_example():
    """Solve the double integrator trajectory optimization problem."""

    # Problem parameters
    dim = 2  # 2D problem
    n = 2 * dim  # State dimension: [x, y, x_dot, y_dot]
    m = dim  # Input dimension: [u_x, u_y]

    # Time horizon
    tf = 5.0
    num_segments = 50
    h = tf / num_segments

    # Initial and goal states
    x0 = jnp.array([1.0, 1.0, 0.0, 0.0])  # Start at (1,1) with zero velocity
    x_goal = jnp.zeros(n)  # Goal at origin with zero velocity

    # Cost function weights
    Q_diag = jnp.ones(n)  # State penalty
    R_diag = 1e-2 * jnp.ones(m)  # Input penalty
    Qf_diag = 100.0 * jnp.ones(n)  # Terminal state penalty

    print("Setting up double integrator trajectory optimization...")
    print(f"  State dimension: {n}")
    print(f"  Input dimension: {m}")
    print(f"  Time horizon: {tf} seconds")
    print(f"  Number of segments: {num_segments}")
    print(f"  Time step: {h:.3f} seconds")
    print(f"  Initial state: {x0}")
    print(f"  Goal state: {x_goal}")

    # Create solver
    solver = ALTROSolver(num_segments)

    # Set dimensions
    solver.set_dimension(n, m)
    solver.set_time_step(h)

    # Set dynamics - ONLY the function, Jacobian computed automatically!
    dynamics_func = create_double_integrator_dynamics(dim)
    solver.set_explicit_dynamics(dynamics_func)

    # Set cost functions using automatic differentiation
    # Running cost
    running_cost = create_lqr_cost_function(Q_diag, R_diag, x_goal, jnp.zeros(m), is_terminal=False)
    solver.set_cost_function(running_cost, k_start=0, k_stop=num_segments)

    # Terminal cost
    terminal_cost = create_lqr_cost_function(
        Qf_diag, jnp.array([]), x_goal, jnp.array([]), is_terminal=True
    )
    solver.set_cost_function(terminal_cost, k_start=num_segments, k_stop=num_segments + 1)

    # Set goal constraint - ONLY the function, Jacobian computed automatically!
    goal_constraint_func = create_goal_constraint(x_goal)
    solver.set_constraint(
        goal_constraint_func,
        n,
        ConstraintType.EQUALITY,
        "Goal Constraint",
        k_start=num_segments,
        k_stop=num_segments + 1,
    )

    # Set initial state
    solver.set_initial_state(x0, n)

    # Initialize solver
    print("\nInitializing solver...")
    solver.initialize()

    # Set solver options
    opts = solver.get_options()
    opts = AltroOptions(
        verbose=Verbosity.OUTER,
        iterations_max=100,
        tol_stationarity=1e-6,
        tol_primal_feasibility=1e-6,
    )
    solver.set_options(opts)

    # Solve
    print("\nSolving trajectory optimization problem...")
    start_time = time.time()
    status = solver.solve()
    solve_time = time.time() - start_time

    # Print results
    print(f"\nSolve completed in {solve_time:.3f} seconds")
    print(f"Status: {status}")
    print(f"Iterations: {solver.get_iterations()}")
    print(f"Final cost: {solver.get_final_objective():.6f}")
    print(f"Primal feasibility: {solver.get_primal_feasibility():.2e}")

    # Extract trajectory
    print("\nFinal trajectory:")
    x_final = solver.get_state(num_segments)
    print(f"  Final state: {x_final}")
    print(f"  Distance to goal: {jnp.linalg.norm(x_final - x_goal):.6f}")

    # Print some intermediate states
    print("\nTrajectory samples:")
    for k in [0, num_segments // 4, num_segments // 2, 3 * num_segments // 4, num_segments]:
        x_k = solver.get_state(k)
        t_k = k * h
        print(f"  t={t_k:.2f}: x={x_k[:2]}, v={x_k[2:]}")

    # Validate solution
    initial_distance = jnp.linalg.norm(x0 - x_goal)
    final_distance = jnp.linalg.norm(x_final - x_goal)

    print("\nValidation:")
    print(f"  Initial distance to goal: {initial_distance:.6f}")
    print(f"  Final distance to goal: {final_distance:.6f}")
    print(f"  Improvement: {initial_distance - final_distance:.6f}")

    success = (
        final_distance < 1e-4
        and solver.get_primal_feasibility() < 1e-4
        and status.value == "Success"
    )

    print(f"  Success: {success}")

    return solver


def demonstrate_automatic_differentiation():
    """Demonstrate the automatic differentiation capabilities."""

    print("\n" + "=" * 60)
    print("AUTOMATIC DIFFERENTIATION DEMONSTRATION")
    print("=" * 60)

    # Create dynamics function
    dynamics_func = create_double_integrator_dynamics(dim=2)

    # Test point
    x_test = jnp.array([1.0, 2.0, 0.5, -0.3])
    u_test = jnp.array([0.1, -0.2])
    h_test = 0.1

    print("\nDynamics function evaluation:")
    print(f"x = {x_test}")
    print(f"u = {u_test}")
    print(f"h = {h_test}")

    # Evaluate dynamics
    x_next = dynamics_func(x_test, u_test, h_test)
    print(f"x_next = f(x, u, h) = {x_next}")

    # Compute Jacobian automatically using JAX
    def dynamics_combined(xu):
        n = len(x_test)
        x_part, u_part = xu[:n], xu[n:]
        return dynamics_func(x_part, u_part, h_test)

    jacobian_func = jax.jacobian(dynamics_combined)
    xu_combined = jnp.concatenate([x_test, u_test])
    jacobian = jacobian_func(xu_combined)

    print("\nJacobian (computed automatically):")
    print(f"Shape: {jacobian.shape}")
    print(f"df/dx:\n{jacobian[:, :4]}")
    print(f"df/du:\n{jacobian[:, 4:]}")

    # Show cost function automatic differentiation
    cost_func = create_lqr_cost_function(
        Q_diag=jnp.ones(4), R_diag=0.01 * jnp.ones(2), x_ref=jnp.zeros(4), u_ref=jnp.zeros(2)
    )

    print("\nCost function evaluation:")
    cost_val = cost_func(x_test, u_test)
    print(f"J(x, u) = {cost_val}")

    # Compute gradients automatically
    grad_x = jax.grad(cost_func, argnums=0)(x_test, u_test)
    grad_u = jax.grad(cost_func, argnums=1)(x_test, u_test)

    print(f"∇_x J = {grad_x}")
    print(f"∇_u J = {grad_u}")

    # Compute Hessians automatically
    hess_xx = jax.hessian(cost_func, argnums=0)(x_test, u_test)
    hess_uu = jax.hessian(cost_func, argnums=1)(x_test, u_test)

    print(f"∇²_xx J shape: {hess_xx.shape}")
    print(f"∇²_uu J shape: {hess_uu.shape}")

    print("\n✅ All derivatives computed automatically by JAX!")
    print("   No manual derivative implementations required!")


if __name__ == "__main__":
    """Run the double integrator example with automatic differentiation."""

    print("JAX-based ALTRO Double Integrator Example")
    print("With Automatic Differentiation")
    print("=" * 50)

    try:
        solver = solve_double_integrator_example()
        print("\nExample completed successfully!")

        # Demonstrate automatic differentiation
        demonstrate_automatic_differentiation()

    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback

        traceback.print_exc()
