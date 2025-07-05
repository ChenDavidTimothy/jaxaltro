"""Double integrator example demonstrating JAX-based ALTRO trajectory optimization.

This example replicates the C++ double integrator test case, showing how to:
1. Set up a trajectory optimization problem
2. Define dynamics and cost functions
3. Add constraints
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

from jaxaltro import ALTROSolver, ConstraintType, Verbosity
from jaxaltro.types import Float


def create_double_integrator_dynamics(dim: int = 2) -> tuple[callable, callable]:
    """Create double integrator dynamics functions.

    State: [position, velocity] (2*dim dimensional)
    Input: [acceleration] (dim dimensional)

    Dynamics: x_{k+1} = A*x_k + B*u_k
    where A = [[I, h*I], [0, I]], B = [[0.5*h^2*I], [h*I]]

    Args:
        dim: Spatial dimension (2 for 2D double integrator)

    Returns:
        Tuple of (dynamics_function, dynamics_jacobian)
    """

    @jax.jit
    def dynamics_function(x: Array, u: Array, h: Float) -> Array:
        """Double integrator dynamics: x_{k+1} = f(x_k, u_k, h)."""
        2 * dim  # State dimension

        # Extract position and velocity
        pos = x[:dim]
        vel = x[dim:]

        # Integrate dynamics
        pos_next = pos + vel * h + 0.5 * u * h**2
        vel_next = vel + u * h

        return jnp.concatenate([pos_next, vel_next])

    @jax.jit
    def dynamics_jacobian(x: Array, u: Array, h: Float) -> Array:
        """Jacobian of double integrator dynamics."""
        n = 2 * dim  # State dimension
        m = dim  # Input dimension

        # Jacobian is constant for linear dynamics
        # J = [A, B] where A is (n,n) and B is (n,m)

        # A matrix: [[I, h*I], [0, I]]
        A = jnp.zeros((n, n))
        A = A.at[:dim, :dim].set(jnp.eye(dim))  # I
        A = A.at[:dim, dim:].set(h * jnp.eye(dim))  # h*I
        A = A.at[dim:, dim:].set(jnp.eye(dim))  # I

        # B matrix: [[0.5*h^2*I], [h*I]]
        B = jnp.zeros((n, m))
        B = B.at[:dim, :].set(0.5 * h**2 * jnp.eye(dim))  # 0.5*h^2*I
        B = B.at[dim:, :].set(h * jnp.eye(dim))  # h*I

        return jnp.hstack([A, B])

    return dynamics_function, dynamics_jacobian


def create_goal_constraint(x_goal: Array) -> tuple[callable, callable]:
    """Create goal constraint functions.

    Args:
        x_goal: Target state

    Returns:
        Tuple of (constraint_function, constraint_jacobian)
    """

    @jax.jit
    def constraint_function(x: Array, u: Array) -> Array:
        """Goal constraint: x - x_goal = 0."""
        return x - x_goal

    @jax.jit
    def constraint_jacobian(x: Array, u: Array) -> Array:
        """Jacobian of goal constraint."""
        n = x.shape[0]
        m = u.shape[0]

        # Jacobian is [I, 0] (identity wrt x, zero wrt u)
        jac = jnp.zeros((n, n + m))
        jac = jac.at[:, :n].set(jnp.eye(n))

        return jac

    return constraint_function, constraint_jacobian


def solve_double_integrator_example() -> None:
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

    # Set dynamics
    dynamics_func, dynamics_jac = create_double_integrator_dynamics(dim)
    solver.set_explicit_dynamics(dynamics_func, dynamics_jac)

    # Set cost function
    solver.set_lqr_cost(n, m, Q_diag, R_diag, x_goal, jnp.zeros(m), k_start=0, k_stop=num_segments)

    # Set terminal cost
    solver.set_lqr_cost(
        n,
        0,
        Qf_diag,
        jnp.array([]),
        x_goal,
        jnp.array([]),
        k_start=num_segments,
        k_stop=num_segments + 1,
    )

    # Set goal constraint
    goal_constraint_func, goal_constraint_jac = create_goal_constraint(x_goal)
    solver.set_constraint(
        goal_constraint_func,
        goal_constraint_jac,
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
    opts.verbose = Verbosity.OUTER
    opts.iterations_max = 100
    opts.tol_stationarity = 1e-6
    opts.tol_primal_feasibility = 1e-6
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


if __name__ == "__main__":
    """Run the double integrator example."""

    print("JAX-based ALTRO Double Integrator Example")
    print("=" * 50)

    try:
        solver = solve_double_integrator_example()
        print("\nExample completed successfully!")

        # Optionally print trajectories
        print("\nTo visualize results, you can extract the full trajectory:")
        print("  states = [solver.get_state(k) for k in range(solver.get_horizon_length() + 1)]")
        print("  inputs = [solver.get_input(k) for k in range(solver.get_horizon_length())]")

    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback

        traceback.print_exc()
