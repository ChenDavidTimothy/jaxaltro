"""Performance benchmarking wrapper for ALTRO examples with pure JAX autodiff."""

import statistics
import time
from collections.abc import Callable
from typing import Any

import jax


def benchmark_solver(
    solver_function: Callable[[], Any], name: str, warmup_runs: int = 2, timing_runs: int = 5
) -> dict[str, float]:
    """Benchmark ALTRO solver with proper methodology.

    Args:
        solver_function: Function that returns configured solver
        name: Descriptive name for benchmark
        warmup_runs: JIT warm-up iterations
        timing_runs: Measurement iterations

    Returns:
        Dictionary with timing statistics
    """
    print(f"\n=== Benchmarking {name} ===")

    # Warm-up for JIT compilation
    print(f"Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        solver = solver_function()
        _ = solver.solve()  # JIT compilation happens here

    # Actual timing measurements
    print(f"Timing ({timing_runs} runs)...")
    total_times = []
    solver_times = []

    for i in range(timing_runs):
        # Force garbage collection
        import gc

        gc.collect()

        # Time total execution
        start_total = time.perf_counter()
        solver = solver_function()
        solver.solve()
        end_total = time.perf_counter()

        total_time = end_total - start_total
        solver_time = solver.get_solve_time_ms() / 1000.0  # Convert to seconds

        total_times.append(total_time)
        solver_times.append(solver_time)

        print(f"  Run {i + 1}: {total_time:.3f}s total, {solver_time:.3f}s solve")

    # Calculate statistics
    results = {
        "total_mean": statistics.mean(total_times),
        "total_std": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        "total_min": min(total_times),
        "total_max": max(total_times),
        "solver_mean": statistics.mean(solver_times),
        "solver_std": statistics.stdev(solver_times) if len(solver_times) > 1 else 0.0,
        "overhead_mean": statistics.mean(
            [t - s for t, s in zip(total_times, solver_times, strict=False)]
        ),
    }

    # Print summary
    print(f"\n{name} Timing Results:")
    print(f"  Total time:  {results['total_mean']:.3f} ± {results['total_std']:.3f}s")
    print(f"  Solve time:  {results['solver_mean']:.3f} ± {results['solver_std']:.3f}s")
    print(f"  Overhead:    {results['overhead_mean']:.3f}s")
    print(f"  Backend:     {jax.default_backend()}")
    print(f"  Device:      {jax.devices()[0].device_kind}")

    return results


def benchmark_double_integrator():
    """Benchmark double integrator example."""
    from example_double_integrator import solve_double_integrator_example

    # Wrapper function for consistent interface
    def solver_factory():
        return solve_double_integrator_example()

    return benchmark_solver(
        solver_factory, "Double Integrator (50 segments)", warmup_runs=2, timing_runs=5
    )


def _create_solver_factory(segments: int) -> Callable[[], Any]:
    """Create solver factory with bound segment count for JAX autodiff benchmarking."""
    import jax.numpy as jnp
    from example_double_integrator import create_double_integrator_dynamics

    from jaxaltro import ALTROSolver, ConstraintType

    def solver_factory():
        # Create scaled problem using pure JAX autodiff
        solver = ALTROSolver(segments)
        solver.set_dimension(4, 2)  # 2D double integrator
        solver.set_time_step(5.0 / segments)

        # Set dynamics - JAX computes Jacobian automatically
        dynamics_func = create_double_integrator_dynamics(2)
        solver.set_explicit_dynamics(dynamics_func)

        # Use built-in LQR cost (now uses JAX autodiff internally)
        Q_diag = jnp.ones(4)
        R_diag = 1e-2 * jnp.ones(2)
        Qf_diag = 100.0 * jnp.ones(4)
        x_goal = jnp.zeros(4)

        # All cost functions now use JAX autodiff internally
        solver.set_lqr_cost(4, 2, Q_diag, R_diag, x_goal, jnp.zeros(2), k_start=0, k_stop=segments)

        # Terminal cost
        solver.set_lqr_cost(
            4,
            0,
            Qf_diag,
            jnp.array([]),
            x_goal,
            jnp.array([]),
            k_start=segments,
            k_stop=segments + 1,
        )

        # Goal constraint - JAX computes Jacobian automatically
        @jax.jit
        def goal_constraint(x, u):
            return x - x_goal

        solver.set_constraint(
            goal_constraint,
            4,
            ConstraintType.EQUALITY,
            "Goal",
            k_start=segments,
            k_stop=segments + 1,
        )

        # Initial state and initialize
        x0 = jnp.array([1.0, 1.0, 0.0, 0.0])
        solver.set_initial_state(x0, 4)
        solver.initialize()

        return solver

    return solver_factory


def benchmark_scaling():
    """Benchmark different problem sizes with pure JAX autodiff."""
    segments_list = [25, 50, 100, 200]
    results = {}

    print("\n=== JAX Autodiff Scaling Study ===")
    print("All derivatives computed automatically by JAX")

    for num_segments in segments_list:
        solver_factory = _create_solver_factory(num_segments)

        results[num_segments] = benchmark_solver(
            solver_factory,
            f"JAX Autodiff ({num_segments} segments)",
            warmup_runs=1,
            timing_runs=3,
        )

    # Print scaling analysis
    print("\n=== JAX Autodiff Scaling Analysis ===")
    print(f"{'Segments':<10} {'Time (s)':<10} {'Time/Seg':<12} {'Speedup':<10}")

    baseline_time = None
    for segments, result in results.items():
        time_per_seg = result["solver_mean"] / segments * 1000  # ms per segment

        if baseline_time is None:
            baseline_time = result["solver_mean"]
            speedup_str = "baseline"
        else:
            speedup = baseline_time / result["solver_mean"] * (segments / 25)  # Normalized speedup
            speedup_str = f"{speedup:.2f}x"

        print(
            f"{segments:<10} {result['solver_mean']:<10.3f} {time_per_seg:<12.2f} ms {speedup_str:<10}"
        )

    return results


def compare_jax_autodiff_performance():
    """Demonstrate the performance benefits of pure JAX autodiff."""
    print("\n=== JAX AUTODIFF PERFORMANCE BENEFITS ===")
    print("1. Single JIT-compiled gradient computation")
    print("2. GPU acceleration for all derivatives")
    print("3. Optimized XLA kernels")
    print("4. No manual gradient/Hessian code duplication")
    print("5. Compile-time optimization across entire cost function")

    # Simple performance demonstration
    import jax.numpy as jnp

    # Create test cost function
    @jax.jit
    def test_cost(x, u):
        Q = jnp.diag(jnp.ones(4))
        R = jnp.diag(0.01 * jnp.ones(2))
        return 0.5 * x.T @ Q @ x + 0.5 * u.T @ R @ u

    x_test = jnp.ones(4)
    u_test = jnp.ones(2)

    # Time JAX autodiff
    start = time.perf_counter()
    for _ in range(1000):
        grad_x = jax.grad(test_cost, argnums=0)(x_test, u_test)
        grad_u = jax.grad(test_cost, argnums=1)(x_test, u_test)
    jax_time = time.perf_counter() - start

    print(f"\nJAX autodiff 1000 gradient computations: {jax_time:.4f}s")
    print(f"Per gradient: {jax_time / 1000 * 1000:.3f}ms")
    print("✅ All optimized by JAX JIT compilation")


if __name__ == "__main__":
    print("JAX-based ALTRO Performance Benchmarking")
    print("Pure JAX Automatic Differentiation")
    print("=" * 50)

    # Single benchmark
    benchmark_double_integrator()

    # Scaling study
    benchmark_scaling()

    # Performance demonstration
    compare_jax_autodiff_performance()
