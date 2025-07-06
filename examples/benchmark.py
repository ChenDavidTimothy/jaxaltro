"""Performance benchmarking wrapper for ALTRO examples."""

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
    for i in range(warmup_runs):
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


def benchmark_scaling():
    """Benchmark different problem sizes."""
    import jax.numpy as jnp
    from example_double_integrator import create_double_integrator_dynamics

    from jaxaltro import ALTROSolver, ConstraintType

    segments_list = [25, 50, 100, 200]
    results = {}

    for num_segments in segments_list:

        def solver_factory():
            # Create scaled problem
            solver = ALTROSolver(num_segments)
            solver.set_dimension(4, 2)  # 2D double integrator
            solver.set_time_step(5.0 / num_segments)

            # Set dynamics - ONLY the function (fixed API)
            dynamics_func = create_double_integrator_dynamics(2)
            solver.set_explicit_dynamics(dynamics_func)

            # Set costs using built-in LQR cost (more efficient than custom functions)
            Q_diag = jnp.ones(4)
            R_diag = 1e-2 * jnp.ones(2)
            Qf_diag = 100.0 * jnp.ones(4)
            x_goal = jnp.zeros(4)

            solver.set_lqr_cost(
                4, 2, Q_diag, R_diag, x_goal, jnp.zeros(2), k_start=0, k_stop=num_segments
            )

            # Terminal cost
            solver.set_lqr_cost(
                4,
                0,
                Qf_diag,
                jnp.array([]),
                x_goal,
                jnp.array([]),
                k_start=num_segments,
                k_stop=num_segments + 1,
            )

            # Goal constraint
            def goal_constraint(x, u):
                return x - x_goal

            solver.set_constraint(
                goal_constraint,
                4,
                ConstraintType.EQUALITY,
                "Goal",
                k_start=num_segments,
                k_stop=num_segments + 1,
            )

            # Initial state and initialize
            x0 = jnp.array([1.0, 1.0, 0.0, 0.0])
            solver.set_initial_state(x0, 4)
            solver.initialize()

            return solver

        results[num_segments] = benchmark_solver(
            solver_factory,
            f"Double Integrator ({num_segments} segments)",
            warmup_runs=1,
            timing_runs=3,
        )

    # Print scaling analysis
    print("\n=== Scaling Analysis ===")
    print(f"{'Segments':<10} {'Time (s)':<10} {'Time/Seg':<12}")
    for segments, result in results.items():
        time_per_seg = result["solver_mean"] / segments * 1000  # ms per segment
        print(f"{segments:<10} {result['solver_mean']:<10.3f} {time_per_seg:<12.2f} ms")

    return results


if __name__ == "__main__":
    # Single benchmark
    benchmark_double_integrator()

    # Scaling study
    benchmark_scaling()
