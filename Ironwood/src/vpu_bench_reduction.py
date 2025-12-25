"""Benchmark reduction operations on VPU.

Operations: reduce_sum, reduce_max, reduce_min, reduce_prod, reduce_and, reduce_or
"""

import jax.lax as lax
import jax.numpy as jnp
from vpu_bench_utils import benchmark_primitive, benchmark_primitive_with_hlo, generate_shapes, print_results


def create_reduction_func(reduce_op, axis=None):
    """Create a reduction function wrapper."""
    def reduction_wrapper(x):
        if axis is None:
            # Reduce over all dimensions
            return reduce_op(x, list(range(len(x.shape))))
        else:
            return reduce_op(x, axis)
    return reduction_wrapper


def benchmark_all_reduction_ops(
    shapes=None,
    dtype=jnp.bfloat16,
    num_warmup=3,
    num_runs=10,
    device_id=0,
    reduction_axes=None,
    dimension=2,
    hlo_dump_dir=None,
):
    """Benchmark all reduction operations.

    Args:
        shapes: List of input shapes to test
        dtype: Data type to use
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        device_id: Device to run on
        reduction_axes: Axes to reduce over. If None, reduce over all axes.
                       Can be int, tuple of ints, or list of tuples for different configs.
    """
    if shapes is None:
        shapes = generate_shapes(dimension=dimension, min_size=1024, max_size=16384, step=1024)

    # Define reduction operations
    reduction_ops = {
        "reduce_sum": lax.reduce_sum,
        "reduce_max": lax.reduce_max,
        "reduce_min": lax.reduce_min,
        "reduce_prod": lax.reduce_prod,
        "reduce_and": lax.reduce_and,
        "reduce_or": lax.reduce_or,
        "reduce_xor": lax.reduce_xor,
    }

    all_results = {}

    # Test different reduction configurations
    if reduction_axes is None:
        if dimension == 1:
            reduction_configs = [
                ("all_axes", None),
                ("axis_0", 0),
            ]
        else:  # dimension == 2
            reduction_configs = [
                ("all_axes", None),
                ("axis_0", 0),
                ("axis_1", 1),
            ]
    elif isinstance(reduction_axes, (int, tuple)):
        reduction_configs = [("custom", reduction_axes)]
    else:
        reduction_configs = reduction_axes

    for op_name, op_func in reduction_ops.items():
        for config_name, axes in reduction_configs:
            full_op_name = f"{op_name}_{config_name}"
            print(f"\nBenchmarking {full_op_name}...")

            reduction_func = create_reduction_func(op_func, axes)

            try:
                results = benchmark_primitive(
                    reduction_func,
                    shapes,
                    dtype=dtype,
                    num_warmup=num_warmup,
                    num_runs=num_runs,
                    num_inputs=1,
                    device_id=device_id,
                )
                all_results[full_op_name] = results
                print_results(full_op_name, results)
            except Exception as e:
                print(f"Error benchmarking {full_op_name}: {e}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark VPU reduction operations")
    parser.add_argument("--min_size", type=int, default=1024, help="Minimum size")
    parser.add_argument("--max_size", type=int, default=16384, help="Maximum size")
    parser.add_argument("--step", type=int, default=1024, help="Size step")
    parser.add_argument("--dimension", type=int, default=2, choices=[1, 2], help="1D or 2D data")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--num_warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--hlo_dump_dir", type=str, default=None, help="Directory to save HLO dumps (optional)")

    args = parser.parse_args()

    shapes = generate_shapes(args.dimension, args.min_size, args.max_size, args.step)
    dtype = getattr(jnp, args.dtype)

    results = benchmark_all_reduction_ops(
        shapes=shapes,
        dtype=dtype,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device_id=args.device_id,
        dimension=args.dimension,
        hlo_dump_dir=args.hlo_dump_dir,
    )