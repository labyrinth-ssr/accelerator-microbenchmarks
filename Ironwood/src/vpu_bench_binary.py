"""Benchmark elementwise binary operations on VPU.

Operations: add, sub, mul, div, pow, max, min, rem, atan2, nextafter
"""

import jax.lax as lax
import jax.numpy as jnp
from vpu_bench_utils import benchmark_primitive, benchmark_primitive_with_hlo, generate_shapes, print_results


def benchmark_all_binary_ops(
    shapes=None,
    dtype=jnp.bfloat16,
    num_warmup=3,
    num_runs=10,
    device_id=0,
    dimension=2,
    hlo_dump_dir=None,
):
    """Benchmark all elementwise binary operations."""
    if shapes is None:
        shapes = generate_shapes(dimension=dimension, min_size=1024, max_size=16384, step=1024)

    # Define binary operations
    binary_ops = {
        "add": lax.add,
        "sub": lax.sub,
        "mul": lax.mul,
        "div": lax.div,
        "rem": lax.rem,
        "pow": lax.pow,
        "max": lax.max,
        "min": lax.min,
        "atan2": lax.atan2,
        "nextafter": lax.nextafter,
    }

    all_results = {}

    for op_name, op_func in binary_ops.items():
        print(f"\nBenchmarking {op_name}...")

        if hlo_dump_dir:
            results = benchmark_primitive_with_hlo(
                op_func,
                op_name,
                shapes,
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=2,  # Binary operations need 2 inputs
                device_id=device_id,
                hlo_dump_dir=hlo_dump_dir,
            )
        else:
            results = benchmark_primitive(
                op_func,
                shapes,
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=2,  # Binary operations need 2 inputs
                device_id=device_id,
            )

        all_results[op_name] = results
        print_results(op_name, results)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark VPU binary operations")
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

    results = benchmark_all_binary_ops(
        shapes=shapes,
        dtype=dtype,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device_id=args.device_id,
        dimension=args.dimension,
        hlo_dump_dir=args.hlo_dump_dir,
    )