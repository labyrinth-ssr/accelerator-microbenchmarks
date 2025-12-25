"""Benchmark elementwise unary operations on VPU.

Operations: neg, abs, sqrt, rsqrt, exp, log, sin, cos, tanh, sigmoid, erf, etc.
"""

import jax.lax as lax
import jax.numpy as jnp
from vpu_bench_utils import benchmark_primitive, benchmark_primitive_with_hlo, generate_shapes, print_results


def benchmark_all_unary_ops(
    shapes=None,
    dtype=jnp.bfloat16,
    num_warmup=3,
    num_runs=10,
    device_id=0,
    dimension=2,
    hlo_dump_dir=None,
):
    """Benchmark all elementwise unary operations."""
    if shapes is None:
        shapes = generate_shapes(dimension=dimension, min_size=1024, max_size=16384, step=1024)

    # Define unary operations
    unary_ops = {
        # Basic arithmetic
        "neg": lax.neg,
        "abs": lax.abs,
        "sign": lax.sign,

        # Square roots
        "sqrt": lax.sqrt,
        "rsqrt": lax.rsqrt,
        "cbrt": lax.cbrt,

        # Exponentials and logarithms
        "exp": lax.exp,
        "exp2": lax.exp2,
        "expm1": lax.expm1,
        "log": lax.log,
        "log1p": lax.log1p,

        # Trigonometric
        "sin": lax.sin,
        "cos": lax.cos,
        "tan": lax.tan,
        "asin": lax.asin,
        "acos": lax.acos,
        "atan": lax.atan,

        # Hyperbolic
        "sinh": lax.sinh,
        "cosh": lax.cosh,
        "tanh": lax.tanh,
        "asinh": lax.asinh,
        "acosh": lax.acosh,
        "atanh": lax.atanh,

        # Special functions
        "erf": lax.erf,
        "erfc": lax.erfc,
        "erf_inv": lax.erf_inv,
        "logistic": lax.logistic,  # sigmoid

        # Rounding
        "ceil": lax.ceil,
        "floor": lax.floor,
        "round": lax.round,

        # Reciprocal
        "reciprocal": lambda x: lax.div(lax.full_like(x, 1.0), x),

        # Special math functions
        "lgamma": lax.lgamma,
        "digamma": lax.digamma,
        "bessel_i0e": lax.bessel_i0e,
        "bessel_i1e": lax.bessel_i1e,
    }

    all_results = {}

    for op_name, op_func in unary_ops.items():
        print(f"\nBenchmarking {op_name}...")
        try:
            results = benchmark_primitive(
                op_func,
                shapes,
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=1,  # Unary operations need 1 input
                device_id=device_id,
            )
            all_results[op_name] = results
            print_results(op_name, results)
        except Exception as e:
            print(f"Error benchmarking {op_name}: {e}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark VPU unary operations")
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

    results = benchmark_all_unary_ops(
        shapes=shapes,
        dtype=dtype,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device_id=args.device_id,
        dimension=args.dimension,
        hlo_dump_dir=args.hlo_dump_dir,
    )