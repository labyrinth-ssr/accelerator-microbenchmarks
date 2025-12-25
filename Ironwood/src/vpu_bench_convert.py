"""Benchmark type conversion operations on VPU.

Operations: convert_element_type, bitcast_convert_type
"""

import jax.lax as lax
import jax.numpy as jnp
from vpu_bench_utils import benchmark_primitive, benchmark_primitive_with_hlo, generate_shapes, print_results


def create_convert_func(target_dtype):
    """Create a type conversion function."""
    def convert_wrapper(x):
        return lax.convert_element_type(x, target_dtype)
    return convert_wrapper


def benchmark_all_convert_ops(
    shapes=None,
    num_warmup=3,
    num_runs=10,
    device_id=0,
    dimension=2,
    hlo_dump_dir=None,
):
    """Benchmark type conversion operations.

    Tests conversions between common types:
    - bf16 <-> fp32
    - bf16 <-> fp16
    - fp32 <-> fp16
    - bf16/fp32/fp16 <-> fp8_e4m3fn
    - bf16/fp32/fp16 <-> fp8_e5m2
    """
    if shapes is None:
        shapes = generate_shapes(dimension=dimension, min_size=1024, max_size=16384, step=1024)

    # Define conversion pairs (source_dtype, target_dtype)
    conversion_pairs = [
        # BF16 conversions
        (jnp.bfloat16, jnp.float32, "bf16_to_fp32"),
        (jnp.float32, jnp.bfloat16, "fp32_to_bf16"),
        (jnp.bfloat16, jnp.float16, "bf16_to_fp16"),
        (jnp.float16, jnp.bfloat16, "fp16_to_bf16"),

        # FP32 conversions
        (jnp.float32, jnp.float16, "fp32_to_fp16"),
        (jnp.float16, jnp.float32, "fp16_to_fp32"),

        # FP8 conversions (e4m3fn)
        (jnp.bfloat16, jnp.float8_e4m3fn, "bf16_to_fp8_e4m3fn"),
        (jnp.float8_e4m3fn, jnp.bfloat16, "fp8_e4m3fn_to_bf16"),
        (jnp.float32, jnp.float8_e4m3fn, "fp32_to_fp8_e4m3fn"),
        (jnp.float8_e4m3fn, jnp.float32, "fp8_e4m3fn_to_fp32"),

        # FP8 conversions (e5m2)
        (jnp.bfloat16, jnp.float8_e5m2, "bf16_to_fp8_e5m2"),
        (jnp.float8_e5m2, jnp.bfloat16, "fp8_e5m2_to_bf16"),
        (jnp.float32, jnp.float8_e5m2, "fp32_to_fp8_e5m2"),
        (jnp.float8_e5m2, jnp.float32, "fp8_e5m2_to_fp32"),

        # Integer conversions
        (jnp.bfloat16, jnp.int32, "bf16_to_int32"),
        (jnp.int32, jnp.bfloat16, "int32_to_bf16"),
        (jnp.float32, jnp.int32, "fp32_to_int32"),
        (jnp.int32, jnp.float32, "int32_to_fp32"),
    ]

    all_results = {}

    for source_dtype, target_dtype, op_name in conversion_pairs:
        print(f"\nBenchmarking {op_name}...")

        convert_func = create_convert_func(target_dtype)

        try:
            results = benchmark_primitive(
                convert_func,
                shapes,
                dtype=source_dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=1,
                device_id=device_id,
            )
            all_results[op_name] = results
            print_results(op_name, results)
        except Exception as e:
            print(f"Error benchmarking {op_name}: {e}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark VPU type conversion operations")
    parser.add_argument("--min_size", type=int, default=1024, help="Minimum size")
    parser.add_argument("--max_size", type=int, default=16384, help="Maximum size")
    parser.add_argument("--step", type=int, default=1024, help="Size step")
    parser.add_argument("--dimension", type=int, default=2, choices=[1, 2], help="1D or 2D data")
    parser.add_argument("--num_warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")
    parser.add_argument("--hlo_dump_dir", type=str, default=None, help="Directory to save HLO dumps (optional)")

    args = parser.parse_args()

    shapes = generate_shapes(args.dimension, args.min_size, args.max_size, args.step)

    results = benchmark_all_convert_ops(
        shapes=shapes,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device_id=args.device_id,
        dimension=args.dimension,
        hlo_dump_dir=args.hlo_dump_dir,
    )