"""Benchmark memory layout operations on VPU.

Operations: transpose, reshape, slice, concatenate, broadcast, pad, reverse
"""

import jax.lax as lax
import jax.numpy as jnp
from vpu_bench_utils import benchmark_primitive, benchmark_primitive_with_hlo, generate_shapes, print_results


def benchmark_all_memory_ops(
    shapes=None,
    dtype=jnp.bfloat16,
    num_warmup=3,
    num_runs=10,
    device_id=0,
    dimension=2,
    hlo_dump_dir=None,
):
    """Benchmark memory layout operations."""
    if shapes is None:
        shapes = generate_shapes(dimension=dimension, min_size=1024, max_size=16384, step=1024)

    all_results = {}

    # Transpose (only for 2D)
    if dimension == 2:
        print("\nBenchmarking transpose...")
        transpose_func = lambda x: lax.transpose(x, (1, 0))
        results = benchmark_primitive(
            transpose_func,
            shapes,
            dtype=dtype,
            num_warmup=num_warmup,
            num_runs=num_runs,
            num_inputs=1,
            device_id=device_id,
        )
        all_results["transpose"] = results
        print_results("transpose", results)

    # Reshape (square to 2x smaller matrices)
    print("\nBenchmarking reshape...")
    reshape_shapes = []
    for m, n in shapes:
        if m >= 2 and n >= 2:
            reshape_shapes.append((m, n))

    def create_reshape_func(original_shape):
        m, n = original_shape
        new_shape = (m * 2, n // 2) if n >= 2 else original_shape
        return lambda x: lax.reshape(x, new_shape)

    reshape_results = {}
    for shape in reshape_shapes:
        reshape_func = create_reshape_func(shape)
        try:
            results = benchmark_primitive(
                reshape_func,
                [shape],
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=1,
                device_id=device_id,
            )
            shape_key = "x".join(map(str, shape))
            reshape_results[shape_key] = results[shape_key]
        except Exception as e:
            print(f"Error reshaping {shape}: {e}")

    all_results["reshape"] = reshape_results
    print_results("reshape", reshape_results)

    # Slice (take middle half)
    print("\nBenchmarking slice...")
    def create_slice_func(shape):
        m, n = shape
        start_m, start_n = m // 4, n // 4
        limit_m, limit_n = 3 * m // 4, 3 * n // 4
        return lambda x: lax.slice(x, (start_m, start_n), (limit_m, limit_n))

    slice_results = {}
    for shape in shapes:
        slice_func = create_slice_func(shape)
        try:
            results = benchmark_primitive(
                slice_func,
                [shape],
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=1,
                device_id=device_id,
            )
            shape_key = "x".join(map(str, shape))
            slice_results[shape_key] = results[shape_key]
        except Exception as e:
            print(f"Error slicing {shape}: {e}")

    all_results["slice"] = slice_results
    print_results("slice", slice_results)

    # Concatenate (along axis 0)
    print("\nBenchmarking concatenate_axis0...")
    concat_func = lambda x, y: lax.concatenate([x, y], dimension=0)
    results = benchmark_primitive(
        concat_func,
        shapes,
        dtype=dtype,
        num_warmup=num_warmup,
        num_runs=num_runs,
        num_inputs=2,
        device_id=device_id,
    )
    all_results["concatenate_axis0"] = results
    print_results("concatenate_axis0", results)

    # Concatenate (along axis 1)
    print("\nBenchmarking concatenate_axis1...")
    concat_func_axis1 = lambda x, y: lax.concatenate([x, y], dimension=1)
    results = benchmark_primitive(
        concat_func_axis1,
        shapes,
        dtype=dtype,
        num_warmup=num_warmup,
        num_runs=num_runs,
        num_inputs=2,
        device_id=device_id,
    )
    all_results["concatenate_axis1"] = results
    print_results("concatenate_axis1", results)

    # Broadcast (add scalar dimension)
    print("\nBenchmarking broadcast_in_dim...")
    def create_broadcast_func(shape):
        m, n = shape
        target_shape = (m, n, 4)  # Broadcast to add dimension
        return lambda x: lax.broadcast_in_dim(x, target_shape, (0, 1))

    broadcast_results = {}
    for shape in shapes:
        broadcast_func = create_broadcast_func(shape)
        try:
            results = benchmark_primitive(
                broadcast_func,
                [shape],
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=1,
                device_id=device_id,
            )
            shape_key = "x".join(map(str, shape))
            broadcast_results[shape_key] = results[shape_key]
        except Exception as e:
            print(f"Error broadcasting {shape}: {e}")

    all_results["broadcast_in_dim"] = broadcast_results
    print_results("broadcast_in_dim", broadcast_results)

    # Reverse (along axis 0)
    print("\nBenchmarking reverse_axis0...")
    reverse_func = lambda x: lax.rev(x, dimensions=(0,))
    results = benchmark_primitive(
        reverse_func,
        shapes,
        dtype=dtype,
        num_warmup=num_warmup,
        num_runs=num_runs,
        num_inputs=1,
        device_id=device_id,
    )
    all_results["reverse_axis0"] = results
    print_results("reverse_axis0", results)

    # Reverse (along axis 1)
    print("\nBenchmarking reverse_axis1...")
    reverse_func_axis1 = lambda x: lax.rev(x, dimensions=(1,))
    results = benchmark_primitive(
        reverse_func_axis1,
        shapes,
        dtype=dtype,
        num_warmup=num_warmup,
        num_runs=num_runs,
        num_inputs=1,
        device_id=device_id,
    )
    all_results["reverse_axis1"] = results
    print_results("reverse_axis1", results)

    # Pad (with zeros)
    print("\nBenchmarking pad...")
    def create_pad_func(shape):
        padding_config = [(10, 10), (10, 10)]  # Pad 10 on each side
        return lambda x: lax.pad(x, jnp.array(0.0, dtype=dtype), padding_config)

    pad_results = {}
    for shape in shapes:
        pad_func = create_pad_func(shape)
        try:
            results = benchmark_primitive(
                pad_func,
                [shape],
                dtype=dtype,
                num_warmup=num_warmup,
                num_runs=num_runs,
                num_inputs=1,
                device_id=device_id,
            )
            shape_key = "x".join(map(str, shape))
            pad_results[shape_key] = results[shape_key]
        except Exception as e:
            print(f"Error padding {shape}: {e}")

    all_results["pad"] = pad_results
    print_results("pad", pad_results)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark VPU memory operations")
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

    results = benchmark_all_memory_ops(
        shapes=shapes,
        dtype=dtype,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        device_id=args.device_id,
        dimension=args.dimension,
        hlo_dump_dir=args.hlo_dump_dir,
    )