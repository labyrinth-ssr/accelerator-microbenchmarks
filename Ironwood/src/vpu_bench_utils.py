"""Simple utility functions for VPU primitive benchmarking.

Focus on timing only, with optional HLO dump support.
"""

import time
import os
from typing import Callable, Tuple, Dict, Any, List, Optional
import jax
import jax.numpy as jnp
import numpy as np


def simple_timeit(
    func: Callable,
    *args,
    num_warmup: int = 3,
    num_runs: int = 10,
    **kwargs
) -> List[float]:
    """Simple timing function using jax.block_until_ready.

    Args:
        func: JIT-compiled function to benchmark
        *args: Arguments to pass to func
        num_warmup: Number of warmup runs
        num_runs: Number of timed runs
        **kwargs: Keyword arguments to pass to func

    Returns:
        List of execution times in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        result = func(*args, **kwargs)
        jax.block_until_ready(result)

    # Timed runs
    times_ms = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000)

    return times_ms


def benchmark_primitive(
    primitive_func: Callable,
    input_shapes: List[Tuple[int, ...]],
    dtype: jnp.dtype = jnp.bfloat16,
    num_warmup: int = 3,
    num_runs: int = 10,
    num_inputs: int = 1,
    device_id: int = 0,
) -> Dict[str, Any]:
    """Benchmark a single primitive across multiple input sizes.

    Args:
        primitive_func: The jax.lax primitive to benchmark (e.g., jax.lax.add)
        input_shapes: List of input shapes to test
        dtype: Data type to use
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        num_inputs: Number of input arrays (1 for unary, 2 for binary, etc.)
        device_id: Device to run on (default 0 for single device)

    Returns:
        Dictionary with benchmark results
    """
    results = {}
    device = jax.devices()[device_id]

    for shape in input_shapes:
        # Create random inputs on the specified device
        key = jax.random.key(42)
        keys = jax.random.split(key, num_inputs)

        inputs = []
        for i in range(num_inputs):
            arr = jax.random.normal(keys[i], shape, dtype=dtype)
            arr = jax.device_put(arr, device)
            inputs.append(arr)

        # JIT compile the primitive
        jitted_func = jax.jit(primitive_func)

        # Run benchmark
        times = simple_timeit(jitted_func, *inputs, num_warmup=num_warmup, num_runs=num_runs)

        shape_key = "x".join(map(str, shape))
        results[shape_key] = {
            "times_ms": times,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
        }

    return results


def generate_2d_shapes(
    min_size: int = 1024,
    max_size: int = 32768,
    step: int = 1024,
) -> List[Tuple[int, int]]:
    """Generate list of 2D square matrix shapes."""
    sizes = range(min_size, max_size + 1, step)
    return [(s, s) for s in sizes]


def generate_1d_shapes(
    min_size: int = 1024,
    max_size: int = 1048576,
    step: int = None,
    multiplier: int = 2,
) -> List[Tuple[int,]]:
    """Generate list of 1D vector shapes.

    Args:
        min_size: Minimum vector size
        max_size: Maximum vector size
        step: Linear step size (if provided, uses linear growth instead of exponential)
        multiplier: Exponential growth multiplier (only used if step is None)

    Returns:
        List of 1D shapes
    """
    shapes = []
    if step is not None:
        # Linear growth
        sizes = range(min_size, max_size + 1, step)
        shapes = [(s,) for s in sizes]
    else:
        # Exponential growth
        size = min_size
        while size <= max_size:
            shapes.append((size,))
            size *= multiplier
    return shapes


def generate_shapes(
    dimension: int,
    min_size: int = 1024,
    max_size: int = 32768,
    step: int = 1024,
    multiplier: int = 2,
) -> List[Tuple]:
    """Generate shapes based on dimension.

    Args:
        dimension: 1 for 1D vectors, 2 for 2D matrices
        min_size: Minimum size
        max_size: Maximum size
        step: Step size (for linear growth)
        multiplier: Multiplier (for exponential growth in 1D)

    Returns:
        List of shapes
    """
    if dimension == 1:
        return generate_1d_shapes(min_size, max_size, step, multiplier)
    elif dimension == 2:
        return generate_2d_shapes(min_size, max_size, step)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}. Use 1 or 2.")


def configure_hlo_dump(hlo_dump_dir: str, enable: bool = True):
    """Configure XLA flags for HLO dumping.

    Args:
        hlo_dump_dir: Directory to save HLO dumps
        enable: Whether to enable HLO dumping
    """
    if enable:
        os.makedirs(hlo_dump_dir, exist_ok=True)

        # Set XLA flags for HLO dumping
        xla_flags = [
            f"--xla_dump_to={hlo_dump_dir}",
            "--xla_enable_dumping=true",
            "--xla_dump_hlo_as_long_text=true",
            "--xla_dump_full_hlo_config=true",
            "--xla_dump_include_timestamp=false",
        ]

        # Get existing XLA_FLAGS if any
        existing_flags = os.environ.get("XLA_FLAGS", "")

        # Combine flags
        if existing_flags:
            combined_flags = existing_flags + " " + " ".join(xla_flags)
        else:
            combined_flags = " ".join(xla_flags)

        os.environ["XLA_FLAGS"] = combined_flags
        print(f"HLO dumping enabled. Output directory: {hlo_dump_dir}")
    else:
        # Remove dump-related flags
        existing_flags = os.environ.get("XLA_FLAGS", "")
        if existing_flags:
            # Filter out dump-related flags
            flags = existing_flags.split()
            filtered_flags = [f for f in flags if not f.startswith("--xla_dump") and not f.startswith("--xla_enable_dumping")]
            os.environ["XLA_FLAGS"] = " ".join(filtered_flags)


def benchmark_primitive_with_hlo(
    primitive_func: Callable,
    primitive_name: str,
    input_shapes: List[Tuple[int, ...]],
    dtype: jnp.dtype = jnp.bfloat16,
    num_warmup: int = 3,
    num_runs: int = 10,
    num_inputs: int = 1,
    device_id: int = 0,
    hlo_dump_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Benchmark a primitive with optional HLO dump support.

    Args:
        primitive_func: The jax.lax primitive to benchmark
        primitive_name: Name of the primitive (for HLO dump naming)
        input_shapes: List of input shapes to test
        dtype: Data type to use
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        num_inputs: Number of input arrays
        device_id: Device to run on
        hlo_dump_dir: Directory for HLO dumps (None to disable)

    Returns:
        Dictionary with benchmark results
    """
    results = {}
    device = jax.devices()[device_id]

    # Configure HLO dumping if requested
    if hlo_dump_dir:
        configure_hlo_dump(hlo_dump_dir, enable=True)

    for shape in input_shapes:
        # Create shape identifier
        shape_str = "_".join(map(str, shape))
        if len(shape) == 2:
            shape_identifier = f"m_{shape[0]}_n_{shape[1]}"
        elif len(shape) == 1:
            shape_identifier = f"size_{shape[0]}"
        else:
            shape_identifier = shape_str

        # Create random inputs on the specified device
        key = jax.random.key(42)
        keys = jax.random.split(key, num_inputs)

        inputs = []
        for i in range(num_inputs):
            arr = jax.random.normal(keys[i], shape, dtype=dtype)
            arr = jax.device_put(arr, device)
            inputs.append(arr)

        # Wrap function with named computation for HLO dumps
        def named_func(*args):
            return primitive_func(*args)

        # Set the function name for HLO dumps
        if hlo_dump_dir:
            func_name = f"{primitive_name}_{shape_identifier}"
            named_func.__name__ = func_name

        # JIT compile the primitive
        jitted_func = jax.jit(named_func)

        # Run benchmark (first run triggers compilation and HLO dump)
        times = simple_timeit(jitted_func, *inputs, num_warmup=num_warmup, num_runs=num_runs)

        shape_key = "x".join(map(str, shape))
        results[shape_key] = {
            "times_ms": times,
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
        }

    return results


def print_results(primitive_name: str, results: Dict[str, Any]):
    """Pretty print benchmark results."""
    print(f"\n{'='*80}")
    print(f"Benchmark: {primitive_name}")
    print(f"{'='*80}")
    print(f"{'Shape':<20} {'Mean (ms)':<15} {'Std (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15}")
    print(f"{'-'*80}")

    for shape_key, metrics in results.items():
        print(f"{shape_key:<20} {metrics['mean_ms']:<15.4f} {metrics['std_ms']:<15.4f} "
              f"{metrics['min_ms']:<15.4f} {metrics['max_ms']:<15.4f}")