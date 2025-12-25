# Binary Operations Benchmarks

This guide explains how to run benchmarks for JAX lax binary operations.

## Available Binary Operations

The following binary operations are available:
- `add` - Addition (X + Y)
- `sub` - Subtraction (X - Y)
- `mul` - Multiplication (X * Y)
- `div` - Division (X / Y)
- `rem` - Remainder (X % Y)
- `pow` - Power (X ** Y)
- `maximum` - Element-wise maximum
- `minimum` - Element-wise minimum
- `atan2` - Arc tangent of Y/X
- `nextafter` - Next representable floating-point value

## Running Binary Operations

### Method 1: Using Individual Functions

Run specific binary operations using their individual functions:

```bash
# Run add operation
python run_benchmark.py --config=configs/binary_ops.yaml

# The config file specifies individual operations:
# - benchmark_name: "add"
#   benchmark_params:
#     - { m: 1024, n: 1024, num_runs: 10 }
```

### Method 2: Using Parameter Sweeps

Run operations with parameter sweeps to test multiple matrix sizes:

```bash
python run_benchmark.py --config=configs/binary_ops_sweep.yaml
```

Example config with parameter sweep:
```yaml
- benchmark_name: "add"
  benchmark_sweep_params:
    - {
        m_range: { start: 1024, end: 8192, multiplier: 2 },
        n: "SAME_AS_m",  # n will be set to the same value as m
        num_runs: 10,
      }
```

This will run the benchmark with m = 1024, 2048, 4096, 8192.

### Method 3: Using the Generic binary_op Function

Use the generic `binary_op` function to run multiple operations with the same parameters:

```yaml
- benchmark_name: "binary_op"
  benchmark_params:
    - { op_name: "add", m: 1024, n: 1024, num_runs: 10 }
    - { op_name: "sub", m: 1024, n: 1024, num_runs: 10 }
    - { op_name: "mul", m: 1024, n: 1024, num_runs: 10 }
```

## Configuration Parameters

### Required Parameters
- `m`: First dimension of the matrices (rows)
- `n`: Second dimension of the matrices (columns)

### Optional Parameters
- `num_runs`: Number of benchmark runs (default: 1)
- `trace_dir`: Directory to save performance traces
- `op_name`: Operation name (only required when using `binary_op` function)

### Special Parameters
- `SAME_AS_m`: Sets the parameter to the same value as `m`
- `m_range`: Sweep `m` values using start, end, and multiplier/increase_by
- `m_list`: Specify a list of values for `m`

## Output

Benchmarks generate the following outputs:
1. **CSV files**: Tab-separated values with timing and performance metrics
2. **HLO graphs**: XLA compiler intermediate representations
3. **Traces**: Performance traces (if trace_dir is specified)

Output location is specified in the config file:
- `csv_path`: Where to save CSV metrics
- `xla_dump_dir`: Where to save HLO graphs
- `trace_dir`: Where to save performance traces

## Example: Running All Binary Ops

To benchmark all binary operations with the same parameters:

1. Create a custom config file:
```yaml
benchmarks:
  - benchmark_name: "add"
    benchmark_params:
      - { m: 2048, n: 2048, num_runs: 20 }
    csv_path: "./results/binary_ops"

  - benchmark_name: "sub"
    benchmark_params:
      - { m: 2048, n: 2048, num_runs: 20 }
    csv_path: "./results/binary_ops"

  # ... repeat for all operations
```

2. Run the benchmark:
```bash
python run_benchmark.py --config=configs/my_binary_ops.yaml --output_path=./results
```

## Programmatic Usage

You can also use the binary operations directly in Python:

```python
from benchmark_compute import binary_op, binary_op_calculate_metrics

# Run a single operation
result = binary_op("add", m=1024, n=1024, num_runs=10)
metrics = binary_op_calculate_metrics(1024, 1024, result["time_ms_list"])
print(metrics)

# Run all operations
ops_to_run = ["add", "sub", "mul", "div", "max", "min"]
for op in ops_to_run:
    result = binary_op(op, m=1024, n=1024, num_runs=10)
    metrics = binary_op_calculate_metrics(1024, 1024, result["time_ms_list"])
    print(f"{op}: {metrics}")
```

## Notes

- All binary operations use bfloat16 data type by default
- The operations are benchmarked with JAX's lax primitives for maximum performance
- Sharding strategy can be configured in `benchmark_compute.py` (default: NO_SHARDING)