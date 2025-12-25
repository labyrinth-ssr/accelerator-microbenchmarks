# VPU Primitive Benchmarks

Simple, focused benchmarks for JAX LAX primitives running on VPU. These benchmarks measure **execution time only** - no profiling, tracing, or HLO dumps.

## Overview

The benchmarks are organized into 6 categories:

1. **Binary Operations** (`vpu_bench_binary.py`)
   - add, sub, mul, div, pow, max, min, rem, atan2, nextafter

2. **Unary Operations** (`vpu_bench_unary.py`)
   - neg, abs, sign, sqrt, rsqrt, cbrt
   - exp, exp2, expm1, log, log1p
   - sin, cos, tan, asin, acos, atan
   - sinh, cosh, tanh, asinh, acosh, atanh
   - erf, erfc, erf_inv, logistic (sigmoid)
   - ceil, floor, round
   - lgamma, digamma, bessel_i0e, bessel_i1e

3. **Comparison Operations** (`vpu_bench_comparison.py`)
   - eq, ne, gt, lt, ge, le
   - select, clamp

4. **Reduction Operations** (`vpu_bench_reduction.py`)
   - reduce_sum, reduce_max, reduce_min, reduce_prod
   - reduce_and, reduce_or, reduce_xor
   - (tested across different axes)

5. **Type Conversion** (`vpu_bench_convert.py`)
   - bf16 ↔ fp32, bf16 ↔ fp16, fp32 ↔ fp16
   - bf16/fp32 ↔ fp8_e4m3fn, bf16/fp32 ↔ fp8_e5m2
   - int32 ↔ bf16/fp32

6. **Memory Operations** (`vpu_bench_memory.py`)
   - transpose, reshape, slice, concatenate
   - broadcast_in_dim, reverse, pad

## Quick Start

### Run all benchmarks (2D data)
```bash
cd Ironwood/src
python run_vpu_benchmarks.py --all --save_json --save_csv
```

### Run all benchmarks (1D data)
```bash
# Benchmark 1D vectors
python run_vpu_benchmarks.py --all --dimension 1 --save_csv
```

### Run specific benchmark categories
```bash
# Binary operations only (2D)
python run_vpu_benchmarks.py --binary

# Unary and comparison operations (1D vectors)
python run_vpu_benchmarks.py --unary --comparison --dimension 1

# Type conversions with custom size range
python run_vpu_benchmarks.py --convert --min_size 2048 --max_size 8192 --step 2048
```

### Run individual benchmark files
```bash
# Binary operations (2D matrices)
python vpu_bench_binary.py --min_size 1024 --max_size 16384 --step 1024

# Unary operations with fp32 (1D vectors)
python vpu_bench_unary.py --dimension 1 --dtype float32 --num_runs 20

# Reduction operations (1D vectors)
python vpu_bench_reduction.py --dimension 1 --min_size 4096 --max_size 1048576
```

## Command-line Options

### Main runner (`run_vpu_benchmarks.py`)

**Benchmark selection:**
- `--all` - Run all benchmarks (default if nothing specified)
- `--binary` - Run binary operations
- `--unary` - Run unary operations
- `--comparison` - Run comparison operations
- `--reduction` - Run reduction operations
- `--convert` - Run type conversion operations
- `--memory` - Run memory operations

**Benchmark parameters:**
- `--min_size SIZE` - Minimum size (default: 1024)
- `--max_size SIZE` - Maximum size (default: 16384)
- `--step SIZE` - Step size between dimensions (default: 1024)
- `--dimension DIM` - Use 1D vectors or 2D matrices: 1 or 2 (default: 2)
- `--dtype DTYPE` - Data type: bfloat16, float32, float16 (default: bfloat16)
- `--num_warmup N` - Number of warmup runs (default: 3)
- `--num_runs N` - Number of timed runs (default: 10)
- `--device_id ID` - Device to run on (default: 0)

**Output options:**
- `--save_json` - Save results to JSON files
- `--save_csv` - Save results to CSV files
- `--output_dir DIR` - Output directory (default: /tmp/vpu_benchmark_results)

## Examples

### Full benchmark suite with different sizes (2D)
```bash
python run_vpu_benchmarks.py \
  --all \
  --min_size 1024 \
  --max_size 32768 \
  --step 1024 \
  --num_runs 20 \
  --save_json --save_csv \
  --output_dir ./results
```

### Full benchmark suite for 1D vectors
```bash
python run_vpu_benchmarks.py \
  --all \
  --dimension 1 \
  --min_size 1024 \
  --max_size 1048576 \
  --step 1024 \
  --num_runs 20 \
  --save_csv \
  --output_dir ./results_1d
```

### Save to CSV only (for data analysis)
```bash
# Save all results to CSV for analysis in pandas/Excel
python run_vpu_benchmarks.py --all --save_csv --output_dir ./csv_results
```

### Quick test with small sizes
```bash
python run_vpu_benchmarks.py \
  --binary --unary \
  --min_size 512 \
  --max_size 2048 \
  --step 512 \
  --num_warmup 1 \
  --num_runs 3
```

### Benchmark specific operations
```bash
# Just binary ops with fp32 (2D)
python vpu_bench_binary.py --dtype float32 --num_runs 50

# Just unary ops on device 1 (1D)
python vpu_bench_unary.py --dimension 1 --device_id 1

# Type conversions only (1D)
python run_vpu_benchmarks.py --convert --dimension 1 --save_json

# Binary ops with 1D vectors (large sizes)
python vpu_bench_binary.py \
  --dimension 1 \
  --min_size 4096 \
  --max_size 4194304 \
  --step 4096
```

## Output Format

### Console Output
Each benchmark prints a table with:
- Shape (e.g., "1024x1024")
- Mean time (ms)
- Standard deviation (ms)
- Min time (ms)
- Max time (ms)

Example:
```
================================================================================
Benchmark: add
================================================================================
Shape                Mean (ms)       Std (ms)        Min (ms)        Max (ms)
--------------------------------------------------------------------------------
1024x1024           0.1234          0.0056          0.1180          0.1320
2048x2048           0.4567          0.0123          0.4420          0.4750
```

### JSON Output (with `--save_json`)
Results are saved as JSON files with structure:
```json
{
  "add": {
    "1024x1024": {
      "times_ms": [0.123, 0.124, ...],
      "mean_ms": 0.1234,
      "std_ms": 0.0056,
      "min_ms": 0.1180,
      "max_ms": 0.1320,
      "median_ms": 0.1230
    },
    ...
  },
  ...
}
```

### CSV Output (with `--save_csv`)
Results are saved as flat CSV files with columns:
```
operation,shape,mean_ms,std_ms,min_ms,max_ms,median_ms
add,1024x1024,0.1234,0.0056,0.1180,0.1320,0.1230
add,2048x2048,0.4567,0.0123,0.4420,0.4750,0.4550
sub,1024x1024,0.1250,0.0058,0.1190,0.1330,0.1245
...
```

CSV files are ideal for:
- Loading into spreadsheet software (Excel, Google Sheets)
- Data analysis with pandas/R
- Plotting with matplotlib/seaborn
- Quick filtering and sorting

## Implementation Details

- **Single device execution** - Benchmarks run on device 0 by default (configurable)
- **No sharding** - Data stays on one device for pure VPU performance measurement
- **Simple timing** - Uses `jax.block_until_ready()` for accurate timing
- **No overhead** - No profiling, tracing, or HLO dumps during benchmarking
- **Warmup runs** - Default 3 warmup iterations before timing
- **Flexible dimensions** - Supports both 1D vectors and 2D matrices
  - **1D mode**: Tests vector operations (e.g., `(1024,)`, `(2048,)`, ...)
  - **2D mode**: Tests matrix operations (e.g., `(1024, 1024)`, `(2048, 2048)`, ...)

## Files

- `vpu_bench_utils.py` - Shared utilities for timing and result formatting
- `vpu_bench_binary.py` - Binary operations benchmarks
- `vpu_bench_unary.py` - Unary operations benchmarks
- `vpu_bench_comparison.py` - Comparison operations benchmarks
- `vpu_bench_reduction.py` - Reduction operations benchmarks
- `vpu_bench_convert.py` - Type conversion benchmarks
- `vpu_bench_memory.py` - Memory operations benchmarks
- `run_vpu_benchmarks.py` - Main runner for all benchmarks

## Adding New Primitives

To add a new primitive:

1. Identify which category it belongs to (binary, unary, etc.)
2. Add it to the appropriate `vpu_bench_*.py` file
3. Add the operation to the ops dictionary in that file

Example for adding a new binary op:
```python
# In vpu_bench_binary.py
binary_ops = {
    ...
    "my_new_op": lax.my_new_op,
}
```

## Notes

- **Default mode**: 2D square matrices (e.g., 1024x1024, 2048x2048)
- **1D mode**: Use `--dimension 1` for vector benchmarks (e.g., (1024,), (2048,))
- Data is generated randomly on each run
- Default dtype is bfloat16 (common for TPU/VPU)
- Results may vary based on system load and hardware state
- For 1D benchmarks, you may want larger max sizes (e.g., `--max_size 1048576`)
- Some operations (like transpose) only apply to 2D data and are skipped in 1D mode