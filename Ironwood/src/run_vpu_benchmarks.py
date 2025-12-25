"""Main script to run all VPU primitive benchmarks.

Usage:
    python run_vpu_benchmarks.py --all
    python run_vpu_benchmarks.py --binary --unary
    python run_vpu_benchmarks.py --benchmark unary --min_size 2048 --max_size 8192
"""

import argparse
import json
import csv
import os
from datetime import datetime
import jax.numpy as jnp

from vpu_bench_utils import generate_shapes
from vpu_bench_binary import benchmark_all_binary_ops
from vpu_bench_unary import benchmark_all_unary_ops
from vpu_bench_comparison import benchmark_all_comparison_ops
from vpu_bench_reduction import benchmark_all_reduction_ops
from vpu_bench_convert import benchmark_all_convert_ops
from vpu_bench_memory import benchmark_all_memory_ops


def save_results_to_json(results, output_dir, benchmark_name):
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nJSON results saved to: {filepath}")
    return filepath


def save_results_to_csv(results, output_dir, benchmark_name):
    """Save benchmark results to CSV file.

    Creates a flat CSV with columns:
    operation, shape, mean_ms, std_ms, min_ms, max_ms, median_ms
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{benchmark_name}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Flatten nested results into rows
    rows = []
    for op_name, shapes_data in results.items():
        if isinstance(shapes_data, dict):
            for shape_key, metrics in shapes_data.items():
                if isinstance(metrics, dict) and 'mean_ms' in metrics:
                    row = {
                        'operation': op_name,
                        'shape': shape_key,
                        'mean_ms': float(metrics['mean_ms']),
                        'std_ms': float(metrics['std_ms']),
                        'min_ms': float(metrics['min_ms']),
                        'max_ms': float(metrics['max_ms']),
                        'median_ms': float(metrics['median_ms']),
                    }
                    rows.append(row)

    # Write CSV
    if rows:
        fieldnames = ['operation', 'shape', 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'median_ms']
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV results saved to: {filepath}")
    else:
        print(f"Warning: No data to save to CSV for {benchmark_name}")

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Run VPU primitive benchmarks")

    # Benchmark selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--binary", action="store_true", help="Run binary ops benchmarks")
    parser.add_argument("--unary", action="store_true", help="Run unary ops benchmarks")
    parser.add_argument("--comparison", action="store_true", help="Run comparison ops benchmarks")
    parser.add_argument("--reduction", action="store_true", help="Run reduction ops benchmarks")
    parser.add_argument("--convert", action="store_true", help="Run type conversion ops benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory ops benchmarks")

    # Benchmark parameters
    parser.add_argument("--min_size", type=int, default=1024, help="Minimum size")
    parser.add_argument("--max_size", type=int, default=16384, help="Maximum size")
    parser.add_argument("--step", type=int, default=1024, help="Size step")
    parser.add_argument("--dimension", type=int, default=2, choices=[1, 2], help="1D or 2D data (default: 2)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--num_warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID")

    # Output options
    parser.add_argument("--output_dir", type=str, default="/data/vpu_benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--save_json", action="store_true", help="Save results to JSON files")
    parser.add_argument("--save_csv", action="store_true", help="Save results to CSV files")
    parser.add_argument("--hlo_dump_dir", type=str, default=None,
                       help="Directory to save HLO dumps (optional)")

    args = parser.parse_args()

    # If no specific benchmark is selected, default to --all
    if not any([args.all, args.binary, args.unary, args.comparison,
                args.reduction, args.convert, args.memory]):
        args.all = True

    # Setup
    shapes = generate_shapes(args.dimension, args.min_size, args.max_size, args.step)
    dtype = getattr(jnp, args.dtype)

    print(f"\n{'='*80}")
    print(f"VPU Primitive Benchmarks")
    print(f"{'='*80}")
    if args.dimension == 1:
        print(f"Shape range: ({args.min_size},) to ({args.max_size},) (step: {args.step})")
    else:
        print(f"Shape range: {args.min_size}x{args.min_size} to {args.max_size}x{args.max_size} (step: {args.step})")
    print(f"Dimension: {args.dimension}D")
    print(f"Data type: {args.dtype}")
    print(f"Warmup runs: {args.num_warmup}")
    print(f"Benchmark runs: {args.num_runs}")
    print(f"Device ID: {args.device_id}")
    print(f"{'='*80}\n")

    all_results = {}

    # Run benchmarks
    if args.all or args.binary:
        print("\n" + "="*80)
        print("BINARY OPERATIONS")
        print("="*80)
        results = benchmark_all_binary_ops(
            shapes=shapes,
            dtype=dtype,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device_id=args.device_id,
            dimension=args.dimension,
            hlo_dump_dir=args.hlo_dump_dir,
        )
        all_results["binary"] = results
        if args.save_json:
            save_results_to_json(results, args.output_dir, "binary_ops")
        if args.save_csv:
            save_results_to_csv(results, args.output_dir, "binary_ops")

    if args.all or args.unary:
        print("\n" + "="*80)
        print("UNARY OPERATIONS")
        print("="*80)
        results = benchmark_all_unary_ops(
            shapes=shapes,
            dtype=dtype,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device_id=args.device_id,
            dimension=args.dimension,
        )
        all_results["unary"] = results
        if args.save_json:
            save_results_to_json(results, args.output_dir, "unary_ops")
        if args.save_csv:
            save_results_to_csv(results, args.output_dir, "unary_ops")

    if args.all or args.comparison:
        print("\n" + "="*80)
        print("COMPARISON OPERATIONS")
        print("="*80)
        results = benchmark_all_comparison_ops(
            shapes=shapes,
            dtype=dtype,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device_id=args.device_id,
            dimension=args.dimension,
        )
        all_results["comparison"] = results
        if args.save_json:
            save_results_to_json(results, args.output_dir, "comparison_ops")
        if args.save_csv:
            save_results_to_csv(results, args.output_dir, "comparison_ops")

    if args.all or args.reduction:
        print("\n" + "="*80)
        print("REDUCTION OPERATIONS")
        print("="*80)
        results = benchmark_all_reduction_ops(
            shapes=shapes,
            dtype=dtype,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device_id=args.device_id,
            dimension=args.dimension,
        )
        all_results["reduction"] = results
        if args.save_json:
            save_results_to_json(results, args.output_dir, "reduction_ops")
        if args.save_csv:
            save_results_to_csv(results, args.output_dir, "reduction_ops")

    if args.all or args.convert:
        print("\n" + "="*80)
        print("TYPE CONVERSION OPERATIONS")
        print("="*80)
        results = benchmark_all_convert_ops(
            shapes=shapes,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device_id=args.device_id,
            dimension=args.dimension,
        )
        all_results["convert"] = results
        if args.save_json:
            save_results_to_json(results, args.output_dir, "convert_ops")
        if args.save_csv:
            save_results_to_csv(results, args.output_dir, "convert_ops")

    if args.all or args.memory:
        print("\n" + "="*80)
        print("MEMORY OPERATIONS")
        print("="*80)
        results = benchmark_all_memory_ops(
            shapes=shapes,
            dtype=dtype,
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
            device_id=args.device_id,
            dimension=args.dimension,
        )
        all_results["memory"] = results
        if args.save_json:
            save_results_to_json(results, args.output_dir, "memory_ops")
        if args.save_csv:
            save_results_to_csv(results, args.output_dir, "memory_ops")

    # Save combined results
    if args.save_json and len(all_results) > 1:
        save_results_to_json(all_results, args.output_dir, "all_benchmarks")
    if args.save_csv and len(all_results) > 1:
        # Flatten all results into a single CSV
        combined_results = {}
        for category, cat_results in all_results.items():
            for op_name, shapes_data in cat_results.items():
                combined_key = f"{category}_{op_name}"
                combined_results[combined_key] = shapes_data
        save_results_to_csv(combined_results, args.output_dir, "all_benchmarks")

    print("\n" + "="*80)
    print("BENCHMARKS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()