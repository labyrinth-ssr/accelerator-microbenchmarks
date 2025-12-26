#!/bin/bash

CONFIG_NAMES="gemm_accum gemm_fp8_b128_fp32_static_scaling gemm_fp8_b128_fp32 gemm_fp8_rowwise_static_scaling gemm_fp8_rowwise_w_dequantize gemm_fp8_rowwise gemm_multiple_run_more gemm_multiple_run gemm_mxfp8_b32_static_scaling gemm_mxfp8_b32 gemm_simple gemm"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/training/${CONFIG}.yaml"

  echo "--- Starting benchmark for ${CONFIG} ---"

  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  echo "--- Finished benchmark for ${CONFIG} ---"
done

# add.yaml,quantization.yaml,transpose_quantization.yaml,quantization_static_scaling.yaml,transpose_quantization_static_scaling.yaml,swiglu_fwd.yaml,rmsnorm_fwd