#!/bin/bash

CONFIG_NAMES=" add_bf16 add_fp8"

for CONFIG in $CONFIG_NAMES
do
  # Construct the full config file path
  CONFIG_FILE="Ironwood/configs/inference/${CONFIG}.yaml"

  echo "--- Starting benchmark for ${CONFIG} ---"

  # Run the python script and wait for it to complete
  python Ironwood/src/run_benchmark.py --config="${CONFIG_FILE}"

  echo "--- Finished benchmark for ${CONFIG} ---"
done

# add.yaml,quantization.yaml,transpose_quantization.yaml,quantization_static_scaling.yaml,transpose_quantization_static_scaling.yaml,swiglu_fwd.yaml,rmsnorm_fwd