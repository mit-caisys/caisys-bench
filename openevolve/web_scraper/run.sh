#!/bin/bash

export TENSOR_PARALLEL_SIZE=2

for config in src/configs/*.yaml; do
    name=$(basename "$config" .yaml)

    primary=$(grep -E '^\s*primary_model:' "$config" | awk -F'"' '{print $2}')
    secondary=$(grep -E '^\s*secondary_model:' "$config" | awk -F'"' '{print $2}')

    primary_name="${primary##*/}"
    if [ "$secondary" != "" ]; then
        secondary_name="${secondary##*/}"
        model_name="${primary_name}_${secondary_name}"
    else
        model_name="$primary_name"
    fi

    output_folder="${TENSOR_PARALLEL_SIZE}_${name}_${model_name}"

    mkdir -p "src/openevolve_output/${output_folder}"

    python ../openevolve-run.py \
        src/initial_program.py \
        src/evaluator.py \
        --config "$config" \
        --output "src/openevolve_output/${output_folder}" \
        --iterations 100 \
        --log-level "DEBUG"
done
