#!/bin/bash

export TENSOR_PARALLEL_SIZE=1

for config in src/configs/*.yaml; do
    name=$(basename "$config" .yaml)
    model_names=$(yq -r '.llm.models[].name' "$config" | awk -F'/' '{print $NF}' | paste -sd '_' -)
    cores=$(yq -r '.cores // ""' "$config")

    output_folder="${TENSOR_PARALLEL_SIZE}_${name}_${model_names}_${cores}"

    mkdir -p "src/openevolve_output/${output_folder}"

    taskset -c "$cores" python ../openevolve-run.py \
        src/initial_program_parallel.rs \
        src/evaluator.py \
        --config "$config" \
        --output "src/openevolve_output/${output_folder}" \
        --iterations 100 \
        --log-level "DEBUG"
done
