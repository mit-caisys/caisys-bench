#!/bin/bash

export TENSOR_PARALLEL_SIZE=1

for config in src/configs/*.yaml; do
    num_diverse_programs=$(yq -r '.prompt.num_diverse_programs // ""' "$config")
    include_artifacts=$(yq -r '.prompt.include_artifacts // ""' "$config")
    model_names=$(yq -r '.llm.models[].name' "$config" | awk -F'/' '{print $NF}' | paste -sd '_' -)
    cores=$(yq -r '.cores // ""' "$config")

    template_dir=$(yq -r '.prompt.template_dir // ""' "$config")
    if [[ -n "$template_dir" && "$template_dir" != "null" ]]; then
        custom_prompt="_custom-prompt"
    else
        custom_prompt=""
    fi

    output_folder="${TENSOR_PARALLEL_SIZE}_${num_diverse_programs}_${include_artifacts}${custom_prompt}_${model_names}_${cores}"

    mkdir -p "src/openevolve_output/${output_folder}"

    taskset -c "$cores" python ../openevolve-run.py \
        src/initial_program.rs \
        src/evaluator.py \
        --config "$config" \
        --output "src/openevolve_output/${output_folder}" \
        --iterations 100 \
        --log-level "DEBUG"
done
