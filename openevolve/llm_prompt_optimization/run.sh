#!/bin/bash
# Wrapper script to run OpenEvolve with the correct dataset

export TENSOR_PARALLEL_SIZE=2
export OPENEVOLVE_PROMPT=src/hotpotqa_prompt.txt

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
        "$OPENEVOLVE_PROMPT" \
        src/evaluator.py \
        --config "$config" \
        --output "src/openevolve_output/${output_folder}" \
        --iterations 100 \
        --log-level "DEBUG"
done

# if [ $# -lt 1 ]; then
#     echo "Usage: $0 <prompt_file> [additional_args...]"
#     echo "Example: $0 src/emotion_prompt.txt"
#     exit 1
# fi
#
# PROMPT_FILE=$1
# shift  # Remove first argument
# # Set the environment variable for the evaluator
# export OPENEVOLVE_PROMPT=$PROMPT_FILE
#
# for config in src/configs/*.yaml; do
#     name=$(basename "$config" .yaml)
#
#     mkdir -p "src/openevolve_output/${TENSOR_PARALLEL_SIZE}_${name}"
#
#     python ../openevolve-run.py \
#         "$PROMPT_FILE" \
#         src/evaluator.py \
#         --config "$config" \
#         --output "src/openevolve_output/${TENSOR_PARALLEL_SIZE}_${name}" \
#         --iterations 10 \
#         --log-level "DEBUG" \
#         "$@"
# done
