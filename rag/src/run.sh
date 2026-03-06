#!/bin/bash

config="../config/config_vllm.yaml"
cores=$(yq -r '.cores // ""' "$config")

taskset -c "$cores" python run_experiment.py "$@" -f "$config"
