#!/bin/bash

for config in configs/*.yaml; do
    cores=$(yq -r '.cores // ""' "$config")

    taskset -c "$cores" python run.py "$config" "$@"
done
