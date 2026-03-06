#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_TOKEN=$(grep '^HF_TOKEN=' "$SCRIPT_DIR/../../.env" | cut -d'=' -f2-)
"$SCRIPT_DIR/../../start_instance/start_vllm.sh" '\"device=0\"' 'google/gemma-3-27b-it' 1 '0-7'
# "$SCRIPT_DIR/../../start_instance/start_vllm.sh" '\"device=0,1\"' 'google/gemma-3-27b-it' 2 '0-7'
