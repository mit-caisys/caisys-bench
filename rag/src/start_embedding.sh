#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_TOKEN=$(grep '^HF_TOKEN=' "$SCRIPT_DIR/../../.env" | cut -d'=' -f2-)
"$SCRIPT_DIR/../../start_instance/start_embedding.sh" '\"device=1\"' 'google/embeddinggemma-300m' 1 '8-15'
# "$SCRIPT_DIR/../../start_instance/start_embedding.sh" '\"device=0,1\"' 'google/embeddinggemma-300m' 2 '8-15'
