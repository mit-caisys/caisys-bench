#!/bin/bash

export HF_TOKEN=$(grep '^HF_TOKEN=' ../../.env | cut -d'=' -f2-)
../../start_instance/start_vllm.sh '\"device=0\"' 'google/gemma-3-27b-it' 1

