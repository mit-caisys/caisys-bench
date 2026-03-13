#!/bin/bash

# install pip library
pip install -r requirements.txt

# create relevant directories
python src/common/path.py

# download dataset
export HF_TOKEN=$(grep '^HF_TOKEN=' ./src/.env | cut -d'=' -f2-)
huggingface-cli download google/frames-benchmark test.tsv --repo-type dataset --local-dir ./dataset/frames-benchmark
