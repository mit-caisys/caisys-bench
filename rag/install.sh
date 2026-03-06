#!/bin/bash

# install pip library
pip install -r requirements.txt

# download dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_TOKEN=$(grep '^HF_TOKEN=' "$SCRIPT_DIR/../.env" | cut -d'=' -f2-)

mkdir dataset
hf download google/frames-benchmark test.tsv --repo-type dataset --local-dir ./dataset/frames-benchmark

cd dataset
./download.sh
cd ..
