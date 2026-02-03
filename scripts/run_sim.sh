#!/bin/bash

# 1. Define Project Path
# We map your local Linux folder (~/evlaformer_lab) to /workspace/evlaformer_lab inside Docker
PROJECT_DIR="$HOME/evlaformer_lab"

# 2. Run Isaac Sim Container
# --rm: Automatically remove container when it exits
# --gpus all: Enable all GPUs
# -v: Mount volumes (Synchornize folders between Host and Container)
docker run --name isaac-sim --entrypoint ./runheadless.native.sh --gpus all --rm \
    -e "ACCEPT_EULA=Y" \
    -v $PROJECT_DIR:/workspace/evlaformer_lab \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    nvcr.io/nvidia/isaac-sim:4.2.0