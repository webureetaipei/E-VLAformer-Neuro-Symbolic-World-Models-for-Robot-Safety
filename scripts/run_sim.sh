#!/bin/bash

# 1. 定義路徑 (讓 Docker 知道你的專案在哪裡)
PROJECT_DIR="$HOME/evlaformer_lab"

# 2. 啟動 Isaac Sim
# -v $PROJECT_DIR:/workspace/evlaformer_lab : 這行是關鍵！它把你的 Linux 資料夾掛載到 Docker 內
docker run --name isaac-sim --entrypoint ./runheadless.native.sh --gpus all --rm \
    -e "ACCEPT_EULA=Y" \
    -v $PROJECT_DIR:/workspace/evlaformer_lab \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    nvcr.io/nvidia/isaac-sim:4.2.0