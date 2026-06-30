#!/bin/bash

set -e

export ROOT=/your_root
export MODEL_DIR=$ROOT/models
export MODEL_NAME=Qwen-4B

mkdir -p $MODEL_DIR

cd $MODEL_DIR

huggingface-cli download Qwen/Qwen3-4B-Base --local-dir $MODEL_NAME --local-dir-use-symlinks False
# huggingface-cli download Qwen/Qwen3-8B-Base --local-dir $MODEL_NAME --local-dir-use-symlinks False