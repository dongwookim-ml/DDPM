#!/bin/bash

# DDPM Multi-GPU Training Launch Script
# This script launches distributed training across all available GPUs

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Activate virtual environment
source ddpm_env/bin/activate

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Launch distributed training
echo "Starting distributed training on 4 GPUs..."
echo "Project: ddpm-cifar10-4gpu"
echo "Effective batch size: 512 (128 per GPU)"
echo "Iterations: 100,000"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    scripts/train_cifar_distributed.py \
    --project_name "ddpm-cifar10-4gpu" \
    --batch_size 512 \
    --iterations 100000 \
    --log_rate 500 \
    --checkpoint_rate 2000 \
    --learning_rate 2e-4

echo "Training completed!"
