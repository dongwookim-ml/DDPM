#!/bin/bash

# DDPM Multi-GPU Training Launch Script
# This script launches distributed training across all available GPUs

# Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=localhost
export MASTER_PORT=12355

# Activate virtual environment
source ddpm_env/bin/activate

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Launch distributed training
echo "Starting distributed training on 8 GPUs..."
echo "Project: ddpm-cifar10-8gpu-full"
echo "Effective batch size: 2048 (256 per GPU)"
echo "Iterations: 200,000"

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    scripts/train/train_cifar_distributed.py \
    --project_name "ddpm-cifar10-8gpu-full" \
    --batch_size 2048 \
    --iterations 200000 \
    --log_rate 1000 \
    --checkpoint_rate 5000 \
    --learning_rate 2e-4 \
    --use_labels False \
    --log_to_wandb False

echo "Training completed!"
