#!/bin/bash

# Quick GPU Benchmark Demo
# This runs a very short benchmark for demonstration purposes

echo "🚀 Quick GPU Benchmark Demo"
echo "============================"

# Activate environment
source ../../ddpm_env/bin/activate

echo "This demo will run a quick comparison between 1 GPU and 8 GPUs"
echo "using a small batch size and limited steps for demonstration."
echo ""

# Check if we want to run the full demo
read -p "Do you want to run the quick benchmark demo? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Demo cancelled."
    exit 0
fi

echo "Starting quick benchmark..."
echo ""

# Run with smaller settings for demo
python ../benchmark_gpu_training.py \
    --batch_size 8 \
    --num_workers 2 \
    --output_dir "demo_benchmark_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Demo completed! 🎉"
echo ""
echo "For a full benchmark with larger batches, run:"
echo "  bash run_gpu_benchmark.sh --batch_size 32"
