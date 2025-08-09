#!/bin/bash

# DDPM Environment Activation Script
# Usage: source activate_ddpm_env.sh

echo "Activating DDPM environment..."

# Activate the virtual environment
source ddpm_env/bin/activate

# Verify the environment
echo "Environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Set project root
export DDPM_ROOT="$(pwd)"
export PYTHONPATH="${DDPM_ROOT}:${PYTHONPATH}"

echo "Environment variables set:"
echo "  DDPM_ROOT: ${DDPM_ROOT}"
echo "  PYTHONPATH: ${PYTHONPATH}"

echo ""
echo "🎉 DDPM environment is ready!"
echo ""
echo "Available scripts:"
echo "  - Training: python scripts/train_cifar.py"
echo "  - Distributed training: bash launch_distributed_training.sh"
echo "  - Sampling: python scripts/sample_images.py"
echo "  - Test setup: python test_setup.py"
echo "  - Test benchmark setup: python test_benchmark_setup.py"
echo "  - GPU benchmark: bash run_gpu_benchmark.sh"
echo "  - Quick benchmark: bash run_gpu_benchmark.sh --batch_size 16"
echo ""
echo "To deactivate: deactivate"
