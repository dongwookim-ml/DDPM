#!/bin/bash

# GPU Benchmark Launcher Script
# Usage: bash run_gpu_benchmark.sh [options]

echo "🚀 DDPM GPU Training Speed Benchmark"
echo "======================================"

# Activate environment
source ddpm_env/bin/activate

# Set environment variables for CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Default parameters
BATCH_SIZE=32
NUM_WORKERS=4
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip_single_gpu)
            SKIP_SINGLE="--skip_single_gpu"
            shift
            ;;
        --skip_multi_gpu)
            SKIP_MULTI="--skip_multi_gpu"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch_size INT        Batch size per GPU (default: 32)"
            echo "  --num_workers INT       Number of data loader workers (default: 4)"
            echo "  --output_dir STR        Output directory (default: benchmark_results_TIMESTAMP)"
            echo "  --skip_single_gpu       Skip single GPU benchmark"
            echo "  --skip_multi_gpu        Skip multi-GPU benchmark"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Number of workers: $NUM_WORKERS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Available GPUs: $(nvidia-smi --list-gpus | wc -l)"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please ensure CUDA is installed."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    echo "❌ No GPUs detected."
    exit 1
fi

echo "🔍 GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | while read line; do
    echo "  GPU $line"
done
echo ""

# Run the benchmark
echo "🏃‍♂️ Starting benchmark..."
python benchmark_gpu_training.py \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    $SKIP_SINGLE \
    $SKIP_MULTI

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Benchmark completed successfully!"
    echo "📊 Results saved to: $OUTPUT_DIR/"
    echo ""
    echo "Generated files:"
    ls -la $OUTPUT_DIR/
else
    echo ""
    echo "❌ Benchmark failed!"
    exit 1
fi
