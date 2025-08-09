# 🚀 DDPM GPU Training Speed Benchmark

A comprehensive benchmarking suite to compare training performance between single GPU and multi-GPU setups for DDPM (Denoising Diffusion Probabilistic Models).

## 📋 Overview

This benchmark suite provides:
- **Automated comparison** between 1 GPU vs 8 GPU training performance
- **Real-time monitoring** of GPU memory usage and training metrics
- **Beautiful visualizations** of performance comparisons
- **Detailed performance reports** with speedup analysis
- **Easy-to-use scripts** for running benchmarks

## 🔧 Setup

### 1. Environment Setup
```bash
# Activate the DDPM environment
source activate_ddpm_env.sh
```

### 2. Verify Setup
```bash
# Test that everything is working
python test_benchmark_setup.py
```

## 🏃‍♂️ Running Benchmarks

### Quick Demo (Recommended for first time)
```bash
# Run a quick demonstration with small batch size
bash demo_benchmark.sh
```

### Full Benchmark
```bash
# Run complete benchmark with default settings
bash run_gpu_benchmark.sh

# Or with custom settings
bash run_gpu_benchmark.sh --batch_size 32 --num_workers 4
```

### Advanced Usage
```bash
# Skip single GPU benchmark (if you only want multi-GPU results)
bash run_gpu_benchmark.sh --skip_single_gpu

# Skip multi-GPU benchmark (if you only want single GPU results)
bash run_gpu_benchmark.sh --skip_multi_gpu

# Custom batch size and output directory
bash run_gpu_benchmark.sh --batch_size 16 --output_dir my_benchmark_results
```

## 📊 Understanding the Results

### Generated Files

After running a benchmark, you'll find these files in the output directory:

```
benchmark_results_TIMESTAMP/
├── gpu_benchmark_comparison.png     # Main comparison visualization
├── performance_summary_table.png    # Detailed performance table
└── benchmark_results.json          # Raw data and metrics
```

### Key Metrics

- **Training Time**: Total time to complete 1 epoch
- **Throughput**: Samples processed per second
- **Memory Usage**: GPU memory consumption over time
- **Speedup**: Performance improvement with multi-GPU
- **Efficiency**: How well the scaling works (ideal is linear)

### Example Results

```
🎉 BENCHMARK COMPLETE!
======================================
Time Speedup: 6.45x
Multi-GPU Efficiency: 80.6%
Results saved to: benchmark_results_20250809_143022/
======================================
```

## 📈 Interpreting Performance

### Speedup Analysis

- **Perfect scaling**: 8x speedup with 8 GPUs (100% efficiency)
- **Good scaling**: 6-7x speedup (75-87% efficiency)
- **Acceptable scaling**: 4-6x speedup (50-75% efficiency)
- **Poor scaling**: <4x speedup (<50% efficiency)

### Factors Affecting Performance

1. **Batch Size**: Larger batches generally scale better
2. **Model Size**: Larger models benefit more from multi-GPU
3. **Communication Overhead**: Network between GPUs
4. **Memory Bandwidth**: GPU memory limitations
5. **Data Loading**: CPU bottlenecks in data pipeline

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   bash run_gpu_benchmark.sh --batch_size 16
   ```

2. **Slow Data Loading**
   ```bash
   # Reduce number of workers
   bash run_gpu_benchmark.sh --num_workers 2
   ```

3. **Distributed Training Fails**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Run only single GPU benchmark
   bash run_gpu_benchmark.sh --skip_multi_gpu
   ```

### System Requirements

- **CUDA-capable GPUs**: 8 GPUs recommended for full benchmark
- **GPU Memory**: At least 8GB per GPU (RTX 3090 with 24GB is ideal)
- **Python**: 3.8 or later
- **PyTorch**: 2.8.0 with CUDA support

## 📋 Script Reference

### Available Scripts

| Script | Purpose |
|--------|---------|
| `activate_ddpm_env.sh` | Activate environment and set paths |
| `test_benchmark_setup.py` | Verify system setup |
| `benchmark_gpu_training.py` | Main benchmark script |
| `run_gpu_benchmark.sh` | Benchmark launcher with options |
| `demo_benchmark.sh` | Quick demonstration |

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch_size` | Batch size per GPU | 32 |
| `--num_workers` | Data loader workers | 4 |
| `--output_dir` | Results directory | `benchmark_results_TIMESTAMP` |
| `--skip_single_gpu` | Skip single GPU test | False |
| `--skip_multi_gpu` | Skip multi-GPU test | False |

## 🔬 Technical Details

### Benchmark Methodology

1. **Dataset**: CIFAR-10 (50,000 training images)
2. **Model**: U-Net with 128 base channels, ~13M parameters
3. **Training**: Exactly 1 epoch for fair comparison
4. **Metrics**: Wall-clock time, GPU memory, samples/second
5. **Distributed**: NCCL backend with DDP

### Monitoring

- **GPU Memory**: Tracked every 50 steps
- **Training Loss**: Recorded for each batch
- **Batch Times**: Individual batch processing times
- **Throughput**: Calculated as total_samples / total_time

### Visualization Features

- **Side-by-side comparisons** of key metrics
- **Memory usage over time** plots
- **Efficiency analysis** charts
- **Detailed summary tables** with all metrics

## 💡 Tips for Best Results

1. **Warm-up runs**: Run a small benchmark first to warm up GPUs
2. **Consistent environment**: Close other GPU applications
3. **Monitor temperatures**: Ensure GPUs don't thermal throttle
4. **Network optimization**: Use InfiniBand if available
5. **Batch size tuning**: Find the largest batch that fits in memory

## 🤝 Contributing

To add new features or improve the benchmark:

1. **Add new metrics**: Modify `PerformanceTracker` class
2. **Enhance visualizations**: Update `create_visualizations()` function
3. **Add new model architectures**: Extend model configuration options
4. **Improve monitoring**: Add system-level metrics (CPU, network)

## 📚 References

- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Original diffusion model paper
- [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - Distributed training guide
- [NCCL](https://developer.nvidia.com/nccl) - Multi-GPU communication library

---

**Happy benchmarking!** 🚀✨
