# DDPM: Denoising Diffusion Probabilistic Models

A complete implementation of Denoising Diffusion Probabilistic Models (DDPM) for image generation, with comprehensive benchmarking tools and organized codebase structure.

## 📁 Repository Structure

```
DDPM/
├── 📋 README.md                      # This file
├── 🔧 setup.py                       # Package installation
├── 📦 requirements.txt               # Dependencies
├── 🚫 .gitignore                     # Git ignore rules
│
├── 📦 ddpm/                          # Core DDPM package
│   ├── 🧠 models/                    # Model implementations
│   │   ├── diffusion.py              # GaussianDiffusion class
│   │   └── unet.py                   # U-Net architecture
│   ├── 🎯 training/                  # Training utilities
│   │   ├── ema.py                    # Exponential Moving Average
│   │   └── utils.py                  # Training utilities
│   └── 🔧 utils/                     # Utility functions
│       └── script_utils.py           # Script helpers
│
├── 🚀 scripts/                       # Training & inference scripts
│   ├── train/                        # Training scripts
│   │   ├── train_cifar.py            # Single GPU CIFAR-10 training
│   │   └── train_cifar_distributed.py # Multi-GPU training
│   ├── inference/                    # Inference scripts
│   │   └── sample_images.py          # Generate samples
│   └── launcher_scripts/             # Launcher utilities
│       └── launch_distributed_training.sh
│
├── 📊 benchmarks/                    # GPU benchmarking system
│   ├── benchmark_gpu_training.py     # Core benchmark script
│   ├── test_benchmark_setup.py       # Setup verification
│   ├── scripts/                      # Benchmark launchers
│   │   ├── run_gpu_benchmark.sh      # Full benchmark
│   │   └── demo_benchmark.sh         # Quick demo
│   └── README.md                     # Benchmark documentation
│
├── 🧪 tests/                         # Test files
│   └── test_setup.py                 # Environment tests
│
├── 🔧 tools/                         # Utility tools
│   ├── activate_ddpm_env.sh          # Environment activation
│   ├── show_benchmark_summary.py     # Benchmark summary
│   └── get-pip.py                    # Pip installer
│
└── 📚 docs/                          # Documentation & resources
    └── resources/                    # Papers, slides, examples
        ├── diffusion_models_report.pdf
        ├── diffusion_models_talk_slides.pdf
        ├── diffusion_sequence_mnist.gif
        └── samples_linear_200.png
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/dongwookim-ml/DDPM.git
cd DDPM

# Activate environment
source tools/activate_ddpm_env.sh
```

### 2. Training
```bash
# Single GPU training
python scripts/train/train_cifar.py --iterations 100000 --batch_size 128

# Multi-GPU training  
bash scripts/launcher_scripts/launch_distributed_training.sh

# Or use the training script directly
python scripts/train/train_cifar_distributed.py --iterations 100000 --batch_size 64
```

### 3. Inference
```bash
# Generate samples from trained model
python scripts/inference/sample_images.py --model_path path/to/checkpoint.pth
```

### 4. Benchmarking
```bash
# Quick demo (recommended first)
bash benchmarks/scripts/demo_benchmark.sh

# Full GPU benchmark (1 GPU vs 8 GPUs)
bash benchmarks/scripts/run_gpu_benchmark.sh

# Custom benchmark
bash benchmarks/scripts/run_gpu_benchmark.sh --batch_size 32 --num_workers 8
```

## 📊 Features

### Core Implementation
- **Complete DDPM implementation** with U-Net architecture
- **Support for conditional and unconditional generation**
- **Exponential Moving Average (EMA)** for stable training
- **Multiple noise schedules** (linear, cosine)
- **Distributed training** with PyTorch DDP

### Benchmarking System
- **1 GPU vs 8 GPU performance comparison**
- **Real-time performance monitoring**
- **Comprehensive metrics** (throughput, scaling efficiency, memory usage)
- **Automatic visualization generation**
- **CSV and JSON result export**

### Developer Tools
- **Organized modular structure**
- **Comprehensive testing suite**
- **Environment management tools**
- **Documentation and examples**

## 🔧 Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU(s)
- PyTorch 2.0+

### Setup
```bash
# Install package in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## 📚 Documentation

- **Benchmarking**: See `benchmarks/README.md` for detailed benchmarking documentation
- **Training**: Check `scripts/train/` for training examples
- **Models**: Explore `ddpm/models/` for model implementations
- **Resources**: Find papers and examples in `docs/resources/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the modular structure
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project implements the DDPM algorithm from:
- **Paper**: "Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
- **ArXiv**: https://arxiv.org/abs/2006.11239

## 🏆 Performance

With 8x RTX 3090 GPUs:
- **Training speedup**: ~6-7x compared to single GPU
- **Scaling efficiency**: ~75-85%
- **Memory usage**: ~8-12GB per GPU
- **Throughput**: ~500-1000 samples/second

## 🔗 Links

- [Original DDPM Paper](https://arxiv.org/abs/2006.11239)
- [Repository](https://github.com/dongwookim-ml/DDPM)
- [Issues](https://github.com/dongwookim-ml/DDPM/issues)
