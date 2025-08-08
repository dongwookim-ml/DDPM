# DDPM - Denoising Diffusion Probabilistic Models

A clean, well-documented implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for image generation. This implementation is based on the paper ["Denoising Diffusion Probabilistic Models" by Ho et al. (2020)](https://arxiv.org/abs/2006.11239).

## 🌟 Key Features

- **Complete DDPM Implementation**: Forward diffusion process, reverse denoising, and sampling
- **U-Net Architecture**: Time-conditioned U-Net with attention mechanisms for noise prediction
- **Class Conditioning**: Support for conditional generation (e.g., specific CIFAR-10 classes)
- **EMA Training**: Exponential Moving Average for stable and high-quality generation
- **Flexible Schedules**: Both linear and cosine noise schedules
- **Comprehensive Documentation**: Every function and class is thoroughly documented

## 📁 Project Structure

```
DDPM/
├── ddpm/                           # Main package
│   ├── __init__.py                # Package initialization with imports
│   ├── diffusion.py               # Core diffusion process implementation
│   ├── unet.py                    # U-Net architecture for noise prediction
│   ├── ema.py                     # Exponential Moving Average utilities
│   ├── utils.py                   # Helper functions
│   └── script_utils.py            # Training/inference utilities
├── scripts/                       # Training and sampling scripts
│   ├── train_cifar.py             # CIFAR-10 training script
│   └── sample_images.py           # Image generation script
├── resources/                     # Documentation and examples
│   ├── diffusion_models_report.pdf
│   ├── diffusion_models_talk_slides.pdf
│   ├── diffusion_sequence_mnist.gif
│   └── samples_linear_200.png
├── setup.py                       # Package installation
└── README.md                      # This file
```

## 🧠 How DDPM Works

### 1. Forward Process (Adding Noise)
The forward process gradually adds Gaussian noise to clean images over T timesteps:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

- **x_0**: Original clean image
- **x_T**: Pure noise
- **β_t**: Noise schedule controlling how much noise to add at step t

### 2. Reverse Process (Denoising)
The reverse process learns to remove noise step by step using a neural network:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

- The U-Net predicts the noise ε_θ(x_t, t) added at each step
- The model is trained to minimize: E[||ε - ε_θ(x_t, t)||²]

### 3. Generation Process
To generate new images:
1. Start with random noise x_T ~ N(0, I)
2. Iteratively denoise: x_{t-1} = (x_t - predicted_noise) / √α_t + noise
3. Continue until x_0 (clean image)

## 🏗️ Architecture Components

### GaussianDiffusion (`ddpm/diffusion.py`)
- **Core diffusion logic**: Forward noising and reverse denoising
- **Beta schedules**: Linear and cosine noise schedules
- **Loss computation**: Training loss for noise prediction
- **Sampling methods**: Generate images and visualize diffusion sequences

### UNet (`ddpm/unet.py`)
- **PositionalEmbedding**: Sinusoidal encoding for timesteps
- **ResidualBlock**: Core building blocks with time/class conditioning
- **AttentionBlock**: Self-attention for capturing long-range dependencies
- **Downsample/Upsample**: Spatial resolution changes with skip connections

### EMA (`ddpm/ema.py`)
- **Exponential Moving Average**: Maintains stable model weights
- **Better Generation**: EMA weights often produce higher quality samples

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd DDPM

# Install dependencies
pip install torch torchvision numpy wandb
```

### Training on CIFAR-10

```bash
# Basic training
python scripts/train_cifar.py --iterations 100000 --batch_size 128

# Training with class conditioning
python scripts/train_cifar.py --use_labels --iterations 100000

# Training with Weights & Biases logging
python scripts/train_cifar.py --log_to_wandb --project_name "ddpm-cifar10"
```

### Generating Images

```bash
# Generate 1000 images
python scripts/sample_images.py 
    --model_path ./logs/model-100000.pth 
    --save_dir ./generated_images 
    --num_images 1000

# Generate class-conditional images
python scripts/sample_images.py 
    --model_path ./logs/model-100000.pth 
    --save_dir ./generated_images 
    --use_labels 
    --num_images 1000
```

## ⚙️ Configuration Options

### Diffusion Parameters
- `--num_timesteps`: Number of diffusion steps (default: 1000)
- `--schedule`: Noise schedule type ("linear" or "cosine")
- `--loss_type`: Loss function ("l1" or "l2")

### Model Architecture
- `--base_channels`: Base number of U-Net channels (default: 128)
- `--channel_mults`: Channel multipliers (default: (1, 2, 2, 2))
- `--attention_resolutions`: Where to apply attention (default: (1,))
- `--dropout`: Dropout rate (default: 0.1)

### Training
- `--learning_rate`: Adam learning rate (default: 2e-4)
- `--batch_size`: Training batch size (default: 128)
- `--ema_decay`: EMA decay rate (default: 0.9999)

## 📊 Results

The model can generate high-quality images after training:

- **CIFAR-10**: Achieves competitive FID scores
- **Class Conditioning**: Can generate specific object classes
- **Progressive Denoising**: Visualize the generation process

## 🔬 Key Implementation Details

### 1. Reparameterization Trick
Instead of learning means directly, the model predicts noise:
```python
# Forward process: add noise directly to x_0
x_t = √(α̅_t) * x_0 + √(1-α̅_t) * ε

# Reverse: predict noise and compute x_{t-1}
x_{t-1} = (x_t - β_t/√(1-α̅_t) * ε_θ(x_t,t)) / √α_t
```

### 2. Time Conditioning
Timesteps are embedded using sinusoidal positional encoding and injected into U-Net residual blocks.

### 3. Class Conditioning (Optional)
Class labels are embedded and added to the time conditioning for controlled generation.

### 4. Attention Mechanism
Self-attention layers capture global image structure at lower resolutions.

## 📚 References

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
2. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) - Nichol & Dhariwal, 2021
3. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) - Dhariwal & Nichol, 2021

## 🤝 Contributing

This implementation prioritizes clarity and educational value. All code is thoroughly documented with:
- Detailed docstrings for every function/class
- Inline comments explaining key concepts
- Mathematical formulations where relevant

## 📄 License

This project is open source and available under the MIT License.

---

*This implementation is designed to be educational and easily understandable. Each component is well-documented to help learn the concepts behind diffusion models.*

An implementation of Denoising Diffusion Probabilistic Models for image generation written in PyTorch. This roughly follows the original code by Ho et al. Unlike their implementation, however, my model allows for class conditioning through bias in residual blocks. 

## Experiments

I have trained the model on MNIST and CIFAR-10 datasets. The model seemed to converge well on the MNIST dataset, producing realistic samples. However, I am yet to report the same CIFAR-10 quality that Ho. et al. provide in their paper. Here are the samples generated with a linear schedule after 2000 epochs:

![Samples after 2000 epochs](resources/samples_linear_200.png)

Here is a sample of a diffusion sequence on MNIST:

<p align="center">
  <img src="resources/diffusion_sequence_mnist.gif" />
</p>


## Resources

I gave a talk about diffusion models, NCSNs, and their applications in audio generation. The [slides are available here](resources/diffusion_models_talk_slides.pdf).

I also compiled a report with what are, in my opinion, the most crucial findings on the topic of denoising diffusion models. It is also [available in this repository](resources/diffusion_models_report.pdf).


## Acknowledgements

I used [Phil Wang's implementation](https://github.com/lucidrains/denoising-diffusion-pytorch) and [the official Tensorflow repo](https://github.com/hojonathanho/diffusion) as a reference for my work.

## Citations

```bibtex
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@inproceedings{anonymous2021improved,
    title   = {Improved Denoising Diffusion Probabilistic Models},
    author  = {Anonymous},
    booktitle = {Submitted to International Conference on Learning Representations},
    year    = {2021},
    url     = {https://openreview.net/forum?id=-NEXDKk8gZ},
    note    = {under review}
}
