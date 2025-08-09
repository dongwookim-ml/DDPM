#!/usr/bin/env python3
"""
Quick test script to verify benchmark setup
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import os

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing data loading...")
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Test dataset download and loading
    dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Test one batch
    for images, labels in loader:
        print(f"✅ Data loading successful: batch shape {images.shape}")
        break
    
    return True

def test_model_creation():
    """Test model creation and basic operations."""
    print("🧪 Testing model creation...")
    
    from ddpm.utils import script_utils
    import argparse
    
    # Test model creation
    model_args = argparse.Namespace(
        # Diffusion process parameters
        num_timesteps=100,  # Smaller for testing
        schedule="linear",
        loss_type="l2",
        use_labels=True,
        schedule_low=1e-4,
        schedule_high=0.02,
        
        # U-Net architecture parameters
        base_channels=64,  # Smaller for testing
        channel_mults=(1, 2, 2),  # Smaller for testing
        num_res_blocks=1,  # Smaller for testing
        time_emb_dim=64 * 4,  # Smaller for testing
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),
        
        # EMA parameters
        ema_decay=0.9999,
        ema_update_rate=1,
    )
    
    model = script_utils.get_diffusion_from_args(model_args)
    
    # Test forward pass
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy data
    x = torch.randn(2, 3, 32, 32).to(device)
    y = torch.randint(0, 10, (2,)).to(device)
    
    # Test forward pass
    loss = model(x, y)
    print(f"✅ Model creation and forward pass successful: loss = {loss.item():.4f}")
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {param_count:,}")
    
    return True

def test_gpu_detection():
    """Test GPU detection and CUDA functionality."""
    print("🧪 Testing GPU detection...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ Detected {gpu_count} GPUs")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Test basic CUDA operations
    device = torch.device('cuda:0')
    x = torch.randn(100, 100).to(device)
    y = torch.mm(x, x.t())
    print(f"✅ CUDA operations working: result shape {y.shape}")
    
    return True

def distributed_worker(rank, world_size):
    """Worker function for distributed training test."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Different port from benchmark
    
    try:
        dist.init_process_group("nccl", rank=rank, world_size=2)
        torch.cuda.set_device(rank)
        
        # Test basic distributed operations
        tensor = torch.ones(2, 2).cuda(rank)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        if rank == 0:
            print(f"✅ Distributed setup successful: tensor sum = {tensor[0,0].item()}")
        
        dist.destroy_process_group()
        return True
    except Exception as e:
        if rank == 0:
            print(f"❌ Distributed setup failed: {e}")
        return False


def test_distributed_setup():
    """Test distributed training setup."""
    print("🧪 Testing distributed setup...")
    
    if torch.cuda.device_count() >= 2:
        mp.spawn(distributed_worker, args=(2,), nprocs=2, join=True)
    else:
        print("⚠️  Need at least 2 GPUs for distributed test, skipping...")
    
    return True

def main():
    """Run all tests."""
    print("🔧 DDPM Benchmark Setup Verification")
    print("=" * 50)
    
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Distributed Setup", test_distributed_setup),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Benchmark is ready to run.")
        print("\nTo run the full benchmark:")
        print("  bash run_gpu_benchmark.sh")
        print("  bash run_gpu_benchmark.sh --batch_size 16  # Smaller batch size")
        print("  bash run_gpu_benchmark.sh --skip_single_gpu  # Only multi-GPU")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
