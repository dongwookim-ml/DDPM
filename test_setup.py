#!/usr/bin/env python3
"""
Test script to verify DDPM environment setup
"""

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
            print(f"✓ Current CUDA device: {torch.cuda.current_device()}")
        
        import torchvision
        print(f"✓ torchvision {torchvision.__version__} imported successfully")
        
        import numpy as np
        print(f"✓ numpy {np.__version__} imported successfully")
        
        import einops
        print(f"✓ einops {einops.__version__} imported successfully")
        
        import wandb
        print(f"✓ wandb {wandb.__version__} imported successfully")
        
        import joblib
        print(f"✓ joblib {joblib.__version__} imported successfully")
        
        # Test DDPM package import
        import ddpm
        print("✓ DDPM package imported successfully")
        
        # Test individual modules
        from ddpm import diffusion, unet, ema, utils, script_utils
        print("✓ All DDPM modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic PyTorch functionality"""
    try:
        import torch
        
        # Test tensor creation
        x = torch.randn(2, 3, 32, 32)
        print(f"✓ Created tensor with shape: {x.shape}")
        
        # Test basic operations
        y = torch.nn.functional.relu(x)
        print(f"✓ Applied ReLU operation")
        
        # Test GPU if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"✓ Moved tensor to GPU: {x_gpu.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("DDPM Environment Setup Test")
    print("=" * 50)
    
    print("\n1. Testing package imports...")
    imports_ok = test_imports()
    
    print("\n2. Testing basic functionality...")
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if imports_ok and functionality_ok:
        print("🎉 Environment setup completed successfully!")
        print("You can now run DDPM training and inference scripts.")
    else:
        print("❌ Environment setup has issues. Please check the errors above.")
    print("=" * 50)
