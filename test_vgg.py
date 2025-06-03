#!/usr/bin/env python3
"""
Script test nhanh để kiểm tra VGG model có hoạt động không
"""

import torch
import torchvision.models as models

def test_vgg_loading():
    """Test load VGG19 với các phương pháp khác nhau"""
    print("Testing VGG19 loading...")
    
    # Method 1: Với weights mới (PyTorch 1.13+)
    try:
        from torchvision.models import VGG19_Weights
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        print("✅ Method 1 (weights): SUCCESS")
        return vgg
    except Exception as e:
        print(f"❌ Method 1 (weights): FAILED - {e}")
    
    # Method 2: Với pretrained=True (cũ)
    try:
        vgg = models.vgg19(pretrained=True)
        print("✅ Method 2 (pretrained=True): SUCCESS")
        return vgg
    except Exception as e:
        print(f"❌ Method 2 (pretrained=True): FAILED - {e}")
    
    # Method 3: Không pretrained
    try:
        vgg = models.vgg19(pretrained=False)
        print("⚠️  Method 3 (pretrained=False): SUCCESS (nhưng không có pretrained weights)")
        return vgg
    except Exception as e:
        print(f"❌ Method 3 (pretrained=False): FAILED - {e}")
    
    return None

def test_vgg_forward():
    """Test forward pass của VGG"""
    vgg = test_vgg_loading()
    
    if vgg is None:
        print("❌ Không thể load VGG model")
        return False
    
    print("\nTesting VGG forward pass...")
    
    try:
        # Test input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Forward pass qua features
        with torch.no_grad():
            features = vgg.features(dummy_input)
        
        print(f"✅ Forward pass SUCCESS - Output shape: {features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("VGG MODEL TEST")
    print("=" * 50)
    
    success = test_vgg_forward()
    
    if success:
        print("\n🎉 VGG test PASSED - Training có thể hoạt động")
    else:
        print("\n💥 VGG test FAILED - Cần sửa PyTorch/torchvision")
        print("\nGợi ý:")
        print("1. pip uninstall torch torchvision")
        print("2. pip install torch==1.12.1 torchvision==0.13.1") 