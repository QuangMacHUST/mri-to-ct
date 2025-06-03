#!/usr/bin/env python3
import torch

print("=" * 50)
print("KIỂM TRA GPU VÀ CUDA")
print("=" * 50)

print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    # Test GPU memory
    device = torch.device('cuda')
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(device)/1024**2:.1f} MB")
    
    # Test tensor creation
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print("✅ GPU tensor creation successful")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ GPU tensor creation failed: {e}")
else:
    print("❌ CUDA not available")
    print("Có thể cần cài PyTorch với CUDA support")

print("=" * 50) 