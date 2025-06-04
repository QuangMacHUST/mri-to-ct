#!/usr/bin/env python3
"""
Script kiểm tra GPU GTX 1650 và memory cho training MRI-to-CT
"""

import torch
import torch.nn as nn

def check_gpu_status():
    """Kiểm tra trạng thái GPU chi tiết"""
    print("=" * 50)
    print("KIỂM TRA GPU VÀ CUDA")
    print("=" * 50)
    
    # Basic CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA không khả dụng!")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # GPU details
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Major: {props.major}, Minor: {props.minor}")
        print(f"  Multi-processor count: {props.multi_processor_count}")
    
    # Memory status
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    
    return True

def test_model_memory():
    """Test memory usage với model tương tự"""
    print("\n" + "=" * 50)
    print("TEST MEMORY USAGE VỚI MODEL MRI-TO-CT")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test với model nhỏ trước
        print("Testing với model Generator nhỏ...")
        
        # Tạo model test nhỏ
        model = nn.Sequential(
            nn.Conv2d(1, 64, 7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 7, padding=3),
            nn.Tanh()
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test với batch size khác nhau
        batch_sizes = [1, 2, 4]
        input_size = (256, 256)
        
        for batch_size in batch_sizes:
            try:
                torch.cuda.empty_cache()
                
                # Tạo input tensor
                x = torch.randn(batch_size, 1, *input_size).to(device)
                
                # Forward pass
                with torch.no_grad():
                    y = model(x)
                
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ Batch size {batch_size}: {memory_used:.2f} GB")
                
                # Thử backward pass
                y = model(x)
                loss = y.mean()
                loss.backward()
                
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ Batch size {batch_size} (with backward): {memory_used:.2f} GB")
                
                del x, y, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Batch size {batch_size}: Out of memory")
                else:
                    print(f"❌ Batch size {batch_size}: {e}")
                    
        print("\n📊 RECOMMENDATIONS:")
        print("- Batch size 1: An toàn cho GTX 1650 4GB")
        print("- Batch size 2: Có thể hoạt động với model nhỏ")
        print("- Batch size 4+: Có thể out of memory")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test memory: {e}")
        return False

def estimate_cyclegan_memory():
    """Ước tính memory cho CycleGAN"""
    print("\n" + "=" * 50)
    print("ƯỚC TÍNH MEMORY CHO CYCLEGAN")
    print("=" * 50)
    
    # Thông số model
    configs = [
        {"name": "Small (6 res blocks)", "res_blocks": 6, "batch_size": 1},
        {"name": "Medium (9 res blocks)", "res_blocks": 9, "batch_size": 1},
        {"name": "Large (12 res blocks)", "res_blocks": 12, "batch_size": 1},
    ]
    
    for config in configs:
        # Ước tính parameters
        # Generator: ~11M params per 3 residual blocks
        gen_params = config["res_blocks"] * 3.7e6  
        total_params = gen_params * 2 + 2.7e6  # 2 generators + 2 discriminators
        
        # Ước tính memory (rough estimate)
        # Model weights: 4 bytes per param
        # Forward activations: ~batch_size * 256 * 256 * channels * 4 bytes * multiple layers
        # Backward gradients: 2x forward
        
        model_memory = total_params * 4 / 1024**3  # GB
        activation_memory = config["batch_size"] * 256 * 256 * 64 * 4 * 10 / 1024**3  # GB
        total_estimated = model_memory + activation_memory * 3  # 3x for forward+backward+optimizer
        
        print(f"{config['name']}:")
        print(f"  Parameters: {total_params/1e6:.1f}M")
        print(f"  Estimated memory: {total_estimated:.2f} GB")
        
        if total_estimated < 3.5:
            print(f"  Status: ✅ An toàn cho GTX 1650")
        elif total_estimated < 4.0:
            print(f"  Status: ⚠️  Có thể hoạt động")
        else:
            print(f"  Status: ❌ Có thể out of memory")
        print()

def main():
    """Main function"""
    print("KIỂM TRA GPU CHO TRAINING MRI-TO-CT CYCLEGAN")
    print("GPU: NVIDIA GTX 1650 (4GB VRAM)")
    
    if not check_gpu_status():
        return
    
    test_model_memory()
    estimate_cyclegan_memory()
    
    print("\n" + "=" * 50)
    print("KHUYẾN NGHỊ FINAL:")
    print("=" * 50)
    print("✅ Sử dụng batch_size = 1")
    print("✅ Sử dụng 6 residual blocks thay vì 9")
    print("✅ Monitor GPU memory trong quá trình training")
    print("✅ Sử dụng torch.cuda.empty_cache() định kỳ")
    print("⚠️  Nếu out of memory, giảm image size xuống 128x128")

if __name__ == "__main__":
    main() 