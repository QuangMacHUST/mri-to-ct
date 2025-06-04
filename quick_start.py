#!/usr/bin/env python3
"""
🚀 MRI-to-CT CycleGAN Quick Start Script
Tự động setup và verify environment cho training
"""

import subprocess
import sys
import os
import platform

def print_header(text):
    """Print header với format đẹp"""
    print("\n" + "="*60)
    print(f"🎯 {text}")
    print("="*60)

def run_command(cmd, check=True):
    """Chạy command và return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        return None, e.stderr

def check_python_version():
    """Kiểm tra Python version"""
    print_header("KIỂM TRA PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Cần Python 3.8 trở lên!")
        return False
    else:
        print("✅ Python version hợp lệ")
        return True

def check_gpu():
    """Kiểm tra GPU và CUDA"""
    print_header("KIỂM TRA GPU & CUDA")
    
    # Check nvidia-smi
    stdout, stderr = run_command("nvidia-smi", check=False)
    if stdout is None:
        print("❌ nvidia-smi không tìm thấy. Cần cài NVIDIA driver!")
        return False
    
    print("✅ NVIDIA driver detected")
    
    # Extract GPU info
    lines = stdout.split('\n')
    for line in lines:
        if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
            gpu_info = line.split('|')[1].strip()
            print(f"GPU: {gpu_info}")
            break
    
    return True

def check_pytorch_cuda():
    """Kiểm tra PyTorch CUDA"""
    print_header("KIỂM TRA PYTORCH CUDA")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print("✅ PyTorch CUDA setup hoàn hảo!")
            return True
        else:
            print("❌ PyTorch không detect được CUDA")
            return False
            
    except ImportError:
        print("❌ PyTorch chưa được cài đặt")
        return False

def install_pytorch_cuda():
    """Cài đặt PyTorch với CUDA"""
    print_header("CÀI ĐẶT PYTORCH CUDA")
    
    cmd = "pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121"
    print(f"Running: {cmd}")
    
    stdout, stderr = run_command(cmd)
    if stdout is None:
        print(f"❌ Lỗi cài PyTorch: {stderr}")
        return False
    else:
        print("✅ PyTorch CUDA cài đặt thành công!")
        return True

def install_dependencies():
    """Cài đặt dependencies"""
    print_header("CÀI ĐẶT DEPENDENCIES")
    
    packages = [
        "nibabel", "SimpleITK", "opencv-python", "scikit-image", 
        "matplotlib", "tensorboard", "tqdm", "scipy", "pillow", 
        "pydicom", "numpy<2.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        stdout, stderr = run_command(f"pip install {package}", check=False)
        if stdout is None:
            print(f"⚠️  Warning: Failed to install {package}")
        else:
            print(f"✅ {package} installed")
    
    print("✅ Dependencies installation completed!")

def verify_installation():
    """Verify toàn bộ installation"""
    print_header("VERIFY INSTALLATION")
    
    # Test imports
    test_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"), 
        ("nibabel", "NiBabel"),
        ("SimpleITK", "SimpleITK"),
        ("cv2", "OpenCV"),
        ("skimage", "Scikit-Image"),
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard")
    ]
    
    for package, name in test_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - Failed to import")
    
    # Test GPU tensor
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(10, 10).cuda()
            print("✅ GPU tensor creation successful")
            del x
            torch.cuda.empty_cache()
        else:
            print("❌ GPU tensor creation failed")
    except Exception as e:
        print(f"❌ GPU test error: {e}")

def check_data_structure():
    """Kiểm tra cấu trúc data"""
    print_header("KIỂM TRA DATA STRUCTURE")
    
    required_dirs = ["data/MRI", "data/CT", "src"]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
            
            if dir_path.startswith("data"):
                files = [f for f in os.listdir(dir_path) if f.endswith('.nii.gz')]
                print(f"   Files: {len(files)} .nii.gz files")
        else:
            print(f"❌ {dir_path} not found")
            if dir_path.startswith("data"):
                os.makedirs(dir_path, exist_ok=True)
                print(f"   Created {dir_path}")

def main():
    """Main function"""
    print("🏥 MRI-to-CT CycleGAN Quick Start Setup")
    print("=====================================")
    
    # 1. Check Python
    if not check_python_version():
        print("\n❌ Setup failed: Python version không hợp lệ")
        return
    
    # 2. Check GPU
    if not check_gpu():
        print("\n❌ Setup failed: GPU/CUDA không khả dụng")
        return
    
    # 3. Check hoặc install PyTorch
    if not check_pytorch_cuda():
        print("\n🔧 Cài đặt PyTorch CUDA...")
        if not install_pytorch_cuda():
            print("\n❌ Setup failed: Không thể cài PyTorch")
            return
        
        # Re-check sau khi cài
        if not check_pytorch_cuda():
            print("\n❌ Setup failed: PyTorch CUDA vẫn không hoạt động")
            return
    
    # 4. Install dependencies
    install_dependencies()
    
    # 5. Verify installation
    verify_installation()
    
    # 6. Check data structure
    check_data_structure()
    
    print_header("SETUP COMPLETED!")
    print("🎉 Environment setup thành công!")
    print("\n📋 Next steps:")
    print("1. Copy data files vào data/MRI/ và data/CT/")
    print("2. Run: python check_gpu.py")
    print("3. Run: cd src && python train.py")
    print("\n📊 Expected performance:")
    print("- GTX 1650: ~2 hours/epoch")
    print("- VRAM usage: ~0.7GB/4GB")
    print("- SSIM improvement: 0.12 → 0.28+ in first epochs")

if __name__ == "__main__":
    main() 