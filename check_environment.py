#!/usr/bin/env python3
"""
Script kiểm tra môi trường và tương thích PyTorch/torchvision
Chạy trước khi training để đảm bảo mọi thứ hoạt động đúng
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Kiểm tra phiên bản Python"""
    print("="*50)
    print("KIỂM TRA PHIÊN BẢN PYTHON")
    print("="*50)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ CẢNH BÁO: Cần Python 3.7 trở lên")
        return False
    else:
        print("✅ Python version phù hợp")
        return True

def check_package_version(package_name, min_version=None):
    """Kiểm tra phiên bản của một package"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"{package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: CHƯA CÀI ĐẶT")
        return False

def check_pytorch_installation():
    """Kiểm tra PyTorch và tương thích CUDA"""
    print("\n" + "="*50)
    print("KIỂM TRA PYTORCH VÀ CUDA")
    print("="*50)
    
    try:
        import torch
        import torchvision
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"Torchvision version: {torchvision.__version__}")
        
        # Kiểm tra CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Lỗi import PyTorch: {e}")
        return False

def test_torchvision_models():
    """Kiểm tra xem torchvision models có hoạt động không"""
    print("\n" + "="*50)
    print("KIỂM TRA TORCHVISION MODELS")
    print("="*50)
    
    try:
        import torch
        import torchvision.models as models
        
        # Test load VGG19 
        print("Đang test VGG19...")
        
        # Thử method mới trước
        try:
            from torchvision.models import VGG19_Weights
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            print("✅ VGG19 với weights mới - OK")
        except:
            # Fallback to old method
            try:
                vgg = models.vgg19(pretrained=True)
                print("✅ VGG19 với pretrained=True - OK")
            except Exception as e:
                print(f"❌ Lỗi load VGG19: {e}")
                return False
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = vgg.features(dummy_input)
        print("✅ VGG19 forward pass - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi test torchvision models: {e}")
        return False

def check_medical_imaging_libs():
    """Kiểm tra thư viện xử lý ảnh y tế"""
    print("\n" + "="*50)
    print("KIỂM TRA THỬ VIỆN XỬ LÝ ẢNH Y TẾ")
    print("="*50)
    
    packages = ['nibabel', 'SimpleITK', 'cv2', 'skimage', 'scipy', 'numpy', 'matplotlib']
    all_ok = True
    
    for package in packages:
        if not check_package_version(package):
            all_ok = False
    
    return all_ok

def test_data_loading():
    """Kiểm tra xem có thể load dữ liệu không"""
    print("\n" + "="*50)
    print("KIỂM TRA LOAD DỮ LIỆU")
    print("="*50)
    
    import os
    
    data_dirs = ['data/MRI', 'data/CT']
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.nii.gz')]
            print(f"{dir_path}: {len(files)} files")
            if len(files) == 0:
                print(f"⚠️  Cảnh báo: Không có file .nii.gz trong {dir_path}")
        else:
            print(f"❌ Thư mục không tồn tại: {dir_path}")
            return False
    
    return True

def suggest_fixes():
    """Đưa ra gợi ý sửa lỗi"""
    print("\n" + "="*50)
    print("GỢI Ý SỬA LỖI")
    print("="*50)
    
    print("Nếu gặp lỗi 'torchvision::nms does not exist', hãy thử:")
    print("1. Gỡ cài đặt PyTorch/torchvision hiện tại:")
    print("   pip uninstall torch torchvision torchaudio")
    print()
    print("2. Cài đặt lại với phiên bản tương thích:")
    print("   # Cho CUDA 11.6:")
    print("   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116")
    print()
    print("   # Cho CPU only:")
    print("   pip install torch==1.12.1+cpu torchvision==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu")
    print()
    print("3. Hoặc cài đặt từ requirements.txt đã được cập nhật:")
    print("   pip install -r requirements.txt")
    print()
    print("4. Nếu vẫn lỗi, thử xóa cache pip:")
    print("   pip cache purge")

def main():
    """Hàm main chạy tất cả kiểm tra"""
    print("KIỂM TRA MÔI TRƯỜNG TRAINING MRI-TO-CT")
    print("Vui lòng đợi trong khi kiểm tra...")
    
    all_checks = [
        check_python_version(),
        check_pytorch_installation(),
        test_torchvision_models(),
        check_medical_imaging_libs(),
        test_data_loading()
    ]
    
    print("\n" + "="*50)
    print("TÓM TẮT KẾT QUẢ")
    print("="*50)
    
    if all(all_checks):
        print("✅ TẤT CẢ KIỂM TRA PASS - SẴN SÀNG TRAINING!")
    else:
        print("❌ CÓ VẤN ĐỀ CẦN SỬA")
        suggest_fixes()

if __name__ == "__main__":
    main() 