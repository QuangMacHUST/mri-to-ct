# 🏥 MRI-to-CT CycleGAN - Setup Instructions

Hướng dẫn setup dự án CycleGAN chuyển đổi ảnh MRI thành CT cho lập kế hoạch xạ trị.

## 🎯 **System Requirements**

### **Hardware:**
- **GPU**: NVIDIA GTX 1650 hoặc tương đương (tối thiểu 4GB VRAM)
- **RAM**: 8GB+ 
- **Storage**: 10GB+ trống

### **Software:**
- **OS**: Windows 10/11, Linux Ubuntu 18.04+
- **Python**: 3.8-3.12
- **CUDA**: 11.7+ (driver 470+)

## 🚀 **Quick Setup**

### **1. Clone Repository**
```bash
git clone <repository-url>
cd mri-to-ct
```

### **2. Kiểm tra CUDA**
```bash
nvidia-smi
```
Đảm bảo CUDA version ≥ 11.7

### **3. Tạo Python Environment**
```bash
# Conda (recommended)
conda create -n mri-ct python=3.11
conda activate mri-ct

# hoặc venv
python -m venv mri-ct
# Windows:
mri-ct\Scripts\activate
# Linux:
source mri-ct/bin/activate
```

### **4. Cài đặt PyTorch với CUDA**
```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### **5. Cài đặt Dependencies**
```bash
pip install nibabel SimpleITK opencv-python scikit-image matplotlib tensorboard tqdm scipy pillow pydicom
```

### **6. Verify Installation**
```bash
python check_gpu.py
```
Expected output:
```
✅ CUDA available: True
✅ GPU 0: NVIDIA GeForce GTX 1650
✅ GPU tensor creation: Successful
```

## 📁 **Data Setup**

### **Cấu trúc thư mục:**
```
mri-to-ct/
├── data/
│   ├── MRI/          # Đặt file brain_001.nii.gz đến brain_046.nii.gz
│   ├── CT/           # Đặt file brain_001.nii.gz đến brain_046.nii.gz
│   └── Test/         # (optional) data test
├── src/              # Source code
└── ...
```

### **Data format:**
- **Format**: `.nii.gz` (NIfTI compressed)
- **Naming**: `brain_001.nii.gz`, `brain_002.nii.gz`, ..., `brain_046.nii.gz`
- **Size**: 46 cặp ảnh MRI-CT tương ứng

## 🏃‍♂️ **Training**

### **1. Test Memory**
```bash
python check_gpu.py
```

### **2. Start Training**
```bash
cd src
python train.py
```

### **3. Monitor Training**
```bash
# Tensorboard
tensorboard --logdir=logs

# Check outputs
ls checkpoints/  # Model checkpoints
ls samples/      # Sample images
```

## 📊 **Expected Performance**

### **Training Speed:**
- **GTX 1650**: ~2 hours/epoch
- **Total time**: ~200 hours cho 100 epochs

### **Memory Usage:**
- **VRAM**: ~0.7GB / 4GB
- **RAM**: ~2-4GB

### **Metrics Progression:**
- **Epoch 0**: SSIM ~0.12
- **Epoch 1**: SSIM ~0.28
- **Epoch 10+**: SSIM >0.5 (expected)

## 🛠 **Troubleshooting**

### **GPU Issues:**
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Memory issues
# Giảm batch_size trong src/train.py từ 1 xuống 1 (đã tối ưu)
# Hoặc giảm n_residual_blocks từ 6 xuống 4
```

### **Import Errors:**
```bash
# Missing cv2
pip install opencv-python

# Missing SimpleITK
pip install SimpleITK

# Missing nibabel
pip install nibabel
```

### **Training Errors:**
```bash
# Negative strides error
# Đã fix trong code, restart training

# SSIM win_size error  
# Đã fix trong code, restart training

# Out of memory
# Giảm batch_size hoặc image size
```

## 📝 **Configuration**

### **Model Settings (src/train.py):**
```python
config = {
    'batch_size': 1,          # Cho GTX 1650 4GB
    'n_residual_blocks': 6,   # Giảm nếu out of memory
    'num_epochs': 100,        # Tăng nếu muốn training lâu hơn
    'lr_G': 0.0002,          # Learning rate Generator
    'lr_D': 0.0002,          # Learning rate Discriminator
}
```

### **Loss Weights:**
```python
lambda_cycle = 10.0       # Cycle consistency
lambda_identity = 5.0     # Identity loss  
lambda_perceptual = 1.0   # Perceptual loss
lambda_adversarial = 1.0  # Adversarial loss
```

## 🎯 **Next Steps**

1. **Start training**: `cd src && python train.py`
2. **Monitor progress**: Check `logs/`, `checkpoints/`, `samples/`
3. **Test model**: `python test.py` (sau khi training)
4. **Evaluate results**: Check metrics trong tensorboard

## 📞 **Support**

- **Hardware**: NVIDIA GTX 1650+ với 4GB+ VRAM
- **Software**: PyTorch 2.2.2 + CUDA 12.1
- **Data**: 46 cặp ảnh MRI-CT brain scans (.nii.gz)

---
**Status**: ✅ Tested successfully trên GTX 1650 4GB
**Performance**: SSIM 0.12 → 0.28 trong 2 epochs đầu 