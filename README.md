# 🏥 MRI-to-CT Synthesis với CycleGAN

Dự án CycleGAN để chuyển đổi ảnh MRI thành CT mô phỏng cho lập kế hoạch xạ trị. **Đã test thành công trên NVIDIA GTX 1650 4GB.**

## 🚀 **Quick Start cho máy mới**

### **1. Automatic Setup (Recommended):**
```bash
git clone <repository-url>
cd mri-to-ct
python quick_start.py
```

### **2. Manual Setup:**
```bash
# Clone project
git clone <repository-url>
cd mri-to-ct

# Cài PyTorch CUDA
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# Cài dependencies  
pip install nibabel SimpleITK opencv-python scikit-image matplotlib tensorboard tqdm

# Verify
python check_gpu.py

# Training
cd src && python train.py
```

## 📊 **Kết quả đã verified**

### **Hardware tested:**
- **GPU**: NVIDIA GTX 1650 4GB VRAM
- **CUDA**: 12.1 
- **PyTorch**: 2.2.2+cu121

### **Performance:**
- **Training speed**: ~2 giờ/epoch với 36 samples
- **Memory usage**: 0.7GB VRAM / 4GB total  
- **SSIM improvement**: 0.12 → 0.28 (126% improvement) trong 2 epochs đầu
- **Model size**: 41.2M parameters

### **Training progression:**
```
Epoch 0: Train Loss: 5.0248 | Val Loss: 2.7144 | SSIM: 0.1252
Epoch 1: Train Loss: 2.2805 | Val Loss: 2.1206 | SSIM: 0.2824 ⬆️ 126%
```

## 📁 **Cấu trúc Project**

```
mri-to-ct/
├── 📄 quick_start.py          # Auto setup script
├── 📄 check_gpu.py            # GPU verification
├── 📄 requirements.txt        # Dependencies (tested versions)
├── 📄 SETUP.md               # Chi tiết setup instructions
├── 📄 CURSOR_CHAT_BACKUP.md  # Hướng dẫn backup chat history
├── 
├── 📁 data/
│   ├── 📁 MRI/               # brain_001.nii.gz → brain_046.nii.gz
│   ├── 📁 CT/                # brain_001.nii.gz → brain_046.nii.gz  
│   └── 📁 Test/              # Test data (optional)
│
├── 📁 src/                   # Source code
│   ├── 📄 train.py           # Training script (GPU optimized)
│   ├── 📄 test.py            # Testing script
│   ├── 📄 models.py          # CycleGAN architecture
│   ├── 📄 data_loader.py     # Data loading với preprocessing
│   ├── 📄 metrics.py         # Evaluation metrics
│   └── 📄 utils.py           # Utility functions
│
├── 📁 checkpoints/           # Model checkpoints (tạo khi training)
├── 📁 logs/                  # Tensorboard logs (tạo khi training)  
└── 📁 samples/               # Sample images (tạo khi training)
```

## 🎯 **Model Architecture**

### **CycleGAN:**
- **Generators**: 2 ResNet với 6 residual blocks (tối ưu cho 4GB VRAM)
- **Discriminators**: 2 PatchGAN 70×70
- **Total parameters**: 41,200,068

### **Loss Functions:**
- **Adversarial Loss**: Standard GAN loss
- **Cycle Consistency Loss**: λ = 10.0  
- **Identity Loss**: λ = 5.0
- **Perceptual Loss**: VGG19 features, λ = 1.0

### **Preprocessing:**
- **N4 Bias Correction**: SimpleITK implementation
- **Otsu Thresholding**: Binary mask tạo
- **Z-score Normalization**: Robust với outliers
- **Data Augmentation**: Rotation, flips, intensity scaling

## 📈 **Metrics Tracking**

- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)  
- **RMSE** (Root Mean Squared Error)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **NCC** (Normalized Cross Correlation)

## 💾 **Backup Chat History cho máy mới**

Xem chi tiết trong [`CURSOR_CHAT_BACKUP.md`](CURSOR_CHAT_BACKUP.md):

### **Quick backup:**
```bash
# Windows - Copy workspace folder
xcopy "%APPDATA%\Cursor\User\workspaceStorage" "D:\backup\cursor-chat\" /E /I

# Manual export
# Ctrl+A → Ctrl+C trong chat tab → Save to .md file
```

### **Essential commands nếu mất chat:**
```bash
# Setup
python quick_start.py

# hoặc manual
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install nibabel SimpleITK opencv-python scikit-image matplotlib tensorboard tqdm

# Test & Train
python check_gpu.py
cd src && python train.py
```

## 🛠 **Known Issues & Solutions**

| Issue | Solution | Status |
|-------|----------|--------|
| `torchvision::nms error` | Use PyTorch 2.2.2+cu121 | ✅ Fixed |
| `Negative strides` | Add `.copy()` trong data_loader.py | ✅ Fixed |
| `SSIM win_size error` | Auto-adjust win_size | ✅ Fixed | 
| `Out of memory` | batch_size=1, n_residual_blocks=6 | ✅ Optimized |
| `Windows multiprocessing` | num_workers=0 | ✅ Fixed |

## 📚 **Documentation**

- **[SETUP.md](SETUP.md)**: Chi tiết setup instructions
- **[README_DETAILED.md](README_DETAILED.md)**: Technical deep dive
- **[CURSOR_CHAT_BACKUP.md](CURSOR_CHAT_BACKUP.md)**: Chat backup guide

## 🎉 **Success Status**

✅ **Training đã verified thành công trên:**
- NVIDIA GTX 1650 4GB
- Windows 11 + Python 3.12  
- PyTorch 2.2.2 + CUDA 12.1
- 46 cặp ảnh brain MRI-CT

**Ready for deployment trên máy tính tương tự!**