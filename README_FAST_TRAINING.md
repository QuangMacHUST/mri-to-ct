# 🚀 Fast MRI-to-CT Training System

## Tổng quan

Hệ thống training nhanh này tách **preprocessing** ra khỏi **training loop** để tăng tốc đáng kể quá trình training. Thay vì preprocessing lại 42 bệnh nhân mỗi epoch, chúng ta:

1. **Preprocessing một lần** và lưu cache 
2. **Training nhanh** chỉ với augmentation

**Kết quả**: Training nhanh hơn **~450x** so với preprocessing realtime!

## 📊 Tại sao nhanh hơn?

### Trước đây (chậm):
```
Mỗi epoch: Preprocessing (42 × 45s) + Training = ~30 phút/epoch
100 epochs = ~50 giờ
```

### Bây giờ (nhanh):
```
Preprocessing 1 lần: ~30 phút
Mỗi epoch: Load cache + Augmentation = ~4s/epoch  
100 epochs = 30 phút + 7 phút = ~37 phút tổng cộng
```

## 🛠️ Cách sử dụng

### 1. Chạy toàn bộ pipeline (đơn giản nhất):

```bash
# Chạy cả preprocessing + training
python run_fast_training.py --epochs 50 --batch_size 6 --use_amp

# Với custom directories
python run_fast_training.py \
    --mri_dir data/MRI \
    --ct_dir data/CT \
    --cache_dir preprocessed_cache \
    --output_dir fast_training_output \
    --epochs 100 \
    --batch_size 8 \
    --use_amp
```

### 2. Hoặc chạy từng bước:

#### Bước 1: Preprocessing và cache (một lần)
```bash
# Preprocess và lưu cache HDF5
python preprocess_and_cache.py --format h5

# Hoặc lưu cache PyTorch tensors (nhanh hơn loading)
python preprocess_and_cache.py --format pt
```

#### Bước 2: Training nhanh với cache
```bash
# Training với cached data
python train_with_cache.py \
    --cache_dir preprocessed_cache \
    --batch_size 8 \
    --epochs 100 \
    --use_amp
```

### 3. Test tốc độ cache:
```bash
python -c "from src.cached_data_loader import test_cached_loader_speed; test_cached_loader_speed()"
```

## 📁 Cấu trúc cache

```
preprocessed_cache/
├── brain_001.h5              # Preprocessed volumes
├── brain_002.h5
├── ...
└── metadata/
    └── preprocessing_info.pkl # Metadata và thống kê
```

## ⚙️ Tham số training được tối ưu

### Google Colab (Tesla T4):
```bash
python run_fast_training.py \
    --batch_size 4 \
    --epochs 100 \
    --workers 2 \
    --use_amp \
    --save_freq 5
```

### GPU mạnh hơn:
```bash
python run_fast_training.py \
    --batch_size 8 \
    --epochs 100 \
    --workers 4 \
    --use_amp \
    --save_freq 5
```

## 🔧 Options chi tiết

### Preprocessing options:
- `--cache_format`: `h5` (HDF5, ít dung lượng) hoặc `pt` (PyTorch, nhanh hơn)
- `--skip_preprocessing`: Bỏ qua preprocessing nếu cache đã tồn tại

### Training options:
- `--batch_size`: Batch size (có thể lớn hơn với cache)
- `--epochs`: Số epochs 
- `--lr`: Learning rate (default: 0.0002)
- `--workers`: Số DataLoader workers
- `--use_amp`: Sử dụng mixed precision (tiết kiệm memory)
- `--aug_prob`: Xác suất augmentation (default: 0.8)

### Save options:
- `--save_freq`: Tần suất lưu checkpoint (epochs)
- `--val_freq`: Tần suất validation (epochs)

## 📈 Monitoring training

Training sẽ hiển thị:
```
Epoch 1/100: 100%|██████| 347/347 [00:04<00:00, 89.2it/s]
G_loss: 0.8234, D_loss: 0.1567, Total: 0.9801

📊 Epoch 1/100 Summary:
   Train Loss: 0.9801
   Val Loss: 0.8543
   Epoch Time: 4.2s
   Est. Remaining: 6.8 minutes
```

## 🎯 Kết quả

Sau training, bạn sẽ có:

```
fast_training_output/
├── checkpoints/
│   ├── best_model.pth        # Model tốt nhất
│   ├── epoch_5.pth          # Checkpoints định kỳ
│   └── epoch_10.pth
├── samples/                  # Sample outputs
└── logs/                    # Training logs
```

## 🔍 Kiểm tra môi trường

```bash
# Kiểm tra môi trường trước khi chạy
python run_fast_training.py --check_env
```

Output:
```
🔍 CHECKING ENVIRONMENT
✅ GPU: NVIDIA GeForce RTX 3080 (10.0GB)
✅ RAM: 32.0GB (available: 28.5GB)
✅ Disk: 500.2GB free
✅ numpy
✅ torch
✅ h5py
✅ Environment check passed!
```

## 💡 Tips tối ưu

### 1. Chọn cache format:
- **HDF5 (`h5`)**: Ít dung lượng, load hơi chậm hơn
- **PyTorch (`pt`)**: Nhiều dung lượng hơn, load nhanh nhất

### 2. Batch size:
- **Tesla T4 (15GB)**: batch_size=4-6
- **RTX 3080 (10GB)**: batch_size=6-8  
- **RTX 4090 (24GB)**: batch_size=12-16

### 3. Workers:
- **Local machine**: 4-8 workers
- **Google Colab**: 2 workers (optimal)

### 4. Mixed precision:
- Luôn dùng `--use_amp` để tiết kiệm memory và tăng tốc

## ⚠️ Lưu ý

1. **Cache size**: ~2-5GB cho 42 bệnh nhân
2. **Preprocessing**: Mất ~30-60 phút một lần duy nhất
3. **Training**: ~4-8s/epoch với cache vs ~30 phút/epoch không cache
4. **Augmentation**: Vẫn random mỗi epoch cho diversity

## 🆘 Troubleshooting

### Lỗi memory:
```bash
# Giảm batch size
--batch_size 2

# Giảm workers  
--workers 1

# Dùng mixed precision
--use_amp
```

### Cache bị hỏng:
```bash
# Xóa cache và tạo lại
rm -rf preprocessed_cache
python preprocess_and_cache.py
```

### Training chậm:
```bash
# Kiểm tra cache format
--cache_format pt  # Nhanh hơn h5

# Tăng workers nếu có đủ CPU
--workers 4
```

## 📚 Technical Details

### Cache preprocessing steps:
1. N4 bias correction (MRI)
2. Brain+skull mask creation  
3. MRI-guided CT artifact removal
4. Outlier clipping
5. Intensity normalization
6. Brain ROI cropping
7. Resize to 256×256
8. Range clamping [0,1]

### Fast augmentation (training time):
1. Convert to [-1,1] range (for model input)
2. Random rotation (±15°) with background=-1
3. Random horizontal flip
4. Random vertical flip  
5. Random intensity scaling (±10%) in [-1,1] range

### Memory optimization:
- HDF5 compression (gzip level 9)
- Mixed precision training
- Gradient checkpointing
- Smart memory cleanup

---

🚀 **Kết quả**: Training 100 epochs từ ~50 giờ xuống ~40 phút! 