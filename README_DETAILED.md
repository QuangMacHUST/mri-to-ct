# Dự án Chuyển đổi MRI thành CT mô phỏng sử dụng CycleGAN - Tài liệu Chi tiết

## 📋 Tổng quan

Dự án này triển khai mô hình **CycleGAN** để chuyển đổi ảnh MRI thành ảnh CT mô phỏng phục vụ cho **lập kế hoạch xạ trị** trong y học. Mô hình sử dụng kiến trúc CycleGAN với PatchGAN discriminator và tích hợp nhiều loss function tiên tiến bao gồm perceptual loss, adversarial loss, cycle consistency loss và identity loss.

### 🎯 Mục tiêu dự án
- **Vấn đề giải quyết**: Tạo ra ảnh CT mô phỏng từ ảnh MRI để hỗ trợ lập kế hoạch xạ trị khi không có sẵn ảnh CT
- **Ý nghĩa y học**: Giảm liều phóng xạ cho bệnh nhân, tiết kiệm chi phí và thời gian
- **Thách thức kỹ thuật**: Đảm bảo độ chính xác cao của ảnh CT mô phỏng để phục vụ mục đích y tế

## 🏗️ Cấu trúc dự án

```
mri-to-ct/
├── data/
│   ├── MRI/                 # Dữ liệu MRI training (brain_001.nii.gz - brain_046.nii.gz)
│   ├── CT/                  # Dữ liệu CT training (brain_001.nii.gz - brain_046.nii.gz)
│   └── Test/
│       ├── MRI/            # Dữ liệu MRI test
│       └── CT/             # Dữ liệu CT test
├── src/
│   ├── data_loader.py      # Tải và tiền xử lý dữ liệu
│   ├── models.py           # Kiến trúc CycleGAN và các mô hình
│   ├── train.py            # Script training
│   ├── test.py             # Script testing và đánh giá
│   ├── metrics.py          # Các metrics đánh giá (MAE, MSE, SSIM, PSNR)
│   └── utils.py            # Các hàm tiện ích
├── requirements.txt        # Các thư viện cần thiết
└── README.md
```

## 🔬 Chi tiết Tiền xử lý Dữ liệu

### 1. N4 Bias Field Correction

**Mục đích**: Hiệu chỉnh độ lệch từ trường trong ảnh MRI

**Lý do sử dụng**:
- Ảnh MRI thường bị nhiễu do sự không đồng nhất của từ trường
- Bias field gây ra độ sáng không đều trên toàn bộ ảnh
- Ảnh hưởng đến chất lượng training và kết quả cuối cùng

**Tham số được sử dụng**:
```python
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.SetMaximumNumberOfIterations([50] * 4)
```
- **50 iterations × 4 levels**: Đảm bảo hiệu chỉnh chính xác qua nhiều level resolution
- **4 levels**: Multi-resolution approach giúp xử lý bias field ở nhiều tần số khác nhau

**Tác động**: Cải thiện contrast và đồng nhất intensity, tạo điều kiện tốt cho các bước xử lý tiếp theo

### 2. Binary Masking với Otsu Thresholding

**Mục đích**: Tạo mask để phân tách vùng não khỏi background

**Lý do sử dụng**:
- Loại bỏ noise và artifacts ở vùng background
- Tập trung training vào vùng quan trọng (mô não)
- Giảm thiểu ảnh hưởng của vùng không chứa thông tin y học

**Quy trình xử lý**:
```python
# Chuẩn hóa về [0, 255]
normalized = ((image_array - image_array.min()) / 
             (image_array.max() - image_array.min()) * 255).astype(np.uint8)

# Otsu thresholding
threshold = filters.threshold_otsu(normalized)
binary_mask = normalized > threshold

# Post-processing
binary_mask = morphology.remove_small_objects(binary_mask, min_size=1000)
binary_mask = ndimage.binary_fill_holes(binary_mask)
```

**Tham số chi tiết**:
- **min_size=1000**: Loại bỏ các object nhỏ hơn 1000 pixels, giữ lại chỉ các vùng mô não chính
- **binary_fill_holes**: Lấp đầy các lỗ hổng nhỏ trong mask, tạo vùng liên tục

**Hiệu quả**: Cải thiện SNR (Signal-to-Noise Ratio) và giảm computational cost

### 3. Intensity Normalization

**Mục đích**: Chuẩn hóa giá trị pixel để ổn định training

**Phương pháp Min-Max normalization**:
```python
masked_values = image_array[mask > 0]
min_val = np.min(masked_values)
max_val = np.max(masked_values)
if max_val > min_val:
    # Min-Max normalization về [0, 1]
    image_array = (image_array - min_val) / (max_val - min_val)

# Áp dụng mask
image_array = image_array * mask

# Chuyển về [-1, 1] để phù hợp với Tanh activation
image_array = image_array * 2.0 - 1.0
```

**Lý do chọn Min-Max**:
- **Full dynamic range utilization**: Sử dụng toàn bộ range [0,1] sau khi đã loại bỏ outliers bằng mask
- **Intuitive interpretation**: 0 = pixel tối nhất trong brain, 1 = pixel sáng nhất trong brain
- **Optimal for generation**: Phù hợp với generative models khi đã có binary mask loại bỏ background
- **Cross-subject consistency**: Cùng tissue type sẽ có similar relative position trong [0,1] range

**Chuyển đổi [-1, 1]**:
- Phù hợp với Tanh activation của Generator output
- Standard practice trong CycleGAN và các generative models
- Đảm bảo symmetric range around zero

### 4. Data Augmentation

**Mục đích**: Tăng cường dữ liệu để tránh overfitting

**Các kỹ thuật được sử dụng**:

#### 4.1 Geometric Transformations
```python
# Random rotation (-10° đến +10°)
if random.random() > 0.5:
    angle = random.uniform(-10, 10)
    mri_slice = ndimage.rotate(mri_slice, angle, reshape=False, mode='nearest')
```
- **Góc xoay [-10°, +10°]**: Mô phỏng sự thay đổi tư thế đầu tự nhiên
- **reshape=False**: Giữ nguyên kích thước ảnh
- **mode='nearest'**: Tránh interpolation artifacts

#### 4.2 Flip Operations
```python
# Horizontal và Vertical flip với xác suất 50%
if random.random() > 0.5:
    mri_slice = np.fliplr(mri_slice)  # Left-Right flip
if random.random() > 0.5:
    mri_slice = np.flipud(mri_slice)  # Up-Down flip
```
- **Xác suất 50%**: Cân bằng giữa dữ liệu gốc và augmented
- **Anatomical consideration**: Phản ánh tính đối xứng tự nhiên của não

#### 4.3 Intensity Scaling
```python
# Random intensity scaling (±10%)
if random.random() > 0.5:
    scale_factor = random.uniform(0.9, 1.1)
    mri_slice = mri_slice * scale_factor
```
- **Scaling range [0.9, 1.1]**: Mô phỏng sự biến đổi contrast tự nhiên
- **Chỉ áp dụng cho MRI**: CT có range cố định nên không nên thay đổi

**Tác động tổng thể**: Tăng gấp 8 lần số lượng dữ liệu hiệu quả, cải thiện khả năng generalization

## 🧠 Kiến trúc Mô hình Chi tiết

### Generator Architecture

**Thiết kế Encoder-Decoder với Residual Blocks**:

```python
# Encoder (downsampling)
- ReflectionPad2d(3) + Conv2d(1→64, kernel=7) + InstanceNorm2d + ReLU
- Conv2d(64→128, kernel=3, stride=2) + InstanceNorm2d + ReLU  # 1/2 resolution
- Conv2d(128→256, kernel=3, stride=2) + InstanceNorm2d + ReLU # 1/4 resolution

# Residual Blocks (9 blocks)
- 9 × ResidualBlock(256 features)

# Decoder (upsampling)  
- ConvTranspose2d(256→128, kernel=3, stride=2) + InstanceNorm2d + ReLU # 1/2 resolution
- ConvTranspose2d(128→64, kernel=3, stride=2) + InstanceNorm2d + ReLU  # full resolution
- ReflectionPad2d(3) + Conv2d(64→1, kernel=7) + Tanh
```

**Lý do thiết kế**:
- **9 Residual Blocks**: Đủ sâu để học complex mappings, tránh vanishing gradient
- **ReflectionPad**: Giảm artifacts ở biên ảnh so với zero padding
- **InstanceNorm**: Phù hợp với style transfer, tốt hơn BatchNorm cho medical images
- **Tanh activation**: Output trong [-1, 1], phù hợp với normalized input

### PatchGAN Discriminator

**Thiết kế 70×70 PatchGAN**:
```python
# Layer 1: Conv2d(1→64, kernel=4, stride=2) + LeakyReLU(0.2)     # No normalization
# Layer 2: Conv2d(64→128, kernel=4, stride=2) + InstanceNorm + LeakyReLU(0.2)
# Layer 3: Conv2d(128→256, kernel=4, stride=2) + InstanceNorm + LeakyReLU(0.2)  
# Layer 4: Conv2d(256→512, kernel=4, stride=1) + InstanceNorm + LeakyReLU(0.2)
# Output:  Conv2d(512→1, kernel=4, stride=1)                     # No activation
```

**Ưu điểm PatchGAN**:
- **Local discrimination**: Đánh giá chất lượng ở mức local patches (70×70)
- **Fewer parameters**: Hiệu quả hơn full-image discriminator
- **Better texture quality**: Tập trung vào chi tiết texture thay vì global structure

## 📊 Chi tiết Loss Functions và Trọng số

### 1. Adversarial Loss (λ = 1.0)

**Công thức**:
```python
loss_gan_ct = F.mse_loss(D_CT(fake_ct), torch.ones_like(D_CT(fake_ct)))
loss_gan_mri = F.mse_loss(D_MRI(fake_mri), torch.ones_like(D_MRI(fake_mri)))
loss_gan = (loss_gan_ct + loss_gan_mri) * 0.5
```

**Lý do chọn MSE thay vì BCE**:
- **Least-squares GAN**: Ổn định training hơn, ít bị mode collapse
- **Smoother gradients**: Gradient penalty tự nhiên gần decision boundary
- **Better convergence**: Phù hợp với medical image domain

**Trọng số λ_adversarial = 1.0**:
- **Baseline weight**: Làm reference cho các loss khác
- **Balanced contribution**: Không quá dominant so với cycle consistency

### 2. Cycle Consistency Loss (λ = 10.0)

**Công thức**:
```python
loss_cycle_mri = F.l1_loss(G_CT2MRI(G_MRI2CT(mri)), mri)
loss_cycle_ct = F.l1_loss(G_MRI2CT(G_CT2MRI(ct)), ct)
loss_cycle = (loss_cycle_mri + loss_cycle_ct) * 0.5
```

**Lý do trọng số cao (λ = 10.0)**:
- **Core constraint**: Đảm bảo tính consistency cơ bản của cycle
- **Prevent mode collapse**: Ngăn generator tạo ra mapping tùy ý
- **Medical accuracy**: Đặc biệt quan trọng trong ứng dụng y tế
- **Empirical validation**: Giá trị 10.0 được verify qua nhiều nghiên cứu CycleGAN

**L1 vs L2 loss**:
- **L1 Loss**: Ít bị blur hơn, preserve sharp edges tốt hơn
- **Robust to outliers**: Quan trọng với medical images có noise

### 3. Identity Loss (λ = 5.0)

**Công thức**:
```python
loss_identity_ct = F.l1_loss(G_MRI2CT(ct), ct)
loss_identity_mri = F.l1_loss(G_CT2MRI(mri), mri)
loss_identity = (loss_identity_ct + loss_identity_mri) * 0.5
```

**Mục đích**:
- **Color preservation**: Giữ nguyên tông màu khi input đã đúng domain
- **Reduce overshoot**: Tránh generator thay đổi quá mức không cần thiết

**Trọng số λ_identity = 5.0**:
- **Moderate constraint**: Không quá restrictive như cycle consistency
- **Medical consideration**: Đảm bảo intensity values được preserve phù hợp
- **Half of cycle weight**: Cân bằng giữa consistency và flexibility

### 4. Perceptual Loss (λ = 1.0)

**Kiến trúc VGG19-based**:
```python
feature_layers = [3, 8, 15, 22]  # ReLU1_2, ReLU2_2, ReLU3_4, ReLU4_4
loss_perceptual = Σ MSE(VGG_i(fake_ct), VGG_i(real_ct))
```

**Lý do chọn VGG19**:
- **Pre-trained features**: Đã học được generic visual patterns
- **Multi-scale representation**: Từ low-level đến high-level features
- **Medical imaging validation**: Hiệu quả đã được chứng minh với medical images

**Feature layers selection**:
- **Layer 3 (ReLU1_2)**: Low-level textures, edges
- **Layer 8 (ReLU2_2)**: Medium-level patterns  
- **Layer 15 (ReLU3_4)**: High-level features
- **Layer 22 (ReLU4_4)**: Semantic representations

**Trọng số λ_perceptual = 1.0**:
- **Complementary role**: Bổ sung cho pixel-wise losses
- **Equal importance**: Với adversarial loss để cân bằng quality
- **Medical relevance**: Đảm bảo similarity ở multiple perception levels

### 5. Tổng hợp Loss Function

**Total Generator Loss**:
```python
L_total = λ_adversarial × L_GAN + 
          λ_cycle × L_cycle + 
          λ_identity × L_identity + 
          λ_perceptual × L_perceptual

L_total = 1.0 × L_GAN + 10.0 × L_cycle + 5.0 × L_identity + 1.0 × L_perceptual
```

**Hierarchy của trọng số**:
1. **L_cycle (10.0)**: Highest priority - đảm bảo consistency cơ bản
2. **L_identity (5.0)**: Medium priority - preserve original characteristics  
3. **L_GAN (1.0)**: Standard priority - realistic generation
4. **L_perceptual (1.0)**: Standard priority - perceptual quality

**Tác động của từng thành phần**:
- **Cycle consistency**: Đảm bảo mapping có thể đảo ngược
- **Identity loss**: Giữ nguyên khi không cần transform
- **Adversarial loss**: Tạo ra ảnh realistic
- **Perceptual loss**: Cải thiện chất lượng visual và texture

## 📈 Metrics Đánh giá

### 1. Mean Absolute Error (MAE)
- **Công thức**: `MAE = (1/N) Σ |y_pred - y_true|`
- **Ý nghĩa**: Sai số trung bình tuyệt đối, đơn vị giống pixel intensity
- **Tầm quan trọng**: Đo lường độ chính xác pixel-level

### 2. Mean Squared Error (MSE)  
- **Công thức**: `MSE = (1/N) Σ (y_pred - y_true)²`
- **Ý nghĩa**: Nhấn mạnh các sai số lớn, penalty cho outliers
- **Sử dụng**: Đánh giá overall reconstruction quality

### 3. Peak Signal-to-Noise Ratio (PSNR)
- **Công thức**: `PSNR = 20 × log₁₀(MAX_I / √MSE)`
- **Đơn vị**: Decibel (dB)
- **Giá trị tốt**: >25 dB cho medical images
- **Ý nghĩa**: Tỷ lệ signal/noise, cao hơn = chất lượng tốt hơn

### 4. Structural Similarity Index (SSIM)
- **Range**: [0, 1], càng gần 1 càng tốt
- **Thành phần**: Luminance × Contrast × Structure
- **Ưu điểm**: Tương quan tốt với perception của con người
- **Medical relevance**: Quan trọng cho texture và structural details

### 5. Normalized Cross Correlation (NCC)
- **Range**: [-1, 1], lý tưởng là 1
- **Ý nghĩa**: Đo correlation giữa hai ảnh
- **Robust**: Không bị ảnh hưởng bởi linear intensity changes

## 🚀 Hướng dẫn Sử dụng

### Cài đặt Dependencies

```bash
git clone <repository-url>
cd mri-to-ct
pip install -r requirements.txt
```

### Training

```bash
cd src
python train.py
```

**Tham số Training có thể điều chỉnh**:
```python
config = {
    'batch_size': 4,          # Tùy thuộc GPU memory
    'num_epochs': 200,        # Đủ để converge
    'lr_G': 0.0002,          # Learning rate cho Generator
    'lr_D': 0.0002,          # Learning rate cho Discriminator  
    'decay_epoch': 100,       # Bắt đầu decay LR
    'decay_epochs': 100       # Số epoch để decay về 0
}
```

### Testing

#### Test một ảnh MRI đơn lẻ:
```bash
python test.py --model_path checkpoints/best_model.pth \
               --test_mode single \
               --mri_path data/MRI/brain_001.nii.gz \
               --output_dir results/
```

#### Test với ground truth để tính metrics:
```bash
python test.py --model_path checkpoints/best_model.pth \
               --test_mode with_gt \
               --mri_path data/MRI/brain_001.nii.gz \
               --ct_path data/CT/brain_001.nii.gz \
               --output_dir results/
```

#### Test trên toàn bộ dataset:
```bash
python test.py --model_path checkpoints/best_model.pth \
               --test_mode dataset \
               --mri_dir data/Test/MRI \
               --ct_dir data/Test/CT \
               --output_dir results/
```

## 📁 Dữ liệu Input/Output

### Dữ liệu đầu vào
- **Format**: NIfTI (.nii.gz)
- **Naming convention**: brain_001.nii.gz đến brain_046.nii.gz
- **Pairing**: Mỗi file MRI có file CT tương ứng cùng tên
- **Preprocessing**: Dữ liệu đã được cắt vùng (cropped) thủ công
- **Resolution**: Tự động resize về 256×256 cho training

### Kết quả đầu ra
- **Synthetic CT**: File .nii.gz với suffix "_synthetic_ct"
- **Comparison images**: File .png hiển thị MRI input, CT mô phỏng, và difference map
- **Metrics report**: Các chỉ số MAE, MSE, SSIM, PSNR, NCC
- **Tensorboard logs**: Theo dõi loss và metrics qua epochs

## ⚙️ Yêu cầu Hệ thống

### Hardware Requirements
- **GPU**: CUDA-enabled GPU với ít nhất 8GB VRAM (khuyến nghị 16GB+)
- **RAM**: Tối thiểu 16GB, khuyến nghị 32GB
- **Storage**: ~20GB cho dữ liệu, checkpoints và logs
- **CPU**: Multi-core processor để data loading

### Software Requirements
- **Python**: 3.7+
- **PyTorch**: 1.9.0+
- **CUDA**: 11.0+ (nếu sử dụng GPU)
- **Operating System**: Linux (khuyến nghị), Windows, macOS

## 📊 Monitoring và Debugging

### Tensorboard Monitoring
```bash
tensorboard --logdir logs/
```

**Metrics được track**:
- Training/Validation losses (G_total, G_gan, G_cycle, G_identity, G_perceptual)
- Discriminator losses (D_CT, D_MRI)
- Image quality metrics (MAE, MSE, SSIM, PSNR)
- Learning rates

### Sample Images
- **Frequency**: Mỗi 5 epochs
- **Location**: `samples/epoch_X/`
- **Content**: Real MRI, Fake CT, Real CT, Reconstructed images

### Checkpoint Strategy
- **Regular saves**: Mỗi 10 epochs
- **Best model**: Dựa trên validation SSIM
- **Resume capability**: Có thể tiếp tục training từ checkpoint

## 🔬 Kết quả Mong đợi

### Performance Benchmarks
- **SSIM**: >0.85 cho high-quality synthesis
- **PSNR**: >25 dB
- **MAE**: <0.1 (với normalized intensity)
- **Training time**: 24-48 giờ trên GPU RTX 3080

### Qualitative Assessment
- **Bone structures**: Rõ nét và chính xác trong CT mô phỏng
- **Soft tissue contrast**: Tương đồng với CT thật
- **Artifacts**: Tối thiểu noise và distortion
- **Anatomical consistency**: Giữ nguyên cấu trúc từ MRI

## ⚠️ Lưu ý Quan trọng

### Medical Applications
1. **Validation requirement**: Kết quả cần được xác thực bởi chuyên gia y tế
2. **Clinical responsibility**: Không sử dụng trực tiếp mà không có supervision
3. **Regulatory compliance**: Tuân thủ các quy định y tế địa phương
4. **Quality assurance**: Kiểm tra kỹ lưỡng trước khi sử dụng lâm sàng

### Technical Considerations
1. **Data quality**: Chất lượng input ảnh hưởng trực tiếp đến output
2. **Domain specificity**: Model được train cho brain images
3. **Generalization**: Cần validation với different scanners/protocols
4. **Computational cost**: Training resource-intensive

### Hyperparameter Tuning
- **Learning rates**: Có thể cần điều chỉnh tùy dataset
- **Loss weights**: Fine-tune dựa trên validation metrics
- **Architecture**: Có thể thay đổi số residual blocks
- **Training schedule**: Điều chỉnh decay timing

## 📚 Tài liệu Tham khảo

### Core Papers
- **CycleGAN**: Zhu, J.Y., et al. "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks." ICCV 2017
- **Perceptual Loss**: Johnson, J., et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution." ECCV 2016
- **PatchGAN**: Isola, P., et al. "Image-to-Image Translation with Conditional Adversarial Networks." CVPR 2017

### Medical Imaging References
- **MRI-CT Synthesis**: Lei, Y., et al. "MRI-only based synthetic CT generation using dense cycle consistent generative adversarial networks." Medical Physics 2019
- **N4 Bias Correction**: Tustison, N.J., et al. "N4ITK: improved N3 bias correction." IEEE TMI 2010

### Implementation Guides
- **README Best Practices**: [How to Write a Good README](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
- **Documentation Standards**: [Make a README](https://www.makeareadme.com/)

## 🤝 Contributing

### Development Guidelines
1. **Code style**: Follow PEP 8 conventions
2. **Documentation**: Comment all functions và classes
3. **Testing**: Add unit tests for new features
4. **Version control**: Use descriptive commit messages

### Bug Reports
- Include system specifications
- Provide error logs và stack traces
- Describe reproduction steps
- Attach sample data if possible

## 📄 License

[Thêm thông tin license phù hợp]

## 👥 Liên hệ

**Nhóm phát triển**: MRI-to-CT Research Team
**Email**: [Thêm email liên hệ]
**Institution**: [Thêm thông tin tổ chức]

---

**Lưu ý**: Đây là dự án nghiên cứu. Kết quả cần được validation và approval từ chuyên gia y tế trước khi sử dụng trong thực tế lâm sàng. 