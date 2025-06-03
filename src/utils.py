import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
from typing import Dict, Optional, Tuple
import cv2

def save_checkpoint(model, optimizer_G, optimizer_D, epoch: int, loss: float, filename: str):
    """
    Lưu checkpoint của model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint đã được lưu tại: {filename}")


def load_checkpoint(model, optimizer_G, optimizer_D, filename: str, device: str = 'cuda') -> int:
    """
    Load checkpoint và khôi phục trạng thái training
    
    Returns:
        epoch: epoch cuối cùng đã train
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint từ {filename}")
        
        checkpoint = torch.load(filename, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        print(f"Checkpoint loaded successfully (epoch {epoch}, loss: {loss:.4f})")
        return epoch
    else:
        print(f"Không tìm thấy checkpoint tại {filename}")
        return 0


def create_sample_images(model, mri_tensor: torch.Tensor, ct_tensor: torch.Tensor, 
                        save_path: str, device: str = 'cuda'):
    """
    Tạo và lưu ảnh mẫu để kiểm tra kết quả
    """
    model.eval()
    
    with torch.no_grad():
        mri = mri_tensor.to(device)
        ct = ct_tensor.to(device)
        
        # Generate fake images
        fake_ct = model.G_MRI2CT(mri)
        fake_mri = model.G_CT2MRI(ct)
        
        # Cycle consistency
        rec_mri = model.G_CT2MRI(fake_ct)
        rec_ct = model.G_MRI2CT(fake_mri)
        
        # Chuyển về numpy
        mri_np = mri[0, 0].cpu().numpy()
        ct_np = ct[0, 0].cpu().numpy()
        fake_ct_np = fake_ct[0, 0].cpu().numpy()
        fake_mri_np = fake_mri[0, 0].cpu().numpy()
        rec_mri_np = rec_mri[0, 0].cpu().numpy()
        rec_ct_np = rec_ct[0, 0].cpu().numpy()
        
        # Tạo figure với subplot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Hàng 1: MRI -> CT
        axes[0, 0].imshow(mri_np, cmap='gray')
        axes[0, 0].set_title('Real MRI')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fake_ct_np, cmap='gray')
        axes[0, 1].set_title('Fake CT (từ MRI)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(rec_mri_np, cmap='gray')
        axes[0, 2].set_title('Reconstructed MRI')
        axes[0, 2].axis('off')
        
        # Hàng 2: CT -> MRI
        axes[1, 0].imshow(ct_np, cmap='gray')
        axes[1, 0].set_title('Real CT')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(fake_mri_np, cmap='gray')
        axes[1, 1].set_title('Fake MRI (từ CT)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(rec_ct_np, cmap='gray')
        axes[1, 2].set_title('Reconstructed CT')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample images đã được lưu tại: {save_path}")


def save_nifti_image(image_array: np.ndarray, save_path: str, reference_path: Optional[str] = None):
    """
    Lưu ảnh numpy array thành file NIfTI
    
    Args:
        image_array: numpy array của ảnh
        save_path: đường dẫn lưu file
        reference_path: đường dẫn file tham chiếu để lấy header và affine
    """
    if reference_path and os.path.exists(reference_path):
        # Sử dụng header từ file tham chiếu
        reference_img = nib.load(reference_path)
        new_img = nib.Nifti1Image(image_array, reference_img.affine, reference_img.header)
    else:
        # Tạo header mặc định
        new_img = nib.Nifti1Image(image_array, np.eye(4))
    
    nib.save(new_img, save_path)
    print(f"Ảnh NIfTI đã được lưu tại: {save_path}")


def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Load và tiền xử lý ảnh NIfTI sử dụng Min-Max normalization
    """
    # Load bằng SimpleITK
    sitk_image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    
    # Lấy slice giữa
    middle_slice = image_array.shape[0] // 2
    slice_2d = image_array[middle_slice]
    
    # Resize nếu cần
    if slice_2d.shape != target_size:
        slice_2d = cv2.resize(slice_2d, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Min-Max normalization về [0,1] rồi chuyển về [-1, 1]
    slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
    slice_2d = slice_2d * 2.0 - 1.0
    
    # Chuyển về tensor
    tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return tensor


def visualize_training_progress(train_losses: list, val_losses: list, save_path: str):
    """
    Vẽ biểu đồ training progress
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training và Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot đã được lưu tại: {save_path}")


def calculate_model_size(model):
    """
    Tính toán kích thước model
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    print(f"Model size: {size_all_mb:.2f} MB")
    return size_all_mb


def compare_images(original: np.ndarray, generated: np.ndarray, save_path: str, 
                  title1: str = "Original", title2: str = "Generated"):
    """
    So sánh hai ảnh cạnh nhau
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ảnh gốc
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(title1)
    axes[0].axis('off')
    
    # Ảnh được tạo
    axes[1].imshow(generated, cmap='gray')
    axes[1].set_title(title2)
    axes[1].axis('off')
    
    # Difference map
    diff = np.abs(original - generated)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference Map')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_gif_from_images(image_folder: str, gif_path: str, duration: int = 500):
    """
    Tạo GIF từ các ảnh trong thư mục
    """
    try:
        from PIL import Image
        
        images = []
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        for filename in image_files:
            img = Image.open(os.path.join(image_folder, filename))
            images.append(img)
        
        if images:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF đã được tạo tại: {gif_path}")
        else:
            print("Không tìm thấy ảnh nào trong thư mục")
            
    except ImportError:
        print("Cần cài đặt Pillow để tạo GIF: pip install Pillow")


def print_model_summary(model):
    """
    In tóm tắt thông tin model
    """
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    # Generator parameters
    g_mri2ct_params = sum(p.numel() for p in model.G_MRI2CT.parameters())
    g_ct2mri_params = sum(p.numel() for p in model.G_CT2MRI.parameters())
    
    # Discriminator parameters
    d_ct_params = sum(p.numel() for p in model.D_CT.parameters())
    d_mri_params = sum(p.numel() for p in model.D_MRI.parameters())
    
    total_params = g_mri2ct_params + g_ct2mri_params + d_ct_params + d_mri_params
    
    print(f"Generator MRI->CT: {g_mri2ct_params:,} parameters")
    print(f"Generator CT->MRI: {g_ct2mri_params:,} parameters")
    print(f"Discriminator CT: {d_ct_params:,} parameters")
    print(f"Discriminator MRI: {d_mri_params:,} parameters")
    print("-" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {calculate_model_size(model):.2f} MB")
    print("="*50)


def setup_logging(log_dir: str):
    """
    Thiết lập logging cho training
    """
    import logging
    
    # Tạo thư mục log nếu chưa có
    os.makedirs(log_dir, exist_ok=True)
    
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__) 