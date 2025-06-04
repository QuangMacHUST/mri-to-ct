import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from skimage import filters, morphology
from scipy import ndimage
import random
from typing import Tuple, Optional, List
import cv2

class MRIToCTDataset(Dataset):
    """
    Dataset class để tải và tiền xử lý dữ liệu MRI và CT
    """
    
    def __init__(self, 
                 mri_dir: str, 
                 ct_dir: str,
                 transform=None,
                 is_training: bool = True,
                 slice_range: Optional[Tuple[int, int]] = None):
        """
        Args:
            mri_dir: đường dẫn thư mục chứa ảnh MRI
            ct_dir: đường dẫn thư mục chứa ảnh CT  
            transform: các phép biến đổi dữ liệu
            is_training: chế độ training hay test
            slice_range: phạm vi slice để lấy (start, end)
        """
        self.mri_dir = mri_dir
        self.ct_dir = ct_dir
        self.transform = transform
        self.is_training = is_training
        self.slice_range = slice_range
        
        # Lấy danh sách các file
        self.mri_files = sorted([f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')])
        self.ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')])
        
        # Đảm bảo số lượng file MRI và CT khớp nhau
        assert len(self.mri_files) == len(self.ct_files), "Số lượng file MRI và CT không khớp"
        
        print(f"Đã tải {len(self.mri_files)} cặp ảnh MRI-CT")
        
    def __len__(self):
        return len(self.mri_files)
    
    def _apply_n4_bias_correction(self, image: sitk.Image) -> sitk.Image:
        """
        Áp dụng N4 bias field correction
        """
        # Chuyển đổi sang float32 để tránh lỗi với 16-bit integer
        if image.GetPixelIDValue() != sitk.sitkFloat32:
            image = sitk.Cast(image, sitk.sitkFloat32)
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        return corrector.Execute(image)
    
    def _create_otsu_mask(self, image_array: np.ndarray) -> np.ndarray:
        """
        Tạo binary mask sử dụng Otsu thresholding
        """
        # Chuẩn hóa ảnh về [0, 255]
        normalized = ((image_array - image_array.min()) / 
                     (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        
        # Áp dụng Otsu threshold
        threshold = filters.threshold_otsu(normalized)
        binary_mask = normalized > threshold
        
        # Loại bỏ các vùng nhỏ và lấp đầy lỗ hổng
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=1000)
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        return binary_mask.astype(np.float32)
    
    def _normalize_intensity(self, image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa cường độ ảnh trong vùng mask sử dụng Min-Max normalization
        """
        masked_values = image_array[mask > 0]
        if len(masked_values) > 0:
            min_val = np.min(masked_values)
            max_val = np.max(masked_values)
            if max_val > min_val:
                # Min-Max normalization về [0, 1]
                image_array = (image_array - min_val) / (max_val - min_val)
            else:
                # Trường hợp tất cả pixel có cùng giá trị
                image_array = np.zeros_like(image_array)
        
        # Áp dụng mask để đảm bảo background = 0
        image_array = image_array * mask
        
        # Chuyển về [-1, 1] để phù hợp với Tanh activation của Generator
        image_array = image_array * 2.0 - 1.0
        
        return image_array
    
    def _augment_data(self, mri_slice: np.ndarray, ct_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Áp dụng data augmentation
        """
        if not self.is_training:
            return mri_slice, ct_slice
            
        # Random rotation (-10 đến 10 độ)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            mri_slice = ndimage.rotate(mri_slice, angle, reshape=False, mode='nearest')
            ct_slice = ndimage.rotate(ct_slice, angle, reshape=False, mode='nearest')
        
        # Random flip horizontal
        if random.random() > 0.5:
            mri_slice = np.fliplr(mri_slice).copy()
            ct_slice = np.fliplr(ct_slice).copy()
            
        # Random flip vertical
        if random.random() > 0.5:
            mri_slice = np.flipud(mri_slice).copy()
            ct_slice = np.flipud(ct_slice).copy()
        
        # Random intensity scaling cho MRI (±10%)
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            mri_slice = mri_slice * scale_factor
            
        return mri_slice, ct_slice
    
    def __getitem__(self, idx):
        """
        Lấy một cặp ảnh MRI-CT
        """
        # Đọc file MRI và CT
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        
        # Load ảnh bằng SimpleITK
        mri_sitk = sitk.ReadImage(mri_path)
        ct_sitk = sitk.ReadImage(ct_path)
        
        # Áp dụng N4 bias correction cho MRI
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        
        # Chuyển về numpy array
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        # Tạo binary mask cho cả MRI và CT
        mri_mask = self._create_otsu_mask(mri_array)
        ct_mask = self._create_otsu_mask(ct_array)
        
        # Chuẩn hóa intensity
        mri_array = self._normalize_intensity(mri_array, mri_mask)
        ct_array = self._normalize_intensity(ct_array, ct_mask)
        
        # Áp dụng mask
        mri_array = mri_array * mri_mask
        ct_array = ct_array * ct_mask
        
        # Lấy slice giữa hoặc random slice nếu đang training
        if self.slice_range:
            start_slice, end_slice = self.slice_range
            slice_idx = random.randint(start_slice, min(end_slice, mri_array.shape[0]-1)) if self.is_training else (start_slice + end_slice) // 2
        else:
            slice_idx = random.randint(0, mri_array.shape[0]-1) if self.is_training else mri_array.shape[0] // 2
        
        mri_slice = mri_array[slice_idx]
        ct_slice = ct_array[slice_idx]
        
        # Data augmentation
        mri_slice, ct_slice = self._augment_data(mri_slice, ct_slice)
        
        # Resize về 256x256 nếu cần
        if mri_slice.shape != (256, 256):
            mri_slice = cv2.resize(mri_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
            ct_slice = cv2.resize(ct_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # Đảm bảo arrays có positive strides
        mri_slice = mri_slice.copy()
        ct_slice = ct_slice.copy()
        
        # Chuyển về tensor và thêm channel dimension
        mri_tensor = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0)    # [1, H, W]
        
        return {
            'mri': mri_tensor,
            'ct': ct_tensor,
            'filename': self.mri_files[idx]
        }


def create_data_loaders(mri_dir: str, 
                       ct_dir: str, 
                       batch_size: int = 4,
                       train_split: float = 0.8,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo DataLoader cho training và validation
    """
    # Tạo dataset đầy đủ
    full_dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=True)
    
    # Chia train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader


def create_test_loader(mri_dir: str, 
                      ct_dir: str, 
                      batch_size: int = 1,
                      num_workers: int = 2) -> DataLoader:
    """
    Tạo DataLoader cho test
    """
    test_dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    return test_loader 