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
        Áp dụng N4 bias field correction để loại bỏ bias field trong MRI
        """
        # Cast về float32 để tránh lỗi với 16-bit signed integer
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
        # Chỉ normalize trong vùng mask
        masked_values = image_array[mask > 0]
        if len(masked_values) > 0:
            min_val = np.min(masked_values)
            max_val = np.max(masked_values)
            if max_val > min_val:
                # Min-Max normalization về [0, 1] chỉ trong vùng mask
                normalized = np.zeros_like(image_array)
                normalized[mask > 0] = (image_array[mask > 0] - min_val) / (max_val - min_val)
                image_array = normalized
            else:
                # Trường hợp tất cả pixel có cùng giá trị
                image_array = np.zeros_like(image_array)
        
        # KHÔNG convert về [-1,1] ở đây, sẽ làm ở cuối
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
            mri_slice = ndimage.rotate(mri_slice, angle, reshape=False, mode='constant', cval=0)
            ct_slice = ndimage.rotate(ct_slice, angle, reshape=False, mode='constant', cval=0)
        
        # Random flip horizontal
        if random.random() > 0.5:
            mri_slice = np.fliplr(mri_slice).copy()
            ct_slice = np.fliplr(ct_slice).copy()
            
        # Random flip vertical
        if random.random() > 0.5:
            mri_slice = np.flipud(mri_slice).copy()
            ct_slice = np.flipud(ct_slice).copy()
        
        # Random intensity scaling cho MRI (±10%) - chỉ trong vùng mask
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            mask = mri_slice > 0  # Detect brain region
            mri_slice[mask] = mri_slice[mask] * scale_factor
            mri_slice = np.clip(mri_slice, 0, 1)  # Clip về [0,1]
            
        return mri_slice, ct_slice
    
    def _remove_head_frame(self, image_array: np.ndarray, modality: str = 'CT') -> Tuple[np.ndarray, np.ndarray]:
        """
        Loại bỏ khung cố định đầu (stereotactic head frame) và các artifacts kim loại
        
        Args:
            image_array: ảnh 3D 
            modality: 'CT' hoặc 'MRI'
        
        Returns:
            cleaned_array: ảnh đã loại bỏ khung
            frame_mask: mask của khung (để debugging)
        """
        cleaned_array = image_array.copy()
        
        if modality == 'CT':
            # CT: Khung kim loại có intensity rất cao - chỉ target extreme values
            normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            
            # Chỉ target top 2% extreme values (metal có intensity rất cao)
            high_intensity_threshold = np.percentile(normalized, 98)  # Stricter threshold
            metal_mask = normalized > high_intensity_threshold
            
            # Remove small noise, keep only large connected components (khung thật)
            metal_mask = morphology.remove_small_objects(metal_mask, min_size=1000)  # Larger min_size
            
            # Smaller dilation để không remove quá nhiều
            kernel = morphology.ball(1)  # Smaller kernel
            metal_mask = morphology.binary_dilation(metal_mask, kernel)
            
            # Set metal regions to median background value instead of percentile
            background_value = np.median(image_array[normalized < 0.3])
            cleaned_array[metal_mask] = background_value
            
        else:  # MRI
            # MRI: Khung có thể tạo signal void hoặc artifacts - ít aggressive hơn
            normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            
            # Chỉ target extreme outliers
            high_threshold = np.percentile(normalized, 98)  # Top 2%
            low_threshold = np.percentile(normalized, 2)    # Bottom 2%
            
            artifact_mask = (normalized > high_threshold) | (normalized < low_threshold)
            
            # Remove small noise - chỉ giữ artifacts lớn
            artifact_mask = morphology.remove_small_objects(artifact_mask, min_size=500)
            
            # Minimal dilation
            kernel = morphology.ball(1) 
            artifact_mask = morphology.binary_dilation(artifact_mask, kernel)
            
            # Set to median background value
            background_value = np.median(image_array[normalized < 0.5])
            cleaned_array[artifact_mask] = background_value
            
            metal_mask = artifact_mask
        
        return cleaned_array, metal_mask.astype(np.float32)
    
    def _crop_brain_roi(self, image_array: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop về ROI chứa não, loại bỏ vùng ngoài không cần thiết
        """
        # Find bounding box của vùng não
        brain_coords = np.where(mask > 0)
        
        if len(brain_coords[0]) == 0:
            # Nếu không tìm thấy brain mask, return original
            return image_array, mask
        
        # Lấy bounding box với padding
        min_z, max_z = brain_coords[0].min(), brain_coords[0].max()
        min_y, max_y = brain_coords[1].min(), brain_coords[1].max()  
        min_x, max_x = brain_coords[2].min(), brain_coords[2].max()
        
        # Add padding để không crop quá sát
        padding = 10
        min_z = max(0, min_z - padding)
        max_z = min(image_array.shape[0], max_z + padding)
        min_y = max(0, min_y - padding)
        max_y = min(image_array.shape[1], max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(image_array.shape[2], max_x + padding)
        
        # Crop image và mask
        cropped_image = image_array[min_z:max_z, min_y:max_y, min_x:max_x]
        cropped_mask = mask[min_z:max_z, min_y:max_y, min_x:max_x]
        
        return cropped_image, cropped_mask
    
    def __getitem__(self, idx):
        """
        Lấy một cặp ảnh MRI-CT với pipeline tiền xử lý logic đúng: N4 → Remove Frame → Mask → Normalize → Augment
        """
        # Đọc file MRI và CT
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        
        # Load ảnh bằng SimpleITK
        mri_sitk = sitk.ReadImage(mri_path)
        ct_sitk = sitk.ReadImage(ct_path)
        
        # BƯỚC 1: Áp dụng N4 bias correction cho MRI (cải thiện chất lượng trước khi detect frame)
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        
        # Chuyển về numpy array
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        # BƯỚC 2: Loại bỏ head frame/stereotactic frame TRƯỚC khi tạo mask
        print(f"Đang loại bỏ head frame cho {self.mri_files[idx]}...")
        mri_array, mri_frame_mask = self._remove_head_frame(mri_array, 'MRI')
        ct_array, ct_frame_mask = self._remove_head_frame(ct_array, 'CT')
        
        # BƯỚC 3: Tạo binary mask (sau khi đã clean frame artifacts)
        mri_mask = self._create_otsu_mask(mri_array)
        ct_mask = self._create_otsu_mask(ct_array)
        
        # BƯỚC 4: Chuẩn hóa intensity với mask - normalize trong vùng sạch
        mri_array = self._normalize_intensity(mri_array, mri_mask)
        ct_array = self._normalize_intensity(ct_array, ct_mask)
        
        # BƯỚC 5: Crop về brain ROI (sau khi đã normalize)
        mri_array, mri_mask = self._crop_brain_roi(mri_array, mri_mask)
        ct_array, ct_mask = self._crop_brain_roi(ct_array, ct_mask)
        
        # BƯỚC 6: Lấy slice giữa hoặc random slice
        if self.slice_range:
            start_slice, end_slice = self.slice_range
            slice_idx = random.randint(start_slice, min(end_slice, mri_array.shape[0]-1)) if self.is_training else (start_slice + end_slice) // 2
        else:
            slice_idx = random.randint(0, mri_array.shape[0]-1) if self.is_training else mri_array.shape[0] // 2
        
        mri_slice = mri_array[slice_idx]
        ct_slice = ct_array[slice_idx]
        
        # BƯỚC 7: Resize về 256x256 (fix import error)
        if mri_slice.shape != (256, 256):
            from scipy.ndimage import zoom
            # Calculate zoom factors
            zoom_h = 256 / mri_slice.shape[0]
            zoom_w = 256 / mri_slice.shape[1]
            mri_slice = zoom(mri_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
            ct_slice = zoom(ct_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
        
        # Clamp để đảm bảo trong [0,1] sau resize
        mri_slice = np.clip(mri_slice, 0, 1)
        ct_slice = np.clip(ct_slice, 0, 1)
        
        # BƯỚC 8: Data augmentation trên [0,1] range (cuối cùng)
        mri_slice, ct_slice = self._augment_data(mri_slice, ct_slice)
        
        # BƯỚC 9: Convert về [-1,1] CUỐI CÙNG để phù hợp với Tanh activation
        mri_slice = mri_slice * 2.0 - 1.0
        ct_slice = ct_slice * 2.0 - 1.0
        
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