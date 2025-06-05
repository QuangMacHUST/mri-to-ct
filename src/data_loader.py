import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from skimage import filters, morphology, measure
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
    
    def _create_brain_with_skull_mask(self, mri_array: np.ndarray) -> np.ndarray:
        """
        Tạo comprehensive mask bao gồm brain tissue + skull để preserve bone structures
        """
        # Step 1: Normalize về [0, 1]
        normalized = (mri_array - mri_array.min()) / (mri_array.max() - mri_array.min())
        
        # Step 2: Multi-threshold approach để capture brain + skull
        otsu_thresh = filters.threshold_otsu(normalized)
        
        # Lower threshold để capture brain tissue (including gray matter)
        brain_thresh = otsu_thresh * 0.6  # Slightly lower để capture more
        
        # Higher threshold để capture bright structures (skull in some MRI sequences)
        skull_thresh = otsu_thresh * 1.2
        
        # Combine brain and potential skull regions
        brain_mask = normalized > brain_thresh
        bright_mask = normalized > skull_thresh
        
        # Step 3: Create comprehensive mask
        # Start với brain mask
        comprehensive_mask = brain_mask.copy()
        
        # Step 4: Morphological operations
        # Remove small noise objects
        comprehensive_mask = morphology.remove_small_objects(comprehensive_mask, min_size=1500)
        
        # Fill holes để có continuous region
        comprehensive_mask = ndimage.binary_fill_holes(comprehensive_mask)
        
        # Step 5: Get largest connected component + surrounding region
        labeled_mask = measure.label(comprehensive_mask)
        if labeled_mask.max() > 0:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0  # Ignore background
            largest_component = np.argmax(component_sizes)
            main_region = (labeled_mask == largest_component)
        else:
            main_region = comprehensive_mask
        
        # Step 6: Expand mask để include skull region
        # Dilation để capture skull structures around brain
        kernel_expand = morphology.ball(3)  # Slightly larger kernel
        expanded_mask = morphology.binary_dilation(main_region, kernel_expand)
        
        # Step 7: Refine với shape constraints
        # Remove regions quá xa brain center
        center_of_mass = ndimage.center_of_mass(main_region)
        
        # Create distance-based refinement
        coords = np.ogrid[0:expanded_mask.shape[0], 0:expanded_mask.shape[1], 0:expanded_mask.shape[2]]
        distances = np.sqrt(
            (coords[0] - center_of_mass[0])**2 +
            (coords[1] - center_of_mass[1])**2 +
            (coords[2] - center_of_mass[2])**2
        )
        
        # Maximum reasonable distance để include skull
        max_brain_radius = np.max(distances[main_region]) * 1.3  # 30% buffer for skull
        distance_mask = distances <= max_brain_radius
        
        # Combine expanded mask với distance constraint
        final_mask = expanded_mask & distance_mask
        
        # Step 8: Final morphological cleanup
        # Gentle closing để smooth contours
        kernel_smooth = morphology.ball(2)
        final_mask = morphology.binary_closing(final_mask, kernel_smooth)
        
        # Fill any remaining holes
        final_mask = ndimage.binary_fill_holes(final_mask)
        
        # Ensure mask is not too large (safety check)
        total_volume = np.prod(mri_array.shape)
        mask_volume = np.sum(final_mask)
        
        if mask_volume > total_volume * 0.7:  # If mask > 70% of image, too large
            print("Warning: Mask too large, falling back to conservative approach")
            # Fall back to original brain mask
            conservative_mask = main_region
            kernel_conservative = morphology.ball(1)
            final_mask = morphology.binary_dilation(conservative_mask, kernel_conservative)
        
        return final_mask.astype(np.float32)
    
    def _apply_mri_mask_to_ct(self, ct_array: np.ndarray, mri_mask: np.ndarray) -> np.ndarray:
        """
        Improved MRI-guided CT artifact removal với better preservation
        """
        # Step 1: Apply MRI mask để loại bỏ couch/headframe
        masked_ct = ct_array.copy()
        
        # Tạo realistic background value (air-like)
        background_region = ct_array[mri_mask == 0]
        if len(background_region) > 0:
            # Air value trong CT thường là -1000 HU, dùng percentile thấp
            background_value = np.percentile(background_region, 10)  # More realistic air value
            # Ensure không quá extreme
            background_value = max(background_value, ct_array.min())
        else:
            background_value = ct_array.min()
        
        # Set vùng ngoài mask thành air-like value
        masked_ct[mri_mask == 0] = background_value
        
        # Step 2: Improved metal artifact detection trong brain region
        brain_region = masked_ct[mri_mask > 0]
        if len(brain_region) > 0:
            # Use more robust statistics
            q95 = np.percentile(brain_region, 95)
            q05 = np.percentile(brain_region, 5)
            q50 = np.percentile(brain_region, 50)  # Median
            
            # More conservative thresholds để preserve normal tissue
            metal_threshold = q95 + 2 * (q95 - q50)  # Detect extreme bright artifacts
            air_threshold = q05 - 2 * (q50 - q05)    # Detect extreme dark artifacts
            
            # Create masks với conservative approach
            metal_mask = (masked_ct > metal_threshold) & (mri_mask > 0)
            air_mask = (masked_ct < air_threshold) & (mri_mask > 0)
            
            # Stricter size requirements để avoid removing normal tissue
            metal_mask = morphology.remove_small_objects(metal_mask, min_size=1000)  # Larger size
            air_mask = morphology.remove_small_objects(air_mask, min_size=1000)
            
            # Replace artifacts với tissue-appropriate values
            if np.any(metal_mask):
                # Metal artifacts -> median của normal brain tissue
                normal_tissue_value = np.median(brain_region[(brain_region >= q05) & (brain_region <= q95)])
                masked_ct[metal_mask] = normal_tissue_value
                
            if np.any(air_mask):
                # Air artifacts -> CSF-like value (slightly above q05)
                csf_value = np.percentile(brain_region, 20)  # Typical CSF range
                masked_ct[air_mask] = csf_value
        
        return masked_ct
    
    def _gentle_outlier_clipping(self, image_array: np.ndarray, mask: np.ndarray, modality: str = 'CT') -> np.ndarray:
        """
        Minimal outlier clipping để preserve natural tissue characteristics hoàn toàn
        """
        clipped = image_array.copy()
        brain_region = image_array[mask > 0]
        
        if len(brain_region) == 0:
            return clipped
        
        if modality == 'CT':
            # CT: Chỉ clip extreme outliers, preserve HU values
            brain_values = brain_region.flatten()
            
            # Very conservative outlier detection using IQR method
            q25, q75 = np.percentile(brain_values, [25, 75])
            iqr = q75 - q25
            
            # 3 * IQR rule (more conservative than 1.5 * IQR)
            lower_bound = q25 - 3 * iqr
            upper_bound = q75 + 3 * iqr
            
            # Clip chỉ extreme outliers
            clipped[mask > 0] = np.clip(clipped[mask > 0], lower_bound, upper_bound)
            
        else:  # MRI
            # MRI: Tương tự với conservative approach
            brain_values = brain_region.flatten()
            
            # Use 2.5 * IQR cho MRI (slightly less conservative)
            q25, q75 = np.percentile(brain_values, [25, 75])
            iqr = q75 - q25
            
            lower_bound = q25 - 2.5 * iqr
            upper_bound = q75 + 2.5 * iqr
            
            # Ensure không clip quá nhiều
            lower_bound = max(lower_bound, np.percentile(brain_values, 0.5))
            upper_bound = min(upper_bound, np.percentile(brain_values, 99.5))
            
            clipped[mask > 0] = np.clip(clipped[mask > 0], lower_bound, upper_bound)
        
        return clipped
    
    def _normalize_intensity(self, image_array: np.ndarray, mask: np.ndarray, modality: str = 'CT') -> np.ndarray:
        """
        Tissue-aware normalization để preserve natural contrast relationships
        """
        # Chỉ normalize trong vùng mask
        masked_values = image_array[mask > 0]
        if len(masked_values) == 0:
            return np.zeros_like(image_array)
        
        normalized = np.zeros_like(image_array)
        
        if modality == 'CT':
            # CT: Preserve bone-tissue contrast for skull + brain
            # Extended HU ranges: CSF: 0-15, Gray: 35-45, White: 20-30, Bone: 700-3000 HU
            
            # Separate soft tissue and bone for better normalization
            soft_tissue_mask = (masked_values >= -100) & (masked_values <= 100)  # Soft tissue range
            bone_mask = masked_values > 150  # Bone/skull range
            
            if np.any(soft_tissue_mask) and np.any(bone_mask):
                # Mixed tissue approach - preserve both soft tissue and bone contrast
                
                # Soft tissue normalization bounds
                soft_tissue_values = masked_values[soft_tissue_mask]
                p5_soft, p95_soft = np.percentile(soft_tissue_values, [5, 95])
                
                # Bone normalization bounds  
                bone_values = masked_values[bone_mask]
                p5_bone, p95_bone = np.percentile(bone_values, [10, 90])  # More conservative for bone
                
                # Combined bounds để preserve full spectrum
                p1 = min(p5_soft, np.percentile(masked_values, 1))
                p99 = max(p95_bone, np.percentile(masked_values, 99))
                
                # Ensure adequate bone contrast
                if p99 - p95_soft < 50:  # Ensure bone stands out from soft tissue
                    p99 = p95_soft + 200  # Add bone contrast range
                    
            else:
                # Fallback to wider percentile range
                p1, p99 = np.percentile(masked_values, [0.5, 99.5])
            
            # Ensure minimum dynamic range for tissue differentiation
            if p99 - p1 < 100:  # Increase threshold for skull inclusion
                mean_val = np.mean(masked_values)
                std_val = np.std(masked_values)
                p1 = max(mean_val - 4 * std_val, np.min(masked_values))
                p99 = min(mean_val + 4 * std_val, np.max(masked_values))
            
            # Robust normalization preserving bone-tissue contrast
            if p99 > p1:
                normalized[mask > 0] = (image_array[mask > 0] - p1) / (p99 - p1)
                # Allow headroom for extreme bone values
                normalized = np.clip(normalized, -0.15, 1.15)  # More headroom for bone
                # Final clip to [0, 1]
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized[mask > 0] = 0.5  # Default middle value
                
        else:  # MRI
            # MRI: Preserve relative intensity relationships
            # Use robust statistics to avoid extreme outliers
            
            # Calculate robust statistics
            q25, q75 = np.percentile(masked_values, [25, 75])
            iqr = q75 - q25
            
            # Define outlier boundaries (1.5 * IQR rule)
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Use these as normalization bounds
            min_val = max(lower_bound, np.percentile(masked_values, 1))
            max_val = min(upper_bound, np.percentile(masked_values, 99))
            
            if max_val > min_val:
                normalized[mask > 0] = (image_array[mask > 0] - min_val) / (max_val - min_val)
                normalized = np.clip(normalized, 0, 1)
            else:
                normalized[mask > 0] = 0.5
        
        return normalized
    
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
        Pipeline tiền xử lý sử dụng MRI mask để loại bỏ couch/headframe từ CT
        """
        # Đọc file MRI và CT
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        
        # Load ảnh bằng SimpleITK
        mri_sitk = sitk.ReadImage(mri_path)
        ct_sitk = sitk.ReadImage(ct_path)
        
        # BƯỚC 1: N4 bias correction chỉ cho MRI
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        
        # Chuyển về numpy array
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        print(f"Xử lý {self.mri_files[idx]} với MRI-guided preprocessing...")
        
        # BƯỚC 2: Tạo comprehensive mask từ MRI (brain + skull, không có couch/headframe)
        mri_mask = self._create_brain_with_skull_mask(mri_array)
        
        # BƯỚC 3: Sử dụng MRI mask để clean up CT (loại bỏ couch/headframe)
        ct_array = self._apply_mri_mask_to_ct(ct_array, mri_mask)
        
        # BƯỚC 4: Gentle outlier clipping only (NO contrast enhancement)
        # Preserve natural tissue characteristics by only removing extreme outliers
        mri_array = self._gentle_outlier_clipping(mri_array, mri_mask, 'MRI')
        ct_array = self._gentle_outlier_clipping(ct_array, mri_mask, 'CT')
        
        # BƯỚC 5: Normalize intensity trong brain region
        mri_array = self._normalize_intensity(mri_array, mri_mask, 'MRI')
        ct_array = self._normalize_intensity(ct_array, mri_mask, 'CT')  # Sử dụng cùng mask
        
        # BƯỚC 6: Crop brain ROI (conservative)
        mri_array, mri_mask_cropped = self._crop_brain_roi(mri_array, mri_mask)
        ct_array, _ = self._crop_brain_roi(ct_array, mri_mask)  # Sử dụng original mask để crop cùng region
        
        # BƯỚC 7: Lấy slice
        if self.slice_range:
            start_slice, end_slice = self.slice_range
            slice_idx = random.randint(start_slice, min(end_slice, mri_array.shape[0]-1)) if self.is_training else (start_slice + end_slice) // 2
        else:
            slice_idx = random.randint(0, mri_array.shape[0]-1) if self.is_training else mri_array.shape[0] // 2
        
        mri_slice = mri_array[slice_idx]
        ct_slice = ct_array[slice_idx]
        
        # BƯỚC 8: Resize
        if mri_slice.shape != (256, 256):
            from scipy.ndimage import zoom
            zoom_h = 256 / mri_slice.shape[0]
            zoom_w = 256 / mri_slice.shape[1]
            mri_slice = zoom(mri_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
            ct_slice = zoom(ct_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
        
        # BƯỚC 9: Final clamp về [0,1]
        mri_slice = np.clip(mri_slice, 0, 1)
        ct_slice = np.clip(ct_slice, 0, 1)
        
        # BƯỚC 10: Data augmentation
        mri_slice, ct_slice = self._augment_data(mri_slice, ct_slice)
        
        # BƯỚC 11: Convert về [-1,1] cho model
        mri_slice = mri_slice * 2.0 - 1.0
        ct_slice = ct_slice * 2.0 - 1.0
        
        # Đảm bảo arrays có positive strides
        mri_slice = mri_slice.copy()
        ct_slice = ct_slice.copy()
        
        # Chuyển về tensor
        mri_tensor = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0)
        
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