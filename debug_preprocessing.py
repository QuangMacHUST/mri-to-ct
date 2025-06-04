#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script debug để kiểm tra chất lượng tiền xử lý ảnh
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

# Import trực tiếp từ file để tránh circular import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import Dataset
import cv2
import random
from typing import Optional, Tuple
from skimage import filters, morphology
from scipy import ndimage

# Copy class MRIToCTDataset để tránh import conflicts
class MRIToCTDataset(Dataset):
    def __init__(self, 
                 mri_dir: str, 
                 ct_dir: str,
                 transform=None,
                 is_training: bool = True,
                 slice_range: Optional[Tuple[int, int]] = None):
        self.mri_dir = mri_dir
        self.ct_dir = ct_dir
        self.transform = transform
        self.is_training = is_training
        self.slice_range = slice_range
        
        # Lấy danh sách file
        self.mri_files = sorted([f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')])
        self.ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')])
        
        # Đảm bảo số lượng file MRI và CT bằng nhau
        assert len(self.mri_files) == len(self.ct_files), f"Số file MRI ({len(self.mri_files)}) và CT ({len(self.ct_files)}) không bằng nhau"
        
        print(f"Đã tải {len(self.mri_files)} cặp ảnh MRI-CT")
    
    def __len__(self):
        return len(self.mri_files)
    
    def _apply_n4_bias_correction(self, image: sitk.Image) -> sitk.Image:
        # Cast về float32 để tránh lỗi với 16-bit signed integer
        image = sitk.Cast(image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        return corrector.Execute(image)
    
    def _create_otsu_mask(self, image_array: np.ndarray) -> np.ndarray:
        normalized = ((image_array - image_array.min()) / 
                     (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        threshold = filters.threshold_otsu(normalized)
        binary_mask = normalized > threshold
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=1000)
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        return binary_mask.astype(np.float32)
    
    def _normalize_intensity(self, image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked_values = image_array[mask > 0]
        if len(masked_values) > 0:
            min_val = np.min(masked_values)
            max_val = np.max(masked_values)
            if max_val > min_val:
                normalized = np.zeros_like(image_array)
                normalized[mask > 0] = (image_array[mask > 0] - min_val) / (max_val - min_val)
                image_array = normalized
            else:
                image_array = np.zeros_like(image_array)
        return image_array
    
    def _remove_head_frame(self, image_array: np.ndarray, modality: str = 'CT'):
        cleaned_array = image_array.copy()
        
        if modality == 'CT':
            normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            high_intensity_threshold = np.percentile(normalized, 98)  # Stricter
            metal_mask = normalized > high_intensity_threshold
            metal_mask = morphology.remove_small_objects(metal_mask, min_size=1000)  # Larger
            kernel = morphology.ball(1)  # Smaller kernel
            metal_mask = morphology.binary_dilation(metal_mask, kernel)
            background_value = np.median(image_array[normalized < 0.3])
            cleaned_array[metal_mask] = background_value
        else:  # MRI
            normalized = (image_array - image_array.min()) / (image_array.max() - image_array.min())
            high_threshold = np.percentile(normalized, 98)  # Top 2%
            low_threshold = np.percentile(normalized, 2)    # Bottom 2%
            artifact_mask = (normalized > high_threshold) | (normalized < low_threshold)
            artifact_mask = morphology.remove_small_objects(artifact_mask, min_size=500)
            kernel = morphology.ball(1)
            artifact_mask = morphology.binary_dilation(artifact_mask, kernel)
            background_value = np.median(image_array[normalized < 0.5])
            cleaned_array[artifact_mask] = background_value
            metal_mask = artifact_mask
        
        return cleaned_array, metal_mask.astype(np.float32)
    
    def _crop_brain_roi(self, image_array: np.ndarray, mask: np.ndarray):
        brain_coords = np.where(mask > 0)
        
        if len(brain_coords[0]) == 0:
            return image_array, mask
        
        min_z, max_z = brain_coords[0].min(), brain_coords[0].max()
        min_y, max_y = brain_coords[1].min(), brain_coords[1].max()  
        min_x, max_x = brain_coords[2].min(), brain_coords[2].max()
        
        padding = 10
        min_z = max(0, min_z - padding)
        max_z = min(image_array.shape[0], max_z + padding)
        min_y = max(0, min_y - padding)
        max_y = min(image_array.shape[1], max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(image_array.shape[2], max_x + padding)
        
        cropped_image = image_array[min_z:max_z, min_y:max_y, min_x:max_x]
        cropped_mask = mask[min_z:max_z, min_y:max_y, min_x:max_x]
        
        return cropped_image, cropped_mask
    
    def __getitem__(self, idx):
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        ct_path = os.path.join(self.ct_dir, self.ct_files[idx])
        
        mri_sitk = sitk.ReadImage(mri_path)
        ct_sitk = sitk.ReadImage(ct_path)
        
        # BƯỚC 1: N4 bias correction TRƯỚC
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        # BƯỚC 2: Remove head frame TRƯỚC mask
        print(f"Đang loại bỏ head frame cho {self.mri_files[idx]}...")
        mri_array, mri_frame_mask = self._remove_head_frame(mri_array, 'MRI')
        ct_array, ct_frame_mask = self._remove_head_frame(ct_array, 'CT')
        
        # BƯỚC 3: Tạo mask sau khi clean
        mri_mask = self._create_otsu_mask(mri_array)
        ct_mask = self._create_otsu_mask(ct_array)
        
        # BƯỚC 4: Normalize
        mri_array = self._normalize_intensity(mri_array, mri_mask)
        ct_array = self._normalize_intensity(ct_array, ct_mask)
        
        # BƯỚC 5: Crop brain ROI  
        mri_array, mri_mask = self._crop_brain_roi(mri_array, mri_mask)
        ct_array, ct_mask = self._crop_brain_roi(ct_array, ct_mask)
        
        slice_idx = mri_array.shape[0] // 2
        mri_slice = mri_array[slice_idx]
        ct_slice = ct_array[slice_idx]
        
        if mri_slice.shape != (256, 256):
            # Fix: Use scipy.ndimage.zoom instead of skimage
            from scipy.ndimage import zoom
            zoom_h = 256 / mri_slice.shape[0]
            zoom_w = 256 / mri_slice.shape[1]
            mri_slice = zoom(mri_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
            ct_slice = zoom(ct_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
        
        mri_slice = np.clip(mri_slice, 0, 1)
        ct_slice = np.clip(ct_slice, 0, 1)
        
        mri_slice = mri_slice * 2.0 - 1.0
        ct_slice = ct_slice * 2.0 - 1.0
        
        mri_tensor = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0)
        
        return {
            'mri': mri_tensor,
            'ct': ct_tensor,
            'filename': self.mri_files[idx]
        }

def visualize_preprocessing():
    """
    Visualize kết quả tiền xử lý
    """
    dataset = MRIToCTDataset(
        mri_dir="data/MRI", 
        ct_dir="data/CT", 
        is_training=False
    )
    
    sample = dataset[0]
    mri_tensor = sample['mri']
    ct_tensor = sample['ct']
    filename = sample['filename']
    
    mri_np = mri_tensor.squeeze(0).numpy()
    ct_np = ct_tensor.squeeze(0).numpy()
    
    mri_display = (mri_np + 1.0) / 2.0
    ct_display = (ct_np + 1.0) / 2.0
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Preprocessing Results: {filename}', fontsize=16)
    
    im1 = axes[0,0].imshow(mri_np, cmap='gray', vmin=-1, vmax=1)
    axes[0,0].set_title(f'MRI [-1,1] range\nMin: {mri_np.min():.3f}, Max: {mri_np.max():.3f}')
    axes[0,0].axis('off')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(mri_display, cmap='gray', vmin=0, vmax=1)
    axes[0,1].set_title(f'MRI [0,1] display\nMin: {mri_display.min():.3f}, Max: {mri_display.max():.3f}')
    axes[0,1].axis('off')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[1,0].imshow(ct_np, cmap='gray', vmin=-1, vmax=1)
    axes[1,0].set_title(f'CT [-1,1] range\nMin: {ct_np.min():.3f}, Max: {ct_np.max():.3f}')
    axes[1,0].axis('off')
    plt.colorbar(im3, ax=axes[1,0])
    
    im4 = axes[1,1].imshow(ct_display, cmap='gray', vmin=0, vmax=1)
    axes[1,1].set_title(f'CT [0,1] display\nMin: {ct_display.min():.3f}, Max: {ct_display.max():.3f}')
    axes[1,1].axis('off')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.tight_layout()
    
    output_path = 'preprocessing_debug.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Đã lưu visualization: {output_path}")
    
    print("\n" + "="*60)
    print("📊 THỐNG KÊ PREPROCESSING")
    print("="*60)
    print(f"📁 File: {filename}")
    print(f"📐 Shape: {mri_np.shape}")
    
    print(f"\n🧠 MRI Stats:")
    print(f"   Range: [{mri_np.min():.4f}, {mri_np.max():.4f}]")
    print(f"   Mean: {mri_np.mean():.4f}")
    print(f"   Std: {mri_np.std():.4f}")
    print(f"   Non-zero pixels: {np.count_nonzero(mri_np)} / {mri_np.size}")
    
    print(f"\n🦴 CT Stats:")
    print(f"   Range: [{ct_np.min():.4f}, {ct_np.max():.4f}]")
    print(f"   Mean: {ct_np.mean():.4f}")
    print(f"   Std: {ct_np.std():.4f}")
    print(f"   Non-zero pixels: {np.count_nonzero(ct_np)} / {ct_np.size}")
    
    brain_pixels_mri = np.count_nonzero(mri_np)
    brain_pixels_ct = np.count_nonzero(ct_np)
    total_pixels = mri_np.size
    
    brain_ratio_mri = brain_pixels_mri / total_pixels
    brain_ratio_ct = brain_pixels_ct / total_pixels
    
    print(f"\n✅ QUALITY CHECK:")
    print(f"   MRI brain ratio: {brain_ratio_mri:.3f} ({brain_ratio_mri*100:.1f}%)")
    print(f"   CT brain ratio: {brain_ratio_ct:.3f} ({brain_ratio_ct*100:.1f}%)")
    
    if brain_ratio_mri > 0.1 and brain_ratio_ct > 0.1:
        print("   🟢 GOOD: Brain masking hiệu quả")
    else:
        print("   🔴 POOR: Brain masking có vấn đề")
    
    if -1.0 <= mri_np.min() <= -0.9 and 0.9 <= mri_np.max() <= 1.0:
        print("   🟢 GOOD: MRI range chuẩn [-1,1]")
    else:
        print(f"   🔴 POOR: MRI range không chuẩn: [{mri_np.min():.3f}, {mri_np.max():.3f}]")
        
    if -1.0 <= ct_np.min() <= -0.9 and 0.9 <= ct_np.max() <= 1.0:
        print("   🟢 GOOD: CT range chuẩn [-1,1]") 
    else:
        print(f"   🔴 POOR: CT range không chuẩn: [{ct_np.min():.3f}, {ct_np.max():.3f}]")
        
    print("="*60)

if __name__ == "__main__":
    print("🔍 DEBUGGING PREPROCESSING PIPELINE...")
    try:
        visualize_preprocessing()
        print("✅ Debug hoàn thành!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc() 