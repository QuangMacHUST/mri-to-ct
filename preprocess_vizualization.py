import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset
import SimpleITK as sitk

def debug_ct_shift_issue():
    """
    Debug để tìm nguyên nhân CT bị lệch sang một bên
    """
    print("=== DEBUG: CT Shift Issue Analysis ===")
    
    # Load raw data
    mri_dir = "data/MRI"
    ct_dir = "data/CT"
    
    print("Loading raw data...")
    mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
    ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
    
    mid_slice = mri_array_raw.shape[0] // 2
    print(f"Analyzing slice {mid_slice}")
    
    # Step by step analysis
    dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False)
    
    # Manually go through preprocessing steps
    print("\n=== Manual Step-by-Step Analysis ===")
    
    # 1. Load và N4 correction
    mri_sitk = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_sitk = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_sitk = dataset._apply_n4_bias_correction(mri_sitk)
    
    mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
    ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
    
    print(f"After N4: MRI shape={mri_array.shape}, CT shape={ct_array.shape}")
    
    # 2. Create brain mask
    mri_mask = dataset._create_brain_with_skull_mask(mri_array)
    print(f"Brain mask coverage: {np.sum(mri_mask)/np.prod(mri_mask.shape)*100:.1f}%")
    
    # 3. Apply mask to CT
    ct_masked = dataset._apply_mri_mask_to_ct(ct_array, mri_mask)
    
    # 4. Clip outliers  
    mri_clipped = dataset._gentle_outlier_clipping(mri_array, mri_mask, 'MRI')
    ct_clipped = dataset._gentle_outlier_clipping(ct_masked, mri_mask, 'CT')
    
    # 5. Normalize
    mri_norm = dataset._normalize_intensity(mri_clipped, mri_mask, 'MRI')
    ct_norm = dataset._normalize_intensity(ct_clipped, mri_mask, 'CT')
    
    # 6. Crop ROI - THIS MIGHT BE THE ISSUE
    print(f"\nBefore crop: MRI shape={mri_norm.shape}, CT shape={ct_norm.shape}")
    
    mri_cropped, mri_mask_cropped = dataset._crop_brain_roi(mri_norm, mri_mask)
    ct_cropped, _ = dataset._crop_brain_roi(ct_norm, mri_mask)
    
    print(f"After crop: MRI shape={mri_cropped.shape}, CT shape={ct_cropped.shape}")
    
    # Analyze crop coordinates
    brain_coords = np.where(mri_mask > 0)
    if len(brain_coords[0]) > 0:
        min_z, max_z = brain_coords[0].min(), brain_coords[0].max()
        min_y, max_y = brain_coords[1].min(), brain_coords[1].max()  
        min_x, max_x = brain_coords[2].min(), brain_coords[2].max()
        
        padding = 10
        crop_min_z = max(0, min_z - padding)
        crop_max_z = min(mri_array.shape[0], max_z + padding)
        crop_min_y = max(0, min_y - padding)
        crop_max_y = min(mri_array.shape[1], max_y + padding)
        crop_min_x = max(0, min_x - padding)
        crop_max_x = min(mri_array.shape[2], max_x + padding)
        
        print(f"\nCrop coordinates:")
        print(f"Z: {crop_min_z}:{crop_max_z} (original: 0:{mri_array.shape[0]})")
        print(f"Y: {crop_min_y}:{crop_max_y} (original: 0:{mri_array.shape[1]})")
        print(f"X: {crop_min_x}:{crop_max_x} (original: 0:{mri_array.shape[2]})")
        
        # Check if crop is centered
        original_center_y = mri_array.shape[1] // 2
        original_center_x = mri_array.shape[2] // 2
        crop_center_y = (crop_min_y + crop_max_y) // 2
        crop_center_x = (crop_min_x + crop_max_x) // 2
        
        shift_y = crop_center_y - original_center_y
        shift_x = crop_center_x - original_center_x
        
        print(f"\nCenter analysis:")
        print(f"Original center: Y={original_center_y}, X={original_center_x}")
        print(f"Crop center: Y={crop_center_y}, X={crop_center_x}")
        print(f"Shift: Y={shift_y}, X={shift_x}")
        
        if abs(shift_y) > 10 or abs(shift_x) > 10:
            print(f"⚠️ FOUND ISSUE: Significant shift detected!")
            print(f"Y shift: {shift_y} pixels, X shift: {shift_x} pixels")
        else:
            print("✓ Crop appears centered")
    
    # 7. Get slice và resize
    mri_slice = mri_cropped[mid_slice if mid_slice < mri_cropped.shape[0] else mri_cropped.shape[0]//2]
    ct_slice = ct_cropped[mid_slice if mid_slice < ct_cropped.shape[0] else ct_cropped.shape[0]//2]
    
    print(f"Slice shape before resize: {mri_slice.shape}")
    
    # Resize to 256x256
    if mri_slice.shape != (256, 256):
        from scipy.ndimage import zoom
        zoom_h = 256 / mri_slice.shape[0]
        zoom_w = 256 / mri_slice.shape[1]
        mri_slice_resized = zoom(mri_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
        ct_slice_resized = zoom(ct_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
        print(f"Zoom factors: H={zoom_h:.3f}, W={zoom_w:.3f}")
    else:
        mri_slice_resized = mri_slice
        ct_slice_resized = ct_slice
    
    # Final steps
    mri_final = np.clip(mri_slice_resized, 0, 1) * 2.0 - 1.0
    ct_final = np.clip(ct_slice_resized, 0, 1) * 2.0 - 1.0
    
    # Compare với processed result từ dataloader
    print(f"\n=== Compare with DataLoader Result ===")
    sample = dataset[0]
    mri_dataloader = sample['mri'].squeeze(0).numpy()
    ct_dataloader = sample['ct'].squeeze(0).numpy()
    
    print(f"Manual processing: MRI {mri_final.shape}, CT {ct_final.shape}")
    print(f"DataLoader result: MRI {mri_dataloader.shape}, CT {ct_dataloader.shape}")
    
    # Check differences
    mri_diff = np.abs(mri_final - mri_dataloader)
    ct_diff = np.abs(ct_final - ct_dataloader)
    
    print(f"Differences: MRI max={mri_diff.max():.6f}, CT max={ct_diff.max():.6f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    fig.suptitle('CT Shift Debug Analysis - Step by Step', fontsize=16)
    
    # Row 1: Original data
    axes[0, 0].imshow(mri_array_raw[mid_slice], cmap='gray')
    axes[0, 0].set_title('Raw MRI')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ct_array_raw[mid_slice], cmap='gray')
    axes[0, 1].set_title('Raw CT')
    axes[0, 1].axis('off')
    
    # Show mask
    axes[0, 2].imshow(mri_mask[mid_slice], cmap='viridis')
    axes[0, 2].set_title('Brain Mask')
    axes[0, 2].axis('off')
    
    # After masking
    axes[0, 3].imshow(mri_norm[mid_slice], cmap='gray')
    axes[0, 3].set_title('MRI Normalized')
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(ct_norm[mid_slice], cmap='gray')
    axes[0, 4].set_title('CT Masked & Normalized')
    axes[0, 4].axis('off')
    
    # Overlay mask on CT to check alignment
    ct_overlay = ct_norm[mid_slice].copy()
    mask_edges = mri_mask[mid_slice]
    axes[0, 5].imshow(ct_overlay, cmap='gray')
    axes[0, 5].contour(mask_edges, colors='red', linewidths=1)
    axes[0, 5].set_title('CT with Mask Overlay')
    axes[0, 5].axis('off')
    
    # Row 2: After cropping
    crop_mid = mri_cropped.shape[0] // 2
    
    axes[1, 0].imshow(mri_cropped[crop_mid], cmap='gray')
    axes[1, 0].set_title('MRI Cropped')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ct_cropped[crop_mid], cmap='gray')
    axes[1, 1].set_title('CT Cropped')
    axes[1, 1].axis('off')
    
    # After resize
    axes[1, 2].imshow(mri_slice_resized, cmap='gray')
    axes[1, 2].set_title('MRI Resized')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(ct_slice_resized, cmap='gray')
    axes[1, 3].set_title('CT Resized')
    axes[1, 3].axis('off')
    
    # Final result
    axes[1, 4].imshow(mri_final, cmap='gray', vmin=-1, vmax=1)
    axes[1, 4].set_title('Final MRI [-1,1]')
    axes[1, 4].axis('off')
    
    axes[1, 5].imshow(ct_final, cmap='gray', vmin=-1, vmax=1)
    axes[1, 5].set_title('Final CT [-1,1]')
    axes[1, 5].axis('off')
    
    # Row 3: DataLoader comparison
    axes[2, 0].imshow(mri_dataloader, cmap='gray', vmin=-1, vmax=1)
    axes[2, 0].set_title('DataLoader MRI')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(ct_dataloader, cmap='gray', vmin=-1, vmax=1)
    axes[2, 1].set_title('DataLoader CT')
    axes[2, 1].axis('off')
    
    # Difference maps
    axes[2, 2].imshow(mri_diff, cmap='hot', vmin=0, vmax=0.1)
    axes[2, 2].set_title(f'MRI Diff (max={mri_diff.max():.4f})')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(ct_diff, cmap='hot', vmin=0, vmax=0.1)
    axes[2, 3].set_title(f'CT Diff (max={ct_diff.max():.4f})')
    axes[2, 3].axis('off')
    
    # Center line analysis
    center_line = ct_dataloader.shape[1] // 2
    axes[2, 4].plot(ct_dataloader[center_line, :], label='DataLoader CT', alpha=0.7)
    axes[2, 4].plot(ct_final[center_line, :], label='Manual CT', alpha=0.7)
    axes[2, 4].set_title('Center Line Profile')
    axes[2, 4].legend()
    axes[2, 4].grid(True, alpha=0.3)
    
    # Check symmetry
    left_half = ct_dataloader[:, :128]
    right_half = np.fliplr(ct_dataloader[:, 128:])
    symmetry_diff = np.abs(left_half - right_half)
    axes[2, 5].imshow(symmetry_diff, cmap='hot')
    axes[2, 5].set_title(f'Symmetry Check\n(mean diff={symmetry_diff.mean():.4f})')
    axes[2, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('ct_shift_debug_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved debug visualization: mô tả tiền xử lý ảnh.png")
    
    return {
        'shift_y': shift_y if 'shift_y' in locals() else 0,
        'shift_x': shift_x if 'shift_x' in locals() else 0,
        'crop_coords': {
            'y': (crop_min_y, crop_max_y) if 'crop_min_y' in locals() else None,
            'x': (crop_min_x, crop_max_x) if 'crop_min_x' in locals() else None
        }
    }

if __name__ == "__main__":
    import torch
    results = debug_ct_shift_issue()
    
    print(f"\n=== DEBUG SUMMARY ===")
    print(f"Y shift: {results['shift_y']} pixels")
    print(f"X shift: {results['shift_x']} pixels")
    
    if abs(results['shift_y']) > 10 or abs(results['shift_x']) > 10:
        print("❌ ISSUE: Significant shift detected in crop ROI!")
        print("Recommend: Adjust crop_brain_roi function to ensure centered cropping")
    else:
        print("✅ Crop appears properly centered")
        print("Issue might be elsewhere - check brain mask creation or other steps") 