import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset

def debug_mri_guided_preprocessing():
    """
    Debug script để test phương pháp MRI-guided preprocessing mới
    """
    print("=== DEBUG: MRI-Guided Preprocessing cho việc loại bỏ CT artifacts ===")
    
    # Setup paths
    mri_dir = "data/MRI"
    ct_dir = "data/CT" 
    
    # Tạo dataset
    dataset = MRIToCTDataset(
        mri_dir=mri_dir,
        ct_dir=ct_dir,
        is_training=False,  # Không dùng augmentation để debug
        slice_range=(30, 60)  # Lấy slice ở giữa
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test với file đầu tiên
    try:
        print("\n--- Test preprocessing với brain_001.nii.gz ---")
        
        # Load raw data để so sánh
        import nibabel as nib
        import SimpleITK as sitk
        
        mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
        ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
        
        mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
        ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
        
        print(f"MRI raw shape: {mri_array_raw.shape}")
        print(f"CT raw shape: {ct_array_raw.shape}")
        
        # Lấy middle slice
        mid_slice = mri_array_raw.shape[0] // 2
        mri_slice_raw = mri_array_raw[mid_slice]
        ct_slice_raw = ct_array_raw[mid_slice]
        
        # Load qua dataset (với preprocessing)
        sample = dataset[0]
        mri_processed = sample['mri'].squeeze(0)  # Remove channel dimension
        ct_processed = sample['ct'].squeeze(0)    # Remove channel dimension
        
        # Convert back từ [-1,1] về [0,1] để visualize
        mri_processed = (mri_processed.numpy() + 1) / 2
        ct_processed = (ct_processed.numpy() + 1) / 2
        
        print(f"MRI processed shape: {mri_processed.shape}")
        print(f"CT processed shape: {ct_processed.shape}")
        
        # Tạo visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('MRI-Guided Preprocessing Results: brain_001.nii.gz', fontsize=16)
        
        # Row 1: MRI
        # Original MRI
        im1 = axes[0, 0].imshow(mri_slice_raw, cmap='gray')
        axes[0, 0].set_title(f'MRI Raw\nRange: [{mri_slice_raw.min():.1f}, {mri_slice_raw.max():.1f}]')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # Processed MRI
        im2 = axes[0, 1].imshow(mri_processed, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'MRI Processed\nRange: [{mri_processed.min():.3f}, {mri_processed.max():.3f}]')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # MRI Brain Mask (tạo để hiển thị)
        dataset_temp = MRIToCTDataset(mri_dir, ct_dir, is_training=False)
        # Simulate lấy raw data
        mri_sitk = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
        mri_sitk = dataset_temp._apply_n4_bias_correction(mri_sitk)
        mri_array_n4 = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        brain_mask = dataset_temp._create_mri_brain_mask(mri_array_n4)
        mask_slice = brain_mask[mid_slice]
        
        im3 = axes[0, 2].imshow(mask_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, 2].set_title(f'MRI Brain Mask\nBrain pixels: {np.sum(mask_slice > 0)}')
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # MRI Overlay
        overlay = mri_slice_raw.copy()
        overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min())
        overlay_rgb = np.stack([overlay_norm, overlay_norm, overlay_norm], axis=-1)
        overlay_rgb[mask_slice > 0, 1] = 1  # Green overlay cho brain region
        axes[0, 3].imshow(overlay_rgb)
        axes[0, 3].set_title('MRI + Brain Mask Overlay\n(Green = Brain)')
        axes[0, 3].axis('off')
        
        # Row 2: CT  
        # Original CT
        im4 = axes[1, 0].imshow(ct_slice_raw, cmap='gray')
        axes[1, 0].set_title(f'CT Raw\nRange: [{ct_slice_raw.min():.1f}, {ct_slice_raw.max():.1f}]')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Processed CT
        im5 = axes[1, 1].imshow(ct_processed, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title(f'CT Processed (MRI-guided)\nRange: [{ct_processed.min():.3f}, {ct_processed.max():.3f}]')
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # CT with MRI mask applied
        ct_temp = dataset_temp._apply_mri_mask_to_ct(ct_array_raw, brain_mask)
        ct_masked_slice = ct_temp[mid_slice]
        ct_masked_norm = (ct_masked_slice - ct_masked_slice.min()) / (ct_masked_slice.max() - ct_masked_slice.min())
        im6 = axes[1, 2].imshow(ct_masked_norm, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title(f'CT + MRI Mask Applied\nArtifacts removed')
        axes[1, 2].axis('off')
        plt.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # Difference map (before vs after)
        ct_raw_norm = (ct_slice_raw - ct_slice_raw.min()) / (ct_slice_raw.max() - ct_slice_raw.min())
        diff_map = np.abs(ct_raw_norm - ct_masked_norm)
        im7 = axes[1, 3].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
        axes[1, 3].set_title('Difference Map\n(Red = Changed regions)')
        axes[1, 3].axis('off')
        plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig('mri_guided_preprocessing_debug.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: mri_guided_preprocessing_debug.png")
        
        # Print statistics
        print(f"\n--- Statistics ---")
        print(f"Original CT range: [{ct_slice_raw.min():.1f}, {ct_slice_raw.max():.1f}]")
        print(f"Processed CT range: [{ct_processed.min():.3f}, {ct_processed.max():.3f}]")
        print(f"Brain mask coverage: {np.sum(mask_slice > 0) / mask_slice.size * 100:.1f}%")
        print(f"Pixels changed: {np.sum(diff_map > 0.1) / diff_map.size * 100:.1f}%")
        
        # Test với file khác nếu có
        if len(dataset) > 1:
            print(f"\n--- Quick test với {dataset.mri_files[1]} ---")
            sample_2 = dataset[1]
            mri_2 = sample_2['mri']
            ct_2 = sample_2['ct']
            print(f"File 2 - MRI range: [{mri_2.min():.3f}, {mri_2.max():.3f}]")
            print(f"File 2 - CT range: [{ct_2.min():.3f}, {ct_2.max():.3f}]")
        
        print(f"\n✓ MRI-guided preprocessing debug completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during debugging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mri_guided_preprocessing() 