import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset
import SimpleITK as sitk

def test_centered_processing():
    """
    Test processing với augmentation tắt để xác định nguyên nhân CT lệch
    """
    print("=== TEST: Centered Processing Analysis ===")
    
    # Test với is_training=False để không có augmentation
    mri_dir = "data/MRI"
    ct_dir = "data/CT"
    
    # Tạo dataset không augmentation
    dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False, slice_range=(55, 56))
    
    # Get một sample
    sample = dataset[0]
    mri_processed = sample['mri'].squeeze(0).numpy()
    ct_processed = sample['ct'].squeeze(0).numpy()
    
    print(f"Processed shape: {mri_processed.shape}")
    print(f"Processed range: MRI [{mri_processed.min():.3f}, {mri_processed.max():.3f}]")
    print(f"Processed range: CT [{ct_processed.min():.3f}, {ct_processed.max():.3f}]")
    
    # Load raw data cho comparison
    mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
    ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
    
    # Raw slices
    mri_raw_slice = mri_array_raw[55]
    ct_raw_slice = ct_array_raw[55]
    
    # Test nhiều files để xem pattern
    print(f"\n=== Testing Multiple Files ===")
    for i in range(3):
        sample_i = dataset[i]
        mri_i = sample_i['mri'].squeeze(0).numpy()
        ct_i = sample_i['ct'].squeeze(0).numpy()
        filename = sample_i['filename']
        
        # Check center of mass
        mri_mask = (mri_i > -0.8)  # Approximate brain mask
        ct_mask = (ct_i > -0.8)
        
        if np.sum(mri_mask) > 0:
            mri_com = np.array([
                np.sum(np.arange(mri_mask.shape[0])[:, None] * mri_mask) / np.sum(mri_mask),
                np.sum(np.arange(mri_mask.shape[1])[None, :] * mri_mask) / np.sum(mri_mask)
            ])
        else:
            mri_com = np.array([128, 128])
            
        if np.sum(ct_mask) > 0:
            ct_com = np.array([
                np.sum(np.arange(ct_mask.shape[0])[:, None] * ct_mask) / np.sum(ct_mask),
                np.sum(np.arange(ct_mask.shape[1])[None, :] * ct_mask) / np.sum(ct_mask)
            ])
        else:
            ct_com = np.array([128, 128])
        
        com_diff = ct_com - mri_com
        
        print(f"{filename}:")
        print(f"  MRI center of mass: [{mri_com[0]:.1f}, {mri_com[1]:.1f}]")
        print(f"  CT center of mass:  [{ct_com[0]:.1f}, {ct_com[1]:.1f}]")
        print(f"  Difference: [{com_diff[0]:.1f}, {com_diff[1]:.1f}]")
        
        if abs(com_diff[0]) > 5 or abs(com_diff[1]) > 5:
            print(f"  ⚠️ MISALIGNMENT detected!")
        else:
            print(f"  ✓ Well aligned")
    
    # Detailed visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Centered Processing Analysis (No Augmentation)', fontsize=14)
    
    # Row 1: Raw data
    axes[0, 0].imshow(mri_raw_slice, cmap='gray')
    axes[0, 0].set_title('Raw MRI')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ct_raw_slice, cmap='gray')
    axes[0, 1].set_title('Raw CT')
    axes[0, 1].axis('off')
    
    # Row 2: Processed data
    axes[1, 0].imshow(mri_processed, cmap='gray', vmin=-1, vmax=1)
    axes[1, 0].set_title('Processed MRI')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ct_processed, cmap='gray', vmin=-1, vmax=1)
    axes[1, 1].set_title('Processed CT')
    axes[1, 1].axis('off')
    
    # Center line profiles
    center_y = 128
    axes[0, 2].plot(mri_raw_slice[center_y, :], label='Raw MRI', alpha=0.7)
    axes[0, 2].plot(ct_raw_slice[center_y, :], label='Raw CT', alpha=0.7)
    axes[0, 2].set_title('Raw Center Line')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 2].plot(mri_processed[center_y, :], label='Processed MRI', alpha=0.7)
    axes[1, 2].plot(ct_processed[center_y, :], label='Processed CT', alpha=0.7)
    axes[1, 2].set_title('Processed Center Line')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Center column profiles  
    center_x = 128
    axes[0, 3].plot(mri_raw_slice[:, center_x], label='Raw MRI', alpha=0.7)
    axes[0, 3].plot(ct_raw_slice[:, center_x], label='Raw CT', alpha=0.7)
    axes[0, 3].set_title('Raw Center Column')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    axes[1, 3].plot(mri_processed[:, center_x], label='Processed MRI', alpha=0.7)
    axes[1, 3].plot(ct_processed[:, center_x], label='Processed CT', alpha=0.7)
    axes[1, 3].set_title('Processed Center Column')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('centered_processing_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: centered_processing_analysis.png")
    
    # Test với augmentation ON
    print(f"\n=== Testing with Augmentation ON ===")
    dataset_aug = MRIToCTDataset(mri_dir, ct_dir, is_training=True, slice_range=(55, 56))
    
    # Test multiple samples để xem random effect
    for i in range(3):
        sample_aug = dataset_aug[0]  # Same file, different random seed
        mri_aug = sample_aug['mri'].squeeze(0).numpy()
        ct_aug = sample_aug['ct'].squeeze(0).numpy()
        
        # Check center of mass again
        mri_mask_aug = (mri_aug > -0.8)
        ct_mask_aug = (ct_aug > -0.8)
        
        if np.sum(mri_mask_aug) > 0:
            mri_com_aug = np.array([
                np.sum(np.arange(mri_mask_aug.shape[0])[:, None] * mri_mask_aug) / np.sum(mri_mask_aug),
                np.sum(np.arange(mri_mask_aug.shape[1])[None, :] * mri_mask_aug) / np.sum(mri_mask_aug)
            ])
        else:
            mri_com_aug = np.array([128, 128])
            
        if np.sum(ct_mask_aug) > 0:
            ct_com_aug = np.array([
                np.sum(np.arange(ct_mask_aug.shape[0])[:, None] * ct_mask_aug) / np.sum(ct_mask_aug),
                np.sum(np.arange(ct_mask_aug.shape[1])[None, :] * ct_mask_aug) / np.sum(ct_mask_aug)
            ])
        else:
            ct_com_aug = np.array([128, 128])
        
        com_diff_aug = ct_com_aug - mri_com_aug
        
        print(f"Sample {i+1} (with augmentation):")
        print(f"  MRI center of mass: [{mri_com_aug[0]:.1f}, {mri_com_aug[1]:.1f}]")
        print(f"  CT center of mass:  [{ct_com_aug[0]:.1f}, {ct_com_aug[1]:.1f}]")
        print(f"  Difference: [{com_diff_aug[0]:.1f}, {com_diff_aug[1]:.1f}]")
        
        if abs(com_diff_aug[0]) > 5 or abs(com_diff_aug[1]) > 5:
            print(f"  ⚠️ AUGMENTATION caused misalignment!")
        else:
            print(f"  ✓ Still aligned")
    
    return {
        'no_aug_com_diff': com_diff if 'com_diff' in locals() else [0, 0],
        'aug_com_diff': com_diff_aug if 'com_diff_aug' in locals() else [0, 0]
    }

if __name__ == "__main__":
    import torch
    results = test_centered_processing()
    
    print(f"\n=== SUMMARY ===")
    print(f"Without augmentation: Center difference = {results['no_aug_com_diff']}")
    print(f"With augmentation: Center difference = {results['aug_com_diff']}")
    
    no_aug_max = max(abs(results['no_aug_com_diff'][0]), abs(results['no_aug_com_diff'][1]))
    aug_max = max(abs(results['aug_com_diff'][0]), abs(results['aug_com_diff'][1]))
    
    if no_aug_max > 5:
        print("❌ ISSUE: CT misaligned even without augmentation!")
        print("Problem likely in: mask creation, crop ROI, or resize step")
    elif aug_max > 5:
        print("❌ ISSUE: Augmentation causing misalignment!")
        print("Problem in: _augment_data function")
    else:
        print("✅ No significant misalignment detected")
        print("Issue might be subtle or in specific cases only") 