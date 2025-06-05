import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset
import SimpleITK as sitk
from skimage import filters, morphology
from scipy import ndimage

def compare_preprocessing_methods():
    """
    So sánh phương pháp preprocessing cũ và mới
    """
    print("=== COMPARISON: Old vs New Preprocessing Methods ===")
    
    # Setup paths
    mri_dir = "data/MRI"
    ct_dir = "data/CT"
    
    # Load raw data
    mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
    ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
    
    mid_slice = mri_array_raw.shape[0] // 2
    mri_slice_raw = mri_array_raw[mid_slice]
    ct_slice_raw = ct_array_raw[mid_slice]
    
    print(f"Raw data loaded - MRI: {mri_array_raw.shape}, CT: {ct_array_raw.shape}")
    
    # === OLD METHOD: Otsu Thresholding ===
    def old_otsu_mask(image_array):
        normalized = ((image_array - image_array.min()) / 
                     (image_array.max() - image_array.min()) * 255).astype(np.uint8)
        threshold = filters.threshold_otsu(normalized)
        binary_mask = normalized > threshold
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=1000)
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        return binary_mask.astype(np.float32)
    
    # Apply old method
    print("\n--- Applying OLD method (Otsu) ---")
    mri_mask_old = old_otsu_mask(mri_array_raw)
    ct_mask_old = old_otsu_mask(ct_array_raw)
    
    # === NEW METHOD: MRI-Guided ===
    print("\n--- Applying NEW method (MRI-guided) ---")
    dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False)
    
    # Apply N4 correction
    mri_n4 = dataset._apply_n4_bias_correction(mri_raw)
    mri_array_n4 = sitk.GetArrayFromImage(mri_n4).astype(np.float32)
    
    # Create MRI brain mask
    mri_mask_new = dataset._create_mri_brain_mask(mri_array_n4)
    
    # Apply MRI mask to CT
    ct_cleaned = dataset._apply_mri_mask_to_ct(ct_array_raw, mri_mask_new)
    
    # Get slices for comparison
    mri_mask_old_slice = mri_mask_old[mid_slice]
    ct_mask_old_slice = ct_mask_old[mid_slice]
    mri_mask_new_slice = mri_mask_new[mid_slice]
    ct_cleaned_slice = ct_cleaned[mid_slice]
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Comparison: Old vs New Preprocessing Methods', fontsize=16)
    
    # Row 1: Original images
    axes[0, 0].imshow(mri_slice_raw, cmap='gray')
    axes[0, 0].set_title('Original MRI')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ct_slice_raw, cmap='gray')
    axes[0, 1].set_title('Original CT\n(with couch/headframe)')
    axes[0, 1].axis('off')
    
    axes[0, 2].text(0.5, 0.5, 'ORIGINAL\nDATA', ha='center', va='center', 
                   fontsize=20, transform=axes[0, 2].transAxes)
    axes[0, 2].axis('off')
    
    axes[0, 3].text(0.5, 0.5, 'brain_001.nii.gz\nMiddle slice', ha='center', va='center',
                   fontsize=14, transform=axes[0, 3].transAxes)
    axes[0, 3].axis('off')
    
    # Row 2: Old method (Otsu)
    axes[1, 0].imshow(mri_mask_old_slice, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'OLD: MRI Otsu Mask\nCoverage: {np.sum(mri_mask_old_slice)/mri_mask_old_slice.size*100:.1f}%')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(ct_mask_old_slice, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title(f'OLD: CT Otsu Mask\nCoverage: {np.sum(ct_mask_old_slice)/ct_mask_old_slice.size*100:.1f}%')
    axes[1, 1].axis('off')
    
    # CT with old mask applied
    ct_old_masked = ct_slice_raw.copy()
    ct_old_masked[ct_mask_old_slice == 0] = ct_slice_raw.min()
    ct_old_norm = (ct_old_masked - ct_old_masked.min()) / (ct_old_masked.max() - ct_old_masked.min())
    axes[1, 2].imshow(ct_old_norm, cmap='gray', vmin=0, vmax=1)
    axes[1, 2].set_title('OLD: CT with Otsu mask\n(Still has artifacts)')
    axes[1, 2].axis('off')
    
    axes[1, 3].text(0.5, 0.5, 'OLD METHOD\n(Otsu Thresholding)\n\n❌ Separate masks\n❌ CT artifacts remain', 
                   ha='center', va='center', fontsize=12, transform=axes[1, 3].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    axes[1, 3].axis('off')
    
    # Row 3: New method (MRI-guided)
    axes[2, 0].imshow(mri_mask_new_slice, cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title(f'NEW: MRI Brain Mask\nCoverage: {np.sum(mri_mask_new_slice)/mri_mask_new_slice.size*100:.1f}%')
    axes[2, 0].axis('off')
    
    # Show MRI mask applied to CT
    ct_with_mri_mask = ct_slice_raw.copy()
    ct_with_mri_mask[mri_mask_new_slice == 0] = np.percentile(ct_slice_raw[mri_mask_new_slice == 0], 25)
    ct_mri_norm = (ct_with_mri_mask - ct_with_mri_mask.min()) / (ct_with_mri_mask.max() - ct_with_mri_mask.min())
    axes[2, 1].imshow(ct_mri_norm, cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title('NEW: CT with MRI mask\n(Couch/headframe removed)')
    axes[2, 1].axis('off')
    
    # Final cleaned CT
    ct_final_norm = (ct_cleaned_slice - ct_cleaned_slice.min()) / (ct_cleaned_slice.max() - ct_cleaned_slice.min())
    axes[2, 2].imshow(ct_final_norm, cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title('NEW: Final cleaned CT\n(Enhanced + normalized)')
    axes[2, 2].axis('off')
    
    axes[2, 3].text(0.5, 0.5, 'NEW METHOD\n(MRI-guided)\n\n✅ Unified mask\n✅ Artifacts removed\n✅ Better brain focus', 
                   ha='center', va='center', fontsize=12, transform=axes[2, 3].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison: preprocessing_comparison.png")
    
    # === QUANTITATIVE COMPARISON ===
    print(f"\n--- QUANTITATIVE COMPARISON ---")
    
    # Mask differences
    print(f"OLD - MRI mask coverage: {np.sum(mri_mask_old_slice)/mri_mask_old_slice.size*100:.1f}%")
    print(f"OLD - CT mask coverage: {np.sum(ct_mask_old_slice)/ct_mask_old_slice.size*100:.1f}%")
    print(f"NEW - MRI mask coverage: {np.sum(mri_mask_new_slice)/mri_mask_new_slice.size*100:.1f}%")
    
    # Mask consistency (overlap between MRI and CT masks)
    old_overlap = np.sum((mri_mask_old_slice > 0) & (ct_mask_old_slice > 0))
    old_union = np.sum((mri_mask_old_slice > 0) | (ct_mask_old_slice > 0))
    old_iou = old_overlap / old_union if old_union > 0 else 0
    
    print(f"OLD - MRI-CT mask IoU: {old_iou:.3f}")
    print(f"NEW - Uses same mask for both: 1.000 (perfect consistency)")
    
    # Artifact removal effectiveness
    # Count high-intensity pixels outside brain (likely artifacts)
    background_old = ct_slice_raw[ct_mask_old_slice == 0]
    background_new = ct_slice_raw[mri_mask_new_slice == 0]
    
    if len(background_old) > 0 and len(background_new) > 0:
        artifacts_old = np.sum(background_old > np.percentile(background_old, 95))
        artifacts_new = np.sum(background_new > np.percentile(background_new, 95))
        
        print(f"OLD - High-intensity artifacts in background: {artifacts_old}")
        print(f"NEW - High-intensity artifacts in background: {artifacts_new}")
        print(f"Artifact reduction: {(1 - artifacts_new/artifacts_old)*100:.1f}%")
    
    # Brain tissue preservation
    brain_old = ct_slice_raw[ct_mask_old_slice > 0]
    brain_new = ct_cleaned_slice[mri_mask_new_slice > 0]
    
    if len(brain_old) > 0 and len(brain_new) > 0:
        print(f"OLD - Brain tissue std: {np.std(brain_old):.1f}")
        print(f"NEW - Brain tissue std: {np.std(brain_new):.1f}")
        print(f"Contrast preservation: {np.std(brain_new)/np.std(brain_old)*100:.1f}%")
    
    print(f"\n✓ Preprocessing comparison completed!")

if __name__ == "__main__":
    compare_preprocessing_methods() 