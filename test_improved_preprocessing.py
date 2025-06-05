import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset
import SimpleITK as sitk

def test_improved_preprocessing():
    """
    Test script cho improved preprocessing pipeline
    """
    print("=== TEST: Improved MRI-Guided Preprocessing ===")
    
    # Setup paths
    mri_dir = "data/MRI"
    ct_dir = "data/CT"
    
    # Load raw data cho comparison
    mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
    ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
    
    mid_slice = mri_array_raw.shape[0] // 2
    print(f"Testing with slice {mid_slice} from brain_001.nii.gz")
    
    # Test với dataset cải thiện
    dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False, slice_range=(mid_slice, mid_slice+1))
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Load processed sample
    sample = dataset[0]
    mri_processed = sample['mri'].squeeze(0)
    ct_processed = sample['ct'].squeeze(0)
    
    # Convert từ [-1,1] về [0,1] để visualize
    mri_vis = (mri_processed.numpy() + 1) / 2
    ct_vis = (ct_processed.numpy() + 1) / 2
    
    # Get intermediate steps để hiểu pipeline
    # Simulate pipeline steps manually
    mri_sitk = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_sitk = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    # N4 correction
    mri_n4 = dataset._apply_n4_bias_correction(mri_sitk)
    mri_array_n4 = sitk.GetArrayFromImage(mri_n4).astype(np.float32)
    
    # Create brain mask
    brain_mask = dataset._create_mri_brain_mask(mri_array_n4)
    
    # Apply mask to CT
    ct_masked = dataset._apply_mri_mask_to_ct(ct_array_raw, brain_mask)
    
    # Gentle clipping
    mri_clipped = dataset._gentle_outlier_clipping(mri_array_n4, brain_mask, 'MRI')
    ct_clipped = dataset._gentle_outlier_clipping(ct_masked, brain_mask, 'CT')
    
    # Normalization
    mri_normalized = dataset._normalize_intensity(mri_clipped, brain_mask, 'MRI')
    ct_normalized = dataset._normalize_intensity(ct_clipped, brain_mask, 'CT')
    
    # Get slices
    mri_raw_slice = mri_array_raw[mid_slice]
    ct_raw_slice = ct_array_raw[mid_slice]
    mask_slice = brain_mask[mid_slice]
    ct_masked_slice = ct_masked[mid_slice]
    ct_clipped_slice = ct_clipped[mid_slice]
    mri_normalized_slice = mri_normalized[mid_slice]
    ct_normalized_slice = ct_normalized[mid_slice]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('Improved MRI-Guided Preprocessing Pipeline: brain_001.nii.gz', fontsize=16)
    
    # Row 1: Original + Brain Mask
    axes[0, 0].imshow(mri_raw_slice, cmap='gray')
    axes[0, 0].set_title(f'Original MRI\nRange: [{mri_raw_slice.min():.0f}, {mri_raw_slice.max():.0f}]')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(ct_raw_slice, cmap='gray')
    axes[0, 1].set_title(f'Original CT (with artifacts)\nRange: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}]')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_slice, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title(f'MRI Brain Mask\nCoverage: {np.sum(mask_slice)/mask_slice.size*100:.1f}%')
    axes[0, 2].axis('off')
    
    # Overlay
    overlay = ct_raw_slice.copy()
    overlay_norm = (overlay - overlay.min()) / (overlay.max() - overlay.min())
    overlay_rgb = np.stack([overlay_norm, overlay_norm, overlay_norm], axis=-1)
    overlay_rgb[mask_slice > 0, 1] = 1  # Green for brain
    axes[0, 3].imshow(overlay_rgb)
    axes[0, 3].set_title('CT + Brain Mask Overlay\n(Green = Brain Region)')
    axes[0, 3].axis('off')
    
    axes[0, 4].text(0.5, 0.5, 'STEP 1:\nOriginal Data\n+\nBrain Mask\nCreation', 
                   ha='center', va='center', fontsize=14, transform=axes[0, 4].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[0, 4].axis('off')
    
    # Row 2: Artifact Removal
    axes[1, 0].imshow((ct_masked_slice - ct_masked_slice.min()) / (ct_masked_slice.max() - ct_masked_slice.min()), 
                     cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('CT + MRI Mask Applied\n(Couch/headframe removed)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow((ct_clipped_slice - ct_clipped_slice.min()) / (ct_clipped_slice.max() - ct_clipped_slice.min()), 
                     cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('CT + Gentle Outlier Clipping\n(Artifacts cleaned)')
    axes[1, 1].axis('off')
    
    # Difference maps
    diff_mask = np.abs(ct_raw_slice - ct_masked_slice) > (ct_raw_slice.max() - ct_raw_slice.min()) * 0.05
    axes[1, 2].imshow(diff_mask, cmap='Reds', vmin=0, vmax=1)
    axes[1, 2].set_title('Masking Changes\n(Red = Changed pixels)')
    axes[1, 2].axis('off')
    
    diff_clip = np.abs(ct_masked_slice - ct_clipped_slice) > (ct_masked_slice.max() - ct_masked_slice.min()) * 0.05
    axes[1, 3].imshow(diff_clip, cmap='Reds', vmin=0, vmax=1)
    axes[1, 3].set_title('Clipping Changes\n(Red = Outliers removed)')
    axes[1, 3].axis('off')
    
    axes[1, 4].text(0.5, 0.5, 'STEP 2:\nArtifact\nRemoval\n\n✓ MRI mask\n✓ Outlier clipping', 
                   ha='center', va='center', fontsize=14, transform=axes[1, 4].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    axes[1, 4].axis('off')
    
    # Row 3: Final Results
    axes[2, 0].imshow(mri_vis, cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title(f'Final MRI\nRange: [{mri_vis.min():.3f}, {mri_vis.max():.3f}]')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(ct_vis, cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title(f'Final CT\nRange: [{ct_vis.min():.3f}, {ct_vis.max():.3f}]')
    axes[2, 1].axis('off')
    
    # Comparison với normalized intermediate
    axes[2, 2].imshow(mri_normalized_slice, cmap='gray', vmin=0, vmax=1)
    axes[2, 2].set_title('MRI Normalized\n(Before [-1,1] conversion)')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(ct_normalized_slice, cmap='gray', vmin=0, vmax=1)
    axes[2, 3].set_title('CT Normalized\n(Before [-1,1] conversion)')
    axes[2, 3].axis('off')
    
    axes[2, 4].text(0.5, 0.5, 'STEP 3:\nNormalization\n+\nFinal Output\n\n✓ Tissue-aware norm\n✓ [-1,1] for training', 
                   ha='center', va='center', fontsize=14, transform=axes[2, 4].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    axes[2, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig('improved_preprocessing_pipeline.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: improved_preprocessing_pipeline.png")
    
    # Quantitative analysis
    print(f"\n--- QUANTITATIVE ANALYSIS ---")
    
    # Brain mask quality
    brain_coverage = np.sum(mask_slice) / mask_slice.size * 100
    print(f"Brain mask coverage: {brain_coverage:.1f}%")
    
    # Artifact removal effectiveness
    pixels_changed_mask = np.sum(diff_mask)
    pixels_changed_clip = np.sum(diff_clip)
    total_pixels = mask_slice.size
    
    print(f"Pixels changed by masking: {pixels_changed_mask} ({pixels_changed_mask/total_pixels*100:.1f}%)")
    print(f"Pixels changed by clipping: {pixels_changed_clip} ({pixels_changed_clip/total_pixels*100:.1f}%)")
    
    # Tissue contrast preservation
    # Compare std in brain region
    brain_region_raw = ct_raw_slice[mask_slice > 0]
    brain_region_final = ct_vis[mask_slice > 0] if np.any(mask_slice) else []
    
    if len(brain_region_raw) > 0 and len(brain_region_final) > 0:
        contrast_raw = np.std(brain_region_raw)
        contrast_final = np.std(brain_region_final)
        contrast_preservation = contrast_final / (contrast_raw / (ct_raw_slice.max() - ct_raw_slice.min())) * 100
        
        print(f"Original brain tissue contrast: {contrast_raw:.1f}")
        print(f"Final brain tissue contrast: {contrast_final:.3f}")
        print(f"Contrast preservation: {contrast_preservation:.1f}%")
    
    # Range analysis
    print(f"\nOriginal CT range: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}]")
    print(f"Final CT range: [{ct_vis.min():.3f}, {ct_vis.max():.3f}]")
    
    # Test với file khác
    if len(dataset) > 1:
        print(f"\n--- Quick test with brain_002.nii.gz ---")
        try:
            dataset_2 = MRIToCTDataset(mri_dir, ct_dir, is_training=False, slice_range=(mid_slice, mid_slice+1))
            sample_2 = dataset_2[1]
            mri_2 = sample_2['mri']
            ct_2 = sample_2['ct']
            print(f"brain_002 - MRI range: [{mri_2.min():.3f}, {mri_2.max():.3f}]")
            print(f"brain_002 - CT range: [{ct_2.min():.3f}, {ct_2.max():.3f}]")
        except Exception as e:
            print(f"Error testing brain_002: {e}")
    
    print(f"\n✓ Improved preprocessing test completed successfully!")
    
    return {
        'brain_coverage': brain_coverage,
        'pixels_changed_mask': pixels_changed_mask / total_pixels * 100,
        'pixels_changed_clip': pixels_changed_clip / total_pixels * 100,
        'final_mri_range': [mri_vis.min(), mri_vis.max()],
        'final_ct_range': [ct_vis.min(), ct_vis.max()]
    }

if __name__ == "__main__":
    results = test_improved_preprocessing()
    print(f"\nFinal Results Summary:")
    for key, value in results.items():
        print(f"  {key}: {value}") 