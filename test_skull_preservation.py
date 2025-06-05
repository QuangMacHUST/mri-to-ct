import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset
import SimpleITK as sitk

def test_skull_preservation():
    """
    Test script để kiểm tra skull preservation trong improved preprocessing
    """
    print("=== TEST: Skull Preservation in MRI-Guided Preprocessing ===")
    
    # Setup paths
    mri_dir = "data/MRI"
    ct_dir = "data/CT"
    
    # Load raw data
    mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
    ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
    
    mid_slice = mri_array_raw.shape[0] // 2
    print(f"Testing skull preservation with slice {mid_slice} from brain_001.nii.gz")
    
    # Create dataset với improved mask
    dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False, slice_range=(mid_slice, mid_slice+1))
    
    # Manual pipeline steps để analyze từng bước
    mri_sitk = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_sitk = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    # N4 correction
    mri_n4 = dataset._apply_n4_bias_correction(mri_sitk)
    mri_array_n4 = sitk.GetArrayFromImage(mri_n4).astype(np.float32)
    
    # Create improved mask (brain + skull)
    comprehensive_mask = dataset._create_brain_with_skull_mask(mri_array_n4)
    
    # Apply mask to CT
    ct_masked = dataset._apply_mri_mask_to_ct(ct_array_raw, comprehensive_mask)
    
    # Full preprocessing
    sample = dataset[0]
    mri_final = (sample['mri'].squeeze(0).numpy() + 1) / 2
    ct_final = (sample['ct'].squeeze(0).numpy() + 1) / 2
    
    # Get slices
    mri_raw_slice = mri_array_raw[mid_slice]
    ct_raw_slice = ct_array_raw[mid_slice]
    mask_slice = comprehensive_mask[mid_slice]
    ct_masked_slice = ct_masked[mid_slice]
    
    # Analyze skull region specifically
    # Identify skull in original CT (high HU values outside brain)
    skull_candidates = (ct_raw_slice > 200) & (ct_raw_slice < 2000)  # Typical bone HU range
    
    # Check skull preservation
    skull_in_mask = skull_candidates & (mask_slice > 0)
    skull_preserved = np.sum(skull_in_mask) / np.sum(skull_candidates) * 100 if np.sum(skull_candidates) > 0 else 0
    
    # Visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Skull Preservation Analysis: brain_001.nii.gz', fontsize=16)
    
    # Row 1: Original data
    axes[0, 0].imshow(mri_raw_slice, cmap='gray')
    axes[0, 0].set_title(f'Original MRI\nRange: [{mri_raw_slice.min():.0f}, {mri_raw_slice.max():.0f}]')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(ct_raw_slice, cmap='gray')
    axes[0, 1].set_title(f'Original CT\nRange: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}]')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Skull detection
    axes[0, 2].imshow(skull_candidates, cmap='Reds', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Detected Skull Regions\n(200-2000 HU, {np.sum(skull_candidates)} pixels)')
    axes[0, 2].axis('off')
    
    # CT với skull highlight
    ct_with_skull = ct_raw_slice.copy()
    ct_norm = (ct_with_skull - ct_with_skull.min()) / (ct_with_skull.max() - ct_with_skull.min())
    ct_rgb = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)
    ct_rgb[skull_candidates, 0] = 1  # Red for skull
    axes[0, 3].imshow(ct_rgb)
    axes[0, 3].set_title('Original CT + Skull Highlight\n(Red = Skull candidates)')
    axes[0, 3].axis('off')
    
    # Row 2: Mask analysis
    axes[1, 0].imshow(mask_slice, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Comprehensive Mask\nCoverage: {np.sum(mask_slice)/mask_slice.size*100:.1f}%')
    axes[1, 0].axis('off')
    
    # Skull preservation check
    axes[1, 1].imshow(skull_in_mask, cmap='Greens', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Skull in Mask\n({np.sum(skull_in_mask)} pixels preserved)')
    axes[1, 1].axis('off')
    
    # Overlay mask on CT
    ct_mask_overlay = ct_norm.copy()
    ct_mask_rgb = np.stack([ct_mask_overlay, ct_mask_overlay, ct_mask_overlay], axis=-1)
    ct_mask_rgb[mask_slice > 0, 1] = 1  # Green for mask
    ct_mask_rgb[skull_in_mask, 0] = 1   # Red for preserved skull
    axes[1, 2].imshow(ct_mask_rgb)
    axes[1, 2].set_title('CT + Mask Overlay\n(Green=Mask, Red=Skull preserved)')
    axes[1, 2].axis('off')
    
    # Skull preservation percentage
    axes[1, 3].text(0.5, 0.5, f'SKULL PRESERVATION\n\n'
                             f'Total skull pixels: {np.sum(skull_candidates)}\n'
                             f'Preserved skull: {np.sum(skull_in_mask)}\n'
                             f'Preservation rate: {skull_preserved:.1f}%\n\n'
                             f'Mask coverage: {np.sum(mask_slice)/mask_slice.size*100:.1f}%',
                   ha='center', va='center', fontsize=12, transform=axes[1, 3].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    axes[1, 3].axis('off')
    
    # Row 3: Final results
    axes[2, 0].imshow(mri_final, cmap='gray', vmin=0, vmax=1)
    axes[2, 0].set_title(f'Final MRI\nRange: [{mri_final.min():.3f}, {mri_final.max():.3f}]')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(ct_final, cmap='gray', vmin=0, vmax=1)
    axes[2, 1].set_title(f'Final CT (Normalized)\nRange: [{ct_final.min():.3f}, {ct_final.max():.3f}]')
    axes[2, 1].axis('off')
    
    # Skull region in final CT
    skull_region_final = ct_final[skull_candidates] if np.any(skull_candidates) else []
    if len(skull_region_final) > 0:
        axes[2, 2].hist(skull_region_final, bins=50, alpha=0.7, color='red', label='Skull region')
        axes[2, 2].hist(ct_final[mask_slice > 0], bins=50, alpha=0.5, color='blue', label='All tissue')
        axes[2, 2].set_title('Final CT Intensity Distribution')
        axes[2, 2].set_xlabel('Normalized Intensity')
        axes[2, 2].set_ylabel('Pixel Count')
        axes[2, 2].legend()
    else:
        axes[2, 2].text(0.5, 0.5, 'No skull region\ndetected', ha='center', va='center', 
                       transform=axes[2, 2].transAxes)
    
    # Overall assessment
    if skull_preserved >= 80:
        status = "✅ EXCELLENT"
        color = "lightgreen"
    elif skull_preserved >= 60:
        status = "⚠️ GOOD"
        color = "lightyellow"
    elif skull_preserved >= 40:
        status = "⚠️ MODERATE"
        color = "orange"
    else:
        status = "❌ POOR"
        color = "lightcoral"
    
    axes[2, 3].text(0.5, 0.5, f'OVERALL ASSESSMENT\n\n'
                             f'{status}\n\n'
                             f'Skull preservation: {skull_preserved:.1f}%\n'
                             f'Mask coverage: {np.sum(mask_slice)/mask_slice.size*100:.1f}%\n\n'
                             f'Target: >80% skull preservation\n'
                             f'with reasonable mask size',
                   ha='center', va='center', fontsize=12, transform=axes[2, 3].transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('skull_preservation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: skull_preservation_analysis.png")
    
    # Quantitative analysis
    print(f"\n--- SKULL PRESERVATION ANALYSIS ---")
    print(f"Original CT range: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}] HU")
    print(f"Skull candidate pixels: {np.sum(skull_candidates)} (200-2000 HU)")
    print(f"Skull pixels in mask: {np.sum(skull_in_mask)}")
    print(f"Skull preservation rate: {skull_preserved:.1f}%")
    print(f"Mask coverage: {np.sum(mask_slice)/mask_slice.size*100:.1f}%")
    
    # Tissue contrast analysis
    if len(skull_region_final) > 0:
        brain_soft_tissue = ct_final[(mask_slice > 0) & (~skull_candidates)]
        if len(brain_soft_tissue) > 0:
            skull_mean = np.mean(skull_region_final)
            soft_tissue_mean = np.mean(brain_soft_tissue)
            contrast_ratio = skull_mean / soft_tissue_mean if soft_tissue_mean > 0 else 0
            
            print(f"\nTissue contrast in final CT:")
            print(f"Skull region mean intensity: {skull_mean:.3f}")
            print(f"Soft tissue mean intensity: {soft_tissue_mean:.3f}")
            print(f"Skull-to-soft tissue contrast ratio: {contrast_ratio:.2f}")
    
    # Check với other files
    print(f"\n--- Quick check other files ---")
    for i in range(1, min(3, len(dataset))):
        try:
            sample_i = dataset[i]
            mri_i = sample_i['mri']
            ct_i = sample_i['ct']
            print(f"brain_{i+1:03d} - MRI: [{mri_i.min():.3f}, {mri_i.max():.3f}], CT: [{ct_i.min():.3f}, {ct_i.max():.3f}]")
        except Exception as e:
            print(f"Error with brain_{i+1:03d}: {e}")
    
    print(f"\n✓ Skull preservation test completed!")
    
    return {
        'skull_preservation_rate': skull_preserved,
        'mask_coverage': np.sum(mask_slice)/mask_slice.size*100,
        'skull_candidates': np.sum(skull_candidates),
        'skull_preserved': np.sum(skull_in_mask),
        'final_ct_range': [ct_final.min(), ct_final.max()],
        'assessment': status
    }

if __name__ == "__main__":
    results = test_skull_preservation()
    print(f"\n=== FINAL RESULTS ===")
    for key, value in results.items():
        print(f"{key}: {value}") 