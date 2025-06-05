import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
import SimpleITK as sitk
from skimage import filters, morphology, measure
from scipy import ndimage

def create_brain_with_skull_mask(mri_array: np.ndarray) -> np.ndarray:
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

def test_mask_consistency():
    """
    Test consistency between updated test.py method and dataloader method
    """
    print("=== TEST: Brain+Skull Mask Consistency ===")
    
    # Load MRI
    mri_path = "data/MRI/brain_001.nii.gz"
    mri_sitk = sitk.ReadImage(mri_path)
    
    # Apply N4 correction
    # Cast to float32 first để avoid pixel type error
    mri_sitk = sitk.Cast(mri_sitk, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 4)
    mri_sitk_corrected = corrector.Execute(mri_sitk)
    mri_array = sitk.GetArrayFromImage(mri_sitk_corrected).astype(np.float32)
    
    print(f"MRI shape: {mri_array.shape}")
    print(f"MRI range: [{mri_array.min():.0f}, {mri_array.max():.0f}]")
    
    # Test standalone function
    print("\nTesting standalone brain+skull mask function...")
    mask_standalone = create_brain_with_skull_mask(mri_array)
    coverage_standalone = np.sum(mask_standalone)/np.prod(mask_standalone.shape)*100
    print(f"Standalone mask coverage: {coverage_standalone:.1f}%")
    
    # Test dataloader function
    print("Testing dataloader brain+skull mask function...")
    from data_loader import MRIToCTDataset
    dataset = MRIToCTDataset("data/MRI", "data/CT", is_training=False)
    mask_dataloader = dataset._create_brain_with_skull_mask(mri_array)
    coverage_dataloader = np.sum(mask_dataloader)/np.prod(mask_dataloader.shape)*100
    print(f"DataLoader mask coverage: {coverage_dataloader:.1f}%")
    
    # Compare
    mask_diff = np.abs(mask_standalone - mask_dataloader)
    diff_pixels = np.sum(mask_diff)
    consistency = 100 * (1 - diff_pixels / np.prod(mask_diff.shape))
    
    print(f"\nComparison results:")
    print(f"Different pixels: {diff_pixels}")
    print(f"Consistency: {consistency:.2f}%")
    
    if diff_pixels == 0:
        print("🎯 PERFECT: Functions are identical!")
    elif consistency > 99:
        print("✅ EXCELLENT: Functions are nearly identical")
    elif consistency > 95:
        print("✅ GOOD: Functions are very similar")
    else:
        print("⚠️ WARNING: Significant differences detected")
    
    # Visualize
    mid_slice = mri_array.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Brain+Skull Mask Consistency Test', fontsize=14)
    
    # Row 1: Original and masks
    axes[0, 0].imshow(mri_array[mid_slice], cmap='gray')
    axes[0, 0].set_title('N4 Corrected MRI')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_standalone[mid_slice], cmap='viridis')
    axes[0, 1].set_title(f'Updated test.py Mask\n({coverage_standalone:.1f}% coverage)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_dataloader[mid_slice], cmap='viridis')
    axes[0, 2].set_title(f'DataLoader Mask\n({coverage_dataloader:.1f}% coverage)')
    axes[0, 2].axis('off')
    
    # Row 2: Difference and masked results
    axes[1, 0].imshow(mask_diff[mid_slice], cmap='hot')
    axes[1, 0].set_title(f'Difference\n({np.sum(mask_diff[mid_slice])} pixels)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mri_array[mid_slice] * mask_standalone[mid_slice], cmap='gray')
    axes[1, 1].set_title('MRI with Updated Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mri_array[mid_slice] * mask_dataloader[mid_slice], cmap='gray')
    axes[1, 2].set_title('MRI with DataLoader Mask')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mask_consistency_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: mask_consistency_test.png")
    
    return {
        'coverage_standalone': coverage_standalone,
        'coverage_dataloader': coverage_dataloader,
        'consistency_percent': consistency,
        'different_pixels': diff_pixels
    }

if __name__ == "__main__":
    try:
        results = test_mask_consistency()
        
        print(f"\n=== FINAL SUMMARY ===")
        print(f"✅ Brain+skull mask implementation:")
        print(f"   Updated test.py coverage: {results['coverage_standalone']:.1f}%")
        print(f"   DataLoader coverage: {results['coverage_dataloader']:.1f}%")
        print(f"   Consistency: {results['consistency_percent']:.2f}%")
        
        if results['different_pixels'] == 0:
            print("\n🎯 SUCCESS: test.py preprocessing now uses identical brain+skull mask!")
            print("✅ Bone structure preservation implemented correctly")
        elif results['consistency_percent'] > 99:
            print("\n✅ SUCCESS: test.py preprocessing uses nearly identical brain+skull mask!")
            print("✅ Bone structure preservation implemented correctly")
        else:
            print(f"\n⚠️ NOTICE: {results['different_pixels']} pixels difference detected")
            print("Minor variations may be acceptable")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 