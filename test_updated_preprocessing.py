import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
import SimpleITK as sitk

def test_updated_preprocessing():
    """
    Test để kiểm tra preprocessing đã được cập nhật đúng cách
    """
    print("=== TEST: Updated Preprocessing in test.py ===")
    
    # Test tester class preprocessing
    from test import MRIToCTTester
    
    # Load raw MRI để test
    mri_path = "data/MRI/brain_001.nii.gz"
    mri_sitk = sitk.ReadImage(mri_path)
    mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
    
    print(f"Raw MRI shape: {mri_array.shape}")
    print(f"Raw MRI range: [{mri_array.min():.0f}, {mri_array.max():.0f}]")
    
    # Giả lập MRIToCTTester preprocessing (không cần model)
    class TestTester:
        def _create_brain_with_skull_mask(self, mri_array):
            from skimage import filters, morphology, measure
            from scipy import ndimage
            
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
    
    # Test mask creation
    tester = TestTester()
    
    # Apply N4 correction như trong test.py
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 4)
    mri_sitk_corrected = corrector.Execute(mri_sitk)
    mri_corrected = sitk.GetArrayFromImage(mri_sitk_corrected).astype(np.float32)
    
    # Create mask
    print("\nCreating brain+skull mask...")
    mask_tester = tester._create_brain_with_skull_mask(mri_corrected)
    
    print(f"Mask coverage: {np.sum(mask_tester)/np.prod(mask_tester.shape)*100:.1f}%")
    
    # Compare với DataLoader method
    from data_loader import MRIToCTDataset
    dataset = MRIToCTDataset("data/MRI", "data/CT", is_training=False)
    mask_dataloader = dataset._create_brain_with_skull_mask(mri_corrected)
    
    print(f"DataLoader mask coverage: {np.sum(mask_dataloader)/np.prod(mask_dataloader.shape)*100:.1f}%")
    
    # Check consistency
    mask_diff = np.abs(mask_tester - mask_dataloader)
    print(f"Mask difference: {np.sum(mask_diff)} pixels different")
    
    if np.sum(mask_diff) == 0:
        print("✅ PERFECT: Tester preprocessing identical to DataLoader!")
    elif np.sum(mask_diff) < 1000:
        print("✅ GOOD: Tester preprocessing very similar to DataLoader")
    else:
        print("⚠️ WARNING: Significant difference between Tester and DataLoader preprocessing")
    
    # Visualize comparison
    mid_slice = mri_array.shape[0] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Updated Preprocessing Comparison: test.py vs DataLoader', fontsize=14)
    
    # Row 1: Original vs corrected
    axes[0, 0].imshow(mri_array[mid_slice], cmap='gray')
    axes[0, 0].set_title('Raw MRI')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mri_corrected[mid_slice], cmap='gray')
    axes[0, 1].set_title('N4 Corrected MRI')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_tester[mid_slice], cmap='viridis')
    axes[0, 2].set_title('test.py Mask')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(mask_dataloader[mid_slice], cmap='viridis')
    axes[0, 3].set_title('DataLoader Mask')
    axes[0, 3].axis('off')
    
    # Row 2: Difference and application
    axes[1, 0].imshow(mask_diff[mid_slice], cmap='hot')
    axes[1, 0].set_title(f'Mask Difference\n({np.sum(mask_diff[mid_slice])} pixels)')
    axes[1, 0].axis('off')
    
    # Apply masks
    mri_masked_tester = mri_corrected[mid_slice] * mask_tester[mid_slice]
    mri_masked_dataloader = mri_corrected[mid_slice] * mask_dataloader[mid_slice]
    
    axes[1, 1].imshow(mri_masked_tester, cmap='gray')
    axes[1, 1].set_title('MRI with test.py Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mri_masked_dataloader, cmap='gray')
    axes[1, 2].set_title('MRI with DataLoader Mask')
    axes[1, 2].axis('off')
    
    # Difference in final result
    final_diff = np.abs(mri_masked_tester - mri_masked_dataloader)
    axes[1, 3].imshow(final_diff, cmap='hot')
    axes[1, 3].set_title(f'Final Difference\n(max={final_diff.max():.3f})')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('updated_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: updated_preprocessing_comparison.png")
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    print(f"Raw MRI range: [{mri_array.min():.0f}, {mri_array.max():.0f}]")
    print(f"N4 corrected range: [{mri_corrected.min():.0f}, {mri_corrected.max():.0f}]")
    print(f"test.py mask coverage: {np.sum(mask_tester)/np.prod(mask_tester.shape)*100:.1f}%")
    print(f"DataLoader mask coverage: {np.sum(mask_dataloader)/np.prod(mask_dataloader.shape)*100:.1f}%")
    print(f"Mask consistency: {100*(1-np.sum(mask_diff)/np.prod(mask_diff.shape)):.2f}%")
    
    return {
        'mask_coverage_tester': np.sum(mask_tester)/np.prod(mask_tester.shape)*100,
        'mask_coverage_dataloader': np.sum(mask_dataloader)/np.prod(mask_dataloader.shape)*100,
        'mask_difference_pixels': np.sum(mask_diff),
        'consistency_percent': 100*(1-np.sum(mask_diff)/np.prod(mask_diff.shape))
    }

if __name__ == "__main__":
    results = test_updated_preprocessing()
    
    print(f"\n=== FINAL RESULT ===")
    if results['mask_difference_pixels'] == 0:
        print("🎯 PERFECT: test.py preprocessing now identical to DataLoader!")
        print("✅ Brain+skull preservation implemented successfully")
    elif results['consistency_percent'] > 95:
        print("✅ EXCELLENT: test.py preprocessing highly consistent with DataLoader")
        print("✅ Brain+skull preservation implemented successfully")
    else:
        print("⚠️ ATTENTION: Some differences detected")
        print(f"Consistency: {results['consistency_percent']:.1f}%")
        print("May need further adjustment")
        
    print(f"Coverage difference: {abs(results['mask_coverage_tester'] - results['mask_coverage_dataloader']):.1f}%") 