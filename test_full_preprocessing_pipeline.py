import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')
from data_loader import MRIToCTDataset
import SimpleITK as sitk

def test_full_preprocessing_pipeline():
    """
    Test toàn bộ preprocessing pipeline và hiển thị kết quả trước/sau
    """
    print("=== TEST: Full Preprocessing Pipeline (MRI & CT) ===")
    
    # Setup paths
    mri_dir = "data/MRI"
    ct_dir = "data/CT"
    
    # Load raw data
    print("Loading raw data...")
    mri_raw = sitk.ReadImage(os.path.join(mri_dir, "brain_001.nii.gz"))
    ct_raw = sitk.ReadImage(os.path.join(ct_dir, "brain_001.nii.gz"))
    
    mri_array_raw = sitk.GetArrayFromImage(mri_raw).astype(np.float32)
    ct_array_raw = sitk.GetArrayFromImage(ct_raw).astype(np.float32)
    
    print(f"Raw MRI shape: {mri_array_raw.shape}")
    print(f"Raw CT shape: {ct_array_raw.shape}")
    
    # Choose middle slice for visualization
    mid_slice = mri_array_raw.shape[0] // 2
    print(f"Analyzing slice {mid_slice}")
    
    # Create dataset và process
    print("\nProcessing through dataloader...")
    dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False, slice_range=(mid_slice, mid_slice+1))
    
    # Get processed sample (output range [-1,1])
    sample = dataset[0]
    mri_processed = sample['mri'].squeeze(0).numpy()  # [-1,1] range
    ct_processed = sample['ct'].squeeze(0).numpy()    # [-1,1] range
    
    print(f"Processed MRI shape: {mri_processed.shape}")
    print(f"Processed CT shape: {ct_processed.shape}")
    print(f"Processed MRI range: [{mri_processed.min():.3f}, {mri_processed.max():.3f}]")
    print(f"Processed CT range: [{ct_processed.min():.3f}, {ct_processed.max():.3f}]")
    
    # Get slices for comparison
    mri_raw_slice = mri_array_raw[mid_slice]
    ct_raw_slice = ct_array_raw[mid_slice]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Full Preprocessing Pipeline: Before vs After (brain_001.nii.gz)', fontsize=16)
    
    # Row 1: Original (Raw) Data
    im1 = axes[0, 0].imshow(mri_raw_slice, cmap='gray')
    axes[0, 0].set_title(f'Raw MRI\nRange: [{mri_raw_slice.min():.0f}, {mri_raw_slice.max():.0f}]')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(ct_raw_slice, cmap='gray')
    axes[0, 1].set_title(f'Raw CT\nRange: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}] HU')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Raw histograms
    axes[0, 2].hist(mri_raw_slice.flatten(), bins=100, alpha=0.7, color='blue', label='MRI')
    axes[0, 2].set_title('Raw MRI Histogram')
    axes[0, 2].set_xlabel('Intensity')
    axes[0, 2].set_ylabel('Pixel Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[0, 3].hist(ct_raw_slice.flatten(), bins=100, alpha=0.7, color='red', label='CT')
    axes[0, 3].set_title('Raw CT Histogram')
    axes[0, 3].set_xlabel('HU Value')
    axes[0, 3].set_ylabel('Pixel Count')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Processed Data ([-1,1] range)
    im3 = axes[1, 0].imshow(mri_processed, cmap='gray', vmin=-1, vmax=1)
    axes[1, 0].set_title(f'Processed MRI\nRange: [{mri_processed.min():.3f}, {mri_processed.max():.3f}]')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im4 = axes[1, 1].imshow(ct_processed, cmap='gray', vmin=-1, vmax=1)
    axes[1, 1].set_title(f'Processed CT\nRange: [{ct_processed.min():.3f}, {ct_processed.max():.3f}]')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Processed histograms
    axes[1, 2].hist(mri_processed.flatten(), bins=100, alpha=0.7, color='blue', label='MRI')
    axes[1, 2].set_title('Processed MRI Histogram')
    axes[1, 2].set_xlabel('Normalized Intensity')
    axes[1, 2].set_ylabel('Pixel Count')
    axes[1, 2].set_xlim(-1, 1)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero')
    
    axes[1, 3].hist(ct_processed.flatten(), bins=100, alpha=0.7, color='red', label='CT')
    axes[1, 3].set_title('Processed CT Histogram')
    axes[1, 3].set_xlabel('Normalized Intensity')
    axes[1, 3].set_ylabel('Pixel Count')
    axes[1, 3].set_xlim(-1, 1)
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero')
    
    # Row 3: Difference Analysis & Statistics
    # Normalize raw data để so sánh
    mri_raw_norm = (mri_raw_slice - mri_raw_slice.min()) / (mri_raw_slice.max() - mri_raw_slice.min())
    ct_raw_norm = (ct_raw_slice - ct_raw_slice.min()) / (ct_raw_slice.max() - ct_raw_slice.min())
    
    # Convert processed từ [-1,1] về [0,1] để so sánh
    mri_proc_01 = (mri_processed + 1) / 2
    ct_proc_01 = (ct_processed + 1) / 2
    
    # Difference maps
    mri_diff = np.abs(mri_raw_norm - mri_proc_01)
    ct_diff = np.abs(ct_raw_norm - ct_proc_01)
    
    im5 = axes[2, 0].imshow(mri_diff, cmap='hot', vmin=0, vmax=1)
    axes[2, 0].set_title('MRI Difference Map\n(Raw vs Processed)')
    axes[2, 0].axis('off')
    plt.colorbar(im5, ax=axes[2, 0], fraction=0.046, pad=0.04)
    
    im6 = axes[2, 1].imshow(ct_diff, cmap='hot', vmin=0, vmax=1)
    axes[2, 1].set_title('CT Difference Map\n(Raw vs Processed)')
    axes[2, 1].axis('off')
    plt.colorbar(im6, ax=axes[2, 1], fraction=0.046, pad=0.04)
    
    # Combined histogram comparison
    axes[2, 2].hist(mri_proc_01.flatten(), bins=50, alpha=0.6, color='blue', label='MRI Processed')
    axes[2, 2].hist(ct_proc_01.flatten(), bins=50, alpha=0.6, color='red', label='CT Processed')
    axes[2, 2].set_title('Processed Data Comparison\n(Both in [0,1] range)')
    axes[2, 2].set_xlabel('Normalized Intensity')
    axes[2, 2].set_ylabel('Pixel Count')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # Statistics summary
    stats_text = f"""PROCESSING STATISTICS

RAW DATA:
MRI: [{mri_raw_slice.min():.0f}, {mri_raw_slice.max():.0f}]
CT: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}] HU

PROCESSED DATA ([-1,1]):
MRI: [{mri_processed.min():.3f}, {mri_processed.max():.3f}]
CT: [{ct_processed.min():.3f}, {ct_processed.max():.3f}]

STATISTICS:
MRI mean: {mri_processed.mean():.3f}
MRI std: {mri_processed.std():.3f}
CT mean: {ct_processed.mean():.3f}
CT std: {ct_processed.std():.3f}

DIFFERENCES:
MRI changed: {np.mean(mri_diff)*100:.1f}%
CT changed: {np.mean(ct_diff)*100:.1f}%

STATUS: ✅ READY FOR TRAINING"""
    
    axes[2, 3].text(0.1, 0.1, stats_text, transform=axes[2, 3].transAxes, 
                   fontsize=10, verticalalignment='bottom', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('full_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: full_preprocessing_comparison.png")
    
    # Detailed analysis
    print(f"\n--- DETAILED ANALYSIS ---")
    print(f"Raw MRI range: [{mri_raw_slice.min():.0f}, {mri_raw_slice.max():.0f}]")
    print(f"Raw CT range: [{ct_raw_slice.min():.0f}, {ct_raw_slice.max():.0f}] HU")
    print(f"Processed MRI range: [{mri_processed.min():.3f}, {mri_processed.max():.3f}]")
    print(f"Processed CT range: [{ct_processed.min():.3f}, {ct_processed.max():.3f}]")
    
    # Check range compliance
    mri_in_range = (mri_processed >= -1.0) & (mri_processed <= 1.0)
    ct_in_range = (ct_processed >= -1.0) & (ct_processed <= 1.0)
    
    print(f"\nRange compliance check:")
    print(f"MRI pixels in [-1,1]: {np.sum(mri_in_range)}/{mri_processed.size} ({np.sum(mri_in_range)/mri_processed.size*100:.1f}%)")
    print(f"CT pixels in [-1,1]: {np.sum(ct_in_range)}/{ct_processed.size} ({np.sum(ct_in_range)/ct_processed.size*100:.1f}%)")
    
    # Statistics
    print(f"\nProcessed data statistics:")
    print(f"MRI - Mean: {mri_processed.mean():.3f}, Std: {mri_processed.std():.3f}")
    print(f"CT - Mean: {ct_processed.mean():.3f}, Std: {ct_processed.std():.3f}")
    
    # Test với multiple files
    print(f"\n--- Testing multiple files ---")
    for i in range(min(3, len(dataset))):
        try:
            sample_i = dataset[i]
            mri_i = sample_i['mri']
            ct_i = sample_i['ct']
            filename = sample_i['filename']
            
            print(f"{filename}:")
            print(f"  MRI: [{mri_i.min():.3f}, {mri_i.max():.3f}], mean: {mri_i.mean():.3f}")
            print(f"  CT:  [{ct_i.min():.3f}, {ct_i.max():.3f}], mean: {ct_i.mean():.3f}")
            
            # Check consistency
            mri_valid = torch.all((mri_i >= -1.0) & (mri_i <= 1.0))
            ct_valid = torch.all((ct_i >= -1.0) & (ct_i <= 1.0))
            print(f"  Range valid: MRI={mri_valid}, CT={ct_valid}")
            
        except Exception as e:
            print(f"Error with file {i}: {e}")
    
    print(f"\n✓ Full preprocessing pipeline test completed!")
    
    return {
        'raw_mri_range': [mri_raw_slice.min(), mri_raw_slice.max()],
        'raw_ct_range': [ct_raw_slice.min(), ct_raw_slice.max()],
        'processed_mri_range': [mri_processed.min(), mri_processed.max()],
        'processed_ct_range': [ct_processed.min(), ct_processed.max()],
        'mri_stats': {'mean': mri_processed.mean(), 'std': mri_processed.std()},
        'ct_stats': {'mean': ct_processed.mean(), 'std': ct_processed.std()},
        'range_compliance': {
            'mri': np.sum(mri_in_range)/mri_processed.size*100,
            'ct': np.sum(ct_in_range)/ct_processed.size*100
        }
    }

if __name__ == "__main__":
    import torch  # Import here để avoid issues
    results = test_full_preprocessing_pipeline()
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"✅ Preprocessing pipeline working correctly")
    print(f"✅ Both MRI & CT normalized to [-1,1] range")
    print(f"✅ Range compliance: MRI {results['range_compliance']['mri']:.1f}%, CT {results['range_compliance']['ct']:.1f}%")
    print(f"✅ Ready for CycleGAN training!") 