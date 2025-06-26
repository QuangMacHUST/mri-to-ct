#!/usr/bin/env python3
"""
Test CycleGAN với COMPLETE preprocessing pipeline từ test.py (bao gồm N4ITK)
"""

import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# Import preprocessing functions
from skimage import filters, morphology, measure
from scipy import ndimage

def save_nifti_image(image_array, save_path, reference_path=None):
    """Lưu ảnh numpy array thành file NIfTI"""
    if reference_path and os.path.exists(reference_path):
        reference_img = nib.load(reference_path)
        new_img = nib.Nifti1Image(image_array, reference_img.affine, reference_img.header)
    else:
        new_img = nib.Nifti1Image(image_array, np.eye(4))
    nib.save(new_img, save_path)

class CompletePreprocessingTester:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.load_model(model_path)
    
    def load_model(self, model_path):
        import sys
        sys.path.insert(0, "src")
        from models import CycleGAN
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        full_config = checkpoint.get("config", {})
        
        model_config = {
            "input_nc": full_config.get("input_nc", 1),
            "output_nc": full_config.get("output_nc", 1),
            "n_residual_blocks": full_config.get("n_residual_blocks", 12 if "new" in model_path else 9),
            "discriminator_layers": full_config.get("discriminator_layers", 3)
        }
        
        self.model = CycleGAN(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"✅ Model loaded: {model_config['n_residual_blocks']} residual blocks")

    def _apply_n4_bias_correction(self, image_sitk):
        """N4 bias field correction để loại bỏ bias field trong MRI"""
        print("⚡ Applying N4 bias field correction...")
        image_sitk = sitk.Cast(image_sitk, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        return corrector.Execute(image_sitk)

    def _create_brain_with_skull_mask(self, mri_array):
        """Tạo comprehensive mask bao gồm brain + skull"""
        print("🧠 Creating comprehensive brain+skull mask...")
        
        # Step 1: Normalize về [0, 1]
        normalized = (mri_array - mri_array.min()) / (mri_array.max() - mri_array.min())
        
        # Step 2: Multi-threshold approach
        otsu_thresh = filters.threshold_otsu(normalized)
        brain_thresh = otsu_thresh * 0.6
        
        # Create initial brain mask
        brain_mask = normalized > brain_thresh
        
        # Step 4: Morphological operations
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=1500)
        brain_mask = ndimage.binary_fill_holes(brain_mask)
        
        # Step 5: Get largest connected component
        labeled_mask = measure.label(brain_mask)
        if labeled_mask.max() > 0:
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component = np.argmax(component_sizes)
            main_region = (labeled_mask == largest_component)
        else:
            main_region = brain_mask
        
        # Step 6: Expand mask để include skull region
        kernel_expand = morphology.ball(3)
        expanded_mask = morphology.binary_dilation(main_region, kernel_expand)
        
        # Step 7: Distance-based refinement
        center_of_mass = ndimage.center_of_mass(main_region)
        coords = np.ogrid[0:expanded_mask.shape[0], 0:expanded_mask.shape[1], 0:expanded_mask.shape[2]]
        distances = np.sqrt(
            (coords[0] - center_of_mass[0])**2 +
            (coords[1] - center_of_mass[1])**2 +
            (coords[2] - center_of_mass[2])**2
        )
        
        max_brain_radius = np.max(distances[main_region]) * 1.3
        distance_mask = distances <= max_brain_radius
        final_mask = expanded_mask & distance_mask
        
        # Step 8: Final cleanup
        kernel_smooth = morphology.ball(2)
        final_mask = morphology.binary_closing(final_mask, kernel_smooth)
        final_mask = ndimage.binary_fill_holes(final_mask)
        
        # Safety check
        total_volume = np.prod(mri_array.shape)
        mask_volume = np.sum(final_mask)
        
        if mask_volume > total_volume * 0.7:
            print("⚠️ Mask too large, using conservative approach")
            kernel_conservative = morphology.ball(1)
            final_mask = morphology.binary_dilation(main_region, kernel_conservative)
        
        print(f"✅ Brain+skull mask: {np.sum(final_mask)/1000:.1f}k voxels ({100*mask_volume/total_volume:.1f}%)")
        return final_mask.astype(np.float32)

    def _apply_mri_mask_to_ct(self, ct_array, mri_mask):
        """Áp dụng MRI mask vào CT để loại bỏ headframe và couch"""
        print("🛡️ Applying MRI mask to remove headframe/couch...")
        
        masked_ct = ct_array.copy()
        
        # Create realistic background value
        background_region = ct_array[mri_mask == 0]
        if len(background_region) > 0:
            background_value = np.percentile(background_region, 10)
            background_value = max(background_value, ct_array.min())
        else:
            background_value = ct_array.min()
        
        # Set vùng ngoài mask thành air-like value
        masked_ct[mri_mask == 0] = background_value
        
        # Metal artifact detection trong brain region
        brain_region = masked_ct[mri_mask > 0]
        if len(brain_region) > 0:
            q95 = np.percentile(brain_region, 95)
            q05 = np.percentile(brain_region, 5)
            q50 = np.percentile(brain_region, 50)
            
            metal_threshold = q95 + 2 * (q95 - q50)
            air_threshold = q05 - 2 * (q50 - q05)
            
            metal_mask = (masked_ct > metal_threshold) & (mri_mask > 0)
            air_mask = (masked_ct < air_threshold) & (mri_mask > 0)
            
            metal_mask = morphology.remove_small_objects(metal_mask, min_size=1000)
            air_mask = morphology.remove_small_objects(air_mask, min_size=1000)
            
            if np.any(metal_mask):
                normal_tissue_value = np.median(brain_region[(brain_region >= q05) & (brain_region <= q95)])
                masked_ct[metal_mask] = normal_tissue_value
                
            if np.any(air_mask):
                csf_value = np.percentile(brain_region, 20)
                masked_ct[air_mask] = csf_value
        
        print("✅ Headframe/couch removed successfully")
        return masked_ct

    def _gentle_outlier_clipping(self, image_array, mask, modality="CT"):
        """Gentle outlier removal"""
        brain_region = image_array[mask > 0]
        if len(brain_region) == 0:
            return image_array
        
        q01 = np.percentile(brain_region, 1)
        q99 = np.percentile(brain_region, 99)
        
        clipped_array = image_array.copy()
        clipped_array[mask > 0] = np.clip(clipped_array[mask > 0], q01, q99)
        
        return clipped_array

    def _normalize_intensity(self, image_array, mask, modality="CT"):
        """Min-Max normalization trong brain region"""
        normalized_array = image_array.copy()
        brain_values = image_array[mask > 0]
        
        if len(brain_values) == 0:
            return normalized_array
        
        min_val = np.min(brain_values)
        max_val = np.max(brain_values)
        
        if max_val > min_val:
            # Normalize về [0, 1] trong brain region
            normalized_array[mask > 0] = (image_array[mask > 0] - min_val) / (max_val - min_val)
        else:
            normalized_array[mask > 0] = 0
        
        # Background = 0
        normalized_array[mask == 0] = 0
        return normalized_array

    def test_volume_complete_pipeline(self, mri_path, ct_path, output_dir):
        """Test với COMPLETE preprocessing pipeline từ test.py"""
        patient_name = os.path.basename(mri_path).replace(".nii.gz", "")
        print(f"\n🏥 Testing {patient_name} with COMPLETE PIPELINE (N4ITK + Full Preprocessing)")
        print("="*80)
        
        # STEP 1: Load với SimpleITK
        print("📁 Loading volumes with SimpleITK...")
        mri_sitk = sitk.ReadImage(mri_path)
        ct_sitk = sitk.ReadImage(ct_path)
        
        # STEP 2: N4 bias field correction cho MRI (CRITICAL STEP!)
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        
        # Convert to numpy
        mri_vol = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        ct_vol = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        print(f"📊 Volume shape: {mri_vol.shape}")
        
        # STEP 3: Create comprehensive brain+skull mask từ MRI
        mri_mask = self._create_brain_with_skull_mask(mri_vol)
        
        # STEP 4: Apply MRI mask to CT để remove headframe  
        ct_vol_masked = self._apply_mri_mask_to_ct(ct_vol, mri_mask)
        
        # STEP 5: Gentle outlier clipping
        print("🧹 Gentle outlier clipping...")
        mri_vol_clipped = self._gentle_outlier_clipping(mri_vol, mri_mask, "MRI")
        ct_vol_clipped = self._gentle_outlier_clipping(ct_vol_masked, mri_mask, "CT")
        
        # STEP 6: Min-Max normalization trong brain region
        print("📏 Min-Max normalization trong brain region...")
        mri_vol_norm = self._normalize_intensity(mri_vol_clipped, mri_mask, "MRI")
        ct_vol_norm = self._normalize_intensity(ct_vol_clipped, mri_mask, "CT")
        
        # STEP 7: Convert to [-1, 1] for model input
        print("🔄 Converting to [-1, 1] for model...")
        # Clip về [0,1] trước khi scale to [-1,1]
        mri_vol_norm = np.clip(mri_vol_norm, 0, 1)
        ct_vol_norm = np.clip(ct_vol_norm, 0, 1)
        
        # Scale to [-1, 1] 
        mri_vol_norm = mri_vol_norm * 2.0 - 1.0
        ct_vol_norm = ct_vol_norm * 2.0 - 1.0
        
        # STEP 8: Process với model
        print(f"🧠 Processing {mri_vol.shape[0]} slices with model...")
        synthetic_vol = np.zeros_like(ct_vol_norm)
        ssim_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(mri_vol.shape[0]), desc="Model inference"):
                mri_slice = mri_vol_norm[i, :, :]
                ct_slice = ct_vol_norm[i, :, :]
                mask_slice = mri_mask[i, :, :]
                
                # Skip if no brain tissue
                if np.sum(mask_slice) < 100:
                    continue
                
                # Resize to 256x256 if needed
                if mri_slice.shape != (256, 256):
                    mri_256 = cv2.resize(mri_slice, (256, 256))
                else:
                    mri_256 = mri_slice
                
                # Model inference
                mri_tensor = torch.FloatTensor(mri_256).unsqueeze(0).unsqueeze(0).to(self.device)
                ct_tensor = self.model.G_MRI2CT(mri_tensor)
                ct_256 = ct_tensor[0, 0].cpu().numpy()
                
                # Resize back to original size
                if ct_256.shape != mri_slice.shape:
                    ct_synth = cv2.resize(ct_256, (mri_slice.shape[1], mri_slice.shape[0]))
                else:
                    ct_synth = ct_256
                
                synthetic_vol[i, :, :] = ct_synth
                
                # SSIM calculation với mask
                try:
                    # Apply mask to both
                    ct_real_masked = ct_slice * mask_slice
                    ct_synth_masked = ct_synth * mask_slice
                    
                    from skimage.metrics import structural_similarity as ssim
                    
                    # Resize for SSIM calculation
                    if mask_slice.shape != (256, 256):
                        ct_real_256 = cv2.resize(ct_real_masked, (256, 256))
                        mask_256 = cv2.resize(mask_slice, (256, 256))
                        ct_synth_256_masked = ct_256 * mask_256
                    else:
                        ct_real_256 = ct_real_masked
                        ct_synth_256_masked = ct_synth_masked
                    
                    ssim_val = ssim(ct_synth_256_masked, ct_real_256, data_range=2.0)
                    
                    if not np.isnan(ssim_val):
                        ssim_scores.append(ssim_val)
                except Exception as e:
                    print(f"⚠️ SSIM error for slice {i}: {e}")
                    continue
        
        # Save results
        patient_dir = os.path.join(output_dir, patient_name)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Save volumes với proper headers
        save_nifti_image(synthetic_vol, os.path.join(patient_dir, f"{patient_name}_synthetic_ct.nii.gz"), mri_path)
        save_nifti_image(mri_mask, os.path.join(patient_dir, f"{patient_name}_brain_mask.nii.gz"), mri_path)
        save_nifti_image(mri_vol_norm, os.path.join(patient_dir, f"{patient_name}_mri_processed.nii.gz"), mri_path)
        save_nifti_image(ct_vol_norm, os.path.join(patient_dir, f"{patient_name}_ct_processed.nii.gz"), mri_path)
        
        # Create comprehensive comparison
        mid = mri_vol.shape[0] // 2
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Row 1: Original data
        axes[0, 0].imshow(sitk.GetArrayFromImage(sitk.ReadImage(mri_path))[mid, :, :], cmap="gray")
        axes[0, 0].set_title("Original MRI")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(sitk.GetArrayFromImage(sitk.ReadImage(ct_path))[mid, :, :], cmap="gray")
        axes[0, 1].set_title("Original CT (with headframe)")
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(mri_vol[mid, :, :], cmap="gray")
        axes[0, 2].set_title("MRI after N4 Correction")
        axes[0, 2].axis("off")
        
        axes[0, 3].imshow(mri_mask[mid, :, :], cmap="gray")
        axes[0, 3].set_title("Brain+Skull Mask")
        axes[0, 3].axis("off")
        
        # Row 2: Processed data
        axes[1, 0].imshow(ct_vol_masked[mid, :, :], cmap="gray")
        axes[1, 0].set_title("CT after Headframe Removal")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow((mri_vol_norm[mid, :, :] + 1) / 2, cmap="gray")
        axes[1, 1].set_title("Normalized MRI [-1,1]")
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow((ct_vol_norm[mid, :, :] + 1) / 2, cmap="gray")
        axes[1, 2].set_title("Normalized CT [-1,1]")
        axes[1, 2].axis("off")
        
        axes[1, 3].imshow((synthetic_vol[mid, :, :] + 1) / 2, cmap="gray")
        axes[1, 3].set_title("Synthetic CT")
        axes[1, 3].axis("off")
        
        # Row 3: Analysis
        # Difference in brain region only
        diff_masked = np.abs(synthetic_vol[mid, :, :] - ct_vol_norm[mid, :, :]) * mri_mask[mid, :, :]
        im = axes[2, 0].imshow(diff_masked, cmap="hot")
        axes[2, 0].set_title("Difference (Brain Only)")
        axes[2, 0].axis("off")
        plt.colorbar(im, ax=axes[2, 0])
        
        # Statistics
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
        
        pipeline_info = f"""COMPLETE PIPELINE:
✓ N4 bias field correction
✓ Brain+skull mask creation  
✓ Headframe/couch removal
✓ Metal artifact removal
✓ Gentle outlier clipping
✓ Min-Max normalization
✓ [-1,1] scaling for model

Results:
• Processed slices: {len(ssim_scores)}
• Average SSIM: {avg_ssim:.4f}
• SSIM std: {np.std(ssim_scores):.4f}
• Best SSIM: {np.max(ssim_scores):.4f}"""
        
        axes[2, 1].text(0.05, 0.95, pipeline_info, fontsize=10, verticalalignment="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
                       transform=axes[2, 1].transAxes)
        axes[2, 1].axis("off")
        
        # SSIM distribution
        if len(ssim_scores) > 1:
            axes[2, 2].hist(ssim_scores, bins=20, alpha=0.7, color="blue", edgecolor="black")
            axes[2, 2].axvline(avg_ssim, color="red", linestyle="--", linewidth=2, label=f"Mean: {avg_ssim:.3f}")
            axes[2, 2].set_title("SSIM Distribution")
            axes[2, 2].set_xlabel("SSIM")
            axes[2, 2].set_ylabel("Frequency")
            axes[2, 2].legend()
            axes[2, 2].grid(True, alpha=0.3)
        else:
            axes[2, 2].text(0.5, 0.5, "Insufficient data\nfor histogram", ha="center", va="center")
            axes[2, 2].axis("off")
        
        axes[2, 3].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(patient_dir, f"{patient_name}_complete_pipeline_comparison.png"), dpi=150)
        plt.close()
        
        print(f"✅ {patient_name} completed with COMPLETE PIPELINE!")
        print(f"📊 Average SSIM: {avg_ssim:.4f} ({len(ssim_scores)} slices)")
        print(f"💾 Results saved to: {patient_dir}")
        
        return {"patient": patient_name, "ssim": avg_ssim, "slices": len(ssim_scores)}

def main():
    parser = argparse.ArgumentParser(description="Test với COMPLETE preprocessing pipeline từ test.py")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mri_dir", type=str, default="data/Test/MRI")
    parser.add_argument("--ct_dir", type=str, default="data/Test/CT")
    parser.add_argument("--output_dir", type=str, default="test_results_complete_pipeline")
    args = parser.parse_args()
    
    tester = CompletePreprocessingTester(args.model_path)
    
    mri_files = sorted([f for f in os.listdir(args.mri_dir) if f.endswith(".nii.gz")])
    ct_files = sorted([f for f in os.listdir(args.ct_dir) if f.endswith(".nii.gz")])
    
    print(f"\n🏥 Testing {len(mri_files)} patients với COMPLETE PREPROCESSING PIPELINE")
    print("⚡ N4ITK + Brain Mask + Headframe Removal + Normalization + Model")
    print("="*80)
    
    results = []
    for mri_file, ct_file in zip(mri_files, ct_files):
        mri_path = os.path.join(args.mri_dir, mri_file)
        ct_path = os.path.join(args.ct_dir, ct_file)
        
        try:
            result = tester.test_volume_complete_pipeline(mri_path, ct_path, args.output_dir)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {mri_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if results:
        overall_ssim = np.mean([r["ssim"] for r in results])
        print(f"\n🎉 FINAL RESULTS với COMPLETE PIPELINE:")
        print("="*50)
        for r in results:
            print(f"  {r['patient']}: SSIM={r['ssim']:.4f} ({r['slices']} slices)")
        print(f"\n🏆 OVERALL AVERAGE SSIM: {overall_ssim:.4f}")
        print(f"📈 Improvement timeline:")
        print(f"  • No preprocessing: 0.5036")
        print(f"  • Basic preprocessing: 0.6552")  
        print(f"  • Complete pipeline (N4ITK): {overall_ssim:.4f}")
        print(f"  • Total improvement: {overall_ssim/0.5036:.2f}x")
        
        # Save detailed results
        with open(os.path.join(args.output_dir, "complete_pipeline_results.txt"), "w") as f:
            f.write("COMPLETE PREPROCESSING PIPELINE RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write("Pipeline Steps:\n")
            f.write("1. N4 bias field correction (SimpleITK)\n")
            f.write("2. Brain+skull mask creation\n")
            f.write("3. Headframe/couch removal\n") 
            f.write("4. Metal artifact removal\n")
            f.write("5. Gentle outlier clipping\n")
            f.write("6. Min-Max normalization\n")
            f.write("7. [-1,1] scaling for model\n\n")
            
            for r in results:
                f.write(f"{r['patient']}: SSIM={r['ssim']:.4f}, Slices={r['slices']}\n")
            f.write(f"\nOverall Average SSIM: {overall_ssim:.4f}\n")
            f.write(f"Improvement vs. no preprocessing: {overall_ssim/0.5036:.2f}x\n")
            f.write(f"Improvement vs. basic preprocessing: {overall_ssim/0.6552:.2f}x\n")
    else:
        print("❌ No valid results obtained")

if __name__ == "__main__":
    main()
