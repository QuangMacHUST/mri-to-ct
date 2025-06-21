import os
import torch
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List
import cv2

from models import CycleGAN
from data_loader import create_test_loader, MRIToCTDataset
from metrics import evaluate_model, print_metrics, MetricsCalculator
from utils import save_nifti_image, compare_images, print_model_summary

class MRIToCTTester:
    """
    Class để test và đánh giá mô hình CycleGAN
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: đường dẫn tới model đã train
            device: thiết bị sử dụng (cuda/cpu)
        """
        self.device = device
        self.model = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """
        Load model từ checkpoint
        """
        print(f"Đang load model từ {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Khôi phục config từ checkpoint
        config = checkpoint.get('config', {
            'input_nc': 1,
            'output_nc': 1,
            'n_residual_blocks': 9,
            'discriminator_layers': 3
        })
        
        # Khởi tạo model
        self.model = CycleGAN(
            input_nc=config['input_nc'],
            output_nc=config['output_nc'],
            n_residual_blocks=config['n_residual_blocks'],
            discriminator_layers=config['discriminator_layers']
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully từ epoch {checkpoint.get('epoch', 'unknown')}")
        print_model_summary(self.model)
    
    def _create_brain_with_skull_mask(self, mri_array: np.ndarray) -> np.ndarray:
        """
        Tạo comprehensive mask bao gồm brain tissue + skull để preserve bone structures
        """
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
    
    def _apply_mri_mask_to_ct(self, ct_array: np.ndarray, mri_mask: np.ndarray) -> np.ndarray:
        """
        Áp dụng MRI mask vào CT để loại bỏ headframe và couch
        """
        from skimage import morphology
        
        # Step 1: Apply MRI mask để loại bỏ couch/headframe
        masked_ct = ct_array.copy()
        
        # Tạo realistic background value (air-like)
        background_region = ct_array[mri_mask == 0]
        if len(background_region) > 0:
            # Air value trong CT thường là -1000 HU, dùng percentile thấp
            background_value = np.percentile(background_region, 10)  # More realistic air value
            # Ensure không quá extreme
            background_value = max(background_value, ct_array.min())
        else:
            background_value = ct_array.min()
        
        # Set vùng ngoài mask thành air-like value
        masked_ct[mri_mask == 0] = background_value
        
        # Step 2: Improved metal artifact detection trong brain region
        brain_region = masked_ct[mri_mask > 0]
        if len(brain_region) > 0:
            # Use more robust statistics
            q95 = np.percentile(brain_region, 95)
            q05 = np.percentile(brain_region, 5)
            q50 = np.percentile(brain_region, 50)  # Median
            
            # More conservative thresholds để preserve normal tissue
            metal_threshold = q95 + 2 * (q95 - q50)  # Detect extreme bright artifacts
            air_threshold = q05 - 2 * (q50 - q05)    # Detect extreme dark artifacts
            
            # Create masks với conservative approach
            metal_mask = (masked_ct > metal_threshold) & (mri_mask > 0)
            air_mask = (masked_ct < air_threshold) & (mri_mask > 0)
            
            # Stricter size requirements để avoid removing normal tissue
            metal_mask = morphology.remove_small_objects(metal_mask, min_size=1000)  # Larger size
            air_mask = morphology.remove_small_objects(air_mask, min_size=1000)
            
            # Replace artifacts với tissue-appropriate values
            if np.any(metal_mask):
                # Metal artifacts -> median của normal brain tissue
                normal_tissue_value = np.median(brain_region[(brain_region >= q05) & (brain_region <= q95)])
                masked_ct[metal_mask] = normal_tissue_value
                
            if np.any(air_mask):
                # Air artifacts -> CSF-like value (slightly above q05)
                csf_value = np.percentile(brain_region, 20)  # Typical CSF range
                masked_ct[air_mask] = csf_value
        
        return masked_ct
    
    def _gentle_outlier_clipping(self, image_array: np.ndarray, mask: np.ndarray, modality: str = 'CT') -> np.ndarray:
        """
        Gentle outlier removal chỉ loại bỏ extreme outliers, preserve normal tissue variation
        """
        # Chỉ xử lý vùng trong mask
        brain_region = image_array[mask > 0]
        
        if len(brain_region) == 0:
            return image_array
        
        # Conservative percentile thresholds
        q01 = np.percentile(brain_region, 1)    # Very low threshold
        q99 = np.percentile(brain_region, 99)   # Very high threshold
        
        # Chỉ clip extreme outliers
        clipped_array = image_array.copy()
        
        # Áp dụng chỉ trong vùng mask
        clipped_array[mask > 0] = np.clip(clipped_array[mask > 0], q01, q99)
        
        return clipped_array
    
    def _normalize_intensity(self, image_array: np.ndarray, mask: np.ndarray, modality: str = 'CT') -> np.ndarray:
        """
        Min-Max normalization trong brain region
        """
        normalized_array = image_array.copy()
        
        # Lấy values trong brain region
        brain_values = image_array[mask > 0]
        
        if len(brain_values) == 0:
            return normalized_array
        
        # Min-Max normalization
        min_val = np.min(brain_values)
        max_val = np.max(brain_values)
        
        if max_val > min_val:
            # Normalize chỉ vùng brain
            normalized_array[mask > 0] = (image_array[mask > 0] - min_val) / (max_val - min_val)
        else:
            # Nếu min == max, set về 0
            normalized_array[mask > 0] = 0
        
        # Vùng ngoài mask giữ nguyên (background)
        normalized_array[mask == 0] = 0
        
        return normalized_array
    
    def _crop_brain_roi(self, image_array: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Crop về ROI chứa não, loại bỏ vùng ngoài không cần thiết
        """
        # Find bounding box của vùng não
        brain_coords = np.where(mask > 0)
        
        if len(brain_coords[0]) == 0:
            # Nếu không tìm thấy brain mask, return original
            return image_array, mask
        
        # Lấy bounding box với padding
        min_z, max_z = brain_coords[0].min(), brain_coords[0].max()
        min_y, max_y = brain_coords[1].min(), brain_coords[1].max()  
        min_x, max_x = brain_coords[2].min(), brain_coords[2].max()
        
        # Add padding để không crop quá sát
        padding = 10
        min_z = max(0, min_z - padding)
        max_z = min(image_array.shape[0], max_z + padding)
        min_y = max(0, min_y - padding)
        max_y = min(image_array.shape[1], max_y + padding)
        min_x = max(0, min_x - padding)
        max_x = min(image_array.shape[2], max_x + padding)
        
        # Crop image và mask
        cropped_image = image_array[min_z:max_z, min_y:max_y, min_x:max_x]
        cropped_mask = mask[min_z:max_z, min_y:max_y, min_x:max_x]
        
        return cropped_image, cropped_mask
    
    def _apply_n4_bias_correction(self, image: sitk.Image) -> sitk.Image:
        """
        Áp dụng N4 bias field correction để loại bỏ bias field trong MRI
        """
        # Cast về float32 để tránh lỗi với 16-bit signed integer
        image = sitk.Cast(image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        return corrector.Execute(image)
    
    def test_single_image(self, mri_path: str, output_dir: str, save_comparison: bool = True) -> Dict[str, float]:
        """
        Test trên một ảnh MRI đơn lẻ với comprehensive preprocessing pipeline
        để loại bỏ headframe và couch
        
        Args:
            mri_path: đường dẫn tới file MRI
            output_dir: thư mục lưu kết quả
            save_comparison: có lưu ảnh so sánh không
            
        Returns:
            Dict chứa metrics nếu có CT ground truth
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Đang xử lý {mri_path}...")
        
        # BƯỚC 1: Load ảnh MRI
        mri_sitk = sitk.ReadImage(mri_path)
        
        # BƯỚC 2: Áp dụng N4 bias correction
        print("Áp dụng N4 bias correction...")
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        
        print(f"MRI shape: {mri_array.shape}")
        
        # BƯỚC 3: Tạo comprehensive brain+skull mask để loại bỏ headframe và couch
        print("Tạo brain+skull mask để loại bỏ headframe và couch...")
        binary_mask = self._create_brain_with_skull_mask(mri_array)
        print(f"Brain mask tạo thành công với {np.sum(binary_mask)} voxels")
        
        # BƯỚC 4: Gentle outlier clipping để loại bỏ extreme artifacts
        print("Áp dụng gentle outlier clipping...")
        mri_array = self._gentle_outlier_clipping(mri_array, binary_mask, 'MRI')
        
        # BƯỚC 5: Normalize intensity trong brain region với Min-Max
        print("Normalize intensity với Min-Max trong brain region...")
        mri_array = self._normalize_intensity(mri_array, binary_mask, 'MRI')
        
        # BƯỚC 6: Crop brain ROI để tập trung vào vùng não
        print("Crop brain ROI...")
        original_shape = mri_array.shape
        mri_array, binary_mask_cropped = self._crop_brain_roi(mri_array, binary_mask)
        
        print(f"Sau khi crop: {original_shape} -> {mri_array.shape}")
        
        # BƯỚC 7: Chuyển về [-1, 1] để phù hợp với Tanh activation của Generator
        # Clip để đảm bảo trong [0,1] trước khi scale về [-1,1]
        mri_array = np.clip(mri_array, 0, 1)
        mri_array = mri_array * 2.0 - 1.0
        
        # BƯỚC 8: Tạo CT mô phỏng cho từng slice
        print("Generating synthetic CT...")
        fake_ct_volume = np.zeros_like(mri_array)
        
        with torch.no_grad():
            for slice_idx in tqdm(range(mri_array.shape[0]), desc="Tạo CT slice"):
                mri_slice = mri_array[slice_idx]
                
                # Resize về 256x256 nếu cần
                if mri_slice.shape != (256, 256):
                    mri_slice_resized = cv2.resize(mri_slice, (256, 256), interpolation=cv2.INTER_LINEAR)
                else:
                    mri_slice_resized = mri_slice
                
                # Chuyển về tensor
                mri_tensor = torch.tensor(mri_slice_resized, dtype=torch.float32)
                mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
                
                # Generate CT
                fake_ct_tensor = self.model.G_MRI2CT(mri_tensor)
                fake_ct_slice = fake_ct_tensor[0, 0].cpu().numpy()
                
                # Resize về kích thước gốc nếu cần
                if fake_ct_slice.shape != mri_slice.shape:
                    fake_ct_slice = cv2.resize(fake_ct_slice, mri_slice.shape[::-1], interpolation=cv2.INTER_LINEAR)
                
                fake_ct_volume[slice_idx] = fake_ct_slice
        
        # BƯỚC 9: Post-processing - đưa về kích thước gốc nếu đã crop
        if fake_ct_volume.shape != original_shape:
            print("Expanding kết quả về kích thước gốc...")
            # Tạo volume gốc với background value
            expanded_ct = np.full(original_shape, -1.0, dtype=np.float32)  # Background = -1 (air-like)
            expanded_mri = np.full(original_shape, -1.0, dtype=np.float32)
            
            # Tìm vị trí để đặt cropped volume vào center
            # Tính toán offset để center volume
            pad_z = (original_shape[0] - fake_ct_volume.shape[0]) // 2
            pad_y = (original_shape[1] - fake_ct_volume.shape[1]) // 2  
            pad_x = (original_shape[2] - fake_ct_volume.shape[2]) // 2
            
            # Đảm bảo không vượt quá boundary
            end_z = min(pad_z + fake_ct_volume.shape[0], original_shape[0])
            end_y = min(pad_y + fake_ct_volume.shape[1], original_shape[1])
            end_x = min(pad_x + fake_ct_volume.shape[2], original_shape[2])
            
            # Place cropped volume vào vị trí center
            expanded_ct[pad_z:end_z, pad_y:end_y, pad_x:end_x] = fake_ct_volume
            expanded_mri[pad_z:end_z, pad_y:end_y, pad_x:end_x] = mri_array
            
            fake_ct_volume = expanded_ct
            mri_array = expanded_mri
        
        # BƯỚC 10: Lưu kết quả
        filename = os.path.basename(mri_path).replace('.nii.gz', '_synthetic_ct.nii.gz')
        output_path = os.path.join(output_dir, filename)
        save_nifti_image(fake_ct_volume, output_path, mri_path)
        
        # Lưu ảnh so sánh cho slice giữa
        if save_comparison:
            middle_slice = fake_ct_volume.shape[0] // 2
            comparison_path = os.path.join(output_dir, filename.replace('.nii.gz', '_comparison.png'))
            
            # Convert về [0,1] để display
            mri_display = np.clip((mri_array[middle_slice] + 1.0) / 2.0, 0, 1)
            ct_display = np.clip((fake_ct_volume[middle_slice] + 1.0) / 2.0, 0, 1)
            
            compare_images(
                mri_display, 
                ct_display,
                comparison_path,
                "MRI Input (Preprocessed)",
                "Synthetic CT"
            )
        
        print(f"CT mô phỏng đã được lưu tại: {output_path}")
        print("Preprocessing pipeline hoàn thành:")
        print("  ✓ N4 bias correction")
        print("  ✓ Brain+skull mask tạo để loại bỏ headframe/couch")
        print("  ✓ Gentle outlier clipping")
        print("  ✓ Min-Max normalization trong brain region")
        print("  ✓ Brain ROI cropping")
        print("  ✓ Synthetic CT generation")
        
        return {}
        
    def test_with_ground_truth(self, mri_path: str, ct_path: str, output_dir: str) -> Dict[str, float]:
        """
        Test với ground truth CT và tính toán metrics
        Áp dụng MRI mask vào CT để loại bỏ headframe và couch
        """
        print(f"Testing với ground truth: MRI={os.path.basename(mri_path)}, CT={os.path.basename(ct_path)}")
        
        # Tạo output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # BƯỚC 1: Load ảnh MRI và CT
        mri_sitk = sitk.ReadImage(mri_path)
        ct_sitk = sitk.ReadImage(ct_path)
        
        # BƯỚC 2: N4 bias correction cho MRI
        mri_sitk = self._apply_n4_bias_correction(mri_sitk)
        
        # Chuyển về numpy
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        print(f"MRI shape: {mri_array.shape}, CT shape: {ct_array.shape}")
        
        # BƯỚC 3: Tạo comprehensive mask từ MRI (brain + skull, không có couch/headframe)
        mri_mask = self._create_brain_with_skull_mask(mri_array)
        print(f"Tạo MRI mask thành công với {np.sum(mri_mask)} voxels")
        
        # BƯỚC 4: Áp dụng MRI mask vào CT để loại bỏ couch/headframe
        print("Áp dụng MRI mask vào CT để loại bỏ headframe và couch...")
        ct_array = self._apply_mri_mask_to_ct(ct_array, mri_mask)
        
        # BƯỚC 5: Gentle outlier clipping
        mri_array = self._gentle_outlier_clipping(mri_array, mri_mask, 'MRI')
        ct_array = self._gentle_outlier_clipping(ct_array, mri_mask, 'CT')
        
        # BƯỚC 6: Normalize intensity trong brain region
        mri_array = self._normalize_intensity(mri_array, mri_mask, 'MRI')
        ct_array = self._normalize_intensity(ct_array, mri_mask, 'CT')  # Sử dụng cùng mask
        
        # BƯỚC 7: Crop brain ROI
        mri_array, mri_mask_cropped = self._crop_brain_roi(mri_array, mri_mask)
        ct_array, _ = self._crop_brain_roi(ct_array, mri_mask)  # Sử dụng cùng original mask
        
        print(f"Sau khi crop: MRI shape: {mri_array.shape}, CT shape: {ct_array.shape}")
        
        # BƯỚC 8: Lấy slice giữa để test
        slice_idx = mri_array.shape[0] // 2
        mri_slice = mri_array[slice_idx]
        ct_slice = ct_array[slice_idx]
        mask_slice = mri_mask_cropped[slice_idx]
        
        # BƯỚC 9: Resize về 256x256
        if mri_slice.shape != (256, 256):
            from scipy.ndimage import zoom
            zoom_h = 256 / mri_slice.shape[0]
            zoom_w = 256 / mri_slice.shape[1]
            mri_slice = zoom(mri_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
            ct_slice = zoom(ct_slice, (zoom_h, zoom_w), order=1, mode='constant', cval=0)
            mask_slice = zoom(mask_slice, (zoom_h, zoom_w), order=0, mode='constant', cval=0)
        
        # BƯỚC 10: Normalize về [-1, 1] cho model
        mri_slice = np.clip(mri_slice, 0, 1)
        ct_slice = np.clip(ct_slice, 0, 1)
        mri_slice = mri_slice * 2.0 - 1.0
        ct_slice = ct_slice * 2.0 - 1.0
        
        # BƯỚC 11: Chuyển về tensor
        mri_tensor = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # BƯỚC 12: Generate synthetic CT
        with torch.no_grad():
            fake_ct = self.model.G_MRI2CT(mri_tensor)
            fake_ct = fake_ct.cpu().numpy()[0, 0]
        
        # BƯỚC 13: Convert về [0, 1] để tính metrics
        real_ct = (ct_slice + 1.0) / 2.0
        fake_ct = (fake_ct + 1.0) / 2.0
        mri_display = (mri_slice + 1.0) / 2.0
        
        # Clip về [0, 1]
        real_ct = np.clip(real_ct, 0, 1)
        fake_ct = np.clip(fake_ct, 0, 1)
        mri_display = np.clip(mri_display, 0, 1)
        
        # BƯỚC 14: Tính toán metrics
        metrics_calc = MetricsCalculator()
        
        # Chỉ tính metrics trong vùng brain (có mask)
        brain_region = mask_slice > 0.1  # Threshold để tạo binary mask
        
        if np.sum(brain_region) > 0:
            mae = metrics_calc.calculate_mae(real_ct[brain_region], fake_ct[brain_region])
            mse = metrics_calc.calculate_mse(real_ct[brain_region], fake_ct[brain_region])
            rmse = metrics_calc.calculate_rmse(real_ct[brain_region], fake_ct[brain_region])
            psnr = metrics_calc.calculate_psnr(real_ct[brain_region], fake_ct[brain_region])
            ssim = metrics_calc.calculate_ssim(real_ct, fake_ct, brain_region)
            ncc = metrics_calc.calculate_ncc(real_ct[brain_region], fake_ct[brain_region])
        else:
            print("Warning: Không tìm thấy brain region để tính metrics")
            mae = mse = rmse = psnr = ssim = ncc = 0.0
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'PSNR': psnr,
            'SSIM': ssim,
            'NCC': ncc
        }
        
        # BƯỚC 15: Lưu kết quả
        filename = os.path.basename(mri_path).replace('.nii.gz', '')
        
        # Tạo comparison image
        comparison_path = os.path.join(output_dir, f"{filename}_comparison_with_mask_applied.png")
        plt.figure(figsize=(20, 5))
        
        # MRI
        plt.subplot(1, 4, 1)
        plt.imshow(mri_display, cmap='gray')
        plt.title('Input MRI')
        plt.axis('off')
        
        # Real CT (with mask applied)
        plt.subplot(1, 4, 2)
        plt.imshow(real_ct, cmap='gray')
        plt.title('Real CT\n(Mask Applied)')
        plt.axis('off')
        
        # Synthetic CT
        plt.subplot(1, 4, 3)
        plt.imshow(fake_ct, cmap='gray')
        plt.title('Synthetic CT')
        plt.axis('off')
        
        # Brain mask
        plt.subplot(1, 4, 4)
        plt.imshow(mask_slice, cmap='gray')
        plt.title('Brain Mask')
        plt.axis('off')
        
        # Thêm metrics text
        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nPSNR: {psnr:.2f}dB\nSSIM: {ssim:.4f}\nNCC: {ncc:.4f}"
        plt.figtext(0.02, 0.02, metrics_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Đã lưu comparison với mask applied tại: {comparison_path}")
        print(f"Metrics (trong brain region): MAE={mae:.4f}, SSIM={ssim:.4f}, PSNR={psnr:.2f}dB")
        
        return metrics
    
    def test_dataset(self, test_loader: DataLoader, output_dir: str) -> Dict[str, float]:
        """
        Test trên toàn bộ test dataset với MRI mask được áp dụng
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        
        print("Đang test trên toàn bộ dataset...")
        print("Lưu ý: Dataset loader đã áp dụng MRI mask vào CT preprocessing")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                mri = batch['mri'].to(self.device)
                ct_real = batch['ct'].to(self.device)
                mask = batch.get('mask', None)  # Lấy mask nếu có
                filename = batch['filename'][0]  # Batch size = 1 cho test
                
                # Generate synthetic CT
                ct_fake = self.model.G_MRI2CT(mri)
                
                # Nếu có mask, áp dụng vào metrics
                if mask is not None:
                    mask = mask.to(self.device)
                    # Tính metrics chỉ trong vùng mask
                    ct_real_masked = ct_real * mask
                    ct_fake_masked = ct_fake * mask
                    metrics = MetricsCalculator.calculate_all_metrics(ct_fake_masked, ct_real_masked)
                else:
                    # Fallback về metrics toàn bộ nếu không có mask
                    metrics = MetricsCalculator.calculate_all_metrics(ct_fake, ct_real)
                
                all_metrics.append(metrics)
                
                # Lưu ảnh so sánh cho một số sample
                if batch_idx < 10:  # Chỉ lưu 10 sample đầu
                    save_path = os.path.join(output_dir, f"test_sample_{batch_idx:03d}.png")
                    
                    mri_np = mri[0, 0].cpu().numpy()
                    ct_real_np = ct_real[0, 0].cpu().numpy()
                    ct_fake_np = ct_fake[0, 0].cpu().numpy()
                    
                    if mask is not None:
                        mask_np = mask[0, 0].cpu().numpy()
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        
                        # Row 1: Input, Outputs, Ground Truth
                        axes[0, 0].imshow(mri_np, cmap='gray')
                        axes[0, 0].set_title('Input MRI')
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(ct_fake_np, cmap='gray')
                        axes[0, 1].set_title('Synthetic CT')
                        axes[0, 1].axis('off')
                        
                        axes[0, 2].imshow(ct_real_np, cmap='gray')
                        axes[0, 2].set_title('Ground Truth CT (Masked)')
                        axes[0, 2].axis('off')
                        
                        # Row 2: Mask, Difference Map, Metrics
                        axes[1, 0].imshow(mask_np, cmap='gray')
                        axes[1, 0].set_title('Brain+Skull Mask')
                        axes[1, 0].axis('off')
                        
                        diff = np.abs(ct_fake_np - ct_real_np) * mask_np
                        im = axes[1, 1].imshow(diff, cmap='hot')
                        axes[1, 1].set_title(f'Difference Map (MAE: {metrics["MAE"]:.4f})')
                        axes[1, 1].axis('off')
                        plt.colorbar(im, ax=axes[1, 1])
                        
                        # Metrics text
                        metrics_text = f"""Metrics (Brain Region):
MAE: {metrics['MAE']:.4f}
MSE: {metrics['MSE']:.4f}
RMSE: {metrics['RMSE']:.4f}
PSNR: {metrics['PSNR']:.2f} dB
SSIM: {metrics['SSIM']:.4f}
NCC: {metrics['NCC']:.4f}"""
                        
                        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                        axes[1, 2].axis('off')
                        
                    else:
                        # Fallback layout nếu không có mask
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        
                        axes[0].imshow(mri_np, cmap='gray')
                        axes[0].set_title('Input MRI')
                        axes[0].axis('off')
                        
                        axes[1].imshow(ct_fake_np, cmap='gray')
                        axes[1].set_title('Synthetic CT')
                        axes[1].axis('off')
                        
                        axes[2].imshow(ct_real_np, cmap='gray')
                        axes[2].set_title('Ground Truth CT')
                        axes[2].axis('off')
                        
                        diff = np.abs(ct_fake_np - ct_real_np)
                        im = axes[3].imshow(diff, cmap='hot')
                        axes[3].set_title(f'Difference (MAE: {metrics["MAE"]:.4f})')
                        axes[3].axis('off')
                        plt.colorbar(im, ax=axes[3])
                    
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
        
        # Tính metrics trung bình
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Test CycleGAN MRI to CT model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn tới model checkpoint')
    parser.add_argument('--test_mode', type=str, choices=['single', 'dataset', 'with_gt'], 
                        default='dataset', help='Chế độ test')
    parser.add_argument('--mri_path', type=str, help='Đường dẫn tới file MRI (cho single mode)')
    parser.add_argument('--ct_path', type=str, help='Đường dẫn tới file CT ground truth')
    parser.add_argument('--mri_dir', type=str, default='data/Test/MRI',
                        help='Thư mục chứa MRI test')
    parser.add_argument('--ct_dir', type=str, default='data/Test/CT',
                        help='Thư mục chứa CT test')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Thư mục lưu kết quả')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device sử dụng (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Kiểm tra device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA không khả dụng, chuyển sang CPU")
        args.device = 'cpu'
    
    # Khởi tạo tester
    tester = MRIToCTTester(args.model_path, args.device)
    
    if args.test_mode == 'single':
        if not args.mri_path:
            print("Cần cung cấp --mri_path cho single mode")
            return
        
        metrics = tester.test_single_image(args.mri_path, args.output_dir)
        if metrics:
            print_metrics(metrics, "Test Results")
    
    elif args.test_mode == 'with_gt':
        if not args.mri_path or not args.ct_path:
            print("Cần cung cấp --mri_path và --ct_path cho with_gt mode")
            return
        
        metrics = tester.test_with_ground_truth(args.mri_path, args.ct_path, args.output_dir)
        print_metrics(metrics, "Test Results")
    
    elif args.test_mode == 'dataset':
        # Tạo test loader
        if not os.path.exists(args.mri_dir) or not os.path.exists(args.ct_dir):
            print("Thư mục test không tồn tại")
            return
        
        test_loader = create_test_loader(args.mri_dir, args.ct_dir, batch_size=1)
        
        # Test trên dataset
        metrics = tester.test_dataset(test_loader, args.output_dir)
        print_metrics(metrics, "Average Test Results")
    
    print(f"Kết quả test đã được lưu tại: {args.output_dir}")


if __name__ == "__main__":
    main() 