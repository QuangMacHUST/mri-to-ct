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
        
    def test_single_image(self, mri_path: str, output_dir: str, save_comparison: bool = True) -> Dict[str, float]:
        """
        Test trên một ảnh MRI đơn lẻ
        
        Args:
            mri_path: đường dẫn tới file MRI
            output_dir: thư mục lưu kết quả
            save_comparison: có lưu ảnh so sánh không
            
        Returns:
            Dict chứa metrics nếu có CT ground truth
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Đang xử lý {mri_path}...")
        
        # Load và tiền xử lý MRI
        mri_sitk = sitk.ReadImage(mri_path)
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        
        # Áp dụng N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50] * 4)
        mri_sitk_corrected = corrector.Execute(mri_sitk)
        mri_array = sitk.GetArrayFromImage(mri_sitk_corrected).astype(np.float32)
        
        # Tạo binary mask và normalize
        from skimage import filters, morphology
        from scipy import ndimage
        
        normalized = ((mri_array - mri_array.min()) / 
                     (mri_array.max() - mri_array.min()) * 255).astype(np.uint8)
        threshold = filters.threshold_otsu(normalized)
        binary_mask = normalized > threshold
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=1000)
        binary_mask = ndimage.binary_fill_holes(binary_mask).astype(np.float32)
        
        # Normalize intensity với Min-Max normalization
        masked_values = mri_array[binary_mask > 0]
        if len(masked_values) > 0:
            min_val = np.min(masked_values)
            max_val = np.max(masked_values)
            if max_val > min_val:
                # Min-Max normalization về [0, 1]
                mri_array = (mri_array - min_val) / (max_val - min_val)
            else:
                # Trường hợp tất cả pixel có cùng giá trị
                mri_array = np.zeros_like(mri_array)
        
        # Áp dụng mask
        mri_array = mri_array * binary_mask
        
        # Chuyển về [-1, 1] để phù hợp với Tanh activation của Generator
        mri_array = mri_array * 2.0 - 1.0
        
        # Tạo CT mô phỏng cho từng slice
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
        
        # Lưu kết quả
        filename = os.path.basename(mri_path).replace('.nii.gz', '_synthetic_ct.nii.gz')
        output_path = os.path.join(output_dir, filename)
        save_nifti_image(fake_ct_volume, output_path, mri_path)
        
        # Lưu ảnh so sánh cho slice giữa
        if save_comparison:
            middle_slice = mri_array.shape[0] // 2
            comparison_path = os.path.join(output_dir, filename.replace('.nii.gz', '_comparison.png'))
            compare_images(
                mri_array[middle_slice], 
                fake_ct_volume[middle_slice],
                comparison_path,
                "MRI Input",
                "Synthetic CT"
            )
        
        print(f"CT mô phỏng đã được lưu tại: {output_path}")
        return {}
        
    def test_with_ground_truth(self, mri_path: str, ct_path: str, output_dir: str) -> Dict[str, float]:
        """
        Test với ground truth CT để tính metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Tạo CT mô phỏng
        self.test_single_image(mri_path, output_dir, save_comparison=False)
        
        # Load ground truth CT
        ct_sitk = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
        
        # Load synthetic CT
        synthetic_ct_path = os.path.join(output_dir, 
                                        os.path.basename(mri_path).replace('.nii.gz', '_synthetic_ct.nii.gz'))
        synthetic_ct_sitk = sitk.ReadImage(synthetic_ct_path)
        synthetic_ct_array = sitk.GetArrayFromImage(synthetic_ct_sitk).astype(np.float32)
        
        # Chuẩn hóa cả hai ảnh về cùng range
        ct_array = (ct_array - ct_array.min()) / (ct_array.max() - ct_array.min() + 1e-8)
        synthetic_ct_array = (synthetic_ct_array - synthetic_ct_array.min()) / (synthetic_ct_array.max() - synthetic_ct_array.min() + 1e-8)
        
        # Chuyển về tensor để tính metrics
        ct_tensor = torch.tensor(ct_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        synthetic_ct_tensor = torch.tensor(synthetic_ct_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Tính metrics
        metrics = MetricsCalculator.calculate_all_metrics(synthetic_ct_tensor, ct_tensor)
        
        # Lưu ảnh so sánh với ground truth
        middle_slice = ct_array.shape[0] // 2
        comparison_path = os.path.join(output_dir, 
                                      os.path.basename(mri_path).replace('.nii.gz', '_gt_comparison.png'))
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Load MRI để hiển thị
        mri_sitk = sitk.ReadImage(mri_path)
        mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
        mri_array = (mri_array - mri_array.min()) / (mri_array.max() - mri_array.min() + 1e-8)
        
        axes[0].imshow(mri_array[middle_slice], cmap='gray')
        axes[0].set_title('Input MRI')
        axes[0].axis('off')
        
        axes[1].imshow(synthetic_ct_array[middle_slice], cmap='gray')
        axes[1].set_title('Synthetic CT')
        axes[1].axis('off')
        
        axes[2].imshow(ct_array[middle_slice], cmap='gray')
        axes[2].set_title('Ground Truth CT')
        axes[2].axis('off')
        
        diff = np.abs(synthetic_ct_array[middle_slice] - ct_array[middle_slice])
        im = axes[3].imshow(diff, cmap='hot')
        axes[3].set_title('Difference Map')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])
        
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return metrics
    
    def test_dataset(self, test_loader: DataLoader, output_dir: str) -> Dict[str, float]:
        """
        Test trên toàn bộ test dataset
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_metrics = []
        
        print("Đang test trên toàn bộ dataset...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
                mri = batch['mri'].to(self.device)
                ct_real = batch['ct'].to(self.device)
                filename = batch['filename'][0]  # Batch size = 1 cho test
                
                # Generate synthetic CT
                ct_fake = self.model.G_MRI2CT(mri)
                
                # Tính metrics
                metrics = MetricsCalculator.calculate_all_metrics(ct_fake, ct_real)
                all_metrics.append(metrics)
                
                # Lưu ảnh so sánh cho một số sample
                if batch_idx < 10:  # Chỉ lưu 10 sample đầu
                    save_path = os.path.join(output_dir, f"test_sample_{batch_idx:03d}.png")
                    
                    mri_np = mri[0, 0].cpu().numpy()
                    ct_real_np = ct_real[0, 0].cpu().numpy()
                    ct_fake_np = ct_fake[0, 0].cpu().numpy()
                    
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