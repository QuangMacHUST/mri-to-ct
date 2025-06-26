import torch
import torch.nn.functional as F
import numpy as np
try:
    # Thử import từ skimage.metrics trước (phiên bản mới)
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    try:
        # Fallback cho phiên bản cũ
        from skimage.measure import compare_ssim as ssim
    except ImportError:
        # Fallback cuối cùng - tự implement SSIM đơn giản
        def ssim(img1, img2, data_range=1.0, win_size=7):
            """Simple SSIM fallback implementation"""
            # Đơn giản hóa SSIM bằng correlation
            img1_flat = img1.flatten()
            img2_flat = img2.flatten()
            
            if len(img1_flat) <= 1:
                return 0.0
                
            # Tính correlation
            corr = np.corrcoef(img1_flat, img2_flat)
            if corr.shape == (2, 2):
                return max(0, min(1, corr[0, 1]))
            else:
                return 0.0

from typing import Dict, Tuple

class MetricsCalculator:
    """
    Class tính toán các metrics đánh giá mô hình
    """
    
    @staticmethod
    def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Mean Absolute Error (MAE)
        """
        return F.l1_loss(pred, target).item()
    
    @staticmethod
    def calculate_mae(pred, target) -> float:
        """
        Alias cho mean_absolute_error với xử lý numpy arrays
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        return MetricsCalculator.mean_absolute_error(pred, target)
    
    @staticmethod
    def calculate_mse(pred, target) -> float:
        """
        Alias cho mean_squared_error với xử lý numpy arrays
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        return MetricsCalculator.mean_squared_error(pred, target)
    
    @staticmethod
    def calculate_rmse(pred, target) -> float:
        """
        Alias cho root_mean_squared_error với xử lý numpy arrays
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        return MetricsCalculator.root_mean_squared_error(pred, target)
    
    @staticmethod
    def calculate_psnr(pred, target, max_value: float = 2.0) -> float:
        """
        Alias cho peak_signal_to_noise_ratio với xử lý numpy arrays
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        return MetricsCalculator.peak_signal_to_noise_ratio(pred, target, max_value)
    
    @staticmethod
    def calculate_ssim(pred, target, mask=None) -> float:
        """
        Alias cho structural_similarity_index với xử lý numpy arrays và mask
        """
        if isinstance(pred, np.ndarray):
            # Nếu là 2D, expand thành batch format [1, 1, H, W]
            if pred.ndim == 2:
                pred = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
            else:
                pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            if target.ndim == 2:
                target = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
            else:
                target = torch.from_numpy(target).float()
        
        # Nếu có mask, áp dụng trước khi tính SSIM
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            
            # Áp dụng mask
            pred = pred * mask
            target = target * mask
            
        return MetricsCalculator.structural_similarity_index(pred, target)
    
    @staticmethod
    def calculate_ncc(pred, target) -> float:
        """
        Alias cho normalized_cross_correlation với xử lý numpy arrays
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0)
        return MetricsCalculator.normalized_cross_correlation(pred, target)
    
    @staticmethod
    def calculate_dice(pred, target, threshold: float = 0.5, mask=None) -> float:
        """
        Tính Dice Coefficient (Sørensen–Dice coefficient) cho medical image segmentation
        
        Args:
            pred: tensor hoặc array dự đoán
            target: tensor hoặc array ground truth
            threshold: ngưỡng để chuyển về binary (default: 0.5 cho normalized data)
            mask: mask để chỉ tính trong vùng quan tâm
        
        Returns:
            DICE score trong khoảng [0, 1] với 1 là tốt nhất
        """
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).float()
        
        return MetricsCalculator.dice_coefficient(pred, target, threshold, mask)
    
    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, mask=None) -> float:
        """
        Tính Dice Coefficient cho medical image evaluation
        
        Formula: DICE = 2 * |A ∩ B| / (|A| + |B|) = 2 * TP / (2*TP + FP + FN)
        
        Args:
            pred: tensor dự đoán shape [batch, channels, height, width] trong range [-1,1] hoặc [0,1]
            target: tensor ground truth cùng shape và range với pred
            threshold: ngưỡng để chuyển về binary
            mask: mask để chỉ tính trong vùng quan tâm (brain region)
        
        Returns:
            DICE score trong khoảng [0, 1] với 1 là perfect match
        """
        # Xử lý tensor sang CPU và numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Convert từ [-1,1] về [0,1] nếu cần
        if pred_np.min() < 0:
            pred_np = (pred_np + 1.0) / 2.0
            target_np = (target_np + 1.0) / 2.0
        
        # Clamp để đảm bảo trong [0,1]
        pred_np = np.clip(pred_np, 0, 1)
        target_np = np.clip(target_np, 0, 1)
        
        dice_scores = []
        
        # Tính DICE cho từng sample trong batch
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]  # [H, W]
            target_img = target_np[i, 0]  # [H, W]
            
            # Áp dụng mask nếu có
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.detach().cpu().numpy()
                else:
                    mask_np = mask
                
                if mask_np.ndim == 4:  # [batch, channels, H, W]
                    mask_slice = mask_np[i, 0]
                elif mask_np.ndim == 3:  # [batch, H, W]
                    mask_slice = mask_np[i]
                else:  # [H, W]
                    mask_slice = mask_np
                
                # Chỉ tính trong vùng mask
                pred_img = pred_img * mask_slice
                target_img = target_img * mask_slice
            
            # Chuyển về binary based on threshold
            pred_binary = (pred_img > threshold).astype(np.float32)
            target_binary = (target_img > threshold).astype(np.float32)
            
            # Tính intersection và union
            intersection = np.sum(pred_binary * target_binary)
            pred_sum = np.sum(pred_binary)
            target_sum = np.sum(target_binary)
            
            # Tính DICE coefficient
            if pred_sum + target_sum == 0:
                # Trường hợp cả hai đều empty (all zeros)
                dice_score = 1.0  # Perfect match khi cả hai đều empty
            else:
                dice_score = (2.0 * intersection) / (pred_sum + target_sum)
            
            dice_scores.append(dice_score)
        
        # Trả về trung bình DICE của batch
        return np.mean(dice_scores)
    
    @staticmethod
    def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Mean Squared Error (MSE)
        """
        return F.mse_loss(pred, target).item()
    
    @staticmethod
    def root_mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Root Mean Squared Error (RMSE)
        """
        return torch.sqrt(F.mse_loss(pred, target)).item()
    
    @staticmethod
    def peak_signal_to_noise_ratio(pred: torch.Tensor, target: torch.Tensor, max_value: float = 2.0) -> float:
        """
        Tính Peak Signal-to-Noise Ratio (PSNR) cho CycleGAN data (range [-1,1])
        
        Args:
            pred: ảnh dự đoán trong range [-1,1]
            target: ảnh ground truth trong range [-1,1]
            max_value: giá trị pixel maximum range (2.0 cho data [-1,1])
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def structural_similarity_index(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        FIXED SSIM Implementation - Giải quyết hoàn toàn plateau 0.8 issue
        
        BUG FIXES:
        1. data_range=1.0 cho normalized [0,1] data (không auto-detect)
        2. Loại bỏ fallback correlation gây plateau
        3. Proper win_size handling
        """
        # Chuyển tensor về numpy và xử lý batch
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Convert từ [-1,1] về [0,1] để phù hợp với SSIM algorithm
        pred_np = (pred_np + 1.0) / 2.0
        target_np = (target_np + 1.0) / 2.0
        
        # Clamp để đảm bảo trong [0,1]
        pred_np = np.clip(pred_np, 0, 1)
        target_np = np.clip(target_np, 0, 1)
        
        ssim_values = []
        
        # Tính SSIM cho từng sample trong batch
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i, 0]  # [H, W]
            target_img = target_np[i, 0]  # [H, W]
            
            # Kiểm tra kích thước tối thiểu
            min_side = min(pred_img.shape)
            if min_side < 7:
                # Resize về 7x7 minimum thay vì fallback correlation
                from scipy.ndimage import zoom
                zoom_factor = 7.0 / min_side
                pred_img = zoom(pred_img, zoom_factor, order=1)
                target_img = zoom(target_img, zoom_factor, order=1)
                min_side = 7
            
            # Tính win_size hợp lý (phải lẻ và <= min_side)
            win_size = min(11, min_side)  # Tăng lên 11 để tăng độ chính xác
            if win_size % 2 == 0:
                win_size -= 1
            win_size = max(3, win_size)  # Minimum 3x3
            
            # CRITICAL FIX: Sử dụng data_range=1.0 cố định cho normalized [0,1] data
            # KHÔNG auto-detect vì đây là nguyên nhân chính của plateau 0.8
            try:
                # Kiểm tra variance trước khi tính SSIM
                pred_var = np.var(pred_img)
                target_var = np.var(target_img)
                
                if pred_var < 1e-7 or target_var < 1e-7:
                    # Images quá uniform, SSIM không meaningful
                    ssim_val = 0.0
                else:
                    # FIXED: data_range=1.0 cho normalized data - đây là key fix!
                    ssim_val = ssim(pred_img, target_img, data_range=1.0, win_size=win_size)
                    
                    # Validation check
                    if np.isnan(ssim_val) or np.isinf(ssim_val):
                        ssim_val = 0.0
                    elif ssim_val < -1.0 or ssim_val > 1.0:
                        # SSIM should be in [-1, 1], clip if outside
                        ssim_val = np.clip(ssim_val, -1.0, 1.0)
                        
            except Exception as e:
                print(f"SSIM calculation failed: {e}")
                ssim_val = 0.0
                
            ssim_values.append(ssim_val)
        
        # Return safe mean
        if ssim_values:
            valid_values = [v for v in ssim_values if not (np.isnan(v) or np.isinf(v))]
            if valid_values:
                mean_ssim = np.mean(valid_values)
                # Final safety check - SSIM vẫn có thể âm cho very different images
                return float(mean_ssim)
            else:
                return 0.0
        else:
            return 0.0
    
    @staticmethod
    def normalized_cross_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Normalized Cross Correlation (NCC) với robust handling
        """
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        ncc_values = []
        
        for i in range(pred.size(0)):
            pred_sample = pred_flat[i]
            target_sample = target_flat[i]
            
            # Compute means
            pred_mean = pred_sample.mean()
            target_mean = target_sample.mean()
            
            # Center the data
            pred_centered = pred_sample - pred_mean
            target_centered = target_sample - target_mean
            
            # Compute standard deviations
            pred_std = torch.sqrt(torch.mean(pred_centered**2))
            target_std = torch.sqrt(torch.mean(target_centered**2))
            
            # Check for zero variance to avoid division by zero
            if pred_std > 1e-8 and target_std > 1e-8:
                # Compute correlation
                correlation = torch.mean(pred_centered * target_centered) / (pred_std * target_std)
                
                # Check for valid result
                if torch.isfinite(correlation):
                    ncc_values.append(correlation.item())
                else:
                    ncc_values.append(0.0)
            else:
                ncc_values.append(0.0)
        
        # Return mean of valid values
        if ncc_values:
            return np.mean(ncc_values)
        else:
            return 0.0
    
    @staticmethod
    def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, max_value: float = 2.0, mask=None) -> Dict[str, float]:
        """
        Tính tất cả metrics cùng lúc cho CycleGAN data (range [-1,1]), bao gồm DICE score cho medical imaging
        
        Args:
            pred: tensor dự đoán
            target: tensor ground truth
            max_value: giá trị maximum cho PSNR calculation
            mask: mask để áp dụng cho DICE calculation trong brain region
        """
        metrics = {}
        
        metrics['MAE'] = MetricsCalculator.mean_absolute_error(pred, target)
        metrics['MSE'] = MetricsCalculator.mean_squared_error(pred, target)
        metrics['RMSE'] = MetricsCalculator.root_mean_squared_error(pred, target)
        metrics['PSNR'] = MetricsCalculator.peak_signal_to_noise_ratio(pred, target, max_value)
        metrics['SSIM'] = MetricsCalculator.structural_similarity_index(pred, target)
        metrics['NCC'] = MetricsCalculator.normalized_cross_correlation(pred, target)
        metrics['DICE'] = MetricsCalculator.dice_coefficient(pred, target, threshold=0.5, mask=mask)
        
        return metrics


class MetricsTracker:
    """
    Class theo dõi metrics qua các epoch
    """
    
    def __init__(self):
        self.metrics_history = {
            'MAE': [],
            'MSE': [],
            'RMSE': [],
            'PSNR': [],
            'SSIM': [],
            'NCC': [],
            'DICE': []
        }
        self.calculator = MetricsCalculator()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, max_value: float = 2.0):
        """
        Cập nhật metrics cho batch hiện tại với CycleGAN data (range [-1,1])
        """
        metrics = self.calculator.calculate_all_metrics(pred, target, max_value)
        
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append(value)
    
    def get_epoch_average(self) -> Dict[str, float]:
        """
        Lấy giá trị trung bình của epoch hiện tại
        """
        epoch_metrics = {}
        for metric_name, values in self.metrics_history.items():
            if values:
                epoch_metrics[metric_name] = np.mean(values)
            else:
                epoch_metrics[metric_name] = 0.0
        
        return epoch_metrics
    
    def reset_epoch(self):
        """
        Reset metrics cho epoch mới
        """
        for metric_name in self.metrics_history:
            self.metrics_history[metric_name] = []
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Lấy metrics tốt nhất (SSIM, PSNR cao nhất, MAE, MSE thấp nhất)
        """
        best_metrics = {}
        
        # Metrics cao hơn thì tốt hơn
        for metric in ['SSIM', 'PSNR', 'NCC', 'DICE']:
            if self.metrics_history[metric]:
                best_metrics[f'best_{metric}'] = max(self.metrics_history[metric])
        
        # Metrics thấp hơn thì tốt hơn
        for metric in ['MAE', 'MSE', 'RMSE']:
            if self.metrics_history[metric]:
                best_metrics[f'best_{metric}'] = min(self.metrics_history[metric])
        
        return best_metrics


def evaluate_model(model, dataloader, device: str = 'cuda') -> Dict[str, float]:
    """
    Đánh giá mô hình trên toàn bộ dataset
    """
    model.eval()
    tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch in dataloader:
            mri = batch['mri'].to(device)
            ct_real = batch['ct'].to(device)
            
            # Generate fake CT
            ct_fake = model.G_MRI2CT(mri)
            
            # Update metrics
            tracker.update(ct_fake, ct_real)
    
    return tracker.get_epoch_average()


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    In metrics theo format đẹp
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        if 'PSNR' in metric_name:
            print(f"{metric_name}: {value:.4f} dB")
        elif metric_name in ['MAE', 'MSE', 'RMSE']:
            print(f"{metric_name}: {value:.6f}")
        else:
            print(f"{metric_name}: {value:.4f}")
    print("-" * 50) 