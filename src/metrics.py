import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
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
    def peak_signal_to_noise_ratio(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
        """
        Tính Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            pred: ảnh dự đoán
            target: ảnh ground truth
            max_value: giá trị pixel maximum (mặc định 1.0 cho ảnh normalized)
        """
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
        return psnr.item()
    
    @staticmethod
    def structural_similarity_index(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Structural Similarity Index (SSIM)
        """
        # Chuyển tensor về numpy và xử lý batch
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        ssim_values = []
        
        # Tính SSIM cho từng sample trong batch
        for i in range(pred_np.shape[0]):
            # Lấy channel đầu tiên nếu có nhiều channel
            pred_img = pred_np[i, 0] if pred_np.shape[1] > 1 else pred_np[i]
            target_img = target_np[i, 0] if target_np.shape[1] > 1 else target_np[i]
            
            # Chuẩn hóa về [0, 1] cho SSIM
            pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min() + 1e-8)
            target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
            
            # Tính SSIM
            ssim_val = ssim(pred_img, target_img, data_range=1.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    @staticmethod
    def normalized_cross_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Normalized Cross Correlation (NCC)
        """
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Normalize
        pred_norm = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_norm = target_flat - target_flat.mean(dim=1, keepdim=True)
        
        # Compute correlation
        numerator = (pred_norm * target_norm).sum(dim=1)
        denominator = torch.sqrt((pred_norm ** 2).sum(dim=1) * (target_norm ** 2).sum(dim=1))
        
        ncc = numerator / (denominator + 1e-8)
        return ncc.mean().item()
    
    @staticmethod
    def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> Dict[str, float]:
        """
        Tính tất cả metrics cùng lúc
        """
        metrics = {}
        
        metrics['MAE'] = MetricsCalculator.mean_absolute_error(pred, target)
        metrics['MSE'] = MetricsCalculator.mean_squared_error(pred, target)
        metrics['RMSE'] = MetricsCalculator.root_mean_squared_error(pred, target)
        metrics['PSNR'] = MetricsCalculator.peak_signal_to_noise_ratio(pred, target, max_value)
        metrics['SSIM'] = MetricsCalculator.structural_similarity_index(pred, target)
        metrics['NCC'] = MetricsCalculator.normalized_cross_correlation(pred, target)
        
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
            'NCC': []
        }
        self.calculator = MetricsCalculator()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0):
        """
        Cập nhật metrics cho batch hiện tại
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
        for metric in ['SSIM', 'PSNR', 'NCC']:
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