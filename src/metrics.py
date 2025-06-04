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
            
            # Kiểm tra NaN/Inf trước khi normalize
            if np.any(np.isnan(pred_img)) or np.any(np.isnan(target_img)) or \
               np.any(np.isinf(pred_img)) or np.any(np.isinf(target_img)):
                ssim_values.append(0.0)
                continue
            
            # Chuẩn hóa về [0, 1] cho SSIM với safe division
            pred_range = pred_img.max() - pred_img.min()
            target_range = target_img.max() - target_img.min()
            
            if pred_range > 1e-8:
                pred_img = (pred_img - pred_img.min()) / pred_range
            else:
                pred_img = np.zeros_like(pred_img)
                
            if target_range > 1e-8:
                target_img = (target_img - target_img.min()) / target_range
            else:
                target_img = np.zeros_like(target_img)
            
            # Tính win_size phù hợp với kích thước ảnh
            min_side = min(pred_img.shape)
            if min_side < 7:
                # Nếu ảnh quá nhỏ, dùng win_size nhỏ hơn
                win_size = max(3, min_side if min_side % 2 == 1 else min_side - 1)
            else:
                win_size = 7  # default
            
            # Tính SSIM với win_size phù hợp
            try:
                ssim_val = ssim(pred_img, target_img, data_range=1.0, win_size=win_size)
                
                # Kiểm tra kết quả SSIM có hợp lệ không
                if np.isnan(ssim_val) or np.isinf(ssim_val):
                    ssim_val = 0.0
                    
            except (ValueError, RuntimeWarning):
                # Fallback nếu vẫn lỗi - dùng correlation thay thế
                pred_flat = pred_img.flatten()
                target_flat = target_img.flatten()
                
                # Safe correlation calculation
                if len(pred_flat) > 1 and np.std(pred_flat) > 1e-8 and np.std(target_flat) > 1e-8:
                    try:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            corr_matrix = np.corrcoef(pred_flat, target_flat)
                            correlation = corr_matrix[0, 1]
                            
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                else:
                    correlation = 0.0
                    
                ssim_val = max(0, correlation)  # Clamp về [0, 1]
                
            ssim_values.append(ssim_val)
        
        # Return safe mean
        if ssim_values:
            return np.mean([v for v in ssim_values if not (np.isnan(v) or np.isinf(v))])
        else:
            return 0.0
    
    @staticmethod
    def normalized_cross_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Tính Normalized Cross Correlation (NCC) với safe division
        """
        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Normalize
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        
        pred_norm = pred_flat - pred_mean
        target_norm = target_flat - target_mean
        
        # Compute correlation với safe division
        numerator = (pred_norm * target_norm).sum(dim=1)
        pred_std = torch.sqrt((pred_norm ** 2).sum(dim=1))
        target_std = torch.sqrt((target_norm ** 2).sum(dim=1))
        denominator = pred_std * target_std
        
        # Safe division với epsilon lớn hơn
        ncc = numerator / (denominator + 1e-8)
        
        # Filter out NaN/Inf values
        ncc = ncc[torch.isfinite(ncc)]
        
        if len(ncc) > 0:
            return ncc.mean().item()
        else:
            return 0.0
    
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