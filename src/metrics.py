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
        Tính Structural Similarity Index (SSIM) cho CycleGAN data (range [-1,1])
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
                # Nếu ảnh quá nhỏ, dùng correlation thay thế
                pred_flat = pred_img.flatten()
                target_flat = target_img.flatten()
                
                # Kiểm tra variance để tránh division by zero
                pred_var = np.var(pred_flat)
                target_var = np.var(target_flat)
                
                if pred_var > 1e-10 and target_var > 1e-10:
                    try:
                        # Manually compute correlation để tránh np.corrcoef warning
                        pred_centered = pred_flat - np.mean(pred_flat)
                        target_centered = target_flat - np.mean(target_flat)
                        
                        correlation = np.sum(pred_centered * target_centered) / (
                            np.sqrt(np.sum(pred_centered**2)) * np.sqrt(np.sum(target_centered**2))
                        )
                        
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = 0.0
                        
                        ssim_val = max(0, min(1, abs(correlation)))
                    except:
                        ssim_val = 0.0
                else:
                    ssim_val = 0.0
                
                ssim_values.append(ssim_val)
                continue
            
            # Tính win_size phù hợp
            win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
            
            # Tính SSIM với data_range=1.0 (vì đã convert về [0,1])
            try:
                ssim_val = ssim(pred_img, target_img, data_range=1.0, win_size=win_size)
                
                # Kiểm tra kết quả hợp lệ
                if np.isnan(ssim_val) or np.isinf(ssim_val):
                    ssim_val = 0.0
                    
            except Exception:
                # Fallback sử dụng correlation manual
                pred_flat = pred_img.flatten()
                target_flat = target_img.flatten()
                
                # Kiểm tra variance để tránh division by zero
                pred_var = np.var(pred_flat)
                target_var = np.var(target_flat)
                
                if pred_var > 1e-10 and target_var > 1e-10:
                    try:
                        # Manually compute correlation để tránh np.corrcoef warning
                        pred_centered = pred_flat - np.mean(pred_flat)
                        target_centered = target_flat - np.mean(target_flat)
                        
                        correlation = np.sum(pred_centered * target_centered) / (
                            np.sqrt(np.sum(pred_centered**2)) * np.sqrt(np.sum(target_centered**2))
                        )
                        
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = 0.0
                        
                        ssim_val = max(0, min(1, abs(correlation)))
                    except:
                        ssim_val = 0.0
                else:
                    ssim_val = 0.0
                
            ssim_values.append(ssim_val)
        
        # Return safe mean
        if ssim_values:
            valid_values = [v for v in ssim_values if not (np.isnan(v) or np.isinf(v))]
            return np.mean(valid_values) if valid_values else 0.0
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
    def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, max_value: float = 2.0) -> Dict[str, float]:
        """
        Tính tất cả metrics cùng lúc cho CycleGAN data (range [-1,1])
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