import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

from cached_data_loader import CachedDataLoaderManager
from optimized_cached_data_loader import OptimizedDataLoaderManager
from volume_based_cached_loader import VolumeCachedDataLoaderManager
from multi_slice_cached_loader import MultiSliceDataLoaderManager
from models import CycleGAN, weights_init_normal
from metrics import MetricsCalculator
from utils import save_checkpoint, load_checkpoint, create_sample_images

class CycleGANTrainer:
    """
    CycleGAN Trainer với cải tiến learning rate scheduling và early stopping
    """
    
    def __init__(self, 
                 config: Dict,
                 device: str = 'cuda',
                 resume_from_checkpoint: bool = True,
                 checkpoint_path: str = None):
        """
        Args:
            config: dictionary chứa các tham số training
            device: thiết bị training (cuda/cpu)
            resume_from_checkpoint: có tải lại checkpoint không
            checkpoint_path: đường dẫn cụ thể đến checkpoint (nếu có)
        """
        self.config = config
        self.device = torch.device(device)
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Training state
        self.current_epoch = 0
        self.best_ssim = 0.0
        self.epochs_without_improvement = 0
        self.max_patience = 25  # Early stopping patience
        
        # Gradient explosion monitoring
        self.gradient_explosion_count = 0
        self.max_explosion_per_epoch = 50  # Cho phép tối đa 50 explosions/epoch
        self.min_lr_threshold = 1e-7  # Minimum learning rate threshold
        
        # EMERGENCY: Warmup strategy to prevent early explosions
        self.warmup_epochs = 5  # Warmup trong 5 epochs đầu
        self.initial_lr_G = config['lr_G']
        self.initial_lr_D = config['lr_D']
        
        # Khởi tạo model
        self.model = CycleGAN(
            input_nc=config['input_nc'],
            output_nc=config['output_nc'],
            n_residual_blocks=config['n_residual_blocks'],
            discriminator_layers=config['discriminator_layers']
        ).to(self.device)
        
        # Khởi tạo trọng số
        self.model.apply(weights_init_normal)
        
        # Optimizers với learning rate thấp hơn
        self.optimizer_G = optim.AdamW(
            list(self.model.G_MRI2CT.parameters()) + list(self.model.G_CT2MRI.parameters()),
            lr=config['lr_G'],
            betas=(config['beta1'], config['beta2']),
            weight_decay= 0.00001
        )
        
        self.optimizer_D = optim.AdamW(
            list(self.model.D_CT.parameters()) + list(self.model.D_MRI.parameters()),
            lr=config['lr_D'],
            betas=(config['beta1'], config['beta2']),
            weight_decay= 0.00001
        )
        
        # Learning rate schedulers - Linear decay như paper gốc + Cosine warmup
        # Medical imaging GANs benefit from stable, gradual LR decay
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=30,
            gamma=0.8
        )
        
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=30,
            gamma=0.9
        )
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Load checkpoint nếu được chỉ định
        if self.resume_from_checkpoint and checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        print(f"Model khởi tạo thành công trên {device}")
        print(f"Tổng số parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Bắt đầu từ epoch: {self.current_epoch}")
        if self.best_ssim > 0:
            print(f"Best SSIM hiện tại: {self.best_ssim:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Tải checkpoint từ đường dẫn cụ thể
        Args:
            checkpoint_path: đường dẫn đến file checkpoint
        """
        if os.path.exists(checkpoint_path):
            try:
                print(f"Đang tải checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer states
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
                
                # Load scheduler states
                self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
                self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
                
                # Load training state
                self.current_epoch = checkpoint['epoch'] + 1  # Bắt đầu từ epoch tiếp theo
                self.best_ssim = checkpoint.get('best_ssim', 0.0)
                
                print(f"✅ Đã tải checkpoint thành công từ epoch {checkpoint['epoch']}")
                print(f"   Checkpoint được lưu tại epoch {checkpoint['epoch']}")
                print(f"   Sẽ tiếp tục training từ epoch {self.current_epoch}")
                
            except Exception as e:
                print(f"❌ Lỗi khi tải checkpoint: {e}")
                print("   Sẽ bắt đầu training từ đầu")
                self.current_epoch = 0
                self.best_ssim = 0.0
        else:
            print(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
    
    def get_current_lr(self):
        """Lấy learning rate hiện tại"""
        return {
            'lr_G': self.optimizer_G.param_groups[0]['lr'],
            'lr_D': self.optimizer_D.param_groups[0]['lr']
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Training một epoch với GPU memory monitoring, gradient clipping và learning rate scheduling
        """
        self.model.train()
        
        # Training metrics
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        # Chi tiết từng loại loss
        total_adv_loss = 0.0
        total_cycle_loss = 0.0
        total_perceptual_loss = 0.0
        total_d_ct_loss = 0.0
        total_d_mri_loss = 0.0
        
        total_metrics = {
            'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 
            'psnr': 0.0, 'ssim': 0.0, 'ncc': 0.0
        }
        
        num_batches = len(train_loader)
        metrics_calculator = MetricsCalculator()
        
        # Progress bar với thông tin learning rate và metrics đầy đủ
        current_lr = self.get_current_lr()
        pbar = tqdm(train_loader, 
                   desc=f"Training Epoch {self.current_epoch+1}: LR_G={current_lr['lr_G']:.6f}")
        
        # Reset gradient explosion count cho epoch mới  
        self.gradient_explosion_count = 0
        
        for i, batch in enumerate(pbar):
            real_mri = batch['mri'].to(self.device)
            real_ct = batch['ct'].to(self.device)
            
            # Check nếu quá nhiều gradient explosions trong epoch này
            if self.gradient_explosion_count > self.max_explosion_per_epoch:
                print(f"🚨 Quá nhiều gradient explosions ({self.gradient_explosion_count}), dừng epoch sớm!")
                break
            
            # Kiểm tra gradient explosion trước khi training
            if self._check_gradient_explosion():
                print(f"⚠️  Phát hiện gradient explosion tại batch {i}, đặt lại learning rate...")
                self._reset_learning_rate()
                self.gradient_explosion_count += 1
                continue
                
            # =============== Train Generator ===============
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            outputs = self.model(real_mri, real_ct)
            
            # Compute generator losses
            g_losses = self.model.generator_loss(real_mri, real_ct, outputs)
            g_losses['total'].backward()
            
            # ULTRA AGGRESSIVE gradient clipping - Medical imaging needs stability
            grad_norm_g = torch.nn.utils.clip_grad_norm_(
                list(self.model.G_MRI2CT.parameters()) + list(self.model.G_CT2MRI.parameters()),
                max_norm=2.0  # RESTORE: 0.1 → 2.0 để model có thể học hiệu quả hơn 
            )
            
            # Check for gradient explosion với threshold hợp lý
            if grad_norm_g > 10.0:  # RESTORE: 1.0 → 10.0 để giảm false positives
                print(f"⚠️ Generator gradient explosion detected: {grad_norm_g:.2f}")
                self.gradient_explosion_count += 1
                # MODERATE RESPONSE: Giảm LR nhẹ hơn
                for g in self.optimizer_G.param_groups:
                    g['lr'] *= 0.8  # MODERATE: 0.5 → 0.8 để không giảm quá mạnh
                    # Emergency brake: Nếu LR quá thấp, set về threshold minimum
                    if g['lr'] < self.min_lr_threshold:
                        g['lr'] = self.min_lr_threshold
                        print(f"🔴 Generator LR hit minimum threshold: {self.min_lr_threshold}")
                self.optimizer_G.zero_grad()
                continue
            
            self.optimizer_G.step()
            
            # =============== Train Discriminator ===============
            self.optimizer_D.zero_grad()
            
            # Compute discriminator losses
            d_losses = self.model.discriminator_loss(real_mri, real_ct, outputs)
            d_losses['total'].backward()
            
            # ULTRA AGGRESSIVE gradient clipping cho Discriminator 
            grad_norm_d = torch.nn.utils.clip_grad_norm_(
                list(self.model.D_CT.parameters()) + list(self.model.D_MRI.parameters()),
                max_norm=2.0  # RESTORE: 0.1 → 2.0 để model có thể học hiệu quả hơn
            )
            
            # Check for gradient explosion với threshold hợp lý  
            if grad_norm_d > 10.0:  # RESTORE: 1.0 → 10.0 để giảm false positives
                print(f"⚠️ Discriminator gradient explosion detected: {grad_norm_d:.2f}")
                self.gradient_explosion_count += 1
                # MODERATE RESPONSE: Giảm LR nhẹ hơn
                for g in self.optimizer_D.param_groups:
                    g['lr'] *= 0.8  # MODERATE: 0.5 → 0.8 để không giảm quá mạnh
                    # Emergency brake: Nếu LR quá thấp, set về threshold minimum
                    if g['lr'] < self.min_lr_threshold:
                        g['lr'] = self.min_lr_threshold
                        print(f"🔴 Discriminator LR hit minimum threshold: {self.min_lr_threshold}")
                self.optimizer_D.zero_grad()
                continue
            
            self.optimizer_D.step()
            
            # ❌ KHÔNG UPDATE SCHEDULER MỖI BATCH - SẼ UPDATE MỖI EPOCH!
            # Scheduler update per batch sẽ khiến LR giảm quá nhanh
            
            # Accumulate losses - tổng
            total_g_loss += g_losses['total'].item()
            total_d_loss += d_losses['total'].item()
            
            # Accumulate chi tiết từng loại loss
            total_adv_loss += g_losses['gan'].item()
            total_cycle_loss += g_losses['cycle'].item()
            total_perceptual_loss += g_losses['perceptual'].item()
            total_d_ct_loss += d_losses['D_CT'].item()
            total_d_mri_loss += d_losses['D_MRI'].item()
            
            # Tính metrics cho batch hiện tại
            with torch.no_grad():
                fake_ct = outputs['fake_ct']
                batch_metrics = metrics_calculator.calculate_all_metrics(
                    fake_ct,  # Truyền tensor trực tiếp
                    real_ct   # Truyền tensor trực tiếp
                )
                
                # Accumulate metrics với key mapping
                metric_key_mapping = {
                    'mae': 'MAE',
                    'mse': 'MSE', 
                    'rmse': 'RMSE',
                    'psnr': 'PSNR',
                    'ssim': 'SSIM',
                    'ncc': 'NCC'
                }
                for key in total_metrics:
                    if metric_key_mapping[key] in batch_metrics:
                        total_metrics[key] += batch_metrics[metric_key_mapping[key]]
            
            # Cập nhật progress bar với chi tiết loss components
            pbar.set_postfix({
                'G_loss': f"{g_losses['total'].item():.4f}",
                'Adv': f"{g_losses['gan'].item():.4f}",
                'Cyc': f"{g_losses['cycle'].item():.4f}",
                'Perc': f"{g_losses['perceptual'].item():.4f}",
                'D_loss': f"{d_losses['total'].item():.4f}",
                'SSIM': f"{batch_metrics['SSIM']:.4f}"
            })
        
        # Average metrics over epoch
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        
        # Average chi tiết các loss components
        avg_adv_loss = total_adv_loss / num_batches
        avg_cycle_loss = total_cycle_loss / num_batches
        avg_perceptual_loss = total_perceptual_loss / num_batches
        avg_d_ct_loss = total_d_ct_loss / num_batches
        avg_d_mri_loss = total_d_mri_loss / num_batches
        
        avg_metrics = {key: value / num_batches for key, value in total_metrics.items()}
        
        return {
            'generator_loss': avg_g_loss,
            'discriminator_loss': avg_d_loss,
            'adversarial_loss': avg_adv_loss,
            'cycle_loss': avg_cycle_loss,
            'perceptual_loss': avg_perceptual_loss,
            'd_ct_loss': avg_d_ct_loss,
            'd_mri_loss': avg_d_mri_loss,
            **avg_metrics
        }
    
    def _check_gradient_explosion(self) -> bool:
        """Kiểm tra gradient explosion"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                if grad_norm > 100.0:  # Threshold cho gradient explosion
                    return True
        return False
    
    def _reset_learning_rate(self):
        """Đặt lại learning rate khi gặp gradient explosion"""
        for g in self.optimizer_G.param_groups:
            g['lr'] *= 0.5  # Giảm learning rate một nửa
        for g in self.optimizer_D.param_groups:
            g['lr'] *= 0.5
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Validation một epoch với chi tiết loss components
        """
        self.model.eval()
        
        # Validation metrics
        total_g_loss = 0.0
        total_d_loss = 0.0
        
        # Chi tiết validation loss components  
        total_val_adv_loss = 0.0
        total_val_cycle_loss = 0.0
        total_val_perceptual_loss = 0.0
        
        total_metrics = {
            'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 
            'psnr': 0.0, 'ssim': 0.0, 'ncc': 0.0
        }
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                real_mri = batch['mri'].to(self.device)
                real_ct = batch['ct'].to(self.device)
                
                # Forward pass
                outputs = self.model(real_mri, real_ct)
                
                # Generator loss (không cần discriminator loss trong validation)
                g_losses = self.model.generator_loss(real_mri, real_ct, outputs)
                
                # Accumulate losses - tổng
                total_g_loss += g_losses['total'].item()
                
                # Accumulate validation loss components
                total_val_adv_loss += g_losses['gan'].item()
                total_val_cycle_loss += g_losses['cycle'].item()
                total_val_perceptual_loss += g_losses['perceptual'].item()
                
                # Compute metrics cho fake_ct
                metrics_calculator = MetricsCalculator()
                batch_metrics = metrics_calculator.calculate_all_metrics(
                    outputs['fake_ct'],  # Truyền tensor trực tiếp
                    real_ct              # Truyền tensor trực tiếp
                )
                
                # Accumulate metrics với key mapping
                metric_key_mapping = {
                    'mae': 'MAE',
                    'mse': 'MSE', 
                    'rmse': 'RMSE',
                    'psnr': 'PSNR',
                    'ssim': 'SSIM',
                    'ncc': 'NCC'
                }
                for key in total_metrics:
                    if metric_key_mapping[key] in batch_metrics:
                        total_metrics[key] += batch_metrics[metric_key_mapping[key]]
        
        # Average losses và metrics
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = 0.0  # Không có D loss trong validation
        
        # Average validation loss components
        avg_val_adv_loss = total_val_adv_loss / num_batches
        avg_val_cycle_loss = total_val_cycle_loss / num_batches
        avg_val_perceptual_loss = total_val_perceptual_loss / num_batches
        
        avg_metrics = {}
        for key in total_metrics:
            avg_metrics[key] = total_metrics[key] / num_batches
        
        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'val_adversarial_loss': avg_val_adv_loss,
            'val_cycle_loss': avg_val_cycle_loss,
            'val_perceptual_loss': avg_val_perceptual_loss,
            **avg_metrics
        }
    
    def log_to_tensorboard(self, 
                          train_losses: Dict[str, float],
                          val_losses: Dict[str, float]):
        """
        Ghi log vào tensorboard với chi tiết loss components
        """
        # Training losses - tổng
        self.writer.add_scalar('Loss/Train_G', train_losses['generator_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Train_D', train_losses['discriminator_loss'], self.current_epoch)
        
        # Training losses - chi tiết components
        self.writer.add_scalar('Loss_Components/Adversarial_Loss', train_losses['adversarial_loss'], self.current_epoch)
        self.writer.add_scalar('Loss_Components/Cycle_Loss', train_losses['cycle_loss'], self.current_epoch)
        self.writer.add_scalar('Loss_Components/Perceptual_Loss', train_losses['perceptual_loss'], self.current_epoch)
        self.writer.add_scalar('Loss_Components/D_CT_Loss', train_losses['d_ct_loss'], self.current_epoch)
        self.writer.add_scalar('Loss_Components/D_MRI_Loss', train_losses['d_mri_loss'], self.current_epoch)
        
        # Validation losses - tổng
        self.writer.add_scalar('Loss/Val_G', val_losses['g_loss'], self.current_epoch)
        self.writer.add_scalar('Loss/Val_D', val_losses['d_loss'], self.current_epoch)
        
        # Validation losses - chi tiết components
        self.writer.add_scalar('Loss_Components/Val_Adversarial_Loss', val_losses['val_adversarial_loss'], self.current_epoch)
        self.writer.add_scalar('Loss_Components/Val_Cycle_Loss', val_losses['val_cycle_loss'], self.current_epoch)
        self.writer.add_scalar('Loss_Components/Val_Perceptual_Loss', val_losses['val_perceptual_loss'], self.current_epoch)
        
        # Training metrics
        for metric_name in ['mae', 'mse', 'rmse', 'psnr', 'ssim', 'ncc']:
            if metric_name in train_losses:
                self.writer.add_scalar(f'Metrics/Train_{metric_name.upper()}', train_losses[metric_name], self.current_epoch)
        
        # Validation metrics
        for metric_name in ['mae', 'mse', 'rmse', 'psnr', 'ssim', 'ncc']:
            if metric_name in val_losses:
                self.writer.add_scalar(f'Metrics/Val_{metric_name.upper()}', val_losses[metric_name], self.current_epoch)
    
    def save_model(self, is_best: bool = False):
        """
        Lưu model checkpoint
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        
        # Lưu checkpoint thường xuyên
        save_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint, save_path)
        
        # Lưu best model
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Đã lưu best model với SSIM: {self.best_ssim:.4f}")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop với early stopping và learning rate monitoring
        """
        print(f"🚀 Bắt đầu training từ epoch {self.current_epoch + 1}")
        print(f"📊 Initial learning rates - G: {self.config['lr_G']:.6f}, D: {self.config['lr_D']:.6f}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            epoch_start_time = time.time()
            
            # EMERGENCY WARMUP: Gradually increase LR trong epochs đầu
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                current_lr_G = self.initial_lr_G * warmup_factor * 0.1  # Start từ 10% initial LR
                current_lr_D = self.initial_lr_D * warmup_factor * 0.1
                
                # Set warmed-up learning rates
                for g in self.optimizer_G.param_groups:
                    g['lr'] = current_lr_G
                for g in self.optimizer_D.param_groups:
                    g['lr'] = current_lr_D
                    
                print(f"🔥 WARMUP Epoch {epoch+1}/{self.warmup_epochs}: LR_G={current_lr_G:.8f}, LR_D={current_lr_D:.8f}")
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Log current learning rates
            current_lr = self.get_current_lr()
            
            # Print epoch results với learning rate info
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - Time: {epoch_time:.2f}s")
            print(f"LR_G: {current_lr['lr_G']:.6f} | LR_D: {current_lr['lr_D']:.6f}")
            print(f"Train Loss: {train_losses['generator_loss'] + train_losses['discriminator_loss']:.4f} | Val Loss: {val_losses['g_loss'] + val_losses['d_loss']:.4f}")
            
            # Print chi tiết loss components
            print("\nLoss Components Details:")
            print("=" * 50)
            print(f"Generator Loss Total: {train_losses['generator_loss']:.4f}")
            print(f"  ├── Adversarial Loss: {train_losses['adversarial_loss']:.4f}")
            print(f"  ├── Cycle Loss:      {train_losses['cycle_loss']:.4f}")
            print(f"  └── Perceptual Loss: {train_losses['perceptual_loss']:.4f}")
            print(f"Discriminator Loss Total: {train_losses['discriminator_loss']:.4f}")
            print(f"  ├── D_CT Loss:       {train_losses['d_ct_loss']:.4f}")
            print(f"  └── D_MRI Loss:      {train_losses['d_mri_loss']:.4f}")
            print("=" * 50)
            
            # Print detailed metrics
            print("\nTrain Metrics:")
            print("-" * 50)
            print(f"MAE: {train_losses['mae']:.6f}")
            print(f"MSE: {train_losses['mse']:.6f}") 
            print(f"RMSE: {train_losses['rmse']:.6f}")
            print(f"PSNR: {train_losses['psnr']:.4f} dB")
            print(f"SSIM: {train_losses['ssim']:.4f}")
            print(f"NCC: {train_losses['ncc']:.4f}")
            print("-" * 50)
            
            print("\nValidation Metrics:")
            print("-" * 50)
            print(f"MAE: {val_losses['mae']:.6f}")
            print(f"MSE: {val_losses['mse']:.6f}")
            print(f"RMSE: {val_losses['rmse']:.6f}")
            print(f"PSNR: {val_losses['psnr']:.4f} dB")
            print(f"SSIM: {val_losses['ssim']:.4f}")
            print(f"NCC: {val_losses['ncc']:.4f}")
            print("-" * 50)
            
            # Log to tensorboard với learning rates
            self.log_to_tensorboard(train_losses, val_losses)
            self.writer.add_scalar('Learning_Rate/Generator', current_lr['lr_G'], epoch)
            self.writer.add_scalar('Learning_Rate/Discriminator', current_lr['lr_D'], epoch)
            
            # Early stopping based on SSIM
            current_ssim = val_losses['ssim']
            
            # Kiểm tra cải thiện
            if current_ssim > self.best_ssim:
                self.best_ssim = current_ssim
                self.epochs_without_improvement = 0
                self.save_model(is_best=True)
                print(f"🎉 New best SSIM: {self.best_ssim:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"⏳ Epochs without improvement: {self.epochs_without_improvement}/{self.max_patience}")
            
            # Kiểm tra SSIM collapse (giảm quá nhanh)
            if epoch > 50 and current_ssim < 0.1:  # SSIM quá thấp
                print(f"⚠️  SSIM collapse detected! Current: {current_ssim:.4f}")
                print("🔄 Reducing learning rate dramatically...")
                
                # Giảm learning rate rất mạnh
                for g in self.optimizer_G.param_groups:
                    g['lr'] *= 0.1
                for g in self.optimizer_D.param_groups:
                    g['lr'] *= 0.1
                
                self.epochs_without_improvement = 0  # Reset patience
            
            # Early stopping
            if self.epochs_without_improvement >= self.max_patience:
                print(f"\n🛑 Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                print(f"🏆 Best SSIM achieved: {self.best_ssim:.4f}")
                break
            
            # Cập nhật current_epoch sau khi hoàn thành epoch
            self.current_epoch = epoch + 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_model()
            
            # Tạo sample images
            if (epoch + 1) % self.config['sample_freq'] == 0:
                self.create_sample_images(val_loader)
        
        total_time = time.time() - start_time
        print(f"\nTraining hoàn thành! Tổng thời gian: {total_time/3600:.2f} giờ")
        print(f"Best SSIM: {self.best_ssim:.4f}")
        
        self.writer.close()
    
    def create_sample_images(self, val_loader):
        """
        Tạo ảnh mẫu để kiểm tra kết quả
        """
        self.model.eval()
        
        with torch.no_grad():
            # Lấy batch đầu tiên từ validation set
            batch = next(iter(val_loader))
            real_mri = batch['mri'][:4].to(self.device)  # Lấy 4 sample đầu
            real_ct = batch['ct'][:4].to(self.device)
            
            # Generate fake images
            fake_ct = self.model.G_MRI2CT(real_mri)
            fake_mri = self.model.G_CT2MRI(real_ct)
            
            # Cycle consistency
            rec_mri = self.model.G_CT2MRI(fake_ct)
            rec_ct = self.model.G_MRI2CT(fake_mri)
            
            # Save sample images
            save_dir = os.path.join(self.config['sample_dir'], f'epoch_{self.current_epoch}')
            os.makedirs(save_dir, exist_ok=True)
            
            # Lưu từng loại ảnh
            self._save_image_batch(real_mri, os.path.join(save_dir, 'real_mri.png'))
            self._save_image_batch(real_ct, os.path.join(save_dir, 'real_ct.png'))
            self._save_image_batch(fake_ct, os.path.join(save_dir, 'fake_ct.png'))
            self._save_image_batch(fake_mri, os.path.join(save_dir, 'fake_mri.png'))
            self._save_image_batch(rec_mri, os.path.join(save_dir, 'rec_mri.png'))
            self._save_image_batch(rec_ct, os.path.join(save_dir, 'rec_ct.png'))
    
    def _save_image_batch(self, images, save_path):
        """
        Lưu batch ảnh thành file
        """
        import matplotlib.pyplot as plt
        
        # Chuyển về numpy
        images_np = images.detach().cpu().numpy()
        batch_size = images_np.shape[0]
        
        # Tạo subplot dựa trên batch size thực tế
        fig, axes = plt.subplots(1, min(4, batch_size), figsize=(16, 4))
        
        # Nếu chỉ có 1 image, axes không phải là array
        if batch_size == 1:
            axes = [axes]
        
        for i in range(min(4, batch_size)):
            img = images_np[i, 0]  # Lấy channel đầu tiên
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """
    Hàm main để chạy training
    """
    # Cấu hình training tối ưu cho GTX 1650 (4GB VRAM) với learning rate thấp hơn
    config = {
        # Data parameters  
        'cache_dir': '../preprocessed_cache',  # Sử dụng cached data
        'batch_size': 4,          # Batch size nhỏ
        'num_workers': 2,         # Tăng workers vì chỉ load cache
        'train_split': 0.8,
        'augmentation_prob': 0.6, # Xác suất augmentation
        
        # Model parameters - Tối ưu với cached data
        'input_nc': 1,
        'output_nc': 1,
        'n_residual_blocks': 9,   
        'discriminator_layers': 3,
        
        # Training parameters - EMERGENCY ULTRA CONSERVATIVE
        # Medical imaging GANs thường dùng LR thấp hơn để tránh mode collapse
        # EMERGENCY: Giảm LR xuống cực thấp để hoàn toàn prevent gradient explosion
        'num_epochs': 150,        # Tăng epochs do medical data cần convergence từ từ
        'lr_G': 0.000005,        # EMERGENCY: 0.00002 → 0.000005 (4x thấp hơn nữa!)
        'lr_D': 0.000002,        # EMERGENCY: 0.00001 → 0.000002 (5x thấp hơn nữa!)
        'beta1': 0.9,            # FIX SSIM PLATEAU: Tăng từ 0.5 → 0.9 (medical imaging needs higher momentum)
        'beta2': 0.99,            # FIX SSIM PLATEAU: Giảm từ 0.999 → 0.9 (prevent second-moment accumulation)
        'decay_epoch': 75,        # Bắt đầu decay tại epoch 75 (50% của 150 epochs)
        'decay_epochs': 75,       # Decay trong 75 epochs cuối
        
        # Directories
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'sample_dir': 'samples',
        
        # Save frequencies - Với cached data training nhanh hơn
        'save_freq': 1,           # Save mỗi 5 epochs
        'sample_freq': 1          # Sample mỗi 5 epochs
    }
    
    # Tạo thư mục cần thiết
    for dir_name in ['checkpoint_dir', 'log_dir', 'sample_dir']:
        os.makedirs(config[dir_name], exist_ok=True)
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng device: {device}")
    
    # Tìm và load checkpoint nếu có
    resume_training = True
    checkpoint_path = None
    
    if os.path.exists(config['checkpoint_dir']):
        # Kiểm tra xem có checkpoint nào không
        checkpoint_files = [f for f in os.listdir(config['checkpoint_dir']) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if checkpoint_files:
            # Sắp xếp checkpoint files theo epoch number
            def extract_epoch(filename):
                try:
                    return int(filename.split('_')[2].split('.pth')[0])
                except:
                    return -1
            
            sorted_checkpoints = sorted(checkpoint_files, key=extract_epoch)
            latest_checkpoint_file = sorted_checkpoints[-1]
            latest_epoch = extract_epoch(latest_checkpoint_file)
            
            print(f"\n🔍 Tìm thấy {len(checkpoint_files)} checkpoints:")
            # Hiển thị 3 checkpoint gần nhất với epoch numbers
            for f in sorted_checkpoints[-3:]:
                epoch_num = extract_epoch(f)
                print(f"   - {f} (epoch {epoch_num})")
            
            print(f"📁 Latest checkpoint: {latest_checkpoint_file} (epoch {latest_epoch})")
            
            choice = input("\n❓ Bạn có muốn tiếp tục training từ checkpoint gần nhất? (y/n): ").lower().strip()
            resume_training = choice in ['y', 'yes', '1', 'true', '']
            
            if resume_training:
                checkpoint_path = os.path.join(config['checkpoint_dir'], latest_checkpoint_file)
            else:
                print("⚠️  Sẽ bắt đầu training từ đầu (checkpoint cũ sẽ không bị xóa)")
                resume_training = False
    
    # Kiểm tra cache tồn tại
    if not os.path.exists(config['cache_dir']):
        print(f"❌ Cache directory not found: {config['cache_dir']}")
        print("   Run: python preprocess_and_cache.py first!")
        return
    
    # Hỏi user chọn loading strategy
    print("\n🤔 Chọn data loading strategy:")
    print("   1. VOLUME-BASED (original): 42 samples/epoch, ~1.5s/epoch (giống data_loader.py cũ)")
    print("   2. MULTI-SLICE (recommended): 42×N slices/epoch, tăng data diversity!")
    print("   3. SLICE-BASED optimized: ~1,260 slices/epoch, ~1.4 phút/epoch")
    print("   4. SLICE-BASED full: ~4,681 slices/epoch, ~5.4 phút/epoch")
    
    choice = input("❓ Chọn strategy (1/2/3/4): ").strip()
    
    if choice == "1" or choice == "":
        loading_strategy = "volume"
    elif choice == "2":
        loading_strategy = "multi_slice"
    elif choice == "3":
        loading_strategy = "slice_optimized"
    elif choice == "4":
        loading_strategy = "slice_full"
    else:
        print("Invalid choice, using volume-based (default)")
        loading_strategy = "volume"
    
        # Tạo cached data loaders theo strategy đã chọn
    if loading_strategy == "volume":
        print("\n🚀 Đang tạo VOLUME-BASED cached data loaders...")
        loader_manager = VolumeCachedDataLoaderManager(config['cache_dir'])
        
        train_loader, val_loader = loader_manager.create_train_val_loaders(
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers'],
            augmentation_prob=config['augmentation_prob']
        )
        
    elif loading_strategy == "multi_slice":
        # Hỏi số slices per patient theo analysis cho SSIM 90%
        print("\n🎯 Chọn số slices per patient (Based on SSIM Analysis):")
        print("   10: Baseline (330 samples/epoch) → Expected SSIM 0.68")
        print("   20: Phase 1 (660 samples/epoch) → Expected SSIM 0.75")  
        print("   50: Phase 2 (1650 samples/epoch) → Expected SSIM 0.87")
        print("   80: Phase 3 (2640 samples/epoch) → Expected SSIM 0.92+ ⭐")
        print("   100: Maximum (3300 samples/epoch) → Expected SSIM 0.95")
        
        slice_choice = input("❓ Chọn số slices (10/20/50/80/100): ").strip()
        
        if slice_choice == "10":
            slices_per_patient = 10
        elif slice_choice == "20":
            slices_per_patient = 20
        elif slice_choice == "50":
            slices_per_patient = 50
        elif slice_choice == "80":
            slices_per_patient = 80
        elif slice_choice == "100":
            slices_per_patient = 100
        else:
            slices_per_patient = 20  # Default to Phase 1
        
        # Điều chỉnh learning rate theo slice count
        original_lr_g = config['lr_G']
        original_lr_d = config['lr_D']
        
        if slices_per_patient >= 80:
            # Very high data - cần LR rất thấp
            config['lr_G'] = 0.00004
            config['lr_D'] = 0.00004
            print(f"   🔧 Adjusted LR to {config['lr_G']} (Very High Data)")
        elif slices_per_patient >= 50:
            # High data - LR thấp
            config['lr_G'] = 0.00005
            config['lr_D'] = 0.00005
            print(f"   🔧 Adjusted LR to {config['lr_G']} (High Data)")
        elif slices_per_patient >= 20:
            # Moderate data - LR moderate
            config['lr_G'] = 0.00008
            config['lr_D'] = 0.00008
            print(f"   🔧 Adjusted LR to {config['lr_G']} (Moderate Data)")
        else:
            # Low data - keep default
            print(f"   🔧 Using default LR {config['lr_G']} (Low Data)")
        
        # Điều chỉnh batch size cho high slice counts
        original_batch_size = config['batch_size']
        if slices_per_patient >= 50:
            # Giảm batch size để fit memory với nhiều data
            config['batch_size'] = 4
            print(f"   🔧 Adjusted batch size to {config['batch_size']} (High Data Volume)")
        
        print(f"\n🚀 Đang tạo MULTI-SLICE cached data loaders với {slices_per_patient} slices/patient...")
        loader_manager = MultiSliceDataLoaderManager(config['cache_dir'])
        
        # Hiển thị statistics
        stats = loader_manager.get_data_statistics(slices_per_patient)
        print(f"\n📊 Data Statistics:")
        print(f"   Samples per epoch: {stats['samples_per_epoch']}")
        print(f"   Data utilization: {stats['data_utilization_percent']:.1f}%")
        print(f"   Improvement vs volume-based: {stats['improvement_vs_volume']}")
        
        train_loader, val_loader = loader_manager.create_train_val_loaders(
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers'],
            slices_per_patient=slices_per_patient,
            augmentation_prob=config['augmentation_prob']
        )
        
        print(f"✅ Multi-slice loaders created!")
        print(f"   Training batches/epoch: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        print(f"   Estimated time/epoch: ~{len(train_loader)*5/60:.1f} minutes")
        
        if slices_per_patient >= 80:
            print(f"\n🎯 TARGET: SSIM 90%+ với {slices_per_patient} slices!")
            print(f"   ⚠️  High training time but expected breakthrough performance")
            print(f"   📈 Progressive strategy recommended if first time")
        
    elif loading_strategy == "slice_optimized":
        print("\n🚀 Đang tạo OPTIMIZED slice-based data loaders...")
        loader_manager = OptimizedDataLoaderManager(config['cache_dir'])
        
        train_loader, val_loader = loader_manager.create_fast_train_val_loaders(
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers'],
            slice_sampling_strategy="every_nth",  # Lấy 60% slices ở giữa
            max_slices_per_patient=30,              # Tối đa 30 slices/bệnh nhân
            augmentation_prob=config['augmentation_prob']
        )
        
    else:  # slice_full
        print("\n📊 Đang tạo FULL slice-based cached data loaders...")
        loader_manager = CachedDataLoaderManager(config['cache_dir'])
        
        # In thông tin cache cho standard loader
        cache_info = loader_manager.get_cache_info()
        print(f"📦 Cache info:")
        print(f"   Total patients: {cache_info['total_patients']}")
        print(f"   Total slices: {cache_info['total_slices']}")
        print(f"   Cache size: {cache_info['cache_size_mb']:.1f} MB")
        print(f"   Save format: {cache_info['save_format']}")
        print(f"🚀 Training sẽ nhanh hơn ~450x so với preprocessing realtime!")
        
        train_loader, val_loader = loader_manager.create_train_val_loaders(
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers'],
            augmentation_prob=config['augmentation_prob']
        )
    
    # Khởi tạo trainer
    trainer = CycleGANTrainer(config, device, resume_from_checkpoint=resume_training, checkpoint_path=checkpoint_path)
    
    # CRITICAL FIX: Reset learning rates nếu đã adjust cho multi-slice
    if loading_strategy == "multi_slice" and resume_training:
        # Đặt lại learning rate sau khi load checkpoint để match config
        current_lr_g = config['lr_G']
        current_lr_d = config['lr_D']
        
        for param_group in trainer.optimizer_G.param_groups:
            param_group['lr'] = current_lr_g
        for param_group in trainer.optimizer_D.param_groups:
            param_group['lr'] = current_lr_d
            
        print(f"🔧 Reset learning rates after checkpoint loading:")
        print(f"   LR_G: {current_lr_g}")
        print(f"   LR_D: {current_lr_d}")
    
    # Debug training state
    print(f"🚀 TRAINING STATE DEBUG:")
    print(f"   - Current epoch (từ checkpoint): {trainer.current_epoch}")
    print(f"   - Sẽ bắt đầu training từ epoch: {trainer.current_epoch + 1}")
    print(f"   - Target epochs: {config['num_epochs']}")
    
    # IMPORTANT: Đặt lại current_epoch về 0-indexed để training loop chạy đúng
    if trainer.current_epoch > 0:
        trainer.current_epoch -= 1
    
    # Hiển thị initial learning rates
    initial_lr = trainer.get_current_lr()
    print(f"📊 Initial learning rates - G: {initial_lr['lr_G']:.6f}, D: {initial_lr['lr_D']:.6f}")
    
    # Bắt đầu training với cached data
    if loading_strategy == "volume":
        print(f"\n🚀 Bắt đầu VOLUME-BASED training...")
        print(f"   - Giống hệt data_loader.py cũ: {len(train_loader.dataset)} samples/epoch")
        print(f"   - Nhưng preprocessing đã cache sẵn → nhanh hơn ~4,784x!")
        print(f"   - Random slice selection mỗi epoch")
        print(f"   - Estimated training time: ~{len(train_loader) * 0.167:.1f}s/epoch")
    else:
        print(f"\n🚀 Bắt đầu SLICE-BASED training với cached data...")
        print(f"   - {len(train_loader.dataset)} samples/epoch")
        print(f"   - Preprocessing đã được cache trước")
        print(f"   - Mỗi epoch chỉ cần load cache + augmentation")
    
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main() 