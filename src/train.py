import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple

from data_loader import create_data_loaders
from models import CycleGAN, weights_init_normal
from metrics import MetricsTracker, print_metrics
from utils import save_checkpoint, load_checkpoint, create_sample_images

class CycleGANTrainer:
    """
    Class training cho CycleGAN
    """
    
    def __init__(self, 
                 config: Dict,
                 device: str = 'cuda'):
        """
        Args:
            config: dictionary chứa các tham số training
            device: thiết bị training (cuda/cpu)
        """
        self.config = config
        self.device = device
        
        # Khởi tạo model
        self.model = CycleGAN(
            input_nc=config['input_nc'],
            output_nc=config['output_nc'],
            n_residual_blocks=config['n_residual_blocks'],
            discriminator_layers=config['discriminator_layers']
        ).to(device)
        
        # Khởi tạo trọng số
        self.model.apply(weights_init_normal)
        
        # Optimizers
        self.optimizer_G = optim.Adam(
            list(self.model.G_MRI2CT.parameters()) + list(self.model.G_CT2MRI.parameters()),
            lr=config['lr_G'],
            betas=(config['beta1'], config['beta2'])
        )
        
        self.optimizer_D = optim.Adam(
            list(self.model.D_CT.parameters()) + list(self.model.D_MRI.parameters()),
            lr=config['lr_D'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config['decay_epoch']) / config['decay_epochs']
        )
        
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optimizer_D,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config['decay_epoch']) / config['decay_epochs']
        )
        
        # Metrics tracker
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Tensorboard writer
        self.writer = SummaryWriter(config['log_dir'])
        
        # Training state
        self.current_epoch = 0
        self.best_ssim = 0.0
        
        print(f"Model khởi tạo thành công trên {device}")
        print(f"Tổng số parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Training một epoch
        """
        self.model.train()
        self.train_metrics.reset_epoch()
        
        epoch_losses = {
            'G_total': 0.0,
            'G_gan': 0.0,
            'G_cycle': 0.0,
            'G_identity': 0.0,
            'G_perceptual': 0.0,
            'D_CT': 0.0,
            'D_MRI': 0.0
        }
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            real_mri = batch['mri'].to(self.device)
            real_ct = batch['ct'].to(self.device)
            
            # =====================================
            # Train Generators
            # =====================================
            self.optimizer_G.zero_grad()
            
            # Forward pass
            outputs = self.model(real_mri, real_ct)
            
            # Generator loss
            g_losses = self.model.generator_loss(real_mri, real_ct, outputs)
            g_losses['total'].backward()
            self.optimizer_G.step()
            
            # =====================================
            # Train Discriminators
            # =====================================
            self.optimizer_D.zero_grad()
            
            # CT Discriminator
            loss_D_CT = self.model.discriminator_loss(
                real_ct, outputs['fake_ct'], self.model.D_CT
            )
            
            # MRI Discriminator  
            loss_D_MRI = self.model.discriminator_loss(
                real_mri, outputs['fake_mri'], self.model.D_MRI
            )
            
            # Total discriminator loss
            loss_D_total = loss_D_CT + loss_D_MRI
            loss_D_total.backward()
            self.optimizer_D.step()
            
            # =====================================
            # Update metrics và losses
            # =====================================
            epoch_losses['G_total'] += g_losses['total'].item()
            epoch_losses['G_gan'] += g_losses['gan'].item()
            epoch_losses['G_cycle'] += g_losses['cycle'].item()
            epoch_losses['G_identity'] += g_losses['identity'].item()
            epoch_losses['G_perceptual'] += g_losses['perceptual'].item()
            epoch_losses['D_CT'] += loss_D_CT.item()
            epoch_losses['D_MRI'] += loss_D_MRI.item()
            
            # Update metrics
            self.train_metrics.update(outputs['fake_ct'], real_ct)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'G_loss': f"{g_losses['total'].item():.4f}",
                    'D_loss': f"{loss_D_total.item():.4f}",
                    'SSIM': f"{self.train_metrics.get_epoch_average().get('SSIM', 0):.4f}"
                })
        
        # Tính average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Validation một epoch
        """
        self.model.eval()
        self.val_metrics.reset_epoch()
        
        val_losses = {
            'G_total': 0.0,
            'G_gan': 0.0,
            'G_cycle': 0.0,
            'G_identity': 0.0,
            'G_perceptual': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                real_mri = batch['mri'].to(self.device)
                real_ct = batch['ct'].to(self.device)
                
                # Forward pass
                outputs = self.model(real_mri, real_ct)
                
                # Generator loss (không cần discriminator loss trong validation)
                g_losses = self.model.generator_loss(real_mri, real_ct, outputs)
                
                # Update losses
                val_losses['G_total'] += g_losses['total'].item()
                val_losses['G_gan'] += g_losses['gan'].item()
                val_losses['G_cycle'] += g_losses['cycle'].item()
                val_losses['G_identity'] += g_losses['identity'].item()
                val_losses['G_perceptual'] += g_losses['perceptual'].item()
                
                # Update metrics
                self.val_metrics.update(outputs['fake_ct'], real_ct)
        
        # Tính average losses
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def log_to_tensorboard(self, 
                          train_losses: Dict[str, float],
                          val_losses: Dict[str, float]):
        """
        Ghi log vào tensorboard
        """
        # Training losses
        for loss_name, loss_value in train_losses.items():
            self.writer.add_scalar(f'Loss/Train_{loss_name}', loss_value, self.current_epoch)
        
        # Validation losses
        for loss_name, loss_value in val_losses.items():
            self.writer.add_scalar(f'Loss/Val_{loss_name}', loss_value, self.current_epoch)
        
        # Training metrics
        train_metrics = self.train_metrics.get_epoch_average()
        for metric_name, metric_value in train_metrics.items():
            self.writer.add_scalar(f'Metrics/Train_{metric_name}', metric_value, self.current_epoch)
        
        # Validation metrics
        val_metrics = self.val_metrics.get_epoch_average()
        for metric_name, metric_value in val_metrics.items():
            self.writer.add_scalar(f'Metrics/Val_{metric_name}', metric_value, self.current_epoch)
        
        # Learning rates
        self.writer.add_scalar('LR/Generator', self.optimizer_G.param_groups[0]['lr'], self.current_epoch)
        self.writer.add_scalar('LR/Discriminator', self.optimizer_D.param_groups[0]['lr'], self.current_epoch)
    
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
        Training loop chính
        """
        print("Bắt đầu training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Get metrics
            train_metrics = self.train_metrics.get_epoch_average()
            val_metrics = self.val_metrics.get_epoch_average()
            
            # Log to tensorboard
            self.log_to_tensorboard(train_losses, val_losses)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch}/{self.config['num_epochs']} - Time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_losses['G_total']:.4f} | Val Loss: {val_losses['G_total']:.4f}")
            print_metrics(train_metrics, "Train")
            print_metrics(val_metrics, "Validation")
            
            # Check if best model
            current_ssim = val_metrics['SSIM']
            is_best = current_ssim > self.best_ssim
            if is_best:
                self.best_ssim = current_ssim
            
            # Save model
            if (epoch + 1) % self.config['save_freq'] == 0 or is_best:
                self.save_model(is_best)
            
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
        
        # Tạo subplot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        for i in range(4):
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
    # Cấu hình training
    config = {
        # Data parameters
        'mri_dir': '../data/MRI',
        'ct_dir': '../data/CT',
        'batch_size': 2,  # Giảm batch size cho CPU
        'num_workers': 1,  # Giảm workers cho CPU
        'train_split': 0.8,
        
        # Model parameters
        'input_nc': 1,
        'output_nc': 1,
        'n_residual_blocks': 9,
        'discriminator_layers': 3,
        
        # Training parameters
        'num_epochs': 200,
        'lr_G': 0.0002,
        'lr_D': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'decay_epoch': 100,
        'decay_epochs': 100,
        
        # Directories
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'sample_dir': 'samples',
        
        # Save frequencies
        'save_freq': 10,
        'sample_freq': 5
    }
    
    # Tạo thư mục cần thiết
    for dir_name in ['checkpoint_dir', 'log_dir', 'sample_dir']:
        os.makedirs(config[dir_name], exist_ok=True)
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Sử dụng device: {device}")
    
    # Tạo data loaders
    print("Đang tạo data loaders...")
    train_loader, val_loader = create_data_loaders(
        config['mri_dir'],
        config['ct_dir'],
        config['batch_size'],
        config['train_split'],
        config['num_workers']
    )
    
    # Khởi tạo trainer
    trainer = CycleGANTrainer(config, device)
    
    # Bắt đầu training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main() 