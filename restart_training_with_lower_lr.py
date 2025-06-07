#!/usr/bin/env python3
"""
Script để restart training từ checkpoint với learning rate thấp hơn
Dùng khi gặp vấn đề SSIM collapse hoặc training instability
"""

import os
import torch
import glob
from src.train import CycleGANTrainer
from src.volume_based_cached_loader import VolumeCachedDataLoaderManager

def find_latest_checkpoint(checkpoint_dir):
    """Tìm checkpoint mới nhất"""
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    # Sắp xếp theo epoch number
    def extract_epoch(filename):
        return int(filename.split('_epoch_')[1].split('.pth')[0])
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return latest_checkpoint, extract_epoch(latest_checkpoint)

def restart_training_with_lower_lr():
    """
    Restart training với learning rate thấp hơn để phục hồi từ SSIM collapse
    """
    
    print("🔄 RESTART TRAINING WITH LOWER LEARNING RATE")
    print("=" * 60)
    
    # Cấu hình với learning rate RẤT THẤP để phục hồi
    config = {
        # Data parameters  
        'cache_dir': '../preprocessed_cache',
        'batch_size': 2,          # Giảm batch size xuống 2 để ổn định hơn
        'num_workers': 2,
        'train_split': 0.8,
        'augmentation_prob': 0.5, # Giảm augmentation để ổn định
        
        # Model parameters
        'input_nc': 1,
        'output_nc': 1,
        'n_residual_blocks': 9,   
        'discriminator_layers': 3,
        
        # Training parameters - LEARNING RATE RẤT THẤP
        'num_epochs': 300,        # Tăng epochs vì LR thấp
        'lr_G': 0.00001,         # Giảm 10x: 0.0001 -> 0.00001
        'lr_D': 0.00001,         # Giảm 10x: 0.0001 -> 0.00001
        'beta1': 0.5,
        'beta2': 0.999,
        'decay_epoch': 100,
        'decay_epochs': 100,
        
        # Directories
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs_restart',  # Log riêng để không bị conflict
        'sample_dir': 'samples_restart',
        
        # Save frequencies
        'save_freq': 2,           # Save thường xuyên hơn
        'sample_freq': 5
    }
    
    # Tạo thư mục
    for dir_name in ['checkpoint_dir', 'log_dir', 'sample_dir']:
        os.makedirs(config[dir_name], exist_ok=True)
    
    # Tìm checkpoint gần nhất
    latest_checkpoint, latest_epoch = find_latest_checkpoint(config['checkpoint_dir'])
    
    if not latest_checkpoint:
        print("❌ Không tìm thấy checkpoint nào để restart!")
        print("   Chạy training bình thường trước.")
        return
    
    print(f"📂 Found latest checkpoint: {latest_checkpoint}")
    print(f"📊 Latest epoch: {latest_epoch}")
    print(f"🔻 New learning rates: G={config['lr_G']:.6f}, D={config['lr_D']:.6f}")
    
    # Xác nhận từ user
    confirm = input(f"\n❓ Restart training từ epoch {latest_epoch} với LR thấp hơn? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes', '']:
        print("❌ Hủy restart training.")
        return
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Sử dụng device: {device}")
    
    # Tạo data loaders
    print("📦 Đang tạo data loaders...")
    loader_manager = VolumeCachedDataLoaderManager(config['cache_dir'])
    
    train_loader, val_loader = loader_manager.create_train_val_loaders(
        batch_size=config['batch_size'],
        train_split=config['train_split'],
        num_workers=config['num_workers'],
        augmentation_prob=config['augmentation_prob']
    )
    
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Tạo trainer và load checkpoint
    print("🏗️  Đang khởi tạo trainer...")
    trainer = CycleGANTrainer(
        config=config,
        device=device,
        resume_from_checkpoint=True  # Sẽ tự động load checkpoint
    )
    
    # Thay đổi learning rate sau khi load checkpoint
    print("🔄 Đang thay đổi learning rate...")
    
    # Update learning rate cho cả optimizer và scheduler
    for param_group in trainer.optimizer_G.param_groups:
        param_group['lr'] = config['lr_G']
    for param_group in trainer.optimizer_D.param_groups:
        param_group['lr'] = config['lr_D']
    
    # Reset schedulers với learning rate mới
    trainer.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer_G, 
        T_0=30,      # Cycle dài hơn vì LR thấp
        T_mult=2,
        eta_min=config['lr_G'] * 0.1  # Min LR = 10% của LR mới
    )
    
    trainer.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        trainer.optimizer_D,
        T_0=30,
        T_mult=2, 
        eta_min=config['lr_D'] * 0.1
    )
    
    # Reset early stopping counter
    trainer.epochs_without_improvement = 0
    trainer.max_patience = 20  # Tăng patience vì LR thấp
    
    current_lr = trainer.get_current_lr()
    print(f"✅ Learning rates updated: G={current_lr['lr_G']:.6f}, D={current_lr['lr_D']:.6f}")
    
    print("\n🚀 BẮT ĐẦU RESTART TRAINING...")
    print(f"📊 Current epoch: {trainer.current_epoch}")
    print(f"🏆 Best SSIM so far: {trainer.best_ssim:.4f}")
    print(f"⏱️  Max patience: {trainer.max_patience}")
    print("=" * 60)
    
    # Bắt đầu training
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        raise
    finally:
        print("\n🔚 Training session ended")

if __name__ == "__main__":
    restart_training_with_lower_lr() 