#!/usr/bin/env python3
"""
Training với Multi-Slice Data Loader
Tăng data diversity để cải thiện SSIM > 0.5
"""

import os
import sys
sys.path.append('src')

from src.train import CycleGANTrainer
from src.multi_slice_cached_loader import MultiSliceDataLoaderManager

def train_with_multi_slice():
    """
    Training với multi-slice data loader để vượt qua SSIM plateau
    """
    
    print("🚀 MULTI-SLICE CYCLEGAN TRAINING")
    print("=" * 60)
    print("🎯 Mục tiêu: Vượt qua SSIM plateau 0.5 bằng data diversity")
    print("📊 Phương pháp: Tăng slices per patient từ 1 → 10")
    
    # Cấu hình tối ưu cho multi-slice training
    config = {
        # Data parameters  
        'cache_dir': 'preprocessed_cache',
        'batch_size': 2,          # Giảm vì có nhiều data hơn
        'num_workers': 2,
        'train_split': 0.8,
        'augmentation_prob': 0.7, # Giảm augmentation vì đã có nhiều data
        
        # Model parameters
        'input_nc': 1,
        'output_nc': 1,
        'n_residual_blocks': 9,   
        'discriminator_layers': 3,
        
        # Training parameters - LR thấp cho stability
        'num_epochs': 100,        
        'lr_G': 0.00005,         # Will be adjusted based on slice count
        'lr_D': 0.00005,          
        'beta1': 0.5,
        'beta2': 0.999,
        
        # Directories
        'checkpoint_dir': 'src/checkpoints_multi_slice',
        'log_dir': 'src/logs_multi_slice',
        'sample_dir': 'src/samples_multi_slice',
        
        # Save frequencies
        'save_freq': 5,           
        'sample_freq': 10
    }
    
    # Tạo thư mục
    for dir_name in ['checkpoint_dir', 'log_dir', 'sample_dir']:
        os.makedirs(config[dir_name], exist_ok=True)
    
    # Hỏi user chọn số slices theo analysis
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
    
    print(f"\n🚀 Selected: {slices_per_patient} slices per patient")
    
    # Điều chỉnh learning rate theo slice count
    if slices_per_patient >= 80:
        # Very high data - cần LR rất thấp
        config['lr_G'] = 0.000025
        config['lr_D'] = 0.000025
        print(f"   🔧 Adjusted LR to {config['lr_G']} (Very High Data)")
    elif slices_per_patient >= 50:
        # High data - LR thấp
        config['lr_G'] = 0.00003
        config['lr_D'] = 0.00003
        print(f"   🔧 Adjusted LR to {config['lr_G']} (High Data)")
    elif slices_per_patient >= 20:
        # Moderate data - LR moderate
        config['lr_G'] = 0.00004
        config['lr_D'] = 0.00004
        print(f"   🔧 Adjusted LR to {config['lr_G']} (Moderate Data)")
    else:
        # Low data - keep default
        print(f"   🔧 Using default LR {config['lr_G']} (Low Data)")
    
    # Kiểm tra cache tồn tại
    if not os.path.exists(config['cache_dir']):
        print(f"❌ Cache directory not found: {config['cache_dir']}")
        print("   Run: python preprocess_and_cache.py first!")
        return
    
    # Kiểm tra GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    # Tạo multi-slice data loaders
    print(f"\n📦 Đang tạo MULTI-SLICE data loaders với {slices_per_patient} slices/patient...")
    loader_manager = MultiSliceDataLoaderManager(config['cache_dir'])
    
    # Hiển thị statistics
    stats = loader_manager.get_data_statistics(slices_per_patient)
    print(f"\n📊 Data Statistics:")
    print(f"   Samples per epoch: {stats['samples_per_epoch']}")
    print(f"   Data utilization: {stats['data_utilization_percent']:.1f}%")
    print(f"   Improvement vs volume-based: {stats['improvement_vs_volume']}")
    print(f"   Improvement vs original: {stats['improvement_vs_original']}")
    
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
    
    # Tạo trainer
    print("\n🏗️  Đang khởi tạo CycleGAN trainer...")
    trainer = CycleGANTrainer(
        config=config,
        device=device,
        resume_from_checkpoint=True
    )
    
    print(f"\n🎯 TRAINING COMPARISON:")
    print(f"   Previous (volume-based): ~11 batches/epoch → SSIM plateau ~0.54")
    print(f"   Current (multi-slice): ~{len(train_loader)} batches/epoch → Expected SSIM > 0.6")
    print(f"   Data increase: {len(train_loader)/11:.1f}x more samples!")
    
    confirm = input(f"\n❓ Bắt đầu multi-slice training? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes', '']:
        print("❌ Training cancelled.")
        return
    
    print("\n🚀 BẮT ĐẦU MULTI-SLICE TRAINING...")
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
        print("\n🔚 Multi-slice training session ended")


if __name__ == "__main__":
    import torch
    train_with_multi_slice() 