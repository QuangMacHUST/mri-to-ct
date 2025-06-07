#!/usr/bin/env python3
"""
Script chạy toàn bộ quy trình fast training:
1. Preprocessing và cache data (một lần)
2. Fast training với cached data
"""

import os
import sys
import argparse
import time
from pathlib import Path

def run_preprocessing(mri_dir: str, ct_dir: str, cache_dir: str, cache_format: str = 'h5'):
    """
    Chạy preprocessing và tạo cache
    """
    print("🔄 STEP 1: PREPROCESSING AND CACHING")
    print("=" * 50)
    
    from preprocess_and_cache import PreprocessingCacher
    
    # Tạo preprocessor
    preprocessor = PreprocessingCacher(
        mri_dir=mri_dir,
        ct_dir=ct_dir,
        cache_dir=cache_dir
    )
    
    # Chạy preprocessing
    print(f"🚀 Starting preprocessing...")
    start_time = time.time()
    
    metadata = preprocessor.preprocess_all_volumes(save_format=cache_format)
    
    preprocessing_time = time.time() - start_time
    
    print(f"\n✅ PREPROCESSING COMPLETE!")
    print(f"   Time taken: {preprocessing_time/60:.1f} minutes")
    print(f"   Cache directory: {cache_dir}")
    print(f"   Format: {cache_format}")
    print(f"   Patients processed: {len(metadata)}")
    print(f"   Total slices: {sum(m['num_slices'] for m in metadata)}")
    
    return metadata, preprocessing_time

def run_training(cache_dir: str, output_dir: str, training_args: dict):
    """
    Chạy fast training với cached data
    """
    print("\n🚀 STEP 2: FAST TRAINING")
    print("=" * 50)
    
    from train_with_cache import FastTrainer
    
    # Tạo trainer
    trainer = FastTrainer(
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    
    # Chạy training
    print(f"🏋️ Starting fast training...")
    start_time = time.time()
    
    results = trainer.train(**training_args)
    
    training_time = time.time() - start_time
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Time taken: {training_time/3600:.1f} hours")
    print(f"   Best validation loss: {results['best_val_loss']:.4f}")
    print(f"   Average epoch time: {results['total_time']/training_args['num_epochs']:.1f}s")
    
    return results, training_time

def estimate_savings(preprocessing_time: float, training_time: float, num_epochs: int):
    """
    Ước tính thời gian tiết kiệm được
    """
    # Thời gian preprocessing trong mỗi epoch nếu không cache
    preprocessing_per_epoch = preprocessing_time  # Full preprocessing mỗi epoch
    
    # Thời gian training thực tế (chỉ augmentation)
    training_per_epoch = training_time / num_epochs
    
    # Thời gian tổng nếu không cache
    time_without_cache = (preprocessing_per_epoch + training_per_epoch) * num_epochs
    
    # Thời gian với cache (preprocessing 1 lần + training nhanh)
    time_with_cache = preprocessing_time + training_time
    
    savings = time_without_cache - time_with_cache
    speedup = time_without_cache / time_with_cache
    
    return savings, speedup

def check_environment():
    """
    Kiểm tra môi trường trước khi chạy
    """
    print("🔍 CHECKING ENVIRONMENT")
    print("=" * 50)
    
    # Kiểm tra GPU
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  No GPU detected - training will be slow on CPU")
    
    # Kiểm tra memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"✅ RAM: {memory.total / 1024**3:.1f}GB (available: {memory.available / 1024**3:.1f}GB)")
    
    # Kiểm tra disk space
    disk = psutil.disk_usage('.')
    print(f"✅ Disk: {disk.free / 1024**3:.1f}GB free")
    
    # Kiểm tra required packages
    required_packages = ['numpy', 'torch', 'h5py', 'scipy', 'tqdm', 'SimpleITK']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\n✅ Environment check passed!")
    return True

def main():
    """
    Main function - chạy toàn bộ pipeline
    """
    parser = argparse.ArgumentParser(description="Fast MRI-to-CT Training Pipeline")
    
    # Data directories
    parser.add_argument('--mri_dir', type=str, default='data/MRI', 
                       help='MRI data directory')
    parser.add_argument('--ct_dir', type=str, default='data/CT',
                       help='CT data directory')
    parser.add_argument('--cache_dir', type=str, default='preprocessed_cache',
                       help='Cache directory for preprocessed data')
    parser.add_argument('--output_dir', type=str, default='fast_training_output',
                       help='Training output directory')
    
    # Preprocessing options
    parser.add_argument('--cache_format', type=str, choices=['h5', 'pt'], default='h5',
                       help='Cache format (h5=HDF5, pt=PyTorch)')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing if cache exists')
    
    # Training options
    parser.add_argument('--batch_size', type=int, default=6,
                       help='Batch size (can be larger with cache)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of DataLoader workers')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Checkpoint save frequency (epochs)')
    parser.add_argument('--val_freq', type=int, default=5,
                       help='Validation frequency (epochs)')
    parser.add_argument('--aug_prob', type=float, default=0.8,
                       help='Data augmentation probability')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use mixed precision training')
    
    # Options
    parser.add_argument('--check_env', action='store_true',
                       help='Only check environment, don\'t run training')
    
    args = parser.parse_args()
    
    print("🚀 FAST MRI-TO-CT TRAINING PIPELINE")
    print("=" * 50)
    print(f"MRI directory: {args.mri_dir}")
    print(f"CT directory: {args.ct_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cache format: {args.cache_format}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Mixed precision: {args.use_amp}")
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed!")
        return 1
    
    if args.check_env:
        print("✅ Environment check complete - ready for training!")
        return 0
    
    # Verify data directories exist
    if not os.path.exists(args.mri_dir):
        print(f"❌ MRI directory not found: {args.mri_dir}")
        return 1
    
    if not os.path.exists(args.ct_dir):
        print(f"❌ CT directory not found: {args.ct_dir}")
        return 1
    
    total_start_time = time.time()
    
    try:
        # STEP 1: Preprocessing (if needed)
        preprocessing_time = 0
        
        if os.path.exists(args.cache_dir) and args.skip_preprocessing:
            print(f"\n✅ Cache found at {args.cache_dir}, skipping preprocessing")
            # Load existing metadata for stats
            import pickle
            metadata_path = os.path.join(args.cache_dir, "metadata", "preprocessing_info.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    cache_metadata = pickle.load(f)
                print(f"   Cached patients: {cache_metadata['total_patients']}")
                print(f"   Cached slices: {cache_metadata['total_slices']}")
        else:
            metadata, preprocessing_time = run_preprocessing(
                args.mri_dir, args.ct_dir, args.cache_dir, args.cache_format
            )
        
        # STEP 2: Fast Training
        training_args = {
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.lr,
            'num_workers': args.workers,
            'save_freq': args.save_freq,
            'val_freq': args.val_freq,
            'augmentation_prob': args.aug_prob,
            'use_amp': args.use_amp
        }
        
        results, training_time = run_training(args.cache_dir, args.output_dir, training_args)
        
        # SUMMARY
        total_time = time.time() - total_start_time
        
        print("\n🎉 PIPELINE COMPLETE!")
        print("=" * 50)
        print(f"📊 SUMMARY:")
        print(f"   Preprocessing time: {preprocessing_time/60:.1f} minutes")
        print(f"   Training time: {training_time/3600:.1f} hours")
        print(f"   Total pipeline time: {total_time/3600:.1f} hours")
        print(f"   Best validation loss: {results['best_val_loss']:.4f}")
        print(f"   Average epoch time: {results['total_time']/args.epochs:.1f}s")
        
        # Estimate savings
        if preprocessing_time > 0:
            savings, speedup = estimate_savings(preprocessing_time, training_time, args.epochs)
            print(f"\n💰 TIME SAVINGS:")
            print(f"   Time saved: {savings/3600:.1f} hours")
            print(f"   Speedup: {speedup:.1f}x faster")
            print(f"   Without cache: {(preprocessing_time + training_time) * args.epochs / speedup / 3600:.1f} hours")
        
        print(f"\n📁 OUTPUTS:")
        print(f"   Preprocessed cache: {args.cache_dir}")
        print(f"   Training results: {args.output_dir}")
        print(f"   Best model: {args.output_dir}/checkpoints/best_model.pth")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 