#!/usr/bin/env python3
"""
Script để chạy CycleGAN training với cached preprocessed data
Tích hợp preprocessing và training trong một pipeline
"""

import os
import sys
import subprocess
import argparse

def run_preprocessing(mri_dir: str = "data/MRI", ct_dir: str = "data/CT", cache_dir: str = "preprocessed_cache"):
    """
    Chạy preprocessing nếu cache chưa tồn tại
    """
    if os.path.exists(cache_dir):
        print(f"✅ Cache already exists at: {cache_dir}")
        return True
    
    print(f"🔄 Cache not found. Running preprocessing...")
    print(f"   MRI directory: {mri_dir}")
    print(f"   CT directory: {ct_dir}")
    print(f"   Cache directory: {cache_dir}")
    
    # Kiểm tra data directories
    if not os.path.exists(mri_dir):
        print(f"❌ MRI directory not found: {mri_dir}")
        return False
    
    if not os.path.exists(ct_dir):
        print(f"❌ CT directory not found: {ct_dir}")
        return False
    
    # Chạy preprocessing
    try:
        cmd = [
            sys.executable, "preprocess_and_cache.py",
            "--mri_dir", mri_dir,
            "--ct_dir", ct_dir,
            "--cache_dir", cache_dir,
            "--format", "h5"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Preprocessing completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def run_cyclegan_training(cache_dir: str = "preprocessed_cache"):
    """
    Chạy CycleGAN training với cached data
    """
    print(f"\n🚀 Starting CycleGAN training with cached data...")
    
    try:
        # Chạy training script
        cmd = [sys.executable, "src/train.py"]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Chạy interactively để user có thể input
        result = subprocess.run(cmd, check=True)
        
        print("✅ Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  Training interrupted by user")
        return False

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="CycleGAN Training with Cached Preprocessing")
    
    parser.add_argument('--mri_dir', type=str, default='data/MRI', 
                       help='MRI data directory')
    parser.add_argument('--ct_dir', type=str, default='data/CT',
                       help='CT data directory')
    parser.add_argument('--cache_dir', type=str, default='preprocessed_cache',
                       help='Cache directory')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing step (assume cache exists)')
    parser.add_argument('--preprocessing_only', action='store_true',
                       help='Only run preprocessing, skip training')
    
    args = parser.parse_args()
    
    print("🔥 CYCLEGAN TRAINING WITH CACHED PREPROCESSING")
    print("=" * 50)
    print(f"MRI directory: {args.mri_dir}")
    print(f"CT directory: {args.ct_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Skip preprocessing: {args.skip_preprocessing}")
    print(f"Preprocessing only: {args.preprocessing_only}")
    
    success = True
    
    # STEP 1: Preprocessing (if needed)
    if not args.skip_preprocessing:
        success = run_preprocessing(args.mri_dir, args.ct_dir, args.cache_dir)
        
        if not success:
            print("❌ Preprocessing failed. Cannot continue to training.")
            return 1
    else:
        if not os.path.exists(args.cache_dir):
            print(f"❌ Cache directory not found: {args.cache_dir}")
            print("   Remove --skip_preprocessing flag to create cache first")
            return 1
        print(f"✅ Skipping preprocessing, using existing cache: {args.cache_dir}")
    
    # STEP 2: Training (if not preprocessing only)
    if not args.preprocessing_only:
        success = run_cyclegan_training(args.cache_dir)
        
        if not success:
            print("❌ Training failed.")
            return 1
    else:
        print("✅ Preprocessing completed. Skipping training as requested.")
    
    if success:
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        
        if not args.preprocessing_only:
            print(f"📁 Training outputs:")
            print(f"   Checkpoints: src/checkpoints/")
            print(f"   Samples: src/samples/")
            print(f"   Logs: src/logs/")
        
        print(f"📁 Preprocessed cache: {args.cache_dir}")
        
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 