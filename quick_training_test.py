import time
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from cached_data_loader import CachedDataLoaderManager
from optimized_cached_data_loader import OptimizedDataLoaderManager

def test_loading_speed():
    """
    So sánh tốc độ loading giữa original và optimized data loader
    """
    cache_dir = "preprocessed_cache"
    batch_size = 4
    
    print("=== SO SÁNH TỐC ĐỘ LOADING ===")
    
    # Test 1: Original CachedDataLoader
    print("\n🔵 Testing ORIGINAL CachedDataLoader...")
    try:
        original_manager = CachedDataLoaderManager(cache_dir)
        train_loader_orig, _ = original_manager.create_train_val_loaders(
            batch_size=batch_size,
            num_workers=2,
            augmentation_prob=0.0  # Disable augmentation for fair comparison
        )
        
        print(f"   Original training batches per epoch: {len(train_loader_orig)}")
        
        # Test loading speed - 20 batches
        start_time = time.time()
        for i, batch in enumerate(train_loader_orig):
            if i >= 20:
                break
        original_time = time.time() - start_time
        
        print(f"   Time for 20 batches: {original_time:.2f}s ({original_time/20:.3f}s per batch)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Optimized CachedDataLoader
    print("\n🟢 Testing OPTIMIZED CachedDataLoader...")
    try:
        optimized_manager = OptimizedDataLoaderManager(cache_dir)
        train_loader_opt, _ = optimized_manager.create_fast_train_val_loaders(
            batch_size=batch_size,
            num_workers=2,
            slice_sampling_strategy="middle_range",
            max_slices_per_patient=30,
            augmentation_prob=0.0  # Disable augmentation for fair comparison
        )
        
        print(f"   Optimized training batches per epoch: {len(train_loader_opt)}")
        
        # Test loading speed - 20 batches
        start_time = time.time()
        for i, batch in enumerate(train_loader_opt):
            if i >= 20:
                break
        optimized_time = time.time() - start_time
        
        print(f"   Time for 20 batches: {optimized_time:.2f}s ({optimized_time/20:.3f}s per batch)")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Tính toán improvement
    print(f"\n📊 RESULTS:")
    print(f"   Original batches/epoch: {len(train_loader_orig)}")
    print(f"   Optimized batches/epoch: {len(train_loader_opt)}")
    print(f"   Batch reduction: {(1 - len(train_loader_opt)/len(train_loader_orig))*100:.1f}%")
    
    # Ước tính training time per epoch
    estimated_orig_epoch = (len(train_loader_orig) / 20) * original_time
    estimated_opt_epoch = (len(train_loader_opt) / 20) * optimized_time
    
    print(f"\n⏱️  ESTIMATED EPOCH TIME:")
    print(f"   Original: {estimated_orig_epoch:.1f}s ({estimated_orig_epoch/60:.1f} minutes)")
    print(f"   Optimized: {estimated_opt_epoch:.1f}s ({estimated_opt_epoch/60:.1f} minutes)")
    print(f"   Time reduction: {(1 - estimated_opt_epoch/estimated_orig_epoch)*100:.1f}%")
    
    print(f"\n🚀 SUMMARY:")
    print(f"   Với optimized loading, mỗi epoch sẽ nhanh hơn ~{(estimated_orig_epoch/estimated_opt_epoch):.1f}x")
    print(f"   100 epochs: {estimated_orig_epoch*100/3600:.1f}h → {estimated_opt_epoch*100/3600:.1f}h")


def check_data_quality():
    """
    Kiểm tra chất lượng dữ liệu từ optimized loader
    """
    print("\n=== KIỂM TRA CHẤT LƯỢNG DỮ LIỆU ===")
    
    cache_dir = "preprocessed_cache"
    
    try:
        # Test different strategies
        strategies = [
            ("middle_range", {"slice_sampling_strategy": "middle_range", "max_slices_per_patient": 30}),
            ("every_4th", {"slice_sampling_strategy": "every_nth", "slice_step": 4}),
            ("random_25", {"slice_sampling_strategy": "random_sample", "max_slices_per_patient": 25})
        ]
        
        for name, params in strategies:
            manager = OptimizedDataLoaderManager(cache_dir)
            train_loader, _ = manager.create_fast_train_val_loaders(
                batch_size=1, 
                num_workers=0,
                **params
            )
            
            # Sample một batch để kiểm tra
            batch = next(iter(train_loader))
            mri = batch['mri'][0, 0].numpy()  # [C, H, W] → [H, W]
            ct = batch['ct'][0, 0].numpy()
            
            print(f"\n🔍 Strategy: {name}")
            print(f"   Total batches: {len(train_loader)}")
            print(f"   MRI range: [{mri.min():.3f}, {mri.max():.3f}]")
            print(f"   CT range: [{ct.min():.3f}, {ct.max():.3f}]")
            print(f"   MRI mean: {mri.mean():.3f}, std: {mri.std():.3f}")
            print(f"   CT mean: {ct.mean():.3f}, std: {ct.std():.3f}")
            
    except Exception as e:
        print(f"❌ Error checking data quality: {e}")


if __name__ == "__main__":
    print("🧪 TESTING OPTIMIZED DATA LOADING")
    print("="*50)
    
    # Kiểm tra cache tồn tại
    if not os.path.exists("preprocessed_cache"):
        print("❌ Cache not found! Run preprocessing first.")
        exit()
    
    # Test loading speed
    test_loading_speed()
    
    # Test data quality
    check_data_quality()
    
    print(f"\n✅ Testing completed!")
    print(f"💡 Recommendation: Sử dụng 'middle_range' strategy với max_slices_per_patient=30") 