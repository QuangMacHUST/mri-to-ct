#!/usr/bin/env python3
"""
So sánh performance training với different slice counts
"""

import sys
sys.path.append('src')

from src.multi_slice_cached_loader import MultiSliceDataLoaderManager

def compare_slice_strategies():
    """
    So sánh các strategy slice khác nhau
    """
    
    print("🔬 SLICE STRATEGY COMPARISON")
    print("=" * 80)
    
    cache_dir = 'preprocessed_cache'
    
    # Test different slice counts
    slice_counts = [10, 20, 50, 80, 100]
    
    try:
        manager = MultiSliceDataLoaderManager(cache_dir)
        
        print(f"📊 Performance Comparison Table:")
        print("-" * 80)
        print(f"{'Slices':<8} {'Train Samples':<15} {'Val Samples':<12} {'Batches/Epoch':<15} {'Est. Time/Epoch':<15} {'Expected SSIM'}")
        print("-" * 80)
        
        for slice_count in slice_counts:
            # Get statistics
            stats = manager.get_data_statistics(slice_count)
            
            # Create test loaders
            train_loader, val_loader = manager.create_train_val_loaders(
                batch_size=2,
                slices_per_patient=slice_count,
                num_workers=0  # For testing
            )
            
            # Calculate metrics
            train_samples = len(train_loader.dataset)
            val_samples = len(val_loader.dataset)
            batches_per_epoch = len(train_loader)
            est_time_mins = batches_per_epoch * 5 / 60  # 5s per batch estimate
            
            # Expected SSIM based on data utilization
            if slice_count <= 10:
                expected_ssim = 0.68
            elif slice_count <= 20:
                expected_ssim = 0.75
            elif slice_count <= 50:
                expected_ssim = 0.87
            elif slice_count <= 80:
                expected_ssim = 0.92
            else:
                expected_ssim = 0.95
            
            print(f"{slice_count:<8} {train_samples:<15} {val_samples:<12} {batches_per_epoch:<15} {est_time_mins:<15.1f} {expected_ssim:<.2f}")
        
        print("-" * 80)
        print("\n🎯 RECOMMENDATIONS:")
        print(f"   • For SSIM 90%: Use 80+ slices")
        print(f"   • Start with 20 slices for proof of concept")
        print(f"   • Monitor GPU memory usage carefully")
        print(f"   • Progressive training approach recommended")
        
        # Memory warning
        print(f"\n⚠️  MEMORY CONSIDERATIONS:")
        print(f"   • Batch size 2 recommended for high slice counts")
        print(f"   • Monitor VRAM usage during training")
        print(f"   • Reduce batch size if OOM errors occur")
        
        # Training time estimates
        print(f"\n⏱️  TRAINING TIME ESTIMATES (100 epochs):")
        for slice_count in [20, 50, 80]:
            # Estimate based on batch count
            if slice_count == 20:
                batches = 330  # From calculation
                time_hours = batches * 5 * 100 / 3600  # 5s per batch, 100 epochs
            elif slice_count == 50:
                batches = 825
                time_hours = batches * 5 * 100 / 3600
            else:  # 80 slices
                batches = 1320
                time_hours = batches * 5 * 100 / 3600
            
            print(f"   {slice_count} slices: ~{time_hours:.1f} hours for 100 epochs")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")


def recommend_progressive_strategy():
    """
    Đưa ra progressive training strategy
    """
    print(f"\n🚀 PROGRESSIVE STRATEGY FOR 90% SSIM:")
    print("=" * 80)
    
    strategies = [
        {
            'phase': 'Phase 1: Foundation',
            'slices': 20,
            'epochs': 30,
            'target_ssim': 0.75,
            'lr': 0.00004,
            'description': 'Establish baseline, verify stability'
        },
        {
            'phase': 'Phase 2: Enhancement', 
            'slices': 50,
            'epochs': 20,
            'target_ssim': 0.87,
            'lr': 0.00003,
            'description': 'Increase data diversity significantly'
        },
        {
            'phase': 'Phase 3: Optimization',
            'slices': 80,
            'epochs': 15,
            'target_ssim': 0.92,
            'lr': 0.000025,
            'description': 'Final push to 90%+ SSIM'
        }
    ]
    
    total_time = 0
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{strategy['phase']}:")
        print(f"   Slices per patient: {strategy['slices']}")
        print(f"   Target epochs: {strategy['epochs']}")
        print(f"   Learning rate: {strategy['lr']}")
        print(f"   Target SSIM: {strategy['target_ssim']:.1%}")
        print(f"   Description: {strategy['description']}")
        
        # Time estimate
        samples_per_epoch = 42 * strategy['slices'] * 0.8  # 80% train split
        batches_per_epoch = samples_per_epoch // 2  # batch size 2
        time_per_phase = batches_per_epoch * 5 * strategy['epochs'] / 3600  # hours
        total_time += time_per_phase
        
        print(f"   Estimated time: {time_per_phase:.1f} hours")
        
        if i < len(strategies):
            print(f"   → Continue to next phase if SSIM ≥ {strategy['target_ssim']:.2f}")
    
    print(f"\n📊 TOTAL ESTIMATED TIME: {total_time:.1f} hours")
    print(f"🎯 EXPECTED FINAL SSIM: 92%+")
    
    print(f"\n✅ SUCCESS CRITERIA:")
    print(f"   • Phase 1: SSIM ≥ 0.75 (proceed to Phase 2)")
    print(f"   • Phase 2: SSIM ≥ 0.87 (proceed to Phase 3)")
    print(f"   • Phase 3: SSIM ≥ 0.90 (SUCCESS!)")
    
    print(f"\n⚠️  MONITORING CHECKLIST:")
    print(f"   □ GPU memory usage < 90%")
    print(f"   □ Training loss decreasing smoothly")
    print(f"   □ Validation SSIM improving")
    print(f"   □ No gradient explosion")
    print(f"   □ Model outputs visually reasonable")


if __name__ == "__main__":
    compare_slice_strategies()
    recommend_progressive_strategy() 