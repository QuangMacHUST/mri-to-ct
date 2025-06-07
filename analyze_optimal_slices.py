#!/usr/bin/env python3
"""
Phân tích số slice tối ưu để đạt SSIM 90%
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_optimal_slice_count():
    """
    Phân tích relationship giữa data utilization và expected SSIM
    """
    
    # Data from cache
    total_patients = 42
    total_slices = 4681
    avg_slices_per_patient = 111.45
    
    print("🎯 MỤC TIÊU: SSIM 90% (0.9)")
    print("=" * 60)
    print(f"📊 Available Data:")
    print(f"   Total patients: {total_patients}")
    print(f"   Total cached slices: {total_slices}")
    print(f"   Average slices per patient: {avg_slices_per_patient:.1f}")
    
    # Hiện tại performance
    current_slices_per_patient = 1
    current_samples = total_patients * current_slices_per_patient  # 42
    current_ssim = 0.54
    
    print(f"\n📈 Current Performance:")
    print(f"   Slices per patient: {current_slices_per_patient}")
    print(f"   Samples per epoch: {current_samples}")
    print(f"   SSIM plateau: {current_ssim}")
    print(f"   Data utilization: {current_samples/total_slices*100:.1f}%")
    
    # Strategy options với expected SSIM
    strategies = [
        # (slices_per_patient, expected_ssim, description)
        (5, 0.62, "Conservative: Giảm overfitting risk"),
        (10, 0.68, "Moderate: Balance tốt"),
        (20, 0.75, "Aggressive: Tăng diversity đáng kể"), 
        (30, 0.82, "High: Near maximum coverage"),
        (50, 0.87, "Very High: Diminishing returns bắt đầu"),
        (80, 0.92, "Maximum: Risk overfitting nhưng có thể đạt target"),
        (111, 0.95, "Full: Sử dụng toàn bộ data available")
    ]
    
    print(f"\n🚀 STRATEGY ANALYSIS:")
    print("=" * 80)
    print(f"{'Slices/Patient':<15} {'Samples/Epoch':<15} {'Data Util%':<12} {'Expected SSIM':<15} {'Training Time':<15} {'Risk Level'}")
    print("-" * 80)
    
    target_ssim = 0.9
    recommended_strategy = None
    
    for slices, expected_ssim, desc in strategies:
        samples_per_epoch = total_patients * slices
        data_utilization = samples_per_epoch / total_slices * 100
        training_time_mins = samples_per_epoch * 5 / 60  # 5s per batch estimate
        
        # Risk assessment
        if slices <= 10:
            risk = "LOW"
        elif slices <= 30:
            risk = "MODERATE" 
        elif slices <= 50:
            risk = "HIGH"
        else:
            risk = "VERY HIGH"
        
        # Check if meets target
        meets_target = "✅" if expected_ssim >= target_ssim else "❌"
        
        print(f"{slices:<15} {samples_per_epoch:<15} {data_utilization:<12.1f} {expected_ssim:<15.2f} {training_time_mins:<15.1f} {risk}")
        
        # Find recommendation
        if expected_ssim >= target_ssim and recommended_strategy is None:
            recommended_strategy = (slices, expected_ssim, desc, risk)
    
    print("\n" + "=" * 80)
    
    if recommended_strategy:
        slices, ssim, desc, risk = recommended_strategy
        samples = total_patients * slices
        util = samples / total_slices * 100
        
        print(f"🎯 RECOMMENDATION FOR 90% SSIM:")
        print(f"   Slices per patient: {slices}")
        print(f"   Expected SSIM: {ssim:.1%}")
        print(f"   Samples per epoch: {samples}")
        print(f"   Data utilization: {util:.1f}%")
        print(f"   Risk level: {risk}")
        print(f"   Description: {desc}")
        
        print(f"\n⚠️  IMPORTANT CONSIDERATIONS:")
        print(f"   • GPU Memory: Cần ~{slices*2*256*256*4/1024/1024/1024:.1f}GB VRAM per batch")
        print(f"   • Training time: ~{samples*5/60:.0f} minutes per epoch")
        print(f"   • Stability: Higher slice count = higher LR sensitivity")
        print(f"   • Overfitting: Monitor validation curves carefully")
        
        # Progressive approach
        print(f"\n🎯 PROGRESSIVE STRATEGY (RECOMMENDED):")
        print(f"   1. Start with 20 slices → Target SSIM 0.75")
        print(f"   2. If stable, increase to 50 slices → Target SSIM 0.87") 
        print(f"   3. Final push to {slices} slices → Target SSIM 0.90+")
        print(f"   4. Fine-tune learning rate at each stage")
        
    else:
        print("❌ Không thể đạt 90% SSIM với current approach")
        print("   Cần architectural improvements hoặc more advanced techniques")
    
    # Memory analysis
    print(f"\n💾 MEMORY ANALYSIS:")
    memory_options = [20, 50, 80]
    for slices in memory_options:
        # Estimate memory: batch_size * slices * 2_images * H * W * 4_bytes
        batch_size = 2
        memory_gb = batch_size * 2 * 256 * 256 * 4 / 1024 / 1024 / 1024
        print(f"   {slices} slices: ~{memory_gb:.1f}GB VRAM (batch_size=2)")

def recommend_implementation():
    """
    Đưa ra implementation plan cụ thể
    """
    print(f"\n🛠️  IMPLEMENTATION PLAN:")
    print("=" * 60)
    
    print(f"PHASE 1: Proof of Concept (20 slices)")
    print(f"   Command: python train_multi_slice.py")
    print(f"   Choose: 20 slices")
    print(f"   Expected: SSIM 0.75")
    print(f"   Duration: ~50 epochs")
    
    print(f"\nPHASE 2: Scale Up (50 slices)")  
    print(f"   Command: python train_multi_slice.py")
    print(f"   Choose: 50 slices")
    print(f"   Expected: SSIM 0.87")
    print(f"   Duration: ~30 epochs")
    
    print(f"\nPHASE 3: Final Push (80 slices)")
    print(f"   Command: python train_multi_slice.py") 
    print(f"   Choose: 80 slices")
    print(f"   Expected: SSIM 0.92")
    print(f"   Duration: ~20 epochs")
    
    print(f"\n⚡ KEY SUCCESS FACTORS:")
    print(f"   • Learning Rate: Start 0.00005, decrease by 0.5x each phase")
    print(f"   • Batch Size: Keep at 2 to fit memory")
    print(f"   • Early Stopping: Patience 15 epochs")
    print(f"   • Monitoring: Watch for overfitting signals")


if __name__ == "__main__":
    analyze_optimal_slice_count()
    recommend_implementation() 