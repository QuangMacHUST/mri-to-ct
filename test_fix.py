#!/usr/bin/env python3
"""
Script test các fix cho lỗi SSIM import và torch.load warning
"""

def test_ssim_import():
    """Test SSIM import"""
    try:
        import sys
        sys.path.append('src')
        from metrics import MetricsCalculator
        print("✅ SSIM import thành công")
        return True
    except ImportError as e:
        print(f"❌ SSIM import lỗi: {e}")
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    try:
        import torch
        import sys
        sys.path.append('src')
        from metrics import MetricsCalculator
        
        # Tạo fake tensors
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randn(1, 1, 64, 64)
        
        # Test calculation
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(pred, target)
        
        print("✅ Metrics calculation thành công:")
        for name, value in metrics.items():
            print(f"   {name}: {value:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation lỗi: {e}")
        return False

def test_torch_load():
    """Test torch.load với weights_only parameter"""
    try:
        import torch
        
        # Tạo dummy checkpoint
        dummy_data = {'test': torch.tensor([1, 2, 3])}
        torch.save(dummy_data, 'test_checkpoint.pth')
        
        # Test load với weights_only=False
        loaded = torch.load('test_checkpoint.pth', weights_only=False)
        print("✅ torch.load với weights_only=False thành công")
        
        # Cleanup
        import os
        os.remove('test_checkpoint.pth')
        return True
        
    except Exception as e:
        print(f"❌ torch.load test lỗi: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TEST CÁC FIX CHO LỖI")
    print("=" * 50)
    
    results = []
    results.append(test_ssim_import())
    results.append(test_metrics_calculation()) 
    results.append(test_torch_load())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 TẤT CẢ CÁC FIX HOẠT ĐỘNG THÀNH CÔNG!")
    else:
        print("❌ VẪN CÒN LỖI CẦN KHẮC PHỤC")
    print("=" * 50) 