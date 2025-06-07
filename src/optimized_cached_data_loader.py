import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import h5py
import os
import pickle
import random
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm

class OptimizedCachedMRIToCTDataset(data.Dataset):
    """
    Tối ưu Dataset để giảm training time bằng cách:
    1. Chỉ lấy một số slices đại diện từ mỗi bệnh nhân
    2. Loại bỏ slices trống hoặc có ít thông tin
    3. Option sampling strategy linh hoạt
    """
    
    def __init__(self, 
                 cache_dir: str,
                 is_training: bool = True,
                 slice_sampling_strategy: str = "middle_range",  # "all", "middle_range", "random_sample", "every_nth"
                 max_slices_per_patient: int = 30,  # Giới hạn slices mỗi bệnh nhân
                 slice_step: int = 4,  # Lấy mỗi slice thứ N (cho every_nth)
                 augmentation_prob: float = 0.8):
        """
        Args:
            cache_dir: thư mục chứa preprocessed cache
            is_training: training mode hay evaluation mode
            slice_sampling_strategy: 
                - "all": lấy tất cả slices (như cũ)
                - "middle_range": lấy 60% slices giữa (bỏ đầu/cuối)
                - "random_sample": random sample số lượng slices
                - "every_nth": lấy mỗi slice thứ N
            max_slices_per_patient: số slices tối đa mỗi bệnh nhân
            slice_step: bước nhảy slice (cho every_nth strategy)
            augmentation_prob: xác suất áp dụng augmentation
        """
        self.cache_dir = cache_dir
        self.is_training = is_training
        self.slice_sampling_strategy = slice_sampling_strategy
        self.max_slices_per_patient = max_slices_per_patient
        self.slice_step = slice_step
        self.augmentation_prob = augmentation_prob
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.save_format = self.metadata['save_format']
        
        # Tạo danh sách slices đã được tối ưu
        self.slice_list = self._build_optimized_slice_list()
        
        print(f"OptimizedCachedMRIToCTDataset initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Patients: {self.metadata['total_patients']}")
        print(f"  Sampling strategy: {slice_sampling_strategy}")
        print(f"  Max slices/patient: {max_slices_per_patient}")
        print(f"  🚀 OPTIMIZED slices: {len(self.slice_list)} (reduced from {self.metadata['total_slices']})")
        print(f"  💾 Memory reduction: {(1 - len(self.slice_list)/self.metadata['total_slices'])*100:.1f}%")
        print(f"  Training mode: {is_training}")
        
    def _load_metadata(self) -> Dict:
        """Load preprocessing metadata"""
        metadata_path = os.path.join(self.cache_dir, "metadata", "preprocessing_info.pkl")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return metadata
    
    def _build_optimized_slice_list(self) -> List[Tuple[str, int]]:
        """
        Tạo danh sách slices đã được tối ưu theo strategy
        """
        slice_list = []
        
        for patient_meta in self.metadata['metadata']:
            patient_id = patient_meta['patient_id']
            total_slices = patient_meta['num_slices']
            
            # Lấy danh sách slice indices theo strategy
            selected_indices = self._select_slice_indices(total_slices)
            
            # Thêm vào slice_list
            for slice_idx in selected_indices:
                slice_list.append((patient_id, slice_idx))
        
        return slice_list
    
    def _select_slice_indices(self, total_slices: int) -> List[int]:
        """
        Chọn slice indices theo strategy
        """
        if self.slice_sampling_strategy == "all":
            return list(range(total_slices))
        
        elif self.slice_sampling_strategy == "middle_range":
            # Lấy 60% slices ở giữa, bỏ 20% đầu và 20% cuối
            start_idx = int(0.2 * total_slices)
            end_idx = int(0.8 * total_slices)
            indices = list(range(start_idx, end_idx))
            
            # Giới hạn số lượng nếu cần
            if len(indices) > self.max_slices_per_patient:
                step = len(indices) // self.max_slices_per_patient
                indices = indices[::step][:self.max_slices_per_patient]
            
            return indices
        
        elif self.slice_sampling_strategy == "random_sample":
            # Random sample số lượng slices
            num_samples = min(self.max_slices_per_patient, total_slices)
            return sorted(random.sample(range(total_slices), num_samples))
        
        elif self.slice_sampling_strategy == "every_nth":
            # Lấy mỗi slice thứ N
            indices = list(range(0, total_slices, self.slice_step))
            
            # Giới hạn số lượng nếu cần
            if len(indices) > self.max_slices_per_patient:
                indices = indices[:self.max_slices_per_patient]
            
            return indices
        
        else:
            raise ValueError(f"Unknown sampling strategy: {self.slice_sampling_strategy}")
    
    def __len__(self):
        return len(self.slice_list)
    
    def __getitem__(self, idx):
        """
        Load một slice đã được preprocessed và áp dụng augmentation
        """
        patient_id, slice_idx = self.slice_list[idx]
        
        # Load preprocessed volume từ cache
        mri_volume, ct_volume = self._load_volume_from_cache(patient_id)
        
        # Lấy slice cụ thể
        mri_slice = mri_volume[slice_idx]  # Shape: (256, 256)
        ct_slice = ct_volume[slice_idx]    # Shape: (256, 256)
        
        # Convert về [-1,1] cho model
        mri_slice = mri_slice * 2.0 - 1.0
        ct_slice = ct_slice * 2.0 - 1.0
        
        # === AUGMENTATION (chỉ khi training và random) ===
        if self.is_training and random.random() < self.augmentation_prob:
            mri_slice, ct_slice = self._apply_augmentation(mri_slice, ct_slice)
        
        # Đảm bảo có positive strides
        mri_slice = mri_slice.copy()
        ct_slice = ct_slice.copy()
        
        # Convert to tensors
        mri_tensor = torch.tensor(mri_slice, dtype=torch.float32).unsqueeze(0)
        ct_tensor = torch.tensor(ct_slice, dtype=torch.float32).unsqueeze(0)
        
        return {
            'mri': mri_tensor,
            'ct': ct_tensor,
            'filename': f"{patient_id}_slice_{slice_idx:03d}"
        }
    
    def _load_volume_from_cache(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load preprocessed volume từ cache"""
        if self.save_format == 'h5':
            return self._load_h5_volume(patient_id)
        elif self.save_format == 'pt':
            return self._load_pytorch_volume(patient_id)
        else:
            raise ValueError(f"Unsupported save format: {self.save_format}")
    
    def _load_h5_volume(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load từ HDF5 file"""
        h5_path = os.path.join(self.cache_dir, f"{patient_id}.h5")
        
        with h5py.File(h5_path, 'r') as f:
            mri_volume = f['mri'][:]
            ct_volume = f['ct'][:]
        
        return mri_volume, ct_volume
    
    def _load_pytorch_volume(self, patient_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load từ PyTorch tensors"""
        pt_path = os.path.join(self.cache_dir, f"{patient_id}.pt")
        
        data = torch.load(pt_path, map_location='cpu', weights_only=True)
        mri_volume = data['mri'].numpy()
        ct_volume = data['ct'].numpy()
        
        return mri_volume, ct_volume
    
    def _apply_augmentation(self, mri_slice: np.ndarray, ct_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Áp dụng data augmentation nhanh"""
        # Random rotation (±15°)
        if random.random() > 0.5:
            from scipy import ndimage
            angle = random.uniform(-15, 15)
            mri_slice = ndimage.rotate(mri_slice, angle, reshape=False, mode='constant', cval=-1)
            ct_slice = ndimage.rotate(ct_slice, angle, reshape=False, mode='constant', cval=-1)
        
        # Random flip horizontal
        if random.random() > 0.5:
            mri_slice = np.fliplr(mri_slice).copy()
            ct_slice = np.fliplr(ct_slice).copy()
            
        # Random flip vertical
        if random.random() > 0.5:
            mri_slice = np.flipud(mri_slice).copy()
            ct_slice = np.flipud(ct_slice).copy()
        
        # Random intensity scaling cho MRI (±10%)
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            mask = mri_slice > -0.8  # Detect brain region
            mri_slice[mask] = mri_slice[mask] * scale_factor
            mri_slice = np.clip(mri_slice, -1, 1)
        
        # Clip về [-1,1]
        mri_slice = np.clip(mri_slice, -1, 1)
        ct_slice = np.clip(ct_slice, -1, 1)
        
        return mri_slice, ct_slice


class OptimizedDataLoaderManager:
    """
    Manager tối ưu để tạo data loaders với giảm thiểu training time
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        
        # Load metadata
        metadata_path = os.path.join(cache_dir, "metadata", "preprocessing_info.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"OptimizedDataLoaderManager initialized:")
        print(f"  Original: {self.metadata['total_patients']} patients, {self.metadata['total_slices']} slices")
    
    def create_fast_train_val_loaders(self, 
                                     batch_size: int = 4,
                                     train_split: float = 0.8,
                                     num_workers: int = 2,
                                     slice_sampling_strategy: str = "middle_range",
                                     max_slices_per_patient: int = 30,
                                     slice_step: int = 4,
                                     augmentation_prob: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Tạo FAST training và validation loaders
        
        Recommended strategies:
        - "middle_range" + max_slices_per_patient=30: Giảm ~75% training time
        - "every_nth" + slice_step=4: Giảm ~75% training time  
        - "random_sample" + max_slices_per_patient=20: Giảm ~80% training time
        """
        
        # Tạo full dataset với tối ưu
        full_dataset = OptimizedCachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=True,
            slice_sampling_strategy=slice_sampling_strategy,
            max_slices_per_patient=max_slices_per_patient,
            slice_step=slice_step,
            augmentation_prob=augmentation_prob
        )
        
        # Chia train/val theo slice
        dataset_size = len(full_dataset)
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Dataset cho validation không có augmentation
        val_dataset_no_aug = OptimizedCachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=False,  # No augmentation
            slice_sampling_strategy=slice_sampling_strategy,
            max_slices_per_patient=max_slices_per_patient,
            slice_step=slice_step,
            augmentation_prob=0.0
        )
        
        # Lấy validation indices
        val_indices = val_dataset.indices
        val_subset = torch.utils.data.Subset(val_dataset_no_aug, val_indices)
        
        # Tạo DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # Tính toán time reduction
        original_batches = self.metadata['total_slices'] // batch_size
        new_batches = len(train_dataset) // batch_size
        time_reduction = (1 - new_batches / original_batches) * 100
        
        print(f"✅ Created OPTIMIZED data loaders:")
        print(f"   Training: {len(train_dataset)} slices ({len(train_dataset)//batch_size} batches)")
        print(f"   Validation: {len(val_subset)} slices ({len(val_subset)//batch_size} batches)")
        print(f"   🚀 Training time reduction: ~{time_reduction:.1f}%")
        print(f"   Original: {original_batches} batches → New: {new_batches} batches")
        
        return train_loader, val_loader


def compare_strategies():
    """
    So sánh các strategies khác nhau để chọn tối ưu nhất
    """
    cache_dir = "preprocessed_cache"
    
    if not os.path.exists(cache_dir):
        print("❌ Cache not found!")
        return
    
    print("=== SO SÁNH SLICE SAMPLING STRATEGIES ===")
    
    strategies = [
        ("all", {"slice_sampling_strategy": "all"}),
        ("middle_range_30", {"slice_sampling_strategy": "middle_range", "max_slices_per_patient": 30}),
        ("middle_range_20", {"slice_sampling_strategy": "middle_range", "max_slices_per_patient": 20}),
        ("every_4th", {"slice_sampling_strategy": "every_nth", "slice_step": 4}),
        ("every_6th", {"slice_sampling_strategy": "every_nth", "slice_step": 6}),
        ("random_25", {"slice_sampling_strategy": "random_sample", "max_slices_per_patient": 25}),
        ("random_20", {"slice_sampling_strategy": "random_sample", "max_slices_per_patient": 20}),
    ]
    
    results = []
    
    for name, params in strategies:
        try:
            dataset = OptimizedCachedMRIToCTDataset(
                cache_dir=cache_dir,
                is_training=True,
                **params
            )
            
            num_slices = len(dataset)
            reduction = (1 - num_slices / 4681) * 100  # 4681 là original total
            
            results.append({
                'name': name,
                'slices': num_slices,
                'reduction': reduction,
                'batches_per_epoch': num_slices // 4  # Batch size = 4
            })
            
        except Exception as e:
            print(f"❌ Strategy {name} failed: {e}")
    
    # Sort by reduction percentage
    results.sort(key=lambda x: x['reduction'], reverse=True)
    
    print("\n📊 RESULTS (sorted by time reduction):")
    print("Strategy".ljust(20) + "Slices".ljust(10) + "Reduction".ljust(12) + "Batches/Epoch")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<20} {result['slices']:<10} {result['reduction']:.1f}%{'':<8} {result['batches_per_epoch']}")
    
    print(f"\n🔥 RECOMMENDATION: 'middle_range_30' hoặc 'every_4th' để giảm ~75% training time!")


if __name__ == "__main__":
    compare_strategies() 