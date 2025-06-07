#!/usr/bin/env python3
"""
Multi-Slice Volume-Based Cached Data Loader
Tăng số lượng slices per patient để cải thiện model learning
"""

import os
import pickle
import random
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import h5py

class MultiSliceVolumeCachedDataset(data.Dataset):
    """
    Enhanced volume-based dataset với nhiều slices per patient
    """
    
    def __init__(self, 
                 cache_dir: str,
                 is_training: bool = True,
                 slices_per_patient: int = 10,  # TĂNG TỪ 1 LÊN 10!
                 slice_range: Optional[Tuple[int, int]] = None,
                 augmentation_prob: float = 0.8):
        
        self.cache_dir = cache_dir
        self.is_training = is_training
        self.slices_per_patient = slices_per_patient
        self.slice_range = slice_range
        self.augmentation_prob = augmentation_prob
        
        # Load metadata để biết save format
        self.metadata = self._load_metadata()
        self.save_format = self.metadata.get('save_format', 'pt')
        
        # Danh sách bệnh nhân
        if self.save_format == 'h5':
            self.patient_ids = [f.replace('.h5', '') for f in os.listdir(cache_dir) 
                              if f.endswith('.h5')]
        else:
            self.patient_ids = [f.replace('.pt', '') for f in os.listdir(cache_dir) 
                              if f.endswith('.pt')]
        
        self.patient_ids.sort()
        
        print(f"MultiSliceVolumeCachedDataset initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Save format: {self.save_format}")
        print(f"  🎯 MULTI-SLICE: {self.slices_per_patient} slices per patient")
        print(f"  🚀 Total samples per epoch: {len(self.patient_ids)} × {self.slices_per_patient} = {len(self.patient_ids) * self.slices_per_patient}")
        print(f"  Training mode: {is_training}")
        print(f"  Augmentation prob: {augmentation_prob if is_training else 'N/A'}")
    
    def _load_metadata(self) -> Dict:
        """Load preprocessing metadata"""
        metadata_path = os.path.join(self.cache_dir, "metadata", "preprocessing_info.pkl")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Fallback metadata
            return {'save_format': 'pt', 'slice_range': (40, 80)}
    
    def __len__(self):
        # Tổng số samples = số bệnh nhân × slices per patient
        return len(self.patient_ids) * self.slices_per_patient
    
    def __getitem__(self, idx):
        """
        Get một slice từ patient
        idx sẽ được mapping thành (patient_idx, slice_in_patient_idx)
        """
        # Mapping idx về patient và slice index
        patient_idx = idx // self.slices_per_patient
        slice_in_patient_idx = idx % self.slices_per_patient
        
        patient_id = self.patient_ids[patient_idx]
        
        # Load volume từ cache
        mri_volume, ct_volume = self._load_volume_from_cache(patient_id)
        
        # Chọn slice index
        if self.is_training:
            # Training: random slice mỗi lần (nhưng reproducible trong cùng epoch)
            random.seed(idx)  # Seed theo idx để consistent trong epoch
            if self.slice_range:
                slice_idx = random.randint(self.slice_range[0], 
                                         min(self.slice_range[1], mri_volume.shape[0] - 1))
            else:
                # Tránh slice đầu/cuối (thường là background)
                valid_range = max(10, int(0.1 * mri_volume.shape[0])), min(int(0.9 * mri_volume.shape[0]), mri_volume.shape[0] - 10)
                slice_idx = random.randint(valid_range[0], valid_range[1])
        else:
            # Validation: spread evenly across volume
            if self.slice_range:
                start_slice, end_slice = self.slice_range
            else:
                start_slice = max(10, int(0.1 * mri_volume.shape[0]))
                end_slice = min(int(0.9 * mri_volume.shape[0]), mri_volume.shape[0] - 10)
            
            # Spread slices evenly
            slice_step = max(1, (end_slice - start_slice) // self.slices_per_patient)
            slice_idx = start_slice + slice_in_patient_idx * slice_step
            slice_idx = min(slice_idx, end_slice)
        
        # Extract slices
        mri_slice = mri_volume[slice_idx]  # [H, W]
        ct_slice = ct_volume[slice_idx]    # [H, W]
        
        # Đảm bảo range [0,1]
        mri_slice = np.clip(mri_slice, 0, 1)
        ct_slice = np.clip(ct_slice, 0, 1)
        
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
            'filename': f"{patient_id}_slice_{slice_idx:03d}_idx_{slice_in_patient_idx}"
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
        """Áp dụng data augmentation"""
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


class MultiSliceDataLoaderManager:
    """
    Manager để tạo multi-slice cached data loaders
    Tăng số lượng samples để model học tốt hơn
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        
        # Load metadata
        metadata_path = os.path.join(cache_dir, "metadata", "preprocessing_info.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"MultiSliceDataLoaderManager initialized:")
        print(f"  🎯 MULTI-SLICE approach")
        print(f"  Patients: {self.metadata['total_patients']}")
        print(f"  Cached slices: {self.metadata['total_slices']}")
        print(f"  🚀 Ready for enhanced sampling!")
    
    def create_train_val_loaders(self, 
                                batch_size: int = 4,
                                train_split: float = 0.8,
                                num_workers: int = 2,
                                slices_per_patient: int = 10,  # MỚI!
                                slice_range: Optional[Tuple[int, int]] = None,
                                augmentation_prob: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Tạo multi-slice training và validation loaders
        """
        # Dataset cho training
        train_dataset = MultiSliceVolumeCachedDataset(
            cache_dir=self.cache_dir,
            is_training=True,
            slices_per_patient=slices_per_patient,
            slice_range=slice_range,
            augmentation_prob=augmentation_prob
        )
        
        # Dataset cho validation
        val_dataset = MultiSliceVolumeCachedDataset(
            cache_dir=self.cache_dir,
            is_training=False,  # No augmentation, deterministic slices
            slices_per_patient=max(3, slices_per_patient // 3),  # Ít slice hơn cho validation
            slice_range=slice_range,
            augmentation_prob=0.0
        )
        
        # Chia patients theo train/val split
        total_patients = len(train_dataset.patient_ids)
        train_patients = int(train_split * total_patients)
        val_patients = total_patients - train_patients
        
        # Chia patient IDs
        all_patient_ids = train_dataset.patient_ids.copy()
        random.seed(42)  # Reproducible split
        random.shuffle(all_patient_ids)
        
        train_patient_ids = all_patient_ids[:train_patients]
        val_patient_ids = all_patient_ids[train_patients:]
        
        # Tạo subset indices
        train_indices = []
        val_indices = []
        
        for i, patient_id in enumerate(train_dataset.patient_ids):
            start_idx = i * slices_per_patient
            end_idx = start_idx + slices_per_patient
            
            if patient_id in train_patient_ids:
                train_indices.extend(range(start_idx, end_idx))
        
        # Validation indices
        val_slices_per_patient = max(3, slices_per_patient // 3)
        for i, patient_id in enumerate(val_dataset.patient_ids):
            start_idx = i * val_slices_per_patient
            end_idx = start_idx + val_slices_per_patient
            
            if patient_id in val_patient_ids:
                val_indices.extend(range(start_idx, end_idx))
        
        # Tạo subsets
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # Tạo DataLoaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True  # Để batch size consistent
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        total_train_samples = len(train_indices)
        total_val_samples = len(val_indices)
        
        print(f"✅ Created MULTI-SLICE cached data loaders:")
        print(f"   Training: {len(train_patient_ids)} patients × {slices_per_patient} slices = {total_train_samples} samples")
        print(f"   Validation: {len(val_patient_ids)} patients × {val_slices_per_patient} slices = {total_val_samples} samples")
        print(f"   🎯 Train batches/epoch: ~{len(train_loader)}")
        print(f"   🚀 Data increase: {slices_per_patient}x more samples than volume-based!")
        print(f"   📊 Estimated training time: ~{len(train_loader)*5/60:.1f} minutes/epoch")
        
        return train_loader, val_loader

    def get_data_statistics(self, slices_per_patient: int = 10) -> Dict:
        """Lấy thống kê về data"""
        total_patients = self.metadata['total_patients']
        total_cached_slices = self.metadata['total_slices']
        
        samples_per_epoch = total_patients * slices_per_patient
        data_utilization = samples_per_epoch / total_cached_slices * 100
        
        return {
            'total_patients': total_patients,
            'total_cached_slices': total_cached_slices,
            'slices_per_patient': slices_per_patient,
            'samples_per_epoch': samples_per_epoch,
            'data_utilization_percent': data_utilization,
            'improvement_vs_volume': f"{slices_per_patient}x",
            'improvement_vs_original': f"{samples_per_epoch/42:.1f}x" if total_patients == 42 else "N/A"
        }


def test_multi_slice_loader():
    """Test function cho multi-slice loader"""
    cache_dir = "../preprocessed_cache"
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        return
    
    print("🧪 Testing Multi-Slice Data Loader...")
    
    # Test different slice counts
    for slices_per_patient in [5, 10, 15]:
        print(f"\n📊 Testing với {slices_per_patient} slices per patient:")
        
        manager = MultiSliceDataLoaderManager(cache_dir)
        stats = manager.get_data_statistics(slices_per_patient)
        
        print(f"   Samples per epoch: {stats['samples_per_epoch']}")
        print(f"   Data utilization: {stats['data_utilization_percent']:.1f}%")
        print(f"   Improvement vs volume-based: {stats['improvement_vs_volume']}")
        
        # Tạo loader test
        train_loader, val_loader = manager.create_train_val_loaders(
            batch_size=3,
            slices_per_patient=slices_per_patient,
            num_workers=0  # 0 for testing
        )
        
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        # Test một batch
        batch = next(iter(train_loader))
        print(f"   Batch MRI shape: {batch['mri'].shape}")
        print(f"   Batch CT shape: {batch['ct'].shape}")
        print(f"   MRI range: [{batch['mri'].min():.3f}, {batch['mri'].max():.3f}]")


if __name__ == "__main__":
    test_multi_slice_loader() 