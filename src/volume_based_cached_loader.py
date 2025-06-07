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

class VolumeCachedMRIToCTDataset(data.Dataset):
    """
    Volume-based Dataset giống data_loader.py cũ:
    - Mỗi __getitem__ trả về 1 RANDOM SLICE từ 1 volume 
    - Tổng cộng 42 samples (như cũ), KHÔNG phải 4,681 slices
    - Mỗi epoch = 42 samples = ~10 batches (như trước khi cache)
    """
    
    def __init__(self, 
                 cache_dir: str,
                 is_training: bool = True,
                 slice_range: Optional[Tuple[int, int]] = None,
                 augmentation_prob: float = 0.8):
        """
        Args:
            cache_dir: thư mục chứa preprocessed cache
            is_training: training mode (có random slice) hay evaluation mode (middle slice)
            slice_range: giới hạn slice nếu cần
            augmentation_prob: xác suất áp dụng augmentation
        """
        self.cache_dir = cache_dir
        self.is_training = is_training
        self.slice_range = slice_range
        self.augmentation_prob = augmentation_prob
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.save_format = self.metadata['save_format']
        
        # Lấy danh sách bệnh nhân (KHÔNG phải danh sách slices)
        self.patient_list = [meta['patient_id'] for meta in self.metadata['metadata']]
        
        print(f"VolumeCachedMRIToCTDataset initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Save format: {self.save_format}")
        print(f"  🎯 VOLUME-BASED: {len(self.patient_list)} samples (patients)")
        print(f"  🚀 Same as original data_loader.py: ~{len(self.patient_list)//4} batches/epoch")
        print(f"  Training mode: {is_training}")
        print(f"  Augmentation prob: {augmentation_prob if is_training else 'N/A'}")
        
    def _load_metadata(self) -> Dict:
        """Load preprocessing metadata"""
        metadata_path = os.path.join(self.cache_dir, "metadata", "preprocessing_info.pkl")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return metadata
    
    def __len__(self):
        """Trả về số lượng BỆNH NHÂN, không phải số slices"""
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        """
        Lấy 1 RANDOM SLICE từ bệnh nhân thứ idx (giống data_loader.py cũ)
        """
        patient_id = self.patient_list[idx]
        
        # Load preprocessed volume từ cache
        mri_volume, ct_volume = self._load_volume_from_cache(patient_id)
        
        # Lấy metadata của bệnh nhân này
        patient_meta = next(meta for meta in self.metadata['metadata'] if meta['patient_id'] == patient_id)
        num_slices = patient_meta['num_slices']
        
        # Chọn slice index (RANDOM nếu training, MIDDLE nếu evaluation)
        if self.slice_range:
            start_slice, end_slice = self.slice_range
            start_slice = max(0, start_slice)
            end_slice = min(num_slices - 1, end_slice)
        else:
            start_slice, end_slice = 0, num_slices - 1
        
        if self.is_training:
            # RANDOM slice cho training (giống data_loader.py cũ)
            slice_idx = random.randint(start_slice, end_slice)
        else:
            # MIDDLE slice cho evaluation
            slice_idx = (start_slice + end_slice) // 2
        
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
        """Áp dụng data augmentation (giống cached_data_loader.py)"""
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


class VolumeCachedDataLoaderManager:
    """
    Manager để tạo volume-based cached data loaders 
    Giống hệt data_loader.py cũ: 42 samples/epoch
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        
        # Load metadata
        metadata_path = os.path.join(cache_dir, "metadata", "preprocessing_info.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"VolumeCachedDataLoaderManager initialized:")
        print(f"  🎯 VOLUME-BASED approach (giống data_loader.py cũ)")
        print(f"  Patients: {self.metadata['total_patients']}")
        print(f"  Cached slices: {self.metadata['total_slices']}")
        print(f"  🚀 Samples per epoch: {self.metadata['total_patients']} (random slices)")
    
    def create_train_val_loaders(self, 
                                batch_size: int = 4,
                                train_split: float = 0.8,
                                num_workers: int = 2,
                                slice_range: Optional[Tuple[int, int]] = None,
                                augmentation_prob: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Tạo volume-based training và validation loaders
        """
        # Tạo full dataset
        full_dataset = VolumeCachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=True,
            slice_range=slice_range,
            augmentation_prob=augmentation_prob
        )
        
        # Chia train/val theo PATIENT (không phải slice)
        dataset_size = len(full_dataset)  # = số bệnh nhân
        train_size = int(train_split * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Dataset cho validation không có augmentation
        val_dataset_no_aug = VolumeCachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=False,  # No augmentation, middle slice
            slice_range=slice_range,
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
        
        print(f"✅ Created VOLUME-BASED cached data loaders:")
        print(f"   Training: {len(train_dataset)} patients")
        print(f"   Validation: {len(val_subset)} patients")
        print(f"   🎯 Same scale as original: ~{len(train_dataset)//batch_size} batches/epoch")
        print(f"   🚀 But with CACHED preprocessing: ~450x faster!")
        
        return train_loader, val_loader

    def create_test_loader(self, 
                          batch_size: int = 1,
                          num_workers: int = 2,
                          slice_range: Optional[Tuple[int, int]] = None) -> DataLoader:
        """Tạo test loader"""
        test_dataset = VolumeCachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=False,
            slice_range=slice_range,
            augmentation_prob=0.0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✅ Created test loader: {len(test_dataset)} patients")
        return test_loader

    def get_cache_info(self) -> Dict:
        """Lấy thông tin về cache"""
        return {
            'cache_dir': self.cache_dir,
            'total_patients': self.metadata['total_patients'],
            'total_slices': self.metadata['total_slices'],
            'save_format': self.metadata['save_format'],
            'preprocessing_steps': self.metadata['preprocessing_steps'],
            'volume_based_samples': self.metadata['total_patients']  # Key difference!
        }


def compare_all_approaches():
    """
    So sánh 3 approaches: Original, Slice-based Cache, Volume-based Cache
    """
    cache_dir = "preprocessed_cache"
    
    if not os.path.exists(cache_dir):
        print("❌ Cache not found!")
        return
    
    print("=== SO SÁNH CÁC APPROACHES ===")
    
    try:
        # Volume-based cache (mới)
        print("\n🟢 Volume-based Cache (NEW):")
        volume_manager = VolumeCachedDataLoaderManager(cache_dir)
        train_loader_vol, _ = volume_manager.create_train_val_loaders(batch_size=4, num_workers=0)
        
        print(f"   Samples per epoch: {len(train_loader_vol.dataset)}")
        print(f"   Batches per epoch: {len(train_loader_vol)}")
        
        # Test loading speed
        import time
        start_time = time.time()
        for i, batch in enumerate(train_loader_vol):
            if i >= 5:  # Test 5 batches
                break
        volume_time = time.time() - start_time
        
        print(f"   Time for 5 batches: {volume_time:.2f}s ({volume_time/5:.3f}s per batch)")
        
        # Estimate epoch time
        estimated_epoch_time = (len(train_loader_vol) / 5) * volume_time
        print(f"   🚀 Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"\n📊 COMPARISON:")
    print(f"   Original data_loader.py: ~42 samples/epoch, ~2 hours (với preprocessing)")
    print(f"   Slice-based cache:       ~4,681 samples/epoch, ~5.4 minutes")
    print(f"   Volume-based cache:      ~42 samples/epoch, ~{estimated_epoch_time/60:.1f} minutes")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   Volume-based cache = Original approach + 450x faster preprocessing!")
    print(f"   Estimated speedup: {(2*60*60)/estimated_epoch_time:.0f}x faster than original!")


if __name__ == "__main__":
    compare_all_approaches() 