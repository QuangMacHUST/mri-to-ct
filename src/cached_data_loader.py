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

class CachedMRIToCTDataset(data.Dataset):
    """
    Fast Dataset loading từ preprocessed cache.
    Chỉ thực hiện augmentation, không preprocessing nặng.
    """
    
    def __init__(self, 
                 cache_dir: str,
                 is_training: bool = True,
                 slice_range: Optional[Tuple[int, int]] = None,
                 augmentation_prob: float = 0.8):
        """
        Args:
            cache_dir: thư mục chứa preprocessed cache
            is_training: training mode (có augmentation) hay evaluation mode
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
        
        # Tạo danh sách tất cả slices có thể training
        self.slice_list = self._build_slice_list()
        
        print(f"CachedMRIToCTDataset initialized:")
        print(f"  Cache directory: {cache_dir}")
        print(f"  Save format: {self.save_format}")
        print(f"  Patients: {self.metadata['total_patients']}")
        print(f"  Available slices: {len(self.slice_list)}")
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
    
    def _build_slice_list(self) -> List[Tuple[str, int]]:
        """
        Tạo danh sách (patient_id, slice_idx) cho tất cả slices có thể training
        """
        slice_list = []
        
        for patient_meta in self.metadata['metadata']:
            patient_id = patient_meta['patient_id']
            num_slices = patient_meta['num_slices']
            
            if self.slice_range:
                start_slice, end_slice = self.slice_range
                start_slice = max(0, start_slice)
                end_slice = min(num_slices - 1, end_slice)
            else:
                start_slice, end_slice = 0, num_slices - 1
            
            # Thêm tất cả slices hợp lệ
            for slice_idx in range(start_slice, end_slice + 1):
                slice_list.append((patient_id, slice_idx))
        
        return slice_list
    
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
        
        # === AUGMENTATION (chỉ khi training và random) - cuối cùng để thống nhất ===
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
        """
        Load preprocessed volume từ cache (H5 hoặc PyTorch)
        """
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
        """
        Áp dụng data augmentation nhanh trên range [-1,1]
        Data đã được resize, normalize và convert về [-1,1] rồi
        """
        # Random rotation (±15°)
        if random.random() > 0.5:
            from scipy import ndimage
            angle = random.uniform(-15, 15)
            mri_slice = ndimage.rotate(mri_slice, angle, reshape=False, mode='constant', cval=-1)  # Background = -1
            ct_slice = ndimage.rotate(ct_slice, angle, reshape=False, mode='constant', cval=-1)   # Background = -1
        
        # Random flip horizontal
        if random.random() > 0.5:
            mri_slice = np.fliplr(mri_slice).copy()
            ct_slice = np.fliplr(ct_slice).copy()
            
        # Random flip vertical
        if random.random() > 0.5:
            mri_slice = np.flipud(mri_slice).copy()
            ct_slice = np.flipud(ct_slice).copy()
        
        # Random intensity scaling cho MRI (±10%) - chỉ trong vùng có data
        if random.random() > 0.5:
            scale_factor = random.uniform(0.9, 1.1)
            mask = mri_slice > -0.8  # Detect brain region (trong range [-1,1])
            mri_slice[mask] = mri_slice[mask] * scale_factor
            mri_slice = np.clip(mri_slice, -1, 1)  # Clip về [-1,1]
        
        # Đảm bảo vẫn trong range [-1,1]
        mri_slice = np.clip(mri_slice, -1, 1)
        ct_slice = np.clip(ct_slice, -1, 1)
        
        return mri_slice, ct_slice


class CachedDataLoaderManager:
    """
    Manager để tạo cached data loaders cho training/validation/test
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        
        # Verify cache exists
        if not os.path.exists(cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
        
        # Load metadata
        metadata_path = os.path.join(cache_dir, "metadata", "preprocessing_info.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"CachedDataLoaderManager initialized with {self.metadata['total_patients']} patients")
    
    def create_train_val_loaders(self, 
                                batch_size: int = 4,
                                train_split: float = 0.8,
                                num_workers: int = 2,
                                slice_range: Optional[Tuple[int, int]] = None,
                                augmentation_prob: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """
        Tạo training và validation loaders
        """
        # Tạo full dataset
        full_dataset = CachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=True,
            slice_range=slice_range,
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
        val_dataset_no_aug = CachedMRIToCTDataset(
            cache_dir=self.cache_dir,
            is_training=False,  # No augmentation cho validation
            slice_range=slice_range,
            augmentation_prob=0.0
        )
        
        # Lấy validation indices để tạo subset
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
        
        print(f"✅ Created cached data loaders:")
        print(f"   Training: {len(train_dataset)} slices")
        print(f"   Validation: {len(val_subset)} slices")
        print(f"   Batch size: {batch_size}")
        print(f"   Num workers: {num_workers}")
        
        return train_loader, val_loader
    
    def create_test_loader(self, 
                          batch_size: int = 1,
                          num_workers: int = 2,
                          slice_range: Optional[Tuple[int, int]] = None) -> DataLoader:
        """
        Tạo test loader (không có augmentation)
        """
        test_dataset = CachedMRIToCTDataset(
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
        
        print(f"✅ Created test loader: {len(test_dataset)} slices")
        return test_loader

    def get_cache_info(self) -> Dict:
        """
        Lấy thông tin về cache
        """
        return {
            'cache_dir': self.cache_dir,
            'total_patients': self.metadata['total_patients'],
            'total_slices': self.metadata['total_slices'],
            'save_format': self.metadata['save_format'],
            'preprocessing_steps': self.metadata['preprocessing_steps'],
            'cache_size_mb': self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Tính size của cache directory"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)  # Convert to MB


def test_cached_loader_speed():
    """
    Test function để so sánh tốc độ load cache vs preprocessing
    """
    cache_dir = "preprocessed_cache"
    
    if not os.path.exists(cache_dir):
        print("❌ Cache not found. Run preprocessing first!")
        return
    
    print("=== TESTING CACHED LOADER SPEED ===")
    
    # Tạo loader manager
    loader_manager = CachedDataLoaderManager(cache_dir)
    
    # In thông tin cache
    cache_info = loader_manager.get_cache_info()
    print(f"Cache info: {cache_info}")
    
    # Tạo test loader
    test_loader = loader_manager.create_test_loader(batch_size=4, num_workers=2)
    
    # Test loading speed
    import time
    
    print("\n🚀 Testing loading speed...")
    start_time = time.time()
    
    for i, batch in enumerate(test_loader):
        if i >= 10:  # Test 10 batches
            break
        print(f"  Batch {i+1}: MRI {batch['mri'].shape}, CT {batch['ct'].shape}")
    
    elapsed = time.time() - start_time
    print(f"\n⚡ Loaded 10 batches in {elapsed:.2f}s ({elapsed/10:.3f}s per batch)")
    print("   This should be much faster than original preprocessing!")


if __name__ == "__main__":
    test_cached_loader_speed() 