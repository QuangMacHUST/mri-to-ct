import numpy as np
import os
import sys
import pickle
import h5py
from tqdm import tqdm
import SimpleITK as sitk
from typing import Tuple, Dict, List
import torch

sys.path.append('src')
from data_loader import MRIToCTDataset

class PreprocessingCacher:
    """
    Class để preprocess tất cả dữ liệu một lần và lưu cache
    """
    
    def __init__(self, mri_dir: str, ct_dir: str, cache_dir: str = "preprocessed_cache"):
        """
        Args:
            mri_dir: thư mục MRI gốc
            ct_dir: thư mục CT gốc  
            cache_dir: thư mục lưu cache
        """
        self.mri_dir = mri_dir
        self.ct_dir = ct_dir
        self.cache_dir = cache_dir
        
        # Tạo thư mục cache
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "mri_volumes"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "ct_volumes"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "metadata"), exist_ok=True)
        
        # Tạo dataset instance để sử dụng preprocessing methods
        self.dataset = MRIToCTDataset(mri_dir, ct_dir, is_training=False)
        
    def preprocess_all_volumes(self, save_format: str = "h5"):
        """
        Preprocess tất cả volumes và lưu cache
        
        Args:
            save_format: 'h5' hoặc 'pt' (torch tensors)
        """
        print("=== PREPROCESSING AND CACHING ALL VOLUMES ===")
        print(f"Found {len(self.dataset.mri_files)} patient volumes")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Save format: {save_format}")
        
        # Danh sách để lưu metadata
        metadata = []
        
        for idx in tqdm(range(len(self.dataset.mri_files)), desc="Preprocessing volumes"):
            filename = self.dataset.mri_files[idx]
            patient_id = filename.replace('.nii.gz', '')
            
            print(f"\nProcessing {filename}...")
            
            # === BƯỚC 1-6: HEAVY PREPROCESSING (chỉ làm 1 lần) ===
            mri_path = os.path.join(self.mri_dir, filename)
            ct_path = os.path.join(self.ct_dir, filename)
            
            # Load và N4 correction
            mri_sitk = sitk.ReadImage(mri_path)
            ct_sitk = sitk.ReadImage(ct_path)
            mri_sitk = self.dataset._apply_n4_bias_correction(mri_sitk)
            
            mri_array = sitk.GetArrayFromImage(mri_sitk).astype(np.float32)
            ct_array = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)
            
            # Create brain+skull mask
            mri_mask = self.dataset._create_brain_with_skull_mask(mri_array)
            
            # Apply mask to CT
            ct_array = self.dataset._apply_mri_mask_to_ct(ct_array, mri_mask)
            
            # Outlier clipping
            mri_array = self.dataset._gentle_outlier_clipping(mri_array, mri_mask, 'MRI')
            ct_array = self.dataset._gentle_outlier_clipping(ct_array, mri_mask, 'CT')
            
            # Intensity normalization
            mri_array = self.dataset._normalize_intensity(mri_array, mri_mask, 'MRI')
            ct_array = self.dataset._normalize_intensity(ct_array, mri_mask, 'CT')
            
            # Crop brain ROI
            mri_array, mri_mask_cropped = self.dataset._crop_brain_roi(mri_array, mri_mask)
            ct_array, _ = self.dataset._crop_brain_roi(ct_array, mri_mask)
            
            # === RESIZE TẤT CẢ SLICES ===
            # Resize tất cả slices về 256x256 để sẵn sàng training
            mri_resized = self._resize_all_slices(mri_array, target_size=(256, 256))
            ct_resized = self._resize_all_slices(ct_array, target_size=(256, 256))
            
            # Clamp về [0,1] 
            mri_resized = np.clip(mri_resized, 0, 1)
            ct_resized = np.clip(ct_resized, 0, 1)
            
            print(f"  Preprocessed shape: {mri_resized.shape}")
            print(f"  MRI range: [{mri_resized.min():.3f}, {mri_resized.max():.3f}]")
            print(f"  CT range: [{ct_resized.min():.3f}, {ct_resized.max():.3f}]")
            
            # === LƯU CACHE ===
            if save_format == "h5":
                self._save_h5(patient_id, mri_resized, ct_resized)
            elif save_format == "pt":
                self._save_pytorch(patient_id, mri_resized, ct_resized)
            
            # Lưu metadata
            metadata.append({
                'patient_id': patient_id,
                'filename': filename,
                'original_shape': mri_array.shape,
                'processed_shape': mri_resized.shape,
                'mri_range': [mri_resized.min(), mri_resized.max()],
                'ct_range': [ct_resized.min(), ct_resized.max()],
                'num_slices': mri_resized.shape[0]
            })
        
        # Lưu metadata
        self._save_metadata(metadata, save_format)
        
        print(f"\n✅ PREPROCESSING COMPLETE!")
        print(f"   Processed {len(metadata)} patients")
        print(f"   Cache saved to: {self.cache_dir}")
        print(f"   Total slices: {sum(m['num_slices'] for m in metadata)}")
        
        return metadata
    
    def _resize_all_slices(self, volume: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize tất cả slices trong volume về target_size
        """
        from scipy.ndimage import zoom
        
        if volume.shape[1:] == target_size:
            return volume
            
        # Tính zoom factors
        zoom_h = target_size[0] / volume.shape[1]
        zoom_w = target_size[1] / volume.shape[2]
        
        # Resize toàn bộ volume
        resized_volume = zoom(volume, (1, zoom_h, zoom_w), order=1, mode='constant', cval=0)
        
        return resized_volume
    
    def _save_h5(self, patient_id: str, mri_volume: np.ndarray, ct_volume: np.ndarray):
        """
        Lưu cache dưới dạng HDF5 (nhanh cho I/O)
        """
        h5_path = os.path.join(self.cache_dir, f"{patient_id}.h5")
        
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('mri', data=mri_volume, compression='gzip', compression_opts=9)
            f.create_dataset('ct', data=ct_volume, compression='gzip', compression_opts=9)
            f.attrs['patient_id'] = patient_id
            f.attrs['shape'] = mri_volume.shape
    
    def _save_pytorch(self, patient_id: str, mri_volume: np.ndarray, ct_volume: np.ndarray):
        """
        Lưu cache dưới dạng PyTorch tensors (nhanh cho loading)
        """
        # Convert to tensors
        mri_tensor = torch.tensor(mri_volume, dtype=torch.float32)
        ct_tensor = torch.tensor(ct_volume, dtype=torch.float32)
        
        # Save
        torch.save({
            'mri': mri_tensor,
            'ct': ct_tensor,
            'patient_id': patient_id,
            'shape': mri_volume.shape
        }, os.path.join(self.cache_dir, f"{patient_id}.pt"))
    
    def _save_metadata(self, metadata: List[Dict], save_format: str):
        """
        Lưu metadata về preprocessing
        """
        metadata_path = os.path.join(self.cache_dir, "metadata", "preprocessing_info.pkl")
        
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'save_format': save_format,
                'total_patients': len(metadata),
                'total_slices': sum(m['num_slices'] for m in metadata),
                'preprocessing_steps': [
                    'N4 bias correction',
                    'Brain+skull mask creation', 
                    'MRI-guided CT artifact removal',
                    'Outlier clipping',
                    'Intensity normalization',
                    'Brain ROI cropping',
                    'Resize to 256x256',
                    'Range clamping [0,1]'
                ]
            }, f)
        
        print(f"   Metadata saved: {metadata_path}")


def main():
    """
    Main function để chạy preprocessing
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess and cache MRI-CT data")
    parser.add_argument('--mri_dir', type=str, default='data/MRI', help='MRI directory')
    parser.add_argument('--ct_dir', type=str, default='data/CT', help='CT directory') 
    parser.add_argument('--cache_dir', type=str, default='preprocessed_cache', help='Cache directory')
    parser.add_argument('--format', type=str, choices=['h5', 'pt'], default='h5', 
                       help='Save format: h5 (HDF5) or pt (PyTorch)')
    
    args = parser.parse_args()
    
    # Tạo preprocessor
    preprocessor = PreprocessingCacher(
        mri_dir=args.mri_dir,
        ct_dir=args.ct_dir, 
        cache_dir=args.cache_dir
    )
    
    # Chạy preprocessing
    metadata = preprocessor.preprocess_all_volumes(save_format=args.format)
    
    print(f"\n🚀 READY FOR FAST TRAINING!")
    print(f"   Use CachedMRIToCTDataset with cache_dir='{args.cache_dir}'")
    print(f"   Training will be much faster now!")


if __name__ == "__main__":
    main() 