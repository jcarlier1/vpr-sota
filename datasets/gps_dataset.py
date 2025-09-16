"""
GPS-based Dataset Loader for Visual Place Recognition
Handles loading images with GPS coordinates from CSV files
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Dict
import math
from pathlib import Path


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the haversine distance between two GPS coordinates in meters.
    """
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


class GPSImageDataset(Dataset):
    """
    Dataset class for GPS-tagged images.
    """
    
    def __init__(
        self,
        csv_path: str,
        base_path: str,
        transform: Optional[transforms.Compose] = None,
        positive_threshold: float = 25.0,  # meters
        negative_threshold: float = 50.0,  # meters
        camera_filter: Optional[str] = None
    ):
        """
        Args:
            csv_path: Path to CSV file with image metadata
            base_path: Base path for images (will be prepended to img_relpath)
            transform: Image transforms to apply
            positive_threshold: Distance threshold for positive pairs (meters)
            negative_threshold: Distance threshold for negative pairs (meters)
            camera_filter: Filter by specific camera (e.g., 'front_left_center')
        """
        self.csv_path = csv_path
        self.base_path = Path(base_path)
        self.transform = transform
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        
        # Load and filter data
        self.data = pd.read_csv(csv_path)
        if camera_filter:
            self.data = self.data[self.data['camera_id'] == camera_filter]
        
        # Reset index after filtering
        self.data.reset_index(drop=True, inplace=True)
        
        # Create GPS coordinate array for fast distance computation
        self.gps_coords = self.data[['lat', 'lon']].values
        
        # Precompute distance matrix for efficient sampling
        self._precompute_distances()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _precompute_distances(self):
        """Precompute distance matrix for positive/negative sampling."""
        n = len(self.data)
        self.positive_pairs = {}
        self.negative_pairs = {}
        
        print(f"Precomputing distances for {n} images...")
        for i in range(n):
            positives = []
            negatives = []
            
            lat1, lon1 = self.gps_coords[i]
            
            for j in range(n):
                if i == j:
                    continue
                    
                lat2, lon2 = self.gps_coords[j]
                dist = haversine_distance(lat1, lon1, lat2, lon2)
                
                if dist <= self.positive_threshold:
                    positives.append(j)
                elif dist >= self.negative_threshold:
                    negatives.append(j)
            
            self.positive_pairs[i] = positives
            self.negative_pairs[i] = negatives
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a dictionary containing the image and metadata.
        """
        row = self.data.iloc[idx]
        
        # Load image
        img_path = self.base_path / row['img_relpath']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'idx': idx,
            'camera_id': row['camera_id'],
            'img_path': str(img_path),
            'timestamp': row['t_nsec'],
            'lat': row['lat'],
            'lon': row['lon'],
            'alt': row['alt'],
            'gps_coords': torch.tensor([row['lat'], row['lon']], dtype=torch.float32)
        }
    
    def get_positive_pairs(self, idx: int) -> List[int]:
        """Get indices of positive pairs for given index."""
        return self.positive_pairs.get(idx, [])
    
    def get_negative_pairs(self, idx: int) -> List[int]:
        """Get indices of negative pairs for given index."""
        return self.negative_pairs.get(idx, [])
    
    def sample_triplet(self, anchor_idx: int) -> Tuple[int, int, int]:
        """
        Sample a triplet (anchor, positive, negative) for given anchor.
        """
        positives = self.get_positive_pairs(anchor_idx)
        negatives = self.get_negative_pairs(anchor_idx)
        
        if not positives or not negatives:
            # Fallback: random sampling
            positive_idx = np.random.choice([i for i in range(len(self)) if i != anchor_idx])
            negative_idx = np.random.choice([i for i in range(len(self)) if i != anchor_idx and i != positive_idx])
        else:
            positive_idx = np.random.choice(positives)
            negative_idx = np.random.choice(negatives)
        
        return anchor_idx, positive_idx, negative_idx


class TripletDataset(Dataset):
    """
    Dataset that returns triplets (anchor, positive, negative) for training.
    """
    
    def __init__(self, base_dataset: GPSImageDataset):
        self.base_dataset = base_dataset
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Returns triplet data."""
        anchor_idx, positive_idx, negative_idx = self.base_dataset.sample_triplet(idx)
        
        anchor = self.base_dataset[anchor_idx]
        positive = self.base_dataset[positive_idx]
        negative = self.base_dataset[negative_idx]
        
        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative
        }


def create_datasets(
    train_csv: str,
    test_csv: str,
    train_base_path: str,
    test_base_path: str,
    camera_filter: Optional[str] = None,
    positive_threshold: float = 25.0,
    negative_threshold: float = 50.0
) -> Tuple[GPSImageDataset, GPSImageDataset]:
    """
    Create train and test datasets.
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = GPSImageDataset(
        csv_path=train_csv,
        base_path=train_base_path,
        transform=train_transform,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        camera_filter=camera_filter
    )
    
    test_dataset = GPSImageDataset(
        csv_path=test_csv,
        base_path=test_base_path,
        transform=test_transform,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
        camera_filter=camera_filter
    )
    
    return train_dataset, test_dataset


def create_dataloaders(
    train_dataset: GPSImageDataset,
    test_dataset: GPSImageDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    triplet_training: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    """
    
    if triplet_training:
        train_dataset = TripletDataset(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_csv = "/media/pragyan/Data/racecar_ws/output/sequences/M-SOLO-SLOW-70-100/poses/poses_minimal_coverage_0.5m.csv"
    test_csv = "/media/pragyan/Data/racecar_ws/output/sequences/M-MULTI-SLOW-KAIST/poses/poses.csv"
    
    # Extract base paths from CSV paths
    train_base_path = os.path.dirname(os.path.dirname(train_csv))
    test_base_path = os.path.dirname(os.path.dirname(test_csv))
    
    print(f"Train base path: {train_base_path}")
    print(f"Test base path: {test_base_path}")
    
    train_dataset, test_dataset = create_datasets(
        train_csv=train_csv,
        test_csv=test_csv,
        train_base_path=train_base_path,
        test_base_path=test_base_path,
        camera_filter="front_left_center"
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test data loading
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=4)
    
    for batch in train_loader:
        print(f"Batch shape: {batch['image'].shape}")
        print(f"GPS coords shape: {batch['gps_coords'].shape}")
        break
