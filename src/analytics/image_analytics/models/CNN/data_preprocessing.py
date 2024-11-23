import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict, Tuple, Optional
from pathlib import Path
from enum import Enum

class ModelType(Enum):
    """Enum for different model types"""
    CNN = "cnn"
    YOLO = "yolo"
    VIT = "vit"
    ENSEMBLE = "ensemble"

class DatasetType(Enum):
    """Enum for different dataset splits"""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

class DatasetConfig:
    """Configuration for dataset paths and parameters"""
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.splits = {
            DatasetType.TRAIN: self.base_path / "train",
            DatasetType.VALID: self.base_path / "valid",
            DatasetType.TEST: self.base_path / "test"
        }
    
    def get_split_paths(self, split_type: DatasetType) -> Tuple[Path, Path]:
        """Get paths for images and classes CSV for a specific split"""
        split_path = self.splits[split_type]
        return split_path / "images", split_path / "_classes.csv"

class AugmentationFactory:
    """Factory class for creating model-specific augmentations"""
    
    @staticmethod
    def create_augmentation(model_type: ModelType, img_size: int = 224) -> Dict[str, A.Compose]:
        """Create augmentation pipeline based on model type"""
        
        base_train_augs = [
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.3),
        ]
        
        if model_type == ModelType.CNN:
            train_transform = A.Compose([
                *base_train_augs,
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        else:  # Ensemble - use CNN style as default
            train_transform = A.Compose([
                *base_train_augs,
                A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0)),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        # Validation transform is simpler and model-specific
        val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406] if model_type != ModelType.YOLO else [0, 0, 0],
                std=[0.229, 0.224, 0.225] if model_type != ModelType.YOLO else [1, 1, 1]
            ),
            ToTensorV2(),
        ])
        
        return {
            'train': train_transform,
            'valid': val_transform,
            'test': val_transform
        }

class ConstellationDataset(Dataset):
    """Enhanced dataset class supporting multiple model types"""
    
    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        transform: Optional[A.Compose] = None,
        model_type: ModelType = ModelType.CNN,
        split_type: DatasetType = DatasetType.TRAIN
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.model_type = model_type
        self.split_type = split_type
        
        # Get class columns (all except filename)
        self.class_columns = [col for col in self.df.columns if col != 'filename']
        
        # Convert labels to float32
        for col in self.class_columns:
            self.df[col] = self.df[col].astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.img_dir / self.df.iloc[idx]['filename']
        
        try:
            # Read image
            image = cv2.imread(str(img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get labels
            labels = self.df.iloc[idx][self.class_columns].values.astype(np.float32)
            
            # Apply transformations
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, torch.FloatTensor(labels)
            
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a black image and zeros for labels as fallback
            black_image = torch.zeros((3, 224, 224))
            zero_labels = torch.zeros(len(self.class_columns))
            return black_image, zero_labels
        

class DataModule:
    """Main data module handling all data operations"""
    
    def __init__(
        self,
        data_config: DatasetConfig,
        model_type: ModelType,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224
    ):
        self.data_config = data_config
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        
        # Create augmentations
        self.transforms = AugmentationFactory.create_augmentation(model_type, img_size)
        
        # Initialize datasets
        self.datasets = self._setup_datasets()
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
    
    def _setup_datasets(self) -> Dict[DatasetType, ConstellationDataset]:
        """Setup datasets for all splits"""
        datasets = {}
        for split_type in DatasetType:
            img_dir, csv_path = self.data_config.get_split_paths(split_type)
            if csv_path.exists() and img_dir.exists():
                datasets[split_type] = ConstellationDataset(
                    csv_path=csv_path,
                    img_dir=img_dir,
                    transform=self.transforms[split_type.value],
                    model_type=self.model_type,
                    split_type=split_type
                )
        return datasets
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights from training data"""
        train_dataset = self.datasets[DatasetType.TRAIN]
        class_counts = train_dataset.df[train_dataset.class_columns].sum()
        total_samples = len(train_dataset)
        
        weights = total_samples / (len(train_dataset.class_columns) * class_counts)
        weights = np.clip(weights, 0.1, 10.0)
        
        return torch.FloatTensor(weights)
    
    def get_dataloader(self, split_type: DatasetType) -> DataLoader:
        """Get dataloader for a specific split"""
        dataset = self.datasets[split_type]
        
        if split_type == DatasetType.TRAIN:
            # Create weighted sampler for training
            sample_weights = torch.zeros(len(dataset))
            for idx in range(len(dataset)):
                labels = dataset.df.iloc[idx][dataset.class_columns].values
                sample_weights[idx] = sum(self.class_weights[i] for i, label in enumerate(labels) if label == 1)
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True
            )
            
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )

def get_loss_function(model_type: ModelType, class_weights: torch.Tensor) -> nn.Module:
    """Get appropriate loss function based on model type"""
    return nn.BCEWithLogitsLoss(pos_weight=class_weights)

def main():
    """Example usage"""
    # Setup paths
    base_path = Path(r"data\constellation_dataset_1")
    
    # Create configuration
    data_config = DatasetConfig(base_path)
    
    # Example for CNN
    cnn_data = DataModule(
        data_config=data_config,
        model_type=ModelType.CNN,
        batch_size=32,
        num_workers=4
    )
    
    # Get dataloaders
    train_loader = cnn_data.get_dataloader(DatasetType.TRAIN)
    val_loader = cnn_data.get_dataloader(DatasetType.VALID)
    test_loader = cnn_data.get_dataloader(DatasetType.TEST)
    
    # Get loss function
    criterion = get_loss_function(ModelType.CNN, cnn_data.class_weights)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
