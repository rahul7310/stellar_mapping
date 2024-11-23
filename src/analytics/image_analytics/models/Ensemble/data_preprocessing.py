import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import ViTImageProcessor
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from enum import Enum

class ModelType(Enum):
    """Enum for different model types in ensemble"""
    CNN = "cnn"
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

class EnsembleDataset(Dataset):
    """Dataset class supporting ensemble model with both CNN and ViT preprocessing"""
    
    def __init__(
        self,
        csv_path: Path,
        img_dir: Path,
        model_type: ModelType,
        transform: Union[A.Compose, Dict[str, A.Compose]],
        vit_processor: Optional[ViTImageProcessor] = None,
        is_training: bool = True
    ):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.model_type = model_type
        self.transform = transform
        self.vit_processor = vit_processor
        self.is_training = is_training
        
        # Get class columns (all except filename)
        self.class_columns = [col for col in self.df.columns if col != 'filename']
        
        # Convert labels to float32
        for col in self.class_columns:
            self.df[col] = self.df[col].astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_dir / self.df.iloc[idx]['filename']
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_array = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image_array = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Get labels
        labels = self.df.iloc[idx][self.class_columns].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float)
        
        if self.model_type == ModelType.ENSEMBLE:
            # Apply both CNN and ViT transforms
            cnn_transformed = self.transform['cnn'](image=image_array)
            vit_transformed = self.transform['vit'](image=image_array)
            
            return {
                'cnn_input': cnn_transformed['image'],
                'vit_input': vit_transformed['image'],
                'labels': labels
            }
            
        else:  # Single model (CNN or ViT)
            transformed = self.transform(image=image_array)
            return {
                'pixel_values': transformed['image'],
                'labels': labels
            }

class DataModule:
    """Data module handling both CNN and ViT data processing"""
    
    def __init__(
        self,
        data_config: DatasetConfig,
        model_type: ModelType,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 224,
        vit_model_name: Optional[str] = "google/vit-base-patch16-224-in21k"
    ):
        self.data_config = data_config
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Initialize ViT processor if needed
        self.vit_processor = None
        if model_type in [ModelType.VIT, ModelType.ENSEMBLE]:
            self.vit_processor = ViTImageProcessor.from_pretrained(vit_model_name)
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
        # Initialize datasets
        self.datasets = self._setup_datasets()
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
    
    def _setup_transforms(self) -> Dict[DatasetType, Dict[ModelType, A.Compose]]:
        """Setup transforms for all model types"""
        transform_dict = {}
        
        for split_type in DatasetType:
            is_train = split_type == DatasetType.TRAIN
            
            # CNN transforms
            cnn_transform = A.Compose([
                A.RandomResizedCrop(self.image_size, self.image_size) if is_train 
                else A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
                A.ShiftScaleRotate(p=0.3) if is_train else A.NoOp(),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3) if is_train else A.NoOp(),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # ViT transforms (same transforms, different normalization)
            vit_transform = A.Compose([
                A.RandomResizedCrop(self.image_size, self.image_size) if is_train 
                else A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5) if is_train else A.NoOp(),
                A.ShiftScaleRotate(p=0.3) if is_train else A.NoOp(),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3) if is_train else A.NoOp(),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ViT normalization
                ToTensorV2()
            ])
            
            # For ensemble, use both transforms
            transform_dict[split_type] = {
                ModelType.CNN: cnn_transform,
                ModelType.VIT: vit_transform,
                ModelType.ENSEMBLE: {
                    'cnn': cnn_transform,
                    'vit': vit_transform
                }
            }
        
        return transform_dict
    
    def _setup_datasets(self) -> Dict[DatasetType, EnsembleDataset]:
        """Setup datasets for all splits"""
        datasets = {}
        
        for split_type in DatasetType:
            img_dir, csv_path = self.data_config.get_split_paths(split_type)
            if csv_path.exists() and img_dir.exists():
                transforms = self.transforms[split_type][self.model_type]
                datasets[split_type] = EnsembleDataset(
                    csv_path=csv_path,
                    img_dir=img_dir,
                    model_type=self.model_type,
                    transform=transforms,
                    vit_processor=self.vit_processor,
                    is_training=split_type == DatasetType.TRAIN
                )
        
        return datasets
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights from training data"""
        train_dataset = self.datasets[DatasetType.TRAIN]
        class_counts = train_dataset.df[train_dataset.class_columns].sum()
        total_samples = len(train_dataset)
        
        weights = total_samples / (len(train_dataset.class_columns) * class_counts)
        weights = np.clip(weights, 0.1, 10.0)  # Clip weights to prevent extreme values
        
        return torch.FloatTensor(weights)
    
    def get_dataloader(
        self,
        split_type: DatasetType,
        weighted_sampling: bool = True
    ) -> DataLoader:
        """Get dataloader for a specific split"""
        dataset = self.datasets[split_type]
        
        if split_type == DatasetType.TRAIN and weighted_sampling:
            # Create weighted sampler for training
            sample_weights = torch.zeros(len(dataset))
            for idx in range(len(dataset)):
                labels = dataset.df.iloc[idx][dataset.class_columns].values
                sample_weights[idx] = sum(self.class_weights[i] for i, label in enumerate(labels) if label > 0)
            
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
    
    def get_all_dataloaders(self) -> Dict[DatasetType, DataLoader]:
        """Get all dataloaders"""
        return {
            split_type: self.get_dataloader(split_type)
            for split_type in DatasetType
        }

def get_loss_function(class_weights: torch.Tensor) -> nn.Module:
    """Get appropriate loss function"""
    return nn.BCEWithLogitsLoss(pos_weight=class_weights)

def main():
    """Example usage"""
    # Setup paths
    base_path = Path(r"data\constellation_dataset_1")
    
    # Create configuration
    data_config = DatasetConfig(base_path)
    
    # Create data module for ensemble
    data_module = DataModule(
        data_config=data_config,
        model_type=ModelType.ENSEMBLE,
        batch_size=32,
        num_workers=4,
        image_size=224,
        vit_model_name="google/vit-base-patch16-224-in21k"
    )
    
    # Get dataloaders
    dataloaders = data_module.get_all_dataloaders()
    
    # Get loss function
    criterion = get_loss_function(data_module.class_weights)
    
    print(f"Training batches: {len(dataloaders[DatasetType.TRAIN])}")
    print(f"Validation batches: {len(dataloaders[DatasetType.VALID])}")
    print(f"Test batches: {len(dataloaders[DatasetType.TEST])}")

if __name__ == "__main__":
    main()