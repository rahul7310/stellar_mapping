import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ConstellationDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        processor: ViTImageProcessor,
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.transform = transform
        self.is_training = is_training
        
        # Get class columns (all except filename)
        self.class_columns = [col for col in self.df.columns if col != 'filename']
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_dir / self.df.iloc[idx]['filename']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), 'black')

        # Get labels
        labels = self.df.iloc[idx][self.class_columns].values.astype(np.float32)
        
        # Apply transforms first if available
        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']
        
        # Then process with ViT processor
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=False if self.transform else True,  # Skip resize if already transformed
            do_normalize=True,  # Always normalize
        )
        
        # Squeeze batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        
        # Add labels
        inputs['labels'] = torch.tensor(labels, dtype=torch.float)
        
        return inputs
class DataModule:
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        train_img_dir: str,
        val_img_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        model_name: str = "google/vit-base-patch16-224-in21k"
    ):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize ViT processor
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Setup augmentations
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self) -> A.Compose:
        return A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            ToTensorV2(),  # Remove Normalize transform since ViT processor handles normalization
        ])
    
    def _get_val_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(224, 224),
            ToTensorV2(),  # Remove Normalize transform since ViT processor handles normalization
        ])
    
    def setup(self):
        """Initialize datasets"""
        self.train_dataset = ConstellationDataset(
            csv_file=self.train_csv,
            img_dir=self.train_img_dir,  # Fixed from self.img_dir
            processor=self.processor,
            transform=self.train_transform,
            is_training=True
        )
        
        self.val_dataset = ConstellationDataset(
            csv_file=self.val_csv,
            img_dir=self.val_img_dir,  # Fixed from self.img_dir
            processor=self.processor,
            transform=self.val_transform,
            is_training=False
        )
        
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.train_dataset.class_columns
