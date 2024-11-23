import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models
import pandas as pd
import numpy as np
from PIL import Image
import os

class ConstellationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Read CSV file with explicit numeric types for all label columns
        dtype_dict = {col: np.float32 for col in ['aquila', 'bootes', 'canis_major', 'canis_minor', 
                                                 'cassiopeia', 'cygnus', 'gemini', 'leo', 'lyra', 
                                                 'moon', 'orion', 'pleiades', 'sagittarius', 
                                                 'scorpius', 'taurus', 'ursa_major']}
        
        self.labels_frame = pd.read_csv(csv_file)
        # Convert the label columns to float32
        for col in self.labels_frame.columns[1:]:  # Skip the filename column
            self.labels_frame[col] = self.labels_frame[col].astype(np.float32)
            
        self.img_dir = img_dir
        self.transform = transform
        self.classes = ['aquila', 'bootes', 'canis_major', 'canis_minor', 
                       'cassiopeia', 'cygnus', 'gemini', 'leo', 'lyra', 
                       'moon', 'orion', 'pleiades', 'sagittarius', 
                       'scorpius', 'taurus', 'ursa_major']

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a black image of the expected size if there's an error
            image = Image.new('RGB', (224, 224), 'black')
        
        # Convert labels to float32 tensor
        labels = self.labels_frame.iloc[idx, 1:].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {img_name}: {str(e)}")
                # Return a transformed black image
                black_image = Image.new('RGB', (224, 224), 'black')
                image = self.transform(black_image)
        
        return image, labels

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict
import timm

class ConstellationCNN(nn.Module):
    """
    Custom CNN for constellation classification using ResNet50 as backbone
    with modifications for multi-label classification
    """
    def __init__(
        self,
        num_classes: int = 16,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        backbone: str = 'resnet50'
    ):
        super(ConstellationCNN, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = backbone
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.feature_extractor = models.resnet50(pretrained=pretrained)
            num_features = self.feature_extractor.fc.in_features
            # Remove the final FC layer
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        elif backbone == 'efficientnet_b0':
            self.feature_extractor = timm.create_model('efficientnet_b0', pretrained=pretrained)
            num_features = self.feature_extractor.num_features
            # Remove the final classifier
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Custom classifier head for multi-label classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Attention module
        self.attention = SpatialAttention()
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def _initialize_weights(self):
        """Initialize the weights of the classifier"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global average pooling
        features = self.gap(features)
        
        # Flatten
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        
        return output

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor
        Returns:
            Attention-weighted tensor
        """
        # Calculate attention weights
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        weights = self.conv(avg_pool)
        
        # Apply attention
        return x * weights

def create_model(
    model_type: str,
    num_classes: int = 16,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('cnn', 'vit', etc.)
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    """
    if model_type.lower() == 'cnn':
        model = ConstellationCNN(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone=kwargs.get('backbone', 'resnet50'),
            dropout_rate=kwargs.get('dropout_rate', 0.5)
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def get_model_info(model: nn.Module) -> Dict:
    """Get model information and statistics"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'total_parameters': num_params,
        'trainable_parameters': num_trainable_params,
        'model_type': model.__class__.__name__,
        'input_size': (224, 224),  # Default size
        'output_size': model.num_classes if hasattr(model, 'num_classes') else None
    }
    
    return model_info