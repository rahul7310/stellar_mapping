import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from typing import Dict, List

class ConstellationViT(nn.Module):
    def __init__(
        self, 
        num_labels: int,
        model_name: str = "google/vit-base-patch16-224-in21k",
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pre-trained ViT
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Modify classifier for multi-label classification
        self.vit.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.vit.config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(512, num_labels)
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vit(pixel_values)
        return outputs.logits
    
    def save_pretrained(self, path: str):
        """Save model and config"""
        self.vit.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, num_labels: int):
        """Load saved model"""
        model = cls(num_labels=num_labels)
        model.vit = ViTForImageClassification.from_pretrained(path)
        return model

def create_model(config: Dict) -> ConstellationViT:
    """Create model from config"""
    return ConstellationViT(
        num_labels=config['num_labels'],
        model_name=config['model_name'],
        dropout=config['dropout']
    )

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
