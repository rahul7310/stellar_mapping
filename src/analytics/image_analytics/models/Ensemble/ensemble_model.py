import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision import models
from typing import Dict, Tuple
import timm

class CNNBackbone(nn.Module):
    """CNN backbone using ResNet50 with custom modifications"""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        backbone: str = 'resnet50'
    ):
        super().__init__()
        
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
        
        # Custom classifier head
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
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both features and logits"""
        # Extract features
        features = self.feature_extractor(x)
        features = self.attention(features)
        features = self.gap(features)
        features = features.view(features.size(0), -1)
        
        # Get logits
        logits = self.classifier(features)
        
        return features, logits

class ViTBackbone(nn.Module):
    """ViT backbone with custom head"""
    
    def __init__(
        self,
        num_classes: int,
        model_name: str = "google/vit-base-patch16-224-in21k",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Load pre-trained ViT
        self.vit = ViTModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both features and logits"""
        # Get ViT features
        outputs = self.vit(pixel_values)
        features = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        
        # Get logits
        logits = self.classifier(features)
        
        return features, logits

class SpatialAttention(nn.Module):
    """Spatial attention module for CNN backbone"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        weights = self.conv(avg_pool)
        return x * weights

class EnsembleModel(nn.Module):
    """Ensemble model combining CNN and ViT with learnable weights"""
    
    def __init__(
        self,
        num_classes: int,
        cnn_config: Dict,
        vit_config: Dict,
        ensemble_dropout: float = 0.3
    ):
        super().__init__()
        
        # Initialize backbones
        self.cnn = CNNBackbone(
            num_classes=num_classes,
            pretrained=cnn_config.get('pretrained', True),
            dropout_rate=cnn_config.get('dropout_rate', 0.5),
            backbone=cnn_config.get('backbone', 'resnet50')
        )
        
        self.vit = ViTBackbone(
            num_classes=num_classes,
            model_name=vit_config.get('model_name', "google/vit-base-patch16-224-in21k"),
            dropout_rate=vit_config.get('dropout_rate', 0.1)
        )
        
        # Get feature dimensions from configs
        cnn_feat_dim = cnn_config.get('feature_dim', 2048)  # ResNet50 default
        vit_feat_dim = vit_config.get('feature_dim', 768)   # ViT base default
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)
        
        # Feature projections to match dimensions before fusion
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(ensemble_dropout)
        )
        
        self.vit_projection = nn.Sequential(
            nn.Linear(vit_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(ensemble_dropout)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(1024, 512),  # 1024 = 512 + 512 (projected dimensions)
            nn.ReLU(),
            nn.Dropout(ensemble_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(ensemble_dropout/2),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(
        self,
        cnn_input: torch.Tensor,
        vit_input: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multiple prediction strategies"""
        # Get predictions from both models
        cnn_features, cnn_logits = self.cnn(cnn_input)
        vit_features, vit_logits = self.vit(vit_input)
        
        # Project features to same dimension
        cnn_proj = self.cnn_projection(cnn_features)
        vit_proj = self.vit_projection(vit_features)
        
        # 1. Weighted average of logits
        ensemble_weights = self.softmax(self.ensemble_weights)
        weighted_logits = ensemble_weights[0] * cnn_logits + ensemble_weights[1] * vit_logits
        
        # 2. Feature fusion
        combined_features = torch.cat([cnn_proj, vit_proj], dim=1)
        fusion_logits = self.fusion_layer(combined_features)
        
        output = {
            'ensemble_logits': weighted_logits,
            'fusion_logits': fusion_logits,
            'cnn_logits': cnn_logits,
            'vit_logits': vit_logits,
            'ensemble_weights': ensemble_weights
        }
        
        if return_features:
            output.update({
                'cnn_features': cnn_features,
                'vit_features': vit_features,
                'cnn_proj_features': cnn_proj,
                'vit_proj_features': vit_proj,
                'combined_features': combined_features
            })
        
        return output
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        return {
            'num_classes': self.num_classes,
            'cnn_params': sum(p.numel() for p in self.cnn.parameters()),
            'vit_params': sum(p.numel() for p in self.vit.parameters()),
            'fusion_params': sum(p.numel() for p in self.fusion_layer.parameters()),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'ensemble_weights': self.softmax(self.ensemble_weights).tolist()
        }

def create_ensemble_model(config: Dict) -> EnsembleModel:
    """Factory function to create ensemble model"""
    model = EnsembleModel(
        num_classes=config['num_classes'],
        cnn_config=config['cnn_config'],
        vit_config=config['vit_config'],
        ensemble_dropout=config.get('ensemble_dropout', 0.3)
    )
    return model

def main():
    """Example usage"""
    config = {
        'num_classes': 16,
        'cnn_config': {
            'pretrained': True,
            'dropout_rate': 0.5,
            'backbone': 'resnet50'
        },
        'vit_config': {
            'model_name': "google/vit-base-patch16-224-in21k",
            'dropout_rate': 0.1
        },
        'ensemble_dropout': 0.3
    }
    
    # Create model
    model = create_ensemble_model(config)
    
    # Print model info
    model_info = model.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    
    # Test forward pass
    batch_size = 4
    cnn_input = torch.randn(batch_size, 3, 224, 224)
    vit_input = torch.randn(batch_size, 3, 224, 224)
    
    outputs = model(cnn_input, vit_input, return_features=True)
    
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

if __name__ == "__main__":
    main()