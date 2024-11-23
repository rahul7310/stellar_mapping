import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = config.get('wandb', {}).get('use_wandb', False)
        if self.use_wandb and not wandb.run:
            wandb.init(
                project=config['wandb']['project'],
                name=config.get('experiment_name', 'vit_training'),
                config=config
            )
        # Setup loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config['learning_rate']),
            weight_decay=float(config['weight_decay'])
        )
        
        # Setup scheduler
        num_training_steps = len(train_loader) * int(config['num_epochs'])
        num_warmup_steps = int(num_training_steps * float(config['warmup_ratio']))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Initialize tracking
        self.best_val_map = 0.0
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(pixel_values)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            np.array(all_preds),
            np.array(all_labels)
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(pixel_values)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        metrics = self.calculate_metrics(
            np.array(all_preds),
            np.array(all_labels)
        )
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def calculate_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        binary_preds = (preds > 0.5).astype(int)
        
        # Per-class metrics
        per_class_metrics = {}
        for i in range(labels.shape[1]):
            tp = np.sum((labels[:, i] == 1) & (binary_preds[:, i] == 1))
            fp = np.sum((labels[:, i] == 0) & (binary_preds[:, i] == 1))
            fn = np.sum((labels[:, i] == 1) & (binary_preds[:, i] == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Overall metrics
        map_score = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
        exact_match = np.mean(np.all(binary_preds == labels, axis=1))
        hamming_loss = np.mean(binary_preds != labels)
        
        return {
            'map': map_score,
            'exact_match': exact_match,
            'hamming_loss': hamming_loss,
            'per_class_metrics': per_class_metrics
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], epoch: int):
        """Save model checkpoint"""
        if metrics['map'] >= self.best_val_map:
            self.best_val_map = metrics['map']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': metrics,
                'config': self.config
            }
            
            torch.save(checkpoint, self.save_dir / 'best_model.pt')
            print(f"Saved best model with MAP: {metrics['map']:.4f}")
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            # Training phase
            train_metrics = self.train_epoch()
            print(f"Train metrics: {train_metrics}")
            
            # Validation phase
            val_metrics = self.validate()
            print(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, epoch)
            
            # Log to wandb if enabled
            print("Logging to wandb at Epoch: ", epoch)

            # Enhanced Wandb logging
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    
                    # Training metrics
                    'train/loss': train_metrics['loss'],
                    'train/map': train_metrics['map'],
                    'train/exact_match': train_metrics['exact_match'],
                    'train/hamming_loss': train_metrics['hamming_loss'],
                    
                    # Validation metrics
                    'val/loss': val_metrics['loss'],
                    'val/map': val_metrics['map'],
                    'val/exact_match': val_metrics['exact_match'],
                    'val/hamming_loss': val_metrics['hamming_loss'],
                }
                
                # Add per-class metrics
                for class_idx, metrics in train_metrics['per_class_metrics'].items():
                    for metric_name, value in metrics.items():
                        log_dict[f'train/class_{class_idx}/{metric_name}'] = value
                
                for class_idx, metrics in val_metrics['per_class_metrics'].items():
                    for metric_name, value in metrics.items():
                        log_dict[f'val/class_{class_idx}/{metric_name}'] = value
                
                wandb.log(log_dict)
