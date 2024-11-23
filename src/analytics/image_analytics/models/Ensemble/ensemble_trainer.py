import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import wandb
import logging
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from data_preprocessing import DataModule, DatasetType
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

class MetricsTracker:
    """Track and compute various training metrics"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics for new epoch"""
        self.predictions = []
        self.targets = []
        self.ensemble_losses = []
        self.cnn_losses = []
        self.vit_losses = []
        self.fusion_losses = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        losses: Dict[str, float]
    ):
        """Update metrics with batch results"""
        self.predictions.append(preds.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        self.ensemble_losses.append(losses['ensemble'])
        self.cnn_losses.append(losses['cnn'])
        self.vit_losses.append(losses['vit'])
        self.fusion_losses.append(losses['fusion'])
    
    def compute_metrics(self) -> Dict:
        """Compute all metrics"""
        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)
        
        # Calculate metrics
        metrics = {
            'loss': {
                'ensemble': np.mean(self.ensemble_losses),
                'cnn': np.mean(self.cnn_losses),
                'vit': np.mean(self.vit_losses),
                'fusion': np.mean(self.fusion_losses),
                'total': np.mean(self.ensemble_losses) + np.mean(self.fusion_losses)
            },
            'per_class_metrics': self._calculate_per_class_metrics(predictions, targets),
            'overall_metrics': self._calculate_overall_metrics(predictions, targets)
        }
        
        return metrics
    
    def _calculate_per_class_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate per-class precision, recall, F1, and AP"""
        # Convert probabilities to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, binary_preds, zero_division=0, average=None
        )
        
        # Calculate Average Precision per class
        ap_scores = [
            average_precision_score(targets[:, i], predictions[:, i])
            for i in range(self.num_classes)
        ]
        
        return {
            'precision': {name: score for name, score in zip(self.class_names, precision)},
            'recall': {name: score for name, score in zip(self.class_names, recall)},
            'f1': {name: score for name, score in zip(self.class_names, f1)},
            'ap': {name: score for name, score in zip(self.class_names, ap_scores)}
        }
    
    def _calculate_overall_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate overall metrics"""
        binary_preds = (predictions > 0.5).astype(int)
        
        # Exact match accuracy
        exact_match = np.mean(np.all(binary_preds == targets, axis=1))
        
        # Mean Average Precision
        mean_ap = average_precision_score(targets, predictions, average='samples')
        
        # Hamming Loss
        hamming_loss = np.mean(binary_preds != targets)
        
        return {
            'exact_match': exact_match,
            'mean_ap': mean_ap,
            'hamming_loss': hamming_loss
        }

class EnsembleTrainer:
    """Trainer class for ensemble model"""
    
    def __init__(
        self,
        model: nn.Module,
        data_module: DataModule,
        config: Dict,
        device: torch.device,
        save_dir: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device
        self.experiment_name = experiment_name or f"ensemble_{datetime.now():%Y%m%d_%H%M%S}"
        self.save_dir = Path(save_dir or "experiments") / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        
        # Setup components
        self._setup_logging()
        self._setup_optimizers()
        self._setup_criterion()
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            num_classes=int(self.config['model']['num_classes']),
            class_names=self.data_module.datasets[DatasetType.TRAIN].class_columns
        )
        
        # Initialize best metrics
        self.best_metrics = {'val_mean_ap': 0.0}
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        # Initialize wandb if enabled
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.experiment_name,
                config=self.config
            )
    
    def _setup_optimizers(self):
        """Setup optimizers for different components"""
        # Optimizer for CNN
        self.cnn_optimizer = optim.AdamW(
            self.model.cnn.parameters(),
            lr=float(self.config['training']['cnn_lr']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # Optimizer for ViT
        self.vit_optimizer = optim.AdamW(
            self.model.vit.parameters(),
            lr=float(self.config['training']['vit_lr']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # Optimizer for ensemble weights and fusion layer
        ensemble_params = list(self.model.fusion_layer.parameters()) + [self.model.ensemble_weights]
        self.ensemble_optimizer = optim.AdamW(
            ensemble_params,
            lr=float(self.config['training']['ensemble_lr']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
    
    def _setup_criterion(self):
        """Setup loss functions"""
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.data_module.class_weights.to(self.device)
        )

    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.metrics_tracker.reset()
        
        train_loader = self.data_module.get_dataloader(DatasetType.TRAIN)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            cnn_inputs = batch['cnn_input'].to(self.device)
            vit_inputs = batch['vit_input'].to(self.device)
            targets = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(cnn_input=cnn_inputs, vit_input=vit_inputs)
            
            # Calculate losses
            ensemble_loss = self.criterion(outputs['ensemble_logits'], targets)
            cnn_loss = self.criterion(outputs['cnn_logits'], targets)
            vit_loss = self.criterion(outputs['vit_logits'], targets)
            fusion_loss = self.criterion(outputs['fusion_logits'], targets)
            
            # Weighted total loss
            total_loss = (
                self.config['training']['loss_weights']['ensemble'] * ensemble_loss +
                self.config['training']['loss_weights']['cnn'] * cnn_loss +
                self.config['training']['loss_weights']['vit'] * vit_loss +
                self.config['training']['loss_weights']['fusion'] * fusion_loss
            )
            
            # Backward pass and optimization
            self.cnn_optimizer.zero_grad()
            self.vit_optimizer.zero_grad()
            self.ensemble_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                        self.config['training']['gradient_clip'])
            
            self.cnn_optimizer.step()
            self.vit_optimizer.step()
            self.ensemble_optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                self.metrics_tracker.update(
                    preds=torch.sigmoid(outputs['ensemble_logits']),
                    targets=targets,
                    losses={
                        'ensemble': ensemble_loss.item(),
                        'cnn': cnn_loss.item(),
                        'vit': vit_loss.item(),
                        'fusion': fusion_loss.item()
                    }
                )
            
            # Update global step
            self.global_step += 1

        self.metrics = self.metrics_tracker.compute_metrics()
            
        return self.metrics

    def get_metrics(self):
        # Return a dictionary with the current metrics (e.g., loss, accuracy)
        return self.metrics

    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        self.metrics_tracker.reset()
        
        val_loader = self.data_module.get_dataloader(DatasetType.VALID)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                cnn_inputs = batch['cnn_input'].to(self.device)
                vit_inputs = batch['vit_input'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(cnn_input=cnn_inputs, vit_input=vit_inputs)
                
                # Calculate losses
                ensemble_loss = self.criterion(outputs['ensemble_logits'], targets)
                cnn_loss = self.criterion(outputs['cnn_logits'], targets)
                vit_loss = self.criterion(outputs['vit_logits'], targets)
                fusion_loss = self.criterion(outputs['fusion_logits'], targets)
                
                # Update metrics
                self.metrics_tracker.update(
                    preds=torch.sigmoid(outputs['ensemble_logits']),
                    targets=targets,
                    losses={
                        'ensemble': ensemble_loss.item(),
                        'cnn': cnn_loss.item(),
                        'vit': vit_loss.item(),
                        'fusion': fusion_loss.item()
                    }
                )
        
        return self.metrics_tracker.compute_metrics()

    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to wandb"""
        log_dict = {
            'epoch': epoch,
            'learning_rate/cnn': self.cnn_optimizer.param_groups[0]['lr'],
            'learning_rate/vit': self.vit_optimizer.param_groups[0]['lr'],
            'learning_rate/ensemble': self.ensemble_optimizer.param_groups[0]['lr'],
            
            # Training metrics
            'train/loss/total': train_metrics['loss']['total'],
            'train/loss/ensemble': train_metrics['loss']['ensemble'],
            'train/loss/cnn': train_metrics['loss']['cnn'],
            'train/loss/vit': train_metrics['loss']['vit'],
            'train/loss/fusion': train_metrics['loss']['fusion'],
            'train/mean_ap': train_metrics['overall_metrics']['mean_ap'],
            'train/exact_match': train_metrics['overall_metrics']['exact_match'],
            'train/hamming_loss': train_metrics['overall_metrics']['hamming_loss'],
            
            # Validation metrics
            'val/loss/total': val_metrics['loss']['total'],
            'val/loss/ensemble': val_metrics['loss']['ensemble'],
            'val/loss/cnn': val_metrics['loss']['cnn'],
            'val/loss/vit': val_metrics['loss']['vit'],
            'val/loss/fusion': val_metrics['loss']['fusion'],
            'val/mean_ap': val_metrics['overall_metrics']['mean_ap'],
            'val/exact_match': val_metrics['overall_metrics']['exact_match'],
            'val/hamming_loss': val_metrics['overall_metrics']['hamming_loss']
        }
        
        # Log per-class metrics
        for class_name in self.data_module.datasets[DatasetType.TRAIN].class_columns:
            # Training
            log_dict.update({
                f'train/class/{class_name}/precision': train_metrics['per_class_metrics']['precision'][class_name],
                f'train/class/{class_name}/recall': train_metrics['per_class_metrics']['recall'][class_name],
                f'train/class/{class_name}/f1': train_metrics['per_class_metrics']['f1'][class_name],
                f'train/class/{class_name}/ap': train_metrics['per_class_metrics']['ap'][class_name]
            })
            
            # Validation
            log_dict.update({
                f'val/class/{class_name}/precision': val_metrics['per_class_metrics']['precision'][class_name],
                f'val/class/{class_name}/recall': val_metrics['per_class_metrics']['recall'][class_name],
                f'val/class/{class_name}/f1': val_metrics['per_class_metrics']['f1'][class_name],
                f'val/class/{class_name}/ap': val_metrics['per_class_metrics']['ap'][class_name]
            })
        
        # Log ensemble weights
        weights = self.model.ensemble_weights.softmax(dim=0)
        log_dict.update({
            'model/cnn_weight': weights[0].item(),
            'model/vit_weight': weights[1].item()
        })
        
        wandb.log(log_dict, step=self.global_step)

    def save_checkpoint(self, metrics: Dict, epoch: int):
        """Save model checkpoint"""
        if metrics['overall_metrics']['mean_ap'] > self.best_metrics['val_mean_ap']:
            self.best_metrics['val_mean_ap'] = metrics['overall_metrics']['mean_ap']
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'cnn_optimizer_state_dict': self.cnn_optimizer.state_dict(),
                'vit_optimizer_state_dict': self.vit_optimizer.state_dict(),
                'ensemble_optimizer_state_dict': self.ensemble_optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }
            
            # Save checkpoint
            checkpoint_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"\nSaved best model with validation MAP: {metrics['overall_metrics']['mean_ap']:.4f}")
            
            # Save latest checkpoint separately if configured
            if self.config['logging']['checkpoint'].get('save_last', False):
                latest_path = self.save_dir / 'latest_model.pt'
                torch.save(checkpoint, latest_path)
            
            # Save periodic checkpoint if configured
            save_frequency = self.config['logging']['checkpoint'].get('save_frequency', 0)
            if save_frequency > 0 and (epoch + 1) % save_frequency == 0:
                periodic_path = self.save_dir / f'model_epoch_{epoch+1}.pt'
                torch.save(checkpoint, periodic_path)

    def train(self, num_epochs: int):
        """Main training loop"""
        logging.info(f"Starting training for {num_epochs} epochs")
        logging.info(f"Training device: {self.device}")
        logging.info(f"Save directory: {self.save_dir}")
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
                
                # Training phase
                train_metrics = self.train_epoch()
                logging.info("\nTraining metrics:")
                logging.info(f"Loss: {train_metrics['loss']['total']:.4f}")
                logging.info(f"Mean AP: {train_metrics['overall_metrics']['mean_ap']:.4f}")
                
                # Validation phase
                val_metrics = self.validate()
                logging.info("\nValidation metrics:")
                logging.info(f"Loss: {val_metrics['loss']['total']:.4f}")
                logging.info(f"Mean AP: {val_metrics['overall_metrics']['mean_ap']:.4f}")
                
                # Log metrics
                if self.config['logging']['use_wandb']:
                    self._log_epoch_metrics(train_metrics, val_metrics, epoch)
                
                # Save checkpoint if improved
                self.save_checkpoint(val_metrics, epoch)
                
                # Print ensemble weights
                weights = self.model.ensemble_weights.softmax(dim=0)
                logging.info("\nEnsemble weights:")
                logging.info(f"CNN weight: {weights[0]:.4f}")
                logging.info(f"ViT weight: {weights[1]:.4f}")
                
                # Early stopping check
                if self._check_early_stopping(val_metrics):
                    logging.info("\nEarly stopping triggered!")
                    break
        
        except Exception as e:
            logging.error(f"Training failed: {str(e)}", exc_info=True)
            raise
        
        finally:
            if self.config['logging']['use_wandb']:
                wandb.finish()
    
    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check if early stopping criteria are met"""
        if not hasattr(self, 'patience_counter'):
            self.patience_counter = 0
            self.best_val_loss = float('inf')
        
        current_loss = val_metrics['loss']['total']
        
        if current_loss < self.best_val_loss:
            self.best_val_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= float(self.config['training']['patience'])
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_optimizer.load_state_dict(checkpoint['cnn_optimizer_state_dict'])
        self.vit_optimizer.load_state_dict(checkpoint['vit_optimizer_state_dict'])
        self.ensemble_optimizer.load_state_dict(checkpoint['ensemble_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metrics = {'val_mean_ap': checkpoint['metrics']['overall_metrics']['mean_ap']}
        
        logging.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        logging.info(f"Best validation MAP: {self.best_metrics['val_mean_ap']:.4f}")

def main():
    """Example usage of EnsembleTrainer"""
    import yaml
    from pathlib import Path
    
    # Load configuration
    config_path = "config/ensemble_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data module
    from data_preprocessing import DatasetConfig, DataModule, ModelType
    data_config = DatasetConfig(config['data_path'])
    data_module = DataModule(
        data_config=data_config,
        model_type=ModelType.ENSEMBLE,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    from ensemble_model import create_ensemble_model
    model = create_ensemble_model(config)
    
    # Create trainer
    trainer = EnsembleTrainer(
        model=model,
        data_module=data_module,
        config=config,
        device=device,
        save_dir=config['save_dir'],
        experiment_name=config['experiment_name']
    )
    
    # Train model
    trainer.train(num_epochs=config['training']['num_epochs'])

if __name__ == "__main__":
    main()