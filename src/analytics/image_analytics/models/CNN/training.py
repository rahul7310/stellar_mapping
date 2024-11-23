import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import Dict, List
from data_preprocessing import DatasetConfig
import wandb
from tqdm import tqdm
import logging
from data_preprocessing import DataModule, ModelType, DatasetType
from CNN import create_model
from lr_schedular import create_scheduler

class MetricsCalculator:
    """Calculate and store various metrics for multi-label classification"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results"""
        self.predictions.append(preds.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self) -> Dict:
        """Compute all metrics"""
        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)
        
        # Calculate metrics
        metrics = {
            'loss': np.mean(self.losses),
            'per_class_metrics': self._calculate_per_class_metrics(predictions, targets),
            'overall_metrics': self._calculate_overall_metrics(predictions, targets),
            'confusion_matrices': self._calculate_confusion_matrices(predictions, targets)
        }
        
        return metrics
    
    def _calculate_per_class_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate per-class precision, recall, F1, and AP"""
        # Convert probabilities to binary predictions
        binary_preds = (predictions > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, binary_preds, zero_division=0
        )
        
        # Calculate Average Precision per class
        ap_scores = [
            average_precision_score(targets[:, i], predictions[:, i])
            for i in range(self.num_classes)
        ]
        
        # Calculate AUC-ROC per class
        auc_scores = [
            roc_auc_score(targets[:, i], predictions[:, i])
            for i in range(self.num_classes)
        ]
        
        return {
            'precision': {name: score for name, score in zip(self.class_names, precision)},
            'recall': {name: score for name, score in zip(self.class_names, recall)},
            'f1': {name: score for name, score in zip(self.class_names, f1)},
            'ap': {name: score for name, score in zip(self.class_names, ap_scores)},
            'auc': {name: score for name, score in zip(self.class_names, auc_scores)}
        }
    
    def _calculate_overall_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate overall metrics"""
        binary_preds = (predictions > 0.5).astype(int)
        
        # Sample-wise accuracy (exact match)
        exact_match = np.all(binary_preds == targets, axis=1).mean()
        
        # Mean Average Precision
        mean_ap = average_precision_score(targets, predictions, average='samples')
        
        # Hamming Loss
        hamming_loss = np.mean(binary_preds != targets)
        
        # Subset Accuracy
        subset_accuracy = np.mean(np.all(binary_preds == targets, axis=1))
        
        return {
            'exact_match': exact_match,
            'mean_ap': mean_ap,
            'hamming_loss': hamming_loss,
            'subset_accuracy': subset_accuracy
        }
    
    def _calculate_confusion_matrices(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """Calculate confusion matrix for each class"""
        binary_preds = (predictions > 0.5).astype(int)
        
        confusion_matrices = {}
        for i, class_name in enumerate(self.class_names):
            cm = confusion_matrix(targets[:, i], binary_preds[:, i])
            confusion_matrices[class_name] = cm.tolist()
        
        return confusion_matrices

class Trainer:
    """Main trainer class for constellation classification"""
    
    def __init__(
        self,
        model: nn.Module,
        data_module: DataModule,
        config: Dict,
        device: torch.device,
        experiment_name: str,
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.data_module.class_weights.to(device)
        )
        
        # Setup learning rate scheduler
        self.scheduler = create_scheduler(
            scheduler_type=config.get('scheduler_type', 'cosine'),
            optimizer=self.optimizer,
            config=config
        )
        
        # Setup metrics calculator
        self.metrics_calculator = MetricsCalculator(
            num_classes=len(self.data_module.datasets[DatasetType.TRAIN].class_columns),
            class_names=self.data_module.datasets[DatasetType.TRAIN].class_columns
        )
        
        # Setup logging
        self.setup_logging(experiment_name)
        
        # Initialize best metrics
        self.best_metrics = {'val_mean_ap': 0.0}
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on config"""
        if self.config['optimizer'] == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def setup_logging(self, experiment_name: str):
        """Setup logging and wandb"""
        self.log_dir = Path('logs') / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        logging.basicConfig(
            filename=self.log_dir / 'training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        if self.use_wandb:
            wandb.init(
                project="constellation-classification",
                name=experiment_name,
                config=self.config
            )
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.metrics_calculator.reset()
        
        train_loader = self.data_module.get_dataloader(DatasetType.TRAIN)
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                predictions = torch.sigmoid(outputs)
                self.metrics_calculator.update(predictions, targets, loss.item())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        metrics = self.metrics_calculator.compute_metrics()
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate the model"""
        self.model.eval()
        self.metrics_calculator.reset()
        
        val_loader = self.data_module.get_dataloader(DatasetType.VALID)
        
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            predictions = torch.sigmoid(outputs)
            self.metrics_calculator.update(predictions, targets, loss.item())
        
        metrics = self.metrics_calculator.compute_metrics()
        return metrics
    
    def train(self, num_epochs: int):
        """Main training loop with enhanced debugging"""
        print("\nStarting training...")
        print(f"Log directory: {self.log_dir}")
        print(f"Number of epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            print("\nTraining metrics:")
            print(f"Loss: {train_metrics['loss']:.4f}")
            print(f"Mean AP: {train_metrics['overall_metrics']['mean_ap']:.4f}")
            
            # Validation phase
            val_metrics = self.validate()
            print("\nValidation metrics:")
            print(f"Loss: {val_metrics['loss']:.4f}")
            print(f"Mean AP: {val_metrics['overall_metrics']['mean_ap']:.4f}")
            print(f"Previous best MAP: {self.best_metrics['val_mean_ap']:.4f}")
            
            # Update learning rate scheduler
            self.scheduler.step(metrics=val_metrics['loss'])
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Check if we should save
            current_map = val_metrics['overall_metrics']['mean_ap']
            if current_map > self.best_metrics['val_mean_ap']:
                print(f"\nImprovement detected! Previous best: {self.best_metrics['val_mean_ap']:.4f}, Current: {current_map:.4f}")
                self.best_metrics['val_mean_ap'] = current_map
                self.save_checkpoint(epoch, val_metrics)
            else:
                print(f"\nNo improvement. Previous best: {self.best_metrics['val_mean_ap']:.4f}, Current: {current_map:.4f}")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to wandb and file"""
        # Log to file
        logging.info(f"Epoch {epoch}")
        logging.info(f"Train metrics: {train_metrics['overall_metrics']}")
        logging.info(f"Val metrics: {val_metrics['overall_metrics']}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'train_mean_ap': train_metrics['overall_metrics']['mean_ap'],
                'val_mean_ap': val_metrics['overall_metrics']['mean_ap'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save checkpoint with verbose debugging"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Get current mean AP and print for verification
        current_map = metrics['overall_metrics']['mean_ap']
        print(f"\nAttempting to save checkpoint:")
        print(f"Current MAP: {current_map}")
        print(f"Best MAP so far: {self.best_metrics['val_mean_ap']}")
        
        checkpoint_path = self.log_dir / 'best_model.pt'
        print(f"Saving checkpoint to: {checkpoint_path}")
            
        try:
            # Verify directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            print(f"Successfully saved checkpoint!")
            logging.info(f"Saved best model with validation MAP: {current_map:.4f}")
            
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            logging.error(f"Failed to save checkpoint: {str(e)}")

def main():
    """Example usage"""
    # Configuration
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 10,

        # Scheduler configuration
        'scheduler_type': 'cosine',  # or 'linear'
        'warmup_epochs': 5,
        'min_lr': 1e-6,
        'max_lr': 1e-3,
        'warmup_start_lr': 1e-7,
        'cosine_cycles': 3,  # Number of cosine annealing cycles
        'plateau_patience': 3,
        'plateau_factor': 0.5,
        'verbose': True
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = Path(r"data\constellation_dataset_1")
    data_module = DataModule(
        data_config=DatasetConfig(base_path),
        model_type=ModelType.CNN,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Create model
    model = create_model(
        model_type='cnn',  # or 'vit'
        num_classes=16,
        pretrained=True,
        backbone='resnet50',  # for CNN
        dropout_rate=0.5
    )

    # Move to device
    model = model.to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        data_module=data_module,
        config=config,
        device=device,
        experiment_name='constellation_training'
    )

    # Train
    trainer.train(num_epochs=10)