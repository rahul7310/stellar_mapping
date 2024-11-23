import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List, Dict

class WarmupCosineLRScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler with:
    1. Linear warmup phase
    2. Cosine annealing with restarts
    3. Adaptive minimum learning rate based on loss plateaus
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        warmup_start_lr: float = 1e-7,
        cycles: int = 1,  # Number of cosine cycles
        plateau_patience: int = 3,
        plateau_factor: float = 0.5,
        verbose: bool = False
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_start_lr = warmup_start_lr
        self.cycles = cycles
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.verbose = verbose
        
        # Plateau detection
        self.plateau_count = 0
        self.best_loss = float('inf')
        
        # Initialize cycle length
        self.cycle_length = (total_epochs - warmup_epochs) // cycles
        
        super().__init__(optimizer, -1)
    
    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
        
        # During warmup phase
        if self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lr()
        
        # After warmup phase
        return self._get_cosine_lr()
    
    def _get_warmup_lr(self) -> List[float]:
        """Calculate LR during warmup phase"""
        alpha = self.last_epoch / self.warmup_epochs
        warmup_factor = alpha * (self.max_lr - self.warmup_start_lr) + self.warmup_start_lr
        return [warmup_factor for _ in self.base_lrs]
    
    def _get_cosine_lr(self) -> List[float]:
        """Calculate LR during cosine annealing phase"""
        epoch_in_cycle = (self.last_epoch - self.warmup_epochs) % self.cycle_length
        cycle_progress = epoch_in_cycle / self.cycle_length
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        
        return [lr for _ in self.base_lrs]
    
    def step(self, metrics=None):
        """Step with optional plateau detection"""
        # Update plateau detection if metrics provided
        if metrics is not None:
            if metrics < self.best_loss:
                self.best_loss = metrics
                self.plateau_count = 0
            else:
                self.plateau_count += 1
            
            # If plateau detected, reduce max_lr
            if self.plateau_count >= self.plateau_patience:
                self.max_lr = max(self.max_lr * self.plateau_factor, self.min_lr)
                self.plateau_count = 0
                if self.verbose:
                    print(f'Plateau detected! Reducing max_lr to {self.max_lr}')
        
        super().step()

class WarmupLinearScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler with:
    1. Linear warmup phase
    2. Linear decay
    3. Adaptive minimum learning rate
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        warmup_start_lr: float = 1e-7,
        plateau_patience: int = 3,
        plateau_factor: float = 0.5,
        verbose: bool = False
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_start_lr = warmup_start_lr
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.verbose = verbose
        
        # Plateau detection
        self.plateau_count = 0
        self.best_loss = float('inf')
        
        super().__init__(optimizer, -1)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lr()
        return self._get_decay_lr()
    
    def _get_warmup_lr(self) -> List[float]:
        """Calculate LR during warmup phase"""
        alpha = self.last_epoch / self.warmup_epochs
        warmup_factor = alpha * (self.max_lr - self.warmup_start_lr) + self.warmup_start_lr
        return [warmup_factor for _ in self.base_lrs]
    
    def _get_decay_lr(self) -> List[float]:
        """Calculate LR during linear decay phase"""
        progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        decay_factor = 1 - progress
        lr = self.min_lr + (self.max_lr - self.min_lr) * decay_factor
        return [lr for _ in self.base_lrs]

def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    config: Dict
) -> _LRScheduler:
    """
    Factory function to create schedulers
    
    Args:
        scheduler_type: Type of scheduler ('cosine', 'linear')
        optimizer: PyTorch optimizer
        config: Configuration dictionary
    
    Returns:
        Initialized scheduler
    """
    common_params = {
        'optimizer': optimizer,
        'warmup_epochs': config.get('warmup_epochs', 5),
        'total_epochs': config['num_epochs'],
        'min_lr': config.get('min_lr', 1e-6),
        'max_lr': config.get('max_lr', config['learning_rate']),
        'warmup_start_lr': config.get('warmup_start_lr', 1e-7),
        'plateau_patience': config.get('plateau_patience', 3),
        'plateau_factor': config.get('plateau_factor', 0.5),
        'verbose': config.get('verbose', False)
    }
    
    if scheduler_type == 'cosine':
        return WarmupCosineLRScheduler(
            **common_params,
            cycles=config.get('cosine_cycles', 1)
        )
    elif scheduler_type == 'linear':
        return WarmupLinearScheduler(**common_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")