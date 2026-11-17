from pathlib import Path
from typing import Optional, Callable, Tuple
import logging
import sys

import torch
import torch.nn as nn
import numpy as np

def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 distances between vectors in x and y.
    
    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)
    
    Returns:
        Tensor of shape (N, M) containing pairwise distances
    """
    # Modern implementation using broadcasting
    x_expanded = x.unsqueeze(2)  # (N, D, 1)
    y_expanded = y.t().unsqueeze(0)  # (1, D, M)
    return torch.sum((x_expanded - y_expanded) ** 2, dim=1)


def make_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert target labels to one-hot encoding.
    
    Args:
        target: Tensor of shape (N,) containing class indices
        num_classes: Number of classes
    
    Returns:
        One-hot encoded tensor of shape (N, num_classes)
    """
    return torch.nn.functional.one_hot(target, num_classes=num_classes).float()

def find_high_activation_crop(
    activation_map: np.ndarray,
    percentile: int = 95
) -> Tuple[int, int, int, int]:
    """
    Find bounding box of high activation region in activation map.
    
    Args:
        activation_map: 2D numpy array of activation values
        percentile: Percentile threshold for high activation
    
    Returns:
        Tuple of (y_min, y_max, x_min, x_max)
    """
    threshold = np.percentile(activation_map, percentile)
    mask = activation_map >= threshold
    
    # Find bounds using numpy operations (more efficient)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        # Fallback if no activation above threshold
        return 0, activation_map.shape[0], 0, activation_map.shape[1]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return y_min, y_max + 1, x_min, x_max + 1

class ImageNetNormalize:
    """ImageNet normalization transform."""
    
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    
    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize image tensor with ImageNet statistics.
        
        Args:
            x: Tensor of shape (B, 3, H, W) or (3, H, W)
        
        Returns:
            Normalized tensor
        """
        mean = torch.tensor(ImageNetNormalize.MEAN, device=x.device)
        std = torch.tensor(ImageNetNormalize.STD, device=x.device)
        
        if x.ndim == 4:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        
        return (x - mean) / std
    
    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Reverse ImageNet normalization."""
        mean = torch.tensor(ImageNetNormalize.MEAN, device=x.device)
        std = torch.tensor(ImageNetNormalize.STD, device=x.device)
        
        if x.ndim == 4:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        
        return x * std + mean

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class ModelCheckpoint:
    """Modern checkpoint manager for saving models."""
    
    def __init__(
        self,
        save_dir: Path,
        min_accuracy: float = 0.0,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            min_accuracy: Minimum accuracy to save model
            save_best_only: If True, only keep the best checkpoint
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.min_accuracy = min_accuracy
        self.save_best_only = save_best_only
        self.best_accuracy = 0.0
        self.best_path = None
    
    def save(
        self,
        model: nn.Module,
        epoch: int,
        accuracy: float,
        suffix: str = '',
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[logging.Logger] = None
    ) -> Optional[Path]:
        """
        Save model checkpoint if conditions are met.
        
        Args:
            model: Model to save
            epoch: Current epoch
            accuracy: Current accuracy
            suffix: Optional suffix for filename
            optimizer: Optional optimizer state to save
            logger: Optional logger for messages
        
        Returns:
            Path to saved checkpoint or None if not saved
        """
        if accuracy < self.min_accuracy:
            return None
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'accuracy': accuracy,
            'model_state_dict': model.state_dict(),
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Generate filename
        filename = f'epoch_{epoch}'
        if suffix:
            filename += f'_{suffix}'
        filename += f'_acc_{accuracy:.4f}.pth'
        save_path = self.save_dir / filename
        
        # Save
        if self.save_best_only:
            if accuracy > self.best_accuracy:
                # Remove old best if exists
                if self.best_path and self.best_path.exists():
                    self.best_path.unlink()
                
                torch.save(checkpoint, save_path)
                self.best_accuracy = accuracy
                self.best_path = save_path
                
                if logger:
                    logger.info(f'Saved new best model: {save_path.name} '
                              f'(accuracy: {accuracy:.2%})')
                return save_path
        else:
            torch.save(checkpoint, save_path)
            if logger:
                logger.info(f'Saved checkpoint: {save_path.name}')
            return save_path
        
        return None
    
    @staticmethod
    def load(
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Tuple[int, float]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to load tensors to
        
        Returns:
            Tuple of (epoch, accuracy)
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['accuracy']

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
