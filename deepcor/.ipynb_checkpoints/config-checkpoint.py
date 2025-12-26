"""Configuration management for DeepCor."""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    latent_dims: Tuple[int, int] = (8, 8)
    hidden_dims: Optional[List[int]] = None
    beta: float = 0.01
    gamma: float = 0.0
    delta: float = 0.0
    scale_MSE_GM: float = 1e3
    scale_MSE_CF: float = 1e3
    scale_MSE_FG: float = 0.0
    do_disentangle: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""

    n_epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 0.001
    optimizer: str = 'adamw'
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    max_grad_norm: float = 5.0
    n_repetitions: int = 20


@dataclass
class DataConfig:
    """Configuration for data processing."""

    n_dummy_scans: int = 0
    apply_censoring: bool = False
    censoring_threshold: float = 0.5
    also_nearby_voxels: bool = True
    confound_columns: List[str] = field(
        default_factory=lambda: ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']
    )


@dataclass
class DeepCorConfig:
    """Complete DeepCor configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


def get_default_config():
    """
    Get default configuration.

    Returns:
        DeepCorConfig with default settings
    """
    return DeepCorConfig()
