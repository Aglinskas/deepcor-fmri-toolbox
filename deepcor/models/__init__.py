"""DeepCor models module."""

from .base import BaseModel
from .cvae import CVAE, cVAE
from .cvae_v1 import CVAE_V1, cVAE_V1
from .registry import get_model, list_models, MODEL_REGISTRY

__all__ = [
    "BaseModel",
    "CVAE",      # Current recommended CVAE implementation (v2)
    "cVAE",      # Alias for CVAE (v2)
    "CVAE_V1",   # Original CVAE implementation
    "cVAE_V1",   # Alias for original CVAE implementation
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
]

