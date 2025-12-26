"""DeepCor utilities module."""

from .io import safe_mkdir
from .helpers import check_gpu_and_speedup

__all__ = [
    'safe_mkdir',
    'check_gpu_and_speedup',
]
