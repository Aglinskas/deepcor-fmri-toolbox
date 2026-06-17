"""I/O utilities for DeepCor."""

import os


def safe_mkdir(path):
    """
    Create directory (and any missing parents) if it doesn't exist.

    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)
