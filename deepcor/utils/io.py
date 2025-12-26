"""I/O utilities for DeepCor."""

import os


def safe_mkdir(path):
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    if not os.path.exists(path):
        os.mkdir(path)
