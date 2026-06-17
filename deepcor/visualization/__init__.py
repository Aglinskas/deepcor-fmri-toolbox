"""DeepCor visualization module."""

from .dashboard import (
    init_track,
    update_track,
    save_track,
    show_dahsboard_v1_marimo,
    show_dahsboard_v1_jupyter,
    show_dahsboard_v2_marimo,
    show_dahsboard_v2_jupyter,
    show_dahsboard_marimo,
    show_dahsboard_jupyter,
    format_progress_title,
)

__all__ = [
    'init_track',
    'update_track',
    'format_progress_title',
    'show_dahsboard_v1_marimo',
    'show_dahsboard_v1_jupyter',
    'show_dahsboard_v2_marimo',
    'show_dahsboard_v2_jupyter',
    'show_dahsboard_marimo',
    'show_dahsboard_jupyter',
    'save_track',
]
