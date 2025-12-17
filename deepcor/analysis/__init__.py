"""DeepCor analysis module."""

from .correlations import (
    correlate_columns,
    calc_corr_map,
    run_correlation_analysis_from_spec,
)
from .contrasts import (
    get_design_matrix,
    get_contrast_val,
    calc_contrast_map,
    run_contrast_analysis_from_spec,
)
from .metrics import (
    calc_mse,
    calc_and_save_compcor,
    average_signal_ensemble,
    correlation,
)

__all__ = [
    'correlate_columns',
    'calc_corr_map',
    'run_correlation_analysis_from_spec',
    'get_design_matrix',
    'get_contrast_val',
    'calc_contrast_map',
    'run_contrast_analysis_from_spec',
    'calc_mse',
    'calc_and_save_compcor',
    'average_signal_ensemble',
    'correlation',
]
