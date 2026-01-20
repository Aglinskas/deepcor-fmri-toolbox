"""
fMRIPrep BIDS Parser - A streamlined interface for working with fMRIPrep derivatives

This module provides an intuitive, Pythonic interface for navigating fMRIPrep
output directories and quickly accessing files needed for downstream fMRI analyses.

Author: Master (Boston College)
Designed for: Computational Neuroscience workflows including denoising, GLM, connectivity

Key Design Principles:
1. Zero-boilerplate file access - get paths in one line
2. Method chaining for intuitive navigation
3. Built-in support for common analysis patterns (denoising, connectivity, etc.)
4. Lazy loading for fast initialization on large datasets
5. Rich metadata extraction from BIDS sidecars
"""

import os
import re
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Union, Any, Iterator, Tuple
from dataclasses import dataclass, field
from functools import cached_property
import pandas as pd
import numpy as np


# =============================================================================
# Data Classes for Structured Access
# =============================================================================

@dataclass
class ConfoundsBundle:
    """
    Structured access to confounds data with common denoising strategies built-in.
    
    Provides easy extraction of confound regressors for various denoising strategies
    without having to remember column names or manually construct regressor matrices.
    """
    tsv_path: str
    json_path: Optional[str] = None
    _df: pd.DataFrame = field(default=None, repr=False)
    _metadata: Dict = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Load confounds data lazily on first access."""
        pass
    
    @cached_property
    def df(self) -> pd.DataFrame:
        """Load and cache the confounds dataframe."""
        return pd.read_csv(self.tsv_path, sep='\t')
    
    @cached_property
    def metadata(self) -> Dict:
        """Load and cache the confounds metadata JSON."""
        if self.json_path and os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                return json.load(f)
        return {}
    
    @property
    def n_timepoints(self) -> int:
        """Number of timepoints/volumes."""
        return len(self.df)
    
    @property
    def available_columns(self) -> List[str]:
        """List all available confound columns."""
        return list(self.df.columns)
    
    # -------------------------------------------------------------------------
    # Motion Parameters
    # -------------------------------------------------------------------------
    
    @property
    def motion_params(self) -> pd.DataFrame:
        """6 rigid-body motion parameters (translations and rotations)."""
        cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        return self.df[[c for c in cols if c in self.df.columns]]
    
    @property
    def motion_params_24(self) -> pd.DataFrame:
        """
        24-parameter motion model: 6 params + derivatives + squared + squared derivatives.
        Commonly used for thorough motion denoising (Friston 24-parameter model).
        """
        base_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        all_cols = []
        for col in base_cols:
            if col in self.df.columns:
                all_cols.append(col)
            # Derivatives
            if f'{col}_derivative1' in self.df.columns:
                all_cols.append(f'{col}_derivative1')
            # Power (squared)
            if f'{col}_power2' in self.df.columns:
                all_cols.append(f'{col}_power2')
            # Derivative squared
            if f'{col}_derivative1_power2' in self.df.columns:
                all_cols.append(f'{col}_derivative1_power2')
        return self.df[all_cols] if all_cols else pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # CompCor Components
    # -------------------------------------------------------------------------
    
    @property
    def acompcor(self) -> pd.DataFrame:
        """Anatomical CompCor components (combined WM+CSF mask)."""
        cols = [c for c in self.df.columns if c.startswith('a_comp_cor_')]
        return self.df[cols]
    
    @property
    def tcompcor(self) -> pd.DataFrame:
        """Temporal CompCor components."""
        cols = [c for c in self.df.columns if c.startswith('t_comp_cor_')]
        return self.df[cols]
    
    def get_acompcor(self, n_components: int = 5, mask: str = 'combined') -> pd.DataFrame:
        """
        Get top N anatomical CompCor components.
        
        Parameters
        ----------
        n_components : int
            Number of components to return (default: 5)
        mask : str
            Mask type: 'combined', 'WM', 'CSF' (requires metadata)
        
        Returns
        -------
        pd.DataFrame
            Selected aCompCor components
        """
        if mask != 'combined' and self.metadata:
            # Filter by mask type using metadata
            valid_cols = []
            for col, info in self.metadata.items():
                if (col.startswith('a_comp_cor_') and 
                    info.get('Mask') == mask and
                    info.get('Retained', True)):
                    valid_cols.append(col)
            valid_cols = sorted(valid_cols)[:n_components]
            return self.df[valid_cols] if valid_cols else pd.DataFrame()
        else:
            cols = sorted([c for c in self.df.columns if c.startswith('a_comp_cor_')])
            return self.df[cols[:n_components]]
    
    # -------------------------------------------------------------------------
    # Global Signals
    # -------------------------------------------------------------------------
    
    @property
    def global_signals(self) -> pd.DataFrame:
        """Global signal, WM, and CSF mean signals."""
        cols = ['global_signal', 'white_matter', 'csf']
        return self.df[[c for c in cols if c in self.df.columns]]
    
    @property 
    def csf(self) -> pd.Series:
        """CSF mean signal."""
        return self.df['csf'] if 'csf' in self.df.columns else None
    
    @property
    def white_matter(self) -> pd.Series:
        """White matter mean signal."""
        return self.df['white_matter'] if 'white_matter' in self.df.columns else None
    
    @property
    def global_signal(self) -> pd.Series:
        """Global (whole-brain) mean signal."""
        return self.df['global_signal'] if 'global_signal' in self.df.columns else None
    
    # -------------------------------------------------------------------------
    # Quality Metrics
    # -------------------------------------------------------------------------
    
    @property
    def framewise_displacement(self) -> pd.Series:
        """Framewise displacement (FD) time series."""
        return self.df['framewise_displacement'] if 'framewise_displacement' in self.df.columns else None
    
    @property
    def dvars(self) -> pd.Series:
        """DVARS (standardized)."""
        return self.df['std_dvars'] if 'std_dvars' in self.df.columns else None
    
    @property
    def mean_fd(self) -> float:
        """Mean framewise displacement."""
        fd = self.framewise_displacement
        return fd.mean() if fd is not None else np.nan
    
    @property
    def max_fd(self) -> float:
        """Maximum framewise displacement."""
        fd = self.framewise_displacement
        return fd.max() if fd is not None else np.nan
    
    def get_high_motion_volumes(self, fd_threshold: float = 0.5) -> np.ndarray:
        """
        Get indices of volumes exceeding FD threshold (for scrubbing).
        
        Parameters
        ----------
        fd_threshold : float
            FD threshold in mm (default: 0.5)
        
        Returns
        -------
        np.ndarray
            Boolean mask where True = high motion volume
        """
        fd = self.framewise_displacement
        if fd is None:
            return np.zeros(self.n_timepoints, dtype=bool)
        return (fd > fd_threshold).values
    
    @property
    def non_steady_state_outliers(self) -> np.ndarray:
        """Indices of non-steady-state volumes."""
        cols = [c for c in self.df.columns if c.startswith('non_steady_state_outlier')]
        if not cols:
            return np.array([])
        # Find volumes marked as outliers
        outlier_df = self.df[cols]
        return np.where(outlier_df.any(axis=1))[0]
    
    # -------------------------------------------------------------------------
    # Cosine Basis (for high-pass filtering)
    # -------------------------------------------------------------------------
    
    @property
    def cosine_basis(self) -> pd.DataFrame:
        """Cosine basis set for DCT-based high-pass filtering."""
        cols = sorted([c for c in self.df.columns if c.startswith('cosine')])
        return self.df[cols] if cols else pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # Pre-built Denoising Strategies
    # -------------------------------------------------------------------------
    
    def get_strategy(self, strategy: str = '24P+aCompCor+GSR', 
                     n_compcor: int = 5) -> pd.DataFrame:
        """
        Get confounds matrix for common denoising strategies.
        
        Parameters
        ----------
        strategy : str
            Denoising strategy name:
            - '6P': 6 motion parameters only
            - '24P': 24-parameter motion model
            - '6P+aCompCor': 6P + anatomical CompCor
            - '24P+aCompCor': 24P + anatomical CompCor
            - '24P+aCompCor+GSR': 24P + aCompCor + global signal regression
            - 'aCompCor': aCompCor only
            - 'tCompCor': temporal CompCor only
            - 'GSR': Global signal regression only
        n_compcor : int
            Number of CompCor components (default: 5)
        
        Returns
        -------
        pd.DataFrame
            Confounds matrix ready for use in denoising
        """
        parts = []
        
        if '24P' in strategy or '24p' in strategy:
            parts.append(self.motion_params_24)
        elif '6P' in strategy or '6p' in strategy:
            parts.append(self.motion_params)
        
        if 'aCompCor' in strategy or 'acompcor' in strategy.lower():
            parts.append(self.get_acompcor(n_components=n_compcor))
        
        if 'tCompCor' in strategy or 'tcompcor' in strategy.lower():
            cols = sorted([c for c in self.df.columns if c.startswith('t_comp_cor_')])[:n_compcor]
            if cols:
                parts.append(self.df[cols])
        
        if 'GSR' in strategy or 'gsr' in strategy.lower():
            if 'global_signal' in self.df.columns:
                parts.append(self.df[['global_signal']])
        
        if not parts:
            return pd.DataFrame()
        
        return pd.concat(parts, axis=1).fillna(0)
    
    def to_matrix(self, columns: List[str] = None, fillna: float = 0) -> np.ndarray:
        """
        Convert confounds to numpy matrix for direct use in analyses.
        
        Parameters
        ----------
        columns : List[str], optional
            Specific columns to include. If None, uses all columns.
        fillna : float
            Value to fill NaN entries (default: 0)
        
        Returns
        -------
        np.ndarray
            Confounds matrix (n_timepoints x n_confounds)
        """
        if columns:
            df = self.df[columns]
        else:
            df = self.df
        return df.fillna(fillna).values


@dataclass
class RunFiles:
    """
    All files associated with a single BOLD run.
    
    Provides structured access to preprocessed BOLD, confounds, masks, and
    associated metadata for a single functional run.
    """
    subject: str
    session: Optional[str]
    task: str
    run: Optional[str]
    root: str
    space: str = None
    
    @property
    def _base_pattern(self) -> str:
        """Build the base filename pattern for this run."""
        parts = [f'sub-{self.subject}']
        if self.session:
            parts.append(f'ses-{self.session}')
        parts.append(f'task-{self.task}')
        if self.run:
            parts.append(f'run-{self.run}')
        return '_'.join(parts)
    
    @property
    def _func_dir(self) -> str:
        """Path to functional directory."""
        parts = [self.root, f'sub-{self.subject}']
        if self.session:
            parts.append(f'ses-{self.session}')
        parts.append('func')
        return os.path.join(*parts)
    
    @property
    def _anat_dir(self) -> str:
        """Path to anatomical directory."""
        parts = [self.root, f'sub-{self.subject}']
        if self.session:
            parts.append(f'ses-{self.session}')
        parts.append('anat')
        return os.path.join(*parts)
    
    def _find_file(self, directory: str, pattern: str) -> Optional[str]:
        """Find a file matching the pattern in directory."""
        if not os.path.exists(directory):
            return None
        for f in os.listdir(directory):
            if re.match(pattern, f):
                return os.path.join(directory, f)
        return None
    
    def _find_files(self, directory: str, pattern: str) -> List[str]:
        """Find all files matching the pattern in directory."""
        if not os.path.exists(directory):
            return []
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if re.match(pattern, f)]
    
    # -------------------------------------------------------------------------
    # BOLD Data
    # -------------------------------------------------------------------------
    
    @property
    def bold_preproc(self) -> Optional[str]:
        """Preprocessed BOLD file in specified space (or native if no space specified)."""
        if self.space:
            pattern = f'{self._base_pattern}_space-{self.space}_desc-preproc_bold\\.nii(\\.gz)?$'
        else:
            pattern = f'{self._base_pattern}_desc-preproc_bold\\.nii(\\.gz)?$'
        return self._find_file(self._func_dir, pattern)
    
    def get_bold(self, space: str = None, desc: str = 'preproc') -> Optional[str]:
        """
        Get BOLD file for specific space and description.
        
        Parameters
        ----------
        space : str, optional
            Output space (e.g., 'MNI152NLin2009cAsym', 'T1w', 'fsnative')
        desc : str
            Description label (default: 'preproc')
        
        Returns
        -------
        str or None
            Path to BOLD file
        """
        if space:
            pattern = f'{self._base_pattern}_space-{space}_desc-{desc}_bold\\.nii(\\.gz)?$'
        else:
            pattern = f'{self._base_pattern}_desc-{desc}_bold\\.nii(\\.gz)?$'
        return self._find_file(self._func_dir, pattern)
    
    @property
    def available_spaces(self) -> List[str]:
        """List all available output spaces for this run."""
        if not os.path.exists(self._func_dir):
            return []
        spaces = set()
        for f in os.listdir(self._func_dir):
            if self._base_pattern in f and '_bold' in f:
                match = re.search(r'_space-([^_]+)_', f)
                if match:
                    spaces.add(match.group(1))
        return sorted(spaces)
    
    # -------------------------------------------------------------------------
    # Masks
    # -------------------------------------------------------------------------
    
    @property
    def brain_mask(self) -> Optional[str]:
        """Brain mask in BOLD space."""
        if self.space:
            pattern = f'{self._base_pattern}_space-{self.space}_desc-brain_mask\\.nii(\\.gz)?$'
        else:
            pattern = f'{self._base_pattern}_desc-brain_mask\\.nii(\\.gz)?$'
        return self._find_file(self._func_dir, pattern)
    
    def get_mask(self, space: str = None, desc: str = 'brain') -> Optional[str]:
        """Get mask file for specific space."""
        if space:
            pattern = f'{self._base_pattern}_space-{space}_desc-{desc}_mask\\.nii(\\.gz)?$'
        else:
            pattern = f'{self._base_pattern}_desc-{desc}_mask\\.nii(\\.gz)?$'
        return self._find_file(self._func_dir, pattern)
    
    # -------------------------------------------------------------------------
    # Confounds
    # -------------------------------------------------------------------------
    
    @property
    def confounds_tsv(self) -> Optional[str]:
        """Path to confounds TSV file."""
        pattern = f'{self._base_pattern}_desc-confounds_timeseries\\.tsv$'
        result = self._find_file(self._func_dir, pattern)
        if not result:
            # Try older naming convention
            pattern = f'{self._base_pattern}_desc-confounds_regressors\\.tsv$'
            result = self._find_file(self._func_dir, pattern)
        return result
    
    @property
    def confounds_json(self) -> Optional[str]:
        """Path to confounds metadata JSON."""
        pattern = f'{self._base_pattern}_desc-confounds_timeseries\\.json$'
        result = self._find_file(self._func_dir, pattern)
        if not result:
            pattern = f'{self._base_pattern}_desc-confounds_regressors\\.json$'
            result = self._find_file(self._func_dir, pattern)
        return result
    
    @cached_property
    def confounds(self) -> Optional[ConfoundsBundle]:
        """ConfoundsBundle object for easy access to confounds data."""
        tsv = self.confounds_tsv
        if tsv:
            return ConfoundsBundle(tsv_path=tsv, json_path=self.confounds_json)
        return None
    
    # -------------------------------------------------------------------------
    # Reference Images
    # -------------------------------------------------------------------------
    
    @property
    def boldref(self) -> Optional[str]:
        """BOLD reference image."""
        if self.space:
            pattern = f'{self._base_pattern}_space-{self.space}_boldref\\.nii(\\.gz)?$'
        else:
            pattern = f'{self._base_pattern}_boldref\\.nii(\\.gz)?$'
        return self._find_file(self._func_dir, pattern)
    
    # -------------------------------------------------------------------------
    # Transformations
    # -------------------------------------------------------------------------
    
    @property
    def bold_to_t1w_transform(self) -> Optional[str]:
        """Affine transform from BOLD to T1w space."""
        pattern = f'{self._base_pattern}_from-boldref_to-T1w_mode-image_desc-.*\\.txt$'
        return self._find_file(self._func_dir, pattern)
    
    @property
    def t1w_to_bold_transform(self) -> Optional[str]:
        """Affine transform from T1w to BOLD space."""
        pattern = f'{self._base_pattern}_from-T1w_to-boldref_mode-image_desc-.*\\.txt$'
        return self._find_file(self._func_dir, pattern)
    
    # -------------------------------------------------------------------------
    # Surface Data (if available)
    # -------------------------------------------------------------------------
    
    def get_surface_bold(self, hemi: str, space: str = 'fsaverage5') -> Optional[str]:
        """
        Get surface-sampled BOLD data.
        
        Parameters
        ----------
        hemi : str
            Hemisphere: 'L' or 'R'
        space : str
            Surface space (default: 'fsaverage5')
        
        Returns
        -------
        str or None
            Path to surface BOLD GIFTI file
        """
        pattern = f'{self._base_pattern}_space-{space}_hemi-{hemi}.*\\.func\\.gii$'
        return self._find_file(self._func_dir, pattern)
    
    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------
    
    @cached_property
    def bold_json(self) -> Dict:
        """BOLD sidecar JSON metadata."""
        pattern = f'{self._base_pattern}_bold\\.json$'
        json_path = self._find_file(self._func_dir, pattern)
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}
    
    @property
    def repetition_time(self) -> Optional[float]:
        """Repetition time (TR) in seconds."""
        return self.bold_json.get('RepetitionTime')
    
    @property
    def task_name(self) -> Optional[str]:
        """Task name from metadata."""
        return self.bold_json.get('TaskName', self.task)
    
    # -------------------------------------------------------------------------
    # Quick Access Bundle
    # -------------------------------------------------------------------------
    
    def get_denoising_bundle(self, space: str = None) -> Dict[str, Any]:
        """
        Get all files needed for denoising in one call.
        
        Parameters
        ----------
        space : str, optional
            Output space. If None, uses instance default.
        
        Returns
        -------
        dict
            Dictionary with keys: 'bold', 'mask', 'confounds', 'confounds_df', 'tr'
        """
        _space = space or self.space
        bold_pattern = f'{self._base_pattern}_space-{_space}_desc-preproc_bold\\.nii(\\.gz)?$' if _space else f'{self._base_pattern}_desc-preproc_bold\\.nii(\\.gz)?$'
        mask_pattern = f'{self._base_pattern}_space-{_space}_desc-brain_mask\\.nii(\\.gz)?$' if _space else f'{self._base_pattern}_desc-brain_mask\\.nii(\\.gz)?$'
        
        return {
            'bold': self._find_file(self._func_dir, bold_pattern),
            'mask': self._find_file(self._func_dir, mask_pattern),
            'confounds': self.confounds_tsv,
            'confounds_df': self.confounds.df if self.confounds else None,
            'tr': self.repetition_time
        }


@dataclass
class SubjectData:
    """
    All data for a single subject.
    
    Provides structured access to anatomical and functional data for a subject,
    with support for multiple sessions.
    """
    subject_id: str
    root: str
    _sessions: List[str] = field(default_factory=list, repr=False)
    
    @property
    def label(self) -> str:
        """Subject label (e.g., 'sub-01')."""
        return f'sub-{self.subject_id}'
    
    @property
    def path(self) -> str:
        """Path to subject directory."""
        return os.path.join(self.root, f'sub-{self.subject_id}')
    
    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------
    
    @cached_property
    def sessions(self) -> List[str]:
        """List of session labels (without 'ses-' prefix)."""
        if self._sessions:
            return self._sessions
        if not os.path.exists(self.path):
            return []
        sessions = []
        for item in os.listdir(self.path):
            if item.startswith('ses-') and os.path.isdir(os.path.join(self.path, item)):
                sessions.append(item.replace('ses-', ''))
        return sorted(sessions)
    
    @property
    def has_sessions(self) -> bool:
        """Whether this subject has multiple sessions."""
        return len(self.sessions) > 0
    
    @property
    def n_sessions(self) -> int:
        """Number of sessions."""
        return len(self.sessions) if self.sessions else 1
    
    # -------------------------------------------------------------------------
    # Anatomical Data
    # -------------------------------------------------------------------------
    
    @property
    def _anat_dir(self) -> str:
        """Path to anatomical directory (handles sessions)."""
        # Check for session-level anat first
        if self.sessions:
            # Use first session's anat if exists
            ses_anat = os.path.join(self.path, f'ses-{self.sessions[0]}', 'anat')
            if os.path.exists(ses_anat):
                return ses_anat
        # Fall back to subject-level anat
        return os.path.join(self.path, 'anat')
    
    def _find_anat_file(self, pattern: str) -> Optional[str]:
        """Find anatomical file matching pattern."""
        # First check subject-level anat
        subj_anat = os.path.join(self.path, 'anat')
        if os.path.exists(subj_anat):
            for f in os.listdir(subj_anat):
                if re.search(pattern, f):
                    return os.path.join(subj_anat, f)
        # Then check session-level
        for ses in self.sessions:
            ses_anat = os.path.join(self.path, f'ses-{ses}', 'anat')
            if os.path.exists(ses_anat):
                for f in os.listdir(ses_anat):
                    if re.search(pattern, f):
                        return os.path.join(ses_anat, f)
        return None
    
    @property
    def t1w_preproc(self) -> Optional[str]:
        """Preprocessed T1w image in native space."""
        pattern = r'sub-' + self.subject_id + r'.*_desc-preproc_T1w\.nii(\.gz)?$'
        return self._find_anat_file(pattern)
    
    @property
    def brain_mask(self) -> Optional[str]:
        """Brain mask in T1w space."""
        pattern = r'sub-' + self.subject_id + r'.*_desc-brain_mask\.nii(\.gz)?$'
        return self._find_anat_file(pattern)
    
    @property
    def gm_probseg(self) -> Optional[str]:
        """Gray matter probability segmentation."""
        pattern = r'sub-' + self.subject_id + r'.*_label-GM_probseg\.nii(\.gz)?$'
        return self._find_anat_file(pattern)
    
    @property
    def wm_probseg(self) -> Optional[str]:
        """White matter probability segmentation."""
        pattern = r'sub-' + self.subject_id + r'.*_label-WM_probseg\.nii(\.gz)?$'
        return self._find_anat_file(pattern)
    
    @property
    def csf_probseg(self) -> Optional[str]:
        """CSF probability segmentation."""
        pattern = r'sub-' + self.subject_id + r'.*_label-CSF_probseg\.nii(\.gz)?$'
        return self._find_anat_file(pattern)
    
    def get_t1w(self, space: str = None, desc: str = 'preproc') -> Optional[str]:
        """
        Get T1w image in specific space.
        
        Parameters
        ----------
        space : str, optional
            Output space (e.g., 'MNI152NLin2009cAsym')
        desc : str
            Description label (default: 'preproc')
        
        Returns
        -------
        str or None
            Path to T1w file
        """
        if space:
            pattern = r'sub-' + self.subject_id + r'.*_space-' + space + r'_desc-' + desc + r'_T1w\.nii(\.gz)?$'
        else:
            pattern = r'sub-' + self.subject_id + r'.*_desc-' + desc + r'_T1w\.nii(\.gz)?$'
        return self._find_anat_file(pattern)
    
    # -------------------------------------------------------------------------
    # Transformations
    # -------------------------------------------------------------------------
    
    @property
    def t1w_to_mni_transform(self) -> Optional[str]:
        """Transform from T1w to MNI space."""
        pattern = r'sub-' + self.subject_id + r'.*_from-T1w_to-MNI.*\.h5$'
        return self._find_anat_file(pattern)
    
    @property
    def mni_to_t1w_transform(self) -> Optional[str]:
        """Transform from MNI to T1w space."""
        pattern = r'sub-' + self.subject_id + r'.*_from-MNI.*_to-T1w.*\.h5$'
        return self._find_anat_file(pattern)
    
    # -------------------------------------------------------------------------
    # Functional Runs
    # -------------------------------------------------------------------------
    
    @cached_property
    def _all_runs(self) -> List[Dict]:
        """Discover all functional runs for this subject."""
        runs = []
        
        # Helper to scan a func directory
        def scan_func_dir(func_dir: str, session: str = None):
            if not os.path.exists(func_dir):
                return
            bold_files = [f for f in os.listdir(func_dir) if '_bold' in f and f.endswith(('.nii', '.nii.gz'))]
            
            for bf in bold_files:
                # Parse entities from filename
                task_match = re.search(r'_task-([^_]+)', bf)
                run_match = re.search(r'_run-([^_]+)', bf)
                
                if task_match:
                    run_info = {
                        'task': task_match.group(1),
                        'run': run_match.group(1) if run_match else None,
                        'session': session
                    }
                    if run_info not in runs:
                        runs.append(run_info)
        
        # Check subject-level func
        scan_func_dir(os.path.join(self.path, 'func'))
        
        # Check session-level func
        for ses in self.sessions:
            scan_func_dir(os.path.join(self.path, f'ses-{ses}', 'func'), session=ses)
        
        return runs
    
    @property
    def tasks(self) -> List[str]:
        """List of unique task labels."""
        return sorted(set(r['task'] for r in self._all_runs))
    
    @property
    def n_runs(self) -> int:
        """Total number of functional runs."""
        return len(self._all_runs)
    
    def get_runs(self, task: str = None, session: str = None, space: str = None) -> List[RunFiles]:
        """
        Get RunFiles objects for functional runs.
        
        Parameters
        ----------
        task : str, optional
            Filter by task label
        session : str, optional
            Filter by session
        space : str, optional
            Default output space for retrieved runs
        
        Returns
        -------
        List[RunFiles]
            List of RunFiles objects
        """
        runs = []
        for r in self._all_runs:
            if task and r['task'] != task:
                continue
            if session and r['session'] != session:
                continue
            runs.append(RunFiles(
                subject=self.subject_id,
                session=r['session'],
                task=r['task'],
                run=r['run'],
                root=self.root,
                space=space
            ))
        return runs
    
    def get_run(self, task: str, run: str = None, session: str = None, space: str = None) -> Optional[RunFiles]:
        """
        Get a specific functional run.
        
        Parameters
        ----------
        task : str
            Task label
        run : str, optional
            Run label (if multiple runs)
        session : str, optional
            Session label
        space : str, optional
            Default output space
        
        Returns
        -------
        RunFiles or None
            RunFiles object if found
        """
        runs = self.get_runs(task=task, session=session, space=space)
        if run:
            runs = [r for r in runs if r.run == run]
        return runs[0] if runs else None
    
    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    
    @property
    def html_report(self) -> Optional[str]:
        """Path to fMRIPrep HTML report."""
        report_path = os.path.join(self.root, f'sub-{self.subject_id}.html')
        return report_path if os.path.exists(report_path) else None
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def iter_runs(self, task: str = None, space: str = None) -> Iterator[RunFiles]:
        """
        Iterate over functional runs.
        
        Parameters
        ----------
        task : str, optional
            Filter by task
        space : str, optional
            Default output space
        
        Yields
        ------
        RunFiles
            RunFiles objects for each run
        """
        for run in self.get_runs(task=task, space=space):
            yield run


# =============================================================================
# Main Parser Class
# =============================================================================

class BIDSFmriprepParser:
    """
    Parser for fMRIPrep BIDS derivatives directories.
    
    Provides a streamlined, Pythonic interface for accessing preprocessed
    fMRI data from fMRIPrep outputs.
    
    Examples
    --------
    >>> # Initialize parser
    >>> fp = BIDSFmriprepParser('./derivatives/fmriprep')
    
    >>> # Quick access to subjects
    >>> fp.subjects  # ['sub-01', 'sub-02', ...]
    >>> fp.n_subjects  # 14
    
    >>> # Access subject data
    >>> sub = fp['sub-01']  # or fp.get_subject('01')
    >>> sub.t1w_preproc
    >>> sub.brain_mask
    
    >>> # Access functional runs
    >>> run = sub.get_run(task='rest', run='01')
    >>> run.bold_preproc
    >>> run.confounds.motion_params
    >>> run.confounds.get_strategy('24P+aCompCor')
    
    >>> # Loop over all subjects and runs
    >>> for sub in fp.iter_subjects():
    ...     for run in sub.iter_runs(task='rest'):
    ...         print(run.bold_preproc)
    """
    
    def __init__(self, root: str, validate: bool = False, default_space: str = None):
        """
        Initialize the BIDS fMRIPrep parser.
        
        Parameters
        ----------
        root : str
            Path to fMRIPrep derivatives directory
        validate : bool
            Whether to validate the directory structure (default: False)
        default_space : str, optional
            Default output space for file queries (e.g., 'MNI152NLin2009cAsym')
        """
        self.root = os.path.abspath(root)
        self.default_space = default_space
        
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Directory not found: {self.root}")
        
        if validate:
            self._validate()
    
    def _validate(self):
        """Basic validation of fMRIPrep output structure."""
        # Check for dataset_description.json
        desc_file = os.path.join(self.root, 'dataset_description.json')
        if not os.path.exists(desc_file):
            warnings.warn("No dataset_description.json found. This may not be a valid BIDS derivatives directory.")
    
    # -------------------------------------------------------------------------
    # Dataset Description
    # -------------------------------------------------------------------------
    
    @cached_property
    def dataset_description(self) -> Dict:
        """Dataset description metadata."""
        desc_file = os.path.join(self.root, 'dataset_description.json')
        if os.path.exists(desc_file):
            with open(desc_file, 'r') as f:
                return json.load(f)
        return {}
    
    @property
    def fmriprep_version(self) -> Optional[str]:
        """fMRIPrep version used for processing."""
        generated_by = self.dataset_description.get('GeneratedBy', [])
        if generated_by:
            for gen in generated_by:
                if gen.get('Name', '').lower() == 'fmriprep':
                    return gen.get('Version')
        # Try legacy PipelineDescription
        pipeline = self.dataset_description.get('PipelineDescription', {})
        if pipeline.get('Name', '').lower() == 'fmriprep':
            return pipeline.get('Version')
        return None
    
    # -------------------------------------------------------------------------
    # Subject Discovery
    # -------------------------------------------------------------------------
    
    @cached_property
    def _subject_ids(self) -> List[str]:
        """Discover all subject IDs."""
        subjects = []
        for item in os.listdir(self.root):
            if item.startswith('sub-') and os.path.isdir(os.path.join(self.root, item)):
                subjects.append(item.replace('sub-', ''))
        return sorted(subjects)
    
    @property
    def subjects(self) -> List[str]:
        """List of subject labels (with 'sub-' prefix)."""
        return [f'sub-{s}' for s in self._subject_ids]
    
    @property
    def subject_ids(self) -> List[str]:
        """List of subject IDs (without 'sub-' prefix)."""
        return self._subject_ids
    
    @property
    def n_subjects(self) -> int:
        """Number of subjects."""
        return len(self._subject_ids)
    
    @property
    def total_subjects(self) -> int:
        """Alias for n_subjects."""
        return self.n_subjects
    
    # -------------------------------------------------------------------------
    # Subject Access
    # -------------------------------------------------------------------------
    
    def get_subject(self, subject_id: str) -> SubjectData:
        """
        Get SubjectData object for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (with or without 'sub-' prefix)
        
        Returns
        -------
        SubjectData
            SubjectData object for the subject
        """
        # Handle both 'sub-01' and '01' formats
        if subject_id.startswith('sub-'):
            subject_id = subject_id.replace('sub-', '')
        
        if subject_id not in self._subject_ids:
            raise KeyError(f"Subject '{subject_id}' not found. Available: {self._subject_ids}")
        
        return SubjectData(subject_id=subject_id, root=self.root)
    
    def __getitem__(self, subject_id: str) -> SubjectData:
        """Allow dictionary-style access: parser['sub-01']"""
        return self.get_subject(subject_id)
    
    def __contains__(self, subject_id: str) -> bool:
        """Check if subject exists: 'sub-01' in parser"""
        if subject_id.startswith('sub-'):
            subject_id = subject_id.replace('sub-', '')
        return subject_id in self._subject_ids
    
    def __len__(self) -> int:
        """Number of subjects."""
        return self.n_subjects
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over subject labels."""
        return iter(self.subjects)
    
    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------
    
    def iter_subjects(self) -> Iterator[SubjectData]:
        """
        Iterate over all subjects.
        
        Yields
        ------
        SubjectData
            SubjectData object for each subject
        """
        for subj_id in self._subject_ids:
            yield SubjectData(subject_id=subj_id, root=self.root)
    
    def iter_runs(self, task: str = None, space: str = None) -> Iterator[Tuple[SubjectData, RunFiles]]:
        """
        Iterate over all functional runs across all subjects.
        
        Parameters
        ----------
        task : str, optional
            Filter by task
        space : str, optional
            Default output space
        
        Yields
        ------
        tuple
            (SubjectData, RunFiles) for each run
        """
        _space = space or self.default_space
        for subject in self.iter_subjects():
            for run in subject.iter_runs(task=task, space=_space):
                yield subject, run
    
    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------
    
    def get_all_bold(self, task: str = None, space: str = None, desc: str = 'preproc') -> List[str]:
        """
        Get all preprocessed BOLD files across subjects.
        
        Parameters
        ----------
        task : str, optional
            Filter by task
        space : str, optional
            Output space
        desc : str
            Description label (default: 'preproc')
        
        Returns
        -------
        List[str]
            List of BOLD file paths
        """
        _space = space or self.default_space
        bold_files = []
        for subject, run in self.iter_runs(task=task, space=_space):
            bold = run.get_bold(space=_space, desc=desc)
            if bold:
                bold_files.append(bold)
        return bold_files
    
    def get_all_confounds(self, task: str = None) -> List[str]:
        """
        Get all confounds TSV files across subjects.
        
        Parameters
        ----------
        task : str, optional
            Filter by task
        
        Returns
        -------
        List[str]
            List of confounds TSV file paths
        """
        confound_files = []
        for subject, run in self.iter_runs(task=task):
            conf = run.confounds_tsv
            if conf:
                confound_files.append(conf)
        return confound_files
    
    def get_all_masks(self, task: str = None, space: str = None) -> List[str]:
        """
        Get all brain masks across subjects.
        
        Parameters
        ----------
        task : str, optional
            Filter by task
        space : str, optional
            Output space
        
        Returns
        -------
        List[str]
            List of mask file paths
        """
        _space = space or self.default_space
        mask_files = []
        for subject, run in self.iter_runs(task=task, space=_space):
            mask = run.get_mask(space=_space)
            if mask:
                mask_files.append(mask)
        return mask_files
    
    # -------------------------------------------------------------------------
    # Quality Control
    # -------------------------------------------------------------------------
    
    def get_motion_summary(self, task: str = None) -> pd.DataFrame:
        """
        Get motion summary statistics for all subjects.
        
        Parameters
        ----------
        task : str, optional
            Filter by task
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: subject, session, task, run, mean_fd, max_fd, n_high_motion
        """
        data = []
        for subject, run in self.iter_runs(task=task):
            if run.confounds:
                conf = run.confounds
                data.append({
                    'subject': subject.subject_id,
                    'session': run.session,
                    'task': run.task,
                    'run': run.run,
                    'mean_fd': conf.mean_fd,
                    'max_fd': conf.max_fd,
                    'n_high_motion_05': conf.get_high_motion_volumes(0.5).sum(),
                    'n_high_motion_09': conf.get_high_motion_volumes(0.9).sum(),
                    'n_timepoints': conf.n_timepoints
                })
        return pd.DataFrame(data)
    
    # -------------------------------------------------------------------------
    # Discovery Methods
    # -------------------------------------------------------------------------
    
    @cached_property
    def available_tasks(self) -> List[str]:
        """List of all unique task labels across all subjects."""
        tasks = set()
        for subject in self.iter_subjects():
            tasks.update(subject.tasks)
        return sorted(tasks)
    
    @cached_property
    def available_spaces(self) -> List[str]:
        """List of all available output spaces."""
        spaces = set()
        for subject, run in self.iter_runs():
            spaces.update(run.available_spaces)
        return sorted(spaces)
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    def summary(self) -> Dict:
        """
        Get a summary of the dataset.
        
        Returns
        -------
        dict
            Summary statistics
        """
        return {
            'root': self.root,
            'n_subjects': self.n_subjects,
            'subjects': self.subjects,
            'tasks': self.available_tasks,
            'spaces': self.available_spaces,
            'fmriprep_version': self.fmriprep_version
        }
    
    def __repr__(self) -> str:
        return f"BIDSFmriprepParser('{self.root}', n_subjects={self.n_subjects})"
    
    def __str__(self) -> str:
        return f"fMRIPrep Derivatives: {self.root}\n  Subjects: {self.n_subjects}\n  Tasks: {self.available_tasks}"


# =============================================================================
# Convenience Function
# =============================================================================

def bids_parser(root: str, **kwargs) -> BIDSFmriprepParser:
    """
    Convenience function to create a BIDSFmriprepParser.
    
    Parameters
    ----------
    root : str
        Path to fMRIPrep derivatives directory
    **kwargs
        Additional arguments passed to BIDSFmriprepParser
    
    Returns
    -------
    BIDSFmriprepParser
        Parser object
    
    Examples
    --------
    >>> fp = bids_parser('./Data/studyforrest-fmriprep')
    >>> fp.subjects
    ['sub-01', 'sub-02', ...]
    """
    return BIDSFmriprepParser(root, **kwargs)


# =============================================================================
# Example Usage and Documentation
# =============================================================================

if __name__ == '__main__':
    # Example usage demonstration
    example_usage = '''
    # ==========================================================================
    # USAGE EXAMPLES
    # ==========================================================================
    
    # Initialize the parser
    fp = bids_parser('./Data/studyforrest-fmriprep')
    
    # -----------------------------------------------------------------------------
    # Basic Navigation
    # -----------------------------------------------------------------------------
    
    # List subjects
    print(fp.subjects)           # ['sub-01', 'sub-02', ...]
    print(fp.total_subjects)     # 14
    print(fp.n_subjects)         # 14 (same as total_subjects)
    
    # Check if subject exists
    'sub-01' in fp               # True
    
    # Access a subject
    sub = fp['sub-01']           # Dictionary-style access
    sub = fp.get_subject('01')   # Alternative method
    
    # Subject properties
    sub.sessions                 # ['ses-01', 'ses-02'] or []
    sub.has_sessions             # True/False
    sub.tasks                    # ['rest', 'movie', ...]
    sub.n_runs                   # 8
    
    # -----------------------------------------------------------------------------
    # Anatomical Data
    # -----------------------------------------------------------------------------
    
    # Native space T1w
    sub.t1w_preproc              # '/path/to/sub-01_desc-preproc_T1w.nii.gz'
    sub.brain_mask               # '/path/to/sub-01_desc-brain_mask.nii.gz'
    sub.gm_probseg               # '/path/to/sub-01_label-GM_probseg.nii.gz'
    sub.wm_probseg               # White matter probability
    sub.csf_probseg              # CSF probability
    
    # Standard space
    sub.get_t1w(space='MNI152NLin2009cAsym')
    
    # Transforms
    sub.t1w_to_mni_transform     # For spatial normalization
    
    # -----------------------------------------------------------------------------
    # Functional Runs
    # -----------------------------------------------------------------------------
    
    # Get a specific run
    run = sub.get_run(task='rest', run='01')
    
    # Or get all runs for a task
    runs = sub.get_runs(task='rest')
    
    # Run properties
    run.bold_preproc             # Preprocessed BOLD
    run.brain_mask               # Brain mask in BOLD space
    run.confounds_tsv            # Confounds file path
    run.boldref                  # BOLD reference image
    run.repetition_time          # TR in seconds
    run.available_spaces         # ['MNI152NLin2009cAsym', 'T1w', ...]
    
    # Get BOLD in specific space
    run.get_bold(space='MNI152NLin2009cAsym')
    
    # Get all files needed for denoising
    bundle = run.get_denoising_bundle(space='MNI152NLin2009cAsym')
    # Returns: {'bold': path, 'mask': path, 'confounds': path, 
    #           'confounds_df': DataFrame, 'tr': 2.0}
    
    # -----------------------------------------------------------------------------
    # Confounds - The Power Feature!
    # -----------------------------------------------------------------------------
    
    conf = run.confounds
    
    # Quick stats
    conf.n_timepoints            # 300
    conf.mean_fd                 # 0.23
    conf.max_fd                  # 1.45
    
    # Motion parameters
    conf.motion_params           # 6-parameter DataFrame
    conf.motion_params_24        # 24-parameter motion model
    
    # CompCor
    conf.acompcor                # All anatomical CompCor components
    conf.tcompcor                # Temporal CompCor
    conf.get_acompcor(n_components=5)  # Top 5 aCompCor
    
    # Global signals
    conf.global_signal           # Global signal time series
    conf.csf                     # CSF signal
    conf.white_matter            # WM signal
    
    # Quality metrics
    conf.framewise_displacement  # FD time series
    conf.dvars                   # Standardized DVARS
    conf.get_high_motion_volumes(fd_threshold=0.5)  # Boolean mask
    conf.non_steady_state_outliers  # Indices of dummy scans
    
    # Pre-built denoising strategies!
    conf.get_strategy('6P')                    # 6 motion params
    conf.get_strategy('24P')                   # 24-param motion model
    conf.get_strategy('24P+aCompCor')          # Motion + CompCor
    conf.get_strategy('24P+aCompCor+GSR')      # + Global signal regression
    
    # Custom confounds matrix
    conf.to_matrix(columns=['trans_x', 'trans_y', 'trans_z'])
    
    # -----------------------------------------------------------------------------
    # Bulk Operations
    # -----------------------------------------------------------------------------
    
    # Get all BOLD files
    all_bold = fp.get_all_bold(task='rest', space='MNI152NLin2009cAsym')
    
    # Get all confounds
    all_conf = fp.get_all_confounds(task='rest')
    
    # Motion summary across dataset
    motion_df = fp.get_motion_summary(task='rest')
    # Returns DataFrame: subject, session, task, run, mean_fd, max_fd, n_high_motion
    
    # -----------------------------------------------------------------------------
    # Looping Patterns
    # -----------------------------------------------------------------------------
    
    # Loop over subjects
    for sub in fp.iter_subjects():
        print(sub.t1w_preproc)
    
    # Loop over all runs
    for sub, run in fp.iter_runs(task='rest'):
        print(f"{sub.subject_id}: {run.bold_preproc}")
    
    # Loop over runs for a subject
    for run in sub.iter_runs():
        print(run.confounds.mean_fd)
    
    # -----------------------------------------------------------------------------
    # Practical Example: Denoising Pipeline
    # -----------------------------------------------------------------------------
    
    import nibabel as nib
    from nilearn.signal import clean
    
    for sub, run in fp.iter_runs(task='rest', space='MNI152NLin2009cAsym'):
        # Get all files
        bold = nib.load(run.bold_preproc)
        
        # Get confounds for your strategy
        confounds = run.confounds.get_strategy('24P+aCompCor', n_compcor=5)
        
        # Get mask
        mask = run.brain_mask
        
        # Check quality
        if run.confounds.mean_fd > 0.5:
            print(f"High motion subject: {sub.subject_id}")
            continue
        
        # Perform denoising
        cleaned = clean(
            bold.get_fdata(),
            confounds=confounds.values,
            t_r=run.repetition_time,
            detrend=True,
            standardize=True
        )
    
    # -----------------------------------------------------------------------------
    # Practical Example: Connectivity Analysis Prep
    # -----------------------------------------------------------------------------
    
    from nilearn.maskers import NiftiLabelsMasker
    
    atlas = 'path/to/schaefer_400.nii.gz'
    masker = NiftiLabelsMasker(atlas)
    
    all_timeseries = {}
    for sub, run in fp.iter_runs(task='rest', space='MNI152NLin2009cAsym'):
        # Extract time series
        ts = masker.fit_transform(
            run.bold_preproc,
            confounds=run.confounds.get_strategy('24P+aCompCor+GSR')
        )
        all_timeseries[sub.subject_id] = ts
    '''
    
    print(example_usage)
