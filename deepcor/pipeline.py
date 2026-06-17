"""High-level API for DeepCor fMRI denoising.

This module exposes :class:`DeepCorDenoiser` (aliased as :class:`DeepCor`), a
scikit-learn-style entry point that wraps the full DeepCor workflow — load data,
train an ensemble of CVAE models with a live training dashboard, then average the
ensemble and write the denoised output (plus preprocessed and CompCor
comparisons).

All version-specific behaviour is isolated in the ``_VERSION_SPECS`` table below,
so supporting a future ``CVAE_V3`` is a single additive entry (see
``wiki/backwards_compatibility.md``).
"""

import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
import torch
import ants

from .models import get_model
from .data import (
    TrainDataset,
    get_obs_noi_list,
    get_obs_noi_list_coords,
    get_confounds,
    array_to_brain,
)
from .training import Trainer, save_brain_signals
from .analysis import average_signal_ensemble, calc_and_save_compcor
from .config import DeepCorConfig
from .utils import safe_mkdir
from . import visualization as viz


# ---------------------------------------------------------------------------
# Version adapter
# ---------------------------------------------------------------------------
# Each spec captures the *only* things that differ between CVAE versions. To add
# a new version, append one spec + a model builder; nothing else in this file
# (or in the notebooks) needs to change.

def _build_v1_model(nTR, in_channels, conf_tensor, config):
    """CVAE_V1: no confounds, a single scalar latent dim per branch, one beta."""
    latent = config.model.latent_dims
    latent_dim = int(latent[0]) if isinstance(latent, (tuple, list)) else int(latent)
    return get_model(
        "v1",
        in_channels=in_channels,
        in_dim=nTR,
        latent_dim=latent_dim,
        beta=config.model.beta,
    )


def _build_v2_model(nTR, in_channels, conf_tensor, config):
    """CVAE (v2): confound-aware, tuple latent dim, disentanglement terms."""
    return get_model(
        "v2",
        conf=conf_tensor,
        in_channels=in_channels,
        in_dim=nTR,
        latent_dim=config.model.latent_dims,
        beta=config.model.beta,
        gamma=config.model.gamma,
        delta=config.model.delta,
        scale_MSE_GM=config.model.scale_MSE_GM,
        scale_MSE_CF=config.model.scale_MSE_CF,
        scale_MSE_FG=config.model.scale_MSE_FG,
        do_disentangle=config.model.do_disentangle,
    )


@dataclass
class _VersionSpec:
    registry_key: str            # key understood by models.get_model
    loader: Callable             # (epi, gm, cf) -> (obs_list, noi_list, gm, cf)
    needs_confounds: bool        # whether the model requires a confounds tensor
    track_version: str           # init_track schema id ('V1', 'V2', ...)
    build_model: Callable        # (nTR, in_channels, conf_tensor, config) -> model


_VERSION_SPECS = {
    "v1": _VersionSpec("v1", get_obs_noi_list, False, "V1", _build_v1_model),
    "v2": _VersionSpec("v2", get_obs_noi_list_coords, True, "V2", _build_v2_model),
}

# Stable aliases that always point at the latest recommended version.
_VERSION_ALIASES = {"latest": "v2", "cvae": "v2"}

# Where outputs land when the caller does not pass an output_dir. Relative, so it
# resolves against the current working directory (notebooks chdir to their own
# folder, matching the StudyForrest layout).
DEFAULT_OUTPUT_DIR = "../Data/DeepCor-Outputs"


def _resolve_spec(model_version):
    key = str(model_version).lower()
    key = _VERSION_ALIASES.get(key, key)
    if key not in _VERSION_SPECS:
        available = sorted(set(_VERSION_SPECS) | set(_VERSION_ALIASES))
        raise ValueError(
            f"Unknown model_version {model_version!r}; available: {available}"
        )
    return key, _VERSION_SPECS[key]


def _as_image(img):
    """Accept either a path or an already-loaded ANTs image."""
    if isinstance(img, str):
        return ants.image_read(img)
    return img


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class DeepCorResult:
    """Paths and artifacts produced by a fit/denoise run.

    Implements ``__fspath__``/``__str__`` returning ``denoised_path`` so it can
    be passed anywhere a path string is expected (e.g. ``ants.image_read(result)``).
    """

    denoised_path: Optional[str] = None
    preproc_path: Optional[str] = None
    compcor_path: Optional[str] = None
    output_dir: Optional[str] = None
    signal_files: List[str] = field(default_factory=list)
    tracks: List[dict] = field(default_factory=list)

    def __fspath__(self):
        return self.denoised_path or ""

    def __str__(self):
        return self.denoised_path or repr(self)


# ---------------------------------------------------------------------------
# High-level denoiser
# ---------------------------------------------------------------------------
class DeepCorDenoiser:
    """High-level, scikit-learn-style API for DeepCor fMRI denoising.

    Typical usage::

        denoiser = DeepCor(model_version="v2")          # config optional
        result = denoiser.fit_denoise(
            epi_path, gm_mask_path, cf_mask_path, confounds_path, output_dir,
        )
        result.denoised_path

    ``fit()`` and ``denoise()`` are also available separately;
    ``fit_denoise()`` is a thin convenience wrapper around the two.
    """

    def __init__(
        self,
        model_version="latest",
        latent_dims=(8, 8),
        n_epochs=100,
        batch_size=1024,
        learning_rate=0.001,
        n_repetitions=20,
        config=None,
        device=None,
        verbose=True,
    ):
        """Initialize the denoiser.

        Args:
            model_version: 'v1', 'v2', 'latest'/'cvae'. Selects loader, model
                and track schema via the version-adapter table.
            latent_dims: (signal_dim, noise_dim). v1 uses the first entry.
            n_epochs, batch_size, learning_rate, n_repetitions: training knobs
                used only when ``config`` is not supplied.
            config: optional DeepCorConfig; if given, overrides the scalar args.
            device: torch device (defaults to GPU when available).
            verbose: print device/progress info.
        """
        self.model_version, self._spec = _resolve_spec(model_version)

        if config is None:
            self.config = DeepCorConfig()
            self.config.model.latent_dims = latent_dims
            self.config.training.n_epochs = n_epochs
            self.config.training.batch_size = batch_size
            self.config.training.learning_rate = learning_rate
            self.config.training.n_repetitions = n_repetitions
        else:
            self.config = config

        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Populated by fit()
        self.models = []
        self.trainers = []
        self.tracks = []
        self.signal_files = []
        self._epi = self._gm = self._cf = None
        self._obs_list = None
        self._fitted_output_dir = None

        if verbose:
            print(f"device is {self.device}")
            print(f"model_version is {self.model_version}")

    # -- internal helpers ---------------------------------------------------
    def _load_confounds(self, confounds):
        """Return a (n_confounds, nTR) array, or None."""
        if confounds is None:
            return None
        if isinstance(confounds, str):
            return get_confounds(
                confounds,
                columns=self.config.data.confound_columns,
                norm="zscore",
            )
        return np.asarray(confounds, dtype=float)

    def _render(self, track, dashboard, on_epoch, epoch, rep):
        """Render the per-epoch dashboard.

        Always renders a *fresh* figure and closes it afterwards (rather than
        reusing one across epochs), so each saved PNG is a single clean
        dashboard rather than every epoch overlaid. Rendering goes through the
        marimo-free Jupyter backend; 'save' disables the in-place display so it
        needs neither marimo nor IPython installed. Returns None.
        """
        if on_epoch is not None:
            on_epoch(track, epoch=epoch, rep=rep)

        if dashboard in (None, False, "none"):
            return None
        if dashboard not in ("jupyter", "save"):
            raise ValueError(
                f"Unknown dashboard mode {dashboard!r}; use 'jupyter', 'save', "
                "None, or pass an on_epoch callback."
            )

        import matplotlib.pyplot as plt

        # 'jupyter' displays in-place + saves; 'save' only renders + saves.
        want_display = dashboard == "jupyter"
        try:
            f = viz.show_dahsboard_jupyter(
                track, fig=None, save_fig=True, display=want_display
            )
        except Exception:
            # IPython unavailable for display: still render + save (no display).
            f = viz.show_dahsboard_jupyter(
                track, fig=None, save_fig=True, display=False
            )
        plt.close(f)  # fresh figure each epoch; bound memory
        return None

    # -- public API ---------------------------------------------------------
    def fit(
        self,
        epi,
        gm_mask,
        cf_mask,
        confounds=None,
        output_dir=None,
        dashboard="save",
        on_epoch=None,
        subject_idx=None,
        run_idx=None,
        verbose=True,
    ):
        """Train the ensemble of models on a single EPI run.

        Args:
            epi, gm_mask, cf_mask: paths or already-loaded ANTs images.
            confounds: path to an fMRIPrep confounds TSV (or a (n_conf, nTR)
                array). Required for confound-aware versions (e.g. v2), ignored
                for v1.
            output_dir: where per-repetition checkpoints, tracks and signal
                NIfTIs are written. Created (with parents) if missing. If None,
                defaults to DEFAULT_OUTPUT_DIR ('../Data/DeepCor-Outputs').
            dashboard: 'save' (save the dashboard PNG each epoch, no interactive
                display — the default), 'jupyter' (also display it in-place), or
                None (no dashboard at all). 'save'/'jupyter' both use the same
                figure; only display differs.
            on_epoch: optional callback ``fn(track, epoch=, rep=)`` invoked every
                epoch — used by marimo notebooks to render live without making
                marimo a dependency of this library.
            subject_idx, run_idx: optional labels written to the config and used
                to name the saved dashboard PNG and the progress line
                (e.g. 'S0R4'). Default to the config values (0).
            verbose: print a per-epoch progress line (subject/run, start time,
                elapsed, ensemble/epoch counters, ETA).

        Returns:
            self
        """
        spec = self._spec
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        safe_mkdir(output_dir)  # creates parents too
        self.config.data.output_dir = output_dir
        if subject_idx is not None:
            self.config.data.subject_idx = subject_idx
        if run_idx is not None:
            self.config.data.run_idx = run_idx

        epi = _as_image(epi)
        gm = _as_image(gm_mask)
        cf = _as_image(cf_mask)

        if verbose:
            print("Preparing training data...")
        obs_list, noi_list, gm, cf = spec.loader(epi, gm, cf)

        conf = self._load_confounds(confounds)
        if spec.needs_confounds and conf is None:
            raise ValueError(
                f"model_version {self.model_version!r} requires confounds; "
                "pass confounds=<path to confounds .tsv>."
            )

        # Optional dummy-scan removal (default 0 => no-op). Applied on the
        # extracted time axis so it is version-agnostic.
        n_dummy = self.config.data.n_dummy_scans
        if n_dummy and n_dummy > 0:
            obs_list = obs_list[..., n_dummy:]
            noi_list = noi_list[..., n_dummy:]
            if conf is not None:
                conf = conf[..., n_dummy:]

        nTR = obs_list.shape[-1]
        in_channels = obs_list.shape[-2]  # data-adaptive: 1 (v1) or 4 (v2)

        train_dataset = TrainDataset(obs_list, noi_list)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True,
        )

        conf_tensor = None
        if conf is not None:
            conf_tensor = torch.tensor(
                np.array([conf for _ in range(self.config.training.batch_size)])
            )

        n_reps = self.config.training.n_repetitions
        T_overall_start = datetime.now()
        for rep in range(n_reps):
            self.config.training.current_ensemble = rep
            try:
                track = viz.init_track(spec.track_version)
                track["T_overall_start"] = T_overall_start

                model = spec.build_model(
                    nTR, in_channels, conf_tensor, self.config
                ).to(self.device)
                trainer = Trainer(
                    model,
                    device=self.device,
                    optimizer_type=self.config.training.optimizer,
                    lr=self.config.training.learning_rate,
                    betas=self.config.training.betas,
                    eps=self.config.training.eps,
                    max_grad_norm=self.config.training.max_grad_norm,
                )

                avg_loss = float("nan")
                for epoch in range(self.config.training.n_epochs):
                    self.config.training.current_epoch = epoch
                    avg_loss = trainer.train_epoch(train_loader)
                    viz.update_track(track, train_loader, model, self.config)
                    self._render(track, dashboard, on_epoch, epoch, rep)
                    if verbose:
                        # Same subject/run/started/elapsed/ETA info as the saved
                        # dashboard title; updates the line in place.
                        print("\r" + viz.format_progress_title(track),
                              end="", flush=True)
                if verbose:
                    print()  # finalize the in-place progress line

                if output_dir:
                    trainer.save_checkpoint(
                        os.path.join(output_dir, f"model_final_ens{rep}.pt"),
                        self.config.training.n_epochs,
                        avg_loss,
                    )
                    viz.save_track(
                        os.path.join(output_dir, f"track_rep_{rep}.pickle"), track
                    )
                    signal_path = os.path.join(output_dir, f"signal_rep_{rep}.nii.gz")
                    save_brain_signals(
                        model, train_dataset, epi, gm,
                        ofn=signal_path, batch_size=512, kind="FG",
                    )
                    self.signal_files.append(signal_path)

                self.models.append(model)
                self.trainers.append(trainer)
                self.tracks.append(track)

            except Exception as e:
                print(f"error on ensemble {rep}, skipping: {e!r}")
                traceback.print_exc()

        # Cache handles so denoise() can run without reloading.
        self._epi, self._gm, self._cf = epi, gm, cf
        self._obs_list = obs_list
        self._fitted_output_dir = output_dir
        return self

    def denoise(self, output_dir=None, verbose=True):
        """Average the trained ensemble and write the denoised output.

        Also writes a preprocessed (undenoised) copy and a CompCor-denoised
        comparison alongside it.

        Returns:
            DeepCorResult with the output paths and the per-repetition tracks.
        """
        output_dir = output_dir or self._fitted_output_dir
        if output_dir is None:
            raise RuntimeError(
                "No output_dir available; pass output_dir to fit() or denoise()."
            )
        if not self.signal_files:
            raise RuntimeError(
                "No per-repetition signals to ensemble; call fit(output_dir=...) "
                "first so signals are written to disk."
            )

        safe_mkdir(output_dir)
        result = DeepCorResult(
            output_dir=output_dir,
            signal_files=list(self.signal_files),
            tracks=list(self.tracks),
        )

        s = self.config.data.subject_idx
        r = self.config.data.run_idx

        if verbose:
            print("Averaging ensemble predictions...")
        result.denoised_path = os.path.join(
            output_dir, f"denoised_deepcor_S{s}_R{r}_avg.nii.gz"
        )
        average_signal_ensemble(self.signal_files, result.denoised_path)

        if verbose:
            print("Saving comparison outputs (preproc, CompCor)...")
        result.preproc_path = os.path.join(output_dir, f"input_data_S{s}_R{r}.nii.gz")
        array_to_brain(
            self._obs_list[:, 0, :],
            self._epi,
            self._gm,
            result.preproc_path,
            inv_z_score=True,
            return_img=False,
        )

        result.compcor_path = os.path.join(
            output_dir, f"denoised_compcor_S{s}_R{r}.nii.gz"
        )
        calc_and_save_compcor(
            self._epi, self._gm, self._cf,
            result.compcor_path, n_components=5, return_img=False,
        )

        if verbose:
            print(f"Denoising complete! Output: {result.denoised_path}")
        return result

    def fit_denoise(
        self,
        epi,
        gm_mask,
        cf_mask,
        confounds=None,
        output_dir=None,
        dashboard="save",
        on_epoch=None,
        subject_idx=None,
        run_idx=None,
        verbose=True,
    ):
        """Convenience wrapper: ``fit(...)`` then ``denoise(...)``.

        Returns:
            DeepCorResult.
        """
        self.fit(
            epi, gm_mask, cf_mask,
            confounds=confounds,
            output_dir=output_dir,
            dashboard=dashboard,
            on_epoch=on_epoch,
            subject_idx=subject_idx,
            run_idx=run_idx,
            verbose=verbose,
        )
        return self.denoise(output_dir=output_dir, verbose=verbose)

    def save(self, path):
        """Save trained models."""
        safe_mkdir(path)
        for i, trainer in enumerate(self.trainers):
            trainer.save_checkpoint(os.path.join(path, f"model_{i}.pt"), 0, 0)


# Stable, friendlier alias for the high-level entry point.
DeepCor = DeepCorDenoiser
