# Glossary

[← Back to the wiki home](README.md)

Short definitions of the terms used across this wiki and the DeepCor codebase.

### aCompCor
Anatomical CompCor: a denoising method that estimates nuisance signals from
anatomical noise regions (white matter and CSF). DeepCor's confounds mask
(`cf_mask`) uses the same kind of noise region. See Behzadi et al. (2007),
https://pmc.ncbi.nlm.nih.gov/articles/PMC2214855/.

### BOLD
Blood-oxygen-level-dependent signal — the contrast fMRI measures as a proxy for
neural activity.

### β (beta)
The weight on the KL-divergence term in the VAE loss (`ModelConfig.beta`).
Higher β = stronger regularization of the latent space. See [CVAEs](CVAEs.md).

### Background (BG)
In the model, the reconstruction built from the **noise latent only** (signal
latent zeroed) — i.e. the estimated noise component. Code: `forward_bg`.

### CompCor
Component-based noise correction: removes principal components estimated from
noise regions. DeepCor writes a CompCor-denoised image as a comparison baseline
(see [outputs](outputs.md)).

### Confounds
Nuisance regressors — here, the **six rigid-body motion parameters** read from an
fMRIPrep confounds TSV. Required by the `v2` model. See
[data preparation](data_preparation.md).

### Confounds mask (`cf_mask`)
The **RONI** — a mask of regions where no signal of interest is expected,
usually **white matter + CSF**. `cf` stands for *confounds*.

### Contrastive VAE (CVAE / cVAE)
A VAE trained on a "target" set (signal + noise) and a "background" set (noise
only), with separate signal and noise latent spaces so it can isolate the
signal. The core of DeepCor. See [CVAEs](CVAEs.md).

### Disentanglement
Encouraging the signal and noise latent spaces to encode independent, non-
overlapping information. Controlled by `gamma`, `delta`, and `do_disentangle`.

### Dummy scans
The first few volumes of a run, often discarded to let the MR signal reach
steady state. Configurable via `DataConfig.n_dummy_scans`.

### Ensemble
The set of independently-trained models (`n_repetitions`) whose denoised outputs
are averaged for a more stable result. See [configuration](configuration.md).

### EPI
Echo-planar imaging — the fast acquisition used for functional (BOLD) runs. In
DeepCor, the `epi` input is the 4D functional image to denoise.

### Epoch
One full pass over the training data. Set per model with
`TrainingConfig.n_epochs`.

### fMRIPrep
A standard fMRI preprocessing pipeline. DeepCor's expected inputs (preprocessed
BOLD, confounds TSV, tissue masks) map directly onto fMRIPrep derivatives. See
[data preparation](data_preparation.md).

### Foreground (FG)
The reconstruction built from the **signal latent only** (noise latent zeroed) —
i.e. the **denoised** signal. This is DeepCor's output. Code: `forward_fg`.

### Frame censoring
Flagging and interpolating high-motion timepoints. Helper functions exist
(`apply_frame_censoring`, `censor_and_interpolate`) but are **not** auto-applied
by the high-level pipeline — see [configuration](configuration.md).

### KL divergence
A measure of how far the learned latent distribution is from the prior; the
regularizing term in the VAE loss, weighted by β.

### Latent space
The compressed, low-dimensional representation a VAE encodes data into. DeepCor
uses two: a **signal latent** (`s`) and a **noise latent** (`z`). Sizes set by
`latent_dims`.

### Noise latent (`z`)
The shared latent meant to capture structured noise present in both the ROI and
RONI.

### Repetition
One member of the ensemble (one trained model). Count set by
`TrainingConfig.n_repetitions`. Note: in the batch scripts, the notebook's `r`
parameter is a separate "run/repeat of the whole analysis" label used in output
filenames — not the same as `n_repetitions`.

### ROI (Region Of Interest)
The **gray-matter mask** (`gm_mask`) — where the signal of interest lives. The
model's "target". Also called the observation (`obs`) set in code.

### RONI (Region Of No Interest)
The **confounds mask** (`cf_mask`) — noise regions (WM + CSF) used to estimate
structured noise. The model's "background". Also called the noise (`noi`) set in
code.

### Signal latent (`s`)
The salient latent meant to capture the signal of interest, present in the ROI
but not the RONI.

### Target (TG)
In the model, the **full** reconstruction of the ROI data (= foreground +
background). Code: `forward_tg`.

### Total correlation
A measure of statistical dependence among latent variables; penalizing it
encourages disentanglement. Weighted by `gamma` (v2).

### Track
A dictionary recording training state and metrics across epochs, consumed by the
training dashboard and saved as `track_rep_*.pickle`. See [outputs](outputs.md).

### VAE
Variational autoencoder — the encoder/decoder generative model DeepCor builds
on. See [CVAEs](CVAEs.md).

---

**See also:** [What CVAEs are and how DeepCor works](CVAEs.md) ·
[Data preparation](data_preparation.md) · [Configuration reference](configuration.md)
