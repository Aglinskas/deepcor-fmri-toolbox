# Backwards Compatibility Guide: CVAE Model Versions

> **Audience:** AI agents and developers modifying the `deepcor` toolbox.
> **Goal:** Write code that works across CVAE model versions (V1, V2, … and
> future V3+) without breaking existing notebooks and pipelines.
>
> Read this **before** editing anything under `deepcor/models/`,
> `deepcor/visualization/`, `deepcor/data/`, or `deepcor/training/`.

---

## 1. The core principle

There is **more than one CVAE model**, and there **will be more in the future**.
Any code you write, fix, or refactor must assume that:

1. **Old versions keep working.** Existing notebooks pin specific versions
   (e.g. `02_StudyForrest-advanced-v1_mo.py` uses `CVAE_V1`,
   `02_StudyForrest-advanced-v2_mo.py` uses `CVAE`). A change made for the
   newest model must **not** alter the behavior of older ones.
2. **New versions are coming.** A `CVAE_V3` (or a different track schema, or a
   new dashboard) will be added later. Structure code so that adding a version
   is a *small, local, additive* change — never a rewrite of shared logic.

When you fix a bug or add a feature, you have exactly **three** acceptable
shapes for the solution. Pick whichever is least invasive:

| Strategy | Use when | Example |
|----------|----------|---------|
| **(A) Version-agnostic** | The logic is genuinely identical across versions | A utility that operates on the returned loss dict's common keys |
| **(B) Single function, version parameter, defaults to latest** | Behavior differs slightly and can be branched cleanly | `init_track(model_version='V1')` |
| **(C) Separate versioned functions + a latest-defaulting dispatcher** | Behavior differs substantially | `show_dahsboard_v1_marimo`, `show_dahsboard_v2_marimo`, with `show_dahsboard_marimo` dispatching to the latest |

**Rule of thumb:** if a caller does not specify a version, the code should
behave as the **latest** version — never crash, never silently no-op.

---

## 2. The version map (current state)

### Models — `deepcor/models/`

| File | Class | Registry keys | Role |
|------|-------|---------------|------|
| `cvae_v1.py` | `CVAE_V1` (alias `cVAE_V1`) | `"v1"` | Original CVAE. No confound decoder. |
| `cvae.py` | `CVAE` (alias `cVAE`) | `"v2"`, `"cvae"`, `"latest"` | Current recommended model. Confound-aware. |

`deepcor/models/registry.py` is the **single source of truth** for "what is
latest":

```python
MODEL_REGISTRY = {
    "v1": CVAE_V1,
    "v2": CVAE,
    "cvae": CVAE,     # stable alias → latest recommended
    "latest": CVAE,   # stable alias → latest recommended
}
```

> **When you add `CVAE_V3`:** create `cvae_v3.py`, add it to
> `models/__init__.py`, add `"v3": CVAE_V3` to the registry, and **repoint
> `"cvae"` and `"latest"` to `CVAE_V3`**. Do not edit `cvae.py` or `cvae_v1.py`
> to "upgrade in place" — old notebooks import those by name.

Prefer `deepcor.models.get_model("latest", **kwargs)` over hard-coding a class
when you want "whatever is newest." Hard-code the specific class
(`CVAE`, `CVAE_V1`) only when you genuinely need that exact version.

---

## 3. How V1 and V2 actually differ

Knowing the concrete differences is what lets you write safe cross-version code.

### 3.1 Constructor signatures differ

```python
# V1 — no confounds, scalar latent dim
CVAE_V1(in_channels=1, in_dim=nTR, latent_dim=8)

# V2 — requires confounds tensor, TUPLE latent dim (signal_dim, noise_dim)
CVAE(conf=torch.tensor(conf), in_channels=4, in_dim=nTR, latent_dim=(8, 8))
```

- `latent_dim` is an **int** in V1 and a **`(z, s)` tuple** in V2. Never assume
  one or the other in shared code.
- V2 takes a required `conf` (confounds) argument; V1 does not.

### 3.2 Input data shape differs (and so does `in_channels`)

The two notebooks use **different data loaders**, which is the root of the
`in_channels` difference:

| Notebook | Loader | `obs_list` shape | `in_channels` |
|----------|--------|------------------|---------------|
| V1 | `deepcor.data.get_obs_noi_list(epi, gm, cf)` | `(N, nTR)` → single channel | `1` |
| V2 | `deepcor.data.get_obs_noi_list_coords(epi, gm, cf)` | `(N, 4, nTR)` | `4` |

In the V2 (`_coords`) loader the channel axis packs
**`[bold_signal, x, y, z]`**: channel 0 is the BOLD timeseries, channels 1–3
are the voxel's spatial coordinates broadcast across time. They serve as
**conditioning channels**.

**Therefore `in_channels` must match the data, not be hard-coded.** Use:

```python
in_channels = obs_list.shape[-2]   # data-adaptive: 1 for V1 loader, 4 for V2 loader
```

The loss and dashboard only reconstruct/score channel 0 (`x[:, 0, :]`), so the
coordinate channels are inputs only. If a future loader adds more conditioning
channels, `obs_list.shape[-2]` keeps working unchanged.

### 3.3 The shared training contract (DO NOT break)

Despite their differences, **both** models expose the same methods so the
trainer and dashboard can be version-agnostic. Keep this contract stable in
every new version:

**Forward methods** (used by `Trainer` and `update_track`):
- `model.forward_tg(x)` → `[recon, input, mu_z, log_var_z, mu_s, log_var_s, z, s]`
- `model.forward_bg(x)` → `[recon, input, mu_z, log_var_z]`
- `model.forward_fg(x)` → `[recon, ...]` (denoised/foreground)

**`loss_function(*args)`** takes 12 positional args in this exact order:

```
recons_tg, input_tg,
tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s, tg_z, tg_s,
recons_bg, input_bg, bg_mu_z, bg_log_var_z
```

**`loss_function` must return a dict containing at least these keys**, because
the trainer and dashboard read them by name:

| Key | Read by | Notes |
|-----|---------|-------|
| `'loss'` | `Trainer.train_epoch` (backprop), dashboard | The only key used for `.backward()`. **Not detached.** |
| `'Reconstruction_Loss'` | `update_track`, `show_dahsboard_v1_marimo` | Logging only. **Detach it** (`.detach()`). |
| `'kld_loss'` | `update_track`, dashboard | Logging only. Detach it. |

> V2 originally **omitted** `'Reconstruction_Loss'`, which crashed the shared
> dashboard (`KeyError`). The fix added it as
> `recons_loss.detach()` (= `recons_loss_roi + recons_loss_roni`). **Any new
> version's `loss_function` must return these three keys.** Extra version-specific
> keys are fine and ignored by version-agnostic consumers.

---

## 4. The visualization / tracking layer

`deepcor/visualization/dashboard.py` is **shared** by all model versions, so it
is the place most likely to break compatibility. It has three version-aware
pieces:

- `init_track(model_version='V1', keys=None)` — builds the per-run tracking
  dict. The `model_version` string is stored in `track['model_version']` and
  drives all downstream dispatch.
- `update_track(track, train_loader, model)` — branches on
  `track['model_version']` to compute and append metrics each epoch.
- `show_dahsboard_v{N}_marimo(track)` — version-specific figure renderer.

### The "track schema version" is NOT the same as the model version

Both notebooks currently call `init_track('V1')` and
`show_dahsboard_v1_marimo`, **even the V2 notebook**. That is intentional: the
*tracked quantities* (loss curves, varexp, in/out correlations, latent
histograms) are identical, so V2 reuses the V1 **track schema**. The `'V1'`
here labels the *dashboard/track layout*, not the model.

Only introduce a `'V2'` track schema when you actually need to track
**different quantities** (e.g. confound-prediction metrics specific to V2).
Until then, reusing `'V1'` is correct and not a bug.

### Patterns to follow in this layer

**`init_track` — default to latest, fail loudly on unknown versions.**
The current code has an `else: pass` branch that leaves `keys = None` and then
crashes on iteration. That is the anti-pattern. Instead, drive it from a
version→keys table and resolve an unspecified version to the latest:

```python
# Sketch of the target shape (mirror registry.py's dict style)
_TRACK_KEYS = {
    "V1": ["loss", "Reconstruction_Loss", "kld_loss",
           "varexp_gm", "varexp_cf", "in_out_corr_gm", "in_out_corr_cf",
           "varexp_gm_fg", "varexp_cf_fg", "in_out_corr_gm_fg", "in_out_corr_cf_fg"],
    # "V2": [...]  # add when a distinct schema is needed
}
LATEST_TRACK_VERSION = "V1"

def init_track(model_version=None, keys=None):
    version = (model_version or LATEST_TRACK_VERSION)
    if keys is None:
        if version not in _TRACK_KEYS:
            raise ValueError(f"Unknown track version {version!r}; "
                             f"available: {sorted(_TRACK_KEYS)}")
        keys = _TRACK_KEYS[version]
    ...
```

**Dashboard renderers — versioned files + a latest-defaulting dispatcher.**
Keep `show_dahsboard_v1_marimo`, `show_dahsboard_v2_marimo`, … as separate
functions. Add a single public entry point that dispatches on the track's
version and defaults to the latest:

```python
_DASHBOARD_RENDERERS = {"V1": show_dahsboard_v1_marimo}  # add "V2": ... later
LATEST_DASHBOARD_VERSION = "V1"

def show_dahsboard_marimo(track):
    """Render the dashboard for track['model_version'] (latest by default)."""
    version = track.get("model_version", LATEST_DASHBOARD_VERSION)
    renderer = _DASHBOARD_RENDERERS.get(version)
    if renderer is None:
        raise ValueError(f"No dashboard renderer for {version!r}; "
                         f"available: {sorted(_DASHBOARD_RENDERERS)}")
    return renderer(track)
```

Existing calls to `show_dahsboard_v1_marimo(track)` keep working untouched;
new code can call `show_dahsboard_marimo(track)` and automatically get the
right (or latest) renderer.

**Whatever you add, export it** in `deepcor/visualization/__init__.py`'s
`__all__` so it is importable as `deepcor.visualization.<name>`.

---

## 5. PyTorch correctness rules (learned from real bugs)

These caused concrete failures while bringing up the V2 notebook. Apply them in
every model version.

### 5.1 Tensors that must move with the model → use `register_buffer`

A plain attribute like `self.confounds = conf.float()` stays on the CPU when the
user calls `model.to(device)`, then explodes with
*"Expected all tensors to be on the same device, but found cuda:0 and cpu"*
during the loss computation.

Register any non-parameter tensor the forward/loss path uses as a **buffer**:

```python
self.register_buffer("confounds", conf)   # moves with .to(), .cuda(), state_dict
```

### 5.2 Normalize input tensor ranks defensively

The V2 loss indexes confounds as `(batch, n_confounds, time)` and loops over
`self.confounds.shape[1]`. A 2D `(n_confounds, time)` tensor made `shape[1]`
mean *time*, producing `IndexError: too many indices`. Promote rank explicitly
instead of assuming the caller's shape:

```python
conf = conf.float()
if conf.dim() == 2:                 # (n_confounds, time) → (1, n_confounds, time)
    conf = conf.unsqueeze(0)
self.register_buffer("confounds", conf)
```

This is robust whether the caller passes 2D or already-batched 3D input.

### 5.3 Detach logging-only tensors

Anything returned purely for plotting/logging (`Reconstruction_Loss`,
`kld_loss`) should be `.detach()`ed in the loss dict. Only `'loss'` stays
attached to the autograd graph. This mirrors `CVAE_V1.loss_function`.

---

## 6. Checklist before you commit a change

- [ ] **Which versions does this touch?** If you edited `cvae.py`, you changed
      **V2 only**. If you edited a shared module (`dashboard.py`, `trainer.py`,
      `data/`, `registry.py`), you changed **all versions** — re-check V1.
- [ ] **Did I edit a versioned file in place to "upgrade" it?** Don't. Add a new
      versioned file/function and repoint `latest`.
- [ ] **Does an unspecified version still resolve to latest** (and not crash or
      no-op)?
- [ ] **Does my `loss_function` still return `loss`, `Reconstruction_Loss`,
      `kld_loss`?**
- [ ] **Do the forward methods (`forward_tg/bg/fg`) keep their return
      structure?**
- [ ] **New public functions exported in the relevant `__init__.py`'s
      `__all__`?**
- [ ] **Did I verify both the V1 and V2 notebooks still run** (or at least the
      one I touched, plus reason about the other)?
- [ ] **Tensors used in forward/loss registered as buffers** so `.to(device)`
      works?

---

## 7. Quick reference: who reads what

```
notebooks (v1, v2)
  └─ deepcor.models.{CVAE_V1 | CVAE}        # pinned per notebook
  └─ deepcor.data.{get_obs_noi_list | get_obs_noi_list_coords}
  └─ deepcor.training.Trainer
        └─ model.forward_tg/bg(x)
        └─ model.loss_function(...)  ──reads──> ['loss']            (backprop)
  └─ deepcor.visualization
        ├─ init_track(model_version) ──────────> track['model_version']
        ├─ update_track(track, loader, model)
        │     └─ model.forward_tg/fg/bg(x)
        │     └─ model.loss_function(...) ──reads──> ['loss','Reconstruction_Loss','kld_loss']
        └─ show_dahsboard_v{N}_marimo(track)  /  show_dahsboard_marimo(track) → latest
```

Keep the **arrows** stable across versions; vary the **implementations** behind
versioned classes/functions.
