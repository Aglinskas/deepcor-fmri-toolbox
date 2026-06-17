# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.21.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="full",
    app_title="deepcor-forrest-simple-v1",
    auto_download=["ipynb", "html"],
)


@app.cell
def _():
    import os
    import marimo as mo
    import deepcor

    os.chdir(mo.notebook_dir())  # all paths relative to the notebook
    return deepcor, mo, os


@app.cell
def _(mo):
    mo.md("""
    # DeepCor — simple training (CVAE v1)

    Same high-level `deepcor.DeepCor` API as the v2 notebook — the **only**
    difference is `model_version="v1"`. The original CVAE has no confound
    conditioning, so `confounds` is optional and ignored here.
    """)
    return


@app.cell
def _(deepcor):
    # GPU check (optional)
    deepcor.utils.check_gpu_and_speedup(tensor_size=(1000, 1000), n_iter=100)
    return


@app.cell
def _(os):
    # ---- Data paths (the only thing you normally edit) ----
    bids_path = "../Data/fMRI-Data/studyforrest-fmriprep/"

    subs = sorted(d for d in os.listdir(bids_path) if d.startswith("sub-"))
    s, r = 0, 4  # subject index, run
    sub_id, run = subs[s], str(r)

    session, task = "ses-localizer", "objectcategories"
    space = "MNI152NLin2009cAsym"
    base = os.path.join(bids_path, sub_id, session, "func")

    epi_path = os.path.join(
        base, f"{sub_id}_{session}_task-{task}_run-{run}_bold_space-{space}_preproc.nii.gz"
    )
    gm_mask_path = os.path.join(bids_path, "mask_roi.nii")
    cf_mask_path = os.path.join(bids_path, "mask_roni.nii")

    output_dir = os.path.join(
        "../Data/DeepCor-Outputs", "forrest-simple-v1", f"S{s}-R{r}-cvae_v1"
    )

    for p in (epi_path, gm_mask_path, cf_mask_path):
        assert os.path.exists(p), f"missing: {p}"
    print("EPI:", epi_path)
    print("output_dir:", output_dir)
    return cf_mask_path, epi_path, gm_mask_path, output_dir, r, s


@app.cell
def _(cf_mask_path, deepcor, epi_path, gm_mask_path, output_dir, r, s):
    # ---- Train + denoise ----
    # Config is optional; omit it for sensible defaults. v1 ignores confounds.
    #
    # No `dashboard`/`on_epoch` => nothing is plotted interactively; the training
    # dashboard is still saved to output_dir every epoch. A per-epoch progress
    # line (subject/run, started, elapsed, ETA) is printed below.
    denoiser = deepcor.DeepCor(model_version="v1")
    denoiser.config.training.n_epochs = 5
    denoiser.config.training.n_repetitions = 5

    result = denoiser.fit_denoise(
        epi_path,
        gm_mask_path,
        cf_mask_path,
        confounds=None,      # v1 has no confound conditioning
        output_dir=output_dir,
        subject_idx=s,
        run_idx=r,
    )
    return denoiser, result


@app.cell
def _(result):
    print("Denoised:", result.denoised_path)
    print("Preproc :", result.preproc_path)
    print("CompCor :", result.compcor_path)
    result.denoised_path
    return


@app.cell
def _(deepcor, os, r, result, s):
    # ---- Overlay dashboard: all repetitions layered onto one figure ----
    # fit_denoise saves one track per repetition as track_rep_{rep}.pickle in
    # output_dir. Reload them and overlay onto a single figure by threading the
    # same `fig` through each call, then save it next to the other outputs.
    # Depending on `result` ties this cell after training in marimo's DAG.
    import warnings
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")

    track_files = sorted(
        os.path.join(result.output_dir, f)
        for f in os.listdir(result.output_dir)
        if f.startswith("track_rep_") and f.endswith(".pickle")
    )
    tracks = [deepcor.data.load_pickle(track_file) for track_file in track_files]

    this_fig = None
    for this_track in tracks:
        try:
            this_fig = deepcor.visualization.show_dahsboard_v1_marimo(
                this_track, fig=this_fig, save_fig=False
            )
        except Exception as e:
            print(f"bad track: {e}")
    if this_fig is not None:
        this_fig.savefig(
            os.path.join(result.output_dir, f"dashboard_S{s}_R{r}.png"),
            dpi=100,
            bbox_inches="tight",
        )
    plt.show()
    return


if __name__ == "__main__":
    app.run()
