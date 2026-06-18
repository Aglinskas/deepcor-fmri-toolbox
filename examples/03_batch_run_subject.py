"""
Per-subject DeepCor runner — invoked by 03_slurm_batch_job.sh.

This script denoises ONE subject, chosen by an index passed on the command line.
A SLURM array job calls it once per subject, so the whole study runs in parallel:

    python 03_batch_run_subject.py <subject_index> [analysis_name]

You normally don't run this by hand — submit 03_slurm_batch_job.sh instead.
See wiki/batch_jobs.md.
"""

import os
import sys
from deepcor import DeepCor

# ===========================================================================
# SUBJECTS & PATHS  — EDIT THESE for your dataset.
# ===========================================================================
# List your subject IDs here, in order. The SLURM array index selects one of
# them (index 0 -> SUBJECTS[0], etc.). Edit to match your subjects.
SUBJECTS = ["sub-01", "sub-02", "sub-03"]   # ... add the rest

# Where your preprocessed data lives. Edit the templates so they resolve to real
# files for each subject. {sub} is replaced with the subject ID.
DATA_DIR = "/path/to/derivatives"
EPI_TEMPLATE       = "{data}/{sub}/func/{sub}_task-rest_desc-preproc_bold.nii.gz"
GM_MASK_TEMPLATE   = "{data}/{sub}/anat/{sub}_label-GM_mask.nii.gz"          # ROI
CF_MASK_TEMPLATE   = "{data}/{sub}/anat/{sub}_label-WMCSF_mask.nii.gz"      # RONI: WM + CSF
CONFOUNDS_TEMPLATE = "{data}/{sub}/func/{sub}_task-rest_desc-confounds_timeseries.tsv"

# Root output directory; per-subject results go in a subfolder under here.
OUTPUT_ROOT = "./deepcor-output"

# ===========================================================================
# HYPERPARAMETERS
# ===========================================================================
# Set LOW for a quick test of the whole batch pipeline. For a REAL run, use
# roughly N_EPOCHS=100, N_REPETITIONS=20.
# (These can also be overridden from the SLURM script via the environment
# variables DEEPCOR_EPOCHS / DEEPCOR_REPS — see 03_slurm_batch_job.sh.)
N_EPOCHS      = int(os.environ.get("DEEPCOR_EPOCHS", 5))
N_REPETITIONS = int(os.environ.get("DEEPCOR_REPS", 5))


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python 03_batch_run_subject.py <subject_index> [analysis_name]")

    idx = int(sys.argv[1])
    analysis_name = sys.argv[2] if len(sys.argv) > 2 else "deepcor-run"

    if idx >= len(SUBJECTS):
        sys.exit(f"Subject index {idx} out of range (only {len(SUBJECTS)} subjects).")
    sub = SUBJECTS[idx]

    # Build this subject's file paths from the templates.
    epi       = EPI_TEMPLATE.format(data=DATA_DIR, sub=sub)
    gm_mask   = GM_MASK_TEMPLATE.format(data=DATA_DIR, sub=sub)
    cf_mask   = CF_MASK_TEMPLATE.format(data=DATA_DIR, sub=sub)
    confounds = CONFOUNDS_TEMPLATE.format(data=DATA_DIR, sub=sub)
    output_dir = os.path.join(OUTPUT_ROOT, analysis_name, sub)

    for label, path in [("EPI", epi), ("GM mask", gm_mask),
                        ("CF mask", cf_mask), ("confounds", confounds)]:
        if not os.path.exists(path):
            sys.exit(f"{label} not found for {sub}: {path}")

    print(f"=== DeepCor: {sub} (index {idx}) | analysis '{analysis_name}' ===")
    print(f"epochs={N_EPOCHS}  repetitions={N_REPETITIONS}")
    print(f"output_dir={output_dir}")

    denoiser = DeepCor(
        model_version="latest",
        n_epochs=N_EPOCHS,
        n_repetitions=N_REPETITIONS,
    )
    result = denoiser.fit_denoise(
        epi=epi,
        gm_mask=gm_mask,
        cf_mask=cf_mask,
        confounds=confounds,
        output_dir=output_dir,
        subject_idx=idx,
        dashboard="save",   # save a dashboard PNG as a per-subject record
        verbose=True,
    )
    print(f"\n{sub} done. Denoised image: {result.denoised_path}")


if __name__ == "__main__":
    main()
