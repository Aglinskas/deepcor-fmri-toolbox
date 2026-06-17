import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns", app_title="DeepCor-dev-rel-v1")


@app.cell
def _():
    import os
    import ants
    import numpy as np
    from matplotlib import pyplot as plt
    import marimo as mo

    return ants, mo, np, os, plt


@app.cell
def _():
    import deepcor

    return (deepcor,)


@app.cell
def _(mo, os):
    os.chdir(mo.notebook_dir())
    return


@app.cell
def _():
    analysis_dir = '../Data/DeepCor-Outputs/test-advanced'
    return


@app.cell
def _():
    return


@app.cell
def _():
    nsubs = 14
    nruns = 4

    folder_fn = 'DeepCor-Forrest-S{s}-R{r}-cvae_v2'

    ffa_roi_fn = '../Data/study-forrest-ROIs/rFFA_final_mask_{sub}_bin.nii.gz'
    ppa_roi_fn = '../Data/study-forrest-ROIs/rPPA.nii.gz_final_mask_{sub}_bin.nii.gz'
    return (ffa_roi_fn,)


@app.cell
def _():
    events_fn_temp = '../Data/study-forrest-events/{sub}_ses-localizer_task-objectcategories_run-{r}_events.tsv'
    return (events_fn_temp,)


@app.cell
def _():
    subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-09','sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20']
    return (subs,)


@app.cell
def _():
    signal_rep_fn = '../Data/DeepCor-Outputs/test-advanced/DeepCor-Forrest-S{s}-R{r}-cvae_v2/signal_S{s}_R{r}_rep_{rep}.nii.gz'
    return (signal_rep_fn,)


@app.cell
def _(ants, deepcor, events_fn_temp, signal_rep_fn, subs):
    epi = ants.image_read(signal_rep_fn.format(s=0,r=1,rep=0))
    X = deepcor.analysis.get_design_matrix(epi,events_fn_temp.format(sub=subs[0],r=1))
    face_reg = X['face'].values
    body_reg = X['body'].values
    return (face_reg,)


@app.cell
def _(ants, ffa_roi_fn, np, os, signal_rep_fn, subs):
    def get_ensemble_data(s,r,return_type='corr2mean'):
        #s = 8
        #r = 1
        sub = subs[s]

        #ims = [ants.image_read(signal_rep_fn.format(s=s,r=r,rep=rep)) for rep in range(5)]
        ims = []
        for rep in range(5):
            fn = signal_rep_fn.format(s=s,r=r,rep=rep)
            if os.path.exists(fn):
                ims.append(ants.image_read(fn))

        ffa_roi = ants.image_read(ffa_roi_fn.format(sub=sub))
    
        ffa_timecourses = np.array([im.numpy()[ffa_roi.numpy()==1].mean(axis=0) for im in ims])

        if return_type=='timecourse':
            return ffa_timecourses
    
        if return_type=='corr2mean':
            return np.array([np.corrcoef(ffa_timecourses.mean(axis=0),ffa_timecourses[i,:])[0,1] for i in range(ffa_timecourses.shape[0])]).mean()
        

    return (get_ensemble_data,)


@app.cell
def _(get_ensemble_data, np):
    corr2mean = np.zeros((14,4))
    for s in range(14):
        print(f'{s}/14')
        for r in range(4):
            corr2mean[s,r] = get_ensemble_data(s,r+1,return_type='corr2mean')
    return (corr2mean,)


@app.cell
def _(corr2mean):
    import seaborn as sns
    sns.heatmap(corr2mean,annot=True)
    return


@app.cell
def _(get_ensemble_data):
    ffa_timecourses = get_ensemble_data(12,2,return_type='timecourse')
    return (ffa_timecourses,)


@app.cell
def _(face_reg, ffa_timecourses, np):
    print(np.corrcoef(face_reg,ffa_timecourses.mean(axis=0))[0,1])
    print(np.corrcoef(face_reg,np.median(ffa_timecourses,axis=0))[0,1])
    return


@app.cell
def _(ffa_timecourses, np):
    [np.corrcoef(ffa_timecourses.mean(axis=0),ffa_timecourses[i,:])[0,1] for i in range(ffa_timecourses.shape[0])]
    return


@app.cell
def _(ffa_timecourses, np, plt):
    plt.plot(ffa_timecourses.transpose())
    plt.plot(ffa_timecourses.mean(axis=0),linewidth=3)
    plt.plot(np.median(ffa_timecourses,axis=0),linewidth=3)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(ffa_timecourses, np, plt):
    band_mode='±1 SD'
    mean_tc = ffa_timecourses.mean(axis=0)
    t = np.arange(ffa_timecourses.shape[1])
    plt.plot(t, mean_tc, lw=2, label='mean')
    if band_mode == 'min-max':
        lo, hi = ffa_timecourses.min(axis=0), ffa_timecourses.max(axis=0)
        band_label = 'min–max'
    else:
        sd = ffa_timecourses.std(axis=0)
        lo, hi = mean_tc - sd, mean_tc + sd
        band_label = '±1 SD'
    plt.fill_between(t, lo, hi, alpha=0.3, label=band_label)
    plt.legend()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
