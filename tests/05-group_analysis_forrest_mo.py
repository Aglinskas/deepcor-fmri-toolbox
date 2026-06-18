import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="DeepCor-group-Forrest")


@app.cell
def _():
    import os
    import ants
    import numpy as np
    from matplotlib import pyplot as plt
    import marimo as mo

    return ants, mo, np, os, plt


@app.cell
def _(mo, os):
    os.chdir(mo.notebook_dir())
    return


@app.cell
def _(np, os):
    np.array(os.listdir('../Data/DeepCor-Outputs/'))
    return


@app.cell
def _():
    analysis_dir = '../Data/DeepCor-Outputs/deepcor-v1-forrest-100-20'
    folder_fn = 'DeepCor-Forrest-S{s}-R{r}-cvae_v1'
    return analysis_dir, folder_fn


@app.cell
def _():
    nsubs = 14
    nruns = 4

    ffa_roi_fn = '../Data/study-forrest-ROIs/rFFA_final_mask_{sub}_bin.nii.gz'
    ppa_roi_fn = '../Data/study-forrest-ROIs/rPPA.nii.gz_final_mask_{sub}_bin.nii.gz'
    return ffa_roi_fn, nsubs, ppa_roi_fn


@app.cell
def _():
    subs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-09','sub-10', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18','sub-19', 'sub-20']
    return (subs,)


@app.cell
def _(ants, np, os):
    s = 0
    r = 1

    #im_fn = os.path.join(analysis_dir,folder_fn.format(s=s,r=r),f'contrast_face_{s}_R{r}_compcor.nii.gz')
    #roi_fn = ffa_roi_fn.format(sub=subs[s])

    def get_roi_data(im_fn,roi_fn):
        assert os.path.exists(im_fn), f'bad im path: {im_fn}'
        assert os.path.exists(im_fn), f'bad roi path: {roi_fn}'

        im = ants.image_read(im_fn)
        roi = ants.image_read(roi_fn)

        assert all((roi.numpy().max()==1,roi.numpy().min()==0)), f'not a mask image: {(roi.numpy().max()==1,roi.numpy().min()==0)}'
        assert len(np.unique(roi.numpy().flatten()))==2, f'non binarized mask, uvals: {np.unique(roi.numpy().flatten())}'

        roi_vals = im.numpy()[roi.numpy()==1]
        #roi_vals=roi_vals[roi_vals!=0] # Drop voxels that are exactly zero, cause their artefactual (mask mismatch probably)
        return roi_vals



    return (get_roi_data,)


@app.cell
def _(analysis_dir, ffa_roi_fn, folder_fn, get_roi_data, os, ppa_roi_fn, subs):
    def get_forrest_roi_data(s,r,task='face',metric='correlation',kind='preproc'):
        # task='place'
        # metric='contrast'
        # kind='deepcor'

        if all((task=='face',metric=='correlation')):
            im_fn = os.path.join(analysis_dir,folder_fn.format(s=s,r=r),f'corr2face_S{s}_R{r}_{kind}.nii.gz')
            roi_fn = ffa_roi_fn.format(sub=subs[s])
        elif all((task=='place',metric=='correlation')):
            im_fn = os.path.join(analysis_dir,folder_fn.format(s=s,r=r),f'corr2place_S{s}_R{r}_{kind}.nii.gz')
            roi_fn = ppa_roi_fn.format(sub=subs[s])
        elif all((task=='face',metric=='contrast')):
            im_fn = os.path.join(analysis_dir,folder_fn.format(s=s,r=r),f'contrast_face_S{s}_R{r}_{kind}.nii.gz')
            roi_fn = ffa_roi_fn.format(sub=subs[s])
        elif all((task=='place',metric=='contrast')):
            im_fn = os.path.join(analysis_dir,folder_fn.format(s=s,r=r),f'contrast_place_S{s}_R{r}_{kind}.nii.gz')
            roi_fn = ppa_roi_fn.format(sub=subs[s])
        else:
            raise Exception(f'Task should be one of [face, place] and metric one of [correlation,contrast], kind one of [preproc, compcor, deepcor] got task={task}, metric={metric}, kind={kind}')

        try:
            roi_vals = get_roi_data(im_fn,roi_fn).mean()
        except:
            #roi_vals=np.nan
            roi_vals=0 # Dev

        return roi_vals

    return (get_forrest_roi_data,)


@app.cell
def _(get_forrest_roi_data, np, nsubs):
    res_cor_face_preproc = np.array([[get_forrest_roi_data(s,r,task='face',metric='correlation',kind='preproc') for r in [1,2,3,4]] for s in range(nsubs)])
    res_cor_face_compcor = np.array([[get_forrest_roi_data(s,r,task='face',metric='correlation',kind='compcor') for r in [1,2,3,4]] for s in range(nsubs)])
    res_cor_face_deepcor = np.array([[get_forrest_roi_data(s,r,task='face',metric='correlation',kind='deepcor') for r in [1,2,3,4]] for s in range(nsubs)])

    res_cor_place_preproc = np.array([[get_forrest_roi_data(s,r,task='place',metric='correlation',kind='preproc') for r in [1,2,3,4]] for s in range(nsubs)])
    res_cor_place_compcor = np.array([[get_forrest_roi_data(s,r,task='place',metric='correlation',kind='compcor') for r in [1,2,3,4]] for s in range(nsubs)])
    res_cor_place_deepcor = np.array([[get_forrest_roi_data(s,r,task='place',metric='correlation',kind='deepcor') for r in [1,2,3,4]] for s in range(nsubs)])

    res_con_face_preproc = np.array([[get_forrest_roi_data(s,r,task='face',metric='contrast',kind='preproc') for r in [1,2,3,4]] for s in range(nsubs)])
    res_con_face_compcor = np.array([[get_forrest_roi_data(s,r,task='face',metric='contrast',kind='compcor') for r in [1,2,3,4]] for s in range(nsubs)])
    res_con_face_deepcor = np.array([[get_forrest_roi_data(s,r,task='face',metric='contrast',kind='deepcor') for r in [1,2,3,4]] for s in range(nsubs)])

    res_con_place_preproc = np.array([[get_forrest_roi_data(s,r,task='place',metric='contrast',kind='preproc') for r in [1,2,3,4]] for s in range(nsubs)])
    res_con_place_compcor = np.array([[get_forrest_roi_data(s,r,task='place',metric='contrast',kind='compcor') for r in [1,2,3,4]] for s in range(nsubs)])
    res_con_place_deepcor = np.array([[get_forrest_roi_data(s,r,task='place',metric='contrast',kind='deepcor') for r in [1,2,3,4]] for s in range(nsubs)])

    res_cor_face_deepcor
    return (
        res_con_face_compcor,
        res_con_face_deepcor,
        res_con_face_preproc,
        res_con_place_compcor,
        res_con_place_deepcor,
        res_con_place_preproc,
        res_cor_face_compcor,
        res_cor_face_deepcor,
        res_cor_face_preproc,
        res_cor_place_compcor,
        res_cor_place_deepcor,
        res_cor_place_preproc,
    )


@app.cell
def _():
    return


@app.function
def pretty_ttest_rel(v1,v2):
    from scipy.stats import ttest_rel
    t,p = ttest_rel(v1,v2)
    degf = len(v1)-1
    d = v1.mean()-v2.mean()

    if p<.001:
        t_statement = f'Δ={d:.2f},t({degf})={t:.2f},p<.001'
    else:
        t_statement = f'Δ={d:.2f},t({degf})={t:.2f},p={p:.3f}'
    return t_statement


@app.cell
def _():
    return


@app.cell
def _(np, plt):
    def plot_bars_ttests(vec,ttl='',dot_size=8,dot_color='red'):
        xs = [0,1,2]
        ys = [v.mean() for v in vec]
        se = [v.std()/np.sqrt(len(v)) for v in vec]

        plt.bar(xs,ys)
        plt.errorbar(xs,ys,se,fmt='r ')
        plt.xticks(xs,[f'NoDenoise\n{ys[0]:.2f}',f'CompCor\n{ys[1]:.2f}',f'DeepCor\n{ys[2]:.2f}'])

        # individual datapoints (jittered dots)
        for x, v in zip(xs, vec):
            jitter = (np.random.rand(len(v)) - 0.5) * 0.25
            plt.scatter(x + jitter, v, s=dot_size, c=dot_color, alpha=0.5, zorder=3)

        # paired t-test significance bars
        def add_sig_bar(i, j, level):
            x1, x2 = xs[i], xs[j]
            top = max(ys[i] + se[i], ys[j] + se[j])
            y = max(ys) + max(se) + level * 0.16 * max(ys)
            h = 0.015 * max(ys)
            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c='k')
            plt.text((x1 + x2) / 2, y + h + .001, pretty_ttest_rel(vec[j], vec[i]),
                     ha='center', va='bottom', fontsize=8)

        add_sig_bar(0, 1, 1)
        add_sig_bar(1, 2, 2)
        add_sig_bar(0, 2, 3)

        plt.title(ttl)

        #plt.show()
    return (plot_bars_ttests,)


@app.cell
def _(
    plot_bars_ttests,
    plt,
    res_con_face_compcor,
    res_con_face_deepcor,
    res_con_face_preproc,
    res_con_place_compcor,
    res_con_place_deepcor,
    res_con_place_preproc,
    res_cor_face_compcor,
    res_cor_face_deepcor,
    res_cor_face_preproc,
    res_cor_place_compcor,
    res_cor_place_deepcor,
    res_cor_place_preproc,
):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_bars_ttests([res_cor_face_preproc.mean(axis=1),res_cor_face_compcor.mean(axis=1),res_cor_face_deepcor.mean(axis=1)],ttl='Correlation w/ Regressor\nFace / FFA')

    plt.subplot(1,2,2)
    plot_bars_ttests([res_cor_place_preproc.mean(axis=1),res_cor_place_compcor.mean(axis=1),res_cor_place_deepcor.mean(axis=1)],ttl='Correlation w/ Regressor\nPlace / PPA')

    plt.show()

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_bars_ttests([res_con_face_preproc.mean(axis=1),res_con_face_compcor.mean(axis=1),res_con_face_deepcor.mean(axis=1)],ttl='Contrast Estimate\nFace / FFA')

    plt.subplot(1,2,2)
    plot_bars_ttests([res_con_place_preproc.mean(axis=1),res_con_place_compcor.mean(axis=1),res_con_place_deepcor.mean(axis=1)],ttl='Contrast Estimate\nPlace / PPA')

    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(plt):
    def plot_scatter(xvals,yvals,xlabel='',ylabel='',ttl=''):
        v1=xvals.flatten()
        v2=yvals.flatten()

        nsubs_, nruns_ = xvals.shape
        labels = [f'S{s+1}R{r+1}' for s in range(nsubs_) for r in range(nruns_)]

        for x, y, lab in zip(v1, v2, labels):
            plt.text(x, y, lab, ha='center', va='center', fontsize=7)

        # x=y reference line
        lo = min(v1.min(), v2.min())
        hi = max(v1.max(), v2.max())
        pad = (hi - lo) * 0.05
        lims = [lo - pad, hi + pad]
        plt.plot(lims, lims, 'k--', lw=1, alpha=0.6, zorder=0)
        plt.xlim(lims); plt.ylim(lims)
        plt.gca().set_aspect('equal')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(ttl)

    return (plot_scatter,)


@app.cell
def _(
    plot_scatter,
    plt,
    res_con_face_compcor,
    res_con_face_deepcor,
    res_con_face_preproc,
    res_con_place_compcor,
    res_con_place_deepcor,
    res_con_place_preproc,
    res_cor_face_compcor,
    res_cor_face_deepcor,
    res_cor_face_preproc,
    res_cor_place_compcor,
    res_cor_place_deepcor,
    res_cor_place_preproc,
):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_scatter(res_cor_face_preproc,res_cor_face_compcor,'preproc','compcor','correlation / face')
    plt.subplot(1,2,2)
    plot_scatter(res_cor_face_compcor,res_cor_face_deepcor,'compcor','deepcor','correlation / face')
    plt.show()


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_scatter(res_cor_place_preproc,res_cor_place_compcor,'preproc','compcor','correlation / place')
    plt.subplot(1,2,2)
    plot_scatter(res_cor_place_compcor,res_cor_place_deepcor,'compcor','deepcor','correlation / place')
    plt.show()


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_scatter(res_con_face_preproc,res_con_face_compcor,'preproc','compcor','contrast / face')
    plt.subplot(1,2,2)
    plot_scatter(res_con_face_compcor,res_con_face_deepcor,'compcor','deepcor','contrast / face')
    plt.show()


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot_scatter(res_con_place_preproc,res_con_place_compcor,'preproc','compcor','contrast / place')
    plt.subplot(1,2,2)
    plot_scatter(res_con_place_compcor,res_con_place_deepcor,'compcor','deepcor','contrast / place')
    plt.show()
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


if __name__ == "__main__":
    app.run()
