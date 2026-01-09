def plot_timeseries(epi,gm,cf):
    plt.figure(figsize=(20,5))
    plt.plot(epi.numpy()[gm.numpy()==1].mean(axis=0))
    plt.title('EPI ROI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')

    plt.figure(figsize=(20,5))
    plt.plot(epi.numpy()[cf.numpy()==1].mean(axis=0))
    plt.title('EPI RONI timeseries')
    plt.ylabel('BOLD')
    plt.xlabel('timepoints')