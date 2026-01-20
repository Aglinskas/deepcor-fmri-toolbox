def show_bashboard_cvae_v1(single_fig=True):
    import sys
    from IPython import display
    nrows=5
    ncols=9
    sp=0
    
    if single_fig==True:
        plt.close()
        sys.stdout.flush()
        display.clear_output(wait=True);
        display.display(plt.gcf());
        plt.figure(figsize=(5*ncols,5*nrows))
    
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(running_loss_L);   plt.title('total loss: {:.2f}'.format(running_loss_L[-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(running_recons_L);   plt.title('Recon Loss: {:.2f}'.format(running_recons_L[-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['ffa_varexp']);   plt.title('FFA varexp: {:.2f}'.format(track['ffa_varexp'][-1]))
    sp+=1;plt.subplot(nrows,ncols,sp);plt.plot(track['c_io']);   plt.title('ffa_io: {:.2f}'.format(track['c_io'][-1]))
    
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(batch_in[0,0,:])
    plt.plot(batch_out[0,0,:])
    plt.plot(model.forward_bg(inputs_gm)[0].detach().cpu().numpy()[0,0,:],'r-')
    plt.plot(model.forward_fg(inputs_gm)[0].detach().cpu().numpy()[0,0,:],'g-')
    plt.title('batch timecourse (single voxel)')
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(ffa_list_coords.mean(axis=0)[0,:])
    plt.plot(recon.mean(axis=0))
    plt.title('FFA AVG')
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(ffa_list_coords.mean(axis=0)[0,:])
    plt.plot(signal.mean(axis=0),'g-')
    plt.plot(face_reg)
    plt.title('FFA SIGNAL')

    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(ffa_list_coords.mean(axis=0)[0,:])
    plt.plot(noise.mean(axis=0),'r-')
    plt.plot(face_reg)
    plt.title('FFA NOISE')

    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(inputs_cf.detach().cpu().numpy()[0,0,:])
    plt.plot(model.forward_bg(inputs_cf)[0].detach().cpu().numpy()[0,0,:])
    plt.plot(model.forward_fg(inputs_cf)[0].detach().cpu().numpy()[0,0,:])
    plt.title('CF batch voxel')
    
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['r_ffa_reg'],'k-')
    plt.plot(track['r_TG_reg'])
    plt.title('R TG-REG {}'.format(track['r_TG_reg'][-1]))
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['r_ffa_reg'],'k-')
    if single_fig==True:
        plt.plot(track['r_FG_reg'],'g-')
    else:
        plt.plot(track['r_FG_reg'])
        
    plt.title('R FG-REG {}'.format(track['r_FG_reg'][-1]))
    
    
    sp+=1;plt.subplot(nrows,ncols,sp);
    plt.plot(track['r_ffa_reg'],'k-')
    plt.plot(track['r_BG_reg'],'r-')
    plt.title('R BG-REG {}'.format(track['r_BG_reg'][-1]))


    #plt.suptitle(f'E:{epoch} T:{elapsed}',y=.91,fontsize=20)
    if single_fig==True:
        plt.suptitle(f'{sub}-R{r}-rep-{rep} E:{epoch}',y=.91,fontsize=20)
        plt.show()