import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal
from itertools import combinations_with_replacement
from numpy import savetxt
import math
from numpy import random
import sklearn.preprocessing  
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn import linear_model
from torch.autograd import Function

import ants # ANTSpy in the toolbox for manipulating MRI files 
from tqdm import tqdm # Easy progress bars
import seaborn as sns

import sys
from IPython import display
import pickle

from nilearn.glm.first_level import make_first_level_design_matrix

class cVAE(nn.Module):

    def __init__(self,conf,in_channels: int,in_dim: int, latent_dim: tuple,hidden_dims: List = None, beta : float = 1, gamma : float = 1, delta : float = 1, scale_MSE_GM : float = 1, scale_MSE_CF : float = 1, scale_MSE_FG : float = 1,do_disentangle = True, freq_exp : float = 1, freq_scale : float = 1) -> None:
        super(cVAE, self).__init__()

        self.latent_dim = latent_dim
        self.latent_dim_z = self.latent_dim[0]
        self.latent_dim_s = self.latent_dim[1]
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.scale_MSE_GM = scale_MSE_GM
        self.scale_MSE_CF = scale_MSE_CF
        self.do_disentangle = do_disentangle
        self.freq_exp = freq_exp
        self.freq_scale = freq_scale
        self.scale_MSE_FG = scale_MSE_FG
        self.confounds = conf.float()
        self.grl = GradientReversalLayer(lambda_=1.0)

        nTR = in_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.decoder_confounds_z = nn.Sequential(
        
                nn.ConvTranspose1d(in_channels=latent_dim[0], out_channels=128, 
                                   kernel_size=nTR, stride=1, bias=False),
            
                # First conv layer: convert 128 channels to 32 channels.
                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
            
                # Second conv layer: convert 32 channels to 16 channels.
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
            
                # Third conv layer: convert 16 channels to 6 channels.
                nn.Conv1d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()  # Normalize the output to [0, 1]
                    )


        self.decoder_confounds_s = nn.Sequential(
        
                nn.ConvTranspose1d(in_channels=latent_dim[1], out_channels=128, 
                                   kernel_size=nTR, stride=1, bias=False),
            
                # First conv layer: convert 128 channels to 32 channels.
                nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
            
                # Second conv layer: convert 32 channels to 16 channels.
                nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(),
            
                # Third conv layer: convert 16 channels to 6 channels.
                nn.Conv1d(in_channels=16, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Sigmoid()  # Normalize the output to [0, 1]
                    )

    

        modules_z = []
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]
        
        self.pad, self.final_size, self.pad_out = compute_padding(self.in_dim)

        # Build Encoder
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_z.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder_z = nn.Sequential(*modules_z)
        self.fc_mu_z = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_z)
        self.fc_var_z = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_z)

        modules_s = []
        in_channels = self.in_channels
        for i in range(len(hidden_dims)):
            h_dim = hidden_dims[i]
            modules_s.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = int(self.pad[-i-1])),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder_s = nn.Sequential(*modules_s)
        self.fc_mu_s = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_s)
        self.fc_var_s = nn.Linear(hidden_dims[-1]*int(self.final_size), self.latent_dim_s)


        # Build Decoder
        modules = []

        #self.decoder_input = nn.Linear(2*latent_dim, hidden_dims[-1] * int(self.final_size))
        self.decoder_input = nn.Linear(self.latent_dim_s+self.latent_dim_z, hidden_dims[-1] * int(self.final_size))

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=int(self.pad_out[-4+i]),
                                    output_padding=int(self.pad_out[-4+i])),
                    nn.LeakyReLU()
                    )
            )


        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=int(self.pad_out[-1]),
                                               output_padding=int(self.pad_out[-1])),
                            nn.LeakyReLU(),
                            nn.Conv1d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1))
           #out_channels

    def encode_z(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_z(input)
  
        result = torch.flatten(result, start_dim=1)


        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_z(result)
        log_var = self.fc_var_z(result)

        return [mu, log_var]

    def encode_s(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_s(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu_s(result)
        log_var = self.fc_var_s(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1,256,int(self.final_size))
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward_tg(self, input: Tensor) -> List[Tensor]: # ORIGINAL
        tg_mu_z, tg_log_var_z = self.encode_z(input)
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_z = self.reparameterize(tg_mu_z, tg_log_var_z)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        #output = self.decode(torch.cat((tg_z, tg_s),1))
        output = self.forward_bg(input)[0]+self.forward_fg(input)[0]
        return  [output, input, tg_mu_z, tg_log_var_z, tg_mu_s, tg_log_var_s,tg_z,tg_s]

    def forward_fg(self, input: Tensor) -> List[Tensor]:
        tg_mu_s, tg_log_var_s = self.encode_s(input)
        tg_s = self.reparameterize(tg_mu_s, tg_log_var_s)
        zeros = torch.zeros(tg_s.shape[0],self.latent_dim_z)
        zeros = zeros.to(self.device)
        output = self.decode(torch.cat((zeros, tg_s),1))
        return  [output, input, tg_mu_s, tg_log_var_s]

    def forward_bg(self, input: Tensor) -> List[Tensor]:
        bg_mu_z, bg_log_var_z = self.encode_z(input)
        bg_z = self.reparameterize(bg_mu_z, bg_log_var_z)
        zeros = torch.zeros(bg_z.shape[0],self.latent_dim_s)
        zeros = zeros.to(self.device)
        output = self.decode(torch.cat((bg_z, zeros),1))
        return  [output, input, bg_mu_z, bg_log_var_z]


    def ncc(self, x, y, eps = 1e-8):
        x = x.flatten(start_dim=1)  # Flatten spatial dimensions
        y = y.flatten(start_dim=1)

        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_std = x.std(dim=1, keepdim=True)
        y_std = y.std(dim=1, keepdim=True)

        ncc = (x - x_mean) * (y - y_mean) / (x_std * y_std + eps)
        ncc = ncc.mean(dim=1)

        return 1 - ncc  # Return 1 - NCC to minimize the loss 

    
        

    def loss_function(self,
                      *args,
                      ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        beta = self.beta
        gamma = self.gamma
        delta = self.delta
                          
        recons_tg = args[0]
        input_tg = args[1]
        tg_mu_z = args[2]
        tg_log_var_z = args[3]
        tg_mu_s = args[4]
        tg_log_var_s = args[5]
        tg_z = args[6]
        tg_s = args[7]
        recons_bg = args[8]
        input_bg = args[9]
        bg_mu_z = args[10]
        bg_log_var_z = args[11]
        
        recons_fg = self.forward_fg(input_tg)[0]
        
        
        # recons_loss_roi = F.mse_loss(recons_tg, input_tg) * self.scale_MSE_GM 
        # recons_loss_roni = F.mse_loss(recons_bg, input_bg) * self.scale_MSE_CF 
        recons_loss_roi = F.mse_loss(recons_tg[:,0,:], input_tg[:,0,:]) * self.scale_MSE_GM #/ batch_size # TG reconstrction loss
        recons_loss_roni = F.mse_loss(recons_bg[:,0,:], input_bg[:,0,:]) * self.scale_MSE_CF # / batch_size # BG reconstrction loss
        #recons_loss_roi+=F.mse_loss(recons_fg[:,0,:], input_tg[:,0,:]) * self.scale_MSE_GM*.01
        recons_loss = recons_loss_roi+recons_loss_roni

        
        

        #recons_loss_fg = F.mse_loss(torch.zeros_like(input_bg[:,0,:]), self.forward_fg(input_bg)[0][:,0,:])*1e2 # Denoised version of RONI, should be all zeros
        #recons_loss_fg+=F.mse_loss(torch.zeros_like(input_bg[:,0,:]), self.forward_bg(recons_fg)[0][:,0,:])*1e2 # Noise features of FG should be all zeros

        confounds_pred_z = self.decoder_confounds_z(torch.unsqueeze(tg_z,2))
        confounds_pred_s = self.decoder_confounds_s(torch.unsqueeze(tg_s,2))

        loss_recon_conf_s = self.grl(F.mse_loss(confounds_pred_s, self.confounds))*1e2
        loss_recon_conf_z = F.mse_loss(confounds_pred_z, self.confounds)*1e2

        ncc_loss_tg = self.ncc(input_tg,recons_tg).mean()*1
        ncc_loss_bg = self.ncc(input_bg,recons_bg).mean()*1

        ncc_loss_conf_s = 0
        for i in range(self.confounds.shape[1]):
            ncc_loss_conf_s +=  self.ncc(self.confounds[:,i,:],confounds_pred_s[:,i,:]).mean()*1e1
        ncc_loss_conf_s = self.grl(ncc_loss_conf_s)

        ncc_loss_conf_z = 0
        for i in range(self.confounds.shape[1]):
            ncc_loss_conf_z += self.ncc(self.confounds[:,i,:],confounds_pred_z[:,i,:]).mean()*1e1

        #ncc_loss_conf_s*=1e2
        #ncc_loss_conf_z*=1e2


        recond_bg = self.forward_bg(input_tg)[0]
        fg_bg_ncc = self.ncc(recond_bg[:,0,:],recons_fg[:,0,:]).mean()
        recons_loss_fg = F.mse_loss(torch.zeros_like(fg_bg_ncc), 1-fg_bg_ncc)*1e4
        
        
        recons_loss+=F.mse_loss(recons_fg[:,0,:], input_tg[:,0,:])*.0001
        recons_loss+=F.mse_loss(recons_bg[:,0,:], input_bg[:,0,:])*.00001

        smoothness_loss = torch.mean((recons_fg[:,0,:][:, 1:] - recons_fg[:,0,:][:, :-1])**2)
        smoothness_loss+=torch.mean((recond_bg[:,0,:][:, 1:] - recond_bg[:,0,:][:, :-1])**2)
        smoothness_loss*=.01
        
        #ncc_losses = ncc_loss_tg+ncc_loss_bg+ncc_loss_conf_s+ncc_loss_conf_z
    
        do_disentangle=self.do_disentangle # Whether to do disentagling 
        disentangle_type = -1 # What type of disentangling to d
        fg_volatility_loss = torch.from_numpy(np.array(0)).to(self.device)
        total_contrastive_loss = torch.from_numpy(np.array(0)).to(self.device)
        ncc_loss_conf_s = torch.from_numpy(np.array(0)).to(self.device)

        
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + tg_log_var_z - tg_mu_z ** 2 - tg_log_var_z.exp(), dim = 1), dim = 0)
        kld_loss += torch.mean(-0.5 * torch.sum(1 + tg_log_var_s - tg_mu_s ** 2 - tg_log_var_s.exp(), dim = 1), dim = 0)
        kld_loss += torch.mean(-0.5 * torch.sum(1 + bg_log_var_z - bg_mu_z ** 2 - bg_log_var_z.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/3
        kld_loss = kld_loss * beta
        

        #loss = torch.sum(recons_loss + kld_loss)
        #loss = torch.sum(recons_loss + kld_loss + recons_loss_fg)
        #loss = torch.sum(recons_loss + kld_loss + recons_loss_fg + ncc_loss_tg + ncc_loss_bg)
        loss = torch.sum(recons_loss + kld_loss + recons_loss_fg + ncc_loss_tg + ncc_loss_bg + loss_recon_conf_s + loss_recon_conf_z + recons_loss_fg + smoothness_loss + ncc_loss_conf_z + ncc_loss_conf_s)
        #loss = torch.sum(recons_loss + kld_loss + ncc_loss_tg + ncc_loss_bg + loss_recon_conf_s + loss_recon_conf_z + ncc_loss_conf_s + ncc_loss_conf_z + recons_loss_fg + smoothness_loss)
        #loss = torch.sum(recons_loss + kld_loss + loss_recon_conf_s + loss_recon_conf_z + ncc_loss_conf_s + +ncc_loss_conf_z +ncc_loss_tg + ncc_loss_bg + recons_loss_fg + smoothness_loss)
        

        return {
            'loss' : loss,
            'kld_loss' : kld_loss,
            'recons_loss_roi' : recons_loss_roi,
            'recons_loss_roni' : recons_loss_roni,
            'loss_recon_conf_s' : loss_recon_conf_s,
            'loss_recon_conf_z' : loss_recon_conf_z,
            'ncc_loss_tg' : ncc_loss_tg,
            'ncc_loss_bg' : ncc_loss_bg,
            'ncc_loss_conf_s' : ncc_loss_conf_s,
            'ncc_loss_conf_z' : ncc_loss_conf_z,
            'smoothness_loss' : smoothness_loss,
            'recons_loss_fg' : recons_loss_fg}
        

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward_fg(x)[0]

    def _compute_log_density_gaussian(self, z, mu, log_var):
            """
            Computes the log density of a Gaussian for each sample in the batch.
            """
            normalization = -0.5 * (math.log(2 * math.pi) + log_var)
            log_prob = normalization - 0.5 * ((z - mu) ** 2 / log_var.exp())
            return log_prob.sum(dim=1)
        
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # Pass the input as is

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_  # Reverse and scale gradients
        return grad_input, None  # Second element corresponds to lambda_, which has no gradient

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    
    
def compute_in(x):
  return (x-3)/2+1

def compute_in_size(x):
  for i in range(4):
    x = compute_in(x)
  return x

def compute_out_size(x):
  return ((((x*2+1)*2+1)*2+1)*2+1)

def compute_padding(x):
  rounding = np.ceil(compute_in_size(x))-compute_in_size(x)
  y = ((((rounding*2)*2)*2)*2)
  pad = bin(int(y)).replace('0b', '')
  if len(pad) < 4:
      for i in range(4-len(pad)):
          pad = '0' + pad
  final_size = compute_in_size(x+y)
  pad_out = bin(int(compute_out_size(final_size)-x)).replace('0b','')
  if len(pad_out) < 4:
      for i in range(4-len(pad_out)):
          pad_out = '0' + pad_out
  return pad,final_size, pad_out