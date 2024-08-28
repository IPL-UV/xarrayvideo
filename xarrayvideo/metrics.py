#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Our libs
from .utils import sample, SEED

#Others
import numpy as np

def SM_SSIM(original, compressed, channel_axis=-1, data_range=1.0, kernel_size=11, levels=1,
            **kwargs):
    '''
         Compute MultiScaleSSIM, Multi-scale Structural Similarity Index Measure.
         This metric is is a generalization of Structural Similarity Index Measure 
         by incorporating image details at different resolution scores.
         https://lightning.ai/docs/torchmetrics/stable/image/multi_scale_structural_similarity.html
         Tries to use pytorch, otherwise reverts to skimage (slower)
    '''
    try:
        from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as smssim
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        import torch

        #torchmetrics is expecting (batch, channel, x, y). We will use t for the batch dimension
        #E.g.: txyc -> tcxy
        co= torch.from_numpy(np.swapaxes(compressed.astype(np.float32), 1, channel_axis))
        o= torch.from_numpy(np.swapaxes(original.astype(np.float32), 1, channel_axis))

        #Using smssim with levels=1 should be equivalent to using ssim, but it is not, so we use both
        if levels>1:
            #Choose betas, depending on levels
            betas= (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)[:levels]
            res= smssim(co, o, kernel_size=kernel_size, reduction='none', betas=betas, **kwargs)
        else:
            res= ssim(co, o, kernel_size=kernel_size, reduction='none', **kwargs)

        #Since there might be nans in some timesteps, lets apply the reduction later
        return np.nanmean(res.numpy())
    
    except Exception as e:
        print('Warning: Could not use torch\'s SM_SSIM, reverting back to slower ssim from skimage')
        from skimage.metrics import structural_similarity as ssim
        assert levels==1, 'Only levels==1 supported with skimage backend'
        return ssim(original, compressed, channel_axis=channel_axis, data_range=data_range)

def SA(original, compressed, channel_dim):
    '''
        Computes spectral angle according to:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6080751
    '''    
    #Sample
    original, compressed= sample(original, compressed)
    
    #To float32
    o, c= original.astype(np.float32), compressed.astype(np.float32)
    
    #Put channel_dim as first dimension (arbitrary, just for consensus)
    #And flatten the rest of the dimensions
    #Final shape: (c, -1)
    o= np.reshape(np.swapaxes(o, channel_dim, 0), (o.shape[channel_dim], -1))
    c= np.reshape(np.swapaxes(c, channel_dim, 0), (c.shape[channel_dim], -1))
    
    #Get angle of each of the -1 elements between o and c
    dot_product= np.einsum('ij,ij->j', o, c)
    prod_of_norms= np.linalg.norm(o, axis=0) * np.linalg.norm(c, axis=0)
    with warnings.catch_warnings(): #We already know there might be nans...
        warnings.simplefilter("ignore")
        theta= np.arccos(dot_product / prod_of_norms)
    
    #Filter out nans
    theta= theta[~np.isnan(theta)]
    
    #Get mean and error
    exp_theta= theta.mean()
    err_theta= np.sqrt(exp_theta/len(theta))
    
    return exp_theta, err_theta
    
    
def SNR(original, compressed, max_value):
    '''
        Peak signal-to-noise ratio (PSNR), signal-to-noise ratio (SNR) and 
        mean square error (MSE) between two images
    '''
    #To float32, filter out nans
    o, c= original.astype(np.float32), compressed.astype(np.float32)
    isnan= np.isnan(o) | np.isnan(c) 
    o, c= o[~isnan], c[~isnan]
    
    #Sample
    o, c= sample(o, c)
    
    #Compute metrics
    mse= np.mean((o-c)**2)
    var= np.mean(o**2)
    if mse == 0.: 
        psnr, snr= 100, 100
    else:
        psnr= 20 * np.log10(max_value / np.sqrt(mse))
        snr= 10 * np.log10(var / mse)
        
    return snr, psnr, mse