#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Others
import numpy as np
from skimage.metrics import structural_similarity as ssim

def SA(original, compressed, channel_dim):
    '''
        Computes spectral angle according to:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6080751
    '''
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
    
    #Compute metrics
    mse= np.mean((o-c)**2)
    var= np.mean(o**2)
    if mse == 0.: 
        psnr, snr= 100, 100
    else:
        psnr= 20 * np.log10(max_value / np.sqrt(mse))
        snr= 10 * np.log10(var / mse)
        
    return snr, psnr, mse
