#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Others
import xarray as xr, numpy as np, ffmpeg
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def safe_eval(s):
    'Evaluates only simple expressions for safety over eval'
    evaluated= None
    try:
        #literal_eval does not support having nan as a value
        parsed= ast.parse(s.replace('nan', 'None'), mode='eval')
        evaluated= ast.literal_eval(parsed)
    except Exception as e:
        print(f'Exception evaulating {s=}: {e}')
    return evaluated

def sanitize_attributes(x):
    attrs_new={}
    for attr, value in x.attrs.items():
        if isinstance(value, dict) or isinstance(value, list) and isinstance(value[0], dict):
            attrs_new= {attr:str(value), **attrs_new}
    return x.assign_attrs(attrs_new)

def sanitize_xarray(x):
    for b in x.variables:
        x[b]= sanitize_attributes(x[b])
    for c in x.coords:
        x[c]= sanitize_attributes(x[c])
    return sanitize_attributes(x)

def to_netcdf(x, *args, **kwargs):
    x= sanitize_xarray(x)
    x.to_netcdf(*args, **kwargs, encoding={var: {'zlib': True} for var in x.variables})

def gap_fill(x:xr.Dataset, fill_bands:List[str], mask_band:str, fill_nans:bool=True, fill_zeros:bool=True,
             fill_values:Optional[List[int]]=None, new_mask='invalid', coord_names=('time', 'variable', 'x', 'y'),
             method:str='last_value'):
    #Checks
    if coord_names[0] not in ['t', 'time', 'frame']: 
        print('Make sure that the first element of `coord_names` is the time dimension')
    
    #Prepare bands array
    x_array= x[fill_bands].to_array()
    array= x_array.transpose(*coord_names).values # t c x y
    array_filled= np.zeros_like(array)
    
    #Get gap mask
    mask= x[[mask_band]].to_array().transpose(*coord_names).values # t 1 x y
    if fill_values == None:
        fill_mask= mask > 0
    else:
        fill_mask= np.isin(mask, fill_values)
    fill_mask= np.concatenate([fill_mask]*array.shape[1], axis=1) # t c, y x
    if fill_nans: 
        fill_mask|= np.isnan(array)
    axis_cxy= tuple(list(range(len(array.shape)))[1:])
    if fill_zeros: 
        fill_mask|= ~np.any(array, axis=axis_cxy, keepdims=True)
        
    #Perform filling
    if method == 'last_value':
        #If there is cloud, use previous value, otherwise, use input values
        array_filled[0]= array[0] #For first timestep, perform no gapfill
        for t in range(1,array.shape[0]): #iterate over time channel
                array_filled[t]= np.where(fill_mask[t], array_filled[t-1], array[t])
                # array_filled[t]= np.where(fill_mask[t] & ~fill_mask[t-1], array_filled[t-1], array[t])
    elif method == 'interp_forward':
        array_filled[0:2]= array[0:2] #For first timestep, perform no gapfill
        for t in range(2,array.shape[0]): #iterate over time channel
            interp= fill_mask[t-1] #x2= x1 + (x1-x0)
            array_filled[t]= np.where(fill_mask[t], 2*array_filled[t-1]-array_filled[t-2], array[t])
    elif method == 'interp':
        from scipy.interpolate import interp1d
        time_indices= np.arange(len(array))
        axis_ct= (0,1)
        mask= ~fill_mask
        array_filled= array.copy()
        if mask.any():
            # If the first frame has missing values, fill them with the mean of the frame
            if ~mask[0].any():
                array_filled[0]= np.nanmean(array_filled, axis=axis_ct)
                mask[0]= True
            # If the last frame has missing values, fill them with the mean of the frame
            if ~mask[-1].any():
                array_filled[-1]= np.nanmean(array_filled, axis=axis_ct)
                mask[-1]= True

            # Use linear interpolation to fill missing values
            mask_t= np.sum(mask, axis=axis_cxy) > 0
            raise NotImplementedError('')
            # interp_func= interp1d(time_indices, array_filled[mask_t], kind="linear")
            # array_filled[~mask]= interp_func(time_indices[~mask])
    else:
        raise AssertionError(f'Unknown {method=}')
    
    #Create new xarray Dataset and add new values
    x_new= x.copy()
    single_band_coord_names= [c for c in coord_names if c!='variable']
    for i,b in enumerate(fill_bands):
        x_new[b]= xr.DataArray(data=array_filled[:,i], dims=x_new[b].coords.dims, 
                               attrs=x_new[b].attrs).transpose(*single_band_coord_names)
    x_new[new_mask]= xr.DataArray(fill_mask[:,0], dims=x_new[mask_band].coords.dims, 
                                  attrs=x_new[b].attrs).transpose(*single_band_coord_names)
    
    return x_new
        
# def reorder_coords(array, coords_in, coords_out):
#     '''
#         Permutes and reorders an array from coords_in into coords_out
#         E.g.: coords_in= ('y', 'x', 't'), coords_out= ('t', 'x', 'y')
#     '''
#     new_order= [coords_out.index(i) for i in coords_in]
#     breakpoint()
#     return np.transpose(array, new_order)

def normalize(array, minmax=(0,1.)):
    'If array is not uint8, clip array to `minmax` and rescale to [0, 255]' 
    if array.dtype != np.uint8:
        array= (array - minmax[0]) / (minmax[1] - minmax[0]) * 255
        array[array > 255]= 255
        array[array < 0]= 0
        array[np.isnan(array)]= 0
        array= array.astype(np.uint8)
    return array
