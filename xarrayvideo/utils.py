#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Others
import xarray as xr, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Set global seed
SEED= 42
MAX_SAMPLES= 1e9

#Examples of formats for ffv1
#yuv420p yuva420p yuva422p yuv444p yuva444p yuv440p yuv422p yuv411p yuv410p bgr0 bgra yuv420p16le 
#yuv422p16le yuv444p16le yuv444p9le yuv422p9le yuv420p9le yuv420p10le yuv422p10le yuv444p10le yuv420p12le 
#yuv422p12le yuv444p12le yuva444p16le yuva422p16le yuva420p16le yuva444p10le yuva422p10le yuva420p10le 
#yuva444p9le yuva422p9le yuva420p9le gray16le gray gbrp9le gbrp10le gbrp12le gbrp14le gbrap10le gbrap12le 
#ya8 gray10le gray12le gbrp16le rgb48le gbrap16le rgba64le gray9le yuv420p14le yuv422p14le yuv444p14le 
#yuv440p10le yuv440p12le

def sample(*arrays, max_samples=MAX_SAMPLES, seed=SEED, axis=0):
    '''
        Randomly sample the same slices for all arrays over a given axis
    '''
    if arrays[0].size > max_samples:
        assert axis==0, 'TODO: if not in axis 0, do swapaxes to 0 before and after'
        rng= np.random.default_rng(seed)
        N= int(max_samples / arrays[0].size * arrays[0].shape[0])
        idx= rng.choice(arrays[0].shape[0], size=N, replace=False)
        return [a[idx] for a in arrays]
    else:
        return arrays

class DRWrapper:
    def __init__(self, n_components=0, params=None, scale=False, max_train=1e6, **kwargs):
        '''
            Standardize and apply PCA by flattening all dimensions except for the last one
        '''
        #Attributes
        self.fitted= False
        self.bands= None
        self.max_train= max_train
        self.rng= np.random.default_rng(SEED)
        
        #Initialize classes           
        if params is None:
            self.scaler= StandardScaler() if scale else None 
            self.n_components= n_components
            self.scale= scale
            self.dr= PCA(n_components=self.n_components, **kwargs)
        else:
            if isinstance(params, str):
                self.set_params_str(params)
            else:
                self.set_params(params)
            self.n_components= self.dr.n_components
            self.scale= self.scaler is not None
    
    def fit(self, X):
        self.fitted= True
        self.bands= X.shape[-1]
        X_flat= np.reshape(X, (-1, self.bands))
        X_flat= X_flat[~np.any(np.isnan(X_flat), axis=-1)]
        if len(X_flat) > self.max_train:
            X_flat= self.rng.choice(X_flat, size=int(self.max_train), replace=False)
        if self.scale: X_flat= self.scaler.fit(X_flat)
        self.dr.fit(X_flat)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def transform(self, X):
        assert self.fitted
        final_shape= list(X.shape)
        final_shape[-1]= self.n_components
        X_flat= np.reshape(X, (-1, self.bands))
        if self.scale: X_flat= self.scaler.transform(X_flat)
        nans= np.any(np.isnan(X_flat), axis=-1)
        X_flat[nans]= 0.
        X_pca_flat= self.dr.transform(X_flat)
        X_flat[nans]= np.nan
        return np.reshape(X_pca_flat, final_shape)
    
    def inverse_transform(self, Y):
        assert self.fitted
        final_shape= list(Y.shape)
        final_shape[-1]= self.bands
        Y_flat= np.reshape(Y, (-1, self.n_components))
        Y_flat= self.dr.inverse_transform(Y_flat)
        if self.scale: Y_flat= self.scaler.inverse_transform(Y_flat)
        return np.reshape(Y_flat, final_shape)
    
    def get_params(self):
        assert self.fitted
        params={}
        params['scaler']= self.scaler.get_params() if self.scale else None
        params['dr']= self.dr.get_params()
        params['dr_matrix']= self.dr.components_
        params['dr_mean']= self.dr.mean_
        params['dr_bands']= self.bands
        return params
    
    def set_params(self, params):
        self.fitted= True
        self.scaler= StandardScaler(**params['scaler']) if params['scaler'] is not None else None
        self.dr= PCA(**params['dr'])
        self.dr.components_= params['dr_matrix']
        self.dr.mean_= params['dr_mean']
        self.bands= params['dr_bands']
        
    def get_params_str(self):
        params= self.get_params()
        params['dr']= str(params['dr'])
        params['dr_matrix']= np2str(params['dr_matrix'])
        params['dr_mean']= np2str(params['dr_mean'])
        params['scaler']= str(params['scaler'])
        params['dr_bands']= str(params['dr_bands'])
        return str(params)
    
    def set_params_str(self, params_str):
        params= safe_eval(params_str)
        params['dr']= safe_eval(params['dr'])
        params['dr_matrix']= str2np(params['dr_matrix']).astype(np.float32)
        params['dr_mean']= str2np(params['dr_mean']).astype(np.float32)
        params['scaler']= safe_eval(params['scaler'])
        params['dr_bands']= int(params['dr_bands'])
        self.set_params(params)

def detect_rgb(fmt):
    for ordering in ['rgb', 'rbg', 'gbr', 'grb', 'brg', 'bgr']:
        if ordering in fmt: 
            return ordering
    return None

def detect_planar(fmt):
    return 'p' in fmt

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
        
def reorder_coords(array, coords_in, coords_out):
    '''
        Permutes and reorders all axis of an array from coords_in into coords_out
        E.g.: coords_in= ('y', 'x', 't'), coords_out= ('t', 'x', 'y')
    '''
    new_order= [coords_in.index(i) for i in coords_out]
    return np.transpose(array, new_order)

def reorder_coords_axis(array, coords_in, coords_out, axis=-1):
    '''
        Permutes and reorders the dimensions within a single axis of an array 
        from coords_in into coords_out
        E.g.: axis=-1, coords_in= ('y', 'x', 't'), coords_out= ('t', 'x', 'y')
    '''
    if coords_in == coords_out: 
        return array
    else:
        new_order= [coords_in.index(i) for i in coords_out]
        #Move reorder axis to position 0, reorder, and then move it back to where it was
        return np.swapaxes(np.swapaxes(array, axis, 0)[new_order], axis, 0)

def is_float(array=None, dtype=None):
    if array is not None:
        return np.issubdtype(array.dtype, np.floating)
    elif dtype is not None:
        return np.issubdtype(dtype, np.floating)
    else:
        raise RuntimeError('Either `array` or `dtype` must be provided')

# def normalize(array, minmax=(0.,1.), bits=8):
#     '''
#         If array is not uint8, clip array to `minmax` and rescale to [0, 2**bits].
#         For bits=8, uint8 is used as output, for bits 9-16, uint16 is used 
#     '''
#     assert bits >=8 and bits <=16, 'Only 8 to 16 bits supported'
#     max_value= 2**bits - 1
#     if array.dtype != np.uint8:
#         array= (array - minmax[0]) / (minmax[1] - minmax[0]) * max_value
#         array[array > max_value]= max_value
#         array[array < 0]= 0
#         array[np.isnan(array)]= 0
#         array= np.round(array).astype(np.uint8 if bits == 8 else np.uint16)
#     return array

# def denormalize(array, minmax=(0.,1.), bits=8):
#     '''
#         Transform to float32, and undo the scaling done in `normalize`
#     '''
#     max_value= 2**bits - 1
#     return array.astype(np.float32) / max_value * (minmax[1] - minmax[0]) + minmax[0]

def normalize(array, minmax, bits=8):
    '''
        If array is not uint8, clip array to `minmax` and rescale to [0, 2**bits-1].
        For bits=8, uint8 is used as output, for bits 9-16, uint16 is used.
        minmax must have shape (B,2), and array must have shape (...,B)
    '''
    if array.dtype == np.uint8: 
        return array
    
    assert bits >=8 and bits <=16, 'Only 8 to 16 bits supported'
    max_value= 2**bits - 1
    array_bands= []
    for c in range(array.shape[-1]):
        array_c= array[...,c]
        array_c= array_c.astype(np.float32)
        array_c= (array_c - minmax[c,0]) / (minmax[c,1] - minmax[c,0]) * max_value
        array_c[array_c > max_value]= max_value
        array_c[array_c < 0]= 0
        array_c[np.isnan(array_c)]= 0
        array_bands.append(array_c)        
    return np.round(np.stack(array_bands, axis=-1)).astype(np.uint8 if bits == 8 else np.uint16)

def denormalize(array, minmax, bits=8):
    '''
        Transform to float32, and undo the scaling done in `normalize`
        minmax must have shape (B,2), and array must have shape (...,B)
    '''
    max_value= 2**bits - 1
    array_bands= []
    for c in range(array.shape[-1]):
        array_c= array[...,c]
        array_c= array_c.astype(np.float32)
        array_c= array_c / max_value * (minmax[c,1] - minmax[c,0]) + minmax[c,0]
        array_bands.append(array_c)     
    return np.stack(array_bands, axis=-1)

def np2str(array:np.ndarray, decimals:int=7):
    return str( (np.trunc(array*10**decimals)/(10**decimals)).tolist() )

def str2np(s:str):
    return np.array(safe_eval(s))