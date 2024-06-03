#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Others
import xarray as xr, numpy as np, ffmpeg
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Own lib
from .utils import (safe_eval, to_netcdf, normalize, denormalize, 
                    detect_planar, detect_rgb, reorder_coords_axis,
                    np2str, str2np)
from .ffmpeg_wrappers import _ffmpeg_read, _ffmpeg_write
from .plot import plot_simple
from .metrics import SA, SNR, SSIM

def get_pix_fmt(params, channels, bits):
    'Get the optimal input / compression pix_fmt'
    #For example, in gbrp16le, the pixel data for the green channel is stored together,
    #followed by the data for the blue channel, and then the red channel (planar arrangement)
    if bits != 8 and sys.byteorder != 'little':
        print('WARNING: System is big endian, but library was only tested in little endian systems!')
    compression= 'lossy'

    if params['c:v'] == 'libx264':
        #For x264: no alpha support
        #gbrp is theoretically not supported, but it works
        #gray is not supported either, x264 just transforms the array to gbrp
        input_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Input / Output
        req_pix_fmt= {1:'gray', 3:'yuv444p'}[channels] #Video 
        if bits == 8: 
            pass
        elif bits in [10]:
            input_pix_fmt+= f'{bits}le'
            req_pix_fmt+= f'{bits}le'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
    elif params['c:v'] == 'libx265':
        #For x265
        input_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Input
        req_pix_fmt= {1:'gray', 3:'yuv444p'}[channels] #Video  
        if bits == 8: 
            pass
        elif bits in [10,12]:
            input_pix_fmt+= f'{bits}le'
            req_pix_fmt+= f'{bits}le'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
    elif params['c:v'] == 'vp9':
        #For vp9
        #gbrap is theoretically not supported, but it works
        #gray is not supported either, vp9 just transforms the array to gbrp
        input_pix_fmt= {1:'gray', 3:'gbrp', 4:'gbrap'}[channels] #Input
        req_pix_fmt= {1:'gray', 3:'yuv444p', 4:'yuva420p'}[channels] #Video   
        if bits == 8: 
            pass
        elif bits in [10,12]:
            assert channels in [1,3],\
                f'For {params["c:v"]=}, {bits=} and {channels=} not supported'
            input_pix_fmt+= f'{bits}le'
            req_pix_fmt+= f'{bits}le'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
        if channels == 4:                 
            print(f'Warning: Lossy compressin with {channels=} is not currently well supported. '+\
                   'It should be possible with c:v=vp9 and format=yuva420p, but it uses '+\
                   ' yuv420p instead and generates bad-looking output...')
    elif params['c:v'] == 'ffv1':
        #For ffv1: many options supported
        req_pix_fmt= {1:'gray', 3:'yuv444p', 4:'yuva444p'}[channels] #Video
        if bits == 8:
            input_pix_fmt= {1:'gray', 3:'bgr0', 4:'bgra'}[channels] #Input
        elif bits in [10,12,16]: #gbrp supports more n of bits, but gray and gbrap do not
            input_pix_fmt= {1:'gray', 3:'gbrp', 4:'gbrap'}[channels] #Input
            input_pix_fmt+= f'{bits}le'
            req_pix_fmt+= f'{bits}le'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported / implemented')
        compression= 'lossless'
    elif params['c:v'] == 'prores_ks':
        assert channels in [3,4] and bits == 10
        req_pix_fmt= {3:'gbrp', 4:'gbrap'}[channels] #Video
        input_pix_fmt= {3:'yuv444p10le', 4:'yuv444p10le'}[channels] #Input
    else:
        raise AssertionError('Only codecs [libx264, libx265, ffv1, vp9] are implemented')
        
    return input_pix_fmt, req_pix_fmt, compression

#Crate a dimensionality reduction wrapper
class DRWrapper:
    def __init__(self, n_components=0, params=None):
        '''
            Standardize and apply PCA by flattening all dimensions except for the last one
        '''
        #Attributes
        self.fitted= False
        self.n_components= n_components
        self.bands= None
        
        #Initialize classes
        self.scaler= StandardScaler()                
        self.dr= PCA(n_components=self.n_components)
        if params is not None:
            self.set_params(self, params)
    
    def fit(self, X):
        self.fitted= True
        self.bands= X.shape[-1]
        X_flat= np.reshape(X, (-1, self.bands))
        X_flat= self.scaler.fit(X_flat)
        self.dr.fit(X_flat)
    
    def fit_transform(self, X):
        self.fitted= True
        final_shape= list(X.shape)
        final_shape[-1]= self.n_components
        self.bands=  X.shape[-1]
        X_flat= np.reshape(X, (-1,self.bands))
        X_flat= self.scaler.fit_transform(X_flat)
        X_pca_flat= self.dr.fit_transform(X_flat)
        return np.reshape(X_pca_flat, final_shape)
    
    def transform(self, X):
        assert self.fitted
        final_shape= list(X.shape)
        final_shape[-1]= self.n_components
        X_flat= np.reshape(X, (-1, self.bands))
        X_flat= self.scaler.transform(X_flat)
        X_pca_flat= self.dr.transform(X_flat)
        return np.reshape(X_pca_flat, final_shape)
    
    def inverse_transform(self, Y):
        assert self.fitted
        final_shape= list(Y.shape)
        final_shape[-1]= self.bands
        Y_flat= np.reshape(Y, (-1, self.n_components))
        Y_flat= self.dr.inverse_transform(Y_flat)
        Y_flat= self.scaler.inverse_transform(Y_flat)
        return np.reshape(Y_flat, final_shape)
    
    def get_params(self):
        params={}
        params['scaler']= self.scaler.get_params()
        params['dr']= self.dr.get_params()
        #TODO: Add matrices
        return params
    
    def set_params(self, params):
        self.fitted= True
        self.scaler.set_params(**params['scaler'])
        self.dr.set_params(**params['dr'])
        #TODO: Add matrices

#Forward function
def xarray2video(x, array_id, conversion_rules, compute_stats=False,
                 output_path='./', fmt='mkv', loglevel='quiet', use_ssim=False, exceptions='raise', 
                 verbose=True):
    '''
        Takes an xarray Dataset as input, and produces an xarray dataset as output, where some
        variables have been saved as video files (with some meta info for reading them back).
        
        Returns a dictionary of results with first level keys being conversion_rules.keys() and path,
        and second level keys being: path, [original_size, compressed_size, compressed, original, 
        mse, psnr, time] (if compute_stats), [ssim] if use_ssim
    '''
    print_fn= print if verbose else lambda *v: None #Disable printing if verbose=False
    results= {} #Some fields are only filled if compute_stats is True
    for name, config in conversion_rules.items():
        #Choose defaults if not all parameters are provided
        bits= 8 #Best support, not best choice for technical images
        params= {
            'c:v': 'libx264',  #[libx264, libx265, vp9, ffv1]
            'preset': 'medium',  #Preset for quality/encoding speed tradeoff: quick, medium, slow (better)
            'crf': 5, #14 default, 11 for higher quality and size
            }
        if len(config) == 5:
            bands, coord_names, use_pca, params, bits= config
        elif len(config) == 4:
            bands, coord_names, use_pca, params= config
            bits= 8 #Best support, not best choice for technical images
        elif len(config) == 3:
            bands, coord_names, use_pca= config
        else:
            raise AssertionError(f'Params: {config} should be: bands, coord_names, use_pca, [params], [bits]')
            
        try:
            #Array -> uint8 or uint16, shape: (t, x, y, c)
            if isinstance(bands, str): bands= [bands]
            if not use_pca:
                assert len(bands) in [1,3,4], f'For {name=} expected to find 1, 3, or 4 bands, found {bands=}'
            channels= len(bands)
            coords= x[bands[0]].coords.dims
            attrs= {b:x[b].attrs for b in bands}
            array= np.stack([x[b].transpose(*coord_names).values for b in bands], axis=-1)
            value_range= np.stack([ np.nanmin(array, axis=tuple(range(array.ndim - 1))), 
                                    np.nanmax(array, axis=tuple(range(array.ndim - 1))) ], axis=1)
            # value_range= np.array([[ np.nanmin(array), np.nanmax(array) ]]*len(bands))
            dtype= array.dtype
            if compute_stats: array_orig= array.copy()
            
            #Use PCA
            if use_pca:
                #Keep only some components
                n_components= 6
                video_files= channels // 3
                if n_components is None:
                    n_components= video_files * 3
                else:
                    assert n_components % 3 == 0 and n_components <= channels
                
                #Initialize classes
                array[np.isnan(array)]= 0
                DR= DRWrapper(n_components=n_components)
                array= DR.fit_transform(array)
                #array= array[:,:n_components]
                
                #Transform limits to the PCA space and overwrite value_range and channel values
                value_range= DR.transform(value_range.T).T
                channels= 3
            else:
                video_files= 1
                                    
            #Normalize
            array= normalize(array, minmax=value_range, bits=bits)
                
            #Get shapes
            x_len, y_len= array.shape[1], array.shape[2]

            #Choose pixel format: only some are supported
            input_pix_fmt, req_pix_fmt, compression= get_pix_fmt(params, channels, bits)
                
            #Deted if planar format
            planar_in= detect_planar(input_pix_fmt)
                
            #Convert rgb <> gbr, etc.
            #This is only relevant for visualizing the output videos
            if not 'rgb' in input_pix_fmt and channels in [3,4]:
                ordering= detect_rgb(input_pix_fmt)
                if ordering is not None:
                    a= 'a' if channels == 4 else ''
                    array= reorder_coords_axis(array, list('rgb'+a), list(ordering+a), axis=-1)
                else:
                    ordering= 'rgb'
                
            #Add custom metainfo used by video2xarray
            metadata= {}
            if use_pca:
                metadata['BANDS']= str(['PCA1', 'PCA2', 'PCA3'])
            else:
                metadata['BANDS']= str([bands]) if isinstance(bands, str) else str(bands)
            metadata['COORDS_DIMS']= str(coords)
            metadata['ATTRS']= str(attrs)
            metadata['FRAMES']= str(array.shape[0])
            metadata['RANGE']= np2str(value_range)
            metadata['OUT_PIX_FMT']= input_pix_fmt
            metadata['REQ_PIX_FMT']= req_pix_fmt
            metadata['PLANAR']= planar_in
            metadata['BITS']= bits
            metadata['NORMALIZED']= compression == 'lossy'
            metadata['CHANNEL_ORDER']= ordering #e.g. gbr

            #Pathing
            output_path= Path(output_path)
            (output_path / array_id).mkdir(exist_ok=True, parents=True)
            comp_names= [name] if not use_pca else [f'{name}_{i}' for i in range(video_files)]
            results[name]= {}
            results[name]['path']= [output_path / array_id / f'{cn}.{fmt}' for cn in comp_names]
            
            #Go over every set of 3 components if use_pca else ignore the loop
            t0= time.time()            
            for i, output_path_video in enumerate(results[name]['path']):
                #Write with ffmpeg
                params['pix_fmt']= req_pix_fmt
                params['r']= 30
                array_in= array if not use_pca else array[...,i*3:(i+1)*3]
                _ffmpeg_write(str(output_path_video), array_in, x_len, y_len, params, planar_in=planar_in,
                              loglevel=loglevel, metadata=metadata, input_pix_fmt=input_pix_fmt)
                
            #Modify minicube to delete the bands we just processed
            x= x.drop_vars(bands)
            t1= time.time()

            #Show stats
            if compute_stats:
                #Size and time
                array_size= array_orig.size * array_orig.itemsize / 2**20 #In Mb (use array_orig dtype)
                video_size= sum([v.stat().st_size / 2**20 for v in results[name]['path']]) #In Mb
                percentage= (video_size / array_size)*100
                bpppb= video_size * 2**20 * 8 / array.size #bits per pixel per band (=bps)

                print_fn(f'{name}: {array_size:.2f}Mb -> {video_size:.2f}Mb '+\
                         f'({percentage:.2f}% of original size, {bpppb:.4f} bpppb) in {t1 - t0:.2f}s'\
                        f'\n - {params=}')
                
                results[name]['original_size']= array_size
                results[name]['compressed_size']= video_size
                results[name]['compression']= video_size / array_size
                results[name]['bpppb']= bpppb
                results[name]['time']= t1 - t0

                #Assess read time and reconstruction quality
                t0= time.time()
                array_comp_list= [_ffmpeg_read(v)[0] for v in results[name]['path']]
                array_comp= np.concatenate(array_comp_list, axis=-1)
                t1= time.time()
                results[name]['d_time']= t1 - t0
                print_fn(f' - Decompression time {results[name]["d_time"]:.2f}s')
                if use_pca: 
                    array_comp= denormalize(array_comp, minmax=value_range, bits=bits)
                    array_comp= DR.inverse_transform(array_comp)
                assert array_comp.shape == array_orig.shape, f'{array_comp.shape=} != {array_orig.shape=}'
                
                if compression == 'lossy' or dtype!=np.uint8:
                    if not use_pca: 
                        array_comp= denormalize(array_comp, minmax=value_range, bits=bits)
                    
                    #Plot last frame
                    if verbose: 
                        plot_simple(array_orig[-1, ..., -3:], max_val=value_range.max(), title='Original')
                        plot_simple(array_comp[-1, ..., -3:], max_val=value_range.max(), title='Compressed')
                        print(f'Saturation values per band {bands}):\n {value_range}')

                    # ssim_arr= ssim(array_orig, array_comp, channel_axis=3 if channels==3 else None)
                    # print_fn(f' - SSIM {ssim_arr:.6f}, read in {t1 - t0:.2f}s')

                    #Process the orginal array in the same way as the video, to be able to compare them
                    # array_bands=[]
                    # for c in range(array_orig.shape[-1]):
                    #     array_c= array_orig[...,c]
                    #     array_c[array_c > value_range[c,1]]= value_range[c,1]
                    #     array_c[array_c < 0]= 0
                    #     array_c[np.isnan(array_c)]= 0
                    #     array_bands.append(array_c)
                    # array_orig_sat= np.stack(array_bands, axis=-1)
                    array_orig_sat= array_orig #TODO
                    
                    if use_ssim:
                        ssim_val, ssim_val_db= SSIM(array_orig_sat, array_comp, channel_dim=-1, 
                                                    max_value=value_range.max())
                        ssim_val2= ssim(array_orig_sat, array_comp, channel_axis=-1, 
                                        data_range=value_range.max()-value_range.min())
                        print_fn(f' - SSIM_sat {ssim_val:.6f} {ssim_val2:.6f} (input saturated)')
                        results[name]['ssim']= ssim_val

                    snr, psnr, mse= SNR(array_orig_sat, array_comp, max_value=value_range.max())
                    exp_sa, err_sa= SA(array_orig_sat, array_comp, channel_dim=-1)
                    print_fn(f' - MSE_sat {mse:.6f} (input saturated)')
                    print_fn(f' - SNR_sat {snr:.4f} (input saturated)')
                    print_fn(f' - PSNR_sat {psnr:.4f} (input saturated)')
                    print_fn(f' - Exp. SA {exp_sa:.4f} (input saturated)')
                    results[name]['snr']= snr
                    results[name]['psnr']= psnr
                    results[name]['mse']= mse
                    results[name]['exp_sa']= exp_sa

                else:
                    acc= np.nanmean(array_comp==array_orig)
                    print_fn(f' - acc {acc:.2f}')
                    results[name]['acc']= acc

                results[name]['compressed']= array_comp
                results[name]['original']= array_orig
                
        except Exception as e:
            print(f'Exception processing {array_id=} {name=}: {e}')
            if exceptions == 'raise': raise e

    #Save the resulting xarray
    results['path']= output_path / array_id / 'x.nc'
    to_netcdf(x, results['path'])
            
    return results

#Bakwards function
def video2xarray(input_path, array_id, fmt='mkv', exceptions='raise', x_name='x', y_name='y'):
    #To path
    path= Path(input_path)
    
    #Load xarray
    x= xr.open_dataset(path / array_id / 'x.nc')
    
    #Go over videos, read them, and integrate them into the dataset
    for video_path in (path / array_id).glob(f'*.{fmt}'):
        
        #Read array and meta
        array, meta_info= _ffmpeg_read(video_path)
        bands= safe_eval(meta_info['BANDS'])
        coords_dims= list(safe_eval(meta_info['COORDS_DIMS']))
        attrs= safe_eval(meta_info['ATTRS'])
        value_range= str2np(meta_info['RANGE'])
        normalized= meta_info['NORMALIZED'] in [True, 'True']
        bits= int(meta_info['BITS'])
        ordering= meta_info['CHANNEL_ORDER']
        
        #I'm not sure why, but saving the video transposes x and y
        x_pos, y_pos= coords_dims.index(x_name), coords_dims.index(y_name)
        coords_dims[x_pos], coords_dims[y_pos]= y_name, x_name
                
        #Rescale array
        if normalized:
            array= denormalize(array, minmax=value_range, bits=bits)
        
        #To rgb if needed
        if ordering != 'rgb':
            a= 'a' if len(bands) == 4 else ''
            array= reorder_coords_axis(array, list(ordering+a), list('rgb'+a), axis=-1)
        
        #Go over bands and set them into the xarray
        for i, (band, attr) in enumerate(zip(bands, attrs.values())):
            try:
                x[band]= xr.DataArray(data=array[...,i], dims=coords_dims, attrs=attr)
            except Exception as e:
                print(f'Exception processing {array_id=} {band=}: {e}')
                if exceptions == 'raise': raise e
            
    return x