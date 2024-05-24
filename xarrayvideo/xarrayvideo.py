#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional

#Others
import xarray as xr, numpy as np, ffmpeg
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

#Own lib
from .utils import safe_eval, to_netcdf, normalize
from .ffmpeg_wrappers import _ffmpeg_read, _ffmpeg_write
from .plot import plot_simple
from .metrics import SA, SNR

#Forward function
def xarray2video(x, array_id, conversion_rules, value_range=(0.,1.), compute_stats=False,
                 lossy_params={'c:v': 'libx264', 'r': 30, 'preset': 'slow', 'crf': 11},
                 lossless_params={'c:v': 'ffv1'}, output_path='./', fmt='mkv',
                 loglevel='quiet', use_ssim=False, exceptions='raise', verbose=True):
    '''
        Takes an xarray Dataset as input, and produces an xarray dataset as output, where some
        variables have been saved as video files (with some meta info for reading them back).
        
        Returns a dictionary of results with first level keys being conversion_rules.keys() and path,
        and second level keys being: path, [original_size, compressed_size, compressed, original, 
        mse, psnr, time] (if compute_stats), [ssim] if use_ssim
    '''
    print_fn= print if verbose else lambda *v: None #Disable printing if verbose=False
    results= {} #Some fields are only filled if compute_stats is True
    for name, (bands, coord_names, compression) in conversion_rules.items():
        try:
            #Array -> uint8, shape: (t, x, y, (3))
            if isinstance(bands, str):
                #Get the array and permute into ordering (t, x, y, (3)), type uint8, and range (0,255)
                channels= 1
                coords= x[bands].coords.dims
                attrs= {bands:x[bands].attrs}
                array= x[bands].transpose(*coord_names).values
                dtype= array.dtype
                if compute_stats: array_orig= array.copy()
                array= normalize(array, minmax=value_range)
                if compression != 'lossless': 
                    print('Warning: as of now, 1-channel compression takes ~3x as much space as it should'
                          ' due to gray pix_fmt not being respected by x264, and hence rgb being used.')
            else:
                assert len(bands) in [3,4], f'For {name=} expected to find 3 or 4 bands, found {bands=}'
                channels= len(bands)
                if channels == 4:
                    print(f'Warning: {channels=} is not currently supported. '+\
                           'It should be possible with c:v=vp9 and format=yuva420p, but it is still '+\
                           'using yuv420p and generating 1.6x as many frames as requested...')
                coords= x[bands[0]].coords.dims
                attrs= {b:x[b].attrs for b in bands}
                array= np.stack([x[b].transpose(*coord_names).values for b in bands], axis=-1)
                dtype= array.dtype
                if compute_stats: array_orig= array.copy()
                array= normalize(array, minmax=value_range)
                
            #Get shapes
            x_len, y_len= array.shape[1], array.shape[2]

            #Choose pixel format: only some are supported, and only in 8bit precision
            #For example, in gbrp16le, the pixel data for the green channel is stored together,
            #followed by the data for the blue channel, and then the red channel (planar arrangement)
            params= lossy_params if compression == 'lossy' else lossless_params
            if params['c:v'] == 'libx264':
                #For x264: no alpha support, gray is somehow NOT working
                input_pix_fmt= {1:'gray', 3:'rgb24'}[channels] #Input / Output
                req_pix_fmt= {1:'gray', 3:'yuv444p'}[channels] #Video
                planar_in= False
                
            elif params['c:v'] == 'libx265':
                #For x265
                input_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Input
                req_pix_fmt= {1:'gray', 3:'yuv444p'}[channels] #Video
                planar_in= True
                
            elif params['c:v'] == 'vp9':
                #For vp9
                input_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Input
                req_pix_fmt= {1:'gray', 3:'yuv444p', 4:'yuva420p'}[channels] #Video
                planar_in= True
                
            elif params['c:v'] == 'ffv1':
                #For ffv1
                input_pix_fmt= {1:'gray', 4:'bgra'}[channels] #Input
                req_pix_fmt= {1:'gray', 4:'bgra'}[channels] #Video
                planar_in= False
                
            else:
                assert compression == 'lossless', \
                    f'For lossy compression only [libx264, libx265, vp9] are supported'
                
            #Convert rgb <> gbr, etc.
            if not 'rgb' in input_pix_fmt:
                pass #TODO: must also be implemented in video2xarray 
                
            #Add custom metainfo
            metadata= {}
            metadata['BANDS']= str([bands]) if isinstance(bands, str) else str(bands)
            metadata['COORDS_DIMS']= str(coords)
            metadata['ATTRS']= str(attrs)
            metadata['FRAMES']= str(array.shape[0])
            metadata['RANGE']= str(value_range)
            metadata['COMPRESSION']= str(compression)
            metadata['OUT_PIX_FMT']= input_pix_fmt
            metadata['REQ_PIX_FMT']= req_pix_fmt
            metadata['PLANAR']= planar_in

            output_path= Path(output_path)
            (output_path / array_id).mkdir(exist_ok=True, parents=True)
            output_path_video= output_path / array_id / f'{name}.{fmt}'
            results[name]= {}
            results[name]['path']= output_path_video

            #Write with ffmpeg
            t0= time.time()
            params['pix_fmt']= req_pix_fmt
            params['r']= 30
            _ffmpeg_write(str(output_path_video), array, x_len, y_len, params, planar_in=planar_in,
                          loglevel=loglevel, metadata=metadata, input_pix_fmt=input_pix_fmt)
            t1= time.time()
                            
            #Modify minicube to delete the bands we just processed
            x= x.drop_vars(bands)

            #Show stats
            if compute_stats:
                #Size and time
                array_size= array.size * array.itemsize / 2**20 #In Mb
                video_size= output_path_video.stat().st_size / 2**20 #In Mb
                percentage= (video_size / array_size)*100

                print_fn(f'{name}: {array_size:.2f}Mb -> {video_size:.2f}Mb '+\
                         f'({percentage:.2f}% of original size) in {t1 - t0:.2f}s'\
                         f'\n - {params=}')
                
                results[name]['original_size']= array_size
                results[name]['compressed_size']= video_size
                results[name]['compression']= video_size / array_size
                results[name]['bpppb']= video_size * 2**20 * 8 / array.size #bits per pixel per band (=bps)
                results[name]['time']= t1 - t0

                #Assess read time and reconstruction quality
                t0= time.time()
                array_comp, metadata= _ffmpeg_read(str(output_path_video))
                t1= time.time()
                results[name]['d_time']= t1 - t0
                print_fn(f' - Decompression time {results[name]["d_time"]:.2f}s')
                assert array_comp.shape == array_orig.shape, f'{array_comp.shape=} != {array_orig.shape=}'

                #Plot last frame
                if verbose: 
                    plot_simple(array_orig[-1], title='Original')
                    plot_simple(array_comp[-1], title='Compressed')
                
                if compression == 'lossy' or dtype!=np.uint8:
                    array_comp= array_comp.astype(np.float32) / 255 * (value_range[1] - value_range[0]) + value_range[0]

                    # ssim_arr= ssim(array_orig, array_comp, channel_axis=3 if channels==3 else None)
                    # print_fn(f' - SSIM {ssim_arr:.6f}, read in {t1 - t0:.2f}s')

                    array_orig_sat= np.copy(array_orig)
                    array_orig_sat[array_orig_sat > value_range[1]]= value_range[1]
                    array_orig_sat[array_orig_sat < value_range[0]]= value_range[0]

                    if use_ssim:
                        ssim_arr2= ssim(array_orig_sat, array_comp, channel_axis=-1 if channels > 1 else None,
                                        data_range=value_range[1]-value_range[0])
                        print_fn(f' - SSIM_sat {ssim_arr2:.6f} (input saturated to [{value_range[0], value_range[1]}])')
                        results[name]['ssim']= ssim_arr2

                    snr, psnr, mse= SNR(array_orig_sat, array_comp, max_value=value_range[1])
                    exp_sa, err_sa= SA(array_orig_sat, array_comp, channel_dim=-1)
                    print_fn(f' - MSE_sat {mse:.6f} (input saturated to [{value_range[0], value_range[1]}])')
                    print_fn(f' - SNR_sat {snr:.4f} (input saturated to [{value_range[0], value_range[1]}])')
                    print_fn(f' - PSNR_sat {psnr:.4f} (input saturated to [{value_range[0], value_range[1]}])')
                    print_fn(f' - Exp. SA {exp_sa:.4f} (input saturated to [{value_range[0], value_range[1]}])')
                    print_fn(f' - Err. SA {err_sa:.4f} (input saturated to [{value_range[0], value_range[1]}])')
                    results[name]['snr']= snr
                    results[name]['psnr']= psnr
                    results[name]['mse']= mse
                    results[name]['exp_sa']= exp_sa
                    results[name]['err_sa']= err_sa

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
        value_range= safe_eval(meta_info['RANGE'])
        compression= str(meta_info['COMPRESSION'])
        
        #I'm not sure why, but saving the video transposes x and y
        x_pos, y_pos= coords_dims.index(x_name), coords_dims.index(y_name)
        coords_dims[x_pos], coords_dims[y_pos]= y_name, x_name
                
        #Rescale array
        if compression == 'lossy':
            array= array.astype(np.float32) / 255 * (value_range[1] - value_range[0]) + value_range[0]
        
        #Go over bands and set them into the xarray
        for i, (band, attr) in enumerate(zip(bands, attrs.values())):
            try:
                data= array[...,i] if len(array.shape) == 4 else array
                x[band]= xr.DataArray(data=data, dims=coords_dims, attrs=attr)
            except Exception as e:
                print(f'Exception processing {array_id=} {band=}: {e}')
                if exceptions == 'raise': raise e
            
    return x