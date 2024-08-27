#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings
from typing import List, Optional
from collections import defaultdict

#Others
import xarray as xr, numpy as np, ffmpeg

#Own lib
from .utils import (safe_eval, to_netcdf, normalize, denormalize, 
                    detect_planar, detect_rgb, reorder_coords_axis,
                    np2str, str2np, DRWrapper, reorder_coords, is_float)
from .ffmpeg_wrappers import _ffmpeg_read, _ffmpeg_write
from .plot import plot_simple
from .metrics import SA, SNR, SM_SSIM

#Globals
IMPLEMENTED_CODECS= ['libx264', 'libx265', 'vp9', 'ffv1', 'mjpeg']

def get_file_fmt(params):
    'Infer optimal file extension from codec name'
    if params['c:v'] in ['libx264', 'libx265', 'vp9', 'ffv1', 'mjpeg']:
        return '.mkv'
    # elif params['c:v'] in ['jpeg2000']:
    #     return r'_%06d.jp2'
    else:
        raise AssertionError(f'Only codecs {IMPLEMENTED_CODECS} are implemented') 
    
def get_compression(params):
    'Infer compression type (lossy / lossless) from codec name'
    is_lossless= lambda p: 'lossless' in p.keys() and p['lossless'] in (1, '1', 'true', True, 'True')
    
    if params['c:v'] in ['libx264', 'libx265', 'mjpeg']:
        return 'lossy'
    elif params['c:v'] in ['vp9']:
        return 'lossless' if is_lossless(params) else 'lossy'
    elif params['c:v'] in ['ffv1']:
        return 'lossless'
    # elif params['c:v'] in ['jpeg2000']: #'jpegxl' not supported until latest ffmpeg builds
    #     return 'lossless' if is_lossless(params) else 'lossy'
    else:
        raise AssertionError(f'Only codecs {IMPLEMENTED_CODECS} are implemented')

def get_pix_fmt(params, channels, bits):
    '''
        Get the optimal input / compression pix_fmt for a given codec and number of bits
        Also check that the requested combination is possible
    '''
    #For example, in gbrp16le, the pixel data for the green channel is stored together,
    #followed by the data for the blue channel, and then the red channel (planar arrangement)
    
    #Check endianness
    if bits != 8 and sys.byteorder != 'little':
        endianness= 'be'
        print('WARNING: System is big endian, but library was only tested in little endian systems!')
    else:
        endianness= 'le'

    #Select optimal pixel format
    if params['c:v'] in ['libx264', 'libx265', 'vp9', 'mjpeg']: #Lossy
        input_pix_fmt= 'gbrp' #Input / Output
        req_pix_fmt= 'yuv444p' #Video 
        
        if bits == 8: pass
        elif (bits == 10 and params['c:v'] in ['libx264', 'libx265', 'vp9'] or
              bits == 12 and params['c:v'] in ['libx265', 'vp9']):
            input_pix_fmt+= f'{bits}{endianness}'
            req_pix_fmt+= f'{bits}{endianness}'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
    elif params['c:v'] in ['ffv1']: #Lossless
        #For ffv1: many options supported
        req_pix_fmt= {1:'gray', 3:'yuv444p', 4:'yuva444p'}[channels] #Video
        if bits == 8:
            input_pix_fmt= {1:'gray', 3:'gbrp', 4:'bgra'}[channels] #Input
        elif bits in [10,12,16]: #gbrp supports more n of bits, but gray and gbrap do not
            input_pix_fmt= {1:'gray', 3:'gbrp', 4:'gbrap'}[channels] #Input
            input_pix_fmt+= f'{bits}{endianness}'
            req_pix_fmt+= f'{bits}{endianness}'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
#     elif params['c:v'] in ['jpeg2000']: #Lossy
#         input_pix_fmt= 'rgb' #Input / Output
#         req_pix_fmt= 'yuv444p' #Video 
        
#         if bits == 8:
#             input_pix_fmt+= '24'
#         elif bits in [10, 12, 16]:
#             input_pix_fmt+= f'{bits*3}{endianness}'
#             req_pix_fmt+= f'{bits}{endianness}'
#         else:
#             raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
    else:
        raise AssertionError(f'Only codecs {IMPLEMENTED_CODECS} are implemented')
        
    return input_pix_fmt, req_pix_fmt

def get_param(possibly_list, position):
    if isinstance(possibly_list, list):
        if position < len(possibly_list): 
            return possibly_list[position]
        else: 
            return possibly_list[-1]
    else:
        return possibly_list
        
#Forward function
def xarray2video(x, array_id, conversion_rules, compute_stats=False,
                 output_path='./', fmt='auto', loglevel='quiet', use_ssim=False, exceptions='raise', 
                 verbose=True, nan_fill=None, all_zeros_is_nan=True, save_dataset=True):
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
        n_components= 'all'
        value_range= None #By default, compute it as [min, max] for every variable
        params= {
            'c:v': 'libx265',  #[libx264, libx265, vp9, ffv1]
            'preset': 'medium',  #Preset for quality/encoding speed tradeoff: quick, medium, slow (better)
            'crf': 3, #14 default, 11 for higher quality and size
            }
        if len(config) == 6:
            bands, coord_names, n_components, params, bits, value_range= config
        elif len(config) == 5:
            bands, coord_names, n_components, params, bits= config
        elif len(config) == 4:
            bands, coord_names, n_components, params= config
            bits= 8 #Best support, not best choice for technical images
        elif len(config) == 3:
            bands, coord_names, n_components= config
        else:
            raise AssertionError(f'Params: {config} should be: bands, coord_names, '
                                 f'[n_components], [params], [bits], [min, max]')
            
        try:
            #Array -> uint8 or uint16, shape: (t, x, y, c)
            if len(coord_names) == 3:
                if isinstance(bands, str): bands= [bands]
                coords= x[bands[0]].coords.dims
                attrs= {b:x[b].attrs for b in bands}
                # array= np.stack([x[b].transpose(*coord_names).values for b in bands], axis=-1)
                #Equivalent but quicker
                array= np.stack([reorder_coords(x[b].values, coords, coord_names) for b in bands], axis=-1)
            elif len(coord_names) == 4: #The bands are already in the xarray itself
                base_band= bands
                coords= x[base_band].coords.dims
                bands= [f'{bands}_{coord_names[-1]}_{v}' for v in x[coord_names[-1]].values]
                attrs= {base_band:x[base_band].attrs}
                # array= x[base_band].transpose(*coord_names).values
                array= reorder_coords(x[base_band].values, coords, coord_names) #Equivalent but quicker
            else:
                raise AssertionError(f'{len(coord_names)=} not in [3,4]')
                
            #Save a copy for comparison if doing compute_stats
            if compute_stats: array_orig= array.copy()
            
            #If all_zeros_is_nan, then mark timeteps with all_zeros as nan
            if all_zeros_is_nan and is_float(array):
                array[~np.any(array, axis=(1,2,3))]= np.nan
            
            #Nanfill?
            if nan_fill is not None:
                array_nans= np.isnan(array)
                if isinstance(nan_fill, int): array[array_nans]= nan_fill
                elif nan_fill == 'mean': array[array_nans]= np.nanmean(array)
                elif nan_fill == 'min': array[array_nans]= np.nanmin(array)
                elif nan_fill == 'max': array[array_nans]= np.nanmax(array)
                else: raise AssertionError(f'{nan_fill=}?')
            
            #Use PCA?
            use_pca= n_components > 0
            if use_pca:
                #Initialize classes, fit transforms
                DR= DRWrapper(n_components=n_components)
                array= DR.fit_transform(array)
                print('Explained variance:', [f"{v*100:.4f}%" for v in DR.dr.explained_variance_ratio_])
                
            #Get compression depending on codec's name
            compression= get_compression(params)
                
            #Store all channels in sets of 3
            #TODO: for now, even lossless compression (which supports 1,3,4 channels) is
            #being stored in groups of 3 channels
            repeats= 3 - (array.shape[-1] % 3) #How many channels we need to repeat
            if repeats!=3:
                array= np.concatenate([array] + [array[...,[-1]*repeats]], axis=-1)
                bands= list(bands) + ([bands[-1]]*repeats)
                assert (array.shape[-1] % 3 == 0) and (len(bands) % 3 == 0),\
                    'Check bug in code, this should never occur'
                
            #Compute minimum and maximum values for every band
            video_files= len(bands) // 3
            if value_range is None:
                value_range= np.stack([ np.nanmin(array, axis=tuple(range(array.ndim - 1))), 
                                        np.nanmax(array, axis=tuple(range(array.ndim - 1))) ], axis=1)
            else: #Or use the provided range
                value_range= np.array([value_range]*len(bands))
        
            #Normalize?
            normalized= compression == 'lossy' or is_float(array_orig)
            if normalized:
                array= normalize(array, minmax=value_range, bits=bits)
                
            #Get shapes
            x_len, y_len= array.shape[1], array.shape[2]

            #Get pixel format: only some are supported
            input_pix_fmt, req_pix_fmt= get_pix_fmt(params, 3, bits)
                
            #Deted if planar format
            planar_in= detect_planar(input_pix_fmt)
            
            #Get automatic format
            if fmt == 'auto': fmt= get_file_fmt(params)
                
            #Convert rgb <> gbr, etc.
            #This is only relevant for visualizing the output videos
            ordering= 'rgb'
            if not 'rgb' in input_pix_fmt:
                ordering= detect_rgb(input_pix_fmt)
                if ordering is None: ordering= 'rgb'

            #Add custom metainfo used by video2xarray
            metadata= {}
            metadata['VERSION']= '0.1'
            metadata['BANDS']= str(bands) if len(coord_names) != 4 else str([base_band])
            metadata['CHANNELS']= 3 #Always 3 now
            metadata['COORDS_DIMS']= str(coords)
            metadata['ATTRS']= str(attrs)
            metadata['FRAMES']= str(array.shape[0])
            metadata['RANGE']= np2str(value_range)
            metadata['OUT_PIX_FMT']= input_pix_fmt
            metadata['REQ_PIX_FMT']= req_pix_fmt
            metadata['PLANAR']= planar_in
            metadata['BITS']= bits
            metadata['NORMALIZED']= normalized
            metadata['CHANNEL_ORDER']= ordering #e.g. gbr
            metadata['PCA_PARAMS']= DR.get_params_str() if use_pca else None

            #Pathing
            output_path= Path(output_path)
            (output_path / array_id).mkdir(exist_ok=True, parents=True)
            comp_names= [f'{name}_{i+1:03d}' for i in range(video_files)]
            results[name]= {}
            results[name]['path']= [output_path / array_id / f'{cn}{fmt}' for cn in comp_names]
            
            #Go over every set of 3 components
            t0= time.time()         
            final_params= {**params} #make copy
            for i, output_path_video in enumerate(results[name]['path']):
                #Write with ffmpeg
                final_params['pix_fmt']= req_pix_fmt
                final_params['r']= 30
                for k in params.keys(): #If the param is a list, index it with i
                    final_params[k]= get_param(params[k], i)
                array_in= array[...,i*3:(i+1)*3]
                array_in= reorder_coords_axis(array_in, list('rgb'), list(ordering), axis=-1)
                metadata['ORDER']= i+1
                _ffmpeg_write(str(output_path_video), array_in, x_len, y_len, final_params, planar_in=planar_in,
                              loglevel=loglevel, metadata=metadata, input_pix_fmt=input_pix_fmt)
                
            #Modify minicube to delete the bands we just processed
            if save_dataset:
                if len(coord_names) == 3:
                    x= x.drop_vars(bands)
                elif len(coord_names) == 4:
                    x= x.drop_vars([base_band])
                    
            #Stop timer
            t1= time.time()

            #Show stats
            if compute_stats:                
                #Size and time
                array_size= array_orig.size * array_orig.itemsize / 2**20 #In Mb (use array_orig dtype)
                video_sizes= [v.stat().st_size / 2**20 for v in results[name]['path']] #In Mb
                video_size= sum(video_sizes)
                percentage= (video_size / array_size)*100
                bpppb= video_size * 2**20 * 8 / array_orig.size #bits per pixel per band (=bps)

                print_fn(f'{name}: {array_size:.2f}Mb -> {[f"{v:.2f}" for v in video_sizes]}Mb '+\
                         f'({percentage:.2f}% of original size, {bpppb:.4f} bpppb) in {t1 - t0:.2f}s'\
                        f'\n - {params=}')
                
                results[name]['original_size']= array_size
                results[name]['compressed_size']= video_size
                results[name]['compression']= video_size / array_size
                results[name]['bpppb']= bpppb
                results[name]['time']= t1 - t0

                #Read
                t0= time.time()
                array_comp_list= [_ffmpeg_read(v)[0] for v in results[name]['path']]
                array_comp_list= [reorder_coords_axis(a, list(ordering), list('rgb'), axis=-1) 
                                  for a in array_comp_list]
                array_comp= np.concatenate(array_comp_list, axis=-1)
                
                #Undo transformations
                if repeats!= 3:
                    array_comp= array_comp[...,:-repeats]
                if use_pca: 
                    array_comp= denormalize(array_comp, minmax=value_range, bits=bits)
                    array_comp= DR.inverse_transform(array_comp)
                
                #Compare
                t1= time.time()
                results[name]['d_time']= t1 - t0
                print_fn(f' - Decompression time {results[name]["d_time"]:.2f}s')
                
                assert array_comp.shape == array_orig.shape, f'{array_comp.shape=} != {array_orig.shape=}'
                
                if normalized:
                    if not use_pca: 
                        array_comp= denormalize(array_comp, minmax=value_range, bits=bits)
                    
                    #Plot last frame
                    if verbose: 
                        plot_simple(array_orig[-1, ..., -3:], max_val=value_range.max(), factor=10, title='Original')
                        plot_simple(array_comp[-1, ..., -3:], max_val=value_range.max(), factor=10, title='Compressed')
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

                    #Setting levels=1, we just do standard SSIM
                    if use_ssim: ssim_val= SM_SSIM(array_orig_sat, array_comp, channel_axis=-1, 
                                                   data_range=value_range.max()-value_range.min(), levels=1)
                    snr, psnr, mse= SNR(array_orig_sat, array_comp, max_value=value_range.max())
                    exp_sa, err_sa= SA(array_orig_sat, array_comp, channel_dim=-1)
                    
                    if use_ssim: print_fn(f' - SSIM_sat {ssim_val:.6f} (input saturated)')
                    print_fn(f' - MSE_sat {mse:.6f} (input saturated)')
                    print_fn(f' - SNR_sat {snr:.4f} (input saturated)')
                    print_fn(f' - PSNR_sat {psnr:.4f} (input saturated)')
                    print_fn(f' - Exp. SA {exp_sa:.4f} (input saturated)')
                    
                    if use_ssim: results[name]['ssim']= ssim_val
                    results[name]['snr']= snr
                    results[name]['psnr']= psnr
                    results[name]['mse']= mse
                    results[name]['exp_sa']= exp_sa

                else:
                    #Plot last frame
                    if verbose: 
                        plot_simple(array_orig[-1, ..., -3:], max_val=2**bits-1, title='Original')
                        plot_simple(array_comp[-1, ..., -3:], max_val=2**bits-1, title='Compressed')
                        
                    acc= np.nanmean(array_comp==array_orig)
                    print_fn(f' - acc {acc:.2f}')
                    results[name]['acc']= acc

                results[name]['compressed']= array_comp
                results[name]['original']= array_orig
                
        except Exception as e:
            print(f'Exception processing {array_id=} {name=}: {e}')
            if exceptions == 'raise': raise e

    #Save the resulting xarray
    if save_dataset:
        results['path']= output_path / array_id / 'x.nc'
        to_netcdf(x, results['path'])
            
    return results

#Backwards function
def video2xarray(input_path, array_id, fmt='auto', exceptions='raise', x_name='x', y_name='y'):
    #To path
    path= Path(input_path)
    
    #Get automatic file format
    if fmt == 'auto': fmt= get_file_fmt(params)
    
    #Load xarray
    x= xr.open_dataset(path / array_id / 'x.nc')
    
    #Read videos
    arrays, metas= defaultdict(list), defaultdict(list)
    for video_path in (path / array_id).glob(f'*{fmt}'):
        array, meta_info= _ffmpeg_read(video_path)
        array= reorder_coords_axis(array, list(meta_info['CHANNEL_ORDER']), list('rgb'), axis=-1)
        bands_key= '_'.join(safe_eval(meta_info['BANDS'])) #E.g.: B01_B02_B03
        arrays[bands_key].append(array)
        metas[bands_key].append(meta_info)
        
    #Put the arrays into the Dataset
    for bands_key in arrays.keys():
        array_list, meta_list= arrays[bands_key], metas[bands_key]
        
        if len(meta_list) > 1:
            #First, we order the arrays using the meta_info of each one
            sort_keys= [int(m['ORDER'])-1 for m in meta_list] #e.g.: [1,3,0,2]
            array_list= [array_list[i] for i in sort_keys]
            
            #We concatenate the final array
            array= np.concatenate(array_list, axis=-1)
        else:
            array= array_list[0]
            
        #Process meta from str
        meta_info= meta_list[0]
        bands= safe_eval(meta_info['BANDS'])
        coords_dims= list(safe_eval(meta_info['COORDS_DIMS']))
        attrs= safe_eval(meta_info['ATTRS'])
        value_range= str2np(meta_info['RANGE'])
        normalized= meta_info['NORMALIZED'] in [True, 'True']
        bits= int(meta_info['BITS'])
        dr_params_str= meta_info['PCA_PARAMS']
        use_pca= dr_params_str not in ['None', None]

        #TODO: I'm not yet sure why, but saving the video transposes x and y
        x_pos, y_pos= coords_dims.index(x_name), coords_dims.index(y_name)
        coords_dims[x_pos], coords_dims[y_pos]= y_name, x_name

        #Rescale array
        if normalized:
            array= denormalize(array, minmax=value_range, bits=bits)
            
        #AFTER denormalizing, we revert PCA if it was used
        if use_pca: 
            DR= DRWrapper(params=dr_params_str)
            array= DR.inverse_transform(array)

        #Go over bands and set them into the xarray
        for i, (band, attr) in enumerate(zip(bands, attrs.values())):
            try:
                x[band]= xr.DataArray(data=array[...,i], dims=coords_dims, attrs=attr)
            except Exception as e:
                print(f'Exception processing {array_id=} {band=}: {e}')
                if exceptions == 'raise': raise e
            
    return x