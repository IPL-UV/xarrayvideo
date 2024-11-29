#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, time, warnings
from typing import List, Optional
from collections import defaultdict

#Others
import xarray as xr, numpy as np, ffmpeg

#Own lib
from .utils import (safe_eval, to_netcdf, normalize, denormalize, 
                    detect_planar, detect_rgb, reorder_coords_axis,
                    np2str, str2np, DRWrapper, reorder_coords, is_float, SEED)
from .ffmpeg_wrappers import _ffmpeg_read, _ffmpeg_write
from .plot import plot_pair
from .metrics import SA, SNR, SSIM

#Globals
#Codecs currently supported
VIDEO_CODECS= ['libx264', 'libx265', 'vp9', 'ffv1', 'hevc_nvenc'] #ffmpeg
IMAGE_CODECS= ['JP2OpenJPEG'] #gdal
EXTENSIONS= ['.mkv', '.jp2']
TRUTHY= (1, '1', 'true', True, 'True', 'YES', 'yes', 'TRUE')
METRICS_MAX_N= 1e8 #Use sampling for metric computation if N_elements is above this number

def get_file_fmt(params):
    'Infer optimal file extension from codec name'
    if 'c:v' in params.keys() and params['c:v'] in VIDEO_CODECS:
        return '.mkv'
    elif 'codec' in params.keys() and params['codec'] in IMAGE_CODECS:
        return '.jp2'
    # elif params['codec'] == 'jpegxl':
    #     return '.jxl'
    else:
        raise AssertionError(f'Only codecs {VIDEO_CODECS} ("c:v" param) and '
                             f'{IMAGE_CODECS} ("codec" param) are implemented') 
    
def get_compression(params):
    'Infer compression type (lossy / lossless) from codec name'
    is_video_lossless= lambda p: 'lossless' in p.keys() and p['lossless'] in TRUTHY
    is_image_lossless= lambda p: \
        'QUALITY' in p.keys() and p['QUALITY']=='100' and\
        'REVERSIBLE' in p.keys() and p['REVERSIBLE']=='YES' and\
        'YCBCR420' in p.keys() and p['YCBCR420']=='NO'
    
    if 'c:v' in params.keys() and params['c:v'] in ['libx264']:
        return 'lossy'
    if 'c:v' in params.keys() and params['c:v'] in ['libx265', 'hevc_nvenc']:
        return 'lossy' if 'x265-params' in params.keys() and \
               'lossless=1' in str(params['x265-params']) else 'lossy'
    elif 'c:v' in params.keys() and params['c:v'] in ['vp9']:
        return 'lossless' if is_video_lossless(params) else 'lossy'
    elif 'c:v' in params.keys() and params['c:v'] in ['ffv1']:
        return 'lossless'
    elif 'codec' in params.keys() and params['codec'] in ['JP2OpenJPEG']:
        return 'lossless' if is_image_lossless(params) else 'lossy'
    else:
        raise AssertionError(f'Only codecs {VIDEO_CODECS} ("c:v" param) and '
                             f'{IMAGE_CODECS} ("codec" param) are implemented') 

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
    if params['c:v'] in ['libx264', 'libx265', 'vp9', 'hevc_nvenc']:
        input_pix_fmt= 'gbrp' #Input / Output
        req_pix_fmt= 'yuv444p' #Video 
        
        if bits == 8: pass
        elif (bits == 10 and params['c:v'] in ['libx264', 'libx265', 'vp9'] or
              bits == 12 and params['c:v'] in ['libx265', 'vp9'] or
              bits == 16 and params['c:v'] in ['hevc_nvenc']):
            input_pix_fmt+= f'{bits}{endianness}'
            req_pix_fmt+= f'{bits}{endianness}'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
    elif params['c:v'] in ['ffv1']: #Lossless
        #For ffv1: many options supported
        req_pix_fmt= {1:'gray', 3:'bgr0', 4:'bgra'}[channels] #Video
        if bits == 8:
            input_pix_fmt= {1:'gray', 3:'gbrp', 4:'bgra'}[channels] #Input
        elif bits in [10,12,16]: #gbrp supports more n of bits, but gray and gbrap do not
            input_pix_fmt= {1:'gray', 3:'gbrp', 4:'gbrap'}[channels] #Input
            input_pix_fmt+= f'{bits}{endianness}'
            req_pix_fmt+= f'{bits}{endianness}'
        else:
            raise AssertionError(f'For {params["c:v"]=}, {bits=} not supported')
    else:
        raise AssertionError(f'Only codecs {VIDEO_CODECS} are implemented')
        
    return input_pix_fmt, req_pix_fmt

def is_image_sequence(params):
    return 'c:v' not in params.keys() and 'codec' in params.keys() \
           and params['codec'] in IMAGE_CODECS

def get_param(possibly_list, position):
    if isinstance(possibly_list, list):
        if position < len(possibly_list): 
            return possibly_list[position]
        else: 
            return possibly_list[-1]
    else:
        return possibly_list
        
#Forward function
def xarray2video(x, array_id, conversion_rules, compute_stats=False, include_data_in_stats=False,
                 output_path='./', fmt='auto', loglevel='quiet', exceptions='raise', 
                 verbose=True, nan_fill=None, all_zeros_is_nan=True, save_dataset=True):
    '''
        Takes an xarray Dataset as input, and creates an xarray dataset as output, where some
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
            #Array -> uint8 or uint16, shape: (t, y, x, c)
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
                            
            #Check if we are encoding a video or an image sequence
            is_sequence= is_image_sequence(params)
            if is_sequence: #We only import gdal if needed
                from .gdal_wrappers import _gdal_read, _gdal_write
            
            #Use PCA?
            use_pca= (n_components > 0)# and not is_sequence
            if use_pca:
                #Initialize classes, fit transforms
                DR= DRWrapper(n_components=n_components)
                array= DR.fit_transform(array)
                print('Explained variance:', [f"{v*100:.4f}%" for v in DR.dr.explained_variance_ratio_])
                
            #Store all channels in sets of 3
            #TODO: even lossless compression (which supports 1,3,4 channels) is stored in sets of 3 channels
            if is_sequence:
                repeats= None
                video_files= 1
            else:
                repeats= 3 - (array.shape[-1] % 3) #How many channels we need to repeat
                if repeats!=3:
                    array= np.concatenate([array] + [array[...,[-1]*repeats]], axis=-1)
                    bands= list(bands) + ([bands[-1]]*repeats)
                    assert (array.shape[-1] % 3 == 0) and (len(bands) % 3 == 0),\
                        'Check bug in code, this should never occur'
                video_files= len(bands) // 3
                
            #Compute minimum and maximum values for every band
            if value_range is None:
                value_range= np.stack([ np.nanmin(array, axis=tuple(range(array.ndim - 1))), 
                                        np.nanmax(array, axis=tuple(range(array.ndim - 1))) ], axis=1)
            else: #Or use the provided range
                value_range= np.array([value_range]*len(bands))
            value_range= value_range.astype(np.float32)
        
            #Normalize data?
            compression= get_compression(params)
            is_int_but_does_not_fit= (array.dtype.itemsize * 8 > bits) and np.nanmax(array) > (2**bits-1)
            normalized= is_float(array) or is_int_but_does_not_fit
            if compression == 'lossless':
                if is_int_but_does_not_fit:
                    print(f'Warning: Eventhough the codec is lossless, the data has {array.dtype.itemsize*8}bits '+\
                          f'AND {np.nanmax(array)=} > {2**bits-1=}. Compression will not be lossless.')
                elif is_float(array):
                    print(f'Warning: Eventhough the codec is lossless, the data is of float type. '+\
                          'Compression will not be lossless.')
            if normalized:
                array= normalize(array, minmax=value_range, bits=bits)
                
            #Get shapes
            x_len, y_len= array.shape[1], array.shape[2]

            #Get pixel format: only some are supported
            if is_sequence:
                input_pix_fmt, req_pix_fmt= None, None
            else:
                input_pix_fmt, req_pix_fmt= get_pix_fmt(params, 3, bits)
                
            #Detect if planar format
            if is_sequence:
                planar_in= None
            else:
                planar_in= detect_planar(input_pix_fmt)
            
            #Get automatic format
            if fmt == 'auto': 
                fmt= get_file_fmt(params)
            else:
                assert fmt in EXTENSIONS, f'{fmt=} must be in {EXTENSIONS}'
                
            #Convert rgb <> gbr, etc.
            #This is only relevant for visualizing the output videos
            ordering= 'rgb'
            if not is_sequence and not 'rgb' in input_pix_fmt:
                ordering= detect_rgb(input_pix_fmt)
                if ordering is None: ordering= 'rgb'

            #Add custom metainfo used by video2xarray
            metadata= {}
            metadata['VERSION']= '0.2'
            metadata['BANDS']= str(bands) if len(coord_names) != 4 else str([base_band])
            metadata['CHANNELS']= 3 #Always 3 now
            metadata['COORDS_DIMS']= str(coord_names)
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
            results[name]= {}
            if is_sequence:
                metadata_path= output_path / array_id / f'{name}.json'
                results[name]['path']= [output_path / array_id / ('%s_{id}%s'%(name, fmt))]
            else:
                comp_names= [f'{name}_{i+1:03d}' for i in range(video_files)]
                results[name]['path']= [output_path / array_id / f'{cn}{fmt}' for cn in comp_names]
            
            #Start timer
            t0= time.time()  
            final_params= {**params} #make copy
            if not is_sequence:
                #Go over every set of 3 components
                for i, output_path_video in enumerate(results[name]['path']):
                    #Write with ffmpeg
                    final_params['pix_fmt']= req_pix_fmt
                    final_params['r']= 30
                    for k in params.keys(): #If the param is a list, index it with i
                        final_params[k]= get_param(params[k], i)
                    array_in= array[...,i*3:(i+1)*3]
                    array_in= reorder_coords_axis(array_in, list('rgb'), list(ordering), axis=-1)
                    metadata['ORDER']= i+1
                    #if verbose: print(f'{output_path_video} -> {array_in.mean()}, {array_in.max()}')
                    _ffmpeg_write(str(output_path_video), array_in, x_len, y_len, final_params, 
                        planar_in=planar_in, loglevel=loglevel, metadata=metadata, input_pix_fmt=input_pix_fmt)
            else:
                codec= final_params['codec']
                del final_params['codec']
                _gdal_write(str(results[name]['path'][0]), metadata_path, array, loglevel=loglevel,
                            codec=codec, metadata=metadata, bits=bits, params=final_params)
                
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
                video_sizes= [v.stat().st_size / 2**20 for v in (output_path / array_id).glob(f'{name}_*')]
                video_size= sum(video_sizes)
                percentage= (video_size / array_size)*100
                bpppb= video_size * 2**20 * 8 / array_orig.size #bits per pixel per band (=bps)

                print_fn(f'{name}: {array_size:.2f}Mb -> {[f"{v:.2f}" for v in video_sizes[-10:]]}Mb '\
                         f'(showing last 10)'+\
                         f'({percentage:.2f}% of original size, {bpppb:.4f} bpppb) in {t1 - t0:.2f}s'\
                         f'\n - {params=}')
                
                results[name]['original_size']= array_size
                results[name]['compressed_size']= video_size
                results[name]['compression']= video_size / array_size
                results[name]['bpppb']= bpppb
                results[name]['time']= t1 - t0

                #Read
                t0= time.time()
                if is_sequence:
                    array_comp= _gdal_read(str(results[name]['path'][0]), metadata_path, loglevel=loglevel)[0]
                else:
                    array_comp_list= [_ffmpeg_read(v)[0] for v in results[name]['path']]
                    array_comp_list= [reorder_coords_axis(a, list(ordering), list('rgb'), axis=-1) 
                                      for a in array_comp_list]
                    array_comp= np.concatenate(array_comp_list, axis=-1)
                
                    #Undo transformations
                    if repeats!= 3:
                        array_comp= array_comp[...,:-repeats]
                  
                #Undo transformations
                if use_pca: 
                    array_comp= denormalize(array_comp, minmax=value_range, bits=bits)
                    array_comp= DR.inverse_transform(array_comp)

                #Compare
                t1= time.time()
                results[name]['d_time']= t1 - t0
                print_fn(f' - Decompression time {results[name]["d_time"]:.2f}s')
                
                assert array_comp.shape == array_orig.shape, f'{array_comp.shape=} != {array_orig.shape=}'
                
                if normalized or compression == 'lossy':
                    if not use_pca and normalized: 
                        array_comp= denormalize(array_comp, minmax=value_range, bits=bits)
                    
                    #Plot last frame
                    if verbose: 
                        plot_pair(array_orig[-1, ..., -3:], array_comp[-1, ..., -3:], 
                                  max_val=value_range.max(), factor=10)
                        print(f'Saturation values per band {bands}):\n {value_range}')
                        
                    #TODO: Process the orginal array in the same way as the video, to be able to compare them
                    array_orig_sat= array_orig 

                    #Torchmetrics is MUCH quicker
                    t0= time.time()
                    try:
                        from torchmetrics.functional.image import structural_similarity_index_measure as ptSSIM
                        from torchmetrics.functional.image import peak_signal_noise_ratio as ptPSNR
                        from torchmetrics.functional.image import spectral_angle_mapper as ptSA
                        from torchmetrics.functional.regression import mean_squared_error as ptMSE
                        import torch
                        
                        #Everything to torch: torchmetrics is expecting (batch, channel, x, y). 
                        #We will use t for the batch. E.g.: txyc -> tcxy
                        comp= torch.from_numpy(np.swapaxes(array_comp.astype(np.float32), 1, -1))
                        orig= torch.from_numpy(np.swapaxes(array_orig_sat.astype(np.float32), 1, -1))
                        
                        #We get rid of any timesteps that have any nans either in orig or comp
                        valid_idx = (torch.isnan(comp) | torch.isnan(orig)).sum(axis=(1,2,3)) == 0
                        comp= comp[valid_idx]
                        orig= orig[valid_idx]
                        
                        #Sample some data?
                        if comp.numel() > METRICS_MAX_N:
                            sample_fraction= METRICS_MAX_N / comp.numel()
                            print(f'Using only {sample_fraction*100:.2f}% of data for metrics computation. '
                                  f'Adjust this by modifying global variable {METRICS_MAX_N=}')
                            torch.manual_seed(SEED)
                            ts= comp.shape[0]
                            idx= [torch.randperm(ts)[:int(ts*sample_fraction)]]
                            comp= comp[idx]
                            orig= orig[idx]
                        
                        ssim_val= ptSSIM(comp, orig).numpy().item()
                        psnr= ptPSNR(comp, orig).numpy().item()
                        eps= 1e-8 #Add small epsilon to avoid division by zero
                        exp_sa= ptSA(comp+eps, orig+eps).numpy().item()
                        mse= ptMSE(comp, orig).numpy().item()
                        
                    except Exception as e:
                        print('It is recommeneded to install the optional dependency `torchmetrics` '
                              f'for much quiecker metric computation. Exception: {e}')
                        ssim_val= SSIM(array_orig_sat, array_comp, channel_axis=-1, 
                                           data_range=value_range.max()-value_range.min())
                        snr, psnr, mse= SNR(array_orig_sat, array_comp, max_value=value_range.max())
                        exp_sa, err_sa= SA(array_orig_sat, array_comp, channel_dim=-1)
                    t1= time.time()
                    
                    print_fn(f' - SSIM_sat {ssim_val:.6f}')
                    print_fn(f' - MSE_sat {mse:.6f}')
                    print_fn(f' - PSNR_sat {psnr:.4f}')
                    print_fn(f' - Exp. SA {exp_sa:.4f} ')
                    print_fn(f'Metrics took {t1-t0:.2f}s to run')
                    
                    results[name]['ssim']= ssim_val
                    results[name]['psnr']= psnr
                    results[name]['mse']= mse
                    results[name]['exp_sa']= exp_sa

                else:
                    #Plot last frame
                    if verbose: 
                        plot_pair(array_orig[-1, ..., -3:], array_comp[-1, ..., -3:], 
                                  max_val=2**bits-1)                        
                    acc= np.nanmean(array_comp==array_orig)
                    print_fn(f' - acc {acc:.2f}')
                    results[name]['acc']= acc

                if include_data_in_stats:
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
def video2xarray(input_path, array_id, exceptions='raise'):
    #To path
    path= Path(input_path)
    
    #Load xarray
    x= xr.open_dataset(path / array_id / 'x.nc')
    
    #Check if it is a sequence of images or a video
    meta_files= list((path / array_id).glob('*.json'))
    is_sequence= len(meta_files)
    
    #Read videos / image sequences along with their metadata
    arrays, metas= defaultdict(list), defaultdict(list)
    if not is_sequence:
        for video_path in (path / array_id).glob(f'*.mkv'):
            array, meta_info= _ffmpeg_read(video_path)
            array= reorder_coords_axis(array, list(meta_info['CHANNEL_ORDER']), list('rgb'), axis=-1)
            bands_key = '_'.join(map(str, safe_eval(meta_info['BANDS']))) #E.g.: B01_B02_B03
            arrays[bands_key].append(array)
            metas[bands_key].append(meta_info)
    else:
        from .gdal_wrappers import _gdal_read
        for metadata_path in meta_files:
            images_path= path / array_id / (metadata_path.stem + '_{id}.jp2') #TODO: use arbitrary extension
            array, meta_info= _gdal_read(images_path, metadata_path)
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

def get_recipe(xarr, t='time', x='longitude', y='latitude', c='level', variables=None, bits=12):
    '''
        Returns a default recipe for compressing a whole xarray using xarray2video function
        This recipe is a dictionary that can be directly used `xarray2video(x, 'data_name', recipe)` 
        or further refined. 
        The syntax is:
        {
         'video_name_1': (
            var or (var1, var2, var3), #Variables present in the xarray
            (t_coord_name, x_coord_name, y_coord_name), #Names of the coordinates in order (t, x, y)
            [Optional] n_components: 0 or n_components 
                - 0: store all bands in sets of 3 channels
                - n_components: store all bands using n_components, which must be divisable by 3
            [Optional] compression parameters dictionary:  
               {
                'c:v': 'libx265',  #[libx264, libx265, vp9, ffv1]
                'preset': 'medium',  #Preset for quality/encoding speed tradeoff: quick, medium, slow (better)
                'crf': 3, #3 default, the lower, the higher the quality
                'x265-params': 'qpmin=0:qpmax=5:psy-rd=0:psy-rdoq=0' #qpmax controls the quality when crf is 0
               },
            [Optional] number of bits (e.g.: 8,10,12,16),
            ),
         'video_name_2': etc.
         }
    '''
    lossy_params= {
        'c:v': 'libx265', 'preset': 'medium', 'crf': [0], 
        'x265-params': 'qpmin=0:qpmax=0.01:psy-rd=0:psy-rdoq=0',
        }
    lossless_params= {
        'c:v': 'ffv1',
    }
    recipe= {}
    if variables is None: variables= xarr.variables
    
    #txy
    vars_txy= [v for v in variables if set(xarr[v].coords.dims) == {t,x,y}]
    vars_txy_lossy= [v for v in vars_txy if is_float(dtype=xarr[v].dtype)]
    if len(vars_txy_lossy): recipe['txy_lossy']= (vars_txy, (t,x,y), 0, lossy_params, bits)
    
    vars_txy_lossless= [v for v in vars_txy if not is_float(dtype=xarr[v].dtype)]
    if len(vars_txy_lossless): recipe['txy_lossless']= (vars_txy, (t,x,y), 0, lossless_params, bits)
    
    #tcxy
    vars_txyc= [v for v in variables if set(xarr[v].coords.dims) == {t,c,x,y}]
    vars_txyc_lossy= [v for v in vars_txyc if is_float(dtype=xarr[v].dtype)]
    for v in vars_txyc_lossy: 
        recipe[f'txyc_lossy_{v}']= (v, (t,x,y,c), 0, lossy_params, bits)
    
    vars_txyc_lossless= [v for v in vars_txyc if not is_float(dtype=xarr[v].dtype)]
    for v in vars_txyc_lossless: 
        recipe[f'txyc_lossless_{v}']= (v, (t,x,y,c), 0, lossless_params, bits)
    
    return recipe