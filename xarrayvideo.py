#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings

#Others
import xarray as xr, numpy as np, ffmpeg
from skimage.metrics import structural_similarity as ssim

def safe_eval(s):
    'Evaluates only simple expressions for safety over eval'
    try:
        #literal_eval does not support having nan as a value
        parsed= ast.parse(s.replace('nan', 'None'), mode='eval')
        evaluated= ast.literal_eval(parsed)
    except Exception as e:
        print(f'Exception evaulating {s=}: {e}')
    return evaluated

def _ffmpeg_read(video_path, loglevel='quiet'):
    #Open the video file using ffmpeg
    probe= ffmpeg.probe(video_path)
    video_info= next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    meta_info= safe_eval(probe['format']['tags']['XARRAY']) #Custom info that we have stored

    #Extract video parameters
    width= int(video_info['width'])
    height= int(video_info['height'])
    
    actual_pix_fmt= video_info['pix_fmt']
    requested_pix_fmt= meta_info['REQ_PIX_FMT']
    output_pix_fmt= meta_info['OUT_PIX_FMT']
    if actual_pix_fmt != requested_pix_fmt:
        print(f'Warning: {requested_pix_fmt=} is different from {actual_pix_fmt=}')
    
    channels= len(safe_eval(meta_info['BANDS']))
    num_frames= int(meta_info['FRAMES'])
    
    #Other ways to infer num_frames. 
    #For now we just save it as metainfo, since not all formats include it
    # num_frames= int(video_info['nb_frames'])
    # dt=  datetime.strptime(video_info['tags']['DURATION'][:12], '%H:%M:%S.%f')
    # seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000
    # fr= eval(video_info['r_frame_rate'])
    # num_frames= int(seconds * fr)

    #Read the video frames into a numpy array
    process = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt=output_pix_fmt, loglevel=loglevel)
        .run_async(pipe_stdout=True)
    )

    #Reshape
    if channels == 1:   output_shape= [num_frames, height, width]
    elif channels == 3: output_shape= [num_frames, height, width, 3]
    elif channels == 4: output_shape= [num_frames, height, width, 4]
    else: raise AssertionError(f'{channels=} not in [1,3,4]')
    data= np.frombuffer(process.stdout.read(), np.uint8)
    assert len(data) == np.prod(output_shape),\
        f'Video {len(data)=} cannot be reshaped into {output_shape=}. '\
        f'Video data is {len(data)/np.prod(output_shape)}x longer'
    video_data= data.reshape(output_shape)

    #Close the ffmpeg process
    process.stdout.close()
    process.wait()
    
    return video_data, meta_info

def _ffmpeg_write(video_path, frame_list, x, y, output_params, input_pix_fmt='gbrp', 
                  loglevel='quiet', metadata={}):
    #Define the input pipe
    input_pipe= ffmpeg.input('pipe:', format='rawvideo', pix_fmt=input_pix_fmt, 
                              s=f'{x}x{y}', framerate=30)

    #Write the numpy array to the input pipe and encode it to the output video file
    params= {**output_params} #Make copy
    if metadata != {}:
        params['metadata']= f'XARRAY={metadata}'
    process= ( ffmpeg
                    .output(input_pipe, video_path, loglevel=loglevel, **params)
                    .overwrite_output()
                    .run_async(pipe_stdin=True, overwrite_output=True)
             )

    #Write the numpy frame_list to the input pipe and close
    for frame in frame_list:
        process.stdin.write(frame.tobytes())

    #Close the ffmpeg process
    process.stdin.close()
    process.wait()

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

def plot_image(x, band_names, mask_name='cloudmask_en', t_name='time', 
               txy_coords=('time', 'variable', 'y', 'x'),
               save_name='default.jpg', cmaps=[None], show=True, plot_idx= [[0,1,2]], 
               limits=[(0,0.3)], ylabels=['RGB']):
    
    from IPython.display import display, Image, HTML
    from txyvis import plot_maps
    import cv2
    
    txy= x[band_names].to_array().transpose(*txy_coords).values #t,y,x,c -> t,c,y,x
    
    if mask_name is None:
        masks= None
    else:
        #t,y,x -> t,1,y,x. Repat mask len(images) times
        masks= [x[[mask_name]].to_array().transpose(*txy_coords).values > 0]*len(plot_idx)
        
    try:
        xlabels= list(map(lambda i_d: f'{i_d[0]} {np.datetime_as_string((i_d[1]), unit="D")}',
                          enumerate(x[t_name].values)))
    except:
        xlabels= np.arange(len(x.shape[1]))
                
    img_comp= plot_maps(
            images=[txy[:,idx] for idx in plot_idx], 
            masks=masks,
            cmaps=cmaps, 
            limits= [(0,0.3)],
            ylabels=ylabels, 
            xlabels=xlabels,
            mask_kwargs=dict(colors= {0:None, 1:'r'}), classes= {1:'Invalid'},
            title=save_name, 
            backend='numpy',
            numpy_backend_kwargs={'size':13, 'color':'black', 'xstep':4,
                                  'labels':'grid', 'font':'OpenSans_Condensed-Regular.ttf'},
            plot_mask_channel=None, matplotlib_backend_kwargs={'text_size':20},
            figsize=(27.5,10),
            stack_every=73, #Stack every year (approx.) 73*5=365
                  )
    
    Path(save_name).parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(save_name, img_comp[...,[2,1,0]])
    if show:
        display(HTML(f'<h4>{Path(save_name).stem}</h4>'))
        display(Image(save_name))
        
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

from typing import List, Optional

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

import matplotlib.pyplot as plt
def plot_simple(img, factor=3, transpose=None, title=None, figsize=(4,4)):
    plt.figure(figsize=figsize)
    max_val= 255 if img.dtype == np.uint8 else 1.
    if transpose is not None:
        plt.imshow(np.minimum(np.transpose(img, transpose)*factor, max_val)) 
    else:
        plt.imshow(np.minimum(img*factor, max_val))
    plt.grid(False)
    plt.axis('off')
    if title: plt.title(title)
    plt.show()

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
                assert channels != 4, f'Warning: {channels=} is not currently supported. '+\
                    'It should be possible with c:v=vp9 and format=yuva420p, but it is still using yuv420p '+\
                    'and generating 1.6x as many frames as requested...'
                coords= x[bands[0]].coords.dims
                attrs= {b:x[b].attrs for b in bands}
                array= np.stack([x[b].transpose(*coord_names).values for b in bands], axis=-1)
                dtype= array.dtype
                if compute_stats: array_orig= array.copy()
                array= normalize(array, minmax=value_range)

            #Update params
            params= lossy_params if compression == 'lossy' else lossless_params
            params['pix_fmt']= req_pix_fmt
            params['r']= 30
                
            #Choose pixel format
            if params['c:v'] == 'libx264':
                #For x264: no alpha support, gray is somehow NOT working
                input_pix_fmt= {1:'gray', 3:'rgb24'}[channels] #Input
                req_pix_fmt= {1:'gray', 3:'yuv444p'}[channels] #Video
                out_pix_fmt= {1:'gray', 3:'rgb24'}[channels] #Output
                
            elif params['c:v'] == 'libx265':
                #For x265: generally worse than x264
                input_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Input
                req_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Video
                out_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Output
            elif params['c:v'] == 'vp9':
                #For vp9
                input_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Input
                req_pix_fmt= {1:'gray', 3:'gbrp', 4:'yuva420p'}[channels] #Video
                out_pix_fmt= {1:'gray', 3:'gbrp'}[channels] #Output
            else:
                assert compression == 'lossless', \
                    f'For lossy compression only [libx264, libx265, vp9] are supported'
                
            #Add custom metainfo
            metadata= {}
            metadata['BANDS']= str([bands]) if isinstance(bands, str) else str(bands)
            metadata['COORDS_DIMS']= str(coords)
            metadata['ATTRS']= str(attrs)
            metadata['FRAMES']= str(array.shape[0])
            metadata['RANGE']= str(value_range)
            metadata['COMPRESSION']= str(compression)
            metadata['OUT_PIX_FMT']= out_pix_fmt
            metadata['REQ_PIX_FMT']= req_pix_fmt

            output_path= Path(output_path)
            (output_path / array_id).mkdir(exist_ok=True, parents=True)
            output_path_video= output_path / array_id / f'{name}.{fmt}'
            results[name]= {}
            results[name]['path']= output_path_video

            #Write with ffmpeg
            t0= time.time()
            frame_list= [f for f in array]
            _ffmpeg_write(str(output_path_video), frame_list, 
                          array.shape[1], array.shape[2], params, loglevel=loglevel, 
                          metadata=metadata, input_pix_fmt=input_pix_fmt)
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