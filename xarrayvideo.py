#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time

#Others
import xarray as xr, numpy as np, ffmpeg
from skimage.metrics import structural_similarity as ssim

input_pix_fmt_dict= {1:'gray', 3:'rgb24', 4:'yuva420p'}
output_pix_fmt_dict= {1:'gray', 3:'rgb24', 4:'yuva420p'}

def safe_eval(s):
    'Evaluates only simple expressions for safety over eval'
    parsed = ast.parse(s, mode='eval')
    return ast.literal_eval(parsed)

def _ffmpeg_read(video_path, loglevel='quiet'):
    #Open the video file using ffmpeg
    probe= ffmpeg.probe(video_path)
    video_info= next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    meta_info= safe_eval(probe['format']['tags']['XARRAY']) #Custom info that we have stored

    #Extract video parameters
    width = int(video_info['width'])
    height = int(video_info['height'])
    
    channels= len(safe_eval(meta_info['BANDS']))
    num_frames= int(meta_info['FRAMES'])
    # num_frames= int(video_info['nb_frames'])
    # dt=  datetime.strptime(video_info['tags']['DURATION'][:12], '%H:%M:%S.%f')
    # seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000
    # fr= eval(video_info['r_frame_rate'])
    # num_frames= int(seconds * fr)

    #Read the video frames into a numpy array
    pix_fmt=input_pix_fmt_dict[channels]
    process = (
        ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt=pix_fmt, loglevel=loglevel)
        .run_async(pipe_stdout=True)
    )

    #Reshape
    if channels == 1:   output_shape= [num_frames, height, width]
    elif channels == 3: output_shape= [num_frames, height, width, 3]
    elif channels == 4: output_shape= [num_frames, height, width, 4]
    else: raise AssertionError(f'{channels=} not in [1,3,4]')
    video_data= np.frombuffer(process.stdout.read(), np.uint8).reshape(output_shape)

    #Close the ffmpeg process
    process.stdout.close()
    process.wait()
    
    return video_data, meta_info

def _ffmpeg_write(video_path, array, output_params, loglevel='quiet', metadata={}, channels=3):
    #Define the input pipe
    pix_fmt=input_pix_fmt_dict[channels]
    input_pipe = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=pix_fmt, 
                              s=f'{array.shape[1]}x{array.shape[2]}', framerate=30)

    #Write the numpy array to the input pipe and encode it to the output video file
    params= {**output_params} #Make copy
    if metadata != {}:
        params['metadata']= f'XARRAY={metadata}'
    process= ( ffmpeg
                    .output(input_pipe, video_path, loglevel=loglevel, **params)
                    .overwrite_output()
                    .run_async(pipe_stdin=True, overwrite_output=True)
             )

    #Write the numpy array to the input pipe and close
    for frame in array:
        process.stdin.write(frame.tobytes())

    #Close the ffmpeg process
    process.stdin.close()
    process.wait()

#Bakwards function
def video2xarray(input_path, array_id, fmt='mkv'):
    #To path
    path= Path(input_path)
    
    #Load xarray
    x= xr.open_dataset(path / array_id / 'x.nc')
    
    #Go over videos, read them, and integrate them into the dataset
    for video_path in (path / array_id).glob(f'*.{fmt}'):
        
        #Read array and meta
        array, meta_info= _ffmpeg_read(video_path)
        bands= safe_eval(meta_info['BANDS'])
        coords_dims= safe_eval(meta_info['COORDS_DIMS'])
        attrs= safe_eval(meta_info['ATTRS'])
        value_range= safe_eval(meta_info['RANGE'])
        compression= str(meta_info['COMPRESSION'])
        
        #Rescale array
        if compression == 'lossy':
            array= array.astype(np.float32) / 255 * (value_range[1] - value_range[0]) + value_range[0]
        
        #Go over bands and set them into the xarray
        for i, (band, attr) in enumerate(zip(bands, attrs.values())):
            data= array[...,i] if len(array.shape) == 4 else array
            x[band]= xr.DataArray(data=data, dims=list(coords_dims), attrs=attr)
            
    return x

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
               txy_coords=('time', 'variable', 'x', 'y'),
               save_name='default.jpg', cmaps=[None], show=True, plot_idx= [[0,1,2]], 
               limits=[(0,0.3)], ylabels=['RGB']):
    
    from IPython.display import display, Image, HTML
    from txyvis import plot_maps
    import cv2
    
    txy= x[band_names].to_array().transpose(*txy_coords).values #t,x,y,c -> t,c,x,y
    
    if mask_name is None:
        masks= None
    else:
        #t,x,y -> t,1,x,y. Repat mask len(images) times
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
    
#Forward function
def xarray2video(x, array_id, conversion_rules, value_range=(0.,1.), compute_stats=False,
                 lossy_params={'c:v': 'libx264', 'r': 30, 'preset': 'slow', 'crf': 11},
                 lossless_params={'c:v': 'ffv1'}, output_path='./', fmt='mkv',
                 loglevel='quiet', use_ssim=False):
    
    array_dict= {} #Only filled if compute_stats is True
    for name, (bands, coord_names, compression) in conversion_rules.items():
        #Array -> uint8, shape: (t, x, y, (3))
        if isinstance(bands, str):
            #Get the array and permute into ordering (t, x, y, (3)), type uint8, and range (0,255)
            channels= 1
            coords= x[bands].coords.dims
            attrs= {bands:x[bands].attrs}
            array= x[bands].transpose(*coord_names).values
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
            if compute_stats: array_orig= array.copy()
            array= normalize(array, minmax=value_range)
        
        #Add custom metainfo
        metadata= {}
        metadata['BANDS']= str([bands]) if isinstance(bands, str) else str(bands)
        metadata['COORDS_DIMS']= str(coords)
        metadata['ATTRS']= str(attrs)
        metadata['FRAMES']= str(array.shape[0])
        metadata['RANGE']= str(value_range)
        metadata['COMPRESSION']= str(compression)
        
        output_path= Path(output_path)
        (output_path / array_id).mkdir(exist_ok=True, parents=True)
        output_path_video= output_path / array_id / f'{name}.{fmt}'

        #Write with ffmpeg
        t0= time.time()
        params= lossy_params if compression == 'lossy' else lossless_params
        params['pix_fmt']= output_pix_fmt_dict[channels]
        params['r']= 30
        _ffmpeg_write(str(output_path_video), array, params, loglevel=loglevel, 
                      channels=channels, metadata=metadata)
        t1= time.time()
        
        #Modify minicube to delete the bands we just processed
        x= x.drop_vars(bands)
        
        #Show stats
        if compute_stats:
            #Size and time
            array_size= array.size * array.itemsize / 2**20 #In Mb
            video_size= output_path_video.stat().st_size / 2**20 #In Mb
            percentage= (video_size / array_size)*100
            
            print(f'{name}: {array_size:.2f}Mb -> {video_size:.2f}Mb '+\
                  f'({percentage:.2f}% of original size) in {t1 - t0:.2f}s'\
                  f'\n - {params=}')
            
            #Assess read time and reconstruction quality
            t0= time.time()
            array2, metadata= _ffmpeg_read(str(output_path_video))
            t1= time.time()
            assert array2.shape == array_orig.shape, f'{array2.shape=} != {array_orig.shape=}'
            
            if compression == 'lossy':
                array2= array2.astype(np.float32) / 255 * (value_range[1] - value_range[0]) + value_range[0]
                
                # ssim_arr= ssim(array_orig, array2, channel_axis=3 if channels==3 else None)
                # print(f' - SSIM {ssim_arr:.6f}, read in {t1 - t0:.2f}s')

                array_orig2= np.copy(array_orig)
                array_orig2[array_orig2 > value_range[1]]= value_range[1]
                array_orig2[array_orig2 < value_range[0]]= value_range[0]
                
                if use_ssim:
                    ssim_arr2= ssim(array_orig2, array2, channel_axis=-1 if channels > 1 else None)
                    print(f' - SSIM_sat {ssim_arr2:.6f} (input saturated to [{value_range[0], value_range[1]}])')
                
                mse= ((array_orig2 - array2)**2).mean()
                print(f' - MSE_sat {mse:.6f} (input saturated to [{value_range[0], value_range[1]}])')

            else:
                acc= (array2==array_orig).mean()
                print(f' - acc {acc:.2f}')
                if acc != 1:
                    mse= ((array_orig2 - array2)**2).mean()
                    print(f' - MSE_sat {mse:.6f} (input saturated to [{value_range[0], value_range[1]}])')
            
            array_dict[name]= (array2, array_orig)
            
    #Save the resulting xarray
    to_netcdf(x, output_path / array_id / f'x.nc')
            
    return array_dict