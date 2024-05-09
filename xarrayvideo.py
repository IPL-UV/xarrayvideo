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
    x= xr.open_dataset(path / array_id / 'x.zarr', engine='zarr')
    
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
        
#Forward function
def xarray2video(x, array_id, conversion_rules, value_range=(0.,1.), compute_stats=False,
                 lossy_params={'c:v': 'libx264', 'r': 30, 'preset': 'slow', 'crf': 11},
                 lossless_params={'c:v': 'ffv1'}, output_path='./', fmt='mkv',
                 loglevel='quiet', use_ssim=False):
    
    def prepare_array(arr, coords_order, normalize=True):
        t_pos, x_pos, y_pos= (coords_order.index(i) for i in (t_name, x_name, y_name))
        arr= np.transpose(arr, (t_pos, x_pos, y_pos))
        if not normalize: 
            return arr
        if arr.dtype != np.uint8:
            arr= (arr - value_range[0]) / (value_range[1] - value_range[0]) * 255
            arr[arr > 255]= 255
            arr[arr < 0]= 0
            arr= arr.astype(np.uint8)
        return arr
    
    array_dict= {} #Only filled if compute_stats is True
    for name, (bands, (t_name, x_name, y_name), compression) in conversion_rules.items():
        #Array -> uint8, shape: (t, x, y, (3))
        if isinstance(bands, str):
            #Get the array and permute into ordering (t, x, y, (3)), type uint8, and range (0,255)
            channels= 1
            coords= x[bands].coords.dims
            attrs= {bands:x[bands].attrs}
            
            if compute_stats: 
                array_orig= prepare_array(x[bands].values, coords, normalize=False)
            array= prepare_array(x[bands].values, coords)
            if compression != 'lossless': 
                print('Warning: as of now, 1-channel compression takes ~3x as much space as it should'
                      ' due to gray pix_fmt not being respected by x264, and hence rgb being used.')
        else:
            assert len(bands) in [3,4], f'For {name=} expected to find 3 or 4 bands, found {bands=}'
            channels= len(bands)
            assert channels != 4, f'Warning: {channels=} is not currently supported. '+\
                'It should be possible with c:v=vp9 and format=yuva420p, but it is still using yuv420p '+\
                'and generating 1.6x as many frames... as requested'
            coords= x[bands[0]].coords.dims
            attrs= {b:x[b].attrs for b in bands}
            
            if compute_stats: 
                array_orig= np.stack([prepare_array(x[b].values, coords, normalize=False)
                                      for b in bands], axis=-1)
            array= np.stack([prepare_array(x[b].values, coords) for b in bands], axis=-1)
        
        #Add metainfo
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
    x.to_zarr(output_path / array_id / f'x.zarr')
            
    return array_dict