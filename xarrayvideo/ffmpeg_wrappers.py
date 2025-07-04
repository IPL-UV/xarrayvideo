#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings, logging
from typing import List, Optional

#Others
import numpy as np, ffmpeg

#Own lib
from .utils import safe_eval

# Configure logger
logger = logging.getLogger(__name__)

def _ffmpeg_read(video_path, loglevel='quiet'):
    #Open the video file using ffmpeg
    probe= ffmpeg.probe(video_path)
    video_info= next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    meta_info= safe_eval(probe['format']['tags']['XARRAY']) #Custom info that we have stored
    
    # meta_info= yaml.safe_load(open(metadata_path))
    # meta_info= safe_eval(open(metadata_path).read())

    #Extract video parameters
    width= int(video_info['width'])
    height= int(video_info['height'])
    
    actual_pix_fmt= video_info['pix_fmt']
    requested_pix_fmt= meta_info['REQ_PIX_FMT']
    output_pix_fmt= meta_info['OUT_PIX_FMT']
    if actual_pix_fmt != requested_pix_fmt:
        logger.warning(f'Requested pixel format {requested_pix_fmt} is different from actual format {actual_pix_fmt}')
    
    planar_out= meta_info['PLANAR'] in [True, 'True']
    bits= int(meta_info['BITS'])
    channels= int(meta_info['CHANNELS'])
    num_frames= int(meta_info['FRAMES'])
    
    #Other ways to infer num_frames. 
    #For now we just save it as metainfo, since not all formats include it
    # num_frames= int(video_info['nb_frames'])
    # dt=  datetime.strptime(video_info['tags']['DURATION'][:12], '%H:%M:%S.%f')
    # seconds = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1000000
    # fr= eval(video_info['r_frame_rate'])
    # num_frames= int(seconds * fr)

    #Read the video frames into a numpy array
    process = ( ffmpeg
        .input(video_path)
        .output('pipe:', format='rawvideo', pix_fmt=output_pix_fmt, loglevel=loglevel)
        .run_async(pipe_stdout=True)
    )

    #Reshape
    if planar_out: 
        output_shape= [num_frames, channels, height, width]
    else:          
        output_shape= [num_frames, height, width, channels]
    
    data= np.frombuffer(process.stdout.read(), np.uint8 if bits==8 else np.uint16)
    #print(f'{video_path} -> {data.mean()}, {data.max()}')
    assert len(data) == np.prod(output_shape),\
        f'Video {len(data)=} cannot be reshaped into {output_shape=}. '\
        f'Video data is {len(data)/np.prod(output_shape):.6f}x longer'
    video_data= data.reshape(output_shape)
    
    #Close the ffmpeg process
    process.stdout.close()
    process.wait()
    
    #Convert back from planar if needed
    if planar_out:
        video_data= np.transpose(video_data, (0, 3, 2, 1)) #(t, c, y, x) > (t, x, y, c)
    else:
        video_data= np.transpose(video_data, (0, 2, 1, 3)) #(t, y, x, c) > (t, x, y, c)

    return video_data, meta_info

def _ffmpeg_write(video_path, array, x, y, output_params, planar_in=True, 
                  input_pix_fmt='gbrp', loglevel='quiet', metadata={}):
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
    #yaml.dump(metadata, open(metadata_path, 'w'))
    # with open(metadata_path, 'w') as f: f.write(metadata)
                
    #Convert to planar if needed
    if planar_in:
        array2= np.transpose(array, (0, 3, 2, 1)) #(t, x, y, c) > (t, c, y, x)
    else:
        array2= np.transpose(array, (0, 2, 1, 3)) #(t, x, y, c) > (t, y, x, c)

    #Write the numpy array to the input pipe and close
    for frame in array2:
        try:
            process.stdin.write(frame.tobytes())
        except Exception as e:
            logger.error('Exception processing frames: Try rerunning with loglevel="verbose" for better error messages.')
            raise e

    #Close the ffmpeg process
    process.stdin.close()
    process.wait()