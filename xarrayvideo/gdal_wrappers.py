#Python std
from datetime import datetime
from pathlib import Path
import ast, sys, os, yaml, time, warnings, glob
from typing import List, Optional

#Others
import numpy as np
from osgeo import gdal, gdal_array
from tqdm.auto import trange, tqdm

def _gdal_read(input_path, metadata_path):
    """
    Reads a sequence of images from disk and their corresponding metadata, then constructs
    a numpy array representing the video data. The images are assumed to be stored with a format
    that includes a sequence number in the filename.

    Parameters:
        input_path (str): Path template for input image files. Should include a placeholder
                          for the sequence number, e.g., './image_{id}.tif'.
        metadata_path (str): Path to the metadata file (YAML format). This file should
                             contain metadata about the images such as bit depth, number of
                             channels, and number of frames.

    Returns:
        tuple: A tuple containing:
            - video_data (numpy.ndarray): A 4D numpy array of shape (T, H, W, C) where T is the
              number of frames, H is the height, W is the width, and C is the number of channels.
            - meta_info (dict): A dictionary containing metadata read from the YAML file.
    
    Raises:
        RuntimeError: If an image file cannot be opened or if any issues occur during processing.
    """
    #Read the meta information
    meta_info= yaml.safe_load(open(metadata_path))
    # meta_info= safe_eval(open(metadata_path).read())

    #Extract parameters
    # bits= int(meta_info['BITS'])
    # channels= int(meta_info['CHANNELS'])
    # num_frames= int(meta_info['FRAMES'])

    #Read the images into a numpy array
    file_list= sorted(glob.glob(str(input_path).format(id='*')))
    images= []
    for i, file_name in tqdm(enumerate(file_list), total=len(file_list), desc=f'Reading {input_path}'):
        dataset= gdal.Open(file_name)
        # if i == 0:
        #     width = dataset.RasterXSize
        #     height = dataset.RasterYSize
        #     pixel_type = band.DataType
        if not dataset: raise RuntimeError(f'Failed to open file {file_name}')
        bands= [dataset.GetRasterBand(band + 1).ReadAsArray() for band in range(dataset.RasterCount)]
        image_data= np.stack(bands, axis=-1)

        images.append(image_data)
        del dataset  #Close the file
    video_data= np.stack(images, axis=0)

    return video_data, meta_info

def _gdal_write(output_path, metadata_path, array, codec='JP2OpenJPEG', metadata={}, 
                bits=16, params={}):
    """
    Write a numpy array to a sequence of JPEG2000 / JPEGXL files with metadata.

    Parameters:
        output_path (str): Path template for output files.
        metadata_path (str): Path for metadata file.
        array (numpy.ndarray): The numpy array to write.
        codec (str): Codec to use (e.g., 'JP2OpenJPEG').
        metadata (dict): Metadata to write to a YAML file.
        bits (int): Number of bits per pixel (8 or 16).
        params (dict): Additional parameters for the JPEG2000 codec.
    """
    # Write the metadata
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    # Create a temporary TIFF driver
    temp_driver = gdal.GetDriverByName("GTiff")
    if not temp_driver:
        raise RuntimeError("GTiff driver is not available.")

    # Create a JPEG2000 driver
    jpeg2000_driver = gdal.GetDriverByName(codec)
    if not jpeg2000_driver:
        raise RuntimeError(f'{codec} driver is not available.')

    # Define the data type based on the bit depth
    data_type = {8: gdal.GDT_Byte, 16: gdal.GDT_UInt16}[bits]

    for i in trange(array.shape[0], desc=f'Writing {output_path}'):
        # Create a temporary TIFF dataset in-memory
        temp_path = "/vsimem/temp.tif"  # Use in-memory file system to avoid disk I/O
        temp_dataset = temp_driver.Create(
            temp_path, 
            array.shape[2], 
            array.shape[1], 
            array.shape[3], 
            data_type
        )
        if temp_dataset is None:
            raise RuntimeError(f"Failed to create temporary dataset at {temp_path}")

        # Write array data to the temporary dataset
        for band in range(array.shape[3]):
            temp_dataset.GetRasterBand(band + 1).WriteArray(array[i, :, :, band])
        
        # Flush and close the temporary dataset
        temp_dataset.FlushCache()
        del temp_dataset
        
        # Define options for JPEG2000 compression
        options_list = [f'{key}={value}' for key, value in params.items()]
        
        # Create the JPEG2000 file using CreateCopy
        jpeg2000_dataset = jpeg2000_driver.CreateCopy(
            output_path.format(id=f'{i:06d}'), 
            gdal.Open(temp_path), 
            options=options_list
        )
        if jpeg2000_dataset is None:
            raise RuntimeError(f"Failed to create JPEG2000 file at {output_path.format(id=f'{i:06d}')}")
        
        # Close the JPEG2000 file
        del jpeg2000_dataset
        
    #Clean up aux files
    for file in Path(output_path).parent.glob('*.aux.xml'): file.unlink()