# xarrayvideo

Save multichannel data from xarray datasets as videos to save up massive amounts of space (e.g. 20-50x compression) with minimal quality loss.

This library provides two functions: `xarray2video` to encode some `xarray` variables into videos, and `video2xarray` to rebuild the xarray from the videos. It can encode lossily or losslessly, supporting all video formats supported by ffmpeg (e.g., x265, vp9) at different bit depths (e.g., 8,10,12,16) as well as all GDAL's supported image formats (e.g. JPEG2000) for frame-by-frame encoding (provided mostly as a comparison baseline).

## Features

- Support for all ffmpeg-supported video codecs (mutichannel images are stored in sets of 3 channels)
- Support for all GDAL-supported image codecs (channels are stored in single image if using e.g. JPEG2000, but every timestep is stored in a different file)
- Planar / non-planar input formats (depends on codec)
- 8 / 10 / 12 / 16 bits (depends on codec)
- Lossy & lossless encoding
- KLT / PCA transform: You can specify a number of principal components (ideally a multiple of 3), and PCA is applied over the channel dimension. Videos are encoded in sets of 3 channels

NOTE: As of now, everything is loaded in memory, which might be a problem for larger datasets. Future versions of this library will allow for navigation within the video files, as well as lazy loading.

## Paper

If you find this library useful, please cite the accompanying paper:
```
TODO
```

## Results

### DeepExtremeCubes

An example `xarray` from the DeepExtremeCubes database is provided as an example. It consists of 495 timesteps of Sentinel2 data and other segmentation maps. 
 - For the Sentinel2 data, the library automatically compresses lossily all 7 bands `'B04','B03','B02','B8A','B05','B06','B07'` into 3 videos (as each video can only encode up to 3 bands at a time). See Figure for compression results.
 - The cloud mask with 5 classes is compressed losslessly to 0.3403 bpppb.
 - Sentinel's Scene Classification Layer (SCL) is compressed losslessly to 0.1001 bpppb respectively.

Here is a plot with some results of different compression approaches for the multiespectral Sentinel 2 data (7 bands). Note that we also use JPEG2000 as a comparison, encoding every timestep at a time, but all bands in a single image (instead of relying on sets of 3 bands as for video compression):

![Results](examples/results_bpppb.png)

For this data, some conclusions can be reached. In summary: `x265 > vp9 >> JPEG2000`
 - 8bits is always inferior to 10-16bits
 - Video compression (either x265 or vp9) is always better than JPEG2000 compression of each timestep separately
 - x265 is slightly better and much quicker than vp9 in compression, so it is preferred
 - JPEG2000 is the slowest for decompression
 - Overall, very high reconstruction quality (PSNR > 50dB) can be achieved at very high compression rates (bpppb < 1, i.e., for 32bit floats this is a compression factor of >32x)
 

Example of compression (1.47% of original size). The quality loss is visually imperceptible.

Original (download for full size):
![Original image](examples/RGB_original.jpg)

Compressed (download for full size):
![Compressed image](examples/RGB_compressed.jpg)

These visualizations were generated using [txyvis](https://github.com/OscarPellicer/txyvis)
 
### ERA5

TODO

### Cesar's cube

TODO

## Installation

```{bash}
#Install base requirements
pip install xarray numpy ffmpeg scikit-image scikit-learn pyyaml zarr netcdf4 ffmpeg-python gdal gcsfs openjpeg tqdm seaborn
#mamba install xarray numpy ffmpeg scikit-image scikit-learn pyyaml zarr netcdf4 ffmpeg-python gcsfs tqdm seaborn
#mamba install -c gdal-master gdal openjpeg

#[Optional] Requiremetns for temporal alignment of video slices
pip install satalign

#[Optional] Requierements for plotting (optional, but `plot_image` calls will fail)
pip install ipython opencv-python
pip install git+https://github.com/OscarPellicer/txyvis.git

#[Optional] Requierements for much quicker metrics
pip install torchmetrics
#mamba install torchmetrics

#Download repo
cd ~
git clone https://github.com/OscarPellicer/xarrayvideo.git
cd xarrayvideo

#Unzip the example xarray
!unzip cube.zip
```

## Examples

To see some examples run `jupyter lab` or VSCode to open `example.ipynb`

The basic syntax (for the DeepExtremesCubes database) is the following:
```
#Load libraries
import xarray as xr
import numpy as np
from xarrayvideo import xarray2video, video2xarray, plot_image

#Define paths
array_id= '-111.49_38.60'
input_path= '../mc_-111.49_38.60_1.2.2_20230702_0.zarr'
output_path= './out'

#Load cube
minicube= xr.open_dataset(input_path, engine='zarr')
minicube['SCL']= minicube['SCL'].astype(np.uint8) #Fixes problem with the dataset
minicube['cloudmask_en']= minicube['cloudmask_en'].astype(np.uint8)

#Set up the compression params
lossless_params= { 'c:v':'ffv1' }
lossy_params= { 'c:v': 'libx265', 'preset': 'medium', 'crf': 1, 
                'x265-params': 'qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0  }
conversion_rules= {
    's2': ( ('B07','B06','B05','B04','B03','B02','B8A'), 
                ('time','x','y'), 0, image_lossy_params, 16),
    'scl': ( 'SCL', ('time','x','y'), 0, lossless_params, 8),
    'cm': ( 'cloudmask_en', ('time','x','y'), 0, lossless_params, 8),
    }
    
#Compress, with compute_stats it takes a bit longer, but shows compression info
arr_dict= xarray2video(minicube, array_id, conversion_rules,
                       output_path=output_path, compute_stats=True,
                       loglevel='verbose', save_dataset= True
                       )  
    
#Decompress
minicube_new= video2xarray(output_path, array_id)

#Plot RGB bands
plot_image(minicube, ['B04','B03','B02'], save_name='./out/RGB original.jpg')
plot_image(minicube_new, ['B04','B03','B02'], save_name='./out/RGB compressed.jpg')
```

## Contact

Contact me: `oscar.pellicer [at] uv.es` or open an Issue