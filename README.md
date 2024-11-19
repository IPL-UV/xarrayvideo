# xarrayvideo

Save multichannel data from xarray datasets as videos to save up massive amounts of space (e.g. 20-50x compression) with minimal quality loss.

This library provides two functions: `xarray2video` to encode some `xarray` variables into videos, and `video2xarray` to rebuild the xarray from the videos. It can encode lossily or losslessly, supporting all video formats supported by ffmpeg (e.g., x265, vp9) at different bit depths (e.g., 8,10,12,16) as well as all GDAL's supported image formats (e.g. JPEG2000) for frame-by-frame encoding (provided mostly as a comparison baseline).

## Features

- Support for all ffmpeg-supported video codecs (mutichannel images are stored in sets of 3 channels)
- Support for all GDAL-supported image codecs (every timestep is stored in a different image file, but all channels are stored together if using e.g. JPEG2000).
- Support for many pixel formats, depending on the codec: 8 / 10 / 12 / 16 bits, planar / non-planar, etc.
- Lossy & lossless encoding are both supported: `libx265` is recommended for lossy, and `ffv1` for lossless
- KLT / PCA transform: You can specify a number of principal components (ideally a multiple of 3), and PCA is applied over the channel dimension. You can choose how many principal components (PCs) to keep, and even particular quality options for different PCs. Videos are encoded then in sets of 3 channels.

## Paper

If you find this library useful, please consider citing the accompanying paper:

```
Paper in progress
```

## Installation

Before installing, GDAL is an optional dependency but is required for some functionality. Installation methods vary by operating system:

Linux and macOS:

```bash
pip install gdal
```

Windows:

```bash
mamba install -c conda-forge gdal
# or
conda install -c conda-forge gdal
```

Test installation:

```bash
git clone https://github.com/OscarPellicer/xarrayvideo.git
cd xarrayvideo
pip install -e .[all] #--no-deps
```

Standard installation (WIP):

```bash
# To install with all dependencies except gdal (very recommended)
pip install xarrayvideo[all] 

# To install with only the base dependencies
pip install xarrayvideo

# Also possible
pip install xarrayvideo[satalign] # For temporal alignment of video slices
pip install xarrayvideo[plotting] # For plotting functionality
pip install xarrayvideo[metrics] # For quicker metrics calculation
```

Step-by-step manual installation:

```bash
# Install base requirements
pip install xarray numpy ffmpeg scikit-image scikit-learn pyyaml zarr netcdf4 ffmpeg-python gdal gcsfs openjpeg tqdm seaborn
# mamba install xarray numpy ffmpeg scikit-image scikit-learn pyyaml zarr netcdf4 ffmpeg-python gcsfs tqdm seaborn
# mamba install -c gdal-master gdal openjpeg

# [Optional] Requirements for temporal alignment of video slices
pip install satalign

# [Optional] Requirements for plotting (optional, but `plot_image` calls will fail)
pip install ipython opencv-python
pip install git+https://github.com/OscarPellicer/txyvis.git

# [Optional] Requirements for much quicker metrics
pip install torchmetrics
# mamba install torchmetrics

# Download repo and install it with not dependencies
cd ~
git clone https://github.com/OscarPellicer/xarrayvideo.git
cd xarrayvideo
pip install -e . --no-deps

# Unzip the example xarray
unzip cube.zip
```

## Examples

To see some examples run `jupyter lab` or VSCode to open the example notebooks: 

 - DynamicEarthnet: `example_dynamicearthnet.ipynb`
 - DeepExtremeCubes dataset: `example_deepextremecubes.ipynb`
 - Custom dataset: `example_customcube.ipynb`
 - ERA5: `example_era5.ipynb`

## Basic usage 

The basic syntax (for the DeepExtremesCubes database) is the following:

```python
# Load libraries
import xarray as xr
import numpy as np
from xarrayvideo import xarray2video, video2xarray, plot_image

# Define paths
array_id= '-111.49_38.60'
input_path= '../mc_-111.49_38.60_1.2.2_20230702_0.zarr'
output_path= './out'

# Load cube
minicube= xr.open_dataset(input_path, engine='zarr')
minicube['SCL']= minicube['SCL'].astype(np.uint8) #Fixes problem with the dataset
minicube['cloudmask_en']= minicube['cloudmask_en'].astype(np.uint8)

# Set up the compression params
lossless_params= { 'c:v':'ffv1' }
lossy_params = {
    'c:v': 'libx265',
    'preset': 'medium',
    'crf': 51,
    'x265-params': 'qpmin=0:qpmax=0.01',
    'tune': 'psnr',
    },
conversion_rules = {
        'rgb': ( ('B04','B03','B02'), ('time','y','x'), 0, lossy_params, 12),
        'ir3': ( ('B8A','B06','B05'), ('time','y','x'), 0, lossy_params, 12),
        'masks': ( ('SCL', 'cloudmask_en', 'invalid'), ('time','y','x'), 0, lossless_params, 8),
        }
    
# Compress, with compute_stats it takes a bit longer, but shows compression info
arr_dict= xarray2video(minicube, array_id, conversion_rules,
                       output_path=output_path, compute_stats=True,
                       loglevel='verbose', save_dataset= True
                       )  
    
# Decompress
minicube_new= video2xarray(output_path, array_id)

# Plot RGB bands
plot_image(minicube, ['B04','B03','B02'], save_name='./out/RGB original.jpg')
plot_image(minicube_new, ['B04','B03','B02'], save_name='./out/RGB compressed.jpg')
```

## Other files

Other interesting files in the repo:

 - `scripts/process_deepextremes.py`: transform the DeepExtremeCubes dataset to `xarrayvideo` format
 - `scripts/process_dynamicearthnet.py`: transform the DynamicEarthNet dataset first to `xarray` and then to `xarrayvideo` format. It also includes a description on how to dowload and use the dataset
 - `scripts/run_tests.py`: run the tests of results section of the paper and generate the tables and plots 
 - `scripts/find_encoders.sh`: find ffpmeg encoders supporting a specfific pixel format
 - `scripts/fix_metadata.py` and `scripts/find_processing_gap.py` were used to find and fix problems in the processing of the data, but should no longer be needed

## Contact

Contact me: `oscar.pellicer [at] uv.es` or open an Issue.
