# xarrayvideo
Save multichannel data from xarray datasets as videos to save up massive amounts of space (e.g. 20-50x compression).

Basically, this library provides two functions: `xarray2video` to encode some `xarray` variables into videos, and `video2xarray` to rebuild the xarray from the videos. As of now, everything is loaded in memory, and lossy compression only accepts videos with 3 chanels (WIP for 1 and 4 channels).

For the provided example `xarray`, we compress bands `'B04','B03','B02'` into video `rgb` (lossy), bands `'B8A','B06','B05'` (lossy) into video ir3, a cloud mask with 5 classes into video `cm` (lossless) and Sentinel's Scene Classification Layer (SCL) into video `scl` (lossless).

```
rgb: 92.81Mb -> 1.37Mb (1.47% of original size) in 11.32s
 - params={'c:v': 'libx265', 'preset': 'slow', 'crf': 6, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 0.18s
 - MSE_sat 0.000020 (input saturated to [(0.0, 1.0)])
 - SNR_sat 39.4120 (input saturated to [(0.0, 1.0)])
 - PSNR_sat 47.0182 (input saturated to [(0.0, 1.0)])
 - Exp. SA 0.0167 (input saturated to [(0.0, 1.0)])
 - Err. SA 0.0001 (input saturated to [(0.0, 1.0)])
 
ir3: 92.81Mb -> 1.37Mb (1.47% of original size) in 12.01s
 - params={'c:v': 'libx265', 'preset': 'slow', 'crf': 6, 'pix_fmt': 'yuv444p10le', 'r': 30}
 - Decompression time 0.20s
 - MSE_sat 0.000027 (input saturated to [(0.0, 1.0)])
 - SNR_sat 38.3215 (input saturated to [(0.0, 1.0)])
 - PSNR_sat 45.7592 (input saturated to [(0.0, 1.0)])
 - Exp. SA 0.0129 (input saturated to [(0.0, 1.0)])
 - Err. SA 0.0000 (input saturated to [(0.0, 1.0)])
 
cm: 7.73Mb -> 0.08Mb (0.98% of original size) in 0.10s
 - params={'vcodec': 'ffv1', 'pix_fmt': 'gray', 'r': 30}
 - acc 1.00
 
scl: 7.73Mb -> 0.29Mb (3.70% of original size) in 0.20s
 - params={'vcodec': 'ffv1', 'pix_fmt': 'gray', 'r': 30}
 - acc 1.00
```

Here is a plot with some very early results:
![Results](examples/results_bpppb.png)

## Features

- Planar / non-planar input formats (depends on video codec)
- 8 / 10 / 12 / 16 bits (depends on video codec)
- Lossy & lossless encoding
- 1 / 3 / 4 channels
  - 1 channel: great lossless compression with `ffv1` (8,10,12,16 bits), bad lossy compression with vp9 (8,10,12 bits) **(WIP: should work better)**
  - 3 channels: great lossless compression with `vp9, lossless 1` (8,10,12 bits), great lossy comporession with `x264` (8,10 bits) or `x265` (8,10,12 bits) 
  - 4 channels: theoretically supported with `vp9` for both lossy and lossless, but currently not working **(WIP)**
- KLT / PCA transform **(WIP)**

## Installation 

```
#Install base requirements
pip install xarray numpy ffmpeg scikit-image

#Requiremetns for temporal alignment
pip install satalign

#Insall requierements for plotting (optional, but `plot_image` calls will fail)
pip install ipython cv2
pip install git+https://github.com/OscarPellicer/txyvis.git

#Install repo
cd ~
git clone https://github.com/OscarPellicer/xarrayvideo.git
cd xarrayvideo

#Unzip the example xarray
!unzip cube.zip
```

Then run `jupyter lab` or VSCode to open example.ipynb

## Examples

Example of compression 92.81Mb -> 1.37Mb (1.47% of original size). The quality loss is visually imperceptible.

Original (download for full size):
![Original image](examples/RGB_original.jpg)

Compressed (download for full size):
![Compressed image](examples/RGB_compressed.jpg)

These visualizations were generated using [txyvis](https://github.com/OscarPellicer/txyvis)