# xarrayvideo
Save multichannel data from xarray datasets as videos to save up massive amounts of space (e.g. >20x compression for the provided example minicube).

Basically, this library provides two functions: `xarray2video` to encode some `xarray` variables into videos, and `video2xarray` to rebuild the xarray from the videos. As of now, everything is loaded in memory, and lossy compression only accepts videos with 3 chanels (WIP for 1 and 4 channels).

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

Example of compression 23.20Mb -> 0.95Mb (4.11% of original size). The quality loss is visually imperceptible.

Original (download for full size):
![Original image](examples/RGB_original.jpg)

Compressed (download for full size):
![Compressed image](examples/RGB_compressed.jpg)

These visualizations were generated using [txyvis](https://github.com/OscarPellicer/txyvis)