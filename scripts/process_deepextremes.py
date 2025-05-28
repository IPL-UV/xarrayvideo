import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
import pandas as pd
import pytortilla
import tacoreader
import json
import tacotoolbox
from xarrayvideo import xarray2video, video2xarray, plot_image, gap_fill
from io import BytesIO
from affine import Affine
from datetime import datetime

## Define the COLLECTION
description_dec = """
## Description

### 📦 Dataset
**DeepExtremeCubes-video** is a storage-efficient, analysis-ready re-packaging of the original
[DeepExtremeCubes](https://doi.org/10.25532/OPARA-703) minicube archive.
All 42 k Sentinel-2 L2A minicubes (2.56 km × 2.56 km, 2016–2022, 7 bands, 5-daily cadence)
were transcoded with **xarrayvideo** into 12-bit H.265/HEVC videos, shrinking the
dataset from ≈ 2.3 TB to ≈ 270 GB at ≈ 56 dB PSNR (~12 ×).

This compact video representation removes I/O bottlenecks yet preserves
scientific fidelity for:
* **Impact mapping** of compound heat-wave & drought events  
* **Forecasting** vegetation stress with ConvLSTM / U-TAE models  
* **Self-supervised pre-training** on long reflectance sequences  

### 🛰️ Sensors
* **Sentinel-2 MSI** (B02, B03, B04, B05, B06, B07, B8A) — 10 m & 20 m upsampled  
* **ERA5-Land** single-pixel time-series (Tair, SM, etc.)  
* **Copernicus DEM 30 m** (static)  
* Cloud & SCL masks from EarthNet Cloud-Mask v1  

Dynamic variables are encoded as multi-channel videos; static rasters
(DEM, land-cover) remain compressed GeoTIFFs.
"""

# -----  publication BibTeX strings ------------------------------------------------
bibtex_den_pub1 = """
@article{pellicer2024explainable,
  title        = {Explainable Earth Surface Forecasting under Extreme Events},
  author       = {Pellicer-Valero, Oscar J and Fernández-Torres, Miguel-Ángel and Ji, Chaonan and Mahecha, Miguel D and Camps-Valls, Gustau},
  journal      = {arXiv preprint arXiv:2410.01770},
  year         = 2024
}
"""

bibtex_den_pub2 = """
@article{ji2025deepextremecubes,
  title     = {DeepExtremeCubes: Earth system spatio-temporal data for assessing compound heatwave and drought impacts},
  author    = {Ji, Chaonan and Fincke, Tonio and Benson, Vitus and Camps-Valls, Gustau and Fernández-Torres, Miguel-Ángel and others},
  journal   = {Scientific Data},
  volume    = {12},
  number    = {1},
  pages     = {149},
  year      = 2025,
  doi       = {10.1038/s41597-025-04447-5}
}
"""
collection_object = tacotoolbox.datamodel.Collection(
    id="deepextremecubesvideo",
    title="DeepExtremeCubes-video: Sentinel-2 Minicubes in Video Format for Compound-Extreme Analysis",
    dataset_version="1.0.0",
    description=description_dec,
    licenses=["cc-by-4.0"],
    extent={
        "spatial": [[-180.0, -60.0, 180.0, 80.0]],
        "temporal": [["2016-01-01T00:00:00Z", "2022-12-31T23:59:59Z"]]
    },
    providers=[{
        "name": "Leipzig University – Remote Sensing Centre",
        "roles": ["host"],
        "links": [{"href": "https://rs.geo.uni-leipzig.de/", "rel": "homepage", "type": "text/html"}]
    }, {
        "name": "Image & Signal Processing (UV) – USMILE project",
        "roles": ["processor"],
        "links": [{"href": "https://ipl.uv.es", "rel": "homepage", "type": "text/html"}]
    }],
    keywords=[
        "remote-sensing", "sentinel-2", "climate-extremes",
        "video-compression", "deep-learning"
    ],
    task="regression",
    curators=[{
        "name": "Oscar J. Pellicer-Valero",
        "organization": "Image & Signal Processing (UV)",
        "links": [{"href": "https://scholar.google.com/citations?user=CCFJshwAAAAJ", "rel": "scholar"}]
    }, {
        "name": "Cesar Aybar",
        "organization": "Image & Signal Processing (UV)",
        "links": [{"href": "https://scholar.google.es/citations?user=rfF51ocAAAAJ", "rel": "scholar"}]
    }, {
        "name": "Julio Contreras",
        "organization": "Image & Signal Processing (UV)",
        "links": [{"href": "https://github.com/JulioContrerasH", "rel": "github"}]
    }],
    split_strategy="none",
    discuss_link={
        "href": "https://huggingface.co/datasets/tacofoundation/DeepExtremeCubes-video/discussions",
        "rel": "discussion", "type": "text/html"
    },
    raw_link={
        "href": "https://doi.org/10.25532/OPARA-703",
        "rel": "source", "type": "text/html"
    },
    optical_data={"sensor": "sentinel2msi"},
    labels={
        "label_classes": [],
        "label_description": "Dataset targets regression/forecasting; no discrete class labels."
    },
    scientific={
        "doi": "10.1038/s41597-025-04447-5",
        "citation": "DeepExtremeCubes-video 1.0.0 (2025). DOI:10.1038/s41597-025-04447-5",
        "summary": "Video-compressed Sentinel-2 minicubes plus ERA5 and DEM for compound heat-wave & drought research.",
        "publications": [
            {
                "doi": "10.48550/arXiv.2410.01770",
                "citation": bibtex_den_pub1,
                "summary": "Pellicer-Valero et al. (2024): ConvLSTM forecasting & explainable AI using DeepExtremeCubes."
            },
            {
                "doi": "10.1038/s41597-025-04447-5",
                "citation": bibtex_den_pub2,
                "summary": "Ji et al. (2025): Original DeepExtremeCubes dataset description in *Scientific Data*."
            }
        ]
    }
)

def to_video(dataset_out_path, images_out_path, 
             conversion_rules, files, debug, align, 
             ignore_existing=True, compute_stats=False):        
    #Run for all cubes
    for i, input_path in (pbar:=tqdm(enumerate(files), total=len(files))):
        try:
            #Print name
            array_id= '_'.join(input_path.stem.split('_')[1:3])
            pbar.set_description(array_id)
            
            #Check if it exists
            if (dataset_out_path / array_id).exists() and ignore_existing:
                continue

            #Load
            minicube= xr.open_dataset(input_path, engine='zarr')
            
            #Fix problems with the dataset
            if 'SCL' in minicube.variables:
                minicube['SCL']= minicube['SCL'].astype(np.uint8)
            if 'cloudmask_en' in minicube.variables: 
                minicube['cloudmask_en']= minicube['cloudmask_en'].astype(np.uint8)
                
            #We drop a variable for now, to have just 2 videos to store
            minicube= minicube.drop_vars('B07')
            bands= ['B04','B03','B02','B8A','B06','B05']
            
            #Align cube
            if align:
                import satalign, satalign.ecc, satalign.pcc #satalign.lgm
                import cv2
                def get_masked_mean_reference(minicube, bands, mask_name='cloudmask_en'):
                    data = minicube[bands].isel(time=slice(74, None)).to_array().transpose('variable', 'time', 'y', 'x')
                    mask = minicube[mask_name].isel(time=slice(74, None)).transpose('time', 'y', 'x')
                    data_np, mask_np= np.array(data), np.array(mask)
                    data_np[:, mask_np > 0] = np.nan
                    data_np = np.nanmean(data_np, axis=1)
                    return data_np, mask_np

                if 'cloudmask_en' in minicube.variables:
                    reference_image, mask= get_masked_mean_reference(minicube, bands=bands, mask_name='cloudmask_en')
                else:
                    reference_image= minicube[bands].isel(time=slice(74,None)).mean("time").to_array().transpose('variable', 'y', 'x')
                    mask= None
                datacube= minicube[bands].to_array().transpose('time', 'variable', 'y', 'x')

                syncmodel= satalign.pcc.PCC( #PCC quicker, ECC more precise
                    interpolation= cv2.INTER_CUBIC,
                    datacube=datacube, # T x C x H x W
                    reference=reference_image, # C x H x W
                    channel="mean", crop_center=110)
                new_cube, warps, errors, error_before, error_after= syncmodel.run()

                #Copy back
                minicube_aligned= minicube.copy()
                for b in bands: minicube_aligned[b]= new_cube.sel(variable=b)
                print(f'{error_before=:.6f}, {error_after=:.6f}, improvement={error_before-error_after:.6f}')
            else:
                minicube_aligned= minicube
                
            #Perform simple gap filling
            minicube_filled= gap_fill(minicube_aligned, fill_bands=bands, mask_band=None, #Do not gapfill clouds, otherise use 'cloudmask_en'
                                      fill_values=[1, 3, 4], method='last_value', new_mask='invalid', 
                                      coord_names=('time', 'variable', 'y', 'x'))

            #Compress
            results= xarray2video( minicube_filled, array_id, conversion_rules,
                                   output_path=dataset_out_path, compute_stats=compute_stats, 
                                   exceptions='ignore', loglevel='quiet',
                                   )  
            
            # Save stats
            if compute_stats:
                with open(dataset_out_path / array_id / 'stats.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, default = lambda d : str(d)) # add serializer for Paths

            #Plot image every 200
            if i%200 == 0:
                minicube_new= video2xarray(dataset_out_path, array_id) 
                plot_image(minicube_new, ['B04','B03','B02'], save_name=str(images_out_path/f'{array_id}.jpg'), 
                           show=False, mask_name='cloudmask_en' if 'cloudmask_en' in minicube.variables else None)

        except Exception as e:
            print(f'Exception processing {array_id=}: {e}')
            import traceback
            traceback.print_exc()
            if debug: raise e

        #Stop?
        if debug and i > 10: break
        
def to_tortilla(dataset_out_path, dataset_out_path_tortilla, registry_path):
    registry_df = pd.read_csv(registry_path, index_col='location_id', dtype={"version": str})
    files = list(dataset_out_path.iterdir())
    registry_df['cube_available'] = False
    for f in tqdm(files):
        try:
            # Skip problematic files
            if f.name not in registry_df.index.values:
                print(f'Skipping {f.name}: not in registry_df')
                continue
            else:
                registry_df.loc[f.name, 'cube_available'] = True

            # Create samples
            sample_rgb = pytortilla.datamodel.Sample(
                id="rgb",
                file_format="BYTES",
                path=f / "rgb_001.mkv",
            )

            # sample_b07 = pytortilla.datamodel.Sample(
            #     id="b07",
            #     file_format="BYTES",
            #     path=output_path + "b07_001.mkv",
            #     data_split="train"
            # )

            sample_ir3 = pytortilla.datamodel.Sample(
                id="ir3",
                file_format="BYTES",
                path=f / "ir3_001.mkv",
            )

            sample_masks = pytortilla.datamodel.Sample(
                id="masks",
                file_format="BYTES",
                path=f / "masks_001.mkv",
            )

            sample_x = pytortilla.datamodel.Sample(
                id="metadata",
                file_format="BYTES",
                path=f / "x.nc",
            )

            # Create samples
            samples = pytortilla.datamodel.Samples(samples=[sample_rgb, sample_ir3, sample_masks, sample_x])

            # Create metrics
            pytortilla.create(samples=samples, output=dataset_out_path_tortilla / f'{f.name}.tortilla')
        
        except Exception as e:
            print(f'Error processing {f.name}: {e}')
            
    # Save csv with new columns indicating availability
    registry_df.to_csv(registry_path.parent / 'dx_video.csv')
    print(f'Available cubes from registry: {registry_df["cube_available"].mean()*100:.2f}%%')
            
def to_taco(dataset_in_path, dataset_out_path_taco, registry_path):
    registry_df = pd.read_csv(registry_path, index_col='location_id', dtype={"version": str})
    tortillas = list(dataset_in_path.iterdir())

    SAMPLE_CONTAINER = []
    for tortilla in tqdm(tortillas, desc='Cooking tortillas'):

        # Load the tortilla file
        tortilla_obj = tacoreader.load(tortilla.as_posix())

        # Load the NetCDF minicube from the tortilla
        minicube = xr.open_dataset(BytesIO(tortilla_obj.read(3)), engine="h5netcdf")

        # Generate STAC properties
        crs_info = minicube.attrs["spatial_ref"]
        x_min, y_min, x_max, y_max = minicube.attrs['spatial_bbox']
        res = minicube.attrs['spatial_res']
        transform = (Affine.translation(x_min.item(), y_max.item()) * Affine.scale(res.item(), -res.item())).to_gdal()

        # Locate cube metadata
        cube_metadata = registry_df.loc[tortilla.stem]

        # Define the sample metadata
        samples = pytortilla.datamodel.Sample(
            id=tortilla.stem,
            file_format="TORTILLA",
            path=tortilla,
            stac_data={
                "crs": crs_info,
                "geotransform": transform,
                "raster_shape": (minicube.sizes["y"], minicube.sizes["x"]),
                "time_start": datetime.fromtimestamp(minicube.time.min().item() / 1e9),
                "time_end": datetime.fromtimestamp(minicube.time.max().item() / 1e9),
            },
            **cube_metadata.to_dict(),
        )

        # Add the sample to the collection
        SAMPLE_CONTAINER.append(samples)

    samples = pytortilla.datamodel.Samples(samples=SAMPLE_CONTAINER)

    tacotoolbox.create(
        samples=samples,
        collection=collection_object,
        output=dataset_out_path_taco,
    )

    # # Acess to the taco file
    # taco = tacoreader.load(dataset_out_path_taco)

    # # Load the NetCDF minicube from the taco
    # netbytes = taco.read(0).read(3)
    # minicube = xr.open_dataset(BytesIO(netbytes), engine="h5netcdf")        

if __name__ == '__main__':
    #Set parameters
    dataset_in_path= Path('/scratch/users/databases/deepextremes/deepextremes-minicubes/full')
    registry_path = Path('/scratch/users/databases/dx.csv')
    
    # dataset_out_path= Path('/scratch/users/databases/deepextremes-video-final')
    dataset_out_path= Path('/scratch/users/databases/deepextremes-video-56psnr')
    dataset_out_path_tortilla= Path('/scratch/users/databases/deepextremes-video-tortilla')
    dataset_out_path_taco= Path('/scratch/users/databases/DeepExtremeCubes-video.taco')
    images_out_path= Path('./deepextremes_images')
    lossy_params = [
                     {'c:v': 'libx265',
                      'preset': 'medium',
                      'crf': 51,
                      'x265-params': 'qpmin=0:qpmax=0.01',
                      'tune': 'psnr',
                     },
                     {'c:v': 'libx265',
                      'preset': 'medium',
                      'crf': 7,
                      'tune': 'psnr',
                     }
                    ][0] #Choose the params
    lossless_params= { 'c:v':'ffv1' }

    conversion_rules= {
        #Sets of 3 channels are the most efficient
        'rgb': ( ('B04','B03','B02'), ('time','y','x'), 0, lossy_params, 12),
        'ir3': ( ('B8A','B06','B05'), ('time','y','x'), 0, lossy_params, 12),
        'masks': ( ('SCL', 'cloudmask_en', 'invalid'), ('time','y','x'), 0, lossless_params, 8),
        }
    debug= False
    align= False
    ignore_existing= True
    compute_stats= False
    
    # Convert the dataset to xarrayvideo format
    if False:
        files= list(dataset_in_path.glob('*/*.zarr'))
        dataset_out_path.mkdir(parents=True, exist_ok=True)
        to_video(dataset_out_path, images_out_path, conversion_rules, files, 
                 debug, align, ignore_existing=ignore_existing, compute_stats=compute_stats)

    # Save each cube in a tortilla
    if False:
        dataset_out_path_tortilla.mkdir(parents=True, exist_ok=True)
        to_tortilla(dataset_out_path, dataset_out_path_tortilla, registry_path)
    
    # Save all tortillas in a taco
    if True:
        dataset_out_path_taco.parent.mkdir(parents=True, exist_ok=True)
        if dataset_out_path_taco.exists():
            raise FileExistsError(f'{dataset_out_path_taco} already exists')
        to_taco(dataset_out_path_tortilla, dataset_out_path_taco, registry_path)