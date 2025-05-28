'''
Instructions for downlading the dynamicearthnet data:

1. Download the data into folder dynamicearthnet (https://mediatum.ub.tum.de/1650201 and https://mediatum.ub.tum.de/1738088)
    echo "m1650201" > /tmp/rsync_password_file_0
    chmod 600 /tmp/rsync_password_file_0
    rsync -av --progress --password-file=/tmp/rsync_password_file_0 rsync://m1650201@dataserv.ub.tum.de/m1650201/ .
    
    echo "m1738088" > /tmp/rsync_password_file_1
    chmod 600 /tmp/rsync_password_file_1
    rsync -av --progress --password-file=/tmp/rsync_password_file_1 rsync://m1738088@dataserv.ub.tum.de/m1738088/ .

2. Unzip it all:
    cd dynamicearthnet
    for file in *.zip; do   folder_name="${file%.zip}";   mkdir -p "$folder_name";   unzip -q "$file" -d "$folder_name"; done
    rm -f *.zip
    rm checksums.sha512
    du -sh
    
3. Download splits in to the data path:
    mkdir dynnet_training_splits
    cd dynnet_training_splits
    wget https://cvg.cit.tum.de/webshare/u/toker/dynnet_training_splits/train.txt
    wget https://cvg.cit.tum.de/webshare/u/toker/dynnet_training_splits/val.txt
    wget https://cvg.cit.tum.de/webshare/u/toker/dynnet_training_splits/test.txt
    
To do inference on the data using the forked repo https://github.com/OscarPellicer/dynnet
Note that this repo has been adapted to use the xarray / xarrayvideo version of the dataset
1. Download the weights into the repo: 
    git clone https://github.com/OscarPellicer/dynnet
    cd ~/dynnet
    wget -r -np -nH --cut-dirs=4 -R "index.html*" https://cvg.cit.tum.de/webshare/u/toker/dynnet_ckpt/

2. Run inference:
    chmod +x run_inference.sh
    ./run_inference.sh
    python inference.py --phase val --config config/defaults.yaml --checkpoint ./weights/single_unet/weekly/best_ckpt.pth --dataset xarray --model single_unet --time weekly --data /scratch/users/databases/dynamicearthnet-xarray/
'''

import xarray as xr
import rasterio
from pathlib import Path
import numpy as np
import datetime
from tqdm import tqdm
import pytortilla
import tacoreader
import json
from xarrayvideo import to_netcdf, plot_image, xarray2video, video2xarray
import tacotoolbox
import tacoreader
import pytortilla
from io import BytesIO

description = """
## Description

### 📦 Dataset
DynamicEarthNet-video is a storage-efficient re-packaging of the original
**DynamicEarthNet** collection. The archive covers seventy-five
1024 × 1024 px regions (≈ 3 m GSD) across the globe, sampled daily from
**1 January 2018 to 31 December 2019**.  Each day provides
four-band PlanetFusion surface-reflectance images (B04 Red, B03 Green,
B02 Blue, B8A Narrow-NIR).  Monthly pixel-wise labels annotate seven
land-cover classes: impervious, agriculture, forest, wetlands, bare soil,
water, snow/ice.

All original GeoTIFF stacks (~525 GB) are transcoded with
[xarrayvideo](https://github.com/IPL-UV/xarrayvideo) to 12-bit H.265/HEVC,
yielding dramatic savings while preserving scientific fidelity:

| Version                     |   Size | PSNR | Ratio |
| --------------------------- | -----: | ---: | ----: |
| Raw GeoTIFF                 | 525 GB |  —   | 1 ×   |
| **DynamicEarthNet-video**   | 8.5 GB | 60 dB| 62 × |
| Extra-compressed (optional) | 2.1 GB | 54 dB| 249 × |

Semantic change-segmentation scores with U-TAE, U-ConvLSTM and 3D-UNet
remain statistically unchanged (Δ mIoU ≤ 0.02 pp) when the compressed
cubes replace the raw imagery.

### 🛰️ Sensors
| Instrument    | Platform (fusion)      | Bands | Native GSD | Role                  |
| ------------- | ---------------------- | ----- | ---------- | --------------------- |
| **PlanetFusion** | PlanetScope + SkySat | RGB+NIR | 3 m | Daily image sequence |

### 🔌 Why videos?
* Removes I/O bottlenecks – train sequence models straight from disk  
* Enables rapid experimentation on high-frequency optical data  
* Facilitates dataset sharing & reproducible change-detection benchmarks
"""
bibtex_publication = """
@inproceedings{toker2022dynamicearthnet,
  title     = {DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation},
  author    = {Toker, Achraf and Kondmann, Lisa and Weber, Manuel and Eisenberger, Martin and Camero, Alfonso and Hu, Jing and Pregel Höderlein, André and Şenaras, Çagatay and Davis, Tyler and Cremers, Daniel and Marchisio, Guido and Zhu, Xiao Xiang and Leal-Taixé, Laura},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2022},
  doi       = {10.48550/arXiv.2203.12560}
}
"""

label_tuples = [
    ("impervious", 0, "Built-up / impervious surfaces"),
    ("agriculture", 1, "Croplands and agricultural areas"),
    ("forest", 2, "Woody vegetation / forest"),
    ("wetlands", 3, "Wetlands & marshes"),
    ("bare_soil", 4, "Bare soil / sparse vegetation"),
    ("water", 5, "Open water bodies"),
    ("snow_ice", 6, "Permanent snow or ice"),
]

label_dicts = [
    {"name": n, "category": idx, "description": desc}
    for n, idx, desc in label_tuples
]

collection_object = tacotoolbox.datamodel.Collection(
    id="dynamicearthnetvideo",
    title="DynamicEarthNet-video: Daily PlanetFusion Image Cubes Compressed as Videos",
    dataset_version="1.0.0",
    description=description,
    licenses=["cc-by-4.0"],
    extent={
        "spatial": [[-180.0, -60.0, 180.0, 75.0]],              # coarse global envelope
        "temporal": [["2018-01-01T00:00:00Z", "2019-12-31T23:59:59Z"]],
    },
    providers=[
        {
            "name": "Technical University of Munich (TUM)",
            "roles": ["producer"],
            "links": [
                {"href": "https://mediatum.ub.tum.de/1650201",
                 "rel": "source", "type": "text/html"}
            ],
        },
        {
            "name": "Planet Labs PBC",
            "roles": ["licensor"],
            "links": [
                {"href": "https://www.planet.com",
                 "rel": "about", "type": "text/html"}
            ],
        },
    ],
    keywords=[
        "remote-sensing", "planet", "change-detection",
        "spatiotemporal", "deep-learning", "video-compression"
    ],
    task="semantic-segmentation",
    curators=[
        {
            "name": "Oscar J. Pellicer-Valero",
            "organization": "Image & Signal Processing (ISP-UV)",
            "email": ["oscar.pellicer@uv.es"],
            "links": [
                {"href": "https://scholar.google.com/citations?user=CCFJshwAAAAJ",
                 "rel": "homepage", "type": "text/html"}
            ],
        },
        {
            "name": "Cesar Aybar",
            "organization": "Image & Signal Processing (ISP-UV)",
            "email": ["cesar.aybar@uv.es"],
        },
        {
            "name": "Julio Contreras",
            "organization": "Image & Signal Processing (ISP-UV)",
            "email": ["julio.contreras@uv.es"],
            "links": [
                {"href": "https://github.com/JulioContrerasH",
                 "rel": "homepage", "type": "text/html"}
            ],
        },
    ],
    split_strategy="none",
    discuss_link={
        "href": "https://huggingface.co/datasets/tacofoundation/DynamicEarthNet-video/discussions",
        "rel": "discussion", "type": "text/html"
    },
    raw_link={
        "href": "https://mediatum.ub.tum.de/1650201",
        "rel": "source", "type": "text/html"
    },
    optical_data={"sensor": "planetfusion"},
    labels={
        "label_classes": label_dicts,
        "label_description": "Monthly 7-class land-cover masks."
    },
    scientific={
        "doi": "10.9999/zenodo.placeholder_denetvideo",
        "citation": "DynamicEarthNet-video (v1.0.0), compressed release of DynamicEarthNet.",
        "summary": "12-bit HEVC videos of daily PlanetFusion cubes; mIoU unchanged after compression.",
        "publications": [
            {
                "doi": "10.48550/arXiv.2203.12560",
                "citation": bibtex_publication,
                "summary": "Original DynamicEarthNet dataset and benchmark (CVPR 2022)."
            }
        ]
    }
)


def find_label_file(label_dir, location_id, location_short, date):
    """Find a label file for a given location and date, supporting multiple date formats."""
    # Create both possible folder paths: sometimes "_" is used, but other times "-" is used
    location_patterns = [
        f"{location_id}_{location_short}",
        f"{location_id}-{location_short}"
    ]
    
    # Create both possible date formats
    date_formats = [
        date.strftime("%Y-%m-%d"),
        date.strftime("%Y_%m_%d")
    ]
    
    # Iterate through all combinations of location patterns and date formats
    for location_pattern in location_patterns:
        location_folder = Path(label_dir) / location_pattern
        for date_str in date_formats:
            label_file = next(location_folder.glob(f"**/*{date_str}.tif"), None)
            if label_file:
                return label_file
    
    # Return None if no label file was found
    return None

def create_xarray_for_location(location_path, label_dirs, band_names=['B', 'G', 'R', 'NIR']):
    """Creates an xarray.DataArray for all the .tif files of a given location."""
    data_arrays = []
    times = []
    labels = []
    label_times = []

    image_files = sorted(location_path.glob("*.tif"))

    # Extract metadata (CRS and transform) from the first image
    # Assumes all GeoTIFFs in a sequence share the same projection and resolution
    first_image_file = image_files[0]
    with rasterio.open(first_image_file) as src_meta:
        crs_wkt = str(src_meta.crs)
        transform_gdal = list(src_meta.transform.to_gdal())
    
    # Add a progress bar for processing image files
    for image_file in tqdm(image_files, desc=f"Processing {location_path.name}", leave=False):
        with rasterio.open(image_file) as src:
            data = src.read()
            times.append(datetime.datetime.strptime(image_file.stem, "%Y-%m-%d"))
            data_arrays.append(data)
            
            # Check for label files in each label directory
            label_data = None
            for l, label_dir in enumerate(label_dirs):
                label_file = find_label_file(label_dir, location_path.parent.name, location_path.parts[-4], times[-1])
                if label_file:
                    with rasterio.open(label_file) as label_src:
                        label_raw = label_src.read() 
                        label_data = np.argmax(label_raw, axis=0).astype(np.uint8) #Undo the one-hot encoding
                    label_times.append(times[-1])
                    labels.append(label_data)
                    break

    # Check that band_names has the appropriate lenght
    assert data_arrays[0].shape[0] == len(band_names), \
        f'{data_arrays[0].shape[0]=} != {len(band_names)=}'
    
    # Create a dictionary to hold each band as a separate DataArray
    data_arrays_dict = {}
    for i, band_name in enumerate(band_names):
        data_arrays_dict[band_name] = xr.DataArray(
            np.array([arr[i] for arr in data_arrays]),
            dims=["time", "y", "x"],
            coords={
                "time": times,
                "y": np.arange(data_arrays[0].shape[1]),
                "x": np.arange(data_arrays[0].shape[2])
            }
        )

    # Create the xarray.DataArray for the labels data
    label_data_array = xr.DataArray(
        np.array(labels),
        dims=["time_month", "y", "x"],
        coords={
            "time_month": label_times,
            "y": np.arange(labels[0].shape[0]),
            "x": np.arange(labels[0].shape[1])
        }
    )

    # Combine both data arrays into a Dataset
    dataset = xr.Dataset(data_arrays_dict)
    dataset["labels"] = label_data_array

    # Store CRS and transform in dataset attributes
    dataset.attrs['crs_wkt'] = crs_wkt
    dataset.attrs['transform_gdal'] = transform_gdal

    return dataset

def to_xarray(planet_root, output_path, label_dirs, skip_existing=True):
    """Process all planet folders and save each location's data as a .zarr file."""
    planet_folders = list(planet_root.glob("planet.*"))

    # Add a progress bar for the overall location processing
    for planet_folder in tqdm(planet_folders, desc="xarray: processing all planet folders"):
        for location_path in tqdm(list(planet_folder.glob("planet/*/*/*/PF-SR")), 
                                  desc=f"Processing locations in {planet_folder.name}", leave=False):
            xarray_output_path = output_path / f"{location_path.parent.name}.nc"
            if xarray_output_path.exists():
                if skip_existing:
                    print(f'Warning: Xarray {xarray_output_path} already exists. Skipping!')
                    continue
                else:
                    print(f'Warning: Xarray {xarray_output_path} already exists. Overwriting!')
            try:
                dataset = create_xarray_for_location(location_path, label_dirs)
                to_netcdf(dataset, xarray_output_path)
            except Exception as e:
                print(f'Error processing {location_path=}: {e}')
            
def extract_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            # Extract the identifier from the second part of the line
            if len(parts) > 1:
                id_part = parts[0].split('/')[5]  # Extracting the identifier
                ids.append(f'{id_part} {id_part} {parts[2]}')
    return ids

def generate_id_files(input_path_subsets, output_path_subsets):
    # Convert string paths to Path objects
    input_path = Path(input_path_subsets)
    output_path = Path(output_path_subsets)

    # List of subset filenames to process
    subsets = ['train.txt', 'val.txt', 'test.txt']

    for subset in subsets:
        input_file_path = input_path / subset
        output_file_path = output_path / subset

        unique_ids = extract_ids(input_file_path)
        print(f'{subset}: {len(unique_ids)} elements')

        # Write unique identifiers to the output file
        with output_file_path.open('w') as output_file:
            for unique_id in sorted(unique_ids):
                output_file.write(f"{unique_id}\n")
            
def to_video(input_path, output_path, images_output_path, conversion_rules, 
             skip_existing=True, compute_stats=False):
    
    files= list(input_path.glob('*.nc'))
    
    # Run for all cubes
    for i, input_path in (pbar:=tqdm(enumerate(files), total=len(files), desc="xarray2video")):
        try:
            # Print name
            array_id= input_path.stem
            pbar.set_description(array_id)
            
            if (output_path / array_id).exists():
                if skip_existing:
                    print(f'Warning: Xarray-video {array_id} already exists. Skipping!')
                    continue
                else:
                    print(f'Warning: Xarray-video {array_id} already exists. Overwriting!')

            # Load
            minicube= xr.open_dataset(input_path)
            minicube['labels']= minicube['labels'].astype(np.uint8)
            
            # Compress
            results= xarray2video(minicube, array_id, conversion_rules,
                           output_path=output_path, compute_stats=compute_stats, 
                           exceptions='ignore', loglevel='quiet',
                           )  
            
            # Save stats
            if compute_stats:
                with open(output_path / array_id / 'stats.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, default = lambda d : str(d)) # add serializer for Paths

            # Plot image every 10
            if True: #i%10 == 0:
                minicube_new= video2xarray(output_path, array_id) 
                plot_image(minicube_new.isel(time=slice(None,None,31)).astype(np.float32)/10000.,
                           list('RGB'), mask_name=None, stack_every=12, show=False, 
                           save_name=str(images_output_path/f'{array_id}.jpg'))

        except Exception as e:
            print(f'Exception processing {array_id=}: {e}')

def generate_id_to_split_map(output_path_subsets):
    id_to_split_map = {}
    subsets = ['train.txt', 'val.txt', 'test.txt']
    for subset in subsets:
        output_file_path = output_path_subsets / subset

        unique_ids = list(set([line.split(' ')[0] for line in output_file_path.read_text().splitlines()]))
        print(f'{subset}: {len(unique_ids)} elements')

        for unique_id in unique_ids:
            id_to_split_map[unique_id] = {'train.txt': 'train', 'val.txt': 'validation', 'test.txt': 'test'}[subset]

    return id_to_split_map

def to_tortilla(input_path, output_path, id_to_split_map):
    """Creates tortilla files for each processed video cube, including data split information."""
    files = list(input_path.iterdir())  # These are directories named with cube_id
    output_path.mkdir(parents=True, exist_ok=True)

    for f_dir_path in tqdm(files, desc="Cooking tortillas"):
        cube_id = f_dir_path.name
        try:
            data_split_value = id_to_split_map.get(cube_id)
            if data_split_value is None:
                print(f"Warning: No data split found for {cube_id}.")

            # Define potential samples based on the expected file structure
            sample_definitions = [
                {"id": "bands_1", "filename": "bands_001.mkv"},
                {"id": "bands_2", "filename": "bands_002.mkv"},
                {"id": "labels", "filename": "labels_001.mkv"},
                {"id": "metadata", "filename": "x.nc"}
            ]

            created_samples = []
            for s_def in sample_definitions:
                file_path = f_dir_path / s_def["filename"]
                # if file_path.exists(): #Let it fail if the file does not exist
                sample = pytortilla.datamodel.Sample(
                    id=s_def["id"],
                    file_format="BYTES",  # Assuming generic bytes, adjust if pytortilla has specific types
                    path=file_path,
                    data_split=data_split_value
                )
                created_samples.append(sample)

            samples_obj = pytortilla.datamodel.Samples(samples=created_samples)
            tortilla_file_path = output_path / f'{cube_id}.tortilla'
            pytortilla.create(samples=samples_obj, output=tortilla_file_path)
        
        except Exception as e:
            print(f'Error processing {cube_id} for tortilla creation: {e}')

def to_taco(input_path, output_path):
    tortillas = list(input_path.iterdir())  # These are .tortilla files

    SAMPLE_CONTAINER = []
    for i, tortilla in enumerate(tqdm(tortillas, desc="Putting tortillas into a taco")):
        if not tortilla.is_file() or tortilla.suffix != '.tortilla':
            print(f"Skipping non-tortilla file: {tortilla}")
            continue
        
        # Load the tortilla file
        tortilla_obj = tacoreader.load(tortilla.as_posix())
        
        # Load the NetCDF minicube from the tortilla
        minicube = xr.open_dataset(BytesIO(tortilla_obj.read(3)), engine="h5netcdf")

        # Define the sample metadata
        samples = pytortilla.datamodel.Sample(
            id=tortilla.stem,
            file_format="TORTILLA",
            path=tortilla,
            stac_data={
                "crs": minicube.attrs['crs_wkt'],
                "geotransform": minicube.attrs['transform_gdal'],
                "raster_shape": (minicube.sizes["y"], minicube.sizes["x"]),
                "time_start": datetime.datetime.fromtimestamp(minicube.time.min().item() / 1e9),
                "time_end": datetime.datetime.fromtimestamp(minicube.time.max().item() / 1e9),
            }
        )

        # Add the sample to the collection
        SAMPLE_CONTAINER.append(samples)
        
    samples = pytortilla.datamodel.Samples(samples=SAMPLE_CONTAINER)

    tacotoolbox.create(
        samples=samples,
        collection=collection_object,
        output=output_path,
    )

    # # Acess to the taco file
    # taco = tacoreader.load(output_path)


    # # Load the NetCDF minicube from the taco
    # netbytes = taco.read(0).read(3)
    # minicube = xr.open_dataset(BytesIO(netbytes), engine="h5netcdf")

if __name__ == '__main__':
    # Set the paths: inputs
    datasets_path= Path("/scratch/users/databases/")
    planet_root = datasets_path / 'dynamicearthnet'
    input_path_subsets = datasets_path / 'dynamicearthnet' / 'dynnet_training_splits'
    label_dirs = [planet_root / "dynamicearthnet_test_labels", planet_root / "labels" / "labels"]

    # Outputs
    output_path_xarray = datasets_path / 'dynamicearthnet-video-final' / 'dynamicearthnet-xarray'
    output_path_video = datasets_path / 'dynamicearthnet-video-final' / 'dynamicearthnet-video-60psnr'
    output_path_tortilla = datasets_path / 'dynamicearthnet-video-final' / 'dynamicearthnet-tortilla'
    output_path_taco = datasets_path / 'dynamicearthnet-video-final' / 'DynamicEarthNet-video.taco'
    images_output_path= Path('./dynamicearthnet_images-60psnr')
    
    # Set the video parameters
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
                    ][1] #Choose the params
    lossless_params= { 'c:v':'ffv1' }
    conversion_rules= {
        'bands': (['R', 'G', 'B', 'NIR'], ('time', 'y', 'x'), 0, lossy_params, 12),
        'labels': ('labels', ('time_month', 'y', 'x'), 0, lossless_params, 8),
        }
    skip_existing= True
    
    # Transform the original dataset (after downloading and unzipping) to xarray
    if False:
        output_path_xarray.mkdir(parents=True, exist_ok=True)
        to_xarray(planet_root, output_path_xarray, label_dirs, skip_existing=skip_existing)
        generate_id_files(input_path_subsets, output_path_xarray)
    
    # Transform the xarray dataset to xarrayvideo
    if False:
        output_path_video.mkdir(parents=True, exist_ok=True)
        to_video(output_path_xarray, output_path_video, images_output_path, conversion_rules,
                 skip_existing=skip_existing)
        generate_id_files(input_path_subsets, output_path_video)
        
    # Save each cube in a tortilla
    if False:
        output_path_tortilla.mkdir(parents=True, exist_ok=True)
        id_to_split_map = generate_id_to_split_map(output_path_video)
        to_tortilla(output_path_video, output_path_tortilla, id_to_split_map)
    
    # Save all tortillas in a taco
    if True:
        output_path_taco.parent.mkdir(parents=True, exist_ok=True)
        if output_path_taco.exists():
            raise FileExistsError(f'{output_path_taco} already exists')
        if output_path_taco.is_dir():
            raise FileExistsError(f'{output_path_taco} is already a directory')
        to_taco(output_path_tortilla, output_path_taco)