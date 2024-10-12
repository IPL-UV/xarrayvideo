'''
Instructions for downlading the dynamicearthnet data and running the repo https://github.com/OscarPellicer/dynnet:

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
    
3. Download the weights into the repo: 
    cd ~/dynnet
    wget -r -np -nH --cut-dirs=4 -R "index.html*" https://cvg.cit.tum.de/webshare/u/toker/dynnet_ckpt/

4. Edit config/defaults.yaml > DATA > ROOT to point to dynamicearthnet path

6. Run inference: 
    python inference.py --config config/defaults.yaml --gpu_ids 0 --resume ./weights/3dconv/weekly/best_ckpt.pth
'''

import xarray as xr
import rasterio
from pathlib import Path
import numpy as np
import datetime
from tqdm import tqdm

from xarrayvideo import to_netcdf
from xarrayvideo import plot_image

def find_label_file(label_dir, location_id, location_short, date):
    """Find a label file for a given location and date."""
    location_folder = label_dir / f"{location_id}_{location_short}" / "Labels" / "Raster"
    date_str = date.strftime("%Y-%m-%d")
    label_file = next(location_folder.glob(f"*/*{date_str}.tif"), None)
    return label_file

def create_xarray_for_location(location_path, label_dirs, band_names=['R', 'G', 'B', 'NIR']):
    """Creates an xarray.DataArray for all the .tif files of a given location."""
    data_arrays = []
    times = []
    labels = []
    label_times = []

    image_files = sorted(location_path.glob("*.tif"))
    
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
                        label_data = label_src.read(1)  # Read the first band of the label
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

    return dataset

def to_xarray(planet_root, output_path, label_dirs, skip_existing=True):
    """Process all planet folders and save each location's data as a .zarr file."""
    planet_folders = list(planet_root.glob("planet.*"))

    # Add a progress bar for the overall location processing
    for planet_folder in tqdm(planet_folders, desc="Processing all planet folders"):
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
            
def extract_unique_ids(file_path):
    unique_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            # Extract the identifier from the second part of the line
            if len(parts) > 1:
                id_part = parts[1].split('/')[2]  # Extracting the identifier
                unique_ids.add(id_part)
    return list(unique_ids)

def generate_unique_id_files(input_path_subsets, output_path_subsets):
    # Convert string paths to Path objects
    input_path = Path(input_path_subsets)
    output_path = Path(output_path_subsets)

    # List of subset filenames to process
    subsets = ['train.txt', 'val.txt', 'test.txt']

    for subset in subsets:
        input_file_path = input_path / subset
        output_file_path = output_path / subset

        unique_ids = extract_unique_ids(input_file_path)
        print(f'{subset}: {len(unique_ids)} elements')

        # Write unique identifiers to the output file
        with output_file_path.open('w') as output_file:
            for unique_id in sorted(unique_ids):  # Optionally sort the unique IDs
                output_file.write(f"{unique_id}\n")
            
def to_video(input_path, output_path, images_output_path, conversion_rules, debug):
    
    files= list(input_path.glob('*.nc'))
    
    # Run for all cubes
    for i, input_path in (pbar:=tqdm(enumerate(files), total=len(files))):
        try:
            # Print name
            array_id= '_'.join(input_path.stem)
            pbar.set_description(array_id)

            # Load
            minicube= xr.open_dataset(input_path)
            
            # Compress
            arr_dict= xarray2video(minicube, array_id, conversion_rules,
                           output_path=output_path, compute_stats=False, 
                           exceptions='ignore', loglevel='quiet',
                           )  

            # Plot image every 10
            if i%10 == 0:
                minicube_new= video2xarray(dataset_out_path, array_id) 
                plot_image(minicube_new, list('RGB'), 
                           save_name=str(images_output_path/f'{array_id}.jpg'), 
                           show=False, mask_name=None)

        except Exception as e:
            print(f'Exception processing {array_id=}: {e}')

if __name__ == '__main__':
    # Set the paths
    datasets_path= Path("/scratch/users/databases/")
    planet_root = datasets_path / 'dynamicearthnet'
    output_path_xarray = datasets_path / 'dynamicearthnet-xarray'
    output_path_video = datasets_path / 'dynamicearthnet-video'
    input_path_subsets = datasets_path / 'dynamicearthnet' / 'dynnet_training_splits'
    images_output_path= Path('./dynamicearthnet_images')
    label_dirs = [planet_root / "dynamicearthnet_test_labels", planet_root / "labels" / "labels"]
    
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
                    ][0] #Choose the params
    lossless_params= { 'c:v':'ffv1' }
    conversion_rules= {
        'bands': (['NIR', 'R', 'G', 'B'], ('time', 'y', 'x'), 0, lossy_params, 12),
        'labels': ('labels', ('time_month', 'y', 'x'), 0, lossless_params, 8),
        }
    
    # Transform the original dataset (after downloading and unzipping) to xarray
    output_path_xarray.mkdir(exist_ok=True)
    to_xarray(planet_root, output_path_xarray, label_dirs)
    
    # Extract subsets from train.txt, val.txt, test.txt
    generate_unique_id_files(input_path_subsets, output_path_xarray)
    
    # Transform the xarray dataset to xarrayvideo
    output_path_video.mkdir(exist_ok=True)
    to_video(output_path_xarray, output_path_video, images_output_path, conversion_rules, debug)