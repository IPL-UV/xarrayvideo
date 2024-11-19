'''
Fixes a bug I had with xarray2video when creating the deepextremes dataset.
Now the bug is fixed, and this fix should no longer be needed.
But I leave it here in case we want to modify the metadata of a dataset for whatever reason.
'''

import ffmpeg
from pathlib import Path
from tqdm import tqdm
import shutil
import tempfile
import os
from datetime import datetime

# Define the base directory where the .mkv files are located
base_dir = Path('/scratch/users/databases/deepextremes-video-XXpsnr')

# Specify the cutoff timestamp
cutoff_time = datetime(2024, 10, 26, 20, 49)

# Get all the folders in the base directory
folders = [folder for folder in base_dir.iterdir() if folder.is_dir()]

# Filter folders based on the creation time
folders = [
    folder for folder in folders 
    if datetime.fromtimestamp(os.path.getctime(folder)) < cutoff_time
]

# Iterate over each filtered folder with a progress bar
for folder in (bar:=tqdm(folders, desc="Processing Folders", unit="folder")):
    mkv_files = list(folder.glob('*.mkv'))  # Get all .mkv files in this folder
    
    # Flag for whether an update is needed
    update_needed = False

    for mkv_file in mkv_files:
        # Get the metadata of the video file
        probe = ffmpeg.probe(mkv_file)

        # Find the XARRAY tag and modify it if present
        xarray_tag = None
        for stream in probe['format']['tags']:
            if 'XARRAY' in stream:
                xarray_tag = probe['format']['tags']['XARRAY']
                break

        # Check if the XARRAY tag needs modification
        # if mkv_file.name.startswith("scl"):
        if xarray_tag and "('time', 'y', 'x')" in xarray_tag:
            new_xarray_tag = xarray_tag.replace("('time', 'y', 'x')", "('time', 'x', 'y')")
            update_needed = True

            # Create a temporary file for output
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mkv') as temp_file:
                temp_file_name = temp_file.name

                # Use ffmpeg to update the metadata tag and write to a temporary file
                (
                    ffmpeg
                    .input(str(mkv_file))
                    .output(temp_file_name,  # Output to a temp file
                            codec='copy',  # Ensures no re-encoding
                            metadata=f'XARRAY={new_xarray_tag}'
                           )
                    .run(overwrite_output=True, quiet=True)
                )

                # Replace the original file with the updated temporary file
                shutil.move(temp_file_name, mkv_file)

        # Update the progress bar description based on whether there was an update
        if update_needed:
            bar.set_description(f"Processing Folders (updated: {folder})")
        else:
            bar.set_description(f"Processing Folders (no updates: {folder})")