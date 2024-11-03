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

if __name__ == '__main__':
    # Define the base directory where the .mkv files are located
    base_dir = Path('/scratch/users/databases/deepextremes-video-XXpsnr')

    # Get all the .mkv files in the directory
    mkv_files = list(base_dir.glob('*/*.mkv'))

    # Iterate over all .mkv files with a progress bar
    for mkv_file in (bar:=tqdm(mkv_files, desc="Processing MKV files", unit="file")):
        # Get the metadata of the video file
        probe = ffmpeg.probe(mkv_file)

        # Find the XARRAY tag and modify it if present
        xarray_tag = None
        for stream in probe['format']['tags']:
            if 'XARRAY' in stream:
                xarray_tag = probe['format']['tags']['XARRAY']
                break

        # Flag for whether an update is needed
        update_needed = False

        # If the XARRAY tag exists, replace the 'COORDS_DIMS' part
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
            bar.set_description(f"Processing MKV files (updated: {mkv_file})")
        else:
            bar.set_description(f"Processing MKV files (no updates: {mkv_file})")
            print(f'Warning: {mkv_file} was not updated')