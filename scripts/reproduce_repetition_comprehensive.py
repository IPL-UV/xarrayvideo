
import xarray as xr
import numpy as np
from xarrayvideo import xarray2video
import logging
import os
import shutil
import argparse

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

# Create synthetic data with variable number of bands
np.random.seed(42)
times = 30
y_dim = 128
x_dim = 128

def create_dataset(n_bands):
    """Create dataset with n_bands"""
    bands = [f'B{i+1}' for i in range(n_bands)]
    data = {}
    for b in bands:
        arr = np.cumsum(np.random.randn(times, y_dim, x_dim), axis=0)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        data[b] = (('time', 'y', 'x'), arr)
    return xr.Dataset(data)

def get_dir_size(start_path):
    """Get total size of .mkv files in directory"""
    total_size = 0
    if not os.path.exists(start_path):
        return 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            if f.endswith('.mkv'):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # Mb

def build_repetition_dataset(ds, band_list):
    """
    Manually build repetition by repeating the LAST channel's values (not zeros!)
    to make channels divisible by 3. This is the true "repetition" method.
    
    Note: xarrayvideo's built-in "repetition" actually pads with zeros (multiplies by 0),
    so this tests TRUE repetition (repeating actual values) which should compress better
    than zero-padding when the last channel has structure.
    
    Returns a new dataset with repeated bands and the list of band names to use.
    """
    n_bands = len(band_list)
    repeats = 3 - (n_bands % 3)
    if repeats == 3:
        # Already divisible by 3, no padding needed
        return ds, band_list
    
    # Create repeated bands (repeat the last channel's actual values)
    ds_rep = ds.copy()
    last_band_data = ds[band_list[-1]].values  # Get the actual values from last band
    padded_bands = list(band_list)
    
    for i in range(repeats):
        repeat_band_name = f'Repeat_{i}'
        ds_rep[repeat_band_name] = (('time', 'y', 'x'), last_band_data.copy())
        padded_bands.append(repeat_band_name)
    
    return ds_rep, padded_bands

def build_zero_padding_mapping(ds, band_list):
    """
    Manually build zero padding by splitting bands into groups of 3,
    padding the last group with zeros if needed.
    
    IMPORTANT: When bands are divisible by 3, we use a SINGLE mapping (like repetition)
    to ensure fair comparison. Only when padding is needed do we split into multiple mappings.
    This tests the true difference: repetition of values vs zero padding.
    
    Returns the mapping dictionary and a dataset with zero bands added.
    """
    n_bands = len(band_list)
    ds_zero = ds.copy()
    
    # Add a zero band if we don't have one
    if 'Zero' not in ds_zero:
        zero_shape = ds[band_list[0]].shape
        ds_zero['Zero'] = (('time', 'y', 'x'), np.zeros(zero_shape))
    
    mapping = {}
    
    # If bands are divisible by 3, use a single mapping (like repetition does)
    # This ensures fair comparison - same number of mappings, same structure
    if n_bands % 3 == 0:
        # Single mapping with all bands - xarrayvideo will split internally
        # This matches the repetition approach structure-wise
        mapping['video_zero'] = (
            band_list,
            ('time', 'y', 'x'),
            0,
            {'c:v': 'libx265', 'preset': 'medium', 'crf': 51, 'x265-params': 'qpmin=0:qpmax=0.01'},
            12
        )
    else:
        # Need padding - split into groups of 3, padding last group
        video_idx = 1
        for i in range(0, n_bands, 3):
            group = band_list[i:i+3]
            # Pad with zeros if needed
            while len(group) < 3:
                group.append('Zero')
            
            video_name = f'video_{video_idx}'
            mapping[video_name] = (
                group,
                ('time', 'y', 'x'),
                0,
                {'c:v': 'libx265', 'preset': 'medium', 'crf': 51, 'x265-params': 'qpmin=0:qpmax=0.01'},
                12
            )
            video_idx += 1
    
    return ds_zero, mapping

def test_compression(n_bands, output_dir):
    """Test compression for a given number of bands"""
    print(f"\n{'='*60}")
    print(f"Testing with {n_bands} bands")
    print(f"{'='*60}")
    
    ds = create_dataset(n_bands)
    band_list = [f'B{i+1}' for i in range(n_bands)]
    
    # Clean up any existing files from previous runs
    case1_name = f'case1_{n_bands}bands'
    case2_name = f'case2_{n_bands}bands'
    import glob
    import shutil
    for case_name in [case1_name, case2_name]:
        case_dir = os.path.join(output_dir, case_name)
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
    
    # Case 1: Repetition (manually built outside)
    ds_rep, padded_bands = build_repetition_dataset(ds, band_list)
    
    # Calculate expected number of videos for repetition
    n_videos_rep = len(padded_bands) // 3
    
    mapping_rep = {
        'video_rep': (
            padded_bands,
            ('time', 'y', 'x'),
            0,
            {'c:v': 'libx265', 'preset': 'medium', 'crf': 51, 'x265-params': 'qpmin=0:qpmax=0.01'},
            12
        )
    }
    
    print(f"  Case 1 (Repetition): Compressing {n_bands} bands (padded to {len(padded_bands)}, {n_videos_rep} video(s))...")
    try:
        xarray2video(ds_rep, case1_name, mapping_rep, compute_stats=False, 
                    output_path=output_dir, verbose=False, save_dataset=False)
        size_rep = get_dir_size(os.path.join(output_dir, case1_name))
        # Count actual video files
        import glob
        actual_videos_rep = len(glob.glob(os.path.join(output_dir, case1_name, '*.mkv')))
        print(f"    Size: {size_rep:.4f} Mb ({actual_videos_rep} video files)")
    except Exception as e:
        print(f"    Failed: {e}")
        size_rep = 0
        actual_videos_rep = 0
    
    # Case 2: Zero Padding (manually built outside)
    ds_zero, mapping_zero = build_zero_padding_mapping(ds, band_list)
    
    # Debug: show what mappings we're creating
    print(f"  Case 2 (Zero Padding): Compressing {n_bands} bands in {len(mapping_zero)} mapping(s)...")
    for map_name, map_config in mapping_zero.items():
        bands_in_map = map_config[0]
        print(f"    Mapping '{map_name}': {bands_in_map}")
    try:
        xarray2video(ds_zero, case2_name, mapping_zero, compute_stats=False,
                    output_path=output_dir, verbose=False, save_dataset=False)
        size_zero = get_dir_size(os.path.join(output_dir, case2_name))
        # Count actual video files
        import glob
        actual_videos_zero = len(glob.glob(os.path.join(output_dir, case2_name, '*.mkv')))
        print(f"    Size: {size_zero:.4f} Mb ({actual_videos_zero} video files)")
    except Exception as e:
        print(f"    Failed: {e}")
        size_zero = 0
        actual_videos_zero = 0
    
    # Debug: Show the difference
    if size_rep > 0 and size_zero > 0:
        print(f"    Debug: Repetition uses {actual_videos_rep} videos, Zero-padding uses {actual_videos_zero} videos")
        # Show individual file sizes and names
        import glob
        rep_files = sorted(glob.glob(os.path.join(output_dir, case1_name, '*.mkv')))
        zero_files = sorted(glob.glob(os.path.join(output_dir, case2_name, '*.mkv')))
        if rep_files:
            rep_sizes = [os.path.getsize(f) / (1024*1024) for f in rep_files]
            rep_names = [os.path.basename(f) for f in rep_files]
            print(f"    Repetition files: {rep_names}")
            print(f"    Repetition file sizes: {[f'{s:.4f}' for s in rep_sizes]}")
        if zero_files:
            zero_sizes = [os.path.getsize(f) / (1024*1024) for f in zero_files]
            zero_names = [os.path.basename(f) for f in zero_files]
            print(f"    Zero-padding files: {zero_names}")
            print(f"    Zero-padding file sizes: {[f'{s:.4f}' for s in zero_sizes]}")
        
        # For bands divisible by 3, they should be identical
        if n_bands % 3 == 0:
            if actual_videos_rep != actual_videos_zero:
                print(f"    WARNING: Different number of videos! Expected same for {n_bands} bands (divisible by 3)")
            elif abs(size_rep - size_zero) > 0.001:
                print(f"    WARNING: Different sizes despite same structure! Rep={size_rep:.4f}, Zero={size_zero:.4f}")
    
    # Compare
    if size_rep > 0 and size_zero > 0:
        ratio = size_rep / size_zero
        if ratio > 1:
            print(f"  Result: Zero Padding is {ratio:.2f}x more efficient")
        else:
            print(f"  Result: Repetition is {1/ratio:.2f}x more efficient")
        return (n_bands, size_rep, size_zero, ratio)
    else:
        print(f"  Result: Could not compare (one or both failed)")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test repetition vs zero padding for arbitrary number of bands')
    parser.add_argument('--bands', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                        help='Number of bands to test (can specify multiple, e.g., --bands 4 5 7 10)')
    parser.add_argument('--output-dir', type=str, default='repetition_test_results',
                        help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Define output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for n_bands in args.bands:
        result = test_compression(n_bands, output_dir)
        if result is not None:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'Bands':<8} {'Repetition (Mb)':<18} {'Zero-Pad (Mb)':<18} {'Ratio (Rep/Zero)':<18}")
        print("-" * 60)
        for n_bands, size_rep, size_zero, ratio in results:
            print(f"{n_bands:<8} {size_rep:<18.4f} {size_zero:<18.4f} {ratio:<18.2f}")

if __name__ == '__main__':
    main()
