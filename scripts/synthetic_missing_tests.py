import xarray as xr
import numpy as np
import pickle
from pathlib import Path
from xarrayvideo import xarray2video, gap_fill
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl

# Setup matplotlib
mpl.rcParams.update(mpl.rcParamsDefault)    
plt.style.use('ggplot')
mpl.rcParams.update({'font.size': 10})

def run_synthetic_missing_tests():
    input_path = Path('../cubos_julio/cubo1_pickle')
    print(f"Loading {input_path}...")
    with open(input_path, 'rb') as file:
        data = pickle.load(file)
    
    minicube_orig = data.to_dataset(dim='band')
    
    # Just use RGB to be fast
    bands = [1, 2, 3]
    minicube_orig = minicube_orig[bands]
    
    # missing_rates = [0.0, 0.1, 0.3, 0.5, 0.8]
    missing_rates = [0.8]
    drop_modes = ['pixel', 'frame']
    gapfill_modes = [True, False]
    
    results = []
    
    test_params = {
        'c:v': 'libx265', 
        'preset': 'medium', 
        'tune': 'psnr', 
        'crf': 51, 
        'x265-params': 'qpmin=0:qpmax=0.01'
    }
    
    conversion_rules = {
        'rgb': (bands, ('time', 'y', 'x'), False, test_params, 12)
    }

    metrics_value_range = (0, 2**16-1)
    
    for rate in missing_rates:
        for drop_mode in drop_modes:
            if rate == 0.0 and drop_mode == 'frame':
                continue # Only run rate=0.0 once
                
            for use_gapfill in gapfill_modes:
                print(f"\n--- Running: rate={rate*100}%, mode={drop_mode}, gapfill={use_gapfill} ---")
                minicube = minicube_orig.copy(deep=True)
                
                if rate > 0:
                    np.random.seed(42)
                    
                    if drop_mode == 'pixel':
                        for b in bands:
                            vals = minicube[b].values.astype(np.float32)
                            mask = np.random.rand(*vals.shape) < rate
                            vals[mask] = np.nan
                            minicube[b].values = vals
                    elif drop_mode == 'frame':
                        time_len = minicube[bands[0]].shape[0]
                        frames_to_drop = np.random.rand(time_len) < rate
                        for b in bands:
                            vals = minicube[b].values.astype(np.float32)
                            vals[frames_to_drop, :, :] = np.nan
                            minicube[b].values = vals
                
                if use_gapfill:
                    print("Applying gap_fill...")
                    # gap_fill requires valid missing mask but we'll ignore it for the test
                    minicube = gap_fill(minicube, fill_bands=bands, mask_band=None,
                                        fill_values=None, method='last_value', new_mask='invalid', 
                                        coord_names=('time', 'variable', 'y', 'x'))
                    # Remove the invalid mask as requested (don't include its weight)
                    minicube = minicube.drop_vars('invalid')
                elif rate > 0:
                    # If no gapfill, replace NaNs with 0 to allow compression
                    for b in bands:
                        vals = minicube[b].values
                        vals[np.isnan(vals)] = 0
                        minicube[b].values = vals
                        
                print("Compressing...")
                out_dir = Path('./testing_synthetic/')
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                    
                try:
                    res = xarray2video(minicube, f'cube_missing_{rate}', conversion_rules, 
                                       compute_stats=True, output_path=out_dir, 
                                       loglevel='quiet', verbose=False, save_dataset=False,
                                       metrics_value_range=metrics_value_range)
                    
                    bpppb = res['rgb']['bpppb']
                    psnr = res['rgb']['psnr']
                    size = res['rgb']['compressed_size']
                    
                    res_dict = {
                        'rate': rate,
                        'drop_mode': 'none' if rate == 0.0 else drop_mode,
                        'use_gapfill': use_gapfill,
                        'psnr': psnr,
                        'bpppb': bpppb,
                        'size_mb': size
                    }
                    results.append(res_dict)
                    print(f"Result: {res_dict}")
                except Exception as e:
                    print(f"Failed: {e}")

    print("\n--- Final Results ---")
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('synthetic_missing_results_0.8.csv', index=False)

if __name__ == '__main__':
    run_synthetic_missing_tests()
