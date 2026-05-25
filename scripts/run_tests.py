'''
Examples:
python scripts/run_tests.py --dataset dynamicearthnet --rules_name 4channels2
python scripts/run_tests.py --dataset deepextremes --rules_name gapfill3
python scripts/run_tests.py --dataset custom --rules_name pca
python scripts/run_tests.py --dataset era5 --rules_name all

To generate sample images:
python scripts/run_tests.py --dataset dynamicearthnet --rules_name img --id 8077_5007 --plot_samples
python scripts/run_tests.py --dataset deepextremes --rules_name img --id 10.38_50.15 --plot_samples
python scripts/run_tests.py --dataset custom --rules_name img --id cubo1 --plot_samples
python scripts/run_tests.py --dataset era5 --rules_name img --plot_samples

To run on slurm:
cd ~
sbatch run_test_xarrayvideo.sh 
'''

import xarray as xr
import numpy as np
from pathlib import Path
import re
from PIL import Image
from xarrayvideo import xarray2video, video2xarray, gap_fill, plot_image, to_netcdf
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings, shutil
import pickle
import pandas as pd
import seaborn as sns
import os

#Reset matplotlib config and change defaults
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib import colors as mcolors
import re
import colorsys
mpl.rcParams.update(mpl.rcParamsDefault)    
plt.style.use('ggplot')
mpl.rcParams.update({'font.size': 10})

#Now plot it! 
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm

rng= np.random.default_rng(seed=42)


def get_external_codec_threads():
    env_value = os.environ.get('XV_CODEC_THREADS') or os.environ.get('SLURM_CPUS_PER_TASK')
    if env_value:
        return max(1, int(env_value))
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)

def get_unique_save_stem(base_path):
    candidate = base_path
    suffix = 1
    while any(candidate.parent.glob(f'{candidate.name}*')):
        candidate = base_path.with_name(f'{base_path.name}_rerun{suffix}')
        suffix += 1
    return candidate


def results_dict_to_dataframe(overall_results, metrics_keep):
    results_metrics = {}
    for cube, cube_results in overall_results.items():
        for test, test_results in cube_results.items():
            for crf, crf_results in test_results.items():
                for bits, bits_results in crf_results.items():
                    for video_name, metadata in bits_results.items():
                        if isinstance(metadata, dict):
                            for metric in metrics_keep:
                                try:
                                    results_metrics[(metric, video_name, test, bits, cube, crf)] = metadata[metric]
                                except Exception as e:
                                    print('Exception:', metadata['path'][0], e)
                                    if 'acc' in metadata.keys():
                                        print(f'{metadata["acc"]=}')
                                        if metadata['acc'] > 0.99999:
                                            print('Assigning best possible value!')
                                            results_metrics[(metric, video_name, test, bits, cube, crf)] = {
                                                'psnr': 100,
                                                'mse': 0,
                                                'exp_sa': 0,
                                                'ssim': 1.,
                                            }[metric]

    results_df = pd.DataFrame.from_dict(results_metrics, orient='index', columns=['value'])
    results_df.index = pd.MultiIndex.from_tuples(results_df.index)
    results_df.index.names = ['metric', 'video_name', 'test_name', 'bits', 'cube', 'crf']
    return results_df


def save_checkpoint_results(overall_results, temp_results_path, save_name, metrics_keep):
    with open(temp_results_path, 'wb') as temp_file:
        pickle.dump(overall_results, temp_file)

    results_df = results_dict_to_dataframe(overall_results, metrics_keep)
    results_df.to_pickle(f'{save_name}.pkl')

def parse_args():
    parser = argparse.ArgumentParser(description="Process command-line options")

    parser.add_argument(
        "--continue_from_temp",
        action="store_true",
        default=False,
        help="If true, load 'results_temp.pkl' and just save/plot whatever is there."
    )
    
    parser.add_argument(
        "--use_saved",
        action="store_true",
        help="If true, use saved .pkl results data (not temp, but final saves); otherwise, run the tests."
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="If true, only one cube per test is performed, and errors are raised."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['deepextremes', 'dynamicearthnet', 'custom', 'era5'],
        default="custom",
        help="Specify the dataset to use. Choices are 'deepextremes', 'dynamicearthnet', 'custom', 'era5'."
    )
    
    parser.add_argument(
        "--rules_name",
        type=str,
        default="11channels",
        help="Specify rule name, e.g., '11channels', '7channels', etc., for save name."
    )
    
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Specify a cube id to load and process."
    )
        
        
    parser.add_argument(
        "--plot_samples",
        action="store_true",
        default=False,
        help="If true, plot some example images."
    )

    parser.add_argument(
        "--plot_pca",
        action="store_true",
        default=False,
        help="If true, include PCA variants in the generated plots when available."
    )

    parser.add_argument(
        "--codec_names",
        type=str,
        default=None,
        help="Comma-separated codec test names to run, e.g. 'libx265,vp9,vvenc,uavs3e'."
    )

    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional maximum number of cubes/samples to benchmark for the selected dataset."
    )

    parser.add_argument(
        "--quality_limit",
        type=int,
        default=None,
        help="Optional maximum number of quality presets to test per codec."
    )

    parser.add_argument(
        "--skip_plot",
        action="store_true",
        default=False,
        help="If true, skip plot generation."
    )

    parser.add_argument(
        "--skip_latex",
        action="store_true",
        default=False,
        help="If true, skip LaTeX table generation."
    )

    parser.add_argument(
        "--extra_saved_results",
        type=str,
        default=None,
        help="Comma-separated additional saved result .pkl files or stems to merge when using --use_saved."
    )

    return parser.parse_args()

import argparse

BIT_LINESTYLES = {8: ':', 10: '--', 12: '-', 16: '-'}
CODEC_COLORS = {
    'libx265': '#1f77b4',
    'vp9': '#8c564b',
    'JP2OpenJPEG': '#bcbd22',
    'vvenc': '#ff7f0e',
    'uavs3e': '#17becf',
    'ffv1': '#7f7f7f',
    'libx265 (PCA)': '#9467bd',
    'libx265 (PCA - 3 bands)': '#e377c2',
    'libx265 (PCA - 9 bands)': '#d62728',
    'libx265 (PCA - 12 bands)': '#2ca02c',
    'libx265 (PCA - all bands)': '#8c6d31',
}
VIDEO_MARKERS = {
    'rgb': 'o',
    'ir3': 's',
    'bands': 'o',
    'all': 'o',
    'relative_humidity': '^',
    'wind_speed': 's',
    'masks': 'D',
}
VIDEO_ORDER = {
    'rgb': 0,
    'bands': 0,
    'all': 0,
    'ir3': 1,
    'relative_humidity': 0,
    'wind_speed': 1,
    'masks': 2,
}
VIDEO_COLOR_VARIANTS = {
    0: (0.0, 0.00),
    1: (0.06, -0.12),
    2: (-0.06, 0.12),
    3: (0.12, -0.06),
    4: (-0.12, 0.06),
}
CODEC_ORDER = {
    'libx265': 0,
    'vp9': 1,
    'JP2OpenJPEG': 2,
    'vvenc': 3,
    'uavs3e': 4,
    'ffv1': 5,
    'libx265 (PCA)': 6,
    'libx265 (PCA - 3 bands)': 7,
    'libx265 (PCA - 9 bands)': 8,
    'libx265 (PCA - 12 bands)': 9,
    'libx265 (PCA - all bands)': 10,
}


def get_codec_color(test_name):
    return CODEC_COLORS.get(test_name, '#333333')


def adjust_color_for_video(base_color, video_name):
    hue_shift, lightness_shift = VIDEO_COLOR_VARIANTS.get(VIDEO_ORDER.get(video_name, 0), (0.0, 0.0))
    red, green, blue = mcolors.to_rgb(base_color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    hue = (hue + hue_shift) % 1.0
    lightness = min(0.82, max(0.22, lightness + lightness_shift))
    adjusted = colorsys.hls_to_rgb(hue, lightness, saturation)
    return adjusted


def empty_legend_handle():
    return plt.Line2D([], [], color='none')


def get_video_marker(video_name):
    return VIDEO_MARKERS.get(video_name, 'o')


def parse_plot_label(label):
    match = re.match(r'(.+?)\s+(\d+)bits\s+(.+)', label)
    if not match:
        return None
    video_name, bits, codec_name = match.groups()
    return video_name.lower(), int(bits), codec_name

def format_legend_items(handles, labels, dataset, plot_pca):
    max_rows = 6
    default_ncols = {'deepextremes': 4, 'dynamicearthnet': 3, 'custom': 4, 'era5': 4}.get(dataset, 4)

    parsed_items = []
    fallback_items = []
    for handle, label in zip(handles, labels, strict=False):
        parsed = parse_plot_label(label)
        if parsed is None:
            fallback_items.append((handle, label))
            continue
        video_name, bits, codec_name = parsed
        if '(PCA' in codec_name and not plot_pca:
            continue
        parsed_items.append(((VIDEO_ORDER.get(video_name, 99), CODEC_ORDER.get(codec_name, 99), -bits), handle, label))

    ordered_items = [(handle, label) for _, handle, label in sorted(parsed_items, key=lambda item: item[0])]
    ordered_items.extend(fallback_items)

    if not ordered_items:
        return handles, labels, default_ncols

    nrows = min(max_rows, max(1, len(ordered_items)))
    ncols = max(default_ncols, int(np.ceil(len(ordered_items) / nrows)))
    columns = []
    for start in range(0, len(ordered_items), nrows):
        column = ordered_items[start:start + nrows]
        if len(column) < nrows:
            column = column + [(empty_legend_handle(), '')] * (nrows - len(column))
        columns.append(column)

    if len(columns) < ncols:
        columns.extend([[(empty_legend_handle(), '')] * nrows for _ in range(ncols - len(columns))])

    reordered_handles = []
    reordered_labels = []
    for column in columns:
        for handle, label in column:
            reordered_handles.append(handle)
            reordered_labels.append(label)
    return reordered_handles, reordered_labels, ncols

def get_conversion_rules(dataset, test, param, bits, **kwargs):
    if dataset == 'deepextremes':
        if test == 'ffv1':
            conversion_rules= {
                # '7 bands': ( bands, ('time','x','y'), 
                #             len(bands) if 'PCA' in test else False, param, bits),
                # 'rgb': ( ('B04','B03','B02'), ('time','y','x'), False, param, bits),
                # 'ir3': ( ('B8A','B06','B05'), ('time','y','x'), False, param, bits),
                'masks': ( ('SCL', 'cloudmask_en', 'invalid'), ('time','y','x'), 0, param, bits),
                }
        elif test == 'default':
             conversion_rules= {
                # '7 bands': ( bands, ('time','x','y'), 
                #             len(bands) if 'PCA' in test else False, param, bits),
                'rgb': ( ('B04','B03','B02'), ('time','y','x'), 
                         3 if '(PCA)' in test else False, param, bits),
                'ir3': ( ('B8A','B06','B05'), ('time','y','x'), 
                         3 if '(PCA)' in test else False, param, bits),
                'masks': ( ('SCL', 'cloudmask_en', 'invalid'), ('time','y','x'), 0, param, bits),
                }  
        else:
             conversion_rules= {
                # '7 bands': ( bands, ('time','x','y'), 
                #             len(bands) if 'PCA' in test else False, param, bits),
                'rgb': ( ('B04','B03','B02'), ('time','y','x'), 
                         3 if '(PCA)' in test else False, param, bits),
                'ir3': ( ('B8A','B06','B05'), ('time','y','x'), 
                         3 if '(PCA)' in test else False, param, bits),
                # 'masks': ( ('SCL', 'cloudmask_en', 'invalid'), ('time','y','x'), 0, param, bits),
                }
    elif dataset == 'dynamicearthnet':
        conversion_rules= {
            'bands': (['R', 'G', 'B', 'NIR'], ('time', 'y', 'x'), 
                      4 if '(PCA)' in test else 3 if '(PCA - 3 bands)' in test else False, param, bits),
            }
    elif dataset == 'custom':
        conversion_rules= {
            # 'rgb': (['B4', 'B3', 'B2'], ('time', 'x', 'y'), False, param, bits),
            'all': (kwargs['bands'], ('time', 'y', 'x'), 
                    9 if '(PCA - 9 bands)' in test 
                    else len(kwargs['bands']) if '(PCA)' in test
                    else False, param, bits),
            }
    elif dataset == 'era5':
        conversion_rules= {
        # 'wind': ( ('10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed'), 
                  # ('time', 'longitude', 'latitude'), False, param, bits),
        'relative_humidity': ('relative_humidity', ('time', 'longitude', 'latitude', 'level'),
           13 if '(PCA)' in test else 12 if '(PCA - 12 bands)' in test else False, param, bits),
        'wind_speed': ('wind_speed', ('time', 'longitude', 'latitude', 'level'), 
           13 if '(PCA)' in test else 12 if '(PCA - 12 bands)' in test else False, param, bits),
        'temp': ('temperature', ('time', 'longitude', 'latitude', 'level'), 
           13 if '(PCA)' in test else 12 if '(PCA - 12 bands)' in test else False, param, bits),
        # 'wind_u': ('u_component_of_wind', ('time', 'longitude', 'latitude', 'level'), 
        #    13 if '(PCA)' in test else 12 if '(PCA - 12 bands)' in test else False, param, bits),
            }
    else:
        raise RuntimeError(f'Unknown {dataset=}')
    return conversion_rules

if __name__ == '__main__':
    
    #____COFNIG____
    
    # Global config
    args= parse_args()
    CONTINUE_FROM_TEMP= args.continue_from_temp #ITf true, load 'results_temp.pkl' and just save / plot whatever is there
    USE_SAVED= args.use_saved #If True, use saved .pkl results data, otherwise run the tests
    DEBUG= args.debug #If True, only one cube per test is performed, and errors are risen
    DATASET= args.dataset #One of ['deepextremes', 'dynamicearthnet', 'custom', 'era5']
    RULES_NAME= args.rules_name #'11channels', '7channels', etc., just for save name
    PLOT_SAMPLES= args.plot_samples
    PLOT_PCA= args.plot_pca
    USER_SAMPLE_LIMIT = args.sample_limit
    QUALITY_LIMIT = args.quality_limit
    SKIP_PLOT = args.skip_plot
    SKIP_LATEX = args.skip_latex

    bands= None
    sample_limit = None
    if DATASET == 'deepextremes':
        #Take N random cubes
        # dataset_in_path= Path('/scratch/users/databases/deepextremes_final')
        dataset_in_path= Path('/data/databases/deepextremes_final')
        cube_paths= np.array(list(dataset_in_path.glob('*.zarr'))) # '*.zarr'
        rng.shuffle(cube_paths)
        bands= ['B04','B03','B02','B8A','B07','B06','B05']
        sample_limit = 10
    elif DATASET == 'dynamicearthnet':
        #Take N random cubes
        # dataset_in_path= Path('/scratch/users/databases/dynamicearthnet-xarray')
        dataset_in_path= Path('/data/databases/dynamicearthnet-xarray')
        cube_paths= np.array(list(dataset_in_path.glob('*.nc')))
        rng.shuffle(cube_paths)
        sample_limit = 5
    elif DATASET == 'custom':
        cube_paths= [Path(f'../cubos_julio/cubo{i}_pickle') for i in range(1,5)]
        bands= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    elif DATASET == 'era5':
        cube_paths= [Path(
        '../since_2022-07-01_2022-07-01_1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')]    
    else:
        raise RuntimeError(f'Unknown {DATASET=}')
        
    if args.id is not None:
        from copy import copy
        cube_paths_orig= copy(cube_paths)
        cube_paths= [c for c in cube_paths if args.id in str(c)]
        assert len(cube_paths), f'{args.id=} was not found in {cube_paths_orig[:10]}...'
    if USER_SAMPLE_LIMIT is not None:
        sample_limit = USER_SAMPLE_LIMIT

    if sample_limit is not None:
        cube_paths= cube_paths[:sample_limit]

    #Define all tests
    crfs= ['Best','Very high','High','Medium','Low','Very low']# + ['Very low 2', 'Very low 3'] 

    #x265 config
    x265_crfs_list= [51, #51, 51, 
                     51, 
                     1, 7, 16, 27, 
                     39, 39, 39, 39, 39, 39]
    x265_param_list= [ 'lossless=1', 
                       # 'qpmin=0:qpmax=0.0000001:psy-rd=0:psy-rdoq=0', 
                       'qpmin=0:qpmax=0.01', 
                       # 'qpmin=0:qpmax=0.1:psy-rd=0:psy-rdoq=0'
                     ] + ['']*10
    x265_params= [{'c:v': 'libx265', 'preset':'medium', 'tune':'psnr', 
                   'crf': x265_crfs_list[i], 'x265-params':x265_param_list[i]} 
                    for i in range(6)] #'tune:psnr' vs psy-rd=0:psy-rdoq=0
    x265_params_PCA= [ {'c:v': 'libx265', 'preset':'medium', 
                        'crf': [x265_crfs_list[i]] + [x265_crfs_list[i+j] for j in range(5)], 
                        'x265-params': [x265_param_list[i]] + [x265_param_list[i+j] for j in range(5)] } 
                       for i in range(6)]
    
    #hevc_nvenc config
    hevc_nvenc_params= [
        {'c:v': 'hevc_nvenc', 'preset': 'medium', 'tune': 'lossless', 'qp': 0},
        {'c:v': 'hevc_nvenc', 'preset': 'medium', 'tune': 'hq', 'qp': 0, 'qp-max': 0.01, 'qp-min': 0},
        {'c:v': 'hevc_nvenc', 'preset': 'medium', 'tune': 'hq', 'qp': 1},
        {'c:v': 'hevc_nvenc', 'preset': 'medium', 'tune': 'hq', 'qp': 7},
        {'c:v': 'hevc_nvenc', 'preset': 'medium', 'tune': 'hq', 'qp': 16},
        {'c:v': 'hevc_nvenc', 'preset': 'medium', 'tune': 'hq', 'qp': 27},
    ] 
    
    #VP9 config
    # vp9_crfs_list= [0, 0, 0, 1, 5, 10, 20, 30]
    # vp9_qmax_list= [0.0001, 0.01, 1, 10000, 10000, 10000]
    vp9_params= [
        {'c:v': 'vp9', 'crf': 0, 'lossless': 1},
        # {'c:v': 'vp9', 'crf': 0, 'arnr-strength': 2, 'qmin': 0, 'qmax': 0.0001, 
        #  'lag-in-frames': 25, 'arnr-maxframes': 7},#Same results as with 'qmax': 0.01
        {'c:v': 'vp9', 'crf': 0, 'arnr-strength': 2, 'qmin': 0, 'qmax': 0.01, 
         'lag-in-frames': 25, 'arnr-maxframes': 7}, #Same results as with 'qmax': 0.01
        # {'c:v': 'vp9', 'crf': 1, 'arnr-strength': 2, 'qmin': 0, 'qmax': 1, 
        #  'lag-in-frames': 25, 'arnr-maxframes': 7}, #Same results as with 'qmax': 0.01
        {'c:v': 'vp9', 'crf': 5, 'arnr-strength': 2, 'lag-in-frames': 25, 'arnr-maxframes': 7},
        {'c:v': 'vp9', 'crf': 12, 'arnr-strength': 2, 'lag-in-frames': 25, 'arnr-maxframes': 7},
        {'c:v': 'vp9', 'crf': 20, 'arnr-strength': 2, 'lag-in-frames': 25, 'arnr-maxframes': 7},
        {'c:v': 'vp9', 'crf': 30, 'arnr-strength': 2, 'lag-in-frames': 25, 'arnr-maxframes': 7},
      ]

    external_codec_threads = get_external_codec_threads()

    vvenc_qp_list = [0, 1, 7, 16, 27, 39]
    vvenc_params = [
        {
            'c:v': 'vvenc',
            'preset': 'medium',
            'qp': qp,
            'qpa': 0,
            'threads': external_codec_threads,
        }
        for qp in vvenc_qp_list
    ]

    uavs3_qp_list = [0, 1, 7, 16, 27, 39]
    uavs3_default_config = './scripts/uavs3e_ra.cfg'
    uavs3_params = [
        {
            'c:v': 'uavs3e',
            'config': uavs3_default_config,
            'qp': qp,
            'threads': external_codec_threads,
            'frm_threads': max(1, external_codec_threads // 4),
        }
        for qp in uavs3_qp_list
    ]

    vtm_qp_list = [0, 1, 7, 16, 27, 39]
    vtm_params = [
        {
            'c:v': 'vtm',
            'qp': qp,
            'wavefront_synchro': 1,
        }
        for qp in vtm_qp_list
    ]

    #JPEG200 params
    if DATASET == 'era5':
        jpeg2000_quality_list= [100, 25, 5, 1, 0.25, 0.05][::-1] #[100, 100, 75, 35, 15, 5, 3, 1]
    else:
        jpeg2000_quality_list= [100, 80, 35, 15, 5, 1] #[100, 100, 75, 35, 15, 5, 3, 1]
    jpeg2000_params= [ {'codec': 'JP2OpenJPEG', 'QUALITY': str(jpeg2000_quality_list[i]), 
                        'REVERSIBLE': 'YES' if i==0 else 'NO', 'YCBCR420':'NO'} for i in range(6)]

    # n_bits= [16]
    # n_bits= [8, 16] #[8,10,12,16] #[8,10,12,16], [12, 16]
    n_bits= [8,10,12,16]
    
    # tests= [ 'JP2OpenJPEG (PCA)'] 
    # tests= ['libx265']
    tests= [ 'JP2OpenJPEG', 'libx265', 'vp9']
    # tests= ['ffv1']
    # tests= ['libx265 (PCA - 9 bands)', 'libx265 (PCA - all bands)', 'libx265', 'vp9', 'JP2OpenJPEG']
    # tests= ['libx265', 'libx265 (PCA)', 'libx265 (PCA - 12 bands)', 'JP2OpenJPEG']
    
    # codec_params= dict(zip(tests, [jpeg2000_params], strict=True))
    # codec_params= dict(zip(tests, [ [{'c:v':'ffv1'}], x265_params, jpeg2000_params, vp9_params], strict=True))
    codec_params_map = {
        'JP2OpenJPEG': jpeg2000_params,
        'libx265': x265_params,
        'vp9': vp9_params,
        'vvenc': vvenc_params,
        'uavs3e': uavs3_params,
        'vtm': vtm_params,
    }
    codec_params= {test_name: codec_params_map[test_name] for test_name in tests}
    # codec_params= dict(zip(tests, [x265_params_PCA, x265_params_PCA, x265_params, vp9_params, jpeg2000_params], strict=True))
    # codec_params= dict(zip(tests, [jpeg2000_params], strict=True))
        
    # Example commands to run PCA tests across modalities:
    # DeepExtremeCubes: python scripts/run_tests.py --dataset deepextremes --rules_name pca
    # DynamicEarthNet:  python scripts/run_tests.py --dataset dynamicearthnet --rules_name pca
    # Custom/SimpleS2:  python scripts/run_tests.py --dataset custom --rules_name pca
    # ERA5:             python scripts/run_tests.py --dataset era5 --rules_name pca
    

    #/home/oscar/.conda/envs/xv/bin/python scripts/run_tests.py --dataset custom --rules_name cs-pca > cs-pca.log 2>&1; /home/oscar/.conda/envs/xv/bin/python scripts/run_tests.py --dataset dynamicearthnet --rules_name de-pca > de-pca.log 2>&1; /home/oscar/.conda/envs/xv/bin/python scripts/run_tests.py --dataset era5 --rules_name e5-pca > e5-pca.log 2>&1; /home/oscar/.conda/envs/xv/bin/python scripts/run_tests.py --dataset deepextremes --rules_name dx-pca > dx-pca.log 2>&1
    
    # If the user specifically wants to run a PCA test for the selected dataset, 
    # we can append the PCA variant to the list of tests depending on the 'rules_name' argument.
    if 'pca' in RULES_NAME.lower():
        if DATASET == 'custom':
            tests += ['libx265 (PCA)', 'libx265 (PCA - 9 bands)']
            codec_params = dict(zip(tests, [jpeg2000_params, x265_params, vp9_params, x265_params_PCA, x265_params_PCA], strict=True))
        elif DATASET == 'era5':
            tests += ['libx265 (PCA)', 'libx265 (PCA - 12 bands)']
            codec_params = dict(zip(tests, [jpeg2000_params, x265_params, vp9_params, x265_params_PCA, x265_params_PCA], strict=True))
        elif DATASET == 'dynamicearthnet':
            tests += ['libx265 (PCA)', 'libx265 (PCA - 3 bands)']
            codec_params = dict(zip(tests, [jpeg2000_params, x265_params, vp9_params, x265_params_PCA, x265_params_PCA], strict=True))
        else:
            tests += ['libx265 (PCA)']
            codec_params = dict(zip(tests, [jpeg2000_params, x265_params, vp9_params, x265_params_PCA], strict=True))

    if args.codec_names:
        tests = [test_name.strip() for test_name in args.codec_names.split(',') if test_name.strip()]
        codec_params = {test_name: codec_params_map[test_name] for test_name in tests}

    if QUALITY_LIMIT is not None:
        crfs = crfs[:QUALITY_LIMIT]
        codec_params = {test_name: params[:QUALITY_LIMIT] for test_name, params in codec_params.items()}
    
    # If you want to run ALL tests (including PCA) at once, uncomment the following block:
    # tests = ['JP2OpenJPEG', 'libx265', 'vp9', 'libx265 (PCA)']
    # codec_params = dict(zip(tests, [jpeg2000_params, x265_params, vp9_params, x265_params_PCA], strict=True))

    if PLOT_SAMPLES:
        tests = ['libx265']
        codec_params = {'libx265': x265_params}
        n_bits = [12]

    metrics_keep= ['compression', 'psnr', 'mse', 'bpppb', 'exp_sa', 'time', 'd_time', 'ssim']
    
    # Set save name and create path    
    requested_save_name_path = Path(f'./results/results_{DATASET}_{RULES_NAME}')
    save_name_path = get_unique_save_stem(requested_save_name_path)
    if save_name_path.name != requested_save_name_path.name:
        print(f'Existing results detected, saving this run as {save_name_path.name}')
    save_name = str(save_name_path)
    load_name = str(requested_save_name_path)
    temp_results_path = save_name_path.with_name(f'{save_name_path.name}_temp.pkl')
    Path(save_name).parent.mkdir(parents=True, exist_ok=True)
    Path('results/samples').mkdir(parents=True, exist_ok=True)
    
    run_testing_dir = Path('./testing') / re.sub(r'[^A-Za-z0-9_.-]+', '_', RULES_NAME)
    run_testing_dir.mkdir(parents=True, exist_ok=True)

    #_____TESTS____

    if not USE_SAVED and not CONTINUE_FROM_TEMP:
        overall_results={}
        for i, input_path in enumerate(pbar:=tqdm(cube_paths, total=len(cube_paths))):
            # For debugging
            if DEBUG and i > 0: break

            #Load data
            metrics_value_range= None #This is needed for comparable PSNR computation
            if DATASET == 'deepextremes':
                metrics_value_range= (0., (2**16-1) / 10000. )
                array_id= '_'.join(input_path.stem.split('_')[1:3])
                minicube= xr.open_dataset(input_path, engine='zarr')
                
                # Ensure required bands are present; skip cube if not
                if 'SCL' not in minicube or 'cloudmask_en' not in minicube:
                    print(f"Skipping cube {array_id} as it is missing 'SCL' or 'cloudmask_en'")
                    continue
                    
                if 'SCL' in minicube:
                    minicube['SCL']= minicube['SCL'].astype(np.uint8) #Fixes problem with the dataset
                if 'cloudmask_en' in minicube:
                    minicube['cloudmask_en']= minicube['cloudmask_en'].astype(np.uint8)
                # minicube= minicube.drop_vars('B07') #We drop a variable for now

                #Align
                #TODO: Alignment is not working properly as of now
                
                #Perform simple gap filling
                #Do not gapfill clouds, otherise use 'cloudmask_en'
                minicube= gap_fill(minicube, fill_bands=bands, mask_band=None,
                                  fill_values=[1, 3, 4], method='last_value', new_mask='invalid', 
                                  coord_names=('time', 'variable', 'y', 'x'))

            elif DATASET == 'dynamicearthnet':
                metrics_value_range= (-2**15, 2**15-1)
                # metrics_value_range= (0, 10000) # Max values are around 13484
                minicube= xr.open_dataset(input_path)
                array_id= input_path.stem

            elif DATASET == 'custom':
                metrics_value_range= (0., 2**16-1)
                with open(input_path, 'rb') as file:
                    data = pickle.load(file)
                minicube= data.to_dataset(dim='band')
                bands= list(minicube.data_vars)
                array_id= file.name

            elif DATASET == 'era5':
                era5= xr.open_dataset(input_path, engine='zarr')
                minicube= era5.sel(time=slice(pd.Timestamp('2022-11-01'), None)) #Choose small time subset
                array_id= 'ERA5_0'
                print(f'Size: {era5.nbytes / 2**30:.3f}Gb')

            else:
                raise RuntimeError(f'Unknown {DATASET=}')

            #Save cube id
            if array_id not in overall_results.keys(): overall_results[array_id]= {}

            #Run tests
            for test, codec_param in codec_params.items():
                if test not in overall_results[array_id].keys(): overall_results[array_id][test]= {}
                for crf, param in zip(crfs, codec_param, strict=False):
                    if crf not in overall_results[array_id][test].keys(): overall_results[array_id][test][crf]= {}
                    for bits in n_bits:
                        #Update pbar
                        pbar.set_description(f'{test=} | {array_id=} | {crf=} | {bits=}')

                        #Skip bits that are not possible for a given codec
                        if 'libx265' in test or 'vp9' in test:
                            if bits not in [8,10,12]: continue
                            # PCA tests only run at 12-bit x265
                            if '(PCA' in test and bits != 12: continue
                            #if 'vp9' in test and bits == 8: continue #this combination hangs ffmpeg for some reason
                        elif 'vvenc' in test or 'uavs3e' in test:
                            if bits not in [8,10]: continue
                        elif 'vtm' in test:
                            if bits not in [8,10,12,16]: continue
                        elif 'hevc_nvenc' in test:
                            if bits not in [16]: continue
                        elif 'JP2OpenJPEG' in test:
                            if bits not in [8,16]: continue
                        elif 'ffv1' in test:
                            if bits not in [8]: continue
                        else:
                            raise AssertionError(f'Unknown test / codec: {test}')

                        #Run with compute_stats
                        conversion_rules= get_conversion_rules(DATASET, test, param, bits, bands=bands)
                        # print(conversion_rules)
                        try:
                            results= xarray2video( minicube, array_id, conversion_rules, 
                                                   include_data_in_stats=PLOT_SAMPLES,
                                                   output_path=run_testing_dir, compute_stats=True,
                                                   loglevel='verbose' if DEBUG else 'quiet',
                                                   verbose=False, save_dataset=False,
                                                   metrics_value_range=metrics_value_range,
                                                   )
                            overall_results[array_id][test][crf][bits]= results
                            
                            #Plot some original / compressed samples
                            if PLOT_SAMPLES:
                                for name in results.keys():
                                    orig, comp= results[name]['original'], results[name]['compressed']
                                    save_name_img= f'{save_name_path.name}_{test}_{bits}_{array_id.replace("/","")}_{name}_{crf}'
                                    factor= {
                                         'deepextremes':{'rgb':5., 'ir3':1.8, 'masks':1}.get(name, 1.), 
                                         'dynamicearthnet':1.1/2000., 
                                         'custom':0.8/2000., 'era5':2./orig.max()}[DATASET]
                                    
                                    timesteps= {'deepextremes':[100,120,140,155], 'dynamicearthnet':[1,60,120,280,310], 
                                                'custom':[1,3,5,7,9], 'era5':[1,20,40,60,80]}[DATASET]
                                    orig_clip= (orig*factor*255).clip(0, 255).astype(np.uint8)
                                    comp_clip= (comp*factor*255).clip(0, 255).astype(np.uint8)
                                    for t in timesteps:
                                        if crf == crfs[0]:
                                            Image.fromarray(orig_clip[t][..., :3], mode='RGB').save(
                                                f'results/samples/{save_name_img}_{t}_orig.png',
                                                compress_level=1,
                                            )
                                        Image.fromarray(comp_clip[t][..., :3], mode='RGB').save(
                                            f'results/samples/{save_name_img}_{t}_comp.png',
                                            compress_level=1,
                                        )

                                # The sample PNGs are already written; drop the large arrays
                                # so plot-sample runs do not exhaust memory when temp results are pickled.
                                for name, metadata in results.items():
                                    if isinstance(metadata, dict):
                                        metadata.pop('original', None)
                                        metadata.pop('compressed', None)

                            #Delete the output folder
                            del_path= results[next(iter(conversion_rules.keys()))]['path'][0].parent
                            shutil.rmtree(del_path)

                        except Exception as e:
                            print(f'{test=} | {array_id=} | {crf=} | {bits=}: Exception: {e}')
                            if DEBUG: raise e
                            #overall_results[array_id][test][crf][bits]= None

            save_checkpoint_results(overall_results, temp_results_path, save_name, metrics_keep)

    #______SAVE DATA_____    
            
    if not USE_SAVED or CONTINUE_FROM_TEMP:
        if CONTINUE_FROM_TEMP:
            conversion_rules= get_conversion_rules(DATASET, 'default', None, None, bands=bands)
            temp_load_path = temp_results_path if temp_results_path.exists() else Path('results_temp.pkl')
            overall_results= pickle.load(open(temp_load_path, 'rb'))
            
        results_df = results_dict_to_dataframe(overall_results, metrics_keep)
        results_df.to_pickle(f'{save_name}.pkl')
    

    #______PLOT______
    PLOT= not SKIP_PLOT
    
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Specify the path to the ttf file directly
    font_path = './results/cmunrm.ttf' 
    fm.fontManager.addfont(font_path)  # Explicitly add the font to font manager
    custom_font = fm.FontProperties(fname=font_path)

    # Set as default font - use the exact font name
    plt.rcParams['font.family'] = 'CMU Serif'  # The actual font family name
    # No need to set font.sans-serif since CMU Serif is a serif font

    #Colors?
    # cmap= plt.get_cmap('gist_rainbow')
    # colors= [cmap(value) for value in np.linspace(0, 1, 15)] #0->1, 15 values
    cmap= plt.get_cmap('tab10')
    colors= [cmap(value) for value in [*np.linspace(0, 1, 8)]*2] #0->1, 12 values

    #Load results?
    if USE_SAVED: 
        # Load the data
        result_frames = [pd.read_pickle(f'{load_name}.pkl')]
        loaded_test_names = set(result_frames[0].index.get_level_values('test_name').unique())
        
        # Check if the secondary dataframe exists
        file_name_16 = f'{load_name[:-1]}8.pkl'
        import os
        if os.path.exists(file_name_16):
            # Load and concatenate the second dataframe if it exists
            print('Using:', file_name_16)
            secondary_df = pd.read_pickle(file_name_16)
            secondary_df = secondary_df[
                ~secondary_df.index.get_level_values('test_name').isin(loaded_test_names)
            ]
            loaded_test_names.update(secondary_df.index.get_level_values('test_name').unique())
            result_frames.append(secondary_df)

        if args.extra_saved_results:
            for extra_name in [item.strip() for item in args.extra_saved_results.split(',') if item.strip()]:
                extra_path = Path(extra_name)
                if extra_path.suffix != '.pkl':
                    extra_path = Path(f'{extra_name}.pkl')
                if not extra_path.is_absolute():
                    extra_path = Path(extra_path)
                if not extra_path.exists():
                    candidate = Path('results') / extra_path.name
                    if candidate.exists():
                        extra_path = candidate
                if not extra_path.exists():
                    raise FileNotFoundError(f'Could not find extra saved results file: {extra_name}')
                print('Using extra saved results:', extra_path)
                extra_df = pd.read_pickle(extra_path)
                extra_df = extra_df[
                    ~extra_df.index.get_level_values('test_name').isin(loaded_test_names)
                ]
                loaded_test_names.update(extra_df.index.get_level_values('test_name').unique())
                result_frames.append(extra_df)

        results_df = pd.concat(result_frames, ignore_index=False)
        results_df = results_df[~results_df.index.duplicated(keep='last')]
            
        conversion_rules= get_conversion_rules(DATASET, 'default', None, None, bands=bands)

    # x_label= 'Compression percentage' #One of {'Compression percentage', 'Compression factor', 'bpppb'}
    x_label= 'bpppb'
    available_tests = set(results_df.index.get_level_values('test_name').unique())
    tests_plot_candidates = ['libx265', 'vvenc', 'uavs3e']
    if PLOT_PCA:
        tests_plot_candidates += [
            'libx265 (PCA)',
            'libx265 (PCA - 3 bands)',
            'libx265 (PCA - 9 bands)',
            'libx265 (PCA - 12 bands)',
            'libx265 (PCA - all bands)',
        ]
    tests_plot_candidates += ['vp9', 'JP2OpenJPEG']
    tests_plot= [test_name for test_name in tests_plot_candidates if test_name in available_tests]
    plot_name_suffix = '_with_pca' if PLOT_PCA else ''
    metrics_plot= {'psnr':'PSNR (dB)', 
                   #'mse':'MSE', #We disable MSE because it is very difficult to plot
                   'ssim':'SSIM', 'exp_sa':'SA (radians)', 
                   # 'time': 'Compression time (s)', #Disable this for lack of space in the paper
                   # 'd_time': 'Decompression time (s)'  #Disable this for lack of space in the paper
                  }
    metrics_plot= {m:label for m, label in metrics_plot.items() if m in metrics_keep}

    if PLOT:
        for i, (metric, y_label) in enumerate(metrics_plot.items()):
            f, ax= plt.subplots(1, 1, figsize=(3,2.5))
            ax.set_ylabel(y_label)

            if y_label == 'MSE':
                ax.set_yscale('log')
                # ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
                # ax.yaxis.set_minor_formatter(mtick.ScalarFormatter())
                # ax.tick_params(axis='y', which='minor', labelsize=6) 
                ax.set_ylim(1e-7, 5e-3)  # Set y limits for MSE to avoid extreme values
            elif y_label == 'SSIM':
                ax.set_ylim(0.98, 1.0)  # Adjust SSIM y-limits for better separation
            elif y_label == 'SA (radians)':
                ax.set_ylim(0., 0.075)
            elif y_label == 'PSNR (dB)':
                ax.set_ylim((35, 90))  

            if x_label == 'Compression factor':
                #If using a factor for compression
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
                ax.xaxis.set_minor_formatter(mtick.ScalarFormatter())
                ax.tick_params(axis='x', which='minor', labelsize=6) 
            elif x_label == 'Compression percentage':
                #If using percentage for compression
                ax.xaxis.set_major_formatter(mtick.PercentFormatter())
                start, end = ax.get_xlim()
            elif x_label == 'bpppb':
                ax.set_xscale('log')
                ax.set_xlim(xmin=0.01, xmax=10.) 
                ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
                # ax.xaxis.set_minor_formatter(mtick.FormatStrFormatter('%.2f'))
                # ax.tick_params(axis='x', which='minor', labelsize=6) 
            else:
                raise AssertionError(f'Unknown {x_label=}')

            for video in conversion_rules.keys():
                # if 'masks' in video: continue
                for test in tests_plot:
                    try:
                        test_video_data = results_df.xs(test, level='test_name').xs(video, level='video_name')
                        available_bits = sorted(test_video_data.index.get_level_values('bits').unique(), reverse=True)
                    except Exception as e:
                        print(f'Error processing {test=}, {video=}, {metric=}: {e}')
                        if 'index' in str(e): 
                            if DEBUG: 
                                breakpoint()
                        continue

                    for bits in available_bits:
                        try:
                            data= test_video_data.xs(bits, level='bits')
                            metric_values= data.xs(metric, level='metric').values.flatten()
                            if x_label == 'Compression factor':
                                compression_values= 1/data.xs('compression', level='metric').values.flatten()
                            elif x_label == 'Compression percentage':
                                compression_values= data.xs('compression', level='metric').values.flatten()
                            elif x_label == 'bpppb':
                                compression_values= data.xs('bpppb', level='metric').values.flatten()
                            else:
                                raise AssertionError(f'Unknown {x_label=}')
                            marker = get_video_marker(video)
                            color = adjust_color_for_video(get_codec_color(test), video)
                            scatter_kws = {'s': 4, 'alpha': 0.25, 'marker': marker}
                            line_kws = {
                                'linestyle': BIT_LINESTYLES[bits],
                                'label': f'{video} {bits}bits {test}',
                                'linewidth': 1.6,
                            }
                            compression_values_plot= compression_values * (100 if x_label == 'Compression percentage' else 1)

                            #Ensure compression_values is sorted
                            sorted_indices= np.argsort(compression_values)[-len(metric_values):]
                            compression_values_sorted= compression_values_plot[sorted_indices]
                            metric_values_sorted= metric_values[sorted_indices]

                            #Lowess smoothing
                            metric_values_smoothed= sm.nonparametric.lowess(metric_values_sorted, compression_values_sorted, 
                                                                            frac=0.6, return_sorted=False)

                            #Scatter plot and lineplot in separate calls
                            ax.scatter(compression_values_sorted, metric_values_sorted, color=color, **scatter_kws)
                            ax.plot(compression_values_sorted, metric_values_smoothed, color=color, **line_kws)

                            # if y_label == 'PSNR (dB)':
                            #     ax.set_ylim((metric_values_smoothed[0], 100))
                        except Exception as e:
                            print(f'Error processing {test=}, {video=}, {metric=}, {bits=}: {e}')
                            if 'index' in str(e): 
                                if DEBUG: 
                                    breakpoint()

            # if metric not in ['time']: 
            #     ax.axhline(metric_values[0], color='darkred', linestyle='--', label='uint8 discretization limit')

            ax.set_xlabel(x_label)
            plt.savefig(f'{save_name}{plot_name_suffix}_{x_label.lower().replace(" ","_")}_{metric}.pdf', dpi=200, bbox_inches='tight')

            if i == 0:
                # Extract the legend
                ax.legend()
                legend = ax.get_legend()
                fig2 = plt.figure()
                capitalize_first = lambda s: s[0].upper() + s[1:].replace('_', ' ') if s else s
                handles = legend.legendHandles
                labels = [capitalize_first(t.get_text()) for t in legend.texts]
                labels = [l.replace('Relative humidity', 'Rel. humidity') for l in labels]
                reordered_handles, reordered_labels, ncols = format_legend_items(handles, labels, DATASET, PLOT_PCA)
                fig2.legend(
                    reordered_handles, 
                    reordered_labels, 
                    ncols=ncols,
                    frameon=False,
                    borderaxespad=0,  # Remove padding between legend and axes
                    handletextpad=0.5,  # Reduce space between handle and text
                )
                plt.savefig(f'{save_name}{plot_name_suffix}_{x_label.lower().replace(" ","_")}_legend.pdf', dpi=200, bbox_inches='tight', 
                            pad_inches=0  # Remove padding around the figure
                           )

    #_____________CREATE LATEX TABLE_____________
                
    if not SKIP_LATEX:
        pd.set_option('display.max_rows', None)
        grouped= results_df.groupby(level=[0,1,2,3,5]).mean() #Group over cube dimension (pos 4)
    #grouped= results_df.xs('12_10.38_50.15', level='video_name')
    
    def create_latex_table(df):
        # Define metric mapping and order
        metric_mapping = {
            'bpppb': 'bpppb',
            'ssim': 'SSIM',
            'psnr': 'PSNR',
            'exp_sa': 'SA',
            'mse': 'MSE',
            'time': '$t_{c}$',
            'd_time': '$t_{d}$'
        }
        metrics_order = ['bpppb', 'ssim', 'mse', 'psnr', 'exp_sa', 'time', 'd_time']
        ignore_metrics = ['ssim', 'exp_sa', 'mse']
        ignore_videos = [] #['masks']
        crf_order = ['Best', 'Very high', 'High', 'Medium', 'Low', 'Very low']
        REVERSE_CRFS = False

        # Get filtered metrics and video names
        metrics = [m for m in metrics_order if m in df.index.get_level_values('metric').unique()]
        metrics = [m for m in metrics if m not in ignore_metrics]
        video_names = sorted(df.index.get_level_values('video_name').unique())
        video_names = [v for v in video_names if v not in ignore_videos]

        latex_str = """\\begin{table*}[t]
    \\centering
    \\caption{Quality comparison of different compression methods}
    \\label{tab:quality_comparison}
    \\begin{tabular}{llr"""

        # Add column format for each video/metric combination (all right-aligned)
        latex_str += "r" * (len(video_names) * len(metrics))
        latex_str += "}\n\\toprule\n"

        # Add video names spanning multiple columns
        latex_str += "& & "
        for video in video_names:
            latex_str += f"\\multicolumn{{{len(metrics)}}}{{c}}{{{video.replace('_', ' ')}}} "
            if video != video_names[-1]:
                latex_str += "& "
        latex_str += "\\\\\n"

        # Add metric names
        latex_str += "Test & Bits & Quality"
        for video in video_names:
            for metric in metrics:
                latex_str += f" & {metric_mapping[metric]}"
        latex_str += "\\\\\n\\midrule\n"

        # Process data
        current_test = None

        # Get unique combinations and sort them
        test_names = sorted(df.index.get_level_values('test_name').unique())

        for test_name in test_names:
            bits_values = sorted(df.xs(test_name, level='test_name').index.get_level_values('bits').unique())

            for bits in bits_values:
                # Get CRF values for this test_name and bits
                df_subset = df.xs((test_name, bits), level=('test_name', 'bits'))
                crf_values = sorted(df_subset.index.get_level_values('crf').unique(),
                                  key=lambda x: crf_order.index(x) if x in crf_order else len(crf_order))

                # Add midrule if changing test_name
                if current_test is not None and current_test != test_name:
                    latex_str += "\\midrule\n"

                for i, crf in enumerate(crf_values):
                    if i == 0:
                        latex_str += f"\\multirow{{{len(crf_values)}}}{{*}}{{{test_name.replace('_', ' ')}}} & "
                        latex_str += f"\\multirow{{{len(crf_values)}}}{{*}}{{{bits}}} & {crf}"
                    else:
                        latex_str += f"& & {crf}"

                    # Add values for each video and metric
                    for video in video_names:
                        for metric in metrics:
                            try:
                                value = df.loc[(metric, video, test_name, bits, 
                                                crf if not REVERSE_CRFS else crf_values[-1-i]), #TODO: Fix bug!
                                               'value']
                                if metric == 'psnr' and value == 100.:
                                    formatted_value= "$\\infty$"
                                else:
                                    formatted_value = f"{value:.3f}" if pd.notnull(value) else ""
                            except KeyError:
                                formatted_value = ""
                            latex_str += f" & {formatted_value}"
                    latex_str += " \\\\\n"

                current_test = test_name

        # Close LaTeX table
        latex_str += """\\bottomrule
    \\end{tabular}
    \\end{table*}"""
    

        return latex_str

    # Print and save LaTeX version
    if not SKIP_LATEX:
        latex_output = create_latex_table(grouped)
        latex_path = f'{save_name}_quality_comparison.tex'
        Path(latex_path).write_text(latex_output + '\n', encoding='utf-8')
        print(latex_output)
        print(f'Saved LaTeX table to {latex_path}')