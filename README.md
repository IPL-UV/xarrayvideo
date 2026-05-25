# xarrayvideo

Save multichannel data from xarray datasets as videos to save up massive amounts of space (e.g. 20-50x compression) with minimal quality loss.

The library revolves around two functions: `xarray2video` encodes selected variables into videos, and `video2xarray` rebuilds the dataset from those videos. The project supports standard ffmpeg codecs such as `libx265`, `vp9`, and `ffv1`, GDAL-backed image codecs such as `JP2OpenJPEG`, and now also direct wrappers for external codecs such as `vvenc` and `uavs3e`.

## Features

- Encode multiband data into groups of three channels per video.
- Mix lossy and lossless outputs in the same dataset export.
- Use ffmpeg-backed video codecs or GDAL-backed image codecs.
- Run optional PCA/KLT over the channel dimension before encoding.
- Benchmark codecs with the same evaluation pipeline used in the paper.

## Paper

If you find this library useful, please consider citing the accompanying paper:

```
Pellicer-Valero, O. J., Aybar, C., & Camps-Valls, G. (2025). Video compression for spatiotemporal Earth system data. arXiv. https://doi.org/10.48550/arXiv.2506.19656
```

## Installation

Base install:

```bash
git clone https://github.com/OscarPellicer/xarrayvideo.git
cd xarrayvideo
pip install -e .[all]
```

If you prefer to install dependencies manually:

```bash
pip install xarray numpy scikit-image scikit-learn pyyaml zarr netcdf4 ffmpeg-python gcsfs pillow tqdm seaborn h5netcdf tacoreader pytortilla tacotoolbox

# Optional helpers
pip install ipython opencv-python
pip install git+https://github.com/OscarPellicer/txyvis.git
pip install torchmetrics

pip install -e . --no-deps
```

GDAL is optional, but required for GDAL-backed image codecs such as `JP2OpenJPEG`.

Linux and macOS:

```bash
pip install gdal
```

Windows:

```bash
mamba install -c conda-forge gdal
# or
conda install -c conda-forge gdal
```

## External codecs: vvenc, uavs3e, and VTM

The main library works with plain ffmpeg alone. You only need the external toolchain if you want to benchmark H.266/VVC or AVS3.

Build the external codecs with the helper script:

```bash
bash scripts/install_vvc_avs3_codecs.sh
```

By default this builds `vvenc`, `vvdec`, `uavs3e`, and `uavs3d`. To restrict the build, set `CODECS_TO_BUILD`:

```bash
CODECS_TO_BUILD=vvenc,uavs3e bash scripts/install_vvc_avs3_codecs.sh
CODECS_TO_BUILD=vtm bash scripts/install_vvc_avs3_codecs.sh
```

Then expose the resulting binaries to the current shell:

```bash
source scripts/activate_codec_tools.sh
```

If your builds live outside `$HOME/codec_toolchains`, pass the root explicitly:

```bash
source scripts/activate_codec_tools.sh /path/to/codec_toolchains
```

Notes:

- `vvenc` and `uavs3e` are the practical external codecs integrated into the benchmark path.
- `VTM` remains optional and significantly slower; it is mainly kept as a reference path.
- The benchmark scripts read `XV_CODEC_THREADS` or `SLURM_CPUS_PER_TASK` to control external codec threading.

## Examples

Open the example notebooks in JupyterLab or VS Code:

- `example_dynamicearthnet.ipynb`
- `example_deepextremecubes.ipynb`
- `example_simples2.ipynb`
- `example_era5.ipynb`

## Basic usage

Example with a DeepExtremeCubes sample:

```python
import xarray as xr
import numpy as np
from xarrayvideo import xarray2video, video2xarray, plot_image

array_id = '-111.49_38.60'
input_path = '../mc_-111.49_38.60_1.2.2_20230702_0.zarr'
output_path = './out'

minicube = xr.open_dataset(input_path, engine='zarr')
minicube['SCL'] = minicube['SCL'].astype(np.uint8)
minicube['cloudmask_en'] = minicube['cloudmask_en'].astype(np.uint8)

lossless_params = {'c:v': 'ffv1'}
lossy_params = {
    'c:v': 'libx265',
    'preset': 'medium',
    'crf': 51,
    'x265-params': 'qpmin=0:qpmax=0.01',
    'tune': 'psnr',
}
conversion_rules = {
    'rgb': (('B04', 'B03', 'B02'), ('time', 'y', 'x'), 0, lossy_params, 12),
    'ir3': (('B8A', 'B06', 'B05'), ('time', 'y', 'x'), 0, lossy_params, 12),
    'masks': (('SCL', 'cloudmask_en', 'invalid'), ('time', 'y', 'x'), 0, lossless_params, 8),
}

# Compress, with compute_stats it takes a bit longer, but shows compression info
arr_dict = xarray2video(
    minicube,
    array_id,
    conversion_rules,
    output_path=output_path,
    compute_stats=True,
    loglevel='verbose',
    save_dataset=True,
)

minicube_new = video2xarray(output_path, array_id)

plot_image(minicube, ['B04', 'B03', 'B02'], save_name='./out/RGB_original.jpg')
plot_image(minicube_new, ['B04', 'B03', 'B02'], save_name='./out/RGB_compressed.jpg')
```

## Testing and benchmarks

There is no dedicated pytest suite in this repository at the moment. The canonical regression checks are the benchmark and validation scripts in `scripts/`.

Main benchmark driver:

```bash
python scripts/run_tests.py --dataset deepextremes --rules_name gapfill3
python scripts/run_tests.py --dataset dynamicearthnet --rules_name 4channels2
python scripts/run_tests.py --dataset custom --rules_name pca
python scripts/run_tests.py --dataset era5 --rules_name all
```

Useful options for fast validation:

```bash
python scripts/run_tests.py \
  --dataset deepextremes \
  --rules_name smoke-vvc-avs3-vtm-fast-qplow \
  --codec_names vvenc,uavs3e,vtm \
  --sample_limit 1 \
  --quality_limit 1 \
  --skip_plot \
  --skip_latex \
  --debug
```

Sample image generation:

```bash
python scripts/run_tests.py --dataset deepextremes --rules_name img --id 10.38_50.15 --plot_samples
python scripts/run_tests.py --dataset dynamicearthnet --rules_name img --id 8077_5007 --plot_samples
python scripts/run_tests.py --dataset custom --rules_name img --id cubo1 --plot_samples
python scripts/run_tests.py --dataset era5 --rules_name img --plot_samples
```

Cluster launchers wrapping `scripts/run_tests.py`:

- `run_cpu_xv_codec_smoke.sh`: minimal external-codec smoke test.
- `run_cpu_xv_codec_calibration.sh`: calibration run for the new codec ladders.
- `run_cpu_xv_custom_codecs.sh`: custom-dataset codec benchmark.
- `run_cpu_xv_deepextremes_codecs.sh`: DeepExtremeCubes codec benchmark.
- `run_cpu_xv_dynamicearthnet_codecs.sh`: DynamicEarthNet codec benchmark.
- `run_cpu_xv_era5_codecs.sh`: ERA5 codec benchmark.

Outputs:

- Final result pickles, plots, and tables are written under `results/`.
- Temporary encoded cubes are written under `testing/`.
- Most benchmark logs are written at the repo root by the shell or SLURM launchers.

Additional regression-oriented scripts:

```bash
python scripts/run_xarrayvideo_single_cube.py --help
python scripts/synthetic_missing_tests.py
python scripts/reproduce_repetition_comprehensive.py --help
```

## TerraCodec comparison workflow

The repository now includes dedicated scripts used for the direct TerraCodec comparisons reported during review.

- `scripts/run_terracodec_tests.py`: benchmark one cube and emit the same MultiIndex pickle layout used by `scripts/run_tests.py`.
- `scripts/run_terracodec_suite.py`: launch the repo's benchmark subset repeatedly across multiple cubes.
- `scripts/check_terracodec_scaling.py`: confirm reflectance scaling before running TerraCodec on public Sentinel-2 cubes.
- `run_gpu_terracodec_node10.slurm` and `run_gpu_terracodec_suite_node10.slurm`: GPU launchers used for those comparisons.
- `scripts/run_xarrayvideo_single_cube.py`: classical codec baseline on the same cube slices used for TerraCodec.

These scripts are intentionally separate from the default ffmpeg-centric workflow because TerraCodec has different environment, hardware, and model assumptions.

## TACO and Tortilla integration

This repo also contains packaging helpers for `.tortilla` and `.taco` datasets.

- `pytortilla` is used to wrap single samples.
- `tacotoolbox` is used to assemble collections of samples.

Relevant scripts:

- `scripts/process_deepextremes.py`: convert DeepExtremeCubes into xarrayvideo/TACO-friendly form.
- `scripts/process_dynamicearthet.py`: convert DynamicEarthNet into xarray first, then into xarrayvideo/TACO form. The filename has a historical typo but is the current tracked script.
- `scripts/download_from_hf.py`: download prepared artifacts from Hugging Face.
- `scripts/upload_taco.py`: upload packaged datasets.
- `scripts/legacy/deepextremecubes_to_tacov2.py` and `scripts/legacy/dynamicearthnet_to_tacov2.py`: migration helpers for older TACO layouts.

## Scripts overview

Benchmarking and validation:

- `scripts/run_tests.py`: main paper benchmark runner.
- `scripts/run_xarrayvideo_single_cube.py`: single-cube rate-distortion check.
- `scripts/reproduce_repetition_comprehensive.py`: repetition-versus-padding compression study.
- `scripts/synthetic_missing_tests.py`: missing-data handling tests.
- `scripts/era5_diagnostics.py`: post-process ERA5 benchmark outputs into diagnostics.

Codec setup and probing:

- `scripts/install_vvc_avs3_codecs.sh`: build external codec toolchains.
- `scripts/activate_codec_tools.sh`: add codec binaries to `PATH`.
- `scripts/find_encoders.sh`: inspect ffmpeg encoder and pixel-format support.

TerraCodec utilities:

- `scripts/run_terracodec_tests.py`: per-cube TerraCodec benchmark.
- `scripts/run_terracodec_suite.py`: multi-cube TerraCodec benchmark orchestration.
- `scripts/check_terracodec_scaling.py`: verify TerraCodec input scaling.

Dataset preparation and packaging:

- `scripts/process_deepextremes.py`: DeepExtremeCubes processing.
- `scripts/process_dynamicearthet.py`: DynamicEarthNet processing.
- `scripts/gen_patches.py`: patch extraction helper.
- `scripts/fix_metadata.py`: metadata repair utility.
- `scripts/upload_taco.py`: Hugging Face upload helper.
- `scripts/download_from_hf.py`: Hugging Face download helper.

Legacy and one-off helpers kept for reproducibility:

- `scripts/legacy/README.md`: quick index for archived helpers.
- `scripts/legacy/check_max_vals.py`: inspect dataset value ranges.
- `scripts/legacy/find_processing_gap.py`: locate processing gaps in prepared datasets.
- `scripts/legacy/measure_ram.py`: memory measurement helper.
- `scripts/legacy/plot_synthetic.py`: plotting helper for synthetic missing-data experiments.
- `scripts/legacy/deepextremecubes_to_tacov2.py`: DeepExtremeCubes TACO migration.
- `scripts/legacy/dynamicearthnet_to_tacov2.py`: DynamicEarthNet TACO migration.
- `scripts/uavs3e_ra.cfg`: AVS3 encoder configuration used by the benchmarks.

The main `scripts/` folder now contains the active workflows. Older migration and ad hoc analysis utilities live under `scripts/legacy/` so the top-level script surface stays focused.

## Contact

Contact: `oscar.pellicer [at] uv.es` or open an Issue.
