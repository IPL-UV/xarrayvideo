"""Benchmark xarrayvideo codecs on a single local cube.

This mirrors the loading and cropping path used by run_terracodec_tests.py so we
can compare a classical xarrayvideo codec against TerraCodec on the exact same
input slice.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xarrayvideo import gap_fill, xarray2video


SIMPLES2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
CRF_LABELS = ["Best", "Very high", "High", "Medium", "Low", "Very low"]

X265_CRFS = [51, 51, 1, 7, 16, 27]
X265_PARAMS = ["lossless=1", "qpmin=0:qpmax=0.01", "", "", "", ""]
VP9_PARAMS = [
    {"c:v": "vp9", "crf": 0, "lossless": 1},
    {"c:v": "vp9", "crf": 0, "arnr-strength": 2, "qmin": 0, "qmax": 0.01, "lag-in-frames": 25, "arnr-maxframes": 7},
    {"c:v": "vp9", "crf": 5, "arnr-strength": 2, "lag-in-frames": 25, "arnr-maxframes": 7},
    {"c:v": "vp9", "crf": 12, "arnr-strength": 2, "lag-in-frames": 25, "arnr-maxframes": 7},
    {"c:v": "vp9", "crf": 20, "arnr-strength": 2, "lag-in-frames": 25, "arnr-maxframes": 7},
    {"c:v": "vp9", "crf": 30, "arnr-strength": 2, "lag-in-frames": 25, "arnr-maxframes": 7},
]
JP2_QUALITIES = [100, 80, 35, 15, 5, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a single xarrayvideo codec on a local cube")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--input-format", choices=["auto", "netcdf", "zarr", "pickle"], default="auto")
    parser.add_argument("--dataset", choices=["generic", "deepextremes"], default="generic")
    parser.add_argument("--bands", type=str, required=True)
    parser.add_argument("--time-dim", type=str, default="time")
    parser.add_argument("--y-dim", type=str, default="y")
    parser.add_argument("--x-dim", type=str, default="x")
    parser.add_argument("--time-range", type=str, default=None)
    parser.add_argument("--spatial-crop", type=str, default=None)
    parser.add_argument("--codec", choices=["libx265", "vp9", "JP2OpenJPEG", "ffv1"], default="libx265")
    parser.add_argument("--quality-index", type=int, default=3, help="0..5 following run_tests.py presets")
    parser.add_argument("--bits", type=int, default=12)
    parser.add_argument("--metrics-value-range", type=str, default="0,65535")
    parser.add_argument("--output-stem", type=Path, default=None)
    parser.add_argument("--cube-name", type=str, default=None)
    parser.add_argument("--keep-artifacts", action="store_true", default=False)
    return parser.parse_args()


def parse_range(spec: str | None) -> slice:
    if spec is None:
        return slice(None)
    start_str, stop_str = spec.split(":", maxsplit=1)
    start = int(start_str) if start_str else None
    stop = int(stop_str) if stop_str else None
    return slice(start, stop)


def parse_spatial_crop(spec: str | None) -> tuple[slice, slice]:
    if spec is None:
        return slice(None), slice(None)
    y_spec, x_spec = spec.split(",", maxsplit=1)
    return parse_range(y_spec), parse_range(x_spec)


def parse_metrics_value_range(spec: str, array: np.ndarray) -> tuple[float, float]:
    if spec == "auto":
        return float(np.nanmin(array)), float(np.nanmax(array))
    min_str, max_str = spec.split(",", maxsplit=1)
    return float(min_str), float(max_str)


def open_dataset(input_path: Path, input_format: str) -> xr.Dataset:
    if input_format == "auto":
        if input_path.suffix == ".nc":
            input_format = "netcdf"
        elif input_path.suffix == ".zarr" or input_path.is_dir():
            input_format = "zarr"
        elif input_path.is_file():
            input_format = "pickle"
        else:
            raise RuntimeError(f"Could not infer input format for {input_path}")

    if input_format == "netcdf":
        return xr.open_dataset(input_path)
    if input_format == "zarr":
        return xr.open_dataset(input_path, engine="zarr")
    if input_format == "pickle":
        with open(input_path, "rb") as file_obj:
            payload = pickle.load(file_obj)
        if isinstance(payload, xr.Dataset):
            return payload
        if isinstance(payload, xr.DataArray):
            return payload.to_dataset(dim="band")
        if hasattr(payload, "to_dataset"):
            return payload.to_dataset(dim="band")
        raise RuntimeError(f"Unsupported pickle payload type: {type(payload)!r}")
    raise RuntimeError(f"Unknown input format {input_format}")


def maybe_alias_simples2_bands(dataset: xr.Dataset) -> xr.Dataset:
    data_vars = list(dataset.data_vars)
    if len(data_vars) != len(SIMPLES2_BANDS):
        return dataset
    expected = [str(index) for index in range(1, len(SIMPLES2_BANDS) + 1)]
    if [str(name) for name in data_vars] != expected:
        return dataset
    return dataset.rename({var_name: band_name for var_name, band_name in zip(data_vars, SIMPLES2_BANDS)})


def build_codec_params(codec: str, quality_index: int) -> tuple[dict, str]:
    if not 0 <= quality_index < len(CRF_LABELS):
        raise RuntimeError(f"quality_index must be in [0, {len(CRF_LABELS) - 1}]")

    if codec == "libx265":
        params = {
            "c:v": "libx265",
            "preset": "medium",
            "tune": "psnr",
            "crf": X265_CRFS[quality_index],
            "x265-params": X265_PARAMS[quality_index],
        }
    elif codec == "vp9":
        params = dict(VP9_PARAMS[quality_index])
    elif codec == "JP2OpenJPEG":
        params = {
            "codec": "JP2OpenJPEG",
            "QUALITY": str(JP2_QUALITIES[quality_index]),
            "REVERSIBLE": "YES" if quality_index == 0 else "NO",
            "YCBCR420": "NO",
        }
    elif codec == "ffv1":
        params = {"c:v": "ffv1"}
    else:
        raise RuntimeError(f"Unsupported codec {codec}")

    return params, CRF_LABELS[quality_index]


def main() -> None:
    args = parse_args()
    dataset = maybe_alias_simples2_bands(open_dataset(args.input_path, args.input_format))
    cube_name = args.cube_name or args.input_path.stem.replace(".zarr", "")
    bands = [band.strip() for band in args.bands.split(",") if band.strip()]

    if args.dataset == "deepextremes":
        missing_required = [band for band in ["SCL", "cloudmask_en"] if band not in dataset]
        if missing_required:
            raise RuntimeError(f"DeepExtremeCubes preset requires {missing_required}")
        dataset["SCL"] = dataset["SCL"].astype(np.uint8)
        dataset["cloudmask_en"] = dataset["cloudmask_en"].astype(np.uint8)
        dataset = gap_fill(
            dataset,
            fill_bands=bands,
            mask_band=None,
            fill_values=[1, 3, 4],
            method="last_value",
            new_mask="invalid",
            coord_names=(args.time_dim, "variable", args.y_dim, args.x_dim),
        )

    time_slice = parse_range(args.time_range)
    y_slice, x_slice = parse_spatial_crop(args.spatial_crop)
    dataset = dataset.isel({args.time_dim: time_slice, args.y_dim: y_slice, args.x_dim: x_slice})

    first_array = np.stack(
        [dataset[band].transpose(args.time_dim, args.y_dim, args.x_dim).values for band in bands],
        axis=-1,
    ).astype(np.float32)
    metrics_value_range = parse_metrics_value_range(args.metrics_value_range, first_array)
    codec_params, crf_label = build_codec_params(args.codec, args.quality_index)

    conversion_rules = {
        "all": (bands, (args.time_dim, args.y_dim, args.x_dim), False, codec_params, args.bits)
    }
    testing_dir = REPO_ROOT / "testing"
    output_stem = args.output_stem or Path("results") / f"results_xarrayvideo_{args.codec}_{cube_name}"
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    results = xarray2video(
        dataset,
        cube_name,
        conversion_rules,
        include_data_in_stats=False,
        output_path=testing_dir,
        compute_stats=True,
        loglevel="quiet",
        verbose=False,
        save_dataset=False,
        metrics_value_range=metrics_value_range,
    )
    row = results["all"]

    results_metrics = {}
    for metric_name in ["original_size", "compressed_size", "compression", "bpppb", "time", "d_time", "ssim", "psnr", "mse", "exp_sa", "acc"]:
        if metric_name in row:
            results_metrics[(metric_name, "all", args.codec, args.bits, cube_name, crf_label)] = row[metric_name]

    results_df = pd.DataFrame.from_dict(results_metrics, orient="index", columns=["value"])
    results_df.index = pd.MultiIndex.from_tuples(results_df.index)
    results_df.index.names = ["metric", "video_name", "test_name", "bits", "cube", "crf"]
    results_df.to_pickle(f"{output_stem}.pkl")

    summary = {
        "cube": cube_name,
        "input_path": str(args.input_path),
        "source_bands": bands,
        "shape": list(first_array.shape),
        "metrics_value_range": list(metrics_value_range),
        "codec": args.codec,
        "crf_label": crf_label,
        "codec_params": codec_params,
        "bits": args.bits,
    }
    with open(f"{output_stem}_summary.json", "w") as file_obj:
        json.dump(summary, file_obj, indent=2)

    if not args.keep_artifacts:
        artifact_dir = Path(row["path"][0]).parent
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)

    print(results_df)


if __name__ == "__main__":
    main()