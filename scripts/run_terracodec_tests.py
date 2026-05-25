"""Benchmark TerraCodec on local Earth observation cubes.

This script keeps TerraCodec isolated from the ffmpeg-centric benchmark path in
``scripts/run_tests.py`` while producing a pickle with the same MultiIndex shape
used by the existing plotting code:

    (metric, video_name, test_name, bits, cube, crf) -> value

The current implementation targets Sentinel-2 L2A style inputs and the
pretrained TerraCodec temporal models. It can run in a strict mode, where all
12 required L2A bands must be present, or in an approximate mode that fills
missing bands to let us probe 10-band cubes without claiming a like-for-like
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TERRACODEC_S2L2A_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]

TERRACODEC_S2L2A_MEAN = np.array(
    [
        793.243,
        924.863,
        1184.553,
        1340.936,
        1671.402,
        2240.082,
        2468.412,
        2563.243,
        2627.704,
        2711.071,
        2416.714,
        1849.625,
    ],
    dtype=np.float32,
)

TERRACODEC_S2L2A_STD = np.array(
    [
        1160.144,
        1201.092,
        1219.943,
        1397.225,
        1400.035,
        1373.136,
        1429.170,
        1485.025,
        1447.836,
        1652.703,
        1471.002,
        1365.307,
    ],
    dtype=np.float32,
)

DEFAULT_COMPRESSIONS = [0.4, 1, 5, 20, 100, 200, 700]
SIMPLES2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
DEFAULT_REFLECTANCE_SCALE = 10000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TerraCodec on an xarray cube")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to a NetCDF or Zarr dataset")
    parser.add_argument(
        "--input-format",
        choices=["auto", "netcdf", "zarr", "pickle"],
        default="auto",
        help="How to open --input-path",
    )
    parser.add_argument(
        "--dataset",
        choices=["generic", "deepextremes"],
        default="generic",
        help="Optional preset loader. 'deepextremes' applies the same gap filling used in run_tests.py.",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default=",".join(TERRACODEC_S2L2A_BANDS),
        help="Comma-separated source band variable names to read from the dataset",
    )
    parser.add_argument("--time-dim", type=str, default="time")
    parser.add_argument("--y-dim", type=str, default="y")
    parser.add_argument("--x-dim", type=str, default="x")
    parser.add_argument(
        "--time-range",
        type=str,
        default=None,
        help="Optional Python-slice-like range start:stop applied on the time dimension",
    )
    parser.add_argument(
        "--spatial-crop",
        type=str,
        default=None,
        help="Optional crop in the form y0:y1,x0:x1 before tiling",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Spatial tile size expected by TerraCodec temporal models",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=4,
        help="Process temporal chunks of this size. Use 0 to encode the full time axis at once.",
    )
    parser.add_argument(
        "--band-fill-mode",
        choices=["strict", "zeros", "mean"],
        default="strict",
        help="How to handle missing TerraCodec L2A bands",
    )
    parser.add_argument(
        "--compression-values",
        type=str,
        default=",".join(map(str, DEFAULT_COMPRESSIONS)),
        help="Comma-separated TerraCodec lambda values",
    )
    parser.add_argument(
        "--model",
        choices=["terracodec_v1_tt_s2l2a", "flextec_v1_s2l2a"],
        default="terracodec_v1_tt_s2l2a",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="FlexTEC quality level. Ignored for terracodec_v1_tt_s2l2a.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Select 'cuda' when available or force CPU/GPU explicitly",
    )
    parser.add_argument(
        "--metrics-value-range",
        type=str,
        default="0,65535",
        help="Value range for PSNR/SSIM as 'min,max' or 'auto'",
    )
    parser.add_argument(
        "--input-scale",
        choices=["auto", "native", "reflectance_0_1"],
        default="auto",
        help="Scale Sentinel-2 inputs before TerraCodec normalization. 'reflectance_0_1' multiplies by --reflectance-scale-factor.",
    )
    parser.add_argument(
        "--reflectance-scale-factor",
        type=float,
        default=DEFAULT_REFLECTANCE_SCALE,
        help="Factor used when --input-scale reflectance_0_1 is selected or auto-detected.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output path without extension. Defaults to results/results_terracodec_<cube_name>",
    )
    parser.add_argument(
        "--cube-name",
        type=str,
        default=None,
        help="Override the cube identifier stored in the output table",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate loading and preprocessing without running TerraCodec",
    )
    return parser.parse_args()


def parse_range(spec: str | None, size: int) -> slice:
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
    return parse_range(y_spec, -1), parse_range(x_spec, -1)


def parse_compressions(spec: str) -> list[float]:
    return [float(value.strip()) for value in spec.split(",") if value.strip()]


def parse_metrics_value_range(spec: str, array: np.ndarray) -> tuple[float, float]:
    if spec == "auto":
        return float(np.nanmin(array)), float(np.nanmax(array))
    min_str, max_str = spec.split(",", maxsplit=1)
    return float(min_str), float(max_str)


def resolve_input_scale(array: np.ndarray, mode: str, reflectance_scale_factor: float) -> tuple[float, str]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 1.0, "empty-input"

    min_value = float(finite.min())
    max_value = float(finite.max())

    if mode == "native":
        return 1.0, "native-requested"
    if mode == "reflectance_0_1":
        return reflectance_scale_factor, "reflectance-requested"

    # Sentinel-2 public releases often store reflectance in [0, 1] while the
    # pretrained TerraCodec checkpoints expect original L2A-like units.
    if min_value >= -0.1 and max_value <= 2.0:
        return reflectance_scale_factor, "reflectance-auto"
    return 1.0, "native-auto"


def apply_input_scale(array: np.ndarray, scale_factor: float) -> np.ndarray:
    if scale_factor == 1.0:
        return array
    return array * np.float32(scale_factor)


def infer_device(torch_module, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is not available")
    return requested


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

    normalized = [str(var_name) for var_name in data_vars]
    expected = [str(index) for index in range(1, len(SIMPLES2_BANDS) + 1)]
    if normalized != expected:
        return dataset

    rename_map = {var_name: band_name for var_name, band_name in zip(data_vars, SIMPLES2_BANDS)}
    return dataset.rename(rename_map)


def load_dataset(args: argparse.Namespace) -> tuple[str, xr.Dataset, list[str]]:
    dataset = maybe_alias_simples2_bands(open_dataset(args.input_path, args.input_format))
    cube_name = args.cube_name or args.input_path.stem.replace(".zarr", "")

    bands = [band.strip() for band in args.bands.split(",") if band.strip()]
    if args.dataset == "deepextremes":
        from xarrayvideo import gap_fill

        missing_required = [band for band in ["SCL", "cloudmask_en"] if band not in dataset]
        if missing_required:
            raise RuntimeError(f"DeepExtremeCubes preset requires {missing_required}")
        if "SCL" in dataset:
            dataset["SCL"] = dataset["SCL"].astype(np.uint8)
        if "cloudmask_en" in dataset:
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

    time_slice = parse_range(args.time_range, dataset.sizes[args.time_dim])
    y_slice, x_slice = parse_spatial_crop(args.spatial_crop)
    dataset = dataset.isel({args.time_dim: time_slice, args.y_dim: y_slice, args.x_dim: x_slice})
    return cube_name, dataset, bands


def stack_bands(dataset: xr.Dataset, bands: list[str], time_dim: str, y_dim: str, x_dim: str) -> np.ndarray:
    missing_bands = [band for band in bands if band not in dataset]
    if missing_bands:
        raise RuntimeError(f"Bands not found in dataset: {missing_bands}")
    array = np.stack(
        [dataset[band].transpose(time_dim, y_dim, x_dim).values for band in bands],
        axis=-1,
    )
    return array.astype(np.float32)


def align_to_terracodec_bands(
    array: np.ndarray,
    source_bands: list[str],
    fill_mode: str,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    band_to_index = {band: index for index, band in enumerate(source_bands)}
    missing = [band for band in TERRACODEC_S2L2A_BANDS if band not in band_to_index]
    if missing and fill_mode == "strict":
        raise RuntimeError(
            "Missing TerraCodec L2A bands "
            f"{missing}. Re-run with --band-fill-mode zeros|mean if you want an approximate probe."
        )

    aligned = []
    observed_mask = []
    for target_band, mean_value in zip(TERRACODEC_S2L2A_BANDS, TERRACODEC_S2L2A_MEAN):
        if target_band in band_to_index:
            aligned.append(array[..., band_to_index[target_band]])
            observed_mask.append(True)
        elif fill_mode == "zeros":
            aligned.append(np.zeros(array.shape[:-1], dtype=np.float32))
            observed_mask.append(False)
        elif fill_mode == "mean":
            aligned.append(np.full(array.shape[:-1], fill_value=mean_value, dtype=np.float32))
            observed_mask.append(False)
        else:
            raise AssertionError(f"Unexpected fill mode {fill_mode}")

    return np.stack(aligned, axis=-1), missing, np.array(observed_mask, dtype=bool)


def pad_to_multiple(array: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    pad_y = (tile_size - (array.shape[1] % tile_size)) % tile_size
    pad_x = (tile_size - (array.shape[2] % tile_size)) % tile_size
    if pad_y == 0 and pad_x == 0:
        return array, (0, 0)
    padded = np.pad(array, ((0, 0), (0, pad_y), (0, pad_x), (0, 0)), mode="edge")
    return padded, (pad_y, pad_x)


def normalize_for_terracodec(array: np.ndarray) -> np.ndarray:
    return (array - TERRACODEC_S2L2A_MEAN.reshape(1, 1, 1, -1)) / TERRACODEC_S2L2A_STD.reshape(1, 1, 1, -1)


def denormalize_from_terracodec(array: np.ndarray) -> np.ndarray:
    return array * TERRACODEC_S2L2A_STD.reshape(1, 1, 1, -1) + TERRACODEC_S2L2A_MEAN.reshape(1, 1, 1, -1)


def iter_time_windows(length: int, window: int) -> list[tuple[int, int]]:
    if window <= 0 or window >= length:
        return [(0, length)]
    return [(start, min(start + window, length)) for start in range(0, length, window)]


def compute_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metrics_value_range: tuple[float, float],
    observed_channel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if observed_channel_mask is not None:
        original = original[..., observed_channel_mask]
        reconstructed = reconstructed[..., observed_channel_mask]

    # TorchMetrics is faster on moderate arrays, but for large SimpleS2 cubes it
    # requires several full tensor copies and can trigger process kills during
    # postprocessing. Fall back to the lower-memory NumPy/xarrayvideo path there.
    estimated_tensor_bytes = original.size * np.dtype(np.float32).itemsize * 2

    try:
        if estimated_tensor_bytes > 1_000_000_000:
            raise RuntimeError("fallback-to-low-memory-metrics")

        import torch
        from torchmetrics.functional.image import peak_signal_noise_ratio as pt_psnr
        from torchmetrics.functional.image import spectral_angle_mapper as pt_sa
        from torchmetrics.functional.image import structural_similarity_index_measure as pt_ssim
        from torchmetrics.functional.regression import mean_squared_error as pt_mse

        comp = torch.from_numpy(np.transpose(reconstructed.astype(np.float32), (0, 3, 1, 2)))
        orig = torch.from_numpy(np.transpose(original.astype(np.float32), (0, 3, 1, 2)))
        valid_idx = (torch.isnan(comp) | torch.isnan(orig)).sum(dim=(1, 2, 3)) == 0
        comp = comp[valid_idx].contiguous()
        orig = orig[valid_idx].contiguous()

        data_range = metrics_value_range[1] - metrics_value_range[0]
        eps = 1e-8
        return {
            "ssim": float(pt_ssim(comp, orig).cpu().numpy().item()),
            "psnr": float(pt_psnr(comp, orig, data_range=data_range).cpu().numpy().item()),
            "mse": float(pt_mse(comp, orig).cpu().numpy().item()),
            "exp_sa": float(pt_sa(comp + eps, orig + eps).cpu().numpy().item()),
        }
    except Exception:
        from xarrayvideo.metrics import SA, SNR, SSIM

        min_val, max_val = metrics_value_range
        ssim_value = SSIM(original, reconstructed, channel_axis=-1, data_range=max_val - min_val)
        _, psnr, mse = SNR(original, reconstructed, max_value=max_val)
        exp_sa, _ = SA(original, reconstructed, channel_dim=-1)
        return {
            "ssim": float(ssim_value),
            "psnr": float(psnr),
            "mse": float(mse),
            "exp_sa": float(exp_sa),
        }


def build_model(model_name: str, compression: float, quality: int, device: str):
    from terracodec import flextec_v1_s2l2a, terracodec_v1_tt_s2l2a

    if model_name == "terracodec_v1_tt_s2l2a":
        return terracodec_v1_tt_s2l2a(pretrained=True, compression=compression, device=device)
    if model_name == "flextec_v1_s2l2a":
        return flextec_v1_s2l2a(pretrained=True, compression=800, device=device), quality
    raise RuntimeError(f"Unsupported model {model_name}")


def get_total_bits(payload: dict) -> int:
    if "bits" in payload:
        bits_value = payload["bits"]
        if hasattr(bits_value, "item"):
            return int(bits_value.item())
        if isinstance(bits_value, np.ndarray):
            return int(bits_value.sum())
        if isinstance(bits_value, (list, tuple)):
            total_bits = 0
            for item in bits_value:
                if hasattr(item, "item"):
                    total_bits += int(item.item())
                elif isinstance(item, np.ndarray):
                    total_bits += int(item.sum())
                else:
                    total_bits += int(item)
            return total_bits
        return int(bits_value)
    raise RuntimeError(f"TerraCodec payload does not expose a top-level 'bits' field: {payload.keys()}")


def log_progress(message: str) -> None:
    print(message, flush=True)


def run_terracodec(
    array: np.ndarray,
    model_name: str,
    compression: float,
    quality: int,
    tile_size: int,
    time_window: int,
    device: str,
    cube_name: str,
) -> tuple[np.ndarray, int, float, float]:
    import torch

    normalized = normalize_for_terracodec(array)
    padded, (pad_y, pad_x) = pad_to_multiple(normalized, tile_size)
    reconstructed = np.zeros_like(padded, dtype=np.float32)
    total_bits = 0
    total_encode_time = 0.0
    total_decode_time = 0.0
    windows = iter_time_windows(padded.shape[0], time_window)
    tiles_per_window = (padded.shape[1] // tile_size) * (padded.shape[2] // tile_size)
    total_tiles = max(1, len(windows) * max(1, tiles_per_window))
    tile_counter = 0

    model_bundle = build_model(model_name, compression, quality, device)
    if isinstance(model_bundle, tuple):
        model, flex_quality = model_bundle
    else:
        model, flex_quality = model_bundle, quality

    log_progress(
        "[terracodec] "
        f"cube={cube_name} compression={compression:g} device={device} "
        f"shape={array.shape} padded_shape={padded.shape} windows={len(windows)} total_tiles={total_tiles}"
    )

    for window_index, (time_start, time_stop) in enumerate(windows, start=1):
        log_progress(
            "[terracodec] "
            f"cube={cube_name} compression={compression:g} window={window_index}/{len(windows)} "
            f"time={time_start}:{time_stop}"
        )
        for y_start in range(0, padded.shape[1], tile_size):
            for x_start in range(0, padded.shape[2], tile_size):
                tile_counter += 1
                tile_label = (
                    "[terracodec] "
                    f"cube={cube_name} compression={compression:g} tile={tile_counter}/{total_tiles} "
                    f"window={window_index}/{len(windows)} y={y_start}:{y_start + tile_size} "
                    f"x={x_start}:{x_start + tile_size}"
                )
                log_progress(f"{tile_label} start")
                tile = padded[time_start:time_stop, y_start : y_start + tile_size, x_start : x_start + tile_size, :]
                tile_tensor = torch.from_numpy(np.transpose(tile, (0, 3, 1, 2))).to(torch.float32)

                encode_t0 = time.time()
                if model_name == "flextec_v1_s2l2a":
                    payload = model.compress(tile_tensor, quality=flex_quality)
                else:
                    payload = model.compress(tile_tensor)
                encode_t1 = time.time()

                decode_t0 = time.time()
                tile_recon = model.decompress(payload)
                decode_t1 = time.time()

                total_bits += get_total_bits(payload)
                total_encode_time += encode_t1 - encode_t0
                total_decode_time += decode_t1 - decode_t0

                log_progress(
                    f"{tile_label} done encode_s={encode_t1 - encode_t0:.2f} "
                    f"decode_s={decode_t1 - decode_t0:.2f} total_bits={total_bits}"
                )

                tile_recon_np = tile_recon.detach().to(dtype=torch.float32, device="cpu").numpy()
                if tile_recon_np.ndim == 5:
                    tile_recon_np = tile_recon_np[0]
                tile_recon_np = np.transpose(tile_recon_np, (0, 2, 3, 1))
                reconstructed[
                    time_start:time_stop,
                    y_start : y_start + tile_size,
                    x_start : x_start + tile_size,
                    :,
                ] = tile_recon_np

    reconstructed = denormalize_from_terracodec(reconstructed)
    if pad_y > 0:
        reconstructed = reconstructed[:, :-pad_y, :, :]
    if pad_x > 0:
        reconstructed = reconstructed[:, :, :-pad_x, :]
    log_progress(
        "[terracodec] "
        f"cube={cube_name} compression={compression:g} finished total_bits={total_bits} "
        f"encode_s={total_encode_time:.2f} decode_s={total_decode_time:.2f}"
    )
    return reconstructed, total_bits, total_encode_time, total_decode_time


def build_results_row(
    original: np.ndarray,
    reconstructed: np.ndarray,
    total_bits: int,
    encode_time: float,
    decode_time: float,
    metrics_value_range: tuple[float, float],
    observed_channel_mask: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = compute_metrics(
        original,
        reconstructed,
        metrics_value_range,
        observed_channel_mask=observed_channel_mask,
    )
    original_size = original.size * original.dtype.itemsize / 2**20
    compressed_size = total_bits / 8 / 2**20
    if observed_channel_mask is None:
        observed_bpppb = float(total_bits / original.size)
    else:
        observed_size = original[..., observed_channel_mask].size
        observed_bpppb = float(total_bits / observed_size)
    return {
        "original_size": float(original_size),
        "compressed_size": float(compressed_size),
        "compression": float(compressed_size / original_size),
        "bpppb": float(total_bits / original.size),
        "observed_bpppb": observed_bpppb,
        "time": float(encode_time),
        "d_time": float(decode_time),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    cube_name, dataset, source_bands = load_dataset(args)
    source_array = stack_bands(dataset, source_bands, args.time_dim, args.y_dim, args.x_dim)
    input_scale_factor, input_scale_reason = resolve_input_scale(
        source_array,
        mode=args.input_scale,
        reflectance_scale_factor=args.reflectance_scale_factor,
    )
    scaled_source_array = apply_input_scale(source_array, input_scale_factor)
    aligned_array, missing_bands, observed_channel_mask = align_to_terracodec_bands(
        scaled_source_array,
        source_bands,
        args.band_fill_mode,
    )
    metrics_value_range = parse_metrics_value_range(args.metrics_value_range, aligned_array)

    summary = {
        "cube": cube_name,
        "input_path": str(args.input_path),
        "source_bands": source_bands,
        "observed_bands": [band for band, keep in zip(TERRACODEC_S2L2A_BANDS, observed_channel_mask) if keep],
        "terracodec_bands": TERRACODEC_S2L2A_BANDS,
        "missing_bands": missing_bands,
        "input_scale": args.input_scale,
        "input_scale_factor": input_scale_factor,
        "input_scale_reason": input_scale_reason,
        "band_fill_mode": args.band_fill_mode,
        "shape": list(aligned_array.shape),
        "metrics_value_range": list(metrics_value_range),
        "model": args.model,
    }

    output_stem = args.output_stem or Path("results") / f"results_terracodec_{cube_name}"
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        with open(f"{output_stem}_summary.json", "w") as file_obj:
            json.dump(summary, file_obj, indent=2)
        print(json.dumps(summary, indent=2))
        return

    import torch

    device = infer_device(torch, args.device)
    summary["device"] = device
    compression_values = parse_compressions(args.compression_values)
    results_metrics: dict[tuple[str, str, str, int, str, str], float] = {}

    log_progress(
        "[terracodec] "
        f"cube={cube_name} starting device={device} compressions={compression_values} "
        f"shape={aligned_array.shape} output={output_stem}"
    )

    for compression in compression_values:
        log_progress(f"[terracodec] cube={cube_name} compression={compression:g} begin")
        reconstructed, total_bits, encode_time, decode_time = run_terracodec(
            aligned_array,
            model_name=args.model,
            compression=compression,
            quality=args.quality,
            tile_size=args.tile_size,
            time_window=args.time_window,
            device=device,
            cube_name=cube_name,
        )
        row = build_results_row(
            original=aligned_array,
            reconstructed=reconstructed,
            total_bits=total_bits,
            encode_time=encode_time,
            decode_time=decode_time,
            metrics_value_range=metrics_value_range,
            observed_channel_mask=observed_channel_mask,
        )
        crf_name = f"lambda-{compression:g}"
        for metric_name, value in row.items():
            results_metrics[(metric_name, "all", args.model, 12, cube_name, crf_name)] = value
        log_progress(
            "[terracodec] "
            f"cube={cube_name} compression={compression:g} metrics="
            f"bpppb={row['bpppb']:.6f} observed_bpppb={row['observed_bpppb']:.6f} "
            f"psnr={row['psnr']:.4f} ssim={row['ssim']:.4f} exp_sa={row['exp_sa']:.6f}"
        )

    results_df = pd.DataFrame.from_dict(results_metrics, orient="index", columns=["value"])
    results_df.index = pd.MultiIndex.from_tuples(results_df.index)
    results_df.index.names = ["metric", "video_name", "test_name", "bits", "cube", "crf"]
    results_df.to_pickle(f"{output_stem}.pkl")

    summary["compressions"] = compression_values
    with open(f"{output_stem}_summary.json", "w") as file_obj:
        json.dump(summary, file_obj, indent=2)

    log_progress(f"[terracodec] cube={cube_name} wrote {output_stem}.pkl and {output_stem}_summary.json")
    print(results_df)


if __name__ == "__main__":
    main()