"""Quick reproducible check for TerraCodec input scaling.

This script exercises the same loading path as ``run_terracodec_tests.py`` and
prints the inferred scale factor that will be applied before TerraCodec
normalization. It is intended as a fast sanity check for Sentinel-2 public
releases that may store reflectance in ``[0, 1]`` instead of original L2A-like
units.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_terracodec_tests import (
    align_to_terracodec_bands,
    apply_input_scale,
    load_dataset,
    resolve_input_scale,
    stack_bands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the scale applied before TerraCodec normalization")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--input-format", choices=["auto", "netcdf", "zarr", "pickle"], default="auto")
    parser.add_argument("--dataset", choices=["generic", "deepextremes"], default="generic")
    parser.add_argument("--bands", type=str, required=True)
    parser.add_argument("--cube-name", type=str, default=None)
    parser.add_argument("--time-dim", type=str, default="time")
    parser.add_argument("--y-dim", type=str, default="y")
    parser.add_argument("--x-dim", type=str, default="x")
    parser.add_argument("--time-range", type=str, default=None)
    parser.add_argument("--spatial-crop", type=str, default=None)
    parser.add_argument("--band-fill-mode", choices=["strict", "zeros", "mean"], default="strict")
    parser.add_argument("--input-scale", choices=["auto", "native", "reflectance_0_1"], default="auto")
    parser.add_argument("--reflectance-scale-factor", type=float, default=10000.0)
    parser.add_argument("--expected-scale-factor", type=float, default=None)
    return parser.parse_args()


def summarize(array: np.ndarray) -> dict[str, float]:
    finite = array[np.isfinite(array)]
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def main() -> None:
    args = parse_args()
    _, dataset, source_bands = load_dataset(args)
    source_array = stack_bands(dataset, source_bands, args.time_dim, args.y_dim, args.x_dim)
    scale_factor, scale_reason = resolve_input_scale(
        source_array,
        mode=args.input_scale,
        reflectance_scale_factor=args.reflectance_scale_factor,
    )
    scaled_array = apply_input_scale(source_array, scale_factor)
    aligned_array, missing_bands, observed_channel_mask = align_to_terracodec_bands(
        scaled_array,
        source_bands,
        args.band_fill_mode,
    )

    if args.expected_scale_factor is not None and not np.isclose(scale_factor, args.expected_scale_factor):
        raise SystemExit(
            f"Expected scale factor {args.expected_scale_factor}, got {scale_factor} ({scale_reason})"
        )

    payload = {
        "input_path": str(args.input_path),
        "source_bands": source_bands,
        "input_scale": args.input_scale,
        "input_scale_factor": scale_factor,
        "input_scale_reason": scale_reason,
        "source_summary": summarize(source_array),
        "scaled_summary": summarize(scaled_array),
        "aligned_summary": summarize(aligned_array),
        "missing_bands": missing_bands,
        "observed_bands": int(observed_channel_mask.sum()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()