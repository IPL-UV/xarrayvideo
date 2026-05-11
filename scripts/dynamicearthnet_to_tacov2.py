"""DynamicEarthNet-video: v1 → v2 migration script."""

import json
import ast
import math
import shutil
from pathlib import Path

import pyarrow as pa
import fsspec
import tacotoolbox
from tacotoolbox.datamodel import Taco, Sample, Tortilla
from tacotoolbox.datamodel.sample import STAC
from tacotoolbox.create import create
import tacoreader
import tacoreader.v1 as tacoreader_v1


# --- Helpers ---
def safe_str(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return str(val)

def safe_list(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val

def safe_float(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


# --- Core ---
def _build_sample(idx: int, datasetv1, f, temp_dir: Path) -> Sample:
    """Build a single v2 sample from v1."""
    level1 = datasetv1.read(idx)
    meta = datasetv1.iloc[idx]

    inner_samples = []
    for _, row in level1.iterrows():
        f.seek(row["tortilla:offset"])
        file_bytes = f.read(row["tortilla:length"])
        inner_samples.append(Sample(
            id=row["tortilla:id"],
            type="FILE",
            path=file_bytes,
            temp_dir=str(temp_dir),
        ))

    sample = Sample(
        id=meta["tortilla:id"],
        type="FOLDER",
        path=Tortilla(inner_samples),
    )

    sample.extend_with(STAC(
        crs=meta["stac:crs"],
        geotransform=meta["stac:geotransform"],
        tensor_shape=meta["stac:raster_shape"],
        time_start=int(meta["stac:time_start"]),
        time_end=int(meta["stac:time_end"]),
    ))

    return sample


def migrate(
    input_file: str,
    output_file: str,
    n_samples: int | None = None,
    temp_dir: str = "trash",
    split_size: str = "4GB",
) -> list[Path]:
    """Migrate DynamicEarthNet-video v1 → v2.

    Args:
        input_file: URL or path to the v1 .taco file.
        output_file: Path for the output .tacozip file.
        n_samples: Number of samples to process. None = all.
        temp_dir: Directory for temporary files.
        split_size: Max size per ZIP file.

    Returns:
        List of created file paths.
    """
    temp_path = Path(temp_dir)
    output_path = Path(output_file)

    datasetv1 = tacoreader_v1.load(input_file)
    total = len(datasetv1) if n_samples is None else min(n_samples, len(datasetv1))

    temp_path.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    root_samples = []
    with fsspec.open(input_file, "rb") as f:
        for idx in range(total):
            sid = datasetv1.iloc[idx]["tortilla:id"]
            print(f"[{idx+1}/{total}] {sid}")
            sample = _build_sample(idx, datasetv1, f, temp_path)
            root_samples.append(sample)

    taco = Taco(
        id="dynamicearthnet-video",
        dataset_version="2.0.0",
        description=(
            "DynamicEarthNet-video is a storage-efficient re-packaging of the original "
            "DynamicEarthNet collection. The archive covers 75 regions (1024x1024 px, ~3 m GSD) "
            "sampled daily from 2018-01-01 to 2019-12-31. Each day is a 4-band PlanetFusion "
            "surface-reflectance image (R, G, B, Narrow-NIR). Monthly pixel-wise labels annotate "
            "7 land-cover classes. All GeoTIFF stacks (~525 GB) are transcoded with xarrayvideo "
            "to 12-bit H.265/HEVC yielding ~62x compression at 77dB PSNR and 250x compression at 71dB PSNR."
        ),
        licenses=["CC-BY-4.0"],
        tasks=["foundation-model"],
        providers=[
            {"name": "Planet Labs PBC", "roles": ["producer"], "url": "https://www.planet.com"},
        ],
        curators=[
            {"name": "Oscar J. Pellicer-Valero", "organization": "Image Signal Processing (ISP)"},
            {"name": "Cesar Aybar", "organization": "Image Signal Processing (ISP)"},
            {"name": "Julio Contreras", "organization": "Image Signal Processing (ISP)"},
        ],
        tortilla=Tortilla(root_samples),
    )

    tacotoolbox.verbose(True)
    print(f"\n{total} samples ready. Creating TACO -> {output_path}...")
    paths = create(
        taco=taco,
        output=str(output_path),
        split_size=split_size,
        consolidate=False,
    )

    if temp_path.exists():
        shutil.rmtree(temp_path)

    print(f"Done -> {[str(p) for p in paths]}")
    return paths


if __name__ == "__main__":
    # BASE = "https://huggingface.co/datasets/isp-uv-es/DynamicEarthNet-video/resolve/main/"
    BASE = "/data/databases/dynamicearthnet-video-final/"

    INPUT = BASE + "DynamicEarthNet-video-53psnr.taco"
    OUTPUT = "/data/databases/dynamicearthnet-video-final/dynamicearthnet-video-71psnr.tacozip"
    paths = migrate(input_file=INPUT, output_file=OUTPUT, n_samples=None, split_size="40GB")

    INPUT = BASE + "DynamicEarthNet-video-60psnr.taco"
    OUTPUT = "/data/databases/dynamicearthnet-video-final/dynamicearthnet-video-77psnr.tacozip"
    paths = migrate(input_file=INPUT, output_file=OUTPUT, n_samples=None, split_size="40GB")

    # Verify
    tacoreader.use("pandas")
    ds = tacoreader.load(str(paths[0]))
    print(ds.read(0))