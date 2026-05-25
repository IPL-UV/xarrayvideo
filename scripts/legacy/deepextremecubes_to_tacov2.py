"""DeepExtremeCubes-video: v1 → v2 migration script."""

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
from tacotoolbox.sample.datamodel import SampleExtension
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


# --- Extension ---
class DeepExtremeCubes(SampleExtension):
    mc_id: str
    version: str
    geometry: str
    events: list
    events_corrected: list | None = None
    pure_class: str | None = None
    mixed_dominant_class: str | None = None
    mixed_second_dominant_class: str | None = None
    continent: str
    group: float | None = None
    has_missing_data: bool = False
    remarks: str | None = None
    s2_l2_bands: float | None = None
    era5: float | None = None
    cci_landcover_map: float | None = None
    copernicus_dem: float | None = None
    de_africa_climatology: float | None = None
    event_arrays: float | None = None
    s2cloudless_cloudmask: float | None = None
    sen2cor_cloudmask: float | None = None
    unetmobv2_cloudmask: float | None = None
    earthnet_cloudmask: float | None = None

    def get_schema(self) -> dict[str, pa.DataType]:
        return {
            "dec:mc_id": pa.string(),
            "dec:version": pa.string(),
            "dec:geometry": pa.string(),
            "dec:events": pa.string(),
            "dec:events_corrected": pa.string(),
            "dec:pure_class": pa.string(),
            "dec:mixed_dominant_class": pa.string(),
            "dec:mixed_second_dominant_class": pa.string(),
            "dec:continent": pa.string(),
            "dec:group": pa.float32(),
            "dec:has_missing_data": pa.bool_(),
            "dec:remarks": pa.string(),
            "dec:s2_l2_bands": pa.float32(),
            "dec:era5": pa.float32(),
            "dec:cci_landcover_map": pa.float32(),
            "dec:copernicus_dem": pa.float32(),
            "dec:de_africa_climatology": pa.float32(),
            "dec:event_arrays": pa.float32(),
            "dec:s2cloudless_cloudmask": pa.float32(),
            "dec:sen2cor_cloudmask": pa.float32(),
            "dec:unetmobv2_cloudmask": pa.float32(),
            "dec:earthnet_cloudmask": pa.float32(),
        }

    def get_field_descriptions(self) -> dict[str, str]:
        return {
            "dec:mc_id": "Minicube identifier",
            "dec:version": "Dataset version",
            "dec:geometry": "WKT polygon geometry",
            "dec:events": "JSON list of (event_id, start_date, end_date) tuples",
            "dec:events_corrected": "Corrected event annotations",
            "dec:pure_class": "Pure land cover class",
            "dec:mixed_dominant_class": "Dominant class in mixed pixels",
            "dec:mixed_second_dominant_class": "Second dominant class in mixed pixels",
            "dec:continent": "Continent name",
            "dec:group": "Group assignment",
            "dec:has_missing_data": "Whether minicube has missing data",
            "dec:remarks": "Additional notes",
            "dec:s2_l2_bands": "Sentinel-2 L2 bands version (-1 = N/A)",
            "dec:era5": "ERA5 reanalysis version (-1 = N/A)",
            "dec:cci_landcover_map": "CCI land cover map version (-1 = N/A)",
            "dec:copernicus_dem": "Copernicus DEM version (-1 = N/A)",
            "dec:de_africa_climatology": "DE Africa climatology version (-1 = N/A)",
            "dec:event_arrays": "Event arrays version (-1 = N/A)",
            "dec:s2cloudless_cloudmask": "S2Cloudless cloud mask version (-1 = N/A)",
            "dec:sen2cor_cloudmask": "Sen2Cor cloud mask version (-1 = N/A)",
            "dec:unetmobv2_cloudmask": "UNetMobV2 cloud mask version (-1 = N/A)",
            "dec:earthnet_cloudmask": "EarthNet cloud mask version (-1 = N/A)",
        }

    def _compute(self, sample) -> pa.Table:
        data = {
            "dec:mc_id": [self.mc_id],
            "dec:version": [self.version],
            "dec:geometry": [self.geometry],
            "dec:events": [json.dumps(self.events)],
            "dec:events_corrected": [json.dumps(self.events_corrected) if self.events_corrected else None],
            "dec:pure_class": [self.pure_class],
            "dec:mixed_dominant_class": [self.mixed_dominant_class],
            "dec:mixed_second_dominant_class": [self.mixed_second_dominant_class],
            "dec:continent": [self.continent],
            "dec:group": [self.group],
            "dec:has_missing_data": [self.has_missing_data],
            "dec:remarks": [self.remarks],
            "dec:s2_l2_bands": [self.s2_l2_bands],
            "dec:era5": [self.era5],
            "dec:cci_landcover_map": [self.cci_landcover_map],
            "dec:copernicus_dem": [self.copernicus_dem],
            "dec:de_africa_climatology": [self.de_africa_climatology],
            "dec:event_arrays": [self.event_arrays],
            "dec:s2cloudless_cloudmask": [self.s2cloudless_cloudmask],
            "dec:sen2cor_cloudmask": [self.sen2cor_cloudmask],
            "dec:unetmobv2_cloudmask": [self.unetmobv2_cloudmask],
            "dec:earthnet_cloudmask": [self.earthnet_cloudmask],
        }
        schema = pa.schema([(k, v) for k, v in self.get_schema().items()])
        return pa.table(data, schema=schema)


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

    sample.extend_with(DeepExtremeCubes(
        mc_id=meta["mc_id"],
        version=str(meta["version"]),
        geometry=meta["geometry"],
        events=safe_list(meta["events"]),
        events_corrected=safe_list(meta.get("events_corrected")),
        pure_class=safe_str(meta.get("pure_class")),
        mixed_dominant_class=safe_str(meta.get("mixed_dominant_class")),
        mixed_second_dominant_class=safe_str(meta.get("mixed_second_dominant_class")),
        continent=meta["continent"],
        group=safe_float(meta.get("group")),
        has_missing_data=bool(meta.get("has_missing_data", False)),
        remarks=safe_str(meta.get("remarks")),
        s2_l2_bands=safe_float(meta.get("s2_l2_bands")),
        era5=safe_float(meta.get("era5")),
        cci_landcover_map=safe_float(meta.get("cci_landcover_map")),
        copernicus_dem=safe_float(meta.get("copernicus_dem")),
        de_africa_climatology=safe_float(meta.get("de_africa_climatology")),
        event_arrays=safe_float(meta.get("event_arrays")),
        s2cloudless_cloudmask=safe_float(meta.get("s2cloudless_cloudmask")),
        sen2cor_cloudmask=safe_float(meta.get("sen2cor_cloudmask")),
        unetmobv2_cloudmask=safe_float(meta.get("unetmobv2_cloudmask")),
        earthnet_cloudmask=safe_float(meta.get("earthnet_cloudmask")),
    ))

    return sample


def migrate(
    input_file: str,
    output_file: str,
    n_samples: int | None = None,
    temp_dir: str = "trash",
    split_size: str = "4GB",
) -> list[Path]:
    """Migrate DeepExtremeCubes v1 → v2.

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
        id="deepextremecubes-video",
        dataset_version="2.0.0",
        description=(
            "DeepExtremeCubes-video is a storage-efficient, analysis-ready re-packaging "
            "of the original DeepExtremeCubes collection. All 42k Sentinel-2 minicubes "
            "(2.56 km x 2.56 km, 2016-2022, 7 bands, 5-daily cadence) have been transcoded "
            "with xarrayvideo into H.265/HEVC videos, achieving ~7.3x lossless-perceptual "
            "compression (~323 GB vs 2.3 TB) at ~68 dB PSNR."
        ),
        licenses=["CC-BY-4.0"],
        tasks=["foundation-model"],
        providers=[
            {"name": "European Space Agency (ESA)", "roles": ["producer"], "url": "https://sentinel.esa.int"},
            {"name": "ECMWF", "roles": ["producer"], "url": "https://climate.copernicus.eu"},
            {"name": "Copernicus DEM", "roles": ["contributor"], "url": "https://land.copernicus.eu"},
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

    # Final cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)

    print(f"Done -> {[str(p) for p in paths]}")
    return paths


if __name__ == "__main__":
    # BASE = "https://huggingface.co/datasets/isp-uv-es/DeepExtremeCubes-video/resolve/main/"
    BASE = "/data/databases/deepextremecubes-video-final/"

    for file in Path(BASE).glob("*.taco"):
        print(f"Migrating {file}...")
        output = file.with_suffix(".tacozip")
        migrate(input_file=str(file), output_file=str(output), n_samples=None, split_size="40GB")

    # Verify
    tacoreader.use("pandas")
    ds = tacoreader.load(str(output))
    print(ds.read(0))