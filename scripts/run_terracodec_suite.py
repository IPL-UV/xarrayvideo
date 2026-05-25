"""Run TerraCodec across the repo's benchmark cube suites.

This orchestrates repeated invocations of ``scripts/run_terracodec_tests.py`` on
the same 10 DeepExtremeCubes samples currently used by ``scripts/run_tests.py``
and on the 4 custom SimpleS2 cubes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_terracodec_tests.py"
DEFAULT_DX_ROOT = Path("/data/databases/deepextremes_final")
DEFAULT_CUSTOM_ROOTS = [REPO_ROOT.parent / "cubos_julio", REPO_ROOT / "cubos_julio"]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "terracodec_suite"
DEFAULT_DX_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B8A"]
DEFAULT_CUSTOM_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
DEFAULT_COMPRESSIONS = "0.4,1,5,20,100,200,700"

# Frozen from the current scripts/run_tests.py selection logic:
#   rng = np.random.default_rng(seed=42)
#   paths = np.array(list(Path('/data/databases/deepextremes_final').glob('*.zarr')))
#   rng.shuffle(paths)
#   paths[:10]
MAIN_TEST_DX_CUBES = [
    "mc_-117.15_51.09_1.3_20231019_0",
    "mc_-68.79_-8.25_1.3_20230928_0",
    "mc_19.88_-16.74_1.3_20231018_0",
    "mc_-68.16_9.87_1.3_20230916_0",
    "mc_5.06_32.48_1.3_20231019_0",
    "mc_60.50_49.75_1.1_20230602_0",
    "mc_28.25_68.17_1.1_20230605_0",
    "mc_24.49_-10.49_1.1_20230611_0",
    "mc_-119.75_49.25_1.1_20230602_0",
    "mc_165.95_62.19_1.3_20231018_0",
]


@dataclass(frozen=True)
class CubeRun:
    dataset_label: str
    cube_name: str
    input_path: str
    input_format: str
    dataset_mode: str
    bands: list[str]
    band_fill_mode: str
    output_stem: str


def compression_output_root(output_root: Path, compression_values: str) -> Path:
    values = [item.strip() for item in compression_values.split(",") if item.strip()]
    if not values:
        return output_root

    normalized = [value.replace(".", "p") for value in values]
    if len(normalized) == 1:
        suffix = f"lambda_{normalized[0]}"
    else:
        suffix = f"lambdas_{'_'.join(normalized)}"

    if output_root.name == suffix:
        return output_root
    return output_root / suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TerraCodec on the benchmark cube suites")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--datasets", type=str, default="custom,deepextremes")
    parser.add_argument("--dx-root", type=Path, default=DEFAULT_DX_ROOT)
    parser.add_argument("--custom-root", type=Path, default=None)
    parser.add_argument("--dx-cubes", type=str, default=",".join(MAIN_TEST_DX_CUBES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--compression-values", type=str, default=DEFAULT_COMPRESSIONS)
    parser.add_argument("--model", choices=["terracodec_v1_tt_s2l2a", "flextec_v1_s2l2a"], default="terracodec_v1_tt_s2l2a")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--time-window", type=int, default=4)
    parser.add_argument("--metrics-value-range", type=str, default="auto")
    parser.add_argument("--input-scale", choices=["auto", "native", "reflectance_0_1"], default="auto")
    parser.add_argument("--reflectance-scale-factor", type=float, default=10000.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-cubes", type=int, default=None)
    parser.add_argument("--cube-filter", type=str, default=None)
    parser.add_argument("--keep-going", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()


def resolve_custom_root(explicit_root: Path | None) -> Path:
    candidates = [explicit_root] if explicit_root is not None else []
    candidates.extend(DEFAULT_CUSTOM_ROOTS)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise RuntimeError(
        "Could not resolve the custom cube root. Pass --custom-root or ensure one of "
        f"{[str(path) for path in DEFAULT_CUSTOM_ROOTS]} exists."
    )


def build_suite(args: argparse.Namespace) -> list[CubeRun]:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    suite: list[CubeRun] = []
    job_output_root = compression_output_root(args.output_root, args.compression_values)

    if "deepextremes" in datasets:
        dx_names = [item.strip() for item in args.dx_cubes.split(",") if item.strip()]
        for cube_name in dx_names:
            input_path = args.dx_root / f"{cube_name}.zarr"
            if not input_path.exists():
                raise RuntimeError(f"DeepExtremeCubes input not found: {input_path}")
            suite.append(
                CubeRun(
                    dataset_label="deepextremes",
                    cube_name=cube_name,
                    input_path=str(input_path),
                    input_format="zarr",
                    dataset_mode="deepextremes",
                    bands=DEFAULT_DX_BANDS,
                    band_fill_mode="mean",
                    output_stem=str(job_output_root / "deepextremes" / cube_name),
                )
            )

    if "custom" in datasets:
        custom_root = resolve_custom_root(args.custom_root)
        for index in range(1, 5):
            cube_name = f"cubo{index}_pickle"
            input_path = custom_root / cube_name
            if not input_path.exists():
                raise RuntimeError(f"Custom cube input not found: {input_path}")
            suite.append(
                CubeRun(
                    dataset_label="custom",
                    cube_name=cube_name,
                    input_path=str(input_path),
                    input_format="pickle",
                    dataset_mode="generic",
                    bands=DEFAULT_CUSTOM_BANDS,
                    band_fill_mode="mean",
                    output_stem=str(job_output_root / "custom" / cube_name),
                )
            )

    cube_filter = args.cube_filter.strip() if args.cube_filter else None
    if cube_filter:
        suite = [item for item in suite if cube_filter in item.cube_name or cube_filter in item.dataset_label]

    suite = suite[args.start_index :]
    if args.max_cubes is not None:
        suite = suite[: args.max_cubes]

    if not suite:
        raise RuntimeError("The requested suite is empty after applying filters/slicing")
    return suite


def build_command(args: argparse.Namespace, cube: CubeRun) -> list[str]:
    return [
        str(args.python_bin),
        str(args.runner),
        "--input-path",
        cube.input_path,
        "--input-format",
        cube.input_format,
        "--dataset",
        cube.dataset_mode,
        "--bands",
        ",".join(cube.bands),
        "--band-fill-mode",
        cube.band_fill_mode,
        "--compression-values",
        args.compression_values,
        "--model",
        args.model,
        "--quality",
        str(args.quality),
        "--device",
        args.device,
        "--tile-size",
        str(args.tile_size),
        "--time-window",
        str(args.time_window),
        "--metrics-value-range",
        args.metrics_value_range,
        "--input-scale",
        args.input_scale,
        "--reflectance-scale-factor",
        str(args.reflectance_scale_factor),
        "--cube-name",
        cube.cube_name,
        "--output-stem",
        cube.output_stem,
    ]


def shell_join(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) if os.name == "nt" else subprocess.list2cmdline([part]) for part in command)


def main() -> None:
    args = parse_args()
    suite = build_suite(args)
    job_output_root = compression_output_root(args.output_root, args.compression_values)
    job_output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "datasets": args.datasets,
        "compression_values": args.compression_values,
        "model": args.model,
        "quality": args.quality,
        "device": args.device,
        "tile_size": args.tile_size,
        "time_window": args.time_window,
        "metrics_value_range": args.metrics_value_range,
        "input_scale": args.input_scale,
        "reflectance_scale_factor": args.reflectance_scale_factor,
        "start_index": args.start_index,
        "max_cubes": args.max_cubes,
        "cube_filter": args.cube_filter,
        "cubes": [asdict(item) for item in suite],
    }
    manifest_path = job_output_root / "suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Suite manifest: {manifest_path}", flush=True)

    failures: list[tuple[str, int]] = []
    for index, cube in enumerate(suite, start=1):
        command = build_command(args, cube)
        print("", flush=True)
        print(f"[{index}/{len(suite)}] {cube.dataset_label}:{cube.cube_name}", flush=True)
        print(shell_join(command), flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        print(
            f"[suite] completed {cube.dataset_label}:{cube.cube_name} return_code={completed.returncode}",
            flush=True,
        )
        if completed.returncode != 0:
            failures.append((cube.cube_name, completed.returncode))
            if not args.keep_going:
                raise SystemExit(completed.returncode)

    if failures:
        for cube_name, return_code in failures:
            print(f"FAILED {cube_name}: return code {return_code}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()