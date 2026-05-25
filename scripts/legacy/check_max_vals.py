import xarray as xr
import numpy as np
from pathlib import Path
import pickle
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from xarrayvideo import video2xarray


def sample_files(file_list, n=50):
    """Return up to n randomly chosen files from file_list."""
    if len(file_list) <= n:
        return file_list
    return random.sample(file_list, n)


def var_stats(ds, skip_vars=()):
    """Return {var: (min, max)} for all numeric variables in ds."""
    results = {}
    for var in ds.data_vars:
        if var in skip_vars:
            continue
        if ds[var].dtype.kind not in 'ifu':
            continue
        arr = ds[var].values
        results[var] = (float(np.nanmin(arr)), float(np.nanmax(arr)))
    return results


def check_dataset_files(label, files, open_fn, skip_vars=()):
    """Open each file, compute per-variable min/max, print summary with
    mean/min/max of per-cube maxima (and minima) across all samples."""
    if not files:
        print(f"{label}: no files found.")
        return

    print(f"\n{label}: checking {len(files)} sample(s)...")

    # per_var_maxima[var] = list of per-cube max values
    # per_var_minima[var] = list of per-cube min values
    per_var_maxima = {}
    per_var_minima = {}

    for fpath in files:
        try:
            ds = open_fn(fpath)
            vstats = var_stats(ds, skip_vars=skip_vars)
            ds.close()
            for var, (vmin, vmax) in vstats.items():
                per_var_maxima.setdefault(var, []).append(vmax)
                per_var_minima.setdefault(var, []).append(vmin)
        except Exception as e:
            print(f"  [{fpath.name}] error: {e}")

    if not per_var_maxima:
        print("  No numeric variables found.")
        return

    # Overall statistics across all cubes and variables
    all_maxima = [v for vals in per_var_maxima.values() for v in vals]
    print(f"  Overall max across all samples: {max(all_maxima):.4f}")

    # Per-variable summary: min/mean/max of per-cube maxima, sorted by global max
    global_max_per_var = {var: max(vals) for var, vals in per_var_maxima.items()}
    print(f"\n  {'Variable':<20}  {'global_min':>12}  {'mean_of_maxima':>16}  "
          f"{'min_of_maxima':>14}  {'max_of_maxima':>14}")
    print("  " + "-" * 80)
    for var, gmx in sorted(global_max_per_var.items(), key=lambda x: -x[1]):
        maxvals = np.array(per_var_maxima[var])
        minvals = np.array(per_var_minima[var])
        print(f"  {var:<20}  {np.min(minvals):>12.4f}  {np.mean(maxvals):>16.4f}  "
              f"{np.min(maxvals):>14.4f}  {np.max(maxvals):>14.4f}")


def check_datasets():
    print("Checking datasets min/max ranges...")

    # ── 1. DynamicEarthNet ────────────────────────────────────────────────────
    path_den = Path('/scratch/users/databases/dynamicearthnet-xarray')
    ncs = list(path_den.glob('*.nc'))
    if ncs:
        check_dataset_files(
            "DynamicEarthNet",
            sample_files(ncs, n=50),
            open_fn=lambda f: xr.open_dataset(f),
            skip_vars=('labels',),  # skip non-spectral variable
        )
    else:
        print("\nDynamicEarthNet: no .nc files found at", path_den)

    # ── 2. DeepExtremeCubes (main path) ───────────────────────────────────────
    path_dx = Path('/scratch/users/databases/deepextremes_final')
    zarrs = list(path_dx.glob('*.zarr'))
    if zarrs:
        check_dataset_files(
            "DeepExtremeCubes (deepextremes_final)",
            sample_files(zarrs, n=50),
            open_fn=lambda f: xr.open_dataset(f, engine='zarr'),
            skip_vars=('SCL', 'cloudmask_en'),
        )
    else:
        print("\nDeepExtremeCubes: no .zarr files found at", path_dx)

    # ── 3. SimpleS2 (all 4 cubes) ─────────────────────────────────────────────
    # Search for pickle files; ../cubos_julio is the real original data path
    cubo_paths = []
    for base in ['cubos_julio', '../cubos_julio']:
        for i in range(1, 5):
            p = Path(base) / f'cubo{i}_pickle'
            if p.is_file():
                cubo_paths.append(p)
        if cubo_paths:
            break  # found at least one, use this base

    if cubo_paths:
        print(f"\nSimpleS2: checking {len(cubo_paths)} cube(s)...")
        for p in cubo_paths:
            try:
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                bands = data.coords['band'].values if 'band' in data.coords else ['all']
                mn = float(data.min().values)
                mx = float(data.max().values)
                mean_val = float(data.mean().values)
                print(f"  {p.name}: min={mn:.2f}, mean={mean_val:.2f}, max={mx:.2f}  "
                      f"(dtype={data.dtype}, bands={list(bands)})")
            except Exception as e:
                print(f"  {p.name}: error — {e}")
    else:
        print("\nSimpleS2: no pickle files found (checked cubos_julio/ and ../cubos_julio/)")


if __name__ == '__main__':
    random.seed(42)
    check_datasets()
