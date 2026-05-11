"""
RAM usage measurement during xarrayvideo compression and decompression.

Measures peak RSS memory for different cube sizes by running each case in
an isolated subprocess, to avoid Python's memory caching polluting results.

Usage:
    python scripts/measure_ram.py
    python scripts/measure_ram.py --single <case_index>  # used internally
"""

import sys, os, gc, time, tempfile, json, subprocess, shutil, argparse
from pathlib import Path
import numpy as np
import xarray as xr
import psutil

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def make_synthetic_cube(n_time, n_y, n_x, n_bands, dtype=np.uint16) -> xr.Dataset:
    """Create a synthetic xarray Dataset with shape (n_time, n_y, n_x) per band."""
    rng = np.random.default_rng(seed=0)
    data = {}
    band_names = [f'B{i+1:02d}' for i in range(n_bands)]
    for b in band_names:
        arr = rng.integers(0, 2**16, size=(n_time, n_y, n_x), dtype=dtype)
        data[b] = xr.DataArray(arr, dims=['time', 'y', 'x'])
    return xr.Dataset(data), band_names


# Test cases: (n_time, n_y, n_x, n_bands, bits, label)
# Note: use only multiples of 3 for n_bands to avoid the padding edge case,
# OR use 7/10 which are valid real-world counts handled by xarrayvideo.
CASES = [
    # DeepExtremeCubes-like: 128x128, ~100 timesteps, 6 bands, 12-bit
    (100, 128, 128,  6, 12, 'Small 6-band 12bit (DX-like)'),
    # SimpleS2 / custom: 256x256, 9 bands, 12-bit
    (100, 256, 256,  9, 12, 'Medium 9-band 12bit (S2-like)'),
    # DynamicEarthNet-like: 1024x1024, 3 bands, 24 months, 12-bit
    ( 24, 1024, 1024, 3, 12, 'Large 3-band 12bit (DEN-like)'),
    # ERA5-like: 1440x721, 3 vars, 12-bit (libx265 max)
    ( 50,  721, 1440, 3, 12, 'ERA5-like 3-band 12bit'),
]


def run_single_case(case_idx: int, tmpdir: str) -> dict:
    """Run a single test case in the current process and return results."""
    from xarrayvideo import xarray2video, video2xarray

    PROCESS = psutil.Process(os.getpid())

    def rss():
        return PROCESS.memory_info().rss / 1024**2

    n_time, n_y, n_x, n_bands, bits, label = CASES[case_idx]
    x265_params = {'c:v': 'libx265', 'preset': 'medium', 'tune': 'psnr', 'crf': 16}

    cube, band_names = make_synthetic_cube(n_time, n_y, n_x, n_bands)
    conversion_rules = {'all': (band_names, ('time', 'y', 'x'), False, x265_params, bits)}

    out_dir = Path(tmpdir, 'ram_test')
    if out_dir.exists():
        shutil.rmtree(out_dir)

    gc.collect()
    rss_baseline = rss()
    uncompressed_mb = cube.nbytes / 1024**2

    # --- encode ---
    t0 = time.perf_counter()
    xarray2video(cube, 'ram_test', conversion_rules,
                 compute_stats=False, output_path=tmpdir, verbose=False)
    encode_time = time.perf_counter() - t0
    rss_peak_encode = rss()

    compressed_mb = sum(
        f.stat().st_size for f in Path(tmpdir, 'ram_test').rglob('*') if f.is_file()
    ) / 1024**2

    # --- decode ---
    gc.collect()
    rss_before_decode = rss()
    t0 = time.perf_counter()
    _ = video2xarray(tmpdir, 'ram_test')
    decode_time = time.perf_counter() - t0
    rss_peak_decode = rss()

    return {
        'label': label,
        'shape': (n_time, n_y, n_x, n_bands),
        'bits': bits,
        'uncompressed_mb': uncompressed_mb,
        'compressed_mb': compressed_mb,
        'rss_baseline_mb': rss_baseline,
        'rss_peak_encode_mb': rss_peak_encode,
        'rss_delta_encode_mb': rss_peak_encode - rss_baseline,
        'rss_before_decode_mb': rss_before_decode,
        'rss_peak_decode_mb': rss_peak_decode,
        'rss_delta_decode_mb': rss_peak_decode - rss_before_decode,
        'encode_time_s': encode_time,
        'decode_time_s': decode_time,
        'compression_ratio': uncompressed_mb / compressed_mb if compressed_mb > 0 else 0,
    }


def print_results(results: list):
    header = (
        f"{'Case':<32} {'Uncomp(MB)':>10} {'Compr(MB)':>9} {'Ratio':>6} "
        f"{'Peak_enc(MB)':>12} {'Overhead_enc':>12} "
        f"{'Peak_dec(MB)':>12} {'Enc(s)':>7} {'Dec(s)':>7}"
    )
    print()
    print(header)
    print('-' * len(header))
    for r in results:
        print(
            f"{r['label']:<32} "
            f"{r['uncompressed_mb']:>10.1f} "
            f"{r['compressed_mb']:>9.1f} "
            f"{r['compression_ratio']:>6.1f}x "
            f"{r['rss_peak_encode_mb']:>12.1f} "
            f"{r['rss_delta_encode_mb']:>12.1f} "
            f"{r['rss_peak_decode_mb']:>12.1f} "
            f"{r['encode_time_s']:>7.2f} "
            f"{r['decode_time_s']:>7.2f}"
        )
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', type=int, default=None,
                        help='Run a single case index (used internally by subprocess mode)')
    parser.add_argument('--tmpdir', type=str, default=None,
                        help='Temp dir to use (used internally by subprocess mode)')
    args = parser.parse_args()

    if args.single is not None:
        # Subprocess mode: run one case, print JSON result, exit
        result = run_single_case(args.single, args.tmpdir)
        print('JSON_RESULT:' + json.dumps(result))
        sys.exit(0)

    # Main mode: spawn a subprocess per case for clean RSS isolation
    all_results = []
    python = sys.executable

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (n_time, n_y, n_x, n_bands, bits, label) in enumerate(CASES):
            uncompressed_mb = (n_time * n_y * n_x * n_bands * 2) / 1024**2
            print(f"[{i+1}/{len(CASES)}] {label}  "
                  f"shape=({n_time},{n_y},{n_x})  {n_bands}bands  {bits}bit  "
                  f"~{uncompressed_mb:.0f}MB uncompressed")

            cmd = [python, __file__, '--single', str(i), '--tmpdir', tmpdir]
            proc = subprocess.run(cmd, capture_output=True, text=True)

            if proc.returncode != 0:
                print(f"  ERROR (returncode={proc.returncode}):")
                for line in proc.stderr.splitlines()[-5:]:
                    print(f"    {line}")
                continue

            # Extract JSON from stdout (there may be x265 info lines too)
            result = None
            for line in proc.stdout.splitlines():
                if line.startswith('JSON_RESULT:'):
                    result = json.loads(line[len('JSON_RESULT:'):])
                    break

            if result is None:
                print(f"  ERROR: no JSON result found in output")
                print(proc.stdout[-500:])
                continue

            all_results.append(result)
            print(f"  -> uncompressed={result['uncompressed_mb']:.1f}MB  "
                  f"compressed={result['compressed_mb']:.1f}MB  "
                  f"ratio={result['compression_ratio']:.1f}x  "
                  f"peak_RSS_encode={result['rss_peak_encode_mb']:.0f}MB  "
                  f"overhead_encode={result['rss_delta_encode_mb']:.0f}MB  "
                  f"peak_RSS_decode={result['rss_peak_decode_mb']:.0f}MB")

    print_results(all_results)

    print("=== SUMMARY ===")
    for r in all_results:
        ratio = r['rss_delta_encode_mb'] / r['uncompressed_mb'] if r['uncompressed_mb'] else 0
        print(f"  {r['label']}: encode overhead / uncompressed = {ratio:.2f}x  "
              f"(uncompressed={r['uncompressed_mb']:.0f}MB, "
              f"overhead={r['rss_delta_encode_mb']:.0f}MB, "
              f"peak={r['rss_peak_encode_mb']:.0f}MB)")
