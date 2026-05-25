#!/bin/bash

CODEC_ROOT="${1:-$HOME/codec_toolchains}"
codec_dirs=(
  "$CODEC_ROOT/vvenc/bin/release-static"
  "$CODEC_ROOT/vvenc/build/bin/release-static"
  "$CODEC_ROOT/vvenc/build/bin"
  "$CODEC_ROOT/vvdec/bin/release-static"
  "$CODEC_ROOT/vvdec/build/bin/release-static"
  "$CODEC_ROOT/vvdec/build/bin"
  "$CODEC_ROOT/uavs3e/build"
  "$CODEC_ROOT/uavs3d/build"
  "$CODEC_ROOT/VVCSoftware_VTM/bin"
  "$CODEC_ROOT/VVCSoftware_VTM/build/bin"
)

missing_dirs=()
for codec_dir in "${codec_dirs[@]}"; do
  if [[ -d "$codec_dir" ]]; then
    PATH="$codec_dir:$PATH"
  else
    missing_dirs+=("$codec_dir")
  fi
done

export PATH

if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ${#missing_dirs[@]} -gt 0 ]]; then
  echo "[codec-path] Warning: some codec build directories are missing:" >&2
  printf '  %s\n' "${missing_dirs[@]}" >&2
fi