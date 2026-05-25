#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-$HOME/codec_toolchains}"
JOBS="${JOBS:-$(nproc)}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"

bootstrap_compilers() {
  local test_dir
  test_dir=$(mktemp -d)
  cat > "$test_dir/test.c" <<'EOF'
int main(void) { return 0; }
EOF

  if command -v gcc >/dev/null 2>&1 && gcc "$test_dir/test.c" -o "$test_dir/test" >/dev/null 2>&1; then
    echo "[setup] Using system compiler from $(command -v gcc)"
    rm -rf "$test_dir"
    return 0
  fi

  echo "[setup] System compiler is unavailable or broken; installing conda-forge compiler toolchain"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"
  mkdir -p "$CONDA_PKGS_DIRS"

  if command -v mamba >/dev/null 2>&1; then
    mamba install -y -c conda-forge compilers sysroot_linux-64 binutils_linux-64
  elif command -v conda >/dev/null 2>&1; then
    conda install -y -c conda-forge compilers sysroot_linux-64 binutils_linux-64
  else
    echo "No conda-compatible package manager available to install compilers" >&2
    rm -rf "$test_dir"
    exit 1
  fi

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" ]]; then
      export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    fi
    if [[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" ]]; then
      export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
    fi
  fi

  if [[ -z "${CC:-}" ]] || [[ -z "${CXX:-}" ]]; then
    echo "Conda compiler installation did not expose CC/CXX as expected" >&2
    rm -rf "$test_dir"
    exit 1
  fi

  echo "[setup] Using conda compiler CC=$CC"
  echo "[setup] Using conda compiler CXX=$CXX"

  "$CC" "$test_dir/test.c" -o "$test_dir/test" >/dev/null 2>&1 || {
    echo "Conda compiler test failed even after installation" >&2
    rm -rf "$test_dir"
    exit 1
  }
  rm -rf "$test_dir"
}

bootstrap_modules() {
  if type module >/dev/null 2>&1; then
    echo "[setup] Loading cluster toolchain modules"
    module load gnu9/9.4.0 cmake/3.21.3 >/dev/null 2>&1 || true
  fi
}

bootstrap_cmake() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/bin" ]]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
  fi

  if command -v cmake >/dev/null 2>&1; then
    echo "[setup] Using cmake from $(command -v cmake)"
    return 0
  fi

  for cmake_bin in \
    /opt/ohpc/pub/utils/cmake/3.21.3/bin/cmake \
    /usr/bin/cmake \
    "$HOME/.local/bin/cmake"
  do
    if [[ -x "$cmake_bin" ]]; then
      export PATH="$(dirname "$cmake_bin"):$PATH"
      echo "[setup] Using existing cmake from $cmake_bin"
      return 0
    fi
  done

  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"
  mkdir -p "$CONDA_PKGS_DIRS"

  if command -v mamba >/dev/null 2>&1; then
    echo "[setup] Installing cmake and ninja with mamba"
    mamba install -y -c conda-forge cmake ninja
  elif command -v conda >/dev/null 2>&1; then
    echo "[setup] Installing cmake and ninja with conda"
    conda install -y -c conda-forge cmake ninja
  elif command -v python >/dev/null 2>&1; then
    echo "[setup] Installing cmake and ninja with pip"
    python -m pip install cmake ninja
  fi

  command -v cmake >/dev/null 2>&1 || {
    echo "cmake is still unavailable after bootstrap attempts" >&2
    exit 1
  }

  if command -v ninja >/dev/null 2>&1; then
    echo "[setup] Using ninja from $(command -v ninja)"
  fi
}

bootstrap_modules
bootstrap_cmake
bootstrap_compilers

if ! command -v ninja >/dev/null 2>&1 && [[ "$GENERATOR" == "Ninja" ]]; then
  echo "Ninja generator requested but ninja is unavailable" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"
echo "[setup] Build root: $ROOT_DIR"
echo "[setup] Build jobs: $JOBS"
echo "[setup] CMake generator: $GENERATOR"

CODECS_TO_BUILD="${CODECS_TO_BUILD:-vvenc,vvdec,uavs3e,uavs3d}"

build_repo() {
    local repo_url="$1"
    local repo_dir="$2"
    shift 2
  local extra_cmake_args=()
    if [[ ! -d "$repo_dir" ]]; then
    echo "[clone] $repo_url -> $repo_dir"
        git clone "$repo_url" "$repo_dir"
  else
    echo "[clone] Reusing existing checkout: $repo_dir"
    fi
    if [[ "$repo_dir" == "uavs3e" ]]; then
      patch_uavs3e "$repo_dir"
    fi
    if [[ "$repo_dir" == "uavs3e" || "$repo_dir" == "uavs3d" ]]; then
      extra_cmake_args+=(
        -DCMAKE_C_FLAGS=-include\ unistd.h\ -D_LARGEFILE64_SOURCE
        -DCMAKE_CXX_FLAGS=-include\ unistd.h\ -D_LARGEFILE64_SOURCE
      )
    fi
  echo "[cmake] Configuring $repo_dir"
    cmake -S "$repo_dir" -B "$repo_dir/build" -G "$GENERATOR" \
      ${CC:+-DCMAKE_C_COMPILER="$CC"} \
      ${CXX:+-DCMAKE_CXX_COMPILER="$CXX"} \
      "${extra_cmake_args[@]}" \
      "$@"
  echo "[build] Building $repo_dir"
    cmake --build "$repo_dir/build" -j "$JOBS"
}

patch_uavs3e() {
  local repo_dir="$1"
  local avx_file="$repo_dir/src/avx2/itdq_avx2.c"
  if [[ -f "$avx_file" ]]; then
    echo "[patch] Applying GCC compatibility fix to $avx_file"
    sed -i 's/(s64\*)block/(long long*)block/g' "$avx_file"
  fi
}

IFS=',' read -r -a codec_targets <<< "$CODECS_TO_BUILD"

for codec_target in "${codec_targets[@]}"; do
  case "$codec_target" in
    vvenc)
      build_repo https://github.com/fraunhoferhhi/vvenc vvenc -DCMAKE_BUILD_TYPE=Release
      ;;
    vvdec)
      build_repo https://github.com/fraunhoferhhi/vvdec vvdec -DCMAKE_BUILD_TYPE=Release
      ;;
    uavs3e)
      build_repo https://github.com/uavs3/uavs3e uavs3e -DCMAKE_BUILD_TYPE=Release -DCOMPILE_10BIT=1
      ;;
    uavs3d)
      build_repo https://github.com/uavs3/uavs3d uavs3d -DCMAKE_BUILD_TYPE=Release -DCOMPILE_10BIT=1
      ;;
    vtm)
      build_repo https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git VVCSoftware_VTM -DCMAKE_BUILD_TYPE=Release
      ;;
    '')
      ;;
    *)
      echo "Unknown codec target '$codec_target'. Supported values: vvenc,vvdec,uavs3e,uavs3d,vtm" >&2
      exit 1
      ;;
  esac
done

cat <<EOF
Build complete.
Add these directories to PATH before running benchmarks:
  $ROOT_DIR/vvenc/bin/release-static
  $ROOT_DIR/vvdec/bin/release-static
  $ROOT_DIR/uavs3e/build
  $ROOT_DIR/uavs3d/build
  $ROOT_DIR/VVCSoftware_VTM/bin

If you run this script manually on the cluster, prefer:
  bash scripts/install_vvc_avs3_codecs.sh

Optional targets:
  CODECS_TO_BUILD=vvenc,vvdec,uavs3e,uavs3d bash scripts/install_vvc_avs3_codecs.sh
  CODECS_TO_BUILD=vvenc,uavs3e bash scripts/install_vvc_avs3_codecs.sh
  CODECS_TO_BUILD=vtm bash scripts/install_vvc_avs3_codecs.sh
EOF