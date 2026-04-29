#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# VHAP deterministic installer for Python 3.11 + PyTorch 2.9.1 + CUDA 12.8.
#
# Usage:
#   bash setup.sh              # create conda env VHAP, install dependencies
#   bash setup.sh --pip-only   # install into the currently active Python
#   bash setup.sh --no-assets  # skip download_assets.sh
#
# Preconditions:
# - CUDA 12.8 toolkit is available. System CUDA at /usr/local/cuda-12.8 is
#   preferred, matching the companion HRAvatar installer.
# - A C++ compiler compatible with CUDA 12.8 is available for nvdiffrast/dlib.
# -----------------------------------------------------------------------------

set -eo pipefail

PIP_ONLY=0
NO_ASSETS=0

for arg in "$@"; do
  case "$arg" in
    --pip-only) PIP_ONLY=1 ;;
    --no-assets) NO_ASSETS=1 ;;
    -h|--help)
      awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
      exit 0
      ;;
    *) echo "[setup.sh] unknown flag: $arg" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="${ENV_NAME:-VHAP}"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
NUMPY_VERSION="2.2.6"
PROTOBUF_VERSION="4.25.5"

pip_no_deps() {
  python -m pip install --no-deps "$@"
}

if [[ ${PIP_ONLY} -eq 0 ]]; then
  echo "[1/5] Creating conda env ${ENV_NAME} (Python ${PYTHON_VERSION})"

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo " -> conda env '${ENV_NAME}' already exists, skipping creation."
  else
    conda create --name "${ENV_NAME}" -y "python=${PYTHON_VERSION}"
  fi

  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
else
  echo "[1/5] Using currently active Python (pip-only mode)"
fi

if [[ "${FORCE_CONDA_CUDA:-0}" == "1" ]]; then
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[setup.sh] FORCE_CONDA_CUDA=1 requires an active conda env." >&2
    exit 1
  fi
  conda install -y -c "nvidia/label/cuda-12.8.1" cuda-toolkit=12.8.1
  if [[ ! -e "${CONDA_PREFIX}/lib64" ]]; then
    ln -s "${CONDA_PREFIX}/lib" "${CONDA_PREFIX}/lib64"
  fi
  export CUDA_HOME="${CONDA_PREFIX}"
elif [[ -z "${CUDA_HOME:-}" || "${CUDA_HOME:-}" == "${CONDA_PREFIX:-__no_conda_prefix__}" ]]; then
  if [[ -d /usr/local/cuda-12.8 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  else
    echo "[setup.sh] CUDA_HOME is not set and no system CUDA path was found." >&2
    echo "[setup.sh] Set CUDA_HOME to CUDA 12.8, or rerun with FORCE_CONDA_CUDA=1." >&2
    exit 1
  fi
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FORCE_CUDA=1

if [[ ${PIP_ONLY} -eq 0 ]]; then
  conda env config vars set CUDA_HOME="${CUDA_HOME}"
  conda env config vars set TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
fi

echo "[setup.sh] CUDA_HOME=${CUDA_HOME}"
echo "[setup.sh] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
nvcc --version || { echo "[setup.sh] nvcc not found on PATH" >&2; exit 1; }

echo "[2/5] Installing pinned core packages"
python -m pip install --upgrade pip==25.2
pip_no_deps editables==0.6

# Install the immovable pins first. Keeping these explicit prevents later
# dependency resolution from drifting the CUDA/numpy/chumpy/protobuf stack.
pip_no_deps "numpy==${NUMPY_VERSION}"
pip_no_deps "protobuf==${PROTOBUF_VERSION}"
# chumpy: install from mattloper master, pinned to a specific SHA where
# chumpy/version.py reports '0.71'. The PyPI release of mattloper/chumpy
# (0.70, 2020) is broken under numpy 2.x; the master branch carries the
# numpy 2 + Python 3.12 fixes. The companion GaussianAvatars cuda128
# branch pins the same SHA, so both stacks agree on chumpy 0.71 and
# never contend over site-packages/chumpy/__init__.py.
#
# Aug-2025 maintenance commit: "ci: drop Python 2 checks; migrate CI to
# CircleCI 2.1 + Python 3.12".
CHUMPY_SHA="580566eafc9ac68b2614b64d6f7aaa84eebb70da"
pip_no_deps "git+https://github.com/mattloper/chumpy.git@${CHUMPY_SHA}"

# PyTorch 2.9.1 cu128 runtime stack. These pins mirror the CUDA 12.8 wheel
# dependency set used by the companion HRAvatar installer.
pip_no_deps nvidia-cublas-cu12==12.8.4.1
pip_no_deps nvidia-cuda-cupti-cu12==12.8.90
pip_no_deps nvidia-cuda-nvrtc-cu12==12.8.93
pip_no_deps nvidia-cuda-runtime-cu12==12.8.90
pip_no_deps nvidia-cudnn-cu12==9.10.2.21
pip_no_deps nvidia-cufft-cu12==11.3.3.83
pip_no_deps nvidia-cufile-cu12==1.13.1.3
pip_no_deps nvidia-curand-cu12==10.3.9.90
pip_no_deps nvidia-cusolver-cu12==11.7.3.90
pip_no_deps nvidia-cusparse-cu12==12.5.8.93
pip_no_deps nvidia-cusparselt-cu12==0.7.1
pip_no_deps nvidia-nccl-cu12==2.27.5
pip_no_deps nvidia-nvjitlink-cu12==12.8.93
pip_no_deps nvidia-nvshmem-cu12==3.3.20
pip_no_deps nvidia-nvtx-cu12==12.8.90
pip_no_deps triton==3.5.1

pip_no_deps "torch==${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
pip_no_deps "torchvision==${TORCHVISION_VERSION}" --index-url https://download.pytorch.org/whl/cu128

# Runtime packages whose resolver choices commonly affect the fixed stack.
pip_no_deps tensorboard==2.20.0
pip_no_deps tensorboard-data-server==0.7.2
pip_no_deps grpcio==1.80.0
pip_no_deps markdown==3.10.2
pip_no_deps werkzeug==3.1.8
pip_no_deps absl-py==2.3.1

echo "[3/5] Installing pinned VHAP Python dependencies"

# Direct VHAP deps plus their runtime deps. Everything is installed with
# --no-deps so pip cannot silently upgrade/downgrade the fixed stack.
pip_no_deps pyyaml==6.0.3
pip_no_deps pillow==12.0.0
pip_no_deps scipy==1.16.3
pip_no_deps opencv-python==4.12.0.88
pip_no_deps ffmpeg-python==0.2.0
pip_no_deps future==1.0.0
pip_no_deps colour-science==0.4.6
pip_no_deps trimesh==4.8.3

# dlib builds from source on many Linux/Python combinations and needs cmake
# on PATH before its setup.py runs.
pip_no_deps cmake==4.1.2
pip_no_deps ninja==1.13.0
hash -r
cmake --version

pip_no_deps dlib==19.24.6
pip_no_deps python-gflags==3.1.2
pip_no_deps imgaug==0.4.0
pip_no_deps lmdb==1.7.5
pip_no_deps lxml==6.0.2
pip_no_deps shapely==2.1.2
pip_no_deps pandas==2.3.3
pip_no_deps gdown==5.2.0
pip_no_deps face-alignment==1.4.1
pip_no_deps dearpygui==1.11.1
pip_no_deps joblib==1.5.2

# tyro / CLI stack.
pip_no_deps tyro==0.8.14
pip_no_deps docstring-parser==0.16
pip_no_deps rich==13.9.4
pip_no_deps shtab==1.7.2
pip_no_deps typing_extensions==4.15.0
pip_no_deps markdown-it-py==3.0.0
pip_no_deps mdurl==0.1.2
pip_no_deps pygments==2.18.0

# matplotlib / image / scipy ecosystem.
pip_no_deps matplotlib==3.10.7
pip_no_deps contourpy==1.3.3
pip_no_deps cycler==0.12.1
pip_no_deps fonttools==4.60.1
pip_no_deps kiwisolver==1.4.9
pip_no_deps packaging==25.0
pip_no_deps pyparsing==3.2.5
pip_no_deps python-dateutil==2.9.0.post0
pip_no_deps six==1.17.0
pip_no_deps imageio==2.37.2
pip_no_deps tifffile==2025.10.16
pip_no_deps lazy_loader==0.4

# torch / torchvision support deps.
pip_no_deps filelock==3.20.0
pip_no_deps fsspec==2025.10.0
pip_no_deps jinja2==3.1.6
pip_no_deps markupsafe==3.0.3
pip_no_deps networkx==3.5
pip_no_deps sympy==1.14.0
pip_no_deps mpmath==1.3.0

# face-alignment / STAR support deps.
pip_no_deps tqdm==4.67.1
pip_no_deps scikit-image==0.25.2
pip_no_deps numba==0.62.1
pip_no_deps llvmlite==0.45.1
pip_no_deps pytz==2025.2
pip_no_deps tzdata==2025.2
pip_no_deps requests==2.32.3
pip_no_deps certifi==2024.8.30
pip_no_deps charset-normalizer==3.4.0
pip_no_deps idna==3.10
pip_no_deps urllib3==2.2.3
pip_no_deps beautifulsoup4==4.12.3
pip_no_deps soupsieve==2.6

# Build backend for editable install with --no-build-isolation.
pip_no_deps hatchling==1.27.0
pip_no_deps pathspec==0.12.1
pip_no_deps pluggy==1.5.0
pip_no_deps trove-classifiers==2025.11.14.15

# Git/direct dependencies used by VHAP. These track environment-specific
# branches in maintained forks instead of immutable commit SHAs.
pip_no_deps "nvdiffrast@git+https://github.com/MTamon/nvdiffrast@cuda128-backface-culling"
pip_no_deps "BackgroundMattingV2@git+https://github.com/MTamon/BackgroundMattingV2@cuda128"
pip_no_deps "STAR@git+https://github.com/MTamon/STAR@cuda128"

echo "[4/5] Installing VHAP in editable mode"
python -m pip install --no-build-isolation --no-deps -e .

echo "[5/5] Sanity check"
python - <<'PY'
import numpy
import torch
import torchvision
import google.protobuf
import chumpy

print("python ok")
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("protobuf", google.protobuf.__version__)
print("chumpy", chumpy.__version__)

assert numpy.__version__ == "2.2.6"
assert torch.__version__.split("+", 1)[0] == "2.9.1"
assert torchvision.__version__.split("+", 1)[0] == "0.24.1"
assert google.protobuf.__version__ == "4.25.5"
assert chumpy.__version__ == "0.71", \
    f"chumpy version drift: got {chumpy.__version__}, expected 0.71 (check CHUMPY_SHA)"
assert hasattr(chumpy, "Ch"), \
    "chumpy.Ch missing - FLAME pickle deserialisation will fail"
PY

echo "[5/5] pip check (informational)"
python -m pip check || true

if [[ ${NO_ASSETS} -eq 0 ]]; then
  echo "[opt] Downloading/checking VHAP assets"
  bash download_assets.sh || echo "[setup.sh] download_assets.sh did not complete; rerun it manually after resolving the message above."
fi

echo "Done."
