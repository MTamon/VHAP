# ----------------------------------------------------------------------------
# 3. pytorch3d v0.7.8 (source build against torch 2.9.1 + CUDA 12.8).
# ----------------------------------------------------------------------------
# HRAvatar imports `pytorch3d.ops` in scene/gaussian_head_model.py and
# `pytorch3d` in utils/loss_utils.py.python -m pip install --no-deps No wheel exists for cu128 yet.
# NOTE: this step compiles a large CUDA extension and can take 10+ minutes.
# Shallow clone + --branch doesn't reliably resolve tags, so we use
# git-init + fetch-by-tag which works regardless of server advertisement.
python -m pip install --no-deps iopath==0.1.10
python -m pip install --no-deps portalocker==3.2.0

PYTORCH3D_TMP="$(mktemp -d)"
git -C "${PYTORCH3D_TMP}" init -q
git -C "${PYTORCH3D_TMP}" remote add origin https://github.com/facebookresearch/pytorch3d.git
git -C "${PYTORCH3D_TMP}" fetch --depth 1 origin tag V0.7.8
git -C "${PYTORCH3D_TMP}" checkout FETCH_HEAD
python -m pip install --no-deps "${PYTORCH3D_TMP}"
rm -rf "${PYTORCH3D_TMP}"