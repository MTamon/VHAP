# Task Scripts

## `check_pinned_deps.py`

This script smoke-checks the three pinned direct Git dependencies in the ways
VHAP actually uses them:

- `nvdiffrast`: imports the package and renders a tiny triangle through
  `vhap.util.render_nvdiffrast.NVDiffRenderer`, including a backward pass.
- `BackgroundMattingV2`: runs `MattingRefine` on synthetic inputs and performs a
  TorchScript save/load round-trip.
- `STAR`: imports STAR/dlib and runs the STAR network forward path on synthetic
  input. By default it does not download STAR assets.

Run all checks:

```bash
python task/check_pinned_deps.py
```

Run individual checks:

```bash
python task/check_pinned_deps.py --check nvdiffrast
python task/check_pinned_deps.py --check background-matting-v2 --device cuda
python task/check_pinned_deps.py --check star --device cuda
```

To also exercise VHAP's asset-backed STAR detector wrapper, which may download
the dlib predictor and STAR checkpoint:

```bash
python task/check_pinned_deps.py --check star --device cuda --star-assets
```

Exit code `0` means the selected smoke checks passed. Exit code `1` means a
dependency imported but failed the VHAP-relevant behavior check. Exit code `2`
means a prerequisite is missing, such as CUDA for `nvdiffrast` or an uninstalled
direct dependency.

For `BackgroundMattingV2`, this script intentionally distinguishes
`torch.jit.load` from `torch.load`: `torch.load` has `weights_only`, while
`torch.jit.load` loads a TorchScript `ScriptModule` and does not expose
`weights_only`. The check therefore verifies TorchScript compatibility
separately from eager `state_dict` loading.

## `compare_pytorch3d_laplacian.py` -> Verified!

This script verifies whether the local replacement in `vhap.util.mesh` is equivalent to the original PyTorch3D path used by `vhap.model.flame`.

It compares:

- `pytorch3d.io.load_obj(...)` vs `vhap.util.mesh.load_obj_mesh(...)`
- `Meshes(...).laplacian_packed().to_dense()` vs `vhap.util.mesh.uniform_laplacian(...)`

### What This Test Means

This is not a comparison of random initialization, probability distributions, or acceptable visual similarity.

For the fixed file `asset/flame/head_template_mesh.obj`, the OBJ parser and uniform Laplacian construction are deterministic. The expected result is exact structural equivalence:

- same vertex tensor shape
- same face index tensor shape
- same UV tensor shape
- same texture face index tensor shape
- same vertex values
- same face indices
- same UV values
- same texture face indices
- same uniform Laplacian matrix

In other words, this is closer to checking a fixed numeric table than checking a stochastic process.

### Good Result

Run:

```bash
python task/compare_pytorch3d_laplacian.py
```

A good result is:

- all printed checks are `OK`
- `laplacian max_abs_diff` is `0`
- `laplacian entries with diff > atol(0.0)` is `0`
- process exit code is `0`

This means the local replacement should be behaviorally equivalent to the PyTorch3D path for this OBJ and this Laplacian use.

### Bad Result

A bad result is:

- any printed check is `NG`
- `laplacian max_abs_diff` is nonzero
- `laplacian entries with diff > atol(0.0)` is nonzero
- process exit code is `1`

This means the local replacement is not a proven drop-in replacement. In that case, do not assume tracking behavior is unchanged. Either fix the local implementation to match PyTorch3D exactly, or keep the PyTorch3D implementation in the pinned target environment.

### Missing Prerequisites

Exit code `2` means the comparison could not run. Typical causes:

- PyTorch3D is not installed in the active environment.
- `asset/flame/head_template_mesh.obj` is missing.

CUDA hardware is not required for this comparison. The test can run on CPU.

### Optional Tolerance

The default tolerance is strict:

```bash
python task/compare_pytorch3d_laplacian.py --atol 0.0
```

Because the expected output is deterministic and mostly index-derived, strict equality is the right default. A nonzero tolerance should only be used for investigation, not as proof of equivalence.
