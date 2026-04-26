#!/usr/bin/env python
"""Compare local FLAME OBJ/Laplacian code against the original PyTorch3D path.

This is a deterministic equivalence check, not a statistical/randomness check.
For the fixed FLAME OBJ, PyTorch3D and the local replacement should produce the
same vertex indices, UV indices, and uniform graph Laplacian. Any mismatch means
the replacement may alter optimization regularization behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBJ = REPO_ROOT / "asset" / "flame" / "head_template_mesh.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vhap.util.mesh.load_obj_mesh/uniform_laplacian with "
            "pytorch3d.io.load_obj and Meshes(...).laplacian_packed()."
        )
    )
    parser.add_argument(
        "--obj",
        type=Path,
        default=DEFAULT_OBJ,
        help=f"OBJ file to compare. Default: {DEFAULT_OBJ}",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for Laplacian comparison. Default: 0.0",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    obj_path = args.obj.resolve()

    if not obj_path.exists():
        print(f"ERROR: OBJ file not found: {obj_path}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(REPO_ROOT))

    try:
        from pytorch3d.io import load_obj
        from pytorch3d.structures import Meshes
    except ImportError as exc:
        print(
            "ERROR: PyTorch3D is required for this comparison but is not installed "
            "in the active Python environment.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    from vhap.util.mesh import load_obj_mesh, uniform_laplacian

    verts_old, faces_old, aux_old = load_obj(str(obj_path), load_textures=False)
    laplacian_old = Meshes(
        verts=[verts_old],
        faces=[faces_old.verts_idx],
    ).laplacian_packed().to_dense()

    verts_new, faces_new, aux_new = load_obj_mesh(obj_path)
    laplacian_new = uniform_laplacian(
        verts_new.shape[0],
        faces_new.verts_idx,
        dtype=laplacian_old.dtype,
        device=laplacian_old.device,
    )

    # Good result:
    #   all checks print OK, laplacian max_abs_diff is 0, and the process exits 0.
    # Bad result:
    #   any NG result, nonzero Laplacian diff, or exit code 1. That means the
    #   local implementation is not a drop-in replacement for this OBJ.
    # Missing prerequisite:
    #   exit code 2, usually because PyTorch3D or the OBJ file is unavailable.
    checks = {
        "verts_shape": verts_old.shape == verts_new.shape,
        "faces_shape": faces_old.verts_idx.shape == faces_new.verts_idx.shape,
        "uv_shape": aux_old.verts_uvs.shape == aux_new.verts_uvs.shape,
        "textures_shape": faces_old.textures_idx.shape == faces_new.textures_idx.shape,
        "verts_equal": torch.equal(verts_old.cpu(), verts_new.cpu()),
        "faces_equal": torch.equal(faces_old.verts_idx.cpu(), faces_new.verts_idx.cpu()),
        "uv_equal": torch.equal(aux_old.verts_uvs.cpu(), aux_new.verts_uvs.cpu()),
        "textures_equal": torch.equal(faces_old.textures_idx.cpu(), faces_new.textures_idx.cpu()),
    }

    diff = (laplacian_old - laplacian_new).abs()
    max_abs_diff = diff.max().item() if diff.numel() else 0.0
    nonzero_diff = int((diff > args.atol).sum().item())
    laplacian_equal = nonzero_diff == 0

    print(f"OBJ: {obj_path}")
    print(f"verts old/new: {tuple(verts_old.shape)} / {tuple(verts_new.shape)}")
    print(f"faces old/new: {tuple(faces_old.verts_idx.shape)} / {tuple(faces_new.verts_idx.shape)}")
    print(f"uvs old/new: {tuple(aux_old.verts_uvs.shape)} / {tuple(aux_new.verts_uvs.shape)}")
    print(f"texture faces old/new: {tuple(faces_old.textures_idx.shape)} / {tuple(faces_new.textures_idx.shape)}")
    print(f"laplacian shape old/new: {tuple(laplacian_old.shape)} / {tuple(laplacian_new.shape)}")
    print(f"laplacian max_abs_diff: {max_abs_diff:.12g}")
    print(f"laplacian entries with diff > atol({args.atol}): {nonzero_diff}")
    print()

    for name, ok in checks.items():
        print(f"{name}: {'OK' if ok else 'NG'}")
    print(f"laplacian_equal: {'OK' if laplacian_equal else 'NG'}")

    if not all(checks.values()) or not laplacian_equal:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
