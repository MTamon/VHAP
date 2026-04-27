#!/usr/bin/env python
"""Smoke-check VHAP's pinned direct dependencies in the ways VHAP uses them.

This is not a full upstream test suite. It deliberately checks only the small
surface that VHAP imports or calls:

- nvdiffrast: CUDA rasterize/interpolate/texture/antialias through
  vhap.util.render_nvdiffrast.NVDiffRenderer.
- BackgroundMattingV2: MattingRefine forward pass, plus TorchScript
  save/load round-trip because the package contains TorchScript export/load
  paths even though VHAP uses eager state_dict inference.
- STAR: import-time compatibility, dlib availability, and STAR network forward.
  The default does not download STAR assets or import VHAP's detector wrapper,
  because that wrapper imports star.asset at module import time. Use
  --star-assets for the full VHAP wrapper path.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class CheckError(RuntimeError):
    """A dependency is installed, but the VHAP-relevant behavior failed."""


class MissingPrerequisite(RuntimeError):
    """The active environment cannot run the requested check."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check VHAP's pinned nvdiffrast, BackgroundMattingV2, and STAR dependencies."
    )
    parser.add_argument(
        "--check",
        choices=("all", "nvdiffrast", "background-matting-v2", "star"),
        default="all",
        help="Which dependency check to run. Default: all.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for checks that can run on CPU. nvdiffrast always requires CUDA. Default: auto.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Synthetic image size for matting/rendering checks. Must be divisible by 4. Default: 64.",
    )
    parser.add_argument(
        "--star-assets",
        action="store_true",
        help="Also import star.asset and instantiate LandmarkDetectorSTAR. This may download assets.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def require_import(module_name: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as exc:  # noqa: BLE001 - import failures are the result here.
        raise MissingPrerequisite(f"cannot import {module_name}: {exc}") from exc


def print_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor_detached = tensor.detach()
    print(
        f"{name}: shape={tuple(tensor_detached.shape)} "
        f"dtype={tensor_detached.dtype} device={tensor_detached.device} "
        f"finite={bool(torch.isfinite(tensor_detached).all().item())}"
    )


def find_cuda_home() -> Path | None:
    candidates = []
    for value in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if value:
            candidates.append(Path(value))

    try:
        from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME
    except Exception:  # noqa: BLE001 - best-effort diagnostics only.
        TORCH_CUDA_HOME = None
    if TORCH_CUDA_HOME:
        candidates.append(Path(TORCH_CUDA_HOME))

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parents[1])

    candidates.extend([Path("/usr/local/cuda-12.8"), Path("/usr/local/cuda")])

    for candidate in candidates:
        if (candidate / "include" / "cuda_runtime.h").exists():
            return candidate
    return None


def require_cuda_toolkit_headers() -> None:
    cuda_home = find_cuda_home()
    if cuda_home is None:
        raise MissingPrerequisite(
            "CUDA Toolkit headers were not found. nvdiffrast builds a C++/CUDA "
            "extension and needs cuda_runtime.h. Set CUDA_HOME to a CUDA Toolkit "
            "install such as /usr/local/cuda-12.8, install conda cuda-toolkit, "
            "or rerun setup.sh with FORCE_CONDA_CUDA=1 and reactivate the env."
        )

    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ["PATH"] = f"{cuda_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    print(f"CUDA_HOME for extension build: {cuda_home}")


def check_nvdiffrast(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise MissingPrerequisite("nvdiffrast check requires torch.cuda.is_available()")

    require_cuda_toolkit_headers()

    sys.path.insert(0, str(REPO_ROOT))
    require_import("nvdiffrast.torch")

    from vhap.util.render_nvdiffrast import NVDiffRenderer

    device = torch.device("cuda")
    image_size = (args.image_size, args.image_size)

    try:
        renderer = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera").to(device)
    except ModuleNotFoundError as exc:
        if exc.name == "nvdiffrast_plugin":
            raise CheckError(
                "nvdiffrast built its PyTorch extension but failed to import it as "
                "'nvdiffrast_plugin'. This is a compatibility issue in the pinned "
                "nvdiffrast fork: nvdiffrast/torch/ops.py ignores the module returned "
                "by torch.utils.cpp_extension.load(...) and then calls "
                "importlib.import_module(...). Patch the fork to cache and return the "
                "load(...) result directly."
            ) from exc
        raise

    verts = torch.tensor(
        [[[-0.55, -0.45, -2.0], [0.55, -0.45, -2.0], [0.0, 0.55, -2.0]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    verts_uv = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], dtype=torch.float32, device=device)
    faces_uv = faces.clone()
    tex = torch.linspace(0.0, 1.0, steps=3 * 8 * 8, dtype=torch.float32, device=device).reshape(1, 3, 8, 8)
    rt = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    focal = args.image_size / 2
    k = torch.tensor(
        [[[focal, 0.0, args.image_size / 2], [0.0, focal, args.image_size / 2], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )

    out = renderer.render_rgba_vis(
        verts=verts,
        faces=faces,
        RT=rt,
        K=k,
        image_size=image_size,
        background_color=[1.0, 1.0, 1.0],
        verts_uv=verts_uv,
        faces_uv=faces_uv,
        tex=tex,
    )

    rgba = out["rgba"]
    print_tensor("nvdiffrast rgba", rgba)
    require(rgba.shape == (1, args.image_size, args.image_size, 4), "unexpected RGBA shape")
    require(bool(torch.isfinite(rgba).all().item()), "RGBA contains non-finite values")
    alpha_sum = float(rgba[..., 3].sum().item())
    print(f"nvdiffrast alpha_sum: {alpha_sum:.6f}")
    require(alpha_sum > 0.0, "rasterized triangle produced no foreground pixels")

    loss = rgba[..., :3].sum()
    loss.backward()
    require(verts.grad is not None, "renderer backward did not produce vertex gradients")
    require(bool(torch.isfinite(verts.grad).all().item()), "vertex gradients contain non-finite values")
    print_tensor("nvdiffrast verts.grad", verts.grad)


def check_background_matting_v2(args: argparse.Namespace) -> None:
    require(args.image_size % 4 == 0, "--image-size must be divisible by 4")

    require_import("BackgroundMattingV2.model")
    from BackgroundMattingV2.model import MattingRefine

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise MissingPrerequisite("--device cuda requested but CUDA is unavailable")

    model = MattingRefine(
        "mobilenetv2",
        backbone_scale=0.25,
        refine_mode="thresholding",
        refine_sample_pixels=80_000,
        refine_threshold=0.01,
        refine_kernel_size=3,
    ).to(device).eval()

    src = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    bgr = torch.rand_like(src)

    with torch.no_grad():
        eager = model(src, bgr)

    require(isinstance(eager, tuple) and len(eager) == 6, "MattingRefine did not return the expected 6-tuple")
    pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm = eager
    for name, tensor in (
        ("pha", pha),
        ("fgr", fgr),
        ("pha_sm", pha_sm),
        ("fgr_sm", fgr_sm),
        ("err_sm", err_sm),
        ("ref_sm", ref_sm),
    ):
        print_tensor(f"BackgroundMattingV2 {name}", tensor)
        require(bool(torch.isfinite(tensor).all().item()), f"{name} contains non-finite values")

    require(pha.shape == (1, 1, args.image_size, args.image_size), "unexpected pha shape")
    require(fgr.shape == (1, 3, args.image_size, args.image_size), "unexpected fgr shape")

    jit_sig = inspect.signature(torch.jit.load)
    load_sig = inspect.signature(torch.load)
    print(f"torch.jit.load signature: {jit_sig}")
    print(f"torch.load signature: {load_sig}")
    require("weights_only" not in jit_sig.parameters, "unexpected torch.jit.load weights_only parameter")
    require("weights_only" in load_sig.parameters, "torch.load has no weights_only parameter in this PyTorch")

    # The upstream package has TorchScript export/load utilities. VHAP does not
    # use them for matting, but this catches TorchScript serialization regressions
    # separately from torch.load(..., weights_only=True) state_dict loading.
    with tempfile.TemporaryDirectory(prefix="vhap-bmv2-jit-") as tmp_dir:
        jit_path = Path(tmp_dir) / "matting_refine_trace.pt"
        traced = torch.jit.trace(model, (src, bgr), strict=False)
        torch.jit.save(traced, str(jit_path))
        loaded = torch.jit.load(str(jit_path), map_location=device)
        loaded.eval()
        with torch.no_grad():
            scripted = loaded(src, bgr)

    max_diff = max(float((a - b).abs().max().item()) for a, b in zip(eager, scripted))
    print(f"BackgroundMattingV2 TorchScript round-trip max_abs_diff: {max_diff:.9g}")
    require(max_diff <= 1e-5, "TorchScript round-trip changed MattingRefine outputs")


def check_star(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(REPO_ROOT))

    require_import("dlib")
    require_import("star.lib.utility")

    import argparse as argparse_module
    from star.lib import utility

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise MissingPrerequisite("--device cuda requested but CUDA is unavailable")

    config = utility.get_config(argparse_module.Namespace(config_name="alignment"))
    config.device_id = torch.cuda.current_device() if device.type == "cuda" else -1
    config.init_instance = lambda: None
    config.logger = None
    utility.set_environment(config)

    net = utility.get_net(config).to(config.device).eval()
    x = torch.rand(1, 3, config.height, config.width, device=config.device)
    with torch.no_grad():
        output, fusionmaps, landmarks = net(x)

    require(isinstance(output, list) and len(output) > 0, "STAR output list is empty")
    require(isinstance(fusionmaps, list) and len(fusionmaps) == config.nstack, "unexpected STAR fusionmaps")
    print_tensor("STAR landmarks", landmarks)
    require(landmarks.shape[-2:] == (68, 2), "unexpected STAR landmark shape")
    require(bool(torch.isfinite(landmarks).all().item()), "STAR landmarks contain non-finite values")

    if args.star_assets:
        import cv2
        import numpy as np
        from vhap.util.landmark_detector_star import LandmarkDetectorSTAR

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.circle(image, (128, 128), 32, (255, 255, 255), -1)
        detector = LandmarkDetectorSTAR()
        bbox, lmks = detector.detect_single_image(image)
        print(f"STAR asset-backed bbox shape: {bbox.shape}")
        print(f"STAR asset-backed landmarks shape: {lmks.shape}")
        require(bbox.shape == (5,), "asset-backed STAR bbox shape mismatch")
        require(lmks.shape == (68, 3), "asset-backed STAR landmarks shape mismatch")


def run_check(name: str, fn: Callable[[argparse.Namespace], None], args: argparse.Namespace) -> int:
    print(f"\n== {name} ==", flush=True)
    try:
        fn(args)
    except MissingPrerequisite as exc:
        print(f"{name}: MISSING PREREQUISITE: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - top-level check runner.
        print(f"{name}: FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"{name}: OK")
    return 0


def main() -> int:
    # Keep extension build/cache output inside the active environment unless the
    # caller intentionally overrides it.
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache" / "torch_extensions"))

    args = parse_args()
    checks: list[tuple[str, Callable[[argparse.Namespace], None]]] = []

    if args.check in ("all", "nvdiffrast"):
        checks.append(("nvdiffrast", check_nvdiffrast))
    if args.check in ("all", "background-matting-v2"):
        checks.append(("BackgroundMattingV2", check_background_matting_v2))
    if args.check in ("all", "star"):
        checks.append(("STAR", check_star))

    exit_code = 0
    for name, fn in checks:
        result = run_check(name, fn, args)
        if result == 1:
            exit_code = 1
        elif result == 2 and exit_code == 0:
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
