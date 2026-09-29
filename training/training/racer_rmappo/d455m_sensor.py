"""Fixed-body D455M depth sensor helpers for the Isaac backend.

The module is safe to import in ordinary Python processes: Omniverse and
Replicator are imported only when :class:`FixedBodyD455MSensor` is created.
This keeps the geometry/preprocessing functions covered by the lightweight
unit-test suite.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _vector(mapping: Mapping[str, Any], key: str, length: int) -> list[float]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"camera.{key} must be a sequence of length {length}")
    result = [float(item) for item in value]
    if len(result) != length:
        raise ValueError(
            f"camera.{key} must contain {length} values, got {len(result)}"
        )
    return result


def validate_d455m_config(cfg: Mapping[str, Any]) -> None:
    """Validate the camera contract shared by the probe and future backend."""
    if not bool(cfg.get("fixed_to_body", False)):
        raise ValueError("camera.fixed_to_body must be true")
    width = int(cfg["width"])
    height = int(cfg["height"])
    if width < 2 or height < 2:
        raise ValueError("camera width and height must be at least two pixels")
    actor_size = [int(value) for value in cfg["actor_resize"]]
    if actor_size != [64, 40]:
        raise ValueError("the current actor contract requires camera.actor_resize=[64, 40]")

    minimum = float(cfg["min_depth_m"])
    maximum = float(cfg["max_depth_m"])
    if not 0.0 < minimum < maximum:
        raise ValueError("camera depth range must satisfy 0 < min_depth_m < max_depth_m")
    for key in ("fx", "fy"):
        if float(cfg[key]) <= 0.0:
            raise ValueError(f"camera.{key} must be positive")
    cx = float(cfg["cx"])
    cy = float(cfg["cy"])
    if not 0.0 <= cx < width or not 0.0 <= cy < height:
        raise ValueError("camera principal point must lie inside the image")

    mount = _vector(cfg, "mount_position_body_m", 3)
    forward = _vector(cfg, "forward_axis_body", 3)
    up = _vector(cfg, "up_axis_body", 3)
    forward_norm = math.sqrt(sum(value * value for value in forward))
    up_norm = math.sqrt(sum(value * value for value in up))
    dot = sum(a * b for a, b in zip(forward, up))
    if forward_norm < 1e-6 or up_norm < 1e-6:
        raise ValueError("camera forward/up axes must be non-zero")
    if abs(dot / (forward_norm * up_norm)) > 1e-4:
        raise ValueError("camera forward_axis_body and up_axis_body must be orthogonal")
    if any(not math.isfinite(value) for value in mount + forward + up):
        raise ValueError("camera mount and axes must be finite")
    if int(cfg.get("warmup_render_frames", 0)) < 1:
        raise ValueError("camera.warmup_render_frames must be at least one")


def depth_to_normalized_inverse(
    depth_m: Tensor,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[Tensor, Tensor]:
    """Convert metric depth to the exact inverse-depth contract used by smoke.

    Invalid, too-near and too-far pixels become zero. A valid measurement at
    ``min_depth_m`` maps to one, while one at ``max_depth_m`` maps to zero.
    The validity mask is returned separately because a far-range zero and an
    invalid zero otherwise have the same actor value.
    """
    if not 0.0 < float(min_depth_m) < float(max_depth_m):
        raise ValueError("depth range must satisfy 0 < min_depth_m < max_depth_m")
    depth = torch.as_tensor(depth_m, dtype=torch.float32)
    valid = (
        torch.isfinite(depth)
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    safe_depth = depth.clamp(float(min_depth_m), float(max_depth_m))
    inv_minimum = 1.0 / float(min_depth_m)
    inv_maximum = 1.0 / float(max_depth_m)
    normalized = (safe_depth.reciprocal() - inv_maximum) / (
        inv_minimum - inv_maximum
    )
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    return normalized.clamp_(0.0, 1.0), valid


def resize_inverse_depth_for_actor(
    normalized_inverse_depth: Tensor,
    output_width: int = 64,
    output_height: int = 40,
) -> Tensor:
    """Downsample inverse depth conservatively for obstacle avoidance.

    Adaptive max pooling keeps the closest valid return in every output cell
    because closer objects have larger normalized inverse depth. This avoids
    bilinear interpolation washing out thin trunks or wall edges.
    """
    if output_width < 1 or output_height < 1:
        raise ValueError("actor depth dimensions must be positive")
    depth = torch.as_tensor(normalized_inverse_depth, dtype=torch.float32)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
        restore = "hw"
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1)
        restore = "bhw"
    elif depth.ndim == 4:
        restore = "bchw"
    else:
        raise ValueError("inverse depth must have shape [H,W], [B,H,W], or [B,C,H,W]")
    if depth.shape[-2] < output_height or depth.shape[-1] < output_width:
        raise ValueError("actor preprocessing only supports downsampling")
    resized = F.adaptive_max_pool2d(depth, (output_height, output_width))
    if restore == "hw":
        return resized[0, 0]
    if restore == "bhw":
        return resized[:, 0]
    return resized


def summarize_depth(
    depth_m: Tensor,
    min_depth_m: float,
    max_depth_m: float,
    center_patch_px: int,
) -> Dict[str, Any]:
    """Return JSON-safe depth diagnostics used by the camera probe."""
    depth = torch.as_tensor(depth_m, dtype=torch.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"depth image must be two-dimensional, got {tuple(depth.shape)}")
    if center_patch_px < 1:
        raise ValueError("center_patch_px must be positive")
    patch = min(int(center_patch_px), int(depth.shape[0]), int(depth.shape[1]))
    if patch % 2 == 0:
        patch -= 1
    row = int(depth.shape[0]) // 2
    column = int(depth.shape[1]) // 2
    radius = patch // 2
    center = depth[row - radius : row + radius + 1, column - radius : column + radius + 1]
    raw_valid = torch.isfinite(depth) & (depth > 0.0)
    range_valid = (
        raw_valid
        & (depth >= float(min_depth_m))
        & (depth <= float(max_depth_m))
    )
    center_valid = (
        torch.isfinite(center)
        & (center >= float(min_depth_m))
        & (center <= float(max_depth_m))
    )
    center_values = center[center_valid]
    range_values = depth[range_valid]
    return {
        "shape": [int(depth.shape[0]), int(depth.shape[1])],
        "finite": bool(torch.isfinite(range_values).all().item()),
        "raw_valid_fraction": float(raw_valid.float().mean().item()),
        "range_valid_fraction": float(range_valid.float().mean().item()),
        "center_valid_fraction": float(center_valid.float().mean().item()),
        "center_median_m": (
            float(center_values.median().item()) if center_values.numel() else None
        ),
        "valid_min_m": float(range_values.min().item()) if range_values.numel() else None,
        "valid_max_m": float(range_values.max().item()) if range_values.numel() else None,
    }


class FixedBodyD455MSensor:
    """One Replicator depth camera rigidly parented to the drone base link."""

    DATA_TYPES = ("distance_to_image_plane", "distance_to_camera")

    def __init__(
        self,
        cfg: Mapping[str, Any],
        parent_prim_path: str,
        simulation_context: Any,
    ) -> None:
        validate_d455m_config(cfg)
        self.cfg = dict(cfg)
        self.sim = simulation_context
        self.prim_path = f"{parent_prim_path}/{cfg.get('prim_name', 'D455M')}"
        self.render_product_path: str | None = None
        self.annotators: Dict[str, Any] = {}
        self._closed = False

        # Runtime-only imports keep this module importable without Isaac Sim.
        from omni.isaac.core.utils.prims import create_prim, is_prim_path_valid
        from pxr import Gf, UsdGeom

        if is_prim_path_valid(self.prim_path):
            raise RuntimeError(f"duplicate D455M prim: {self.prim_path}")
        origin = Gf.Vec3d(0.0, 0.0, 0.0)
        forward = Gf.Vec3d(*_vector(cfg, "forward_axis_body", 3))
        up = Gf.Vec3d(*_vector(cfg, "up_axis_body", 3))
        camera_to_body = Gf.Matrix4d(1.0).SetLookAt(origin, forward, up).GetInverse()
        quaternion = camera_to_body.ExtractRotationQuat()
        # Isaac Sim 2023.1's bundled pxr exposes quaternion components as
        # properties (the same API used by the vendored OmniDrones camera).
        orientation_wxyz = (
            float(quaternion.real),
            *(float(value) for value in quaternion.imaginary),
        )
        prim = create_prim(
            self.prim_path,
            prim_type="Camera",
            translation=tuple(_vector(cfg, "mount_position_body_m", 3)),
            orientation=orientation_wxyz,
        )
        camera = UsdGeom.Camera(prim)
        focal_length_mm = float(cfg.get("focal_length_mm", 24.0))
        width = float(cfg["width"])
        height = float(cfg["height"])
        fx = float(cfg["fx"])
        fy = float(cfg["fy"])
        camera.GetFocalLengthAttr().Set(focal_length_mm)
        camera.GetHorizontalApertureAttr().Set(width * focal_length_mm / fx)
        camera.GetVerticalApertureAttr().Set(height * focal_length_mm / fy)
        camera.GetHorizontalApertureOffsetAttr().Set(
            (float(cfg["cx"]) - width * 0.5) * focal_length_mm / fx
        )
        camera.GetVerticalApertureOffsetAttr().Set(
            (float(cfg["cy"]) - height * 0.5) * focal_length_mm / fy
        )
        camera.GetClippingRangeAttr().Set(
            Gf.Vec2f(float(cfg["min_depth_m"]), float(cfg["max_depth_m"]))
        )
        camera.GetFocusDistanceAttr().Set(float(cfg["max_depth_m"]))

    def initialize(self) -> None:
        """Create the render product after SimulationContext.reset()."""
        if self.render_product_path is not None:
            return
        import omni.replicator.core as rep

        render_product = rep.create.render_product(
            self.prim_path,
            resolution=(int(self.cfg["width"]), int(self.cfg["height"])),
        )
        path = render_product if isinstance(render_product, str) else render_product.path
        self.render_product_path = str(path)
        for name in self.DATA_TYPES:
            annotator = rep.AnnotatorRegistry.get_annotator(name, device="cpu")
            annotator.attach(self.render_product_path)
            self.annotators[name] = annotator

    @staticmethod
    def _as_depth_tensor(data: Any, height: int, width: int) -> Tensor:
        """Copy Replicator output into a stable CPU tensor."""
        import numpy as np

        array = np.asarray(data)
        if array.size != height * width:
            raise RuntimeError(
                f"unexpected Replicator depth size {array.size}; expected {height * width}"
            )
        return torch.from_numpy(array.reshape(height, width).copy()).to(torch.float32)

    def capture(self, warmup_frames: int | None = None) -> Dict[str, Tensor]:
        """Render fresh frames and return both axial and radial metric depth."""
        if self.render_product_path is None:
            raise RuntimeError("D455M sensor must be initialized before capture")
        frames = int(
            self.cfg["warmup_render_frames"]
            if warmup_frames is None
            else warmup_frames
        )
        for _ in range(max(frames, 1)):
            self.sim.render()
        height = int(self.cfg["height"])
        width = int(self.cfg["width"])
        return {
            name: self._as_depth_tensor(annotator.get_data(), height, width)
            for name, annotator in self.annotators.items()
        }

    def close(self) -> None:
        """Detach annotators before Kit unloads renderer plugins."""
        if self._closed:
            return
        self._closed = True
        if self.render_product_path is not None:
            for annotator in self.annotators.values():
                try:
                    annotator.detach([self.render_product_path])
                except TypeError:
                    annotator.detach(self.render_product_path)
        self.annotators.clear()
        self.render_product_path = None
