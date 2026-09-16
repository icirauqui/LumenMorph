from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np

from ..camera import CameraIntrinsics
from ..deformation import (
    compute_localized_weight,
    deformed_z,
    sample_localized_wave_components,
    sample_wave_components,
)
from ..noise import apply_photometric_noise
from ..plane import add_landmarks_and_rigid_regions
from ..pipeline import GenerationCanceledError
from ..renderer import render_mesh
from ..terrain import build_static_mesh, flatten_vertices
from ..tube import (
    build_tube_static_geometry,
    deformed_tube_mesh,
    tube_areas_from_localized_components,
    tube_deformation_weight,
    tube_vertices_from_radius,
)
from ..tube_fea import build_tube_fea_responses
from ..types import DatasetConfig, LocalizedWaveComponent

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("OpenCV (cv2) is required for preview rendering") from exc


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _rotation_from_lookat(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(up_hint, forward))
    if np.linalg.norm(right) < 1e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = _normalize(np.cross(forward, right))
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def _preview_intrinsics(width: int, height: int, fov_deg: float) -> CameraIntrinsics:
    w = max(16, int(width))
    h = max(16, int(height))
    fov = float(np.deg2rad(np.clip(fov_deg, 10.0, 170.0)))
    fx = 0.5 * w / np.tan(0.5 * fov)
    fy = fx
    return CameraIntrinsics(
        width=w,
        height=h,
        fx=float(fx),
        fy=float(fy),
        cx=0.5 * w,
        cy=0.5 * h,
    )


def _scene_view_pose(
    vertices_world: np.ndarray,
    *,
    yaw_deg: float,
    pitch_deg: float,
    distance_scale: float,
    fov_deg: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    vmin = np.min(vertices_world, axis=0)
    vmax = np.max(vertices_world, axis=0)
    center = 0.5 * (vmin + vmax)

    yaw = float(np.deg2rad(yaw_deg))
    pitch = float(np.deg2rad(np.clip(pitch_deg, -85.0, 85.0)))
    direction = np.array(
        [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ],
        dtype=np.float32,
    )
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(_normalize(direction), up))) > 0.98:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye_hint = center + direction
    R_wc = _rotation_from_lookat(eye_hint.astype(np.float32), center.astype(np.float32), up)
    right = R_wc[:, 0]
    cam_up = R_wc[:, 1]

    offsets = vertices_world - center.reshape(1, 3)
    depth_offsets = offsets @ direction
    x_offsets = np.abs(offsets @ right)
    y_offsets = np.abs(offsets @ cam_up)
    tan_x = max(float(np.tan(0.5 * np.deg2rad(np.clip(fov_deg, 10.0, 170.0)))), 1e-3)
    tan_y = max(tan_x * (max(16, int(height)) / float(max(16, int(width)))), 1e-3)
    margin = 0.86
    fit_distance = float(
        np.max(
            depth_offsets
            + np.maximum(
                x_offsets / (tan_x * margin),
                y_offsets / (tan_y * margin),
            )
        )
    )
    fit_distance = max(fit_distance, float(np.max(depth_offsets)) + 0.1, 0.1)
    dist = fit_distance * max(0.25, float(distance_scale))
    eye = center + dist * direction
    return eye.astype(np.float32), _rotation_from_lookat(eye.astype(np.float32), center.astype(np.float32), up)


def _encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode preview frame")
    return bytes(buf.tobytes())


def _preview_texture(texture_enabled: bool, value: int = 196) -> np.ndarray:
    if not texture_enabled:
        return np.full((1, 1, 3), int(np.clip(value, 0, 255)), dtype=np.uint8)
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    checker = ((xx // 8 + yy // 8) % 2).astype(np.uint8)
    tex = np.zeros((n, n, 3), dtype=np.uint8)
    tex[:, :, 0] = 78 + checker * 42
    tex[:, :, 1] = 128 + checker * 58
    tex[:, :, 2] = 112 + checker * 38
    return tex


def _progress_cb(
    cb: Callable[[str, float, str], None] | None,
    *,
    stage: str,
    percent: float,
    message: str,
) -> None:
    if cb is None:
        return
    cb(stage, float(np.clip(percent, 0.0, 1.0)), message)


def render_preview_frames(
    cfg: DatasetConfig,
    *,
    frame_count: int,
    fps: int,
    yaw_deg: float,
    pitch_deg: float,
    distance_scale: float,
    fov_deg: float,
    width: int,
    height: int,
    texture_enabled: bool = False,
    bg_bgr: tuple[int, int, int] = (14, 14, 14),
    localized_components_override: list[LocalizedWaveComponent] | None = None,
    cancel_event: Any | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> tuple[list[bytes], dict[str, Any]]:
    n_frames = max(1, int(frame_count))
    fps_runtime = max(1, int(fps))
    intr = _preview_intrinsics(width=width, height=height, fov_deg=fov_deg)
    cfg_runtime = replace(cfg, sequence=replace(cfg.sequence, num_frames=n_frames, fps=fps_runtime))
    rng = np.random.default_rng(cfg_runtime.seed)
    preset_name = cfg_runtime.default_preset_cycle[0]
    preset = cfg_runtime.difficulty_presets[preset_name]
    preview_texture = _preview_texture(texture_enabled)

    frames: list[bytes] = []
    _progress_cb(progress_callback, stage="setup", percent=0.0, message="Preparing preview scene")

    if cfg_runtime.scene_type == "plane":
        x_grid, y_grid, base_z, triangles, uv = build_static_mesh(cfg_runtime.sequence, rng)
        assert cfg_runtime.plane is not None
        base_z, rigid_mask, _landmarks, _rigid_fraction, _rigid_pattern = add_landmarks_and_rigid_regions(
            x_grid=x_grid,
            y_grid=y_grid,
            base_z=base_z,
            cfg=cfg_runtime.plane,
            rng=rng,
        )
        rigid_weight_vertices = rigid_mask.reshape(-1).astype(np.float32)
        components = sample_wave_components(rng, preset)
        verts0 = flatten_vertices(x_grid, y_grid, base_z)
        t_wc, R_wc = _scene_view_pose(
            verts0,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            distance_scale=distance_scale,
            fov_deg=fov_deg,
            width=width,
            height=height,
        )
        for frame_idx in range(n_frames):
            if cancel_event is not None and bool(cancel_event.is_set()):
                raise GenerationCanceledError("Preview canceled")
            t_s = frame_idx / float(fps_runtime)
            z_candidate, _, _, _ = deformed_z(base_z, x_grid, y_grid, t_s, components)
            z_grid = base_z + (z_candidate - base_z) * (1.0 - rigid_mask)
            verts = flatten_vertices(x_grid, y_grid, z_grid)
            image = render_mesh(
                vertices_world=verts,
                triangles=triangles,
                uv_vertices=uv,
                texture_bgr=preview_texture,
                intr=intr,
                R_wc=R_wc,
                t_wc=t_wc,
                bg_color=bg_bgr,
                rigid_weight_vertices=rigid_weight_vertices,
                rigid_tint_bgr=tuple(cfg_runtime.plane.rigid_tint_bgr) if cfg_runtime.plane.rigid_colorize else None,
                rigid_tint_strength=cfg_runtime.plane.rigid_tint_strength if cfg_runtime.plane.rigid_colorize else 0.0,
                shade_faces=True,
            )
            image = apply_photometric_noise(image, rng, cfg_runtime.photometric)
            frames.append(_encode_png(image))
            _progress_cb(
                progress_callback,
                stage="frames",
                percent=(frame_idx + 1) / float(n_frames),
                message=f"Preview frame {frame_idx + 1}/{n_frames}",
            )
    else:
        assert cfg_runtime.tube is not None
        geom = build_tube_static_geometry(cfg_runtime.tube, preset, rng)
        components = sample_wave_components(rng, preset)
        localized_components = list(localized_components_override) if localized_components_override is not None else None
        if localized_components_override is None and not cfg_runtime.tube.deformation.areas and cfg_runtime.tube.localized_deform_count > 0:
            localized_components = sample_localized_wave_components(rng, preset, cfg_runtime.tube)
        fea_responses = build_tube_fea_responses(geom, cfg_runtime.tube)
        verts0 = tube_vertices_from_radius(geom, geom.base_radius)
        t_wc, R_wc = _scene_view_pose(
            verts0,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            distance_scale=distance_scale,
            fov_deg=fov_deg,
            width=width,
            height=height,
        )
        for frame_idx in range(n_frames):
            if cancel_event is not None and bool(cancel_event.is_set()):
                raise GenerationCanceledError("Preview canceled")
            t_s = frame_idx / float(fps_runtime)
            t_norm = frame_idx / float(max(n_frames - 1, 1))
            verts, _radius, _displacement_grid, _, _, _ = deformed_tube_mesh(
                geom=geom,
                tube_cfg=cfg_runtime.tube,
                t_s=t_s,
                components=components,
                localized_components=localized_components,
                fea_responses=fea_responses,
                t_norm=t_norm,
            )
            tint_weights = None
            if cfg_runtime.tube.localized_tint_enabled:
                areas = (
                    tube_areas_from_localized_components(localized_components)
                    if localized_components
                    else cfg_runtime.tube.deformation.areas
                )
                if areas:
                    tint_weights = tube_deformation_weight(
                        geom,
                        areas,
                        flow_mode=cfg_runtime.tube.deformation.flow_mode,
                        flow_cycles=cfg_runtime.tube.deformation.flow_cycles,
                        t_norm=t_norm,
                    ).reshape(-1).astype(np.float32)

            image = render_mesh(
                vertices_world=verts,
                triangles=geom.triangles,
                uv_vertices=geom.uv,
                texture_bgr=preview_texture,
                intr=intr,
                R_wc=R_wc,
                t_wc=t_wc,
                bg_color=bg_bgr,
                rigid_weight_vertices=tint_weights,
                rigid_tint_bgr=tuple(cfg_runtime.tube.localized_tint_bgr)
                if cfg_runtime.tube.localized_tint_enabled
                else None,
                rigid_tint_strength=cfg_runtime.tube.localized_tint_strength
                if cfg_runtime.tube.localized_tint_enabled
                else 0.0,
                shade_faces=True,
            )
            image = apply_photometric_noise(image, rng, cfg_runtime.photometric)
            frames.append(_encode_png(image))
            _progress_cb(
                progress_callback,
                stage="frames",
                percent=(frame_idx + 1) / float(n_frames),
                message=f"Preview frame {frame_idx + 1}/{n_frames}",
            )

    meta = {
        "frame_count": n_frames,
        "fps": fps_runtime,
        "width": intr.width,
        "height": intr.height,
        "preset": preset_name,
        "scene_type": cfg_runtime.scene_type,
    }
    return frames, meta
