from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..config import canonical_config_dict, load_config_dict
from ..deformation import (
    compute_localized_weight,
    deformed_z,
    sample_localized_wave_components,
    sample_wave_components,
)
from ..plane import add_landmarks_and_rigid_regions
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
from ..types import CameraIntrinsics, DatasetConfig, LocalizedWaveComponent

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("OpenCV (cv2) is required for realtime preview streaming") from exc


@dataclass
class RealtimeCameraSettings:
    yaw_deg: float = 42.0
    pitch_deg: float = 18.0
    distance_scale: float = 1.45
    fov_deg: float = 55.0
    width: int = 384
    height: int = 216
    background_bgr: tuple[int, int, int] = (14, 14, 14)
    jpeg_quality: int = 82
    texture_enabled: bool = False


@dataclass
class RealtimePlaybackState:
    playing: bool = True
    fps: int = 30
    direction: int = 1
    frame_count: int = 90
    t_norm: float = 0.0


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
    w = max(32, int(width))
    h = max(32, int(height))
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
    tan_y = max(tan_x * (max(32, int(height)) / float(max(32, int(width)))), 1e-3)
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


def _draft_checker_texture(size: int = 128) -> np.ndarray:
    n = max(32, int(size))
    yy, xx = np.mgrid[0:n, 0:n]
    checker = ((xx // 8 + yy // 8) % 2).astype(np.uint8)
    stripes = ((np.sin(xx * 0.25) * 0.5 + 0.5) * 255.0).astype(np.uint8)
    tex = np.zeros((n, n, 3), dtype=np.uint8)
    tex[:, :, 0] = 70 + checker * 50
    tex[:, :, 1] = 120 + checker * 70
    tex[:, :, 2] = 95 + (stripes // 3)
    return tex


def _flat_preview_texture(value: int = 196) -> np.ndarray:
    return np.full((1, 1, 3), int(np.clip(value, 0, 255)), dtype=np.uint8)


def _clip_background(values: list[int] | tuple[int, int, int] | None) -> tuple[int, int, int]:
    if values is None:
        return (14, 14, 14)
    x = list(values)[:3]
    while len(x) < 3:
        x.append(14)
    return (int(np.clip(x[0], 0, 255)), int(np.clip(x[1], 0, 255)), int(np.clip(x[2], 0, 255)))


def _topology_signature(cfg: DatasetConfig, draft_mode: bool) -> str:
    preset_name = cfg.default_preset_cycle[0]
    preset = cfg.difficulty_presets[preset_name]
    payload: dict[str, Any] = {
        "scene_type": cfg.scene_type,
        "seed": cfg.seed,
        "preset_name": preset_name,
        "preset_curve_scale": preset.curve_scale,
        "draft_mode": bool(draft_mode),
    }
    if cfg.scene_type == "tube":
        assert cfg.tube is not None
        payload["tube"] = {
            "axis": [
                {
                    "kind": s.kind,
                    "length": s.length,
                    "direction": s.direction,
                    "angle_deg": s.angle_deg,
                    "radius": s.radius,
                }
                for s in cfg.tube.axis.sections
            ],
            "dimensions": {
                "radial_samples": cfg.tube.radial_samples,
                "longitudinal_samples": cfg.tube.longitudinal_samples,
                "base_radius": cfg.tube.base_radius,
                "min_radius": cfg.tube.min_radius,
                "shell_thickness": cfg.tube.shell_thickness,
            },
            "surface": {
                "bump_count": cfg.tube.bump_count,
                "bump_amp_min": cfg.tube.bump_amp_min,
                "bump_amp_max": cfg.tube.bump_amp_max,
                "bump_sigma_long": cfg.tube.bump_sigma_long,
                "bump_sigma_theta": cfg.tube.bump_sigma_theta,
            },
        }
    else:
        assert cfg.plane is not None
        payload["sequence"] = {
            "mesh_nx": cfg.sequence.mesh_nx,
            "mesh_ny": cfg.sequence.mesh_ny,
            "x_min": cfg.sequence.x_min,
            "x_max": cfg.sequence.x_max,
            "y_min": cfg.sequence.y_min,
            "y_max": cfg.sequence.y_max,
        }
        payload["plane"] = {
            "landmark_count": cfg.plane.landmark_count,
            "landmark_height_min": cfg.plane.landmark_height_min,
            "landmark_height_max": cfg.plane.landmark_height_max,
            "landmark_sigma_min": cfg.plane.landmark_sigma_min,
            "landmark_sigma_max": cfg.plane.landmark_sigma_max,
            "rigid_ratio": cfg.plane.rigid_ratio,
            "rigid_include_landmarks": cfg.plane.rigid_include_landmarks,
            "rigid_edge_softness": cfg.plane.rigid_edge_softness,
        }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _apply_draft_profile(cfg: DatasetConfig) -> DatasetConfig:
    if cfg.scene_type == "tube":
        assert cfg.tube is not None
        has_fea = any(
            area.enabled and area.kind == "fea_c3d6" and area.force_n > 0.0
            for area in cfg.tube.deformation.areas
        )
        radial_cap = 40 if has_fea else 24
        longitudinal_cap = 72 if has_fea else 48
        tube = replace(
            cfg.tube,
            radial_samples=min(cfg.tube.radial_samples, radial_cap),
            longitudinal_samples=min(cfg.tube.longitudinal_samples, longitudinal_cap),
            bump_count=min(cfg.tube.bump_count, 20),
        )
        photo = replace(cfg.photometric, gaussian_sigma=0.0, gain_jitter=0.0)
        return replace(cfg, tube=tube, photometric=photo)
    seq = replace(
        cfg.sequence,
        mesh_nx=min(cfg.sequence.mesh_nx, 24),
        mesh_ny=min(cfg.sequence.mesh_ny, 24),
    )
    photo = replace(cfg.photometric, gaussian_sigma=0.0, gain_jitter=0.0)
    return replace(cfg, sequence=seq, photometric=photo)


class PreviewSession:
    def __init__(
        self,
        *,
        session_id: str,
        config_payload: dict[str, Any],
        camera: RealtimeCameraSettings,
        playback: RealtimePlaybackState,
        localized_components: list[LocalizedWaveComponent] | None,
        draft_mode: bool,
    ) -> None:
        self.id = session_id
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name=f"preview-session-{session_id[:8]}", daemon=True)

        self._draft_mode = bool(draft_mode)
        self._config_payload = json.loads(json.dumps(config_payload))
        self._cfg_full = load_config_dict(self._config_payload)
        self._cfg_runtime = _apply_draft_profile(self._cfg_full) if self._draft_mode else self._cfg_full
        self._camera = camera
        self._playback = playback
        self._localized_override = list(localized_components) if localized_components is not None else None

        self._status = "rebuilding"
        self._error: str | None = None
        self._seq = 0
        self._frame_bytes: bytes | None = None
        self._render_ms = 0.0
        self._server_ts = time.time()
        self._last_frame_monotonic = 0.0
        self._last_play_tick: float | None = None

        self._needs_rebuild_static = True
        self._needs_resample_dynamic = True
        self._topology_signature = ""
        self._static: dict[str, Any] = {}
        self._dynamic: dict[str, Any] = {}
        self._texture = _draft_checker_texture(128)

    def start(self) -> None:
        self._thread.start()
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)

    def wait_for_first_frame(self, timeout_s: float = 3.0) -> bool:
        deadline = time.time() + max(timeout_s, 0.05)
        while time.time() < deadline:
            with self._lock:
                if self._frame_bytes is not None:
                    return True
                if self._status == "error":
                    return False
            time.sleep(0.01)
        return False

    def stream_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.id,
                "seq": self._seq,
                "frame": self._frame_bytes,
                "status": self._status,
                "error": self._error,
                "render_ms": self._render_ms,
                "server_ts": self._server_ts,
                "t_norm": self._playback.t_norm,
                "playing": self._playback.playing,
                "fps": self._playback.fps,
            }

    def public_state(self) -> dict[str, Any]:
        snap = self.stream_snapshot()
        snap.pop("frame", None)
        snap["draft_mode"] = self._draft_mode
        with self._lock:
            snap["camera"] = {
                "yaw_deg": self._camera.yaw_deg,
                "pitch_deg": self._camera.pitch_deg,
                "distance_scale": self._camera.distance_scale,
                "fov_deg": self._camera.fov_deg,
                "width": self._camera.width,
                "height": self._camera.height,
                "background_bgr": list(self._camera.background_bgr),
                "jpeg_quality": self._camera.jpeg_quality,
                "texture_enabled": self._camera.texture_enabled,
            }
            snap["playback"] = {
                "playing": self._playback.playing,
                "fps": self._playback.fps,
                "direction": self._playback.direction,
                "frame_count": self._playback.frame_count,
                "t_norm": self._playback.t_norm,
            }
        return snap

    def export_config_payload(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._config_payload))

    def export_localized_override(self) -> list[LocalizedWaveComponent] | None:
        with self._lock:
            if self._localized_override is None:
                return None
            return list(self._localized_override)

    def patch_config(
        self,
        *,
        config_payload: dict[str, Any] | None = None,
        localized_components: list[LocalizedWaveComponent] | None | object = None,
    ) -> None:
        with self._lock:
            if config_payload is not None:
                cfg_full = load_config_dict(config_payload)
                cfg_runtime = _apply_draft_profile(cfg_full) if self._draft_mode else cfg_full
                topo_new = _topology_signature(cfg_runtime, self._draft_mode)
                topo_old = _topology_signature(self._cfg_runtime, self._draft_mode)
                self._config_payload = json.loads(json.dumps(canonical_config_dict(cfg_full)))
                self._cfg_full = cfg_full
                self._cfg_runtime = cfg_runtime
                if topo_new != topo_old:
                    self._needs_rebuild_static = True
                    self._status = "rebuilding"
                self._needs_resample_dynamic = True
            if localized_components is not None:
                self._localized_override = list(localized_components)
                self._needs_resample_dynamic = True
            self._error = None
        self._wake.set()

    def patch_camera(
        self,
        *,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        distance_scale: float | None = None,
        fov_deg: float | None = None,
        width: int | None = None,
        height: int | None = None,
        background_bgr: list[int] | None = None,
        jpeg_quality: int | None = None,
        texture_enabled: bool | None = None,
    ) -> None:
        with self._lock:
            if yaw_deg is not None:
                self._camera.yaw_deg = float(yaw_deg)
            if pitch_deg is not None:
                self._camera.pitch_deg = float(pitch_deg)
            if distance_scale is not None:
                self._camera.distance_scale = float(distance_scale)
            if fov_deg is not None:
                self._camera.fov_deg = float(fov_deg)
            if width is not None:
                self._camera.width = max(32, int(width))
            if height is not None:
                self._camera.height = max(32, int(height))
            if background_bgr is not None:
                self._camera.background_bgr = _clip_background(background_bgr)
            if jpeg_quality is not None:
                self._camera.jpeg_quality = int(np.clip(jpeg_quality, 35, 97))
            if texture_enabled is not None:
                self._camera.texture_enabled = bool(texture_enabled)
        self._wake.set()

    def patch_playback(
        self,
        *,
        playing: bool | None = None,
        fps: int | None = None,
        direction: int | None = None,
        frame_count: int | None = None,
        t_norm: float | None = None,
    ) -> None:
        with self._lock:
            if playing is not None:
                self._playback.playing = bool(playing)
                if not self._playback.playing:
                    self._last_play_tick = None
            if fps is not None:
                self._playback.fps = max(1, int(fps))
            if direction is not None:
                self._playback.direction = 1 if int(direction) >= 0 else -1
            if frame_count is not None:
                self._playback.frame_count = max(1, int(frame_count))
            if t_norm is not None:
                self._playback.t_norm = float(np.clip(t_norm, 0.0, 1.0))
                self._last_play_tick = None
        self._wake.set()

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                playing = self._playback.playing
                target_fps = max(1, self._playback.fps)
            target_period = 1.0 / float(target_fps)
            now = time.monotonic()
            should_render = False
            timeout = 0.04

            if self._wake.is_set():
                self._wake.clear()
                should_render = True
            elif playing:
                if self._last_frame_monotonic <= 0.0 or (now - self._last_frame_monotonic) >= target_period:
                    should_render = True
                else:
                    timeout = min(timeout, target_period - (now - self._last_frame_monotonic))

            if not should_render:
                self._wake.wait(timeout=max(timeout, 0.005))
                continue

            started = time.perf_counter()
            try:
                frame, t_norm = self._render_once(time.monotonic())
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(self._camera.jpeg_quality, 35, 97))],
                )
                if not ok:
                    raise RuntimeError("Failed to encode realtime frame")
                render_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._frame_bytes = bytes(encoded.tobytes())
                    self._seq += 1
                    self._status = "ok"
                    self._error = None
                    self._render_ms = render_ms
                    self._playback.t_norm = t_norm
                    self._server_ts = time.time()
                    self._last_frame_monotonic = time.monotonic()
            except Exception as exc:  # pragma: no cover - runtime defensive path
                with self._lock:
                    self._status = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._server_ts = time.time()
                    self._last_frame_monotonic = time.monotonic()

    def _render_once(self, now_mono: float) -> tuple[np.ndarray, float]:
        cfg = self._cfg_runtime
        if self._needs_rebuild_static:
            with self._lock:
                self._status = "rebuilding"
                self._error = None
            self._rebuild_static_cache(cfg)
            self._needs_rebuild_static = False
            self._needs_resample_dynamic = True
            self._topology_signature = _topology_signature(cfg, self._draft_mode)

        if self._needs_resample_dynamic:
            self._resample_dynamic_components(cfg)
            self._needs_resample_dynamic = False

        with self._lock:
            t_norm = float(np.clip(self._playback.t_norm, 0.0, 1.0))
            if self._playback.playing:
                if self._last_play_tick is None:
                    self._last_play_tick = now_mono
                dt = max(0.0, now_mono - self._last_play_tick)
                self._last_play_tick = now_mono
                span = max(self._playback.frame_count - 1, 1)
                t_norm = (t_norm + self._playback.direction * dt * self._playback.fps / float(span)) % 1.0
            else:
                self._last_play_tick = None

            cam = replace(self._camera)
            localized_override = list(self._localized_override) if self._localized_override is not None else None

        if cfg.scene_type == "tube":
            return self._render_tube_frame(cfg, cam, t_norm=t_norm, localized_override=localized_override), t_norm
        return self._render_plane_frame(cfg, cam, t_norm=t_norm), t_norm

    def _rebuild_static_cache(self, cfg: DatasetConfig) -> None:
        rng = np.random.default_rng(cfg.seed)
        preset = cfg.difficulty_presets[cfg.default_preset_cycle[0]]
        if cfg.scene_type == "tube":
            assert cfg.tube is not None
            geom = build_tube_static_geometry(cfg.tube, preset, rng)
            self._static = {"geom": geom}
            return
        assert cfg.plane is not None
        x_grid, y_grid, base_z, triangles, uv = build_static_mesh(cfg.sequence, rng)
        base_z, rigid_mask, _, _, _ = add_landmarks_and_rigid_regions(
            x_grid=x_grid,
            y_grid=y_grid,
            base_z=base_z,
            cfg=cfg.plane,
            rng=rng,
        )
        self._static = {
            "x_grid": x_grid,
            "y_grid": y_grid,
            "base_z": base_z,
            "triangles": triangles,
            "uv": uv,
            "rigid_mask": rigid_mask,
            "rigid_weight": rigid_mask.reshape(-1).astype(np.float32),
        }

    def _resample_dynamic_components(self, cfg: DatasetConfig) -> None:
        rng = np.random.default_rng(cfg.seed + 99173)
        preset = cfg.difficulty_presets[cfg.default_preset_cycle[0]]
        components = sample_wave_components(rng, preset)
        localized_base: list[LocalizedWaveComponent] = []
        if cfg.scene_type == "tube":
            assert cfg.tube is not None
            if not cfg.tube.deformation.areas and cfg.tube.localized_deform_count > 0:
                localized_base = sample_localized_wave_components(rng, preset, cfg.tube)
            fea_responses = build_tube_fea_responses(self._static["geom"], cfg.tube)
        else:
            fea_responses = []
        self._dynamic = {
            "components": components,
            "localized_base": localized_base,
            "fea_responses": fea_responses,
        }

    def _render_plane_frame(
        self,
        cfg: DatasetConfig,
        cam: RealtimeCameraSettings,
        *,
        t_norm: float,
    ) -> np.ndarray:
        st = self._static
        components = self._dynamic["components"]
        x_grid = st["x_grid"]
        y_grid = st["y_grid"]
        base_z = st["base_z"]
        rigid_mask = st["rigid_mask"]
        t_s = t_norm * max((cfg.sequence.num_frames - 1) / max(cfg.sequence.fps, 1), 1.0 / max(cfg.sequence.fps, 1))
        z_candidate, _, _, _ = deformed_z(base_z, x_grid, y_grid, t_s, components)
        z_grid = base_z + (z_candidate - base_z) * (1.0 - rigid_mask)
        verts = flatten_vertices(x_grid, y_grid, z_grid)

        intr = _preview_intrinsics(cam.width, cam.height, cam.fov_deg)
        t_wc, R_wc = _scene_view_pose(
            verts,
            yaw_deg=cam.yaw_deg,
            pitch_deg=cam.pitch_deg,
            distance_scale=cam.distance_scale,
            fov_deg=cam.fov_deg,
            width=cam.width,
            height=cam.height,
        )
        return render_mesh(
            vertices_world=verts,
            triangles=st["triangles"],
            uv_vertices=st["uv"],
            texture_bgr=self._texture if cam.texture_enabled else _flat_preview_texture(),
            intr=intr,
            R_wc=R_wc,
            t_wc=t_wc,
            bg_color=cam.background_bgr,
            rigid_weight_vertices=st["rigid_weight"],
            rigid_tint_bgr=tuple(cfg.plane.rigid_tint_bgr) if cfg.plane and cfg.plane.rigid_colorize else None,
            rigid_tint_strength=cfg.plane.rigid_tint_strength if cfg.plane and cfg.plane.rigid_colorize else 0.0,
            shade_faces=True,
        )

    def _render_tube_frame(
        self,
        cfg: DatasetConfig,
        cam: RealtimeCameraSettings,
        *,
        t_norm: float,
        localized_override: list[LocalizedWaveComponent] | None,
    ) -> np.ndarray:
        assert cfg.tube is not None
        geom = self._static["geom"]
        components = self._dynamic["components"]
        localized_components = localized_override if localized_override is not None else self._dynamic["localized_base"] or None
        fea_responses = self._dynamic.get("fea_responses") or []
        t_s = t_norm * max((cfg.sequence.num_frames - 1) / max(cfg.sequence.fps, 1), 1.0 / max(cfg.sequence.fps, 1))
        verts, _radius, _displacement_grid, _, _, _ = deformed_tube_mesh(
            geom=geom,
            tube_cfg=cfg.tube,
            t_s=t_s,
            components=components,
            localized_components=localized_components,
            fea_responses=fea_responses,
            t_norm=t_norm,
        )

        intr = _preview_intrinsics(cam.width, cam.height, cam.fov_deg)
        t_wc, R_wc = _scene_view_pose(
            verts,
            yaw_deg=cam.yaw_deg,
            pitch_deg=cam.pitch_deg,
            distance_scale=cam.distance_scale,
            fov_deg=cam.fov_deg,
            width=cam.width,
            height=cam.height,
        )

        tint_weights = None
        if cfg.tube.localized_tint_enabled:
            areas = (
                tube_areas_from_localized_components(localized_components)
                if localized_components
                else cfg.tube.deformation.areas
            )
            if areas:
                tint_weights = tube_deformation_weight(
                    geom,
                    areas,
                    flow_mode=cfg.tube.deformation.flow_mode,
                    flow_cycles=cfg.tube.deformation.flow_cycles,
                    t_norm=t_norm,
                ).reshape(-1).astype(np.float32)

        return render_mesh(
            vertices_world=verts,
            triangles=geom.triangles,
            uv_vertices=geom.uv,
            texture_bgr=self._texture if cam.texture_enabled else _flat_preview_texture(),
            intr=intr,
            R_wc=R_wc,
            t_wc=t_wc,
            bg_color=cam.background_bgr,
            rigid_weight_vertices=tint_weights,
            rigid_tint_bgr=tuple(cfg.tube.localized_tint_bgr) if cfg.tube.localized_tint_enabled else None,
            rigid_tint_strength=cfg.tube.localized_tint_strength if cfg.tube.localized_tint_enabled else 0.0,
            shade_faces=True,
        )


class PreviewSessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, PreviewSession] = {}

    def create_session(
        self,
        *,
        config_payload: dict[str, Any],
        camera: RealtimeCameraSettings,
        playback: RealtimePlaybackState,
        localized_components: list[LocalizedWaveComponent] | None,
        draft_mode: bool,
    ) -> PreviewSession:
        cfg = load_config_dict(config_payload)
        canonical = canonical_config_dict(cfg)
        session_id = uuid.uuid4().hex
        session = PreviewSession(
            session_id=session_id,
            config_payload=canonical,
            camera=camera,
            playback=playback,
            localized_components=localized_components,
            draft_mode=draft_mode,
        )
        with self._lock:
            self._sessions[session_id] = session
        session.start()
        session.wait_for_first_frame(timeout_s=2.5)
        return session

    def get_session(self, session_id: str) -> PreviewSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            raise KeyError(session_id)
        session.close()

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
