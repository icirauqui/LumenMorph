from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import canonical_config_dict, config_hash, load_config, load_config_dict, tube_config_warnings
from ..pipeline import generate_dataset
from ..tube import build_tube_static_geometry
from ..tube_fea import build_tube_fea_responses
from ..types import LocalizedWaveComponent
from .jobs import JobManager
from .models import (
    FullPreviewFromSessionRequest,
    GenerateJobRequest,
    NormalizeConfigRequest,
    PreviewJobRequest,
    PreviewSessionCameraPatchRequest,
    PreviewSessionConfigPatchRequest,
    PreviewSessionCreateRequest,
    PreviewSessionPlaybackPatchRequest,
    TubeFeaPreviewRequest,
    localized_components_or_none,
)
from .preview import render_preview_frames
from .realtime import PreviewSessionManager, RealtimeCameraSettings, RealtimePlaybackState

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIGS_DIR = ROOT_DIR / "configs"


def _default_output_dir() -> Path:
    return ROOT_DIR / "outputs"


def _resolve_output_dir(output_dir: str | None) -> Path:
    raw = str(output_dir or "").strip()
    path = Path(raw).expanduser() if raw else _default_output_dir()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve(strict=False)


def _select_output_dir_with_dialog() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - depends on local desktop support
        raise RuntimeError("Native directory dialog is unavailable on this backend host") from exc

    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=str(_default_output_dir().parent),
            title="Select dataset output directory",
        )
    except Exception as exc:  # pragma: no cover - depends on local desktop support
        raise RuntimeError("Native directory dialog is unavailable on this backend host") from exc
    finally:
        if "root" in locals():
            root.destroy()
    if not selected:
        return None
    return _resolve_output_dir(selected)


def _flatten_dict(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_dict(key, v, out)
    else:
        out[prefix] = value


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _control_schema() -> dict[str, Any]:
    plane_cfg = canonical_config_dict(load_config(CONFIGS_DIR / "default.json"))
    tube_cfg = canonical_config_dict(load_config(CONFIGS_DIR / "tube_default.json"))
    flat_plane: dict[str, Any] = {}
    flat_tube: dict[str, Any] = {}
    _flatten_dict("", plane_cfg, flat_plane)
    _flatten_dict("", tube_cfg, flat_tube)

    keys = sorted(set(flat_plane) | set(flat_tube))
    metadata: dict[str, dict[str, Any]] = {
        "seed": {"min": 0, "step": 1},
        "scene_type": {"enum": ["plane", "tube"]},
        "sequence.num_frames": {"min": 1, "step": 1},
        "sequence.fps": {"min": 1, "step": 1},
        "output.write_frames": {"enum": [True, False]},
        "output.write_mp4": {"enum": [True, False]},
        "output.write_mesh_state": {"enum": [True, False]},
        "tube.deformation.flow_mode": {"enum": ["linear", "cyclic"]},
        "tube.deformation.flow_cycles": {"min": 0.0, "step": 0.1},
        "tube.deformation.fea.young_modulus_pa": {"min": 1.0, "step": 1000000.0},
        "tube.deformation.fea.poisson_ratio": {"min": -0.99, "max": 0.49, "step": 0.01},
        "tube.deformation.fea.axial_elements": {"min": 2, "step": 1},
        "tube.deformation.fea.theta_elements": {"min": 6, "step": 1},
    }
    fields: dict[str, Any] = {}
    for key in keys:
        default_tube = flat_tube.get(key)
        default_plane = flat_plane.get(key)
        default = default_tube if default_tube is not None else default_plane
        fields[key] = {
            "type": _infer_type(default),
            "default": default,
            "defaults_by_scene": {
                "plane": default_plane,
                "tube": default_tube,
            },
        }
        if key in metadata:
            fields[key].update(metadata[key])

    return {
        "scene_types": ["plane", "tube"],
        "enums": {
            "tube.deformation.flow_mode": ["linear", "cyclic"],
        },
        "groups": {
            "root": [
                "version",
                "seed",
                "scene_type",
                "sequence_count.train",
                "sequence_count.val",
                "sequence_count.test",
                "default_preset_cycle",
            ],
            "intrinsics": [
                "intrinsics.width",
                "intrinsics.height",
                "intrinsics.fx",
                "intrinsics.fy",
                "intrinsics.cx",
                "intrinsics.cy",
            ],
            "sequence": [
                "sequence.num_frames",
                "sequence.fps",
                "sequence.mesh_nx",
                "sequence.mesh_ny",
                "sequence.x_min",
                "sequence.x_max",
                "sequence.y_min",
                "sequence.y_max",
            ],
            "camera": [
                "camera.base_x",
                "camera.base_y",
                "camera.base_z",
                "camera.forward_drift_per_s",
                "camera.orbit_radius",
                "camera.orbit_frequency_hz",
                "camera.lateral_amplitude",
                "camera.lateral_frequency_hz",
                "camera.lookahead_distance",
                "camera.lookat_height",
                "camera.roll_jitter_deg",
                "camera.pitch_jitter_deg",
                "camera.yaw_jitter_deg",
            ],
            "texture": [
                "texture.width",
                "texture.height",
                "texture.min_orb_features",
                "texture.max_attempts",
            ],
            "photometric": [
                "photometric.gaussian_sigma",
                "photometric.gain_jitter",
            ],
            "output": [
                "output.write_frames",
                "output.write_mp4",
                "output.write_mesh_state",
            ],
            "validation": [
                "validation.min_orb_features_mean",
                "validation.min_orb_features_frame",
                "validation.min_frame_motion",
                "validation.min_camera_clearance",
                "validation.max_abs_mesh_z",
            ],
            "plane": [
                "plane.landmark_count",
                "plane.landmark_height_min",
                "plane.landmark_height_max",
                "plane.landmark_sigma_min",
                "plane.landmark_sigma_max",
                "plane.rigid_ratio",
                "plane.rigid_include_landmarks",
                "plane.rigid_edge_softness",
                "plane.rigid_colorize",
                "plane.rigid_tint_bgr",
                "plane.rigid_tint_strength",
                "plane.flip_vertical_view",
            ],
            "tube": [
                "tube.axis.sections",
                "tube.dimensions.radial_samples",
                "tube.dimensions.longitudinal_samples",
                "tube.dimensions.base_radius",
                "tube.dimensions.min_radius",
                "tube.dimensions.shell_thickness",
                "tube.dimensions.cap_ends",
                "tube.surface.bump_count",
                "tube.surface.bump_amp_min",
                "tube.surface.bump_amp_max",
                "tube.surface.bump_sigma_long",
                "tube.surface.bump_sigma_theta",
                "tube.camera.offset_ratio",
                "tube.camera.wobble_ratio",
                "tube.camera.wobble_hz",
                "tube.camera.lookahead_distance",
                "tube.camera.end_margin_distance",
                "tube.deformation.flow_mode",
                "tube.deformation.flow_cycles",
                "tube.deformation.fea.young_modulus_pa",
                "tube.deformation.fea.poisson_ratio",
                "tube.deformation.fea.axial_elements",
                "tube.deformation.fea.theta_elements",
                "tube.deformation.localized_tint_enabled",
                "tube.deformation.localized_tint_bgr",
                "tube.deformation.localized_tint_strength",
                "tube.deformation.areas",
            ],
        },
        "fields": fields,
    }


def _load_presets() -> list[dict[str, Any]]:
    presets: list[dict[str, Any]] = []
    for path in sorted(CONFIGS_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        try:
            canonical_payload = canonical_config_dict(load_config_dict(payload))
        except Exception:
            canonical_payload = payload
        presets.append(
            {
                "name": path.stem,
                "path": str(path),
                "scene_type": payload.get("scene_type", "plane"),
                "config": canonical_payload,
                "warnings": tube_config_warnings(payload),
            }
        )
    return presets


def _as_localized_components(
    request: GenerateJobRequest | PreviewJobRequest,
) -> list[LocalizedWaveComponent] | None:
    return localized_components_or_none(request.localized_components)


def _freeze_global_waves_for_static(cfg_payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cfg_payload))
    presets = (out.get("difficulty_presets") or {})
    if isinstance(presets, dict):
        for v in presets.values():
            if not isinstance(v, dict):
                continue
            v["wave_count"] = 0
            v["amp_min"] = 0.0
            v["amp_max"] = 0.0
            v["omega_min"] = 0.0
            v["omega_max"] = 0.0
    return out


def _static_extreme_components(
    components: list[LocalizedWaveComponent],
    *,
    sign: int,
) -> list[LocalizedWaveComponent]:
    phase = math.pi / 2.0 if sign >= 0 else -math.pi / 2.0
    out: list[LocalizedWaveComponent] = []
    for c in components:
        amp_base = float(abs(c.amp))
        amp_out = float(abs(c.amp_out if c.amp_out is not None else amp_base))
        amp_in = float(abs(c.amp_in if c.amp_in is not None else amp_base))
        out.append(
            replace(
                c,
                amp=amp_out if sign >= 0 else -amp_in,
                amp_out=amp_out,
                amp_in=amp_in,
                fx=0.0,
                fy=0.0,
                omega=0.0,
                phase=phase,
                transversal_only=True,
                flow_dir_s=0.0,
                flow_dir_theta=0.0,
            )
        )
    return out


def create_app() -> FastAPI:
    jobs = JobManager()
    preview_sessions = PreviewSessionManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            preview_sessions.close_all()

    app = FastAPI(title="LumenMorph local API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    @app.get("/v1/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "service": "lumenmorph-local-api", "version": "0.1.0"}

    @app.get("/v1/schema")
    async def schema() -> dict[str, Any]:
        return _control_schema()

    @app.get("/v1/presets")
    async def presets() -> dict[str, Any]:
        return {"presets": _load_presets()}

    @app.get("/v1/output/defaults")
    async def output_defaults() -> dict[str, Any]:
        default_output_dir = _default_output_dir()
        return {
            "repo_root": str(ROOT_DIR),
            "default_output_dir": str(default_output_dir),
            "exists": default_output_dir.exists(),
        }

    @app.post("/v1/output/select-directory")
    async def select_output_directory() -> dict[str, Any]:
        try:
            selected = _select_output_dir_with_dialog()
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        if selected is None:
            return {"selected": False, "output_dir": str(_default_output_dir())}
        return {"selected": True, "output_dir": str(selected)}

    @app.post("/v1/config/normalize")
    async def normalize_config(body: NormalizeConfigRequest) -> dict[str, Any]:
        warnings = tube_config_warnings(body.config)
        try:
            cfg = load_config_dict(body.config)
        except Exception as exc:
            return {"ok": False, "errors": [str(exc)], "warnings": []}
        return {
            "ok": True,
            "errors": [],
            "warnings": warnings,
            "normalized_config": canonical_config_dict(cfg),
            "config_hash": config_hash(cfg),
        }

    @app.post("/v1/tube/fea-preview")
    async def tube_fea_preview(body: TubeFeaPreviewRequest) -> dict[str, Any]:
        try:
            cfg = load_config_dict(body.config)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc
        if cfg.scene_type != "tube" or cfg.tube is None:
            raise HTTPException(status_code=400, detail="FEA preview requires scene_type='tube'")

        tube = cfg.tube
        if body.longitudinal_samples is not None or body.radial_samples is not None:
            tube = replace(
                tube,
                longitudinal_samples=max(2, int(body.longitudinal_samples or tube.longitudinal_samples)),
                radial_samples=max(8, int(body.radial_samples or tube.radial_samples)),
            )
            cfg = replace(cfg, tube=tube)

        def _solve() -> dict[str, Any]:
            rng = np.random.default_rng(cfg.seed)
            preset = cfg.difficulty_presets[cfg.default_preset_cycle[0]]
            geom = build_tube_static_geometry(tube, preset, rng)
            responses = build_tube_fea_responses(geom, tube)
            return {
                "longitudinal_samples": int(tube.longitudinal_samples),
                "radial_samples": int(tube.radial_samples),
                "responses": [
                    {
                        "area_id": response.area_id,
                        "applied_s_norm": response.applied_s_norm,
                        "applied_theta_norm": response.applied_theta_norm,
                        "applied_node": response.applied_node,
                        "applied_axial_index": response.applied_axial_index,
                        "applied_theta_index": response.applied_theta_index,
                        "force_vector_local": response.force_vector_local.astype(float).tolist(),
                        "displacement_grid": response.displacement_grid.astype(float).tolist(),
                    }
                    for response in responses
                ],
            }

        # The SciPy sparse solve used by the C3D6 helper can hang in a Python
        # worker thread on some OpenBLAS/SuperLU builds, so keep this solve on
        # the request thread. The frontend triggers it explicitly via the
        # refresh button and then caches the returned displacement grid.
        return _solve()

    @app.post("/v1/preview/jobs")
    async def start_preview_job(body: PreviewJobRequest) -> dict[str, Any]:
        try:
            cfg = load_config_dict(body.config)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc

        localized_override = _as_localized_components(body)
        bg = body.preview_camera.background_bgr
        bg_bgr = (
            int(bg[0]) if len(bg) > 0 else 14,
            int(bg[1]) if len(bg) > 1 else 14,
            int(bg[2]) if len(bg) > 2 else 14,
        )

        def _runner(ctx) -> dict[str, Any]:
            frames, meta = render_preview_frames(
                cfg=cfg,
                frame_count=body.frame_count,
                fps=body.fps,
                yaw_deg=body.preview_camera.yaw_deg,
                pitch_deg=body.preview_camera.pitch_deg,
                distance_scale=body.preview_camera.distance_scale,
                fov_deg=body.preview_camera.fov_deg,
                width=body.preview_camera.width,
                height=body.preview_camera.height,
                texture_enabled=body.preview_camera.texture_enabled,
                bg_bgr=bg_bgr,
                localized_components_override=localized_override,
                cancel_event=ctx.cancel_event,
                progress_callback=ctx.report,
            )
            return {"frames": frames, "result": meta}

        try:
            job_id = jobs.create_job(kind="preview", runner=_runner)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job_id": job_id}

    @app.post("/v1/preview/sessions")
    async def create_preview_session(body: PreviewSessionCreateRequest) -> dict[str, Any]:
        try:
            cfg = load_config_dict(body.config)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc
        localized_override = localized_components_or_none(body.localized_components)
        camera = RealtimeCameraSettings(
            yaw_deg=body.preview_camera.yaw_deg,
            pitch_deg=body.preview_camera.pitch_deg,
            distance_scale=body.preview_camera.distance_scale,
            fov_deg=body.preview_camera.fov_deg,
            width=body.preview_camera.width,
            height=body.preview_camera.height,
            background_bgr=tuple(body.preview_camera.background_bgr[:3]) if body.preview_camera.background_bgr else (14, 14, 14),
            jpeg_quality=body.preview_camera.jpeg_quality,
            texture_enabled=body.preview_camera.texture_enabled,
        )
        playback = RealtimePlaybackState(
            playing=body.playback.playing,
            fps=body.playback.fps,
            direction=body.playback.direction,
            frame_count=body.playback.frame_count,
            t_norm=body.playback.t_norm,
        )
        session = preview_sessions.create_session(
            config_payload=canonical_config_dict(cfg),
            camera=camera,
            playback=playback,
            localized_components=localized_override,
            draft_mode=bool(body.draft_profile),
        )
        return session.public_state()

    @app.get("/v1/preview/sessions/{session_id}")
    async def get_preview_session(session_id: str) -> dict[str, Any]:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc
        return session.public_state()

    @app.patch("/v1/preview/sessions/{session_id}/config")
    async def patch_preview_session_config(
        session_id: str,
        body: PreviewSessionConfigPatchRequest,
    ) -> dict[str, Any]:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc
        cfg_payload: dict[str, Any] | None = None
        if body.config is not None:
            try:
                cfg = load_config_dict(body.config)
                cfg_payload = canonical_config_dict(cfg)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid config patch: {exc}") from exc
        localized = localized_components_or_none(body.localized_components)
        session.patch_config(config_payload=cfg_payload, localized_components=localized)
        return session.public_state()

    @app.patch("/v1/preview/sessions/{session_id}/camera")
    async def patch_preview_session_camera(
        session_id: str,
        body: PreviewSessionCameraPatchRequest,
    ) -> dict[str, Any]:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc
        session.patch_camera(
            yaw_deg=body.yaw_deg,
            pitch_deg=body.pitch_deg,
            distance_scale=body.distance_scale,
            fov_deg=body.fov_deg,
            width=body.width,
            height=body.height,
            background_bgr=body.background_bgr,
            jpeg_quality=body.jpeg_quality,
            texture_enabled=body.texture_enabled,
        )
        return session.public_state()

    @app.patch("/v1/preview/sessions/{session_id}/playback")
    async def patch_preview_session_playback(
        session_id: str,
        body: PreviewSessionPlaybackPatchRequest,
    ) -> dict[str, Any]:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc
        session.patch_playback(
            playing=body.playing,
            fps=body.fps,
            direction=body.direction,
            frame_count=body.frame_count,
            t_norm=body.t_norm,
        )
        return session.public_state()

    @app.delete("/v1/preview/sessions/{session_id}")
    async def delete_preview_session(session_id: str) -> dict[str, Any]:
        try:
            preview_sessions.delete_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc
        return {"ok": True, "session_id": session_id}

    @app.get("/v1/preview/sessions/{session_id}/frame/latest")
    async def preview_session_latest_frame(session_id: str) -> Response:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc

        snap = session.stream_snapshot()
        headers = {
            "X-Preview-Seq": str(int(snap.get("seq", 0))),
            "X-Preview-TNorm": str(float(snap.get("t_norm", 0.0))),
            "X-Preview-Render-Ms": str(float(snap.get("render_ms", 0.0))),
            "X-Preview-Status": str(snap.get("status", "unknown")),
            "X-Preview-Server-Ts": str(float(snap.get("server_ts", 0.0))),
            "X-Preview-Playing": "1" if bool(snap.get("playing", False)) else "0",
            "X-Preview-Fps": str(int(snap.get("fps", 0))),
        }
        error = snap.get("error")
        if error:
            headers["X-Preview-Error"] = str(error)

        frame = snap.get("frame")
        if frame is None:
            payload = {
                "ok": False,
                "session_id": session_id,
                "status": snap.get("status", "rebuilding"),
                "error": error,
            }
            return JSONResponse(content=payload, status_code=202, headers=headers)

        return Response(content=frame, media_type="image/jpeg", headers=headers)

    @app.websocket("/v1/preview/sessions/{session_id}/stream")
    async def stream_preview_session(session_id: str, websocket: WebSocket) -> None:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        last_seq = -1
        last_status: str | None = None
        try:
            while True:
                snap = session.stream_snapshot()
                if snap["seq"] != last_seq and snap["frame"] is not None:
                    await websocket.send_json(
                        {
                            "type": "meta",
                            "session_id": snap["session_id"],
                            "seq": snap["seq"],
                            "t_norm": snap["t_norm"],
                            "render_ms": snap["render_ms"],
                            "status": snap["status"],
                            "error": snap["error"],
                            "server_ts": snap["server_ts"],
                            "playing": snap["playing"],
                            "fps": snap["fps"],
                        }
                    )
                    await websocket.send_bytes(snap["frame"])
                    last_seq = snap["seq"]
                    last_status = str(snap["status"])
                elif str(snap["status"]) != str(last_status):
                    await websocket.send_json(
                        {
                            "type": "meta",
                            "session_id": snap["session_id"],
                            "seq": snap["seq"],
                            "t_norm": snap["t_norm"],
                            "render_ms": snap["render_ms"],
                            "status": snap["status"],
                            "error": snap["error"],
                            "server_ts": snap["server_ts"],
                            "playing": snap["playing"],
                            "fps": snap["fps"],
                        }
                    )
                    last_status = str(snap["status"])
                await asyncio.sleep(0.02)
        except WebSocketDisconnect:
            return

    @app.post("/v1/preview/sessions/{session_id}/full-preview/jobs")
    async def start_full_preview_for_session(
        session_id: str,
        body: FullPreviewFromSessionRequest,
    ) -> dict[str, Any]:
        try:
            session = preview_sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown preview session: {session_id}") from exc

        config_payload = session.export_config_payload()
        localized_override = localized_components_or_none(body.localized_components)
        if localized_override is None:
            localized_override = session.export_localized_override()
        try:
            cfg = load_config_dict(config_payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid session config: {exc}") from exc

        if body.preview_camera is not None:
            yaw_deg = float(body.preview_camera.yaw_deg)
            pitch_deg = float(body.preview_camera.pitch_deg)
            distance_scale = float(body.preview_camera.distance_scale)
            fov_deg = float(body.preview_camera.fov_deg)
            width = int(body.preview_camera.width)
            height = int(body.preview_camera.height)
            background_bgr = list(body.preview_camera.background_bgr)
            texture_enabled = bool(body.preview_camera.texture_enabled)
        else:
            session_state = session.public_state()
            camera = (session_state.get("camera") or {})
            yaw_deg = float(camera.get("yaw_deg", 42.0))
            pitch_deg = float(camera.get("pitch_deg", 18.0))
            distance_scale = float(camera.get("distance_scale", 1.45))
            fov_deg = float(camera.get("fov_deg", 55.0))
            width = int(camera.get("width", 384))
            height = int(camera.get("height", 216))
            background_bgr = list(camera.get("background_bgr", [14, 14, 14]))
            texture_enabled = bool(camera.get("texture_enabled", False))

        bg = background_bgr
        bg_bgr = (
            int(bg[0]) if len(bg) > 0 else 14,
            int(bg[1]) if len(bg) > 1 else 14,
            int(bg[2]) if len(bg) > 2 else 14,
        )

        def _runner(ctx) -> dict[str, Any]:
            frames, meta = render_preview_frames(
                cfg=cfg,
                frame_count=body.frame_count,
                fps=body.fps,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                distance_scale=distance_scale,
                fov_deg=fov_deg,
                width=width,
                height=height,
                texture_enabled=texture_enabled,
                bg_bgr=bg_bgr,
                localized_components_override=localized_override,
                cancel_event=ctx.cancel_event,
                progress_callback=ctx.report,
            )
            return {"frames": frames, "result": {**meta, "source_session_id": session_id}}

        try:
            job_id = jobs.create_job(kind="preview", runner=_runner)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job_id": job_id}

    @app.post("/v1/generate/jobs")
    async def start_generate_job(body: GenerateJobRequest) -> dict[str, Any]:
        try:
            cfg = load_config_dict(body.config)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid config: {exc}") from exc

        localized_override = _as_localized_components(body)
        out_dir = str(_resolve_output_dir(body.output_dir))

        def _runner(ctx) -> dict[str, Any]:
            quick = body.quick
            max_sequences = quick.max_sequences
            override_num_frames = quick.override_num_frames
            override_fps = quick.override_fps
            localized_for_run = None
            if body.mode == "quick":
                if max_sequences is None:
                    max_sequences = 1
                localized_for_run = localized_override if cfg.scene_type == "tube" else None
                flatten_single_sequence = bool(max_sequences == 1)
                manifest = generate_dataset(
                    cfg=cfg,
                    out_dir=out_dir,
                    max_sequences=max_sequences,
                    override_num_frames=override_num_frames,
                    override_fps=override_fps,
                    show_progress=bool(body.show_progress),
                    localized_components_override=localized_for_run,
                    flatten_single_sequence=flatten_single_sequence,
                    cancel_event=ctx.cancel_event,
                    progress_callback=ctx.report,
                )
                return {
                    "result": {
                        "mode": body.mode,
                        "output_dir": out_dir,
                        "manifest": manifest,
                    }
                }
            if body.mode == "full":
                max_sequences = None
                localized_for_run = None
                manifest = generate_dataset(
                    cfg=cfg,
                    out_dir=out_dir,
                    max_sequences=max_sequences,
                    override_num_frames=override_num_frames,
                    override_fps=override_fps,
                    show_progress=bool(body.show_progress),
                    localized_components_override=localized_for_run,
                    cancel_event=ctx.cancel_event,
                    progress_callback=ctx.report,
                )
                return {
                    "result": {
                        "mode": body.mode,
                        "output_dir": out_dir,
                        "manifest": manifest,
                    }
                }

            # static_triplet: export 3 full sequence runs from the same setup:
            # rigid (no localized deformation), negative extreme, positive extreme.
            if cfg.scene_type != "tube" or cfg.tube is None:
                raise RuntimeError("mode=static_triplet requires scene_type='tube'")

            static_cfg_payload = _freeze_global_waves_for_static(canonical_config_dict(cfg))
            rigid_out = str(Path(out_dir) / "rigid")
            negative_out = str(Path(out_dir) / "negative")
            positive_out = str(Path(out_dir) / "positive")

            if localized_override is not None and len(localized_override) > 0:
                variant_specs: list[tuple[str, str, dict[str, Any], list[LocalizedWaveComponent] | None]] = [
                    ("rigid", rigid_out, static_cfg_payload, []),
                    ("negative", negative_out, static_cfg_payload, _static_extreme_components(localized_override, sign=-1)),
                    ("positive", positive_out, static_cfg_payload, _static_extreme_components(localized_override, sign=1)),
                ]
            else:
                areas = ((static_cfg_payload.get("tube") or {}).get("deformation") or {}).get("areas") or []
                if not areas:
                    raise RuntimeError("mode=static_triplet requires at least one localized area")

                def _area_variant(sign: int) -> dict[str, Any]:
                    payload = json.loads(json.dumps(static_cfg_payload))
                    phase = math.pi / 2.0 if sign >= 0 else -math.pi / 2.0
                    out_areas = []
                    for item in areas:
                        area = dict(item)
                        area["frequency_hz"] = 0.0
                        area["phase_rad"] = phase
                        area["mode"] = "radial"
                        area["spatial_cycles_s"] = 0.0
                        area["spatial_cycles_theta"] = 0.0
                        area["flow_dir_s"] = 0.0
                        area["flow_dir_theta"] = 0.0
                        out_areas.append(area)
                    payload["tube"]["deformation"]["areas"] = out_areas
                    return payload

                rigid_payload = json.loads(json.dumps(static_cfg_payload))
                rigid_payload["tube"]["deformation"]["areas"] = []
                variant_specs = [
                    ("rigid", rigid_out, rigid_payload, None),
                    ("negative", negative_out, _area_variant(-1), None),
                    ("positive", positive_out, _area_variant(1), None),
                ]

            manifests: dict[str, Any] = {}
            for idx, (name, variant_out, variant_payload, variant_components) in enumerate(variant_specs):
                progress_offset = float(idx) / float(len(variant_specs))

                def _variant_progress(stage: str, percent: float, message: str) -> None:
                    mapped = progress_offset + float(percent) / float(len(variant_specs))
                    ctx.report(
                        stage=f"static_triplet.{name}.{stage}",
                        percent=mapped,
                        message=f"[{name}] {message}",
                    )

                static_cfg = load_config_dict(variant_payload)
                manifests[name] = generate_dataset(
                    cfg=static_cfg,
                    out_dir=variant_out,
                    max_sequences=1,
                    override_num_frames=override_num_frames,
                    override_fps=override_fps,
                    show_progress=bool(body.show_progress),
                    localized_components_override=variant_components,
                    flatten_single_sequence=True,
                    cancel_event=ctx.cancel_event,
                    progress_callback=_variant_progress,
                )

            return {
                "result": {
                    "mode": body.mode,
                    "output_dir": out_dir,
                    "variants": {
                        "rigid": rigid_out,
                        "negative": negative_out,
                        "positive": positive_out,
                    },
                    "manifests": manifests,
                }
            }

        try:
            job_id = jobs.create_job(kind="generate", runner=_runner)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    async def job_status(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc

    @app.get("/v1/jobs/{job_id}/frames/{frame_idx}")
    async def preview_frame(job_id: str, frame_idx: int) -> Response:
        try:
            frame = jobs.get_frame(job_id, frame_idx)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame index: {frame_idx}") from exc
        return Response(content=frame, media_type="image/png")

    @app.get("/v1/jobs/{job_id}/result")
    async def job_result(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get_result(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app
