from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

from .types import (
    CameraIntrinsics,
    CameraTrajectoryConfig,
    DatasetConfig,
    DifficultyPreset,
    OutputToggles,
    PlaneSceneConfig,
    PhotometricConfig,
    SequenceConfig,
    TextureConfig,
    TubeAreaMode,
    TubeAxisConfig,
    TubeAxisSection,
    TubeCameraConfig,
    TubeDeformationArea,
    TubeDeformationConfig,
    TubeDeformationDirection,
    TubeFEAConfig,
    TubeDimensionsConfig,
    TubeSurfaceConfig,
    TubeSceneConfig,
    TubeTurnDirection,
    ValidationConfig,
)


def _require_keys(data: Dict[str, Any], keys: list[str], ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing keys in {ctx}: {missing}")


def _normalize_fea_load_distribution(value: object) -> str:
    """Return a supported FEA load-distribution mode, defaulting to "point"."""
    if value is None:
        return "point"
    text = str(value).strip().lower()
    if text in {"point", "node", "point_load"}:
        return "point"
    if text in {"gaussian_patch", "gaussian", "patch", "distributed"}:
        return "gaussian_patch"
    raise ValueError(
        f"Unsupported fea.load_distribution: {value!r}. Use 'point' or 'gaussian_patch'."
    )


def _normalize_flow_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"linear", "cyclic"}:
        raise ValueError(f"Unsupported tube.deformation_flow_mode: {value!r}")
    return mode


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(default)


def _section_arc_length(item: TubeAxisSection) -> float:
    if item.kind == "turn":
        return abs(math.radians(float(item.angle_deg))) * max(float(item.radius), 1e-6)
    return max(float(item.length), 1e-6)


def _axis_total_length(axis: TubeAxisConfig) -> float:
    return float(sum(_section_arc_length(s) for s in axis.sections))


def _parse_tube_axis(raw: Dict[str, Any], legacy_length: float) -> TubeAxisConfig:
    axis_raw = raw.get("axis")
    sections_raw = axis_raw.get("sections") if isinstance(axis_raw, dict) else None
    sections: list[TubeAxisSection] = []
    if isinstance(sections_raw, list):
        for idx, item in enumerate(sections_raw):
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "straight")).strip().lower()
            if kind == "turn":
                direction = str(item.get("direction", "right")).strip().lower()
                if direction not in {"up", "down", "left", "right"}:
                    raise ValueError(f"Unsupported tube.axis.sections[{idx}].direction: {direction!r}")
                sections.append(
                    TubeAxisSection(
                        kind="turn",
                        direction=direction,  # type: ignore[arg-type]
                        angle_deg=_as_float(item.get("angle_deg"), 45.0),
                        radius=max(_as_float(item.get("radius"), 5.0), 1e-6),
                        length=max(_as_float(item.get("length"), 0.0), 0.0),
                    )
                )
            elif kind == "straight":
                sections.append(
                    TubeAxisSection(
                        kind="straight",
                        length=max(_as_float(item.get("length"), legacy_length), 1e-6),
                    )
                )
            else:
                raise ValueError(f"Unsupported tube.axis.sections[{idx}].kind: {kind!r}")
    if sections:
        return TubeAxisConfig(sections=sections)
    return TubeAxisConfig(sections=_legacy_axis_sections(raw, legacy_length))


def _legacy_axis_sections(raw: Dict[str, Any], legacy_length: float) -> list[TubeAxisSection]:
    length = max(_as_float(raw.get("length"), legacy_length), 1e-6)
    amp_x = _as_float(raw.get("curve_amplitude_x"), 0.0)
    amp_z = _as_float(raw.get("curve_amplitude_z"), 0.0)
    freq_x = abs(_as_float(raw.get("curve_frequency_x"), 0.85))
    freq_z = abs(_as_float(raw.get("curve_frequency_z"), 0.7))
    turns: list[tuple[TubeTurnDirection, float, float]] = []
    if abs(amp_x) > 1e-6:
        angle = min(120.0, max(12.0, 24.0 + 14.0 * freq_x + 8.0 * abs(amp_x)))
        turns.append(("right" if amp_x >= 0 else "left", angle, abs(amp_x)))
    if abs(amp_z) > 1e-6:
        angle = min(120.0, max(12.0, 24.0 + 14.0 * freq_z + 8.0 * abs(amp_z)))
        turns.append(("up" if amp_z >= 0 else "down", angle, abs(amp_z)))
    if not turns:
        return [TubeAxisSection(kind="straight", length=length)]

    turn_arc_total = min(length * 0.45, max(length * 0.18, length * 0.12 * len(turns)))
    straight_total = max(length - turn_arc_total, length * 0.25)
    straight_each = straight_total / float(len(turns) + 1)
    turn_arc_each = max((length - straight_total) / float(len(turns)), 1e-6)
    out: list[TubeAxisSection] = []
    for direction, angle, _amp in turns:
        out.append(TubeAxisSection(kind="straight", length=straight_each))
        radius = turn_arc_each / max(abs(math.radians(angle)), 1e-6)
        out.append(TubeAxisSection(kind="turn", direction=direction, angle_deg=angle, radius=radius))
    out.append(TubeAxisSection(kind="straight", length=straight_each))
    return out


def _parse_tube_deformation_area(raw: Dict[str, Any], index: int) -> TubeDeformationArea:
    direction_raw = raw.get("direction") if isinstance(raw.get("direction"), dict) else {}
    kind = str(raw.get("kind", "localized")).strip().lower()
    if kind not in {"localized", "fea_c3d6"}:
        raise ValueError(f"Unsupported tube.deformation.areas[{index}].kind: {kind!r}")
    mode = str(raw.get("mode", "radial")).strip().lower()
    if mode not in {"radial", "travelling"}:
        raise ValueError(f"Unsupported tube.deformation.areas[{index}].mode: {mode!r}")
    omega = _as_float(raw.get("omega"), 0.0)
    legacy_amp = _as_float(raw.get("amp"), 0.2)
    return TubeDeformationArea(
        id=str(raw.get("id") or f"area_{index + 1}"),
        enabled=_as_bool(raw.get("enabled"), True),
        center_s=float(min(1.0, max(0.0, _as_float(raw.get("center_s"), 0.5)))),
        center_theta=float(_as_float(raw.get("center_theta"), 0.2) % 1.0),
        sigma_s=max(_as_float(raw.get("sigma_s"), 0.1), 1e-4),
        sigma_theta=max(_as_float(raw.get("sigma_theta"), 0.1), 1e-4),
        kind=kind,  # type: ignore[arg-type]
        direction=TubeDeformationDirection(
            radial=_as_float(direction_raw.get("radial") if isinstance(direction_raw, dict) else None, 1.0),
            tangent=_as_float(direction_raw.get("tangent") if isinstance(direction_raw, dict) else None, 0.0),
            circumferential=_as_float(
                direction_raw.get("circumferential") if isinstance(direction_raw, dict) else None,
                0.0,
            ),
        ),
        amplitude_out=abs(_as_float(raw.get("amplitude_out", raw.get("amp_out")), legacy_amp)),
        amplitude_in=abs(_as_float(raw.get("amplitude_in", raw.get("amp_in")), legacy_amp)),
        force_n=max(_as_float(raw.get("force_n"), 800.0), 0.0),
        frequency_hz=max(_as_float(raw.get("frequency_hz"), omega / (2.0 * math.pi) if omega else 0.2), 0.0),
        phase_rad=_as_float(raw.get("phase_rad", raw.get("phase")), 0.0),
        mode=mode,  # type: ignore[arg-type]
        spatial_cycles_s=_as_float(raw.get("spatial_cycles_s", raw.get("fy")), 0.0),
        spatial_cycles_theta=_as_float(raw.get("spatial_cycles_theta", raw.get("fx")), 0.0),
        flow_dir_s=_as_float(raw.get("flow_dir_s"), 0.0) if mode == "travelling" else 0.0,
        flow_dir_theta=_as_float(raw.get("flow_dir_theta"), 0.0) if mode == "travelling" else 0.0,
    )


def _legacy_deformation_areas(raw: Dict[str, Any]) -> list[TubeDeformationArea]:
    count = max(0, _as_int(raw.get("localized_deform_count"), 0))
    if count == 0:
        return []
    s_min = float(min(1.0, max(0.0, _as_float(raw.get("localized_longitudinal_min"), 0.5))))
    s_max = float(min(1.0, max(0.0, _as_float(raw.get("localized_longitudinal_max"), s_min))))
    if s_max < s_min:
        s_min, s_max = s_max, s_min
    theta_center = _as_float(raw.get("localized_theta_center"), 0.2) % 1.0
    theta_jitter = max(0.0, _as_float(raw.get("localized_theta_jitter"), 0.0))
    sigma_s = max(_as_float(raw.get("localized_sigma_long_min"), 0.1), 1e-4)
    sigma_theta = max(_as_float(raw.get("localized_sigma_theta_min"), 0.1), 1e-4)
    amp = max(0.0, 0.15 * _as_float(raw.get("localized_amp_scale"), 1.0))
    out: list[TubeDeformationArea] = []
    for i in range(count):
        offset = 0.0 if count <= 1 else (i - (count - 1) * 0.5) / max(count - 1, 1)
        out.append(
            TubeDeformationArea(
                id=f"area_{i + 1}",
                enabled=True,
                center_s=float(min(1.0, max(0.0, 0.5 * (s_min + s_max) + 0.08 * offset))),
                center_theta=float((theta_center + theta_jitter * offset) % 1.0),
                sigma_s=sigma_s,
                sigma_theta=sigma_theta,
                kind="localized",
                amplitude_out=amp,
                amplitude_in=amp,
                force_n=800.0,
                frequency_hz=0.2,
                phase_rad=0.0,
                mode="radial",
            )
        )
    return out


def _parse_tube_deformation(raw: Dict[str, Any]) -> TubeDeformationConfig:
    deformation_raw = raw.get("deformation") if isinstance(raw.get("deformation"), dict) else {}
    fea_raw = deformation_raw.get("fea") if isinstance(deformation_raw.get("fea"), dict) else {}
    areas_raw = deformation_raw.get("areas") if isinstance(deformation_raw, dict) else None
    areas: list[TubeDeformationArea] = []
    if isinstance(areas_raw, list):
        for idx, item in enumerate(areas_raw):
            if isinstance(item, dict):
                areas.append(_parse_tube_deformation_area(item, idx))
    else:
        embedded = raw.get("localized_components")
        if isinstance(embedded, list):
            for idx, item in enumerate(embedded):
                if isinstance(item, dict):
                    areas.append(_parse_tube_deformation_area(item, idx))
        else:
            areas = _legacy_deformation_areas(raw)

    flow_mode = (
        deformation_raw.get("flow_mode")
        if isinstance(deformation_raw, dict) and deformation_raw.get("flow_mode") is not None
        else raw.get("deformation_flow_mode", "cyclic")
    )
    flow_cycles = (
        deformation_raw.get("flow_cycles")
        if isinstance(deformation_raw, dict) and deformation_raw.get("flow_cycles") is not None
        else raw.get("deformation_flow_cycles", 1.0)
    )
    tint_bgr = (
        deformation_raw.get("localized_tint_bgr")
        if isinstance(deformation_raw, dict) and deformation_raw.get("localized_tint_bgr") is not None
        else raw.get("localized_tint_bgr", (48, 185, 255))
    )
    tint_list = list(tint_bgr) if isinstance(tint_bgr, (list, tuple)) else [48, 185, 255]
    while len(tint_list) < 3:
        tint_list.append(0)
    return TubeDeformationConfig(
        areas=areas,
        fea=TubeFEAConfig(
            young_modulus_pa=max(_as_float(fea_raw.get("young_modulus_pa"), 1.0e6), 1e-12),
            poisson_ratio=float(min(0.49 - 1e-9, max(-0.99 + 1e-9, _as_float(fea_raw.get("poisson_ratio"), 0.30)))),
            axial_elements=max(_as_int(fea_raw.get("axial_elements"), 8), 2),
            theta_elements=max(_as_int(fea_raw.get("theta_elements"), 16), 6),
            load_distribution=_normalize_fea_load_distribution(fea_raw.get("load_distribution")),
            load_sigma_cutoff=max(_as_float(fea_raw.get("load_sigma_cutoff"), 3.0), 0.5),
        ),
        flow_mode=_normalize_flow_mode(flow_mode),  # type: ignore[arg-type]
        flow_cycles=_as_float(flow_cycles, 1.0),
        localized_tint_enabled=_as_bool(
            deformation_raw.get("localized_tint_enabled")
            if isinstance(deformation_raw, dict) and deformation_raw.get("localized_tint_enabled") is not None
            else raw.get("localized_tint_enabled"),
            False,
        ),
        localized_tint_bgr=(int(tint_list[0]), int(tint_list[1]), int(tint_list[2])),
        localized_tint_strength=_as_float(
            deformation_raw.get("localized_tint_strength")
            if isinstance(deformation_raw, dict) and deformation_raw.get("localized_tint_strength") is not None
            else raw.get("localized_tint_strength"),
            0.18,
        ),
    )


def _tube_scene_config_from_raw(raw: Dict[str, Any]) -> TubeSceneConfig:
    legacy_length = max(_as_float(raw.get("length"), 42.0), 1e-6)
    axis = _parse_tube_axis(raw, legacy_length=legacy_length)
    dimensions_raw = raw.get("dimensions") if isinstance(raw.get("dimensions"), dict) else {}
    surface_raw = raw.get("surface") if isinstance(raw.get("surface"), dict) else {}
    camera_raw = raw.get("camera") if isinstance(raw.get("camera"), dict) else {}
    deformation = _parse_tube_deformation(raw)
    dimensions = TubeDimensionsConfig(
        radial_samples=max(_as_int(dimensions_raw.get("radial_samples") if isinstance(dimensions_raw, dict) else None, _as_int(raw.get("radial_samples"), 64)), 3),
        longitudinal_samples=max(
            _as_int(dimensions_raw.get("longitudinal_samples") if isinstance(dimensions_raw, dict) else None, _as_int(raw.get("longitudinal_samples"), 128)),
            2,
        ),
        base_radius=max(
            _as_float(dimensions_raw.get("base_radius") if isinstance(dimensions_raw, dict) else None, _as_float(raw.get("base_radius"), 2.4)),
            1e-6,
        ),
        min_radius=max(
            _as_float(dimensions_raw.get("min_radius") if isinstance(dimensions_raw, dict) else None, _as_float(raw.get("min_radius"), 0.95)),
            0.0,
        ),
        shell_thickness=max(
            _as_float(
                dimensions_raw.get("shell_thickness") if isinstance(dimensions_raw, dict) else None,
                _as_float(raw.get("shell_thickness"), 0.25),
            ),
            0.0,
        ),
        cap_ends=_as_bool(
            dimensions_raw.get("cap_ends") if isinstance(dimensions_raw, dict) else None,
            False,
        ),
    )
    surface = TubeSurfaceConfig(
        bump_count=max(_as_int(surface_raw.get("bump_count") if isinstance(surface_raw, dict) else None, _as_int(raw.get("bump_count"), 0)), 0),
        bump_amp_min=_as_float(surface_raw.get("bump_amp_min") if isinstance(surface_raw, dict) else None, _as_float(raw.get("bump_amp_min"), 0.0)),
        bump_amp_max=_as_float(surface_raw.get("bump_amp_max") if isinstance(surface_raw, dict) else None, _as_float(raw.get("bump_amp_max"), 0.0)),
        bump_sigma_long=max(
            _as_float(surface_raw.get("bump_sigma_long") if isinstance(surface_raw, dict) else None, _as_float(raw.get("bump_sigma_long"), 0.055)),
            1e-4,
        ),
        bump_sigma_theta=max(
            _as_float(surface_raw.get("bump_sigma_theta") if isinstance(surface_raw, dict) else None, _as_float(raw.get("bump_sigma_theta"), 0.06)),
            1e-4,
        ),
    )
    camera_lookahead = _as_float(
        camera_raw.get("lookahead_distance") if isinstance(camera_raw, dict) else None,
        _as_float(raw.get("lookahead_distance"), 2.4),
    )
    camera = TubeCameraConfig(
        offset_ratio=_as_float(camera_raw.get("offset_ratio") if isinstance(camera_raw, dict) else None, _as_float(raw.get("camera_offset_ratio"), 0.0)),
        wobble_ratio=_as_float(camera_raw.get("wobble_ratio") if isinstance(camera_raw, dict) else None, _as_float(raw.get("camera_wobble_ratio"), 0.0)),
        wobble_hz=_as_float(camera_raw.get("wobble_hz") if isinstance(camera_raw, dict) else None, _as_float(raw.get("camera_wobble_hz"), 0.45)),
        lookahead_distance=camera_lookahead,
        end_margin_distance=max(
            _as_float(
                camera_raw.get("end_margin_distance") if isinstance(camera_raw, dict) else None,
                0.0,
            ),
            0.0,
        ),
    )
    length = _axis_total_length(axis)
    return TubeSceneConfig(
        axis=axis,
        dimensions=dimensions,
        surface=surface,
        camera=camera,
        deformation=deformation,
        length=length,
        radial_samples=dimensions.radial_samples,
        longitudinal_samples=dimensions.longitudinal_samples,
        base_radius=dimensions.base_radius,
        min_radius=dimensions.min_radius,
        shell_thickness=dimensions.shell_thickness,
        bump_count=surface.bump_count,
        bump_amp_min=surface.bump_amp_min,
        bump_amp_max=surface.bump_amp_max,
        bump_sigma_long=surface.bump_sigma_long,
        bump_sigma_theta=surface.bump_sigma_theta,
        curve_amplitude_x=_as_float(raw.get("curve_amplitude_x"), 0.0),
        curve_amplitude_z=_as_float(raw.get("curve_amplitude_z"), 0.0),
        curve_frequency_x=_as_float(raw.get("curve_frequency_x"), 0.0),
        curve_frequency_z=_as_float(raw.get("curve_frequency_z"), 0.0),
        camera_offset_ratio=camera.offset_ratio,
        camera_wobble_ratio=camera.wobble_ratio,
        camera_wobble_hz=camera.wobble_hz,
        lookahead_distance=camera.lookahead_distance,
        localized_deform_count=len(deformation.areas),
        localized_longitudinal_min=min((a.center_s for a in deformation.areas), default=0.15),
        localized_longitudinal_max=max((a.center_s for a in deformation.areas), default=0.85),
        localized_theta_center=deformation.areas[0].center_theta if deformation.areas else 0.25,
        localized_theta_jitter=0.0,
        localized_sigma_long_min=min((a.sigma_s for a in deformation.areas), default=0.04),
        localized_sigma_long_max=max((a.sigma_s for a in deformation.areas), default=0.10),
        localized_sigma_theta_min=min((a.sigma_theta for a in deformation.areas), default=0.05),
        localized_sigma_theta_max=max((a.sigma_theta for a in deformation.areas), default=0.12),
        localized_amp_scale=1.0,
        localized_negative_ratio=0.0,
        localized_tint_enabled=deformation.localized_tint_enabled,
        localized_tint_bgr=deformation.localized_tint_bgr,
        localized_tint_strength=deformation.localized_tint_strength,
        deformation_flow_mode=deformation.flow_mode,
        deformation_flow_cycles=deformation.flow_cycles,
    )


def tube_config_warnings(raw: Dict[str, Any]) -> list[str]:
    if "tube" not in raw or not isinstance(raw.get("tube"), dict):
        return []
    tube = raw["tube"]
    if isinstance(tube.get("axis"), dict) and isinstance(tube.get("dimensions"), dict):
        return []
    return ["Migrated legacy tube config to Tube V2 axis/dimensions/surface/camera/deformation schema."]


def _default_plane_scene_config() -> PlaneSceneConfig:
    return PlaneSceneConfig(
        landmark_count=0,
        landmark_height_min=1.8,
        landmark_height_max=4.8,
        landmark_sigma_min=1.0,
        landmark_sigma_max=3.5,
        rigid_ratio=0.0,
        rigid_include_landmarks=True,
        rigid_edge_softness=1,
        rigid_colorize=False,
        rigid_tint_bgr=[40, 180, 255],
        rigid_tint_strength=0.35,
        flip_vertical_view=False,
    )


def load_config(path: str | Path) -> DatasetConfig:
    path = Path(path)
    raw = json.loads(path.read_text())
    return load_config_dict(raw)


def load_config_dict(raw: Dict[str, Any]) -> DatasetConfig:

    _require_keys(
        raw,
        [
            "version",
            "seed",
            "sequence_count",
            "default_preset_cycle",
            "intrinsics",
            "sequence",
            "camera",
            "texture",
            "photometric",
            "output",
            "validation",
            "difficulty_presets",
        ],
        "root",
    )

    intr = raw["intrinsics"]
    seq = raw["sequence"]
    cam = raw["camera"]
    tex = raw["texture"]
    photo = raw["photometric"]
    out = raw["output"]
    val = dict(raw["validation"])
    val.setdefault("min_orb_features_frame", 1.0)
    presets_raw = raw["difficulty_presets"]

    scene_type = str(raw.get("scene_type", "plane")).strip().lower()
    if scene_type not in {"plane", "tube"}:
        raise ValueError(f"Unsupported scene_type: {scene_type}")

    plane_cfg = _default_plane_scene_config()
    if "plane" in raw:
        p_raw = dict(raw["plane"])
        d = _default_plane_scene_config()
        p_raw.setdefault("landmark_count", d.landmark_count)
        p_raw.setdefault("landmark_height_min", d.landmark_height_min)
        p_raw.setdefault("landmark_height_max", d.landmark_height_max)
        p_raw.setdefault("landmark_sigma_min", d.landmark_sigma_min)
        p_raw.setdefault("landmark_sigma_max", d.landmark_sigma_max)
        p_raw.setdefault("rigid_ratio", d.rigid_ratio)
        p_raw.setdefault("rigid_include_landmarks", d.rigid_include_landmarks)
        p_raw.setdefault("rigid_edge_softness", d.rigid_edge_softness)
        p_raw.setdefault("rigid_colorize", d.rigid_colorize)
        p_raw.setdefault("rigid_tint_bgr", d.rigid_tint_bgr)
        p_raw.setdefault("rigid_tint_strength", d.rigid_tint_strength)
        p_raw.setdefault("flip_vertical_view", d.flip_vertical_view)
        plane_cfg = PlaneSceneConfig(**p_raw)

    tube_cfg = None
    if scene_type == "tube":
        if "tube" not in raw:
            raise ValueError("Missing 'tube' section for scene_type='tube'")
        t_raw = dict(raw["tube"])
        if "localized_components" in raw and "localized_components" not in t_raw:
            t_raw["localized_components"] = raw["localized_components"]
        tube_cfg = _tube_scene_config_from_raw(t_raw)

    presets: Dict[str, DifficultyPreset] = {}
    for name, item in presets_raw.items():
        p = dict(item)
        p.setdefault("curve_scale", 1.0)
        presets[name] = DifficultyPreset(name=name, **p)

    cfg = DatasetConfig(
        version=raw["version"],
        seed=int(raw["seed"]),
        scene_type=scene_type,
        sequence_count={k: int(v) for k, v in raw["sequence_count"].items()},
        default_preset_cycle=[str(x) for x in raw["default_preset_cycle"]],
        intrinsics=CameraIntrinsics(**intr),
        sequence=SequenceConfig(**seq),
        camera=CameraTrajectoryConfig(**cam),
        texture=TextureConfig(**tex),
        photometric=PhotometricConfig(**photo),
        output=OutputToggles(**out),
        validation=ValidationConfig(**val),
        plane=plane_cfg,
        tube=tube_cfg,
        difficulty_presets=presets,
    )

    for name in cfg.default_preset_cycle:
        if name not in cfg.difficulty_presets:
            raise ValueError(f"Unknown preset in default_preset_cycle: {name}")

    for split in ("train", "val", "test"):
        if split not in cfg.sequence_count:
            raise ValueError(f"sequence_count missing '{split}'")

    if cfg.scene_type == "tube" and cfg.tube is None:
        raise ValueError("scene_type='tube' requires tube config")

    return cfg


def canonical_config_dict(cfg: DatasetConfig) -> Dict[str, Any]:
    presets = {
        name: {
            "wave_count": p.wave_count,
            "amp_min": p.amp_min,
            "amp_max": p.amp_max,
            "freq_min": p.freq_min,
            "freq_max": p.freq_max,
            "omega_min": p.omega_min,
            "omega_max": p.omega_max,
            "drift_scale": p.drift_scale,
            "jitter_scale": p.jitter_scale,
            "curve_scale": p.curve_scale,
        }
        for name, p in sorted(cfg.difficulty_presets.items())
    }
    payload = {
        "version": cfg.version,
        "seed": cfg.seed,
        "scene_type": cfg.scene_type,
        "sequence_count": cfg.sequence_count,
        "default_preset_cycle": cfg.default_preset_cycle,
        "intrinsics": {
            "width": cfg.intrinsics.width,
            "height": cfg.intrinsics.height,
            "fx": cfg.intrinsics.fx,
            "fy": cfg.intrinsics.fy,
            "cx": cfg.intrinsics.cx,
            "cy": cfg.intrinsics.cy,
        },
        "sequence": {
            "num_frames": cfg.sequence.num_frames,
            "fps": cfg.sequence.fps,
            "mesh_nx": cfg.sequence.mesh_nx,
            "mesh_ny": cfg.sequence.mesh_ny,
            "x_min": cfg.sequence.x_min,
            "x_max": cfg.sequence.x_max,
            "y_min": cfg.sequence.y_min,
            "y_max": cfg.sequence.y_max,
        },
        "camera": {
            "base_x": cfg.camera.base_x,
            "base_y": cfg.camera.base_y,
            "base_z": cfg.camera.base_z,
            "forward_drift_per_s": cfg.camera.forward_drift_per_s,
            "orbit_radius": cfg.camera.orbit_radius,
            "orbit_frequency_hz": cfg.camera.orbit_frequency_hz,
            "lateral_amplitude": cfg.camera.lateral_amplitude,
            "lateral_frequency_hz": cfg.camera.lateral_frequency_hz,
            "lookahead_distance": cfg.camera.lookahead_distance,
            "lookat_height": cfg.camera.lookat_height,
            "roll_jitter_deg": cfg.camera.roll_jitter_deg,
            "pitch_jitter_deg": cfg.camera.pitch_jitter_deg,
            "yaw_jitter_deg": cfg.camera.yaw_jitter_deg,
        },
        "texture": {
            "width": cfg.texture.width,
            "height": cfg.texture.height,
            "min_orb_features": cfg.texture.min_orb_features,
            "max_attempts": cfg.texture.max_attempts,
        },
        "photometric": {
            "gaussian_sigma": cfg.photometric.gaussian_sigma,
            "gain_jitter": cfg.photometric.gain_jitter,
        },
        "output": {
            "write_frames": cfg.output.write_frames,
            "write_mp4": cfg.output.write_mp4,
            "write_mesh_state": cfg.output.write_mesh_state,
        },
        "validation": {
            "min_orb_features_mean": cfg.validation.min_orb_features_mean,
            "min_orb_features_frame": cfg.validation.min_orb_features_frame,
            "min_frame_motion": cfg.validation.min_frame_motion,
            "min_camera_clearance": cfg.validation.min_camera_clearance,
            "max_abs_mesh_z": cfg.validation.max_abs_mesh_z,
        },
        "difficulty_presets": presets,
    }

    if cfg.plane is not None:
        payload["plane"] = {
            "landmark_count": cfg.plane.landmark_count,
            "landmark_height_min": cfg.plane.landmark_height_min,
            "landmark_height_max": cfg.plane.landmark_height_max,
            "landmark_sigma_min": cfg.plane.landmark_sigma_min,
            "landmark_sigma_max": cfg.plane.landmark_sigma_max,
            "rigid_ratio": cfg.plane.rigid_ratio,
            "rigid_include_landmarks": cfg.plane.rigid_include_landmarks,
            "rigid_edge_softness": cfg.plane.rigid_edge_softness,
            "rigid_colorize": cfg.plane.rigid_colorize,
            "rigid_tint_bgr": cfg.plane.rigid_tint_bgr,
            "rigid_tint_strength": cfg.plane.rigid_tint_strength,
            "flip_vertical_view": cfg.plane.flip_vertical_view,
        }

    if cfg.tube is not None:
        tube = cfg.tube
        payload["tube"] = {
            "axis": {
                "sections": [
                    {
                        "kind": section.kind,
                        **(
                            {"length": section.length}
                            if section.kind == "straight"
                            else {
                                "direction": section.direction,
                                "angle_deg": section.angle_deg,
                                "radius": section.radius,
                            }
                        ),
                    }
                    for section in tube.axis.sections
                ],
            },
            "dimensions": {
                "radial_samples": tube.radial_samples,
                "longitudinal_samples": tube.longitudinal_samples,
                "base_radius": tube.base_radius,
                "min_radius": tube.min_radius,
                "shell_thickness": tube.shell_thickness,
                "cap_ends": tube.dimensions.cap_ends,
            },
            "surface": {
                "bump_count": tube.bump_count,
                "bump_amp_min": tube.bump_amp_min,
                "bump_amp_max": tube.bump_amp_max,
                "bump_sigma_long": tube.bump_sigma_long,
                "bump_sigma_theta": tube.bump_sigma_theta,
            },
            "camera": {
                "offset_ratio": tube.camera_offset_ratio,
                "wobble_ratio": tube.camera_wobble_ratio,
                "wobble_hz": tube.camera_wobble_hz,
                "lookahead_distance": tube.lookahead_distance,
                "end_margin_distance": tube.camera.end_margin_distance,
            },
            "deformation": {
                "flow_mode": tube.deformation.flow_mode,
                "flow_cycles": tube.deformation.flow_cycles,
                "fea": {
                    "young_modulus_pa": tube.deformation.fea.young_modulus_pa,
                    "poisson_ratio": tube.deformation.fea.poisson_ratio,
                    "axial_elements": tube.deformation.fea.axial_elements,
                    "theta_elements": tube.deformation.fea.theta_elements,
                    "load_distribution": tube.deformation.fea.load_distribution,
                    "load_sigma_cutoff": tube.deformation.fea.load_sigma_cutoff,
                },
                "localized_tint_enabled": tube.deformation.localized_tint_enabled,
                "localized_tint_bgr": list(tube.deformation.localized_tint_bgr),
                "localized_tint_strength": tube.deformation.localized_tint_strength,
                "areas": [
                    {
                        "id": area.id,
                        "enabled": area.enabled,
                        "kind": area.kind,
                        "center_s": area.center_s,
                        "center_theta": area.center_theta,
                        "sigma_s": area.sigma_s,
                        "sigma_theta": area.sigma_theta,
                        "direction": {
                            "radial": area.direction.radial,
                            "tangent": area.direction.tangent,
                            "circumferential": area.direction.circumferential,
                        },
                        "amplitude_out": area.amplitude_out,
                        "amplitude_in": area.amplitude_in,
                        "force_n": area.force_n,
                        "frequency_hz": area.frequency_hz,
                        "phase_rad": area.phase_rad,
                        "mode": area.mode,
                        "spatial_cycles_s": area.spatial_cycles_s,
                        "spatial_cycles_theta": area.spatial_cycles_theta,
                        "flow_dir_s": area.flow_dir_s,
                        "flow_dir_theta": area.flow_dir_theta,
                    }
                    for area in tube.deformation.areas
                ],
            },
        }

    return payload


def config_hash(cfg: DatasetConfig) -> str:
    payload = json.dumps(canonical_config_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
