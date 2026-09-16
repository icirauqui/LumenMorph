from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deformation import compute_localized_displacement, deformed_z
from .types import CameraTrajectoryConfig, DifficultyPreset, LocalizedWaveComponent, TubeDeformationArea, TubeSceneConfig, WaveComponent


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


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _quat_from_rot(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _wrapped_distance(a: np.ndarray, b: float) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


@dataclass(frozen=True)
class TubeStaticGeometry:
    centerline: np.ndarray
    tangent: np.ndarray
    frame_right: np.ndarray
    frame_up: np.ndarray
    theta: np.ndarray
    s_norm: np.ndarray
    theta_norm_grid: np.ndarray
    s_norm_grid: np.ndarray
    base_radius: np.ndarray
    triangles: np.ndarray
    uv: np.ndarray


@dataclass(frozen=True)
class TubeCameraPose:
    t_wc: np.ndarray
    R_wc: np.ndarray
    q_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class TubeFEAAreaResponse:
    area_id: str
    displacement_grid: np.ndarray
    applied_s_norm: float
    applied_theta_norm: float
    applied_node: int
    applied_axial_index: int
    applied_theta_index: int
    force_vector_local: np.ndarray
    load_distribution: str = "point"
    load_node_count: int = 1


def _build_tube_triangles(n_long: int, n_rad: int, cap_ends: bool = False) -> np.ndarray:
    tris = []
    for j in range(n_long - 1):
        for i in range(n_rad):
            i_next = (i + 1) % n_rad
            idx0 = j * n_rad + i
            idx1 = j * n_rad + i_next
            idx2 = (j + 1) * n_rad + i
            idx3 = (j + 1) * n_rad + i_next
            tris.append((idx0, idx2, idx1))
            tris.append((idx1, idx2, idx3))
    if cap_ends:
        # Close each terminal ring without introducing extra displacement
        # vertices. The fan uses existing ring vertices so saved deformation
        # grids and FEA state retain their longitudinal-by-radial shape.
        for i in range(1, n_rad - 1):
            tris.append((0, i, i + 1))
        end = (n_long - 1) * n_rad
        for i in range(1, n_rad - 1):
            tris.append((end, end + i + 1, end + i))
    return np.asarray(tris, dtype=np.int32)


def _section_length(section) -> float:
    if section.kind == "turn":
        return abs(float(np.deg2rad(section.angle_deg))) * max(float(section.radius), 1e-6)
    return max(float(section.length), 1e-6)


def _advance_axis_state(
    pos: np.ndarray,
    tangent: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    section,
    distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dist = float(np.clip(distance, 0.0, _section_length(section)))
    if section.kind == "straight":
        return (pos + tangent * dist).astype(np.float32), tangent, right, up

    total_angle = abs(float(np.deg2rad(section.angle_deg)))
    sign = 1.0
    if section.direction in {"left", "down"}:
        sign = -1.0
    radius = max(float(section.radius), 1e-6)
    total_arc = max(total_angle * radius, 1e-6)
    alpha = total_angle * (dist / total_arc)
    c = float(np.cos(alpha))
    s = float(np.sin(alpha))

    if section.direction in {"left", "right"}:
        bend = right
        new_pos = pos + radius * (s * tangent + sign * (1.0 - c) * bend)
        new_tangent = _normalize(c * tangent + sign * s * bend)
        new_right = _normalize(-sign * s * tangent + c * bend)
        return new_pos.astype(np.float32), new_tangent.astype(np.float32), new_right.astype(np.float32), up

    bend = up
    new_pos = pos + radius * (s * tangent + sign * (1.0 - c) * bend)
    new_tangent = _normalize(c * tangent + sign * s * bend)
    new_up = _normalize(-sign * s * tangent + c * bend)
    return new_pos.astype(np.float32), new_tangent.astype(np.float32), right, new_up.astype(np.float32)


def _build_centerline(cfg: TubeSceneConfig, preset: DifficultyPreset, rng: np.random.Generator) -> np.ndarray:
    del preset, rng
    n_long = cfg.longitudinal_samples
    sections = cfg.axis.sections
    total_length = max(float(sum(_section_length(s) for s in sections)), 1e-6)

    starts: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]] = []
    pos = np.zeros(3, dtype=np.float32)
    tangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    arc_cursor = 0.0
    for section in sections:
        starts.append((arc_cursor, pos.copy(), tangent.copy(), right.copy(), up.copy(), section))
        pos, tangent, right, up = _advance_axis_state(pos, tangent, right, up, section, _section_length(section))
        arc_cursor += _section_length(section)

    samples = np.linspace(0.0, total_length, n_long, dtype=np.float32)
    centerline = np.zeros((n_long, 3), dtype=np.float32)
    section_idx = 0
    for i, dist in enumerate(samples):
        while section_idx < len(starts) - 1 and dist > starts[section_idx + 1][0]:
            section_idx += 1
        start_arc, start_pos, start_tangent, start_right, start_up, section = starts[section_idx]
        local_dist = float(dist - start_arc)
        p, _, _, _ = _advance_axis_state(start_pos, start_tangent, start_right, start_up, section, local_dist)
        centerline[i] = p
    return centerline.astype(np.float32)


def _build_frames(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent = np.gradient(centerline, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    tangent = (tangent / tangent_norm).astype(np.float32)

    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    frame_right = np.zeros_like(tangent)
    frame_up = np.zeros_like(tangent)

    for j in range(tangent.shape[0]):
        t = tangent[j]
        right = np.cross(up_world, t)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(alt_up, t)
        right = _normalize(right)
        up = _normalize(np.cross(t, right))
        frame_right[j] = right
        frame_up[j] = up

    return tangent.astype(np.float32), frame_right.astype(np.float32), frame_up.astype(np.float32)


def _build_base_radius(cfg: TubeSceneConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_long = cfg.longitudinal_samples
    n_rad = cfg.radial_samples

    s_norm = np.linspace(0.0, 1.0, n_long, dtype=np.float32)
    theta_norm = np.linspace(0.0, 1.0, n_rad, endpoint=False, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s_norm, theta_norm, indexing="ij")

    radius = np.full((n_long, n_rad), cfg.base_radius, dtype=np.float32)

    for _ in range(cfg.bump_count):
        c_s = float(rng.uniform(0.0, 1.0))
        c_t = float(rng.uniform(0.0, 1.0))
        amp = float(rng.uniform(cfg.bump_amp_min, cfg.bump_amp_max))
        if float(rng.uniform(0.0, 1.0)) < 0.45:
            amp = -0.85 * amp

        ds = s_grid - c_s
        dt = _wrapped_distance(theta_grid, c_t)
        bump = amp * np.exp(
            -0.5 * ((ds / max(cfg.bump_sigma_long, 1e-4)) ** 2 + (dt / max(cfg.bump_sigma_theta, 1e-4)) ** 2)
        )
        radius += bump.astype(np.float32)

    radius = np.maximum(radius, cfg.min_radius).astype(np.float32)
    return radius, s_grid.astype(np.float32), theta_grid.astype(np.float32)


def _tube_uv(n_long: int, n_rad: int) -> np.ndarray:
    s_norm = np.linspace(0.0, 1.0, n_long, dtype=np.float32)
    theta_norm = np.linspace(0.0, 1.0, n_rad, endpoint=False, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s_norm, theta_norm, indexing="ij")
    uv = np.stack([theta_grid, s_grid], axis=-1).reshape(-1, 2)
    return uv.astype(np.float32)


def build_tube_static_geometry(
    cfg: TubeSceneConfig,
    preset: DifficultyPreset,
    rng: np.random.Generator,
) -> TubeStaticGeometry:
    centerline = _build_centerline(cfg, preset, rng)
    tangent, frame_right, frame_up = _build_frames(centerline)
    base_radius, s_grid, theta_grid = _build_base_radius(cfg, rng)

    triangles = _build_tube_triangles(
        cfg.longitudinal_samples,
        cfg.radial_samples,
        cap_ends=cfg.dimensions.cap_ends,
    )
    uv = _tube_uv(cfg.longitudinal_samples, cfg.radial_samples)
    theta = np.linspace(0.0, 2.0 * np.pi, cfg.radial_samples, endpoint=False, dtype=np.float32)
    s_norm = np.linspace(0.0, 1.0, cfg.longitudinal_samples, dtype=np.float32)

    return TubeStaticGeometry(
        centerline=centerline,
        tangent=tangent,
        frame_right=frame_right,
        frame_up=frame_up,
        theta=theta,
        s_norm=s_norm,
        theta_norm_grid=theta_grid,
        s_norm_grid=s_grid,
        base_radius=base_radius,
        triangles=triangles,
        uv=uv,
    )


def tube_vertices_from_radius(geom: TubeStaticGeometry, radius: np.ndarray) -> np.ndarray:
    cos_t = np.cos(geom.theta)[None, :, None]
    sin_t = np.sin(geom.theta)[None, :, None]
    dirs = cos_t * geom.frame_right[:, None, :] + sin_t * geom.frame_up[:, None, :]
    verts = geom.centerline[:, None, :] + radius[:, :, None] * dirs
    return verts.reshape(-1, 3).astype(np.float32)


def _tube_basis_grids(geom: TubeStaticGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_t = np.cos(geom.theta)[None, :, None]
    sin_t = np.sin(geom.theta)[None, :, None]
    radial = cos_t * geom.frame_right[:, None, :] + sin_t * geom.frame_up[:, None, :]
    circum = -sin_t * geom.frame_right[:, None, :] + cos_t * geom.frame_up[:, None, :]
    tangent = np.broadcast_to(geom.tangent[:, None, :], radial.shape)
    return radial.astype(np.float32), tangent.astype(np.float32), circum.astype(np.float32)


def _area_flow_progress(t_norm: float | None, flow_mode: str, flow_cycles: float) -> float:
    tn = float(np.clip(0.0 if t_norm is None else t_norm, 0.0, 1.0))
    if str(flow_mode).strip().lower() == "linear":
        return tn
    return float(np.sin(2.0 * np.pi * max(float(flow_cycles), 0.0) * tn))


def _area_envelope(
    theta_grid: np.ndarray,
    s_grid: np.ndarray,
    area: TubeDeformationArea,
    *,
    flow_mode: str,
    flow_cycles: float,
    t_norm: float | None,
) -> np.ndarray:
    progress = 0.0 if area.mode == "radial" else _area_flow_progress(t_norm, flow_mode, flow_cycles)
    center_s = float(np.clip(area.center_s + area.flow_dir_s * progress, 0.0, 1.0))
    center_theta = float((area.center_theta + area.flow_dir_theta * progress) % 1.0)
    ds = s_grid - center_s
    dt = _wrapped_distance(theta_grid, center_theta)
    return np.exp(
        -0.5 * ((ds / max(area.sigma_s, 1e-4)) ** 2 + (dt / max(area.sigma_theta, 1e-4)) ** 2)
    ).astype(np.float32)


def _area_signal(
    theta_grid: np.ndarray,
    s_grid: np.ndarray,
    t_s: float,
    area: TubeDeformationArea,
) -> np.ndarray:
    if area.mode == "travelling":
        arg = (
            2.0 * np.pi * (area.spatial_cycles_theta * theta_grid + area.spatial_cycles_s * s_grid + area.frequency_hz * t_s)
            + area.phase_rad
        )
    else:
        arg = 2.0 * np.pi * area.frequency_hz * t_s + area.phase_rad
    wave = np.asarray(np.sin(arg), dtype=np.float32)
    return (
        float(abs(area.amplitude_out)) * np.maximum(wave, 0.0)
        + float(abs(area.amplitude_in)) * np.minimum(wave, 0.0)
    ).astype(np.float32)


def tube_areas_from_localized_components(components: list[LocalizedWaveComponent]) -> list[TubeDeformationArea]:
    areas: list[TubeDeformationArea] = []
    for idx, c in enumerate(components):
        amp_base = float(abs(c.amp))
        areas.append(
            TubeDeformationArea(
                id=f"legacy_{idx + 1}",
                enabled=True,
                center_s=float(c.center_s),
                center_theta=float(c.center_theta),
                sigma_s=float(c.sigma_s),
                sigma_theta=float(c.sigma_theta),
                kind="localized",
                amplitude_out=float(abs(c.amp_out if c.amp_out is not None else amp_base)),
                amplitude_in=float(abs(c.amp_in if c.amp_in is not None else amp_base)),
                force_n=800.0,
                frequency_hz=float(abs(c.omega) / (2.0 * np.pi)),
                phase_rad=float(c.phase),
                mode="radial" if c.transversal_only else "travelling",
                spatial_cycles_theta=float(c.fx),
                spatial_cycles_s=float(c.fy),
                flow_dir_s=0.0 if c.transversal_only else float(c.flow_dir_s),
                flow_dir_theta=0.0 if c.transversal_only else float(c.flow_dir_theta),
            )
        )
    return areas


def tube_deformation_weight(
    geom: TubeStaticGeometry,
    areas: list[TubeDeformationArea],
    *,
    flow_mode: str,
    flow_cycles: float,
    t_norm: float | None,
) -> np.ndarray:
    weight = np.zeros_like(geom.base_radius, dtype=np.float32)
    for area in areas:
        if not area.enabled:
            continue
        weight = np.maximum(
            weight,
            _area_envelope(
                geom.theta_norm_grid,
                geom.s_norm_grid,
                area,
                flow_mode=flow_mode,
                flow_cycles=flow_cycles,
                t_norm=t_norm,
            ),
        )
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def deformed_tube_mesh(
    geom: TubeStaticGeometry,
    tube_cfg: TubeSceneConfig,
    t_s: float,
    components: list[WaveComponent],
    localized_components: list[LocalizedWaveComponent] | None = None,
    fea_responses: list[TubeFEAAreaResponse] | None = None,
    t_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    zero = np.zeros_like(geom.base_radius, dtype=np.float32)
    radial_delta, global_amp, global_freq, phase_shift = deformed_z(
        zero,
        geom.theta_norm_grid,
        geom.s_norm_grid,
        t_s,
        components,
    )
    tangent_delta = np.zeros_like(radial_delta, dtype=np.float32)
    circum_delta = np.zeros_like(radial_delta, dtype=np.float32)
    areas = (
        tube_areas_from_localized_components(localized_components)
        if localized_components is not None
        else list(tube_cfg.deformation.areas)
    )
    for area in areas:
        if not area.enabled:
            continue
        envelope = _area_envelope(
            geom.theta_norm_grid,
            geom.s_norm_grid,
            area,
            flow_mode=tube_cfg.deformation.flow_mode,
            flow_cycles=tube_cfg.deformation.flow_cycles,
            t_norm=t_norm,
        )
        scalar = _area_signal(geom.theta_norm_grid, geom.s_norm_grid, t_s, area) * envelope
        d = np.array(
            [area.direction.radial, area.direction.tangent, area.direction.circumferential],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(d))
        if norm < 1e-8:
            d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            d = d / norm
        radial_delta += scalar * float(d[0])
        tangent_delta += scalar * float(d[1])
        circum_delta += scalar * float(d[2])

    radial_basis, tangent_basis, circum_basis = _tube_basis_grids(geom)
    displacement_grid = (
        radial_delta[:, :, None] * radial_basis
        + tangent_delta[:, :, None] * tangent_basis
        + circum_delta[:, :, None] * circum_basis
    ).astype(np.float32)
    if fea_responses:
        for response in fea_responses:
            area = next((a for a in tube_cfg.deformation.areas if a.id == response.area_id), None)
            if area is None or not area.enabled or area.kind != "fea_c3d6":
                continue
            scale = 0.5 * (1.0 - np.cos(2.0 * np.pi * float(area.frequency_hz) * float(t_s) + float(area.phase_rad)))
            displacement_grid += (float(scale) * response.displacement_grid).astype(np.float32)

    radial_component = np.sum(displacement_grid * radial_basis, axis=2).astype(np.float32)
    min_correction = np.maximum(float(tube_cfg.min_radius) - (geom.base_radius + radial_component), 0.0).astype(np.float32)
    if np.any(min_correction > 0.0):
        displacement_grid = (displacement_grid + min_correction[:, :, None] * radial_basis).astype(np.float32)
        radial_component = np.sum(displacement_grid * radial_basis, axis=2).astype(np.float32)
    base_vertices = tube_vertices_from_radius(geom, geom.base_radius).reshape(geom.base_radius.shape[0], geom.base_radius.shape[1], 3)
    vertices = (base_vertices + displacement_grid).reshape(-1, 3).astype(np.float32)
    radius = (geom.base_radius + radial_component).astype(np.float32)
    return vertices, radius, displacement_grid, global_amp, global_freq, phase_shift


def deformed_tube_radius(
    geom: TubeStaticGeometry,
    t_s: float,
    components: list[WaveComponent],
    min_radius: float,
    localized_components: list[LocalizedWaveComponent] | None = None,
    flow_mode: str = "cyclic",
    flow_cycles: float = 1.0,
    t_norm: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    radius, global_amp, global_freq, phase_shift = deformed_z(
        geom.base_radius,
        geom.theta_norm_grid,
        geom.s_norm_grid,
        t_s,
        components,
    )
    if localized_components:
        radius += compute_localized_displacement(
            geom.theta_norm_grid,
            geom.s_norm_grid,
            t_s,
            localized_components,
            flow_mode=flow_mode,
            flow_cycles=flow_cycles,
            t_norm=t_norm,
        )
    radius = np.maximum(radius, min_radius).astype(np.float32)
    return radius, global_amp, global_freq, phase_shift


def _interp_curve(arr: np.ndarray, progress: float) -> np.ndarray:
    n = arr.shape[0]
    if n <= 1:
        return arr[0]
    p = float(np.clip(progress, 0.0, 1.0))
    idx_f = p * (n - 1)
    i0 = int(np.floor(idx_f))
    i1 = min(i0 + 1, n - 1)
    a = idx_f - i0
    return ((1.0 - a) * arr[i0] + a * arr[i1]).astype(np.float32)


class TubeCameraTrajectory:
    def __init__(
        self,
        geom: TubeStaticGeometry,
        tube_cfg: TubeSceneConfig,
        cam_cfg: CameraTrajectoryConfig,
        preset: DifficultyPreset,
        rng: np.random.Generator,
        fps: int,
        num_frames: int,
    ) -> None:
        self.geom = geom
        self.tube_cfg = tube_cfg
        self.cam_cfg = cam_cfg
        self.preset = preset
        self.fps = fps
        self.num_frames = num_frames

        self.phase_wobble = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_orbit = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_roll = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_pitch = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_yaw = float(rng.uniform(0.0, 2.0 * np.pi))

    def _progress_at(self, t_s: float) -> float:
        duration = max((self.num_frames - 1) / max(float(self.fps), 1e-8), 1e-8)
        p = np.clip(t_s / duration, 0.0, 1.0)
        return float(p)

    def pose_at(self, t_s: float) -> TubeCameraPose:
        raw_progress = self._progress_at(t_s)
        # An explicitly requested unseen tail keeps the eye behind its look-at
        # target and the terminal boundary outside a wide-FOV image. A zero
        # margin retains the original full-length trajectory for compatibility.
        length = max(self.tube_cfg.length, 1e-8)
        lookahead = float(np.clip(self.tube_cfg.lookahead_distance / length, 1e-4, 0.45))
        end_margin_distance = max(float(self.tube_cfg.camera.end_margin_distance), 0.0)
        if end_margin_distance > 0.0:
            end_margin = float(np.clip(end_margin_distance / length, 1e-4, 0.45))
            max_progress = max(1.0 - lookahead - end_margin, 0.05)
            progress = raw_progress * max_progress
        else:
            progress = raw_progress
        center = _interp_curve(self.geom.centerline, progress)
        tangent = _normalize(_interp_curve(self.geom.tangent, progress))
        right = _normalize(_interp_curve(self.geom.frame_right, progress))

        right = _normalize(right - tangent * float(np.dot(right, tangent)))
        up = _normalize(np.cross(tangent, right))

        base_r = float(np.min(_interp_curve(self.geom.base_radius, progress)))
        wobble = self.tube_cfg.camera_wobble_ratio * self.preset.jitter_scale * np.sin(
            2.0 * np.pi * self.tube_cfg.camera_wobble_hz * t_s + self.phase_wobble
        )
        offset_ratio = float(np.clip(self.tube_cfg.camera_offset_ratio + wobble, -0.5, 0.5))
        orbit_ang = 2.0 * np.pi * (0.13 * self.preset.curve_scale) * t_s + self.phase_orbit

        offset = offset_ratio * base_r * (np.cos(orbit_ang) * right + np.sin(orbit_ang) * up)
        eye = center + offset.astype(np.float32)

        target_center = _interp_curve(self.geom.centerline, min(1.0, progress + lookahead))

        R_wc = _rotation_from_lookat(eye, target_center, up_hint=up)

        roll = np.deg2rad(self.cam_cfg.roll_jitter_deg * self.preset.jitter_scale) * np.sin(0.7 * t_s + self.phase_roll)
        pitch = np.deg2rad(self.cam_cfg.pitch_jitter_deg * self.preset.jitter_scale) * np.sin(0.9 * t_s + self.phase_pitch)
        yaw = np.deg2rad(self.cam_cfg.yaw_jitter_deg * self.preset.jitter_scale) * np.sin(0.8 * t_s + self.phase_yaw)
        R_jitter = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        R_wc = (R_wc @ R_jitter).astype(np.float32)

        q = _quat_from_rot(R_wc.astype(np.float64))
        return TubeCameraPose(t_wc=eye.astype(np.float32), R_wc=R_wc, q_xyzw=q)

    def clearance_at(self, t_s: float, radius_grid: np.ndarray, cam_pos: np.ndarray) -> float:
        # pose_at reserves axial room for look-ahead, so raw normalized time is
        # not the camera's centerline coordinate. Resolve the actual local ring
        # geometrically; this also remains correct if the trajectory mapping is
        # changed later.
        centerline = np.asarray(self.geom.centerline, dtype=np.float64)
        ring_idx = int(np.argmin(np.linalg.norm(centerline - cam_pos[None, :], axis=1)))
        center = self.geom.centerline[ring_idx]
        right = _normalize(self.geom.frame_right[ring_idx])
        up = _normalize(self.geom.frame_up[ring_idx])
        radial_dist = np.sqrt(float(np.dot(cam_pos - center, right)) ** 2 + float(np.dot(cam_pos - center, up)) ** 2)

        ring = radius_grid[ring_idx]
        min_local_radius = float(np.min(ring))
        return min_local_radius - radial_dist
