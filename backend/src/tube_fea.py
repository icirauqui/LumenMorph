from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .tube import TubeFEAAreaResponse, TubeStaticGeometry
from .types import TubeDeformationArea, TubeSceneConfig


def fea_cyclic_scale(frequency_hz: float, phase_rad: float, t_s: float) -> float:
    return float(0.5 * (1.0 - np.cos(2.0 * np.pi * float(frequency_hz) * float(t_s) + float(phase_rad))))


def _continuum_fem_src() -> Path:
    env_path = os.environ.get("CONTINUUM_FEM_SRC")
    if env_path:
        return Path(env_path).expanduser().resolve(strict=False)
    project_root = Path(__file__).resolve().parents[2]
    bundled = project_root / "continuum-fem" / "src"
    if bundled.is_dir():
        return bundled
    return project_root.parent / "continuum-fem" / "src"


def _load_patch_solver():
    """Return the explicit ContinuumFEM linear-solid implementation."""
    src = _continuum_fem_src()
    if not src.exists():
        raise RuntimeError(f"ContinuumFEM src directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    try:
        from continuum_fem.fea.solid import LinearSolidSolver
        from continuum_fem.systems.solid_builder import SolidBuilder
    except ImportError as exc:  # pragma: no cover - depends on local dependency installation
        raise RuntimeError(
            "Distributed (gaussian_patch) FEA loads require ContinuumFEM and scipy. "
            "Install backend dependencies including scipy before using load_distribution='gaussian_patch'."
        ) from exc
    return SolidBuilder, LinearSolidSolver


def _patch_load_weights(
    area: TubeDeformationArea,
    n_axial: int,
    n_theta: int,
    sigma_cutoff: float,
) -> dict[tuple[int, int], float]:
    """Gaussian weights over outer nodes inside the area's sigma_s/sigma_theta footprint.

    Fixed end rings (axial index 0 and n_axial) are excluded so the clamped
    boundary cannot silently absorb part of the requested load.
    """
    sigma_s = max(float(area.sigma_s), 1e-6)
    sigma_theta = max(float(area.sigma_theta), 1e-6)
    s_nodes = np.linspace(0.0, 1.0, n_axial + 1, dtype=np.float64)
    theta_nodes = np.arange(n_theta, dtype=np.float64) / float(n_theta)

    weights: dict[tuple[int, int], float] = {}
    for i in range(1, n_axial):
        ws = (s_nodes[i] - float(area.center_s)) / sigma_s
        if abs(ws) > sigma_cutoff:
            continue
        for j in range(n_theta):
            d_theta = theta_nodes[j] - float(area.center_theta)
            d_theta -= round(d_theta)  # shortest periodic distance
            wt = d_theta / sigma_theta
            if abs(wt) > sigma_cutoff:
                continue
            weights[(i, j)] = float(np.exp(-0.5 * (ws * ws + wt * wt)))

    if not weights:
        axial_idx = int(np.clip(round(float(area.center_s) * n_axial), 1, n_axial - 1))
        theta_idx = int(round(float(area.center_theta) * n_theta)) % n_theta
        weights[(axial_idx, theta_idx)] = 1.0
    return weights


def _area_direction_at_theta(
    area: TubeDeformationArea, theta_norm: float
) -> np.ndarray:
    """Return the normalized straight-cylinder local load direction.

    The configured basis is centreline tangent, radial and circumferential.
    Earlier FEA code used only the sign of ``radial`` and silently ignored the
    other two declared components; retained radial-only configurations are
    unchanged by this correction.
    """

    angle = 2.0 * np.pi * float(theta_norm)
    axial = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    radial = np.asarray([0.0, np.cos(angle), np.sin(angle)], dtype=np.float64)
    circumferential = np.asarray(
        [0.0, -np.sin(angle), np.cos(angle)], dtype=np.float64
    )
    direction = (
        float(area.direction.tangent) * axial
        + float(area.direction.radial) * radial
        + float(area.direction.circumferential) * circumferential
    )
    magnitude = float(np.linalg.norm(direction))
    if not np.isfinite(magnitude) or magnitude <= 1e-12:
        raise ValueError(f"Degenerate FEA load direction for area {area.id!r}")
    return direction / magnitude


def _solve_patch_response(
    *,
    length: float,
    inner_radius: float,
    outer_radius: float,
    n_axial: int,
    n_theta: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    area: TubeDeformationArea,
    sigma_cutoff: float,
):
    """Solve the tube response to `area`'s force spread over its Gaussian footprint.

    Each loaded node follows the area's declared local radial, centreline-
    tangent and circumferential components at that node. Weights are scaled so
    the net force resolved onto the patch-centre direction equals the
    configured ``force_n``; the solve stays linear, so the response remains
    exactly proportional to ``force_n``.
    """
    SolidBuilder, LinearSolidSolver = _load_patch_solver()
    solid = SolidBuilder.build_hollow_cylinder_c3d6(
        length=float(length),
        inner_radius=float(inner_radius),
        outer_radius=float(outer_radius),
        n_axial=int(n_axial),
        n_theta=int(n_theta),
        E=float(young_modulus_pa),
        nu=float(poisson_ratio),
        fix_ends=True,
    )

    def node_id(i: int, j: int, radial_idx: int) -> int:
        return ((i * int(n_theta) + (j % int(n_theta))) * 2) + radial_idx

    centre_dir = _area_direction_at_theta(area, float(area.center_theta))

    weights = _patch_load_weights(area, int(n_axial), int(n_theta), float(sigma_cutoff))
    vectors: dict[tuple[int, int], np.ndarray] = {}
    for (i, j), w in weights.items():
        vectors[(i, j)] = w * _area_direction_at_theta(
            area, float(j) / float(n_theta)
        )

    net = np.sum(np.stack(list(vectors.values()), axis=0), axis=0)
    projection = float(np.dot(net, centre_dir))
    if abs(projection) < 1e-12:
        raise ValueError(f"Degenerate FEA load patch for area {area.id!r}: net force cancels")
    scale = float(area.force_n) / projection

    loads = np.zeros_like(solid.loads)
    for (i, j), vec in vectors.items():
        loads[node_id(i, j, 1)] = vec * scale
    solid.loads = loads
    solid.validate()

    result = LinearSolidSolver.solve(solid)
    displacement = np.zeros((int(n_axial) + 1, int(n_theta), 3), dtype=np.float64)
    for i in range(int(n_axial) + 1):
        for j in range(int(n_theta)):
            displacement[i, j, :] = result.u[node_id(i, j, 1)]

    axial_idx = int(np.clip(round(float(area.center_s) * int(n_axial)), 1, int(n_axial) - 1))
    theta_idx = int(round(float(area.center_theta) * int(n_theta))) % int(n_theta)
    s_nodes = np.linspace(0.0, 1.0, int(n_axial) + 1, dtype=np.float64)
    return (
        displacement,
        centre_dir * float(area.force_n),
        float(s_nodes[axial_idx]),
        float(theta_idx) / float(n_theta),
        int(node_id(axial_idx, theta_idx, 1)),
        axial_idx,
        theta_idx,
        len(weights),
    )


def _load_solver():
    src = _continuum_fem_src()
    if not src.exists():
        raise RuntimeError(f"ContinuumFEM src directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    try:
        from continuum_fem.systems.tube_c3d6_response import solve_hollow_cylinder_c3d6_force_response
    except ImportError as exc:  # pragma: no cover - depends on local dependency installation
        raise RuntimeError(
            "C3D6 FEA deformation requires ContinuumFEM and scipy. "
            "Install backend dependencies including scipy before using fea_c3d6 areas."
        ) from exc
    return solve_hollow_cylinder_c3d6_force_response


def _area_force_vector_local(area: TubeDeformationArea) -> np.ndarray:
    return _area_direction_at_theta(area, float(area.center_theta)) * float(
        area.force_n
    )


def _interp_periodic_response(
    response: np.ndarray,
    s_grid: np.ndarray,
    theta_grid: np.ndarray,
) -> np.ndarray:
    n_axial = response.shape[0] - 1
    n_theta = response.shape[1]

    s_pos = np.clip(s_grid, 0.0, 1.0) * float(n_axial)
    s0 = np.floor(s_pos).astype(np.int64)
    s0 = np.clip(s0, 0, n_axial - 1)
    s1 = s0 + 1
    sa = (s_pos - s0.astype(np.float32)).astype(np.float32)

    theta_pos = (theta_grid % 1.0) * float(n_theta)
    t0 = np.floor(theta_pos).astype(np.int64) % n_theta
    t1 = (t0 + 1) % n_theta
    ta = (theta_pos - np.floor(theta_pos)).astype(np.float32)

    v00 = response[s0, t0]
    v01 = response[s0, t1]
    v10 = response[s1, t0]
    v11 = response[s1, t1]
    v0 = v00 * (1.0 - ta[:, :, None]) + v01 * ta[:, :, None]
    v1 = v10 * (1.0 - ta[:, :, None]) + v11 * ta[:, :, None]
    return (v0 * (1.0 - sa[:, :, None]) + v1 * sa[:, :, None]).astype(np.float32)


def _local_xyz_to_world_grid(geom: TubeStaticGeometry, local_disp: np.ndarray) -> np.ndarray:
    tangent = np.broadcast_to(geom.tangent[:, None, :], local_disp.shape)
    right = np.broadcast_to(geom.frame_right[:, None, :], local_disp.shape)
    up = np.broadcast_to(geom.frame_up[:, None, :], local_disp.shape)
    return (
        local_disp[:, :, 0:1] * tangent
        + local_disp[:, :, 1:2] * right
        + local_disp[:, :, 2:3] * up
    ).astype(np.float32)


def build_tube_fea_responses(geom: TubeStaticGeometry, tube_cfg: TubeSceneConfig) -> list[TubeFEAAreaResponse]:
    fea_areas = [a for a in tube_cfg.deformation.areas if a.enabled and a.kind == "fea_c3d6" and a.force_n > 0.0]
    if not fea_areas:
        return []

    fea_cfg = tube_cfg.deformation.fea
    distributed = str(fea_cfg.load_distribution) == "gaussian_patch"
    solve_response = None if distributed else _load_solver()
    outer_radius = max(float(tube_cfg.base_radius), 1e-6)
    thickness = max(float(tube_cfg.shell_thickness), outer_radius * 0.02, 1e-6)
    inner_radius = max(outer_radius - thickness, 1e-6)
    if inner_radius >= outer_radius:
        inner_radius = outer_radius * 0.95

    out: list[TubeFEAAreaResponse] = []
    for area in fea_areas:
        if distributed:
            (
                raw_disp,
                force_vector,
                applied_s_norm,
                applied_theta_norm,
                applied_node,
                applied_axial_index,
                applied_theta_index,
                load_node_count,
            ) = _solve_patch_response(
                length=max(float(tube_cfg.length), 1e-6),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                n_axial=int(fea_cfg.axial_elements),
                n_theta=int(fea_cfg.theta_elements),
                young_modulus_pa=float(fea_cfg.young_modulus_pa),
                poisson_ratio=float(fea_cfg.poisson_ratio),
                area=area,
                sigma_cutoff=float(fea_cfg.load_sigma_cutoff),
            )
        else:
            force_vector = _area_force_vector_local(area)
            response = solve_response(
                length=max(float(tube_cfg.length), 1e-6),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                n_axial=int(fea_cfg.axial_elements),
                n_theta=int(fea_cfg.theta_elements),
                young_modulus_pa=float(fea_cfg.young_modulus_pa),
                poisson_ratio=float(fea_cfg.poisson_ratio),
                center_s=float(area.center_s),
                center_theta=float(area.center_theta),
                force_vector=force_vector,
            )
            raw_disp = response.displacement
            applied_s_norm = float(response.applied_s_norm)
            applied_theta_norm = float(response.applied_theta_norm)
            applied_node = int(response.applied_node)
            applied_axial_index = int(response.applied_axial_index)
            applied_theta_index = int(response.applied_theta_index)
            load_node_count = 1

        local_disp = _interp_periodic_response(
            raw_disp,
            geom.s_norm_grid,
            geom.theta_norm_grid,
        )
        out.append(
            TubeFEAAreaResponse(
                area_id=area.id,
                displacement_grid=_local_xyz_to_world_grid(geom, local_disp),
                applied_s_norm=float(applied_s_norm),
                applied_theta_norm=float(applied_theta_norm),
                applied_node=int(applied_node),
                applied_axial_index=int(applied_axial_index),
                applied_theta_index=int(applied_theta_index),
                force_vector_local=np.asarray(force_vector, dtype=np.float32),
                load_distribution=str(fea_cfg.load_distribution),
                load_node_count=int(load_node_count),
            )
        )
    return out
