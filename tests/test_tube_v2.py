from dataclasses import replace

import numpy as np

from backend.src.config import load_config_dict
from backend.src.camera import project_points, world_to_camera
from backend.src.tube import TubeCameraTrajectory, build_tube_static_geometry, deformed_tube_mesh
from backend.src.tube_fea import build_tube_fea_responses, fea_cyclic_scale
from backend.src.types import (
    CameraTrajectoryConfig,
    CameraIntrinsics,
    DifficultyPreset,
    TubeDeformationArea,
    TubeDeformationConfig,
    TubeDeformationDirection,
)


def _tube_cfg(payload_tube):
    base = {
        "version": "0.2.2",
        "seed": 7,
        "scene_type": "tube",
        "sequence_count": {"train": 1, "val": 0, "test": 0},
        "default_preset_cycle": ["p"],
        "intrinsics": {"width": 64, "height": 48, "fx": 60.0, "fy": 60.0, "cx": 32.0, "cy": 24.0},
        "sequence": {"num_frames": 4, "fps": 4, "mesh_nx": 8, "mesh_ny": 8, "x_min": -1.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "camera": {
            "base_x": 0.0,
            "base_y": 0.0,
            "base_z": 0.0,
            "forward_drift_per_s": 0.0,
            "orbit_radius": 0.0,
            "orbit_frequency_hz": 0.0,
            "lateral_amplitude": 0.0,
            "lateral_frequency_hz": 0.0,
            "lookahead_distance": 1.0,
            "lookat_height": 0.0,
            "roll_jitter_deg": 0.0,
            "pitch_jitter_deg": 0.0,
            "yaw_jitter_deg": 0.0,
        },
        "texture": {"width": 64, "height": 64, "min_orb_features": 1, "max_attempts": 1},
        "photometric": {"gaussian_sigma": 0.0, "gain_jitter": 0.0},
        "output": {"write_frames": False, "write_mp4": False, "write_mesh_state": True},
        "validation": {"min_orb_features_mean": 0.0, "min_frame_motion": 0.0, "min_camera_clearance": -1.0, "max_abs_mesh_z": 100.0},
        "difficulty_presets": {
            "p": {
                "wave_count": 0,
                "amp_min": 0.0,
                "amp_max": 0.0,
                "freq_min": 0.0,
                "freq_max": 0.0,
                "omega_min": 0.0,
                "omega_max": 0.0,
                "drift_scale": 1.0,
                "jitter_scale": 1.0,
                "curve_scale": 1.0,
            }
        },
        "tube": payload_tube,
    }
    cfg = load_config_dict(base)
    assert cfg.tube is not None
    return cfg.tube, cfg.difficulty_presets["p"]


def test_tube_v2_turn_sections_change_centerline_direction():
    tube, preset = _tube_cfg(
        {
            "axis": {
                "sections": [
                    {"kind": "straight", "length": 2.0},
                    {"kind": "turn", "direction": "right", "angle_deg": 90.0, "radius": 2.0},
                    {"kind": "straight", "length": 2.0},
                ]
            },
            "dimensions": {"radial_samples": 12, "longitudinal_samples": 36, "base_radius": 1.0, "min_radius": 0.3},
            "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
            "camera": {
                "offset_ratio": 0.0,
                "wobble_ratio": 0.0,
                "wobble_hz": 0.0,
                "lookahead_distance": 1.0,
            },
            "deformation": {"areas": [], "flow_mode": "cyclic", "flow_cycles": 1.0},
        }
    )
    geom = build_tube_static_geometry(tube, preset, np.random.default_rng(1))
    assert geom.centerline[-1, 0] > 2.0
    assert geom.centerline[-1, 1] > 2.0
    dot = np.sum(geom.tangent * geom.frame_right, axis=1)
    assert np.max(np.abs(dot)) < 1e-4


def test_tube_v2_optional_end_caps_close_both_terminal_rings():
    tube, preset = _tube_cfg(
        {
            "axis": {"sections": [{"kind": "straight", "length": 4.0}]},
            "dimensions": {
                "radial_samples": 12,
                "longitudinal_samples": 20,
                "base_radius": 1.0,
                "min_radius": 0.3,
                "cap_ends": True,
            },
            "surface": {
                "bump_count": 0,
                "bump_amp_min": 0.0,
                "bump_amp_max": 0.0,
                "bump_sigma_long": 0.05,
                "bump_sigma_theta": 0.05,
            },
            "camera": {
                "offset_ratio": 0.0,
                "wobble_ratio": 0.0,
                "wobble_hz": 0.0,
                "lookahead_distance": 1.0,
            },
            "deformation": {"areas": [], "flow_mode": "cyclic", "flow_cycles": 1.0},
        }
    )
    geom = build_tube_static_geometry(tube, preset, np.random.default_rng(1))
    side_triangles = 2 * (tube.longitudinal_samples - 1) * tube.radial_samples
    cap_triangles = 2 * (tube.radial_samples - 2)
    assert geom.triangles.shape == (side_triangles + cap_triangles, 3)


def test_tube_v2_left_and_down_turns_keep_forward_arc_progress():
    cases = [
        ("left", lambda end: (end[0] < -2.0 and end[1] > 2.0)),
        ("down", lambda end: (end[2] < -2.0 and end[1] > 2.0)),
    ]
    for direction, predicate in cases:
        tube, preset = _tube_cfg(
            {
                "axis": {
                    "sections": [
                        {"kind": "straight", "length": 2.0},
                        {"kind": "turn", "direction": direction, "angle_deg": 90.0, "radius": 2.0},
                        {"kind": "straight", "length": 2.0},
                    ]
                },
                "dimensions": {"radial_samples": 12, "longitudinal_samples": 48, "base_radius": 1.0, "min_radius": 0.3},
                "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
                "camera": {"offset_ratio": 0.0, "wobble_ratio": 0.0, "wobble_hz": 0.0, "lookahead_distance": 1.0},
                "deformation": {"areas": [], "flow_mode": "cyclic", "flow_cycles": 1.0},
            }
        )
        geom = build_tube_static_geometry(tube, preset, np.random.default_rng(1))
        assert predicate(geom.centerline[-1]), f"{direction} turn ended at {geom.centerline[-1]}"


def test_tube_v2_vector_deformation_has_tangent_component_and_radius_clamp():
    tube, preset = _tube_cfg(
        {
            "axis": {"sections": [{"kind": "straight", "length": 4.0}]},
            "dimensions": {"radial_samples": 16, "longitudinal_samples": 24, "base_radius": 1.0, "min_radius": 0.8},
            "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
            "camera": {"offset_ratio": 0.0, "wobble_ratio": 0.0, "wobble_hz": 0.0, "lookahead_distance": 1.0},
            "deformation": {"areas": [], "flow_mode": "cyclic", "flow_cycles": 1.0},
        }
    )
    area = TubeDeformationArea(
        id="a",
        enabled=True,
        center_s=0.5,
        center_theta=0.25,
        sigma_s=0.3,
        sigma_theta=0.3,
        direction=TubeDeformationDirection(radial=-1.0, tangent=1.0, circumferential=0.0),
        amplitude_out=1.0,
        amplitude_in=1.0,
        frequency_hz=0.0,
        phase_rad=np.pi / 2.0,
        mode="radial",
    )
    tube = replace(tube, deformation=TubeDeformationConfig(areas=[area], flow_mode="cyclic", flow_cycles=1.0))
    geom = build_tube_static_geometry(tube, preset, np.random.default_rng(1))
    _, radius, displacement_grid, *_ = deformed_tube_mesh(geom, tube, 0.0, [])
    assert float(np.min(radius)) >= 0.8 - 1e-6
    tangent_component = np.sum(displacement_grid * geom.tangent[:, None, :], axis=2)
    assert float(np.max(tangent_component)) > 0.2


def test_tube_fea_area_response_is_finite_and_cyclic():
    tube, preset = _tube_cfg(
        {
            "axis": {"sections": [{"kind": "straight", "length": 4.0}]},
            "dimensions": {
                "radial_samples": 12,
                "longitudinal_samples": 16,
                "base_radius": 1.0,
                "min_radius": 0.35,
                "shell_thickness": 0.2,
            },
            "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
            "camera": {"offset_ratio": 0.0, "wobble_ratio": 0.0, "wobble_hz": 0.0, "lookahead_distance": 1.0},
            "deformation": {
                "areas": [
                    {
                        "id": "fea_1",
                        "enabled": True,
                        "kind": "fea_c3d6",
                        "center_s": 0.5,
                        "center_theta": 0.25,
                        "sigma_s": 0.1,
                        "sigma_theta": 0.1,
                        "direction": {"radial": 1.0, "tangent": 0.0, "circumferential": 0.0},
                        "amplitude_out": 0.0,
                        "amplitude_in": 0.0,
                        "force_n": 120.0,
                        "frequency_hz": 1.0,
                        "phase_rad": 0.0,
                        "mode": "radial",
                        "spatial_cycles_s": 0.0,
                        "spatial_cycles_theta": 0.0,
                        "flow_dir_s": 0.0,
                        "flow_dir_theta": 0.0,
                    }
                ],
                "fea": {"young_modulus_pa": 1.0e6, "poisson_ratio": 0.30, "axial_elements": 4, "theta_elements": 8},
                "flow_mode": "cyclic",
                "flow_cycles": 1.0,
            },
        }
    )
    geom = build_tube_static_geometry(tube, preset, np.random.default_rng(1))
    fea_responses = build_tube_fea_responses(geom, tube)
    assert len(fea_responses) == 1
    assert np.isfinite(fea_responses[0].displacement_grid).all()
    assert float(np.linalg.norm(fea_responses[0].displacement_grid)) > 0.0
    assert fea_responses[0].applied_node >= 0
    np.testing.assert_allclose(fea_responses[0].force_vector_local, [0.0, 0.0, 120.0], atol=1e-5)
    inward_area = replace(tube.deformation.areas[0], direction=TubeDeformationDirection(radial=-1.0, tangent=0.0, circumferential=0.0))
    inward_tube = replace(tube, deformation=replace(tube.deformation, areas=[inward_area]))
    inward_response = build_tube_fea_responses(geom, inward_tube)[0]
    np.testing.assert_allclose(inward_response.force_vector_local, [0.0, 0.0, -120.0], atol=1e-5)

    mixed_area = replace(
        tube.deformation.areas[0],
        direction=TubeDeformationDirection(
            radial=-1.0, tangent=2.0, circumferential=2.0
        ),
    )
    mixed_tube = replace(tube, deformation=replace(tube.deformation, areas=[mixed_area]))
    mixed_response = build_tube_fea_responses(geom, mixed_tube)[0]
    expected = 120.0 * np.asarray([2.0, -2.0, -1.0]) / 3.0
    np.testing.assert_allclose(mixed_response.force_vector_local, expected, atol=1e-5)

    assert fea_cyclic_scale(1.0, 0.0, 0.0) == 0.0
    assert fea_cyclic_scale(1.0, 0.0, 0.5) == 1.0
    assert abs(fea_cyclic_scale(1.0, 0.0, 1.0)) < 1e-12

    verts0, _radius0, disp0, *_ = deformed_tube_mesh(geom, tube, 0.0, [], fea_responses=fea_responses)
    verts_peak, _radius_peak, disp_peak, *_ = deformed_tube_mesh(geom, tube, 0.5, [], fea_responses=fea_responses)
    assert np.isfinite(verts0).all()
    assert np.isfinite(verts_peak).all()
    assert float(np.linalg.norm(disp0)) < 1e-8
    assert float(np.linalg.norm(disp_peak)) > 0.0


def test_tube_export_camera_projects_theta_top_above_bottom():
    tube, preset = _tube_cfg(
        {
            "axis": {"sections": [{"kind": "straight", "length": 4.0}]},
            "dimensions": {"radial_samples": 16, "longitudinal_samples": 24, "base_radius": 1.0, "min_radius": 0.3},
            "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
            "camera": {
                "offset_ratio": 0.0,
                "wobble_ratio": 0.0,
                "wobble_hz": 0.0,
                "lookahead_distance": 1.0,
                "end_margin_distance": 1.5,
            },
            "deformation": {"areas": [], "flow_mode": "cyclic", "flow_cycles": 1.0},
        }
    )
    geom = build_tube_static_geometry(tube, preset, np.random.default_rng(1))
    cam_cfg = CameraTrajectoryConfig(
        base_x=0.0,
        base_y=0.0,
        base_z=0.0,
        forward_drift_per_s=0.0,
        orbit_radius=0.0,
        orbit_frequency_hz=0.0,
        lateral_amplitude=0.0,
        lateral_frequency_hz=0.0,
        lookahead_distance=1.0,
        lookat_height=0.0,
        roll_jitter_deg=0.0,
        pitch_jitter_deg=0.0,
        yaw_jitter_deg=0.0,
    )
    camera = TubeCameraTrajectory(
        geom=geom,
        tube_cfg=tube,
        cam_cfg=cam_cfg,
        preset=preset,
        rng=np.random.default_rng(2),
        fps=4,
        num_frames=4,
    )
    pose = camera.pose_at(0.0)
    ring_idx = 8
    center = geom.centerline[ring_idx]
    top = center + geom.frame_up[ring_idx] * 0.8
    bottom = center - geom.frame_up[ring_idx] * 0.8
    intr = CameraIntrinsics(width=160, height=120, fx=100.0, fy=100.0, cx=80.0, cy=60.0)

    uv, valid = project_points(world_to_camera(np.stack([top, bottom]).astype(np.float32), pose.R_wc, pose.t_wc), intr)

    assert valid.all()
    assert uv[0, 1] < uv[1, 1]

    final_pose = camera.pose_at((camera.num_frames - 1) / camera.fps)
    assert np.isfinite(final_pose.t_wc).all()
    assert np.isfinite(final_pose.R_wc).all()
    # The last eye must retain both its lookahead and the separate unseen tail.
    distance_to_endpoint = float(np.linalg.norm(final_pose.t_wc - geom.centerline[-1]))
    required_margin = tube.lookahead_distance + tube.camera.end_margin_distance
    assert distance_to_endpoint > 0.99 * required_margin


def _patch_mode_tube(force_n: float = 120.0, load_distribution: str = "gaussian_patch"):
    return _tube_cfg(
        {
            "axis": {"sections": [{"kind": "straight", "length": 4.0}]},
            "dimensions": {
                "radial_samples": 12,
                "longitudinal_samples": 16,
                "base_radius": 1.0,
                "min_radius": 0.35,
                "shell_thickness": 0.2,
            },
            "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
            "camera": {"offset_ratio": 0.0, "wobble_ratio": 0.0, "wobble_hz": 0.0, "lookahead_distance": 1.0},
            "deformation": {
                "areas": [
                    {
                        "id": f"fea_{tag}",
                        "enabled": True,
                        "kind": "fea_c3d6",
                        "center_s": 0.5,
                        "center_theta": theta,
                        "sigma_s": 0.1,
                        "sigma_theta": 0.1,
                        "direction": {"radial": -1.0, "tangent": 0.0, "circumferential": 0.0},
                        "amplitude_out": 0.0,
                        "amplitude_in": 0.0,
                        "force_n": force_n,
                        "frequency_hz": 1.0,
                        "phase_rad": 0.0,
                        "mode": "radial",
                        "spatial_cycles_s": 0.0,
                        "spatial_cycles_theta": 0.0,
                        "flow_dir_s": 0.0,
                        "flow_dir_theta": 0.0,
                    }
                    for tag, theta in (("top", 0.25), ("bottom", 0.75))
                ],
                "fea": {
                    "young_modulus_pa": 1.0e6,
                    "poisson_ratio": 0.30,
                    "axial_elements": 6,
                    "theta_elements": 12,
                    "load_distribution": load_distribution,
                },
                "flow_mode": "cyclic",
                "flow_cycles": 1.0,
            },
        }
    )


def test_fea_load_distribution_defaults_to_point_and_patch_spreads_the_load():
    """gaussian_patch must spread one area's force over its sigma footprint."""
    point_tube, preset = _patch_mode_tube(load_distribution="point")
    patch_tube, _ = _patch_mode_tube(load_distribution="gaussian_patch")
    assert point_tube.deformation.fea.load_distribution == "point"
    # the default, when the key is absent, stays "point" so earlier datasets reproduce
    default_tube, _ = _tube_cfg(
        {
            "axis": {"sections": [{"kind": "straight", "length": 4.0}]},
            "dimensions": {"radial_samples": 12, "longitudinal_samples": 16, "base_radius": 1.0, "min_radius": 0.35, "shell_thickness": 0.2},
            "surface": {"bump_count": 0, "bump_amp_min": 0.0, "bump_amp_max": 0.0, "bump_sigma_long": 0.05, "bump_sigma_theta": 0.05},
            "camera": {"offset_ratio": 0.0, "wobble_ratio": 0.0, "wobble_hz": 0.0, "lookahead_distance": 1.0},
            "deformation": {"areas": [], "fea": {"axial_elements": 4, "theta_elements": 8}},
        }
    )
    assert default_tube.deformation.fea.load_distribution == "point"

    geom = build_tube_static_geometry(point_tube, preset, np.random.default_rng(1))
    point_responses = build_tube_fea_responses(geom, point_tube)
    patch_responses = build_tube_fea_responses(geom, patch_tube)

    assert [r.load_node_count for r in point_responses] == [1, 1]
    assert all(r.load_distribution == "gaussian_patch" for r in patch_responses)
    assert all(r.load_node_count > 1 for r in patch_responses)

    for responses in (point_responses, patch_responses):
        for r in responses:
            assert np.isfinite(r.displacement_grid).all()
            assert float(np.linalg.norm(r.displacement_grid)) > 0.0

    # opposing pair still cancels, so the patch adds no net force to the tube
    net = patch_responses[0].force_vector_local + patch_responses[1].force_vector_local
    assert float(np.linalg.norm(net)) < 1e-4

    # spreading the same total force over an area is more compliant per node,
    # so it produces a shallower dimple than concentrating it on one node
    point_peak = float(np.linalg.norm(point_responses[0].displacement_grid, axis=-1).max())
    patch_peak = float(np.linalg.norm(patch_responses[0].displacement_grid, axis=-1).max())
    assert patch_peak < point_peak

    mixed_area = replace(
        patch_tube.deformation.areas[0],
        direction=TubeDeformationDirection(
            radial=-1.0, tangent=2.0, circumferential=2.0
        ),
    )
    mixed_tube = replace(
        patch_tube,
        deformation=replace(patch_tube.deformation, areas=[mixed_area]),
    )
    mixed_response = build_tube_fea_responses(geom, mixed_tube)[0]
    expected = 120.0 * np.asarray([2.0, -2.0, -1.0]) / 3.0
    np.testing.assert_allclose(mixed_response.force_vector_local, expected, atol=1e-5)
    assert mixed_response.load_node_count > 1


def test_fea_patch_response_is_linear_in_force_and_rejects_bad_mode():
    """The patch solve stays linear, so force calibration is exact."""
    tube_1x, preset = _patch_mode_tube(force_n=120.0)
    tube_3x, _ = _patch_mode_tube(force_n=360.0)
    geom = build_tube_static_geometry(tube_1x, preset, np.random.default_rng(1))
    d1 = build_tube_fea_responses(geom, tube_1x)[0].displacement_grid
    d3 = build_tube_fea_responses(geom, tube_3x)[0].displacement_grid
    np.testing.assert_allclose(d3, 3.0 * d1, rtol=1e-4, atol=1e-7)

    import pytest

    with pytest.raises(ValueError):
        _patch_mode_tube(load_distribution="nonsense")
