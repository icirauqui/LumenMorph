from dataclasses import replace
import json
import sys

import numpy as np

from backend.src.config import canonical_config_dict, load_config, load_config_dict
from backend.src.pipeline import generate_dataset
from backend.src.tube_fea import _continuum_fem_src
from backend.src.validator import validate_dataset


def _continuum_fem_tube_response_solver():
    continuum_fem_src = _continuum_fem_src()
    if str(continuum_fem_src) not in sys.path:
        sys.path.insert(0, str(continuum_fem_src))
    from continuum_fem.systems.tube_c3d6_response import solve_hollow_cylinder_c3d6_force_response

    return solve_hollow_cylinder_c3d6_force_response


def test_tube_pipeline_smoke(tmp_path):
    cfg = load_config("configs/tube_default.json")
    assert cfg.tube is not None

    cfg = replace(
        cfg,
        sequence_count={"train": 1, "val": 0, "test": 0},
        intrinsics=replace(cfg.intrinsics, width=160, height=120, fx=130.0, fy=130.0, cx=80.0, cy=60.0),
        sequence=replace(cfg.sequence, num_frames=10, fps=10),
        texture=replace(cfg.texture, width=256, height=256, min_orb_features=80, max_attempts=4),
        validation=replace(
            cfg.validation,
            min_orb_features_mean=20.0,
            min_frame_motion=1e-4,
            min_camera_clearance=0.01,
            max_abs_mesh_z=60.0,
        ),
        tube=replace(
            cfg.tube,
            length=8.0,
            radial_samples=22,
            longitudinal_samples=24,
            base_radius=1.4,
            min_radius=0.5,
            bump_count=12,
            localized_deform_count=2,
            localized_theta_center=0.2,
            localized_theta_jitter=0.05,
            localized_amp_scale=1.1,
            localized_tint_enabled=True,
            localized_tint_strength=0.2,
        ),
    )

    out_dir = tmp_path / "tube_dataset"
    manifest = generate_dataset(cfg, out_dir)

    assert manifest["scene_type"] == "tube"
    assert manifest["total_sequences"] == 1

    result = validate_dataset(out_dir)
    assert result.ok, f"validation failed: {result.errors}"

    seq_dir = out_dir / "train" / "seq_000"
    seq_meta = json.loads((seq_dir / "sequence_meta.json").read_text())
    assert seq_meta["scene_type"] == "tube"
    assert seq_meta["mesh_state_type"] == "tube_displacement_xyz_grid"
    assert seq_meta["localized_deformation_count"] == 2
    assert len(seq_meta["localized_wave_components"]) == 2
    assert seq_meta["localized_tint_enabled"] is True


def test_tube_single_sequence_flat_layout(tmp_path):
    cfg = load_config("configs/tube_default.json")
    assert cfg.tube is not None

    cfg = replace(
        cfg,
        sequence_count={"train": 1, "val": 0, "test": 0},
        intrinsics=replace(cfg.intrinsics, width=160, height=120, fx=130.0, fy=130.0, cx=80.0, cy=60.0),
        sequence=replace(cfg.sequence, num_frames=8, fps=8),
        texture=replace(cfg.texture, width=256, height=256, min_orb_features=60, max_attempts=4),
        validation=replace(
            cfg.validation,
            min_orb_features_mean=10.0,
            min_frame_motion=1e-4,
            min_camera_clearance=0.01,
            max_abs_mesh_z=60.0,
        ),
        tube=replace(
            cfg.tube,
            localized_deform_count=1,
            localized_theta_center=0.2,
            localized_theta_jitter=0.0,
            localized_amp_scale=1.0,
        ),
    )

    out_dir = tmp_path / "tube_flat"
    manifest = generate_dataset(
        cfg,
        out_dir,
        max_sequences=1,
        flatten_single_sequence=True,
    )

    assert manifest["layout"] == "flat_single_sequence"
    assert manifest["total_sequences"] == 1
    assert manifest["sequences"][0]["relative_path"] == "."
    assert (out_dir / "poses.csv").exists()
    assert (out_dir / "deformation_params.csv").exists()
    assert (out_dir / "mesh_static.npz").exists()
    assert (out_dir / "sequence_meta.json").exists()
    assert not (out_dir / "train").exists()

    result = validate_dataset(out_dir)
    assert result.ok, f"validation failed: {result.errors}"


def test_regenerating_shorter_sequence_removes_stale_frames(tmp_path):
    cfg = load_config("configs/tube_default.json")
    cfg = replace(
        cfg,
        sequence_count={"train": 1, "val": 0, "test": 0},
        intrinsics=replace(cfg.intrinsics, width=96, height=72, fx=80.0,
                           fy=80.0, cx=48.0, cy=36.0),
        sequence=replace(cfg.sequence, num_frames=5, fps=5),
        texture=replace(cfg.texture, width=128, height=128,
                        min_orb_features=1, max_attempts=1),
        validation=replace(cfg.validation, min_orb_features_mean=0.0,
                           min_frame_motion=0.0, min_camera_clearance=-10.0),
    )
    out_dir = tmp_path / "regenerated"
    generate_dataset(cfg, out_dir, max_sequences=1, flatten_single_sequence=True)
    shorter = replace(cfg, sequence=replace(cfg.sequence, num_frames=3))
    generate_dataset(
        shorter, out_dir, max_sequences=1, flatten_single_sequence=True
    )

    assert len(list((out_dir / "frames").glob("*.png"))) == 3
    assert len(list((out_dir / "mesh_z").glob("*.npy"))) == 3
    result = validate_dataset(out_dir)
    assert not any("count mismatch" in error.lower() for error in result.errors)


def test_tube_clearance_uses_camera_local_ring(tmp_path):
    cfg = load_config("configs/tube_default.json")
    cfg = replace(
        cfg,
        sequence_count={"train": 1, "val": 0, "test": 0},
        intrinsics=replace(cfg.intrinsics, width=96, height=72, fx=80.0,
                           fy=80.0, cx=48.0, cy=36.0),
        sequence=replace(cfg.sequence, num_frames=6, fps=6),
        texture=replace(cfg.texture, width=128, height=128,
                        min_orb_features=1, max_attempts=1),
        validation=replace(cfg.validation, min_orb_features_mean=0.0,
                           min_orb_features_frame=0.0,
                           min_frame_motion=0.0, min_camera_clearance=0.01),
    )
    out_dir = tmp_path / "clearance"
    generate_dataset(cfg, out_dir, max_sequences=1)
    result = validate_dataset(out_dir)
    clearance_errors = [error for error in result.errors
                        if "camera clearance" in error.lower()]
    assert not clearance_errors, clearance_errors
    assert result.stats["min_clearance"] >= 0.01


def test_tube_fea_generation_writes_displacement_and_metadata(tmp_path):
    payload = canonical_config_dict(load_config("configs/tube_default.json"))
    payload["seed"] = 12
    payload["sequence_count"] = {"train": 1, "val": 0, "test": 0}
    payload["default_preset_cycle"] = ["easy"]
    payload["sequence"]["num_frames"] = 3
    payload["sequence"]["fps"] = 2
    payload["intrinsics"] = {"width": 96, "height": 72, "fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 36.0}
    payload["texture"] = {"width": 64, "height": 64, "min_orb_features": 1, "max_attempts": 1}
    payload["output"] = {"write_frames": False, "write_mp4": False, "write_mesh_state": True}
    payload["validation"]["min_camera_clearance"] = -10.0
    for preset in payload["difficulty_presets"].values():
        preset["wave_count"] = 0
        preset["amp_min"] = 0.0
        preset["amp_max"] = 0.0
        preset["omega_min"] = 0.0
        preset["omega_max"] = 0.0
    payload["tube"] = {
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
            "flow_mode": "cyclic",
            "flow_cycles": 1.0,
            "fea": {"young_modulus_pa": 1.0e6, "poisson_ratio": 0.30, "axial_elements": 4, "theta_elements": 8},
            "localized_tint_enabled": False,
            "localized_tint_bgr": [48, 185, 255],
            "localized_tint_strength": 0.18,
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
        },
    }
    cfg = load_config_dict(payload)

    out_dir = tmp_path / "tube_fea_dataset"
    generate_dataset(cfg, out_dir, max_sequences=1)

    seq_dir = out_dir / "train" / "seq_000"
    first = np.load(seq_dir / "mesh_z" / "000000.npy")
    peak = np.load(seq_dir / "mesh_z" / "000001.npy")
    assert first.shape == (16, 12, 3)
    assert np.isfinite(peak).all()
    assert float(np.linalg.norm(first)) < 1e-8
    assert float(np.linalg.norm(peak)) > 0.0
    meta = json.loads((seq_dir / "sequence_meta.json").read_text())
    assert meta["fea_deformation_count"] == 1
    assert meta["fea_deformation_responses"][0]["area_id"] == "fea_1"


def test_generated_fea_displacements_match_direct_continuum_fem_solve(tmp_path):
    solve_response = _continuum_fem_tube_response_solver()
    payload = canonical_config_dict(load_config("configs/tube_default.json"))
    payload["seed"] = 123
    payload["sequence_count"] = {"train": 1, "val": 0, "test": 0}
    payload["default_preset_cycle"] = ["easy"]
    payload["sequence"]["num_frames"] = 5
    payload["sequence"]["fps"] = 4
    payload["intrinsics"] = {"width": 96, "height": 72, "fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 36.0}
    payload["texture"] = {"width": 64, "height": 64, "min_orb_features": 1, "max_attempts": 1}
    payload["output"] = {"write_frames": False, "write_mp4": False, "write_mesh_state": True}
    payload["validation"]["min_camera_clearance"] = -10.0
    for preset in payload["difficulty_presets"].values():
        preset["wave_count"] = 0
        preset["amp_min"] = 0.0
        preset["amp_max"] = 0.0
        preset["omega_min"] = 0.0
        preset["omega_max"] = 0.0

    length = 4.0
    outer_radius = 1.0
    shell_thickness = 0.2
    axial_elements = 4
    theta_elements = 8
    force_n = 150.0
    center_s = 0.5
    center_theta = 0.25
    frequency_hz = 1.0
    payload["tube"] = {
        "axis": {"sections": [{"kind": "straight", "length": length}]},
        "dimensions": {
            "radial_samples": theta_elements,
            "longitudinal_samples": axial_elements + 1,
            "base_radius": outer_radius,
            "min_radius": 0.25,
            "shell_thickness": shell_thickness,
        },
        "surface": {
            "bump_count": 0,
            "bump_amp_min": 0.0,
            "bump_amp_max": 0.0,
            "bump_sigma_long": 0.05,
            "bump_sigma_theta": 0.05,
        },
        "camera": {"offset_ratio": 0.0, "wobble_ratio": 0.0, "wobble_hz": 0.0, "lookahead_distance": 1.0},
        "deformation": {
            "flow_mode": "cyclic",
            "flow_cycles": 1.0,
            "fea": {
                "young_modulus_pa": 1.0e6,
                "poisson_ratio": 0.30,
                "axial_elements": axial_elements,
                "theta_elements": theta_elements,
            },
            "localized_tint_enabled": False,
            "localized_tint_bgr": [48, 185, 255],
            "localized_tint_strength": 0.18,
            "areas": [
                {
                    "id": "fea_oracle",
                    "enabled": True,
                    "kind": "fea_c3d6",
                    "center_s": center_s,
                    "center_theta": center_theta,
                    "sigma_s": 0.1,
                    "sigma_theta": 0.1,
                    "direction": {"radial": 1.0, "tangent": 0.0, "circumferential": 0.0},
                    "amplitude_out": 0.0,
                    "amplitude_in": 0.0,
                    "force_n": force_n,
                    "frequency_hz": frequency_hz,
                    "phase_rad": 0.0,
                    "mode": "radial",
                    "spatial_cycles_s": 0.0,
                    "spatial_cycles_theta": 0.0,
                    "flow_dir_s": 0.0,
                    "flow_dir_theta": 0.0,
                }
            ],
        },
    }
    cfg = load_config_dict(payload)

    out_dir = tmp_path / "tube_fea_oracle_dataset"
    generate_dataset(cfg, out_dir, max_sequences=1, show_progress=False)
    seq_dir = out_dir / "train" / "seq_000"

    theta = 2.0 * np.pi * center_theta
    force_vector = np.asarray([0.0, np.cos(theta) * force_n, np.sin(theta) * force_n], dtype=np.float64)
    response = solve_response(
        length=length,
        inner_radius=outer_radius - shell_thickness,
        outer_radius=outer_radius,
        n_axial=axial_elements,
        n_theta=theta_elements,
        young_modulus_pa=1.0e6,
        poisson_ratio=0.30,
        center_s=center_s,
        center_theta=center_theta,
        force_vector=force_vector,
    )
    assert response.displacement.shape == (axial_elements + 1, theta_elements, 3)
    assert response.applied_s_norm == center_s
    assert response.applied_theta_norm == center_theta

    static = np.load(seq_dir / "mesh_static.npz")
    expected_full = (
        response.displacement[:, :, 0:1] * static["tangent"][:, None, :]
        + response.displacement[:, :, 1:2] * static["frame_right"][:, None, :]
        + response.displacement[:, :, 2:3] * static["frame_up"][:, None, :]
    ).astype(np.float32)
    assert float(np.linalg.norm(expected_full)) > 0.0

    expected_scales = {
        0: 0.0,
        1: 0.5,
        2: 1.0,
        3: 0.5,
        4: 0.0,
    }
    for frame_idx, scale in expected_scales.items():
        exported = np.load(seq_dir / "mesh_z" / f"{frame_idx:06d}.npy")
        np.testing.assert_allclose(
            exported,
            expected_full * np.float32(scale),
            rtol=3e-5,
            atol=5e-8,
            err_msg=f"frame {frame_idx} exported displacement differs from direct ContinuumFEM solve",
        )

    meta = json.loads((seq_dir / "sequence_meta.json").read_text())
    assert meta["mesh_state_type"] == "tube_displacement_xyz_grid"
    assert meta["fea_deformation_count"] == 1
    assert meta["fea_deformation_responses"][0]["area_id"] == "fea_oracle"
    assert meta["fea_deformation_responses"][0]["applied_node"] == response.applied_node
    assert meta["fea_deformation_responses"][0]["applied_axial_index"] == response.applied_axial_index
    assert meta["fea_deformation_responses"][0]["applied_theta_index"] == response.applied_theta_index
    np.testing.assert_allclose(meta["fea_deformation_responses"][0]["force_vector_local"], force_vector, atol=1e-5)
