from backend.src.config import canonical_config_dict, config_hash, load_config, load_config_dict


def test_tube_flow_defaults_are_back_compatible():
    cfg = load_config("configs/tube_default.json")
    assert cfg.tube is not None
    assert cfg.tube.deformation_flow_mode == "cyclic"
    assert cfg.tube.deformation_flow_cycles == 1.0
    assert cfg.tube.camera.end_margin_distance == 0.0
    assert cfg.tube.dimensions.cap_ends is False


def test_tube_flow_fields_roundtrip_and_hash_stability():
    cfg = load_config("configs/tube_default.json")
    assert cfg.tube is not None
    payload = canonical_config_dict(cfg)
    payload["tube"]["deformation"]["flow_mode"] = "linear"
    payload["tube"]["deformation"]["flow_cycles"] = 2.5

    cfg2 = load_config_dict(payload)
    payload2 = canonical_config_dict(cfg2)
    assert payload2["tube"]["deformation"]["flow_mode"] == "linear"
    assert payload2["tube"]["deformation"]["flow_cycles"] == 2.5
    assert config_hash(cfg2) == config_hash(cfg2)


def test_legacy_tube_config_migrates_to_v2_schema():
    legacy = canonical_config_dict(load_config("configs/tube_default.json"))
    legacy["tube"] = {
        "length": 12.0,
        "radial_samples": 16,
        "longitudinal_samples": 18,
        "base_radius": 1.4,
        "min_radius": 0.4,
        "bump_count": 1,
        "bump_amp_min": 0.01,
        "bump_amp_max": 0.02,
        "bump_sigma_long": 0.05,
        "bump_sigma_theta": 0.05,
        "curve_amplitude_x": 0.0,
        "curve_amplitude_z": 0.0,
        "curve_frequency_x": 0.0,
        "curve_frequency_z": 0.0,
        "camera_offset_ratio": 0.0,
        "camera_wobble_ratio": 0.0,
        "camera_wobble_hz": 0.2,
        "lookahead_distance": 1.0,
        "localized_deform_count": 1,
        "localized_longitudinal_min": 0.5,
        "localized_longitudinal_max": 0.5,
        "localized_theta_center": 0.25,
        "localized_theta_jitter": 0.0,
        "localized_sigma_long_min": 0.1,
        "localized_sigma_long_max": 0.1,
        "localized_sigma_theta_min": 0.1,
        "localized_sigma_theta_max": 0.1,
        "localized_amp_scale": 1.0,
        "localized_negative_ratio": 0.0,
        "localized_tint_enabled": True,
        "localized_tint_bgr": [180, 70, 200],
        "localized_tint_strength": 0.2,
        "deformation_flow_mode": "linear",
        "deformation_flow_cycles": 1.5,
    }
    cfg = load_config_dict(legacy)
    migrated = canonical_config_dict(cfg)
    assert migrated["tube"]["axis"]["sections"] == [{"kind": "straight", "length": 12.0}]
    assert migrated["tube"]["dimensions"]["radial_samples"] == 16
    assert migrated["tube"]["dimensions"]["shell_thickness"] == 0.25
    assert len(migrated["tube"]["deformation"]["areas"]) == 1
    assert migrated["tube"]["deformation"]["flow_mode"] == "linear"


def test_tube_shell_thickness_roundtrips_in_v2_schema():
    payload = canonical_config_dict(load_config("configs/tube_default.json"))
    payload["tube"]["dimensions"]["shell_thickness"] = 0.42

    cfg = load_config_dict(payload)
    assert cfg.tube is not None
    assert cfg.tube.shell_thickness == 0.42
    assert canonical_config_dict(cfg)["tube"]["dimensions"]["shell_thickness"] == 0.42


def test_tube_fea_fields_roundtrip_and_hash_stability():
    payload = canonical_config_dict(load_config("configs/tube_default.json"))
    payload["tube"]["deformation"]["fea"] = {
        "young_modulus_pa": 1.5e6,
        "poisson_ratio": 0.32,
        "axial_elements": 4,
        "theta_elements": 8,
    }
    payload["tube"]["deformation"]["areas"] = [
        {
            "id": "fea_1",
            "enabled": True,
            "kind": "fea_c3d6",
            "center_s": 0.5,
            "center_theta": 0.25,
            "sigma_s": 0.1,
            "sigma_theta": 0.1,
            "direction": {"radial": 1.0, "tangent": 0.0, "circumferential": 0.0},
            "amplitude_out": 0.2,
            "amplitude_in": 0.2,
            "force_n": 150.0,
            "frequency_hz": 0.5,
            "phase_rad": 0.0,
            "mode": "radial",
            "spatial_cycles_s": 0.0,
            "spatial_cycles_theta": 0.0,
            "flow_dir_s": 0.0,
            "flow_dir_theta": 0.0,
        }
    ]

    cfg = load_config_dict(payload)
    assert cfg.tube is not None
    out = canonical_config_dict(cfg)
    assert out["tube"]["deformation"]["fea"]["young_modulus_pa"] == 1.5e6
    assert out["tube"]["deformation"]["fea"]["axial_elements"] == 4
    assert out["tube"]["deformation"]["areas"][0]["kind"] == "fea_c3d6"
    assert out["tube"]["deformation"]["areas"][0]["force_n"] == 150.0
    assert config_hash(cfg) == config_hash(load_config_dict(out))
