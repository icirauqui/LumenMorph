from dataclasses import replace
import json

import numpy as np

from backend.src.config import config_hash, load_config
from backend.src.pipeline import generate_dataset
from backend.src.validator import validate_dataset


def test_pipeline_smoke(tmp_path):
    cfg = load_config("configs/default.json")
    cfg = replace(
        cfg,
        sequence_count={"train": 1, "val": 1, "test": 0},
        intrinsics=replace(cfg.intrinsics, width=160, height=120, fx=130.0, fy=130.0, cx=80.0, cy=60.0),
        sequence=replace(cfg.sequence, num_frames=10, fps=10, mesh_nx=20, mesh_ny=20, x_min=-6.0, x_max=6.0, y_min=0.0, y_max=12.0),
        plane=replace(
            cfg.plane,
            landmark_count=0,
            rigid_ratio=0.0,
            rigid_colorize=False,
            flip_vertical_view=False,
        ),
        texture=replace(cfg.texture, width=256, height=256, min_orb_features=80, max_attempts=4),
        validation=replace(
            cfg.validation,
            min_orb_features_mean=20.0,
            min_frame_motion=1e-4,
            min_camera_clearance=0.02,
            max_abs_mesh_z=50.0,
        ),
    )

    out_dir = tmp_path / "dataset"
    manifest = generate_dataset(cfg, out_dir)

    assert manifest["total_sequences"] == 2
    assert len(manifest["splits"]["train"]) == 1
    assert len(manifest["splits"]["val"]) == 1

    result = validate_dataset(out_dir)
    assert result.ok, f"validation failed: {result.errors}"

    # Regression sanity: deterministic metadata + mesh for fixed seed/config.
    out_dir_2 = tmp_path / "dataset_2"
    generate_dataset(cfg, out_dir_2)

    seq1 = out_dir / "train" / "seq_000"
    seq2 = out_dir_2 / "train" / "seq_000"

    meta1 = json.loads((seq1 / "sequence_meta.json").read_text())
    meta2 = json.loads((seq2 / "sequence_meta.json").read_text())
    assert meta1["seed"] == meta2["seed"]
    assert meta1["preset"] == meta2["preset"]
    assert meta1["wave_components"] == meta2["wave_components"]

    mesh1 = np.load(seq1 / "mesh_z" / "000000.npy")
    mesh2 = np.load(seq2 / "mesh_z" / "000000.npy")
    assert np.allclose(mesh1, mesh2)

    assert config_hash(cfg) == config_hash(cfg)
