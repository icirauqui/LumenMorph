import numpy as np

from backend.src.plane import add_landmarks_and_rigid_regions
from backend.src.types import PlaneSceneConfig


def test_plane_landmarks_and_rigid_mask_generation():
    x, y = np.meshgrid(np.linspace(-5, 5, 40), np.linspace(0, 12, 40))
    base = np.zeros_like(x, dtype=np.float32)

    cfg = PlaneSceneConfig(
        landmark_count=3,
        landmark_height_min=1.5,
        landmark_height_max=3.0,
        landmark_sigma_min=0.8,
        landmark_sigma_max=1.6,
        rigid_ratio=0.3,
        rigid_include_landmarks=True,
        rigid_edge_softness=1,
        rigid_colorize=True,
        rigid_tint_bgr=[40, 180, 255],
        rigid_tint_strength=0.4,
        flip_vertical_view=False,
    )

    z, rigid, landmarks, rigid_fraction, rigid_pattern = add_landmarks_and_rigid_regions(
        x_grid=x,
        y_grid=y,
        base_z=base,
        cfg=cfg,
        rng=np.random.default_rng(7),
    )

    assert z.shape == base.shape
    assert rigid.shape == base.shape
    assert len(landmarks) == 3
    assert rigid_pattern in {"corner", "side", "band_horizontal", "band_vertical", "band_diag_pos", "band_diag_neg"}
    assert float(np.max(z)) > 0.5
    assert 0.15 <= rigid_fraction <= 0.7

    # Fully rigid points should not move under deformation blending.
    disp = np.ones_like(z, dtype=np.float32) * 0.8
    z_deformed = z + disp * (1.0 - rigid)
    assert np.allclose(z_deformed[rigid >= 0.999], z[rigid >= 0.999])
