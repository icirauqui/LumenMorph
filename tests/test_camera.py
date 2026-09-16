import numpy as np

from backend.src.api.preview import _preview_intrinsics, _scene_view_pose
from backend.src.camera import project_points, world_to_camera
from backend.src.types import CameraIntrinsics


def test_projection_identity_pose():
    intr = CameraIntrinsics(width=640, height=480, fx=100.0, fy=100.0, cx=320.0, cy=240.0)

    points_world = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
        ],
        dtype=np.float32,
    )
    R_wc = np.eye(3, dtype=np.float32)
    t_wc = np.zeros(3, dtype=np.float32)

    points_cam = world_to_camera(points_world, R_wc, t_wc)
    uv, valid = project_points(points_cam, intr)

    assert valid.all()
    assert np.allclose(uv[0], [320.0, 240.0], atol=1e-5)
    assert np.allclose(uv[1], [340.0, 240.0], atol=1e-5)
    assert np.allclose(uv[2], [320.0, 220.0], atol=1e-5)


def test_preview_camera_default_fit_shows_long_tube_bounds():
    length = 42.0
    radius = 2.4
    vertices = np.array(
        [
            [x, y, z]
            for x in (0.0, length)
            for y in (-radius, radius)
            for z in (-radius, radius)
        ],
        dtype=np.float32,
    )
    intr = _preview_intrinsics(width=1000, height=1200, fov_deg=55.0)
    t_wc, R_wc = _scene_view_pose(
        vertices,
        yaw_deg=42.0,
        pitch_deg=18.0,
        distance_scale=1.45,
        fov_deg=55.0,
        width=intr.width,
        height=intr.height,
    )

    points_cam = world_to_camera(vertices, R_wc, t_wc)
    uv, valid = project_points(points_cam, intr)

    assert valid.all()
    assert float(np.min(uv[:, 0])) > 0.0
    assert float(np.max(uv[:, 0])) < intr.width
    assert float(np.min(uv[:, 1])) > 0.0
    assert float(np.max(uv[:, 1])) < intr.height
