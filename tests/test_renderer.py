import numpy as np

from backend.src.renderer import (
    _perspective_correct_grid,
    perspective_correct_barycentric,
    rasterize_mesh_buffers,
    render_mesh,
)
from backend.src.types import CameraIntrinsics


def test_zbuffer_keeps_near_triangle():
    intr = CameraIntrinsics(width=64, height=48, fx=60.0, fy=60.0, cx=32.0, cy=24.0)

    vertices = np.array(
        [
            [-0.5, -0.4, 4.0],
            [0.5, -0.4, 4.0],
            [0.0, 0.5, 4.0],
            [-0.5, -0.4, 2.0],
            [0.5, -0.4, 2.0],
            [0.0, 0.5, 2.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    uv = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    # texture[1,0] -> green (far), texture[0,1] -> red (near)
    texture = np.array(
        [
            [[255, 0, 0], [0, 0, 255]],
            [[0, 255, 0], [0, 255, 255]],
        ],
        dtype=np.uint8,
    )

    image = render_mesh(
        vertices_world=vertices,
        triangles=triangles,
        uv_vertices=uv,
        texture_bgr=texture,
        intr=intr,
        R_wc=np.eye(3, dtype=np.float32),
        t_wc=np.zeros(3, dtype=np.float32),
    )

    center = image[24, 32]
    assert np.array_equal(center, np.array([0, 0, 255], dtype=np.uint8))

    buffers = rasterize_mesh_buffers(
        vertices_world=vertices,
        triangles=triangles,
        intr=intr,
        R_wc=np.eye(3, dtype=np.float32),
        t_wc=np.zeros(3, dtype=np.float32),
    )
    assert buffers.triangle_index[24, 32] == 1
    assert np.isclose(np.sum(buffers.screen_barycentric[24, 32]), 1.0)
    assert np.isclose(buffers.depth[24, 32], 2.0)


def test_perspective_correct_barycentric_reprojects_to_screen_point():
    vertices = np.array(
        [[-1.0, -0.5, 2.0], [1.0, -0.5, 4.0], [0.0, 1.0, 3.0]],
        dtype=np.float64,
    )
    projected = vertices[:, :2] / vertices[:, 2, None]
    screen_weights = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    target = screen_weights @ projected

    physical_weights = perspective_correct_barycentric(
        screen_weights, vertices[:, 2]
    )
    physical_point = physical_weights @ vertices
    assert np.allclose(physical_point[:2] / physical_point[2], target)

    affine_point = screen_weights @ vertices
    assert not np.allclose(affine_point[:2] / affine_point[2], target)


def test_perspective_grid_matches_analytic_barycentrics_and_depth():
    screen = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    depths = np.array([2.0, 4.0, 3.0], dtype=np.float64)
    p0, p1, p2, depth = _perspective_correct_grid(
        screen[0:1], screen[1:2], screen[2:3], *depths
    )
    physical = perspective_correct_barycentric(screen, depths)
    assert np.allclose(np.array([p0[0], p1[0], p2[0]]), physical)
    assert np.isclose(depth[0], physical @ depths)


def test_flat_texture_fast_path_renders_constant_color():
    intr = CameraIntrinsics(width=48, height=36, fx=50.0, fy=50.0, cx=24.0, cy=18.0)
    vertices = np.array(
        [
            [-0.5, -0.4, 3.0],
            [0.5, -0.4, 3.0],
            [0.0, 0.5, 3.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.zeros((3, 2), dtype=np.float32)
    flat_gray = np.array([[[196, 196, 196]]], dtype=np.uint8)

    image = render_mesh(
        vertices_world=vertices,
        triangles=triangles,
        uv_vertices=uv,
        texture_bgr=flat_gray,
        intr=intr,
        R_wc=np.eye(3, dtype=np.float32),
        t_wc=np.zeros(3, dtype=np.float32),
    )

    assert np.array_equal(image[18, 24], np.array([196, 196, 196], dtype=np.uint8))


def test_face_shading_changes_flat_preview_color():
    intr = CameraIntrinsics(width=48, height=36, fx=50.0, fy=50.0, cx=24.0, cy=18.0)
    vertices = np.array(
        [
            [-0.5, -0.4, 3.0],
            [0.5, -0.4, 3.0],
            [0.0, 0.5, 3.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.zeros((3, 2), dtype=np.float32)
    flat_gray = np.array([[[196, 196, 196]]], dtype=np.uint8)

    image = render_mesh(
        vertices_world=vertices,
        triangles=triangles,
        uv_vertices=uv,
        texture_bgr=flat_gray,
        intr=intr,
        R_wc=np.eye(3, dtype=np.float32),
        t_wc=np.zeros(3, dtype=np.float32),
        shade_faces=True,
    )

    center = image[18, 24]
    assert not np.array_equal(center, np.array([196, 196, 196], dtype=np.uint8))
    assert np.all(center > 120)
