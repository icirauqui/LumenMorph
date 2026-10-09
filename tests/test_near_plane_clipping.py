"""Independent original-triangle ray checks, without a clipping oracle."""
import numpy as np
import pytest
from backend.src.dense_truth import flow_to_frame, query_surface_depth, render_rgb_and_truth
from backend.src.renderer import render_mesh
from backend.src.types import CameraIntrinsics

INTR = CameraIntrinsics(width=32, height=24, fx=18., fy=19., cx=16., cy=12.)
TRI = np.array([[0, 1, 2]], dtype=np.int32)
UV = np.array([[.13, .22], [.78, .31], [.42, .86]])
R = np.eye(3)
C = np.zeros(3)


def ray_oracle(vertices):
    rows, cols = np.indices((INTR.height, INTR.width))
    rays = np.stack(((cols+.5-INTR.cx)/INTR.fx, (INTR.cy-rows-.5)/INTR.fy,
                     np.ones_like(rows)), axis=-1)
    bary = np.full((*rows.shape, 3), np.nan)
    depth = np.full(rows.shape, np.inf)
    for row, col in np.ndindex(rows.shape):
        matrix = np.column_stack((vertices[1]-vertices[0], vertices[2]-vertices[0], -rays[row, col]))
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        a, b, z = np.linalg.solve(matrix, -vertices[0])
        weights = np.array([1-a-b, a, b])
        if z >= 1e-5 and np.min(weights) > 1e-8:
            depth[row, col] = z
            bary[row, col] = weights
    return depth, bary


@pytest.mark.parametrize('vertices', [
    [[-.8,-.6,-.2], [.9,-.5,1.3], [0.,.8,1.1]],
    [[-.8,-.6,0.], [.9,-.5,1.3], [0.,.8,1.1]],
    [[-.8,-.6,-.4], [.9,-.5,-.1], [0.,.8,1.1]],
    [[-.8,-.6,1e-5], [.9,-.5,1.3], [0.,.8,1.1]],
])
def test_clipped_material_matches_original_triangle_rays(vertices):
    vertices = np.array(vertices)
    texture = np.arange(16*16*3, dtype=np.uint16).reshape(16,16,3).astype(np.uint8)
    rgb, truth = render_rgb_and_truth(vertices, TRI, UV, texture, INTR, R, C, high_precision=True)
    expected_depth, expected_bary = ray_oracle(vertices)
    expected = np.isfinite(expected_depth)
    assert expected.sum() > 10
    np.testing.assert_array_equal(truth['valid'], expected)
    assert np.all(truth['triangle_id'][expected] == 0)
    np.testing.assert_allclose(truth['depth_z'][expected], expected_depth[expected], rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(truth['material_barycentric'][expected], expected_bary[expected], rtol=2e-6, atol=1e-7)
    rows, cols = np.nonzero(expected)
    query = query_surface_depth(np.column_stack((cols+.5,rows+.5)), vertices, TRI, INTR, R, C, high_precision=True)
    np.testing.assert_allclose(query, expected_depth[expected], rtol=1e-8, atol=1e-9)
    tex_uv = truth['material_barycentric'][expected].astype(float) @ UV
    tx = (tex_uv[:,0]*15).astype(int)
    ty = ((1-tex_uv[:,1])*15).astype(int)
    np.testing.assert_array_equal(rgb[expected], texture[ty,tx])
    preview = render_mesh(vertices, TRI, UV, texture, INTR, R, C)
    np.testing.assert_array_equal(preview[expected], rgb[expected])
    flow = flow_to_frame(truth, vertices, TRI, INTR, R, C, high_precision=True)
    assert np.all(flow['visible'][expected])
    np.testing.assert_allclose(flow['flow_uv'][expected], 0., atol=2e-5)


def test_fully_behind_triangle_stays_invalid():
    vertices = np.array([[-.8,-.6,-.2],[.9,-.5,-1.3],[0.,.8,0.]])
    _, truth = render_rgb_and_truth(vertices, TRI, UV, np.zeros((1,1,3),dtype=np.uint8), INTR, R, C)
    assert not truth['valid'].any()
    assert np.all(truth['triangle_id'] == -1)
    assert np.isnan(truth['material_barycentric']).all()
    assert np.isinf(query_surface_depth([[16.,12.]],vertices,TRI,INTR,R,C)).all()


@pytest.mark.parametrize('reverse_order', [False, True])
def test_nearer_unclipped_occluder_retains_own_identity(reverse_order):
    crossing = np.array([[-.8,-.6,-.2],[.9,-.5,1.3],[0.,.8,1.1]])
    cover = np.array([[-2.,-2.,.01],[2.,-2.,.01],[0.,2.,.01]])
    vertices = np.concatenate((crossing,cover))
    faces = np.array([[0,1,2],[3,4,5]])
    if reverse_order:
        faces = faces[::-1]
    _, truth = render_rgb_and_truth(vertices,faces,np.tile(UV,(2,1)),np.zeros((1,1,3),dtype=np.uint8),INTR,R,C,high_precision=True)
    assert truth['valid'].all()
    assert np.all(truth['triangle_id'] == (0 if reverse_order else 1))
    weights = truth['material_barycentric']
    assert np.isfinite(weights).all() and np.all(weights >= 0)
    np.testing.assert_allclose((weights @ cover)[...,2],.01,atol=1e-8)
