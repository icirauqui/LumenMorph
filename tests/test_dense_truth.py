import numpy as np
import json
from pathlib import Path

from backend.src.dense_truth import flow_to_frame, query_surface_depth, render_rgb_and_truth
from backend.src.renderer import rasterize_mesh_buffers
from backend.src.types import CameraIntrinsics


INTR = CameraIntrinsics(width=48, height=36, fx=30., fy=30., cx=24., cy=18.)
IDENTITY = np.eye(3, dtype=np.float32)
ZERO = np.zeros(3, dtype=np.float32)


def render(vertices, triangles, uv=None, texture=None):
    if uv is None:
        uv = np.zeros((len(vertices), 2), dtype=np.float32)
    if texture is None:
        texture = np.full((1, 1, 3), 128, dtype=np.uint8)
    return render_rgb_and_truth(vertices, triangles, uv, texture, INTR, IDENTITY, ZERO)


def test_slanted_triangle_truth_matches_independent_ray_intersection_and_rgb():
    vertices = np.array([[-2., -1.5, 2.], [2., -1.5, 5.], [0., 2., 3.]])
    triangles = np.array([[0, 1, 2]])
    uv = np.array([[0., 0.], [1., 0.], [.5, 1.]])
    texture = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    rgb, truth = render(vertices, triangles, uv, texture)
    r, c = 18, 24
    ray = np.array([(c + .5 - INTR.cx) / INTR.fx, (INTR.cy - r - .5) / INTR.fy, 1.])
    p0, p1, p2 = vertices
    # Solve p0 + a*(p1-p0) + b*(p2-p0) = depth*ray independently.
    a, b, z = np.linalg.solve(np.column_stack((p1 - p0, p2 - p0, -ray)), -p0)
    bary = np.array([1 - a - b, a, b])
    assert truth["valid"][r, c]
    np.testing.assert_allclose(truth["material_barycentric"][r, c], bary, atol=1e-6)
    np.testing.assert_allclose(truth["depth_z"][r, c], z, atol=1e-6)
    material_uv = bary @ uv
    expected_color = texture[int((1 - material_uv[1]) * 15), int(material_uv[0] * 15)]
    np.testing.assert_array_equal(rgb[r, c], expected_color)
    expected_normal = np.cross(p1 - p0, p2 - p0)
    expected_normal /= np.linalg.norm(expected_normal)
    np.testing.assert_allclose(truth["normal_world"][r, c], expected_normal, atol=1e-6)
    invalid = ~truth["valid"]
    assert np.all(np.isnan(truth["depth_z"][invalid]))
    assert np.all(truth["triangle_id"][invalid] == -1)


def test_corrected_depth_ownership_differs_from_preserved_affine_audit():
    # Both triangles have the same image footprint. At center, the slanted
    # triangle lies ahead of z=3 under the physical ray and behind it if affine.
    projected = np.array([[-.8, -.7], [.8, -.7], [0., .8]])
    depths = np.array([1.5, 5., 5.])
    slanted = np.column_stack((projected * depths[:, None], depths))
    flat = np.column_stack((projected * 3.7, np.full(3, 3.7)))
    vertices = np.concatenate((slanted, flat))
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    _, truth = render(vertices, triangles)
    old = rasterize_mesh_buffers(vertices, triangles, INTR, IDENTITY, ZERO)
    assert old.triangle_index[18, 24] == 1
    assert truth["triangle_id"][18, 24] == 0


def test_identity_flow_zero_and_subpixel_visibility_for_slanted_surface():
    vertices = np.array([[-2., -1.5, 2.], [2., -1.5, 5.], [0., 2., 3.]])
    triangles = np.array([[0, 1, 2]])
    _, truth = render(vertices, triangles)
    flow = flow_to_frame(truth, vertices, triangles, INTR, IDENTITY, ZERO)
    np.testing.assert_allclose(flow["flow_uv"][truth["valid"]], 0., atol=1e-5)
    assert np.all(flow["visible"][truth["valid"]])
    assert not np.any(flow["occluded"] | flow["unresolved"])


def test_flow_tracks_material_under_translation_and_detects_occlusion():
    # Source has a near triangle at the right, separate from the far triangle.
    back = np.array([[-1., -1., 4.], [1., -1., 4.], [0., 1., 4.]])
    near = back / 2
    near[:, 0] += 2
    vertices = np.concatenate((back, near))
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    _, truth = render(vertices, triangles)
    target = vertices.copy()
    target[3:, 0] -= 2  # foreground moves over the unchanged background
    flow = flow_to_frame(truth, target, triangles, INTR, IDENTITY, ZERO)
    assert truth["triangle_id"][18, 24] == 0
    assert flow["occluded"][18, 24]
    assert flow["flow_valid"][18, 24]
    np.testing.assert_allclose(flow["flow_uv"][18, 24], 0, atol=1e-5)
    foreground = truth["triangle_id"] == 1
    assert np.any(foreground)
    np.testing.assert_allclose(flow["flow_uv"][foreground, 0], -30., atol=1e-5)
    assert np.all(flow["visible"][foreground])


def test_out_of_image_and_behind_camera_are_not_occlusions():
    vertices = np.array([[-1., -1., 4.], [1., -1., 4.], [0., 1., 4.]])
    triangles = np.array([[0, 1, 2]])
    _, truth = render(vertices, triangles)
    outside = flow_to_frame(truth, vertices + [100, 0, 0], triangles, INTR, IDENTITY, ZERO)
    assert np.all(outside["out_of_frame"][truth["valid"]])
    assert not np.any(outside["occluded"])
    behind = flow_to_frame(truth, vertices - [0, 0, 10], triangles, INTR, IDENTITY, ZERO)
    assert np.all(behind["behind_camera"][truth["valid"]])
    assert not np.any(behind["flow_valid"])


def test_continuous_depth_query_matches_ray_away_from_pixel_centers():
    vertices = np.array([[-2., -1.5, 2.], [2., -1.5, 5.], [0., 2., 3.]])
    triangles = np.array([[0, 1, 2]])
    query_uv = np.array([[24.137, 18.871], [23.919, 17.242]])
    depths = query_surface_depth(query_uv, vertices, triangles, INTR, IDENTITY, ZERO)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    rays = np.column_stack(((query_uv[:, 0] - INTR.cx) / INTR.fx,
                            (INTR.cy - query_uv[:, 1]) / INTR.fy, np.ones(2)))
    expected = (normal @ vertices[0]) / (rays @ normal)
    np.testing.assert_allclose(depths, expected, rtol=1e-6)


def test_near_plane_discard_is_explicit_and_not_false_depth():
    vertices = np.array([[-.1, -.1, -1.], [.1, -.1, 2.], [0., .1, 2.]])
    _, truth = render(vertices, np.array([[0, 1, 2]]))
    assert not np.any(truth["valid"])
    assert np.all(np.isnan(truth["depth_z"]))


def test_periodic_tube_u_uses_seam_colors_without_changing_material_truth():
    vertices = np.array([[-2., -1.5, 2.], [2., -1.5, 5.], [0., 2., 3.]])
    triangles = np.array([[0, 1, 2]])
    uv = np.array([[.98, .3], [.02, .3], [.99, .3]])
    texture = np.full((2, 101, 3), [0, 0, 255], dtype=np.uint8)
    texture[:, :5] = [0, 255, 0]
    texture[:, -5:] = [0, 255, 0]
    historical, old_truth = render(vertices, triangles, uv, texture)
    periodic, truth = render_rgb_and_truth(vertices, triangles, uv, texture, INTR,
                                          IDENTITY, ZERO, periodic_u=True)
    valid = truth["valid"]
    assert np.any(np.all(historical[valid] == [0, 0, 255], axis=1))
    assert np.all(periodic[valid] == [0, 255, 0])
    for key in truth:
        np.testing.assert_array_equal(truth[key], old_truth[key])
    # Check an asymmetric texel independently from stored physical weights.
    texture[0, :, 0] = np.arange(101, dtype=np.uint8)
    texture[1, :, 0] = np.arange(101, dtype=np.uint8)
    periodic, truth = render_rgb_and_truth(vertices, triangles, uv, texture, INTR,
                                          IDENTITY, ZERO, periodic_u=True)
    expected_u = (truth["material_barycentric"][18, 24] @ np.array([.98, 1.02, .99])) % 1.
    assert periodic[18, 24, 0] == int(expected_u * 100)
    shifted_uv = uv.copy()
    shifted_uv[:, 0] += np.array([1., -2., 3.])
    shifted, shifted_truth = render_rgb_and_truth(vertices, triangles, shifted_uv, texture,
                                                 INTR, IDENTITY, ZERO, periodic_u=True)
    np.testing.assert_array_equal(shifted, periodic)
    for key in truth:
        np.testing.assert_array_equal(shifted_truth[key], truth[key])


def test_periodic_u_leaves_nonseam_triangles_bitwise_unchanged():
    vertices = np.array([[-2., -1.5, 2.], [2., -1.5, 5.], [0., 2., 3.]])
    triangles = np.array([[0, 1, 2]])
    uv = np.array([[.31, .2], [.63, .5], [.48, .7]])
    texture = np.random.default_rng(17).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    historical, old_truth = render(vertices, triangles, uv, texture)
    periodic, truth = render_rgb_and_truth(vertices, triangles, uv, texture, INTR,
                                          IDENTITY, ZERO, periodic_u=True)
    np.testing.assert_array_equal(periodic, historical)
    for key in truth:
        np.testing.assert_array_equal(truth[key], old_truth[key])


def test_high_precision_grazing_triangle_matches_independent_world_ray():
    fixture = json.loads((Path(__file__).parent/'fixtures/grazing_triangle_r1.json').read_text())
    vertices = np.array(fixture['vertices_world'], dtype=np.float32)
    rotation = np.array(fixture['R_wc'], dtype=np.float32)
    center = np.array(fixture['C_world'], dtype=np.float32)
    intr = CameraIntrinsics(**fixture['intrinsics'])
    row, col = fixture['pixel_row'], fixture['pixel_column']
    triangles = np.array([[0,1,2]])
    uv = np.zeros((3,2), dtype=np.float32)
    texture = np.full((1,1,3),128,dtype=np.uint8)
    # Exact declared transpose transform, not an orthogonalized approximation
    # to the stored float32 camera rotation. Solve directly in world space.
    camera_ray = np.array([(col+.5-intr.cx)/intr.fx,(intr.cy-row-.5)/intr.fy,1.])
    world_ray = np.linalg.solve(rotation.astype(float).T, camera_ray)
    p0,p1,p2 = vertices.astype(float)
    a,b,depth = np.linalg.solve(np.column_stack([p1-p0,p2-p0,-world_ray]),center.astype(float)-p0)
    expected_bary = np.array([1-a-b,a,b])
    old_rgb, old = render_rgb_and_truth(vertices,triangles,uv,texture,intr,rotation,center)
    explicit_rgb, explicit = render_rgb_and_truth(vertices,triangles,uv,texture,intr,rotation,center,high_precision=False)
    np.testing.assert_array_equal(old_rgb,explicit_rgb)
    for key in old:
        np.testing.assert_array_equal(old[key],explicit[key])
    assert abs(float(old['depth_z'][row,col])-depth) > 2e-4
    _, fixed = render_rgb_and_truth(vertices,triangles,uv,texture,intr,rotation,center,high_precision=True)
    assert fixed['triangle_id'][row,col] == 0
    assert fixed['depth_z'].dtype == np.float32
    assert fixed['material_barycentric'].dtype == np.float32
    assert abs(float(fixed['depth_z'][row,col])-depth) < .6*np.spacing(np.float32(depth))
    np.testing.assert_allclose(fixed['material_barycentric'][row,col],expected_bary,atol=3e-8,rtol=0)
    queried = query_surface_depth([[col+.5,row+.5]],vertices,triangles,intr,rotation,center,high_precision=True)
    np.testing.assert_allclose(queried,[depth],atol=1e-8,rtol=0)
    flow = flow_to_frame(fixed,vertices,triangles,intr,rotation,center,high_precision=True)
    assert np.all(flow['visible'][fixed['valid']])
    np.testing.assert_allclose(flow['flow_uv'][fixed['valid']],0.,atol=1e-5)


def test_high_precision_zbuffer_retains_sub_float32_depth_ordering():
    # Two surfaces differ below a float32 depth ULP. Construct the far face
    # first; quantizing the internal z-buffer would wrongly keep that face.
    depth = 4.0
    far = np.array([[-1.,-1.,depth+1e-8],[1.,-1.,depth+1e-8],[0.,1.,depth+1e-8]])
    near = far.copy();near[:,2]=depth
    vertices = np.concatenate([far,near])
    triangles = np.array([[0,1,2],[3,4,5]])
    buffers = rasterize_mesh_buffers(vertices,triangles,INTR,IDENTITY,ZERO,
                                     perspective_correct=True,high_precision=True)
    assert buffers.depth.dtype == np.float64
    assert buffers.screen_barycentric.dtype == np.float64
    assert buffers.triangle_index[18,24] == 1
