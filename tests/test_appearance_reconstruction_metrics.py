import struct

import numpy as np
import pytest

from backend.src.appearance_reconstruction_metrics import (
    F, fit_similarity, pose_metrics, read_images, read_points, point_triangle_squared, surface_distances,
)


def test_similarity_recovers_proper_gauge_and_rejects_unidentified_gauge():
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
    y = 2.3*x@R.T+[7, 8, 9]
    fit = fit_similarity(x, y)
    np.testing.assert_allclose(fit['R'], R, atol=1e-14)
    np.testing.assert_allclose(fit['scale'], 2.3, atol=1e-14)
    np.testing.assert_allclose(fit['t'], [7, 8, 9], atol=1e-14)
    for bad in [np.zeros((4, 3)), np.c_[np.arange(4), np.zeros((4, 2))]]:
        with pytest.raises(ValueError):
            fit_similarity(bad, bad)


def test_pose_metrics_respect_world_and_camera_reflection_and_missing_coverage():
    centers = np.array([[0, 0, 0], [.01, 0, 0], [0, .02, 0], [0, 0, .03]])
    a = .3
    R = np.array([[1., 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    states = {'C_world_meter': centers, 'R_wc': np.repeat(R[None], 4, axis=0)}
    names = [f'{i:06d}.png' for i in range(4)]
    images = {n: {'C': centers[i]@F.T, 'R_wc': F@R@F} for i, n in enumerate(names)}
    score, _ = pose_metrics(images, names, states)
    assert score['ate_mm']['max'] < 1e-12 and score['orientation_degrees']['max'] < 2e-6
    partial, _ = pose_metrics({k: images[k] for k in names[:3]}, names, states)
    assert partial['registered_fraction'] == .75 and partial['missing_names'] == [names[3]]
    absent, fit = pose_metrics({}, names, states)
    assert fit is None and 'unavailable_reason' in absent and absent['registered_fraction'] == 0


def test_binary_images_read_extended_keypoints_and_refuse_standard_short_record(tmp_path):
    p = tmp_path/'images.bin'
    header = struct.pack('<Q', 1)+struct.pack('<I7dI', 7, 1, 0, 0, 0, -1, -2, -3, 1)+b'000000.png\0'+struct.pack('<Q', 1)
    p.write_bytes(header+struct.pack('<2d2Q', .5, .5, 3, 4))
    image = read_images(p)['000000.png']
    np.testing.assert_array_equal(image['C'], [1, 2, 3])
    p.write_bytes(header+struct.pack('<2dQ', .5, .5, 3))
    with pytest.raises(ValueError, match='Truncated extended'):
        read_images(p)


def test_binary_points_preserve_ids_coordinates_and_validate_tracks(tmp_path):
    p = tmp_path/'points3D.bin'
    data = struct.pack('<Q', 1)+struct.pack('<Q3d3Bd', 42, .1, .2, .3, 1, 2, 3, .4)+struct.pack('<Q', 1)+struct.pack('<II', 7, 9)
    p.write_bytes(data)
    ids, xyz = read_points(p)
    assert ids.tolist() == [42]
    np.testing.assert_array_equal(xyz, [[.1, .2, .3]])
    p.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='Truncated point track'):
        read_points(p)


def test_triangle_distance_plane_edges_corners_and_skinny_triangle():
    tri = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
    for p, squared in [([.2, .3, 2], 4), ([.6, .6, 0], .02), ([-1, -2, 0], 5), ([2, 0, 0], 1)]:
        np.testing.assert_allclose(point_triangle_squared(np.array(p), tri), [squared], atol=1e-14)
    thin = np.array([[[0., 0, 0], [1, 0, 0], [1, 1e-10, 0]]])
    np.testing.assert_allclose(point_triangle_squared(np.array([.75, 2e-11, 1]), thin), [1.], atol=1e-14)


def test_conservative_tree_pruning_matches_all_triangles_with_large_remote_centroid():
    pytest.importorskip('scipy')
    # The huge triangle is close to the query despite a far-away centroid.
    triangles = np.array([[[-100., -100, 0], [100, -100, 0], [0, 100, 0]],
                          [[0, 0, 2.], [1, 0, 2], [0, 1, 2]],
                          [[10, 0, -3.], [11, 0, -3], [10, 1, -3]]])
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    rng = np.random.default_rng(42)
    points = np.r_[[[0, 0, .01]], rng.uniform(-5, 5, size=(20, 3))]
    reference = np.array([np.sqrt(point_triangle_squared(p, triangles).min()) for p in points])
    np.testing.assert_allclose(surface_distances(points, vertices, faces), reference, atol=1e-13)


def test_unused_vertices_do_not_supply_an_invalid_surface_distance_upper_bound():
    pytest.importorskip('scipy')
    vertices = np.array([[-1., -1, 2], [1, -1, 2], [0, 1, 2], [0, 0, 0]])
    faces = np.array([[0, 1, 2]])
    np.testing.assert_allclose(surface_distances(np.array([[0., 0, 0]]), vertices, faces), [2.])
