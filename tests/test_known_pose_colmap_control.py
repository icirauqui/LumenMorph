import sqlite3

import numpy as np

from backend.build_known_pose_colmap_control import (
    _format_fork_observations,
    GENERATOR_TO_COLMAP_CAMERA,
    GENERATOR_TO_COLMAP_WORLD,
    _copy_database_bundle,
    _database_bundle_hashes,
    _database_keypoints,
    _quaternion_wxyz_from_rotation,
)
from backend.evaluate_colmap_tube import _rotation_from_quaternion_wxyz


def test_rotation_quaternion_round_trip():
    rotation = np.array(
        [[0.36, -0.48, 0.8], [0.8, 0.60, 0.0], [-0.48, 0.64, 0.60]],
        dtype=np.float64,
    )
    quaternion = _quaternion_wxyz_from_rotation(rotation)
    recovered = _rotation_from_quaternion_wxyz(quaternion)

    assert np.allclose(recovered, rotation)


def test_paired_reflections_preserve_rendered_pixels_with_proper_rotation():
    rotation_wc_generator = np.array(
        [[0.36, -0.48, 0.8], [0.8, 0.60, 0.0], [-0.48, 0.64, 0.60]],
        dtype=np.float64,
    )
    centre_generator = np.array([1.2, -0.4, 2.1])
    point_camera_generator = np.array([0.3, -0.2, 4.0])
    point_generator = (
        rotation_wc_generator @ point_camera_generator + centre_generator
    )

    centre_colmap = GENERATOR_TO_COLMAP_WORLD @ centre_generator
    point_colmap = GENERATOR_TO_COLMAP_WORLD @ point_generator
    rotation_wc_colmap = (
        GENERATOR_TO_COLMAP_WORLD
        @ rotation_wc_generator
        @ GENERATOR_TO_COLMAP_CAMERA
    )
    point_camera_colmap = rotation_wc_colmap.T @ (point_colmap - centre_colmap)

    fx, fy, cx, cy = 560.0, 545.0, 400.0, 300.0
    rendered = np.array(
        [
            fx * point_camera_generator[0] / point_camera_generator[2] + cx,
            cy - fy * point_camera_generator[1] / point_camera_generator[2],
        ]
    )
    projected_colmap = np.array(
        [
            fx * point_camera_colmap[0] / point_camera_colmap[2] + cx,
            fy * point_camera_colmap[1] / point_camera_colmap[2] + cy,
        ]
    )

    assert np.isclose(np.linalg.det(rotation_wc_colmap), 1.0)
    assert point_camera_colmap[2] > 0.0
    assert np.allclose(projected_colmap, rendered)


def test_database_keypoints_reads_xy_and_validates_blob_shape():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)"
    )
    raw = np.array([[1.25, 2.5, 0.7, 0.1], [3.0, 4.0, 0.8, 0.2]], dtype=np.float32)
    connection.execute(
        "INSERT INTO keypoints VALUES (?, ?, ?, ?)", (7, 2, 4, raw.tobytes())
    )

    keypoints = _database_keypoints(connection)

    assert set(keypoints) == {7}
    assert np.allclose(keypoints[7], [[1.25, 2.5], [3.0, 4.0]])


def test_fork_observation_format_includes_deformable_point_identifier():
    keypoints = np.array([[1.25, 2.5], [3.0, 4.0]], dtype=np.float64)
    assert _format_fork_observations(keypoints) == (
        "1.25 2.5 -1 -1 3 4 -1 -1"
    )


def test_copy_database_bundle_preserves_source_and_companions(tmp_path):
    source = tmp_path / "source.db"
    source.write_bytes(b"database")
    (tmp_path / "source.db-wal").write_bytes(b"wal")
    (tmp_path / "source.db-shm").write_bytes(b"shm")
    before = _database_bundle_hashes(source)

    destination = tmp_path / "copy.db"
    _copy_database_bundle(source, destination)

    assert destination.read_bytes() == b"database"
    assert (tmp_path / "copy.db-wal").read_bytes() == b"wal"
    assert (tmp_path / "copy.db-shm").read_bytes() == b"shm"
    assert _database_bundle_hashes(source) == before
