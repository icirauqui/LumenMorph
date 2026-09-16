import numpy as np

from backend.evaluate_colmap_tube import (
    _fit_orientation_similarity,
    _fit_similarity,
    _select_similarity,
)


def test_fit_similarity_recovers_known_transform():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scale = 2.5
    translation = np.array([4.0, -3.0, 7.0])
    target = (scale * (rotation @ source.T)).T + translation

    fit_scale, fit_rotation, fit_translation = _fit_similarity(source, target)

    assert np.isclose(fit_scale, scale)
    assert np.allclose(fit_rotation, rotation)
    assert np.allclose(fit_translation, translation)


def test_fit_orientation_similarity_recovers_known_transform():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    scale = 2.5
    translation = np.array([4.0, -3.0, 7.0])
    target = (scale * (rotation @ source.T)).T + translation
    source_rotations = np.repeat(np.eye(3)[None, :, :], len(source), axis=0)
    target_rotations = np.repeat(rotation[None, :, :], len(source), axis=0)

    fit_scale, fit_rotation, fit_translation = _fit_orientation_similarity(
        source, target, source_rotations, target_rotations
    )

    assert np.isclose(fit_scale, scale)
    assert np.allclose(fit_rotation, rotation)
    assert np.allclose(fit_translation, translation)


def test_select_similarity_uses_positions_for_curved_trajectory():
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.3]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = (2.0 * (rotation @ source.T)).T + np.array([3.0, 4.0, 5.0])
    source_rotations = np.repeat(np.eye(3)[None, :, :], len(source), axis=0)
    # Deliberately inconsistent orientations must not bias a curved path fit.
    target_rotations = np.repeat(np.eye(3)[None, :, :], len(source), axis=0)

    scale, fit_rotation, translation, source_name, _ = _select_similarity(
        source, target, source_rotations, target_rotations
    )

    assert source_name == "registered_camera_centres"
    assert np.isclose(scale, 2.0)
    assert np.allclose(fit_rotation, rotation)
    assert np.allclose(translation, [3.0, 4.0, 5.0])


def test_select_similarity_uses_orientations_for_collinear_roll_gauge():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    target = (3.0 * (rotation @ source.T)).T + np.array([2.0, 3.0, 4.0])
    source_rotations = np.repeat(np.eye(3)[None, :, :], len(source), axis=0)
    target_rotations = np.repeat(rotation[None, :, :], len(source), axis=0)

    scale, fit_rotation, translation, source_name, _ = _select_similarity(
        source, target, source_rotations, target_rotations
    )

    assert source_name == "camera_orientations_for_rank_deficient_position_gauge"
    assert np.isclose(scale, 3.0)
    assert np.allclose(fit_rotation, rotation)
    assert np.allclose(translation, [2.0, 3.0, 4.0])
