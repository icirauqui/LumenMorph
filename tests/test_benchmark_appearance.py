import numpy as np
import pytest

from backend.src.benchmark_appearance import (
    srgb_to_linear, exposed_rgb, proper_rotation, assert_shared_geometry, GEOMETRY_LABEL_KEYS,
)


def test_camera_response_roundtrip_all_8bit_codes_and_exposure_clipping():
    codes = np.repeat(np.arange(256)[:, None], 3, axis=1).astype(np.uint8)
    np.testing.assert_array_equal(exposed_rgb(srgb_to_linear(codes/255), 1), codes)
    # Independent sRGB anchors and exact linear gain before encoding/clipping.
    np.testing.assert_allclose(srgb_to_linear([[.04045, .5, 1.]]),
                               [[.0031308049535603713, .21404114048223255, 1]], atol=1e-14)
    np.testing.assert_array_equal(exposed_rgb([[0., .5, 2.]], 2), [[0, 255, 255]])


@pytest.mark.parametrize('bad', [[[-1., 0, 0]], [[np.nan, 0, 0]], [[np.inf, 0, 0]]])
def test_invalid_radiance_is_not_silently_clipped(bad):
    with pytest.raises(ValueError):
        exposed_rgb(bad, 1)


def test_rounded_rotation_is_corrected_but_reflection_and_distortion_rejected():
    theta = .123
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    corrected = proper_rotation(R.astype(np.float32))
    np.testing.assert_allclose(corrected.T@corrected, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(corrected, R, atol=1e-8)
    for bad in [np.diag([-1, 1, 1]), np.diag([1.01, 1, 1]), np.full((3, 3), np.nan)]:
        with pytest.raises(ValueError):
            proper_rotation(bad)


def test_geometry_sharing_checks_masks_identity_and_nan_background():
    a = {k: np.array([1., np.nan]) for k in GEOMETRY_LABEL_KEYS}
    a['mesh_topology_sha256'] = np.asarray('abc')
    assert_shared_geometry(a, a)
    for key in GEOMETRY_LABEL_KEYS:
        b = dict(a)
        b[key] = np.asarray('def') if key == 'mesh_topology_sha256' else np.array([2., np.nan])
        with pytest.raises(ValueError, match=key):
            assert_shared_geometry(a, b)


def test_full_adapter_render_preserves_source_and_uses_converted_truth(tmp_path):
    pytest.importorskip('mitsuba')
    import cv2
    import json
    from backend.src.benchmark_appearance import BaselineAppearanceSequence, sha256
    from backend.src.endoscopic_renderer import AppearanceSettings
    package = tmp_path/'suite/v2'
    family, moving, public = package/'evaluation/family', package/'evaluation/moving', package/'public'
    for p in [family, moving, public]:
        p.mkdir(parents=True)
    vertices = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]], dtype=np.float32)
    np.savez(family/'geometry.npz', vertices=vertices, triangles=np.array([[0, 1, 2], [0, 2, 3]]),
             uv=np.array([[.1, .1], [.4, .1], [.4, .9], [.1, .9]]), base_radius=np.ones(4))
    np.savez(moving/'states.npz', vertices=vertices[None], R_wc=np.eye(3)[None].astype(np.float32),
             C_world=np.zeros((1, 3)), time_seconds=np.array([6.]))
    cv2.imwrite(str(family/'texture.png'), np.full((8, 8, 3), 128, dtype=np.uint8))
    (public/'intrinsics.json').write_text(json.dumps(dict(width=8, height=8, fx=12, fy=12, cx=4, cy=4)))
    (package/'DATASET.json').write_text(json.dumps(dict(schema='meshmeld_tubular_benchmark_v1',
        family='f01', condition='fem', split='development')))
    before = {str(p): sha256(p) for p in package.rglob('*') if p.is_file()}
    seq = BaselineAppearanceSequence(package)
    assert not seq.albedo.flags.writeable
    rgb, truth, metadata = seq.render(0, AppearanceSettings(material='diffuse', samples_per_pixel=1, threads=1))
    assert rgb.shape == (8, 8, 3) and truth['valid'].all()
    np.testing.assert_allclose(truth['depth_z'], .03, atol=1e-8)
    assert metadata['baseline_adapter']['source_to_meter_scale'] == .015
    assert metadata['baseline_adapter']['time_seconds'] == 6
    assert {str(p): sha256(p) for p in package.rglob('*') if p.is_file()} == before
    for bad in [-1, 1, True]:
        with pytest.raises(IndexError):
            seq.state(bad)
