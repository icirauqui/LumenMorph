import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

pytest.importorskip('mitsuba')
from backend.src.appearance_suite import freeze, run, render_frame, publish_frame
from backend.src.appearance_reader import AppearancePublic, AppearanceEvaluation
from backend.src.benchmark_appearance import BaselineAppearanceSequence, sha256
from backend.src.appearance_validation import validate_dense


def fixture(tmp_path):
    baseline = tmp_path/'baseline'
    package = baseline/'v2'
    geometry = package/'evaluation/family'
    moving = package/'evaluation/moving'
    for directory in [geometry, moving, package/'public']:
        directory.mkdir(parents=True)
    v = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]], dtype=np.float32)
    np.savez(geometry/'geometry.npz', vertices=v, triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
             uv=np.array([[.1, .1], [.4, .1], [.4, .9], [.1, .9]], dtype=np.float32), base_radius=np.ones(4))
    np.savez(moving/'states.npz', vertices=np.stack([v, v+[.1, 0, 0]]).astype(np.float32),
             R_wc=np.repeat(np.eye(3)[None], 2, axis=0).astype(np.float32),
             C_world=np.array([[0, 0, 0], [.05, 0, 0]], dtype=np.float32), time_seconds=np.array([6., 7.]))
    cv2.imwrite(str(geometry/'texture.png'), np.full((8, 8, 3), 128, dtype=np.uint8))
    (package/'public/intrinsics.json').write_text(json.dumps(dict(width=8, height=8, fx=12, fy=12, cx=4, cy=4)))
    (package/'DATASET.json').write_text(json.dumps(dict(schema='meshmeld_tubular_benchmark_v1',
        family='f01', condition='fem', split='development')))
    config = {'schema': 'tubular_appearance_spec_v1', 'families': [dict(family='f01', package='v2', split='development')],
        'resolution_divisor': 1, 'median_radius_meter': .015, 'frame_count': 2,
        'samples_per_pixel': 2, 'threads': 1, 'seed': 4, 'wall_cap_seconds': 120, 'storage_cap_bytes': 16*2**20}
    spec = tmp_path/'spec.json'; spec.write_text(json.dumps(config))
    protocol = tmp_path/'protocol.md'; protocol.write_text('Small analytic test fixture, not production evidence.\n')
    output = tmp_path/'output'
    freeze(output, baseline, spec, protocol)
    return baseline, output, config


def test_suite_roundtrip_material_identity_and_portable_public_boundary(tmp_path, monkeypatch):
    baseline, output, _ = fixture(tmp_path)
    inputs_before = {str(p): sha256(p) for p in baseline.rglob('*') if p.is_file()}
    run(output)
    report = json.loads((output/'GENERATION_COMPLETE.json').read_text())
    assert report['shared_truth_frames'] == 2 and report['rgb_images'] == 10
    family = output/'families/f01'
    evaluation = AppearanceEvaluation(family)
    dense = evaluation.dense(0)
    good = dense['valid']
    points = evaluation.material_points(1, dense['triangle_id'][good], dense['material_barycentric'][good])
    # Exact tracer barycentrics have tiny sum rounding; use the declared stored
    # states to compute the independent material displacement, not another ray hit.
    b = dense['material_barycentric'][good]
    np.testing.assert_allclose(points-dense['position_world'][good],
        np.c_[.0015*b.sum(axis=1), np.zeros((good.sum(), 2))], atol=2e-9)
    p = evaluation.project_material(1, dense['triangle_id'][good], b)
    assert p['forward'].all() and 'Not evaluated' in p['visibility']
    assert len(list((family/'evaluation/dense').glob('*.npz'))) == 2
    assert len(list((family/'evaluation/radiance').glob('*/*.npz'))) == 6
    assert not any((family/'staging').iterdir())
    for condition in evaluation.conditions:
        assert evaluation.rgb(0, condition).shape == (8, 8, 3)
    assert np.array_equal(evaluation.radiance(0, 'diffuse_symmetric'), evaluation.radiance(0, 'diffuse_exposure_half'))
    assert {str(p): sha256(p) for p in baseline.rglob('*') if p.is_file()} == inputs_before
    # Public reader works without evaluator data and never opens an NPZ.
    moved = tmp_path/'relocated'
    shutil.move(str(output), moved)
    hidden = moved/'families/f01/evaluation'
    hidden.rename(moved/'privileged_hidden')
    monkeypatch.setattr(np, 'load', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Public reader touched truth')))
    public = AppearancePublic(moved/'families/f01')
    assert public.rgb(1).shape == (8, 8, 3)
    with pytest.raises(ValueError):
        public.rgb(0, '../../evaluation')
    with pytest.raises(IndexError):
        public.rgb(-1)


def test_committed_tamper_and_incomplete_stage_are_refused(tmp_path):
    baseline, output, config = fixture(tmp_path)
    run(output)
    family = output/'families/f01'
    seq = BaselineAppearanceSequence(baseline/'v2')
    assert render_frame(family, seq, 0, config) is False
    victim = family/'public/diffuse_symmetric/rgb/000000.png'
    victim.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Committed frame changed'):
        render_frame(family, seq, 0, config)
    (family/'commits/000001.json').unlink()
    stage = family/'staging/000001'; stage.mkdir()
    with pytest.raises(RuntimeError, match='Incomplete uncommitted'):
        render_frame(family, seq, 1, config)


def test_partial_publish_resumes_without_regeneration(tmp_path):
    family = tmp_path/'family'; stage = family/'staging/000000'
    stage.mkdir(parents=True)
    a, b = stage/'a.txt', stage/'b.txt'
    a.write_text('one'); b.write_text('two')
    manifest = {'frame': 0, 'files': {'a.txt': sha256(a), 'b.txt': sha256(b)}}
    (stage/'READY.json').write_text(json.dumps(manifest))
    a.rename(family/'a.txt')
    publish_frame(family, 0, stage)
    assert (family/'a.txt').read_text() == 'one' and (family/'b.txt').read_text() == 'two'
    assert (family/'commits/000000.json').is_file() and not stage.exists()


def test_independent_ray_check_detects_wrong_stored_pose(tmp_path):
    _, output, _ = fixture(tmp_path)
    run(output)
    e = AppearanceEvaluation(output/'families/f01')
    s, dense = e.state(0), e.dense(0)
    with pytest.raises(ValueError, match='Geometry/reprojection'):
        validate_dense(s['vertices_meter'], e.geometry['triangles'], e.camera,
            s['R_wc'], s['C_world_meter']+[.001, 0, 0], dense)
