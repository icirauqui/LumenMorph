"""Tiny analytic end-to-end evaluator fixture, not benchmark evidence."""
import hashlib
import json
import struct

import numpy as np

from backend.src.appearance_reconstruction_metrics import evaluate_study, F


def test_saved_models_to_paired_metrics_preserves_low_coverage_and_repeat_control(tmp_path):
    data, study = tmp_path/'data', tmp_path/'study'
    data.mkdir(); study.mkdir()
    hashes = {}

    def save(relative, **arrays):
        path = data/relative; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

    vertices = np.array([[-.02, -.02, .03], [.02, -.02, .03], [0, .02, .03]])
    centers = np.tile(np.array([[0., 0, 0], [.01, 0, 0], [0, .01, 0]]), (30, 1))
    conditions = ['diffuse_symmetric', 'diffuse_asymmetric', 'coated_symmetric',
                  'diffuse_exposure_half', 'diffuse_exposure_double']
    jobs = []
    for family in ['f01', 'f04', 'f05', 'f06']:
        save(f'families/{family}/evaluation/states.npz',
            vertices_meter=np.repeat(vertices[None], 90, axis=0),
            C_world_meter=centers, R_wc=np.repeat(np.eye(3)[None], 90, axis=0))
        save(f'families/{family}/evaluation/geometry.npz', triangles=np.array([[0, 1, 2]]))
        for frame in range(60, 75):
            save(f'families/{family}/evaluation/dense/{frame:06d}.npz', valid=np.ones((1, 1), dtype=bool),
                triangle_id=np.zeros((1, 1), dtype=int), material_barycentric=np.array([[[.25, .25, .5]]]))
        selected = conditions+(['diffuse_symmetric_repeat'] if family == 'f01' else [])
        for condition in selected:
            job_id = family+'__'+condition; jobs.append(job_id)
            job = study/'jobs'/job_id
            report = {'execution_finished': True, 'inputs_unchanged': True, 'models': {}}
            for label, frames in [('global', [0, 1, 2]), ('local', [60, 61, 62])]:
                model = job/'models'/label/'0'; model.mkdir(parents=True)
                blob = struct.pack('<Q', 3)
                for k, frame in enumerate(frames):
                    center = F@centers[frame]
                    blob += struct.pack('<I7dI', k+1, 1, 0, 0, 0, *(-center), 1)
                    blob += f'{frame:06d}.png'.encode()+b'\0'+struct.pack('<Q', 0)
                (model/'images.bin').write_bytes(blob)
                point = struct.pack('<Q', 1)+struct.pack('<Q3d3Bd', 1, 0, 0, .03, 128, 128, 128, 0)+struct.pack('<Q', 0)
                (model/'points3D.bin').write_bytes(point)
                report['models'][label] = {'mapper_completed': True, 'candidates': [{
                    'path': str(model.relative_to(job)), 'converted': True, 'parse_error': None,
                    'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model.iterdir()}}]}
            (job/'REPORT.json').write_text(json.dumps(report))
    (data/'RELEASE_VALIDATION.json').write_text(json.dumps({'all_data_checks_passed': True, 'verified_file_sha256': hashes}))
    (study/'EXECUTION_COMPLETE.json').write_text(json.dumps({'all_20_conditions_accounted_for': True,
        'repeat_control_completed': True, 'jobs': jobs}))
    result = evaluate_study(data, study, tmp_path/'evaluation')
    assert len(result['rows']) == 42 and len(result['paired_common_support']) == 34
    assert sum(r['repeat_control'] for r in result['rows']) == 2
    for row in result['rows']:
        assert row['registered_expected'] == 3
        assert row['registered_fraction'] == (3/90 if row['model'] == 'global' else 3/15)
        assert row['poses']['ate_mm']['max'] < 1e-9
        if row['model'] == 'local':
            assert row['local_geometry']['surface_accuracy_mm']['max'] < 1e-9
            assert row['local_geometry']['fixed_visible_support_nearest_point_mm']['max'] < 1e-9
    assert all(abs(p['ate_rmse_difference_mm']) < 1e-12 for p in result['paired_common_support'])
