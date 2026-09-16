"""Separate saved-data audit, frozen-source replay and reader relocation checks.

Full mode requires terminal complete generation. Prefix mode checks only fixed
committed frames and can never issue a release-approval result.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

CONDITIONS = {
    'diffuse_symmetric': ('diffuse_symmetric', 1.),
    'diffuse_asymmetric': ('diffuse_asymmetric', 1.),
    'coated_symmetric': ('coated_symmetric', 1.),
    'diffuse_exposure_half': ('diffuse_symmetric', .5),
    'diffuse_exposure_double': ('diffuse_symmetric', 2.),
}
TRACED = ('diffuse_symmetric', 'diffuse_asymmetric', 'coated_symmetric')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def saved_rgb(path):
    a = cv2.imread(str(path))
    if a is None:
        raise ValueError('Unreadable PNG: '+str(path))
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def encode_radiance(radiance, gain):
    a = np.asarray(radiance)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('Invalid saved linear radiance')
    x = np.clip(a*gain, 0., 1.)
    return np.rint(255*np.where(x <= .0031308, 12.92*x, 1.055*np.power(x, 1/2.4)-.055)).astype(np.uint8)


def plane_ray_audit(vertices, faces, R, C, intrinsics, positions, valid):
    """Plane intersection plus cross-product barycentrics, separate from producer MT."""
    triangles = vertices[faces]
    a, ab, ac = triangles[:, 0], triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    normal = np.cross(ab, ac)
    nn = np.einsum('ij,ij->i', normal, normal)
    if np.any(nn <= 0):
        raise ValueError('Degenerate saved surface')
    numerator = np.einsum('ij,ij->i', a-C, normal)
    height, width = valid.shape
    chosen = np.unique(np.linspace(0, width*height-1, 121, dtype=int))
    maximum, mismatch = 0., 0
    for index in chosen:
        row, col = divmod(int(index), width)
        dc = np.array([(col+.5-intrinsics['cx'])/intrinsics['fx'],
                       (intrinsics['cy']-row-.5)/intrinsics['fy'], 1.])
        direction = np.linalg.solve(R.T, dc)
        den = normal@direction
        safe = np.abs(den) > 1e-18
        t = np.full(len(faces), -1.)
        t[safe] = numerator[safe]/den[safe]
        delta = C+t[:, None]*direction-a
        b = np.einsum('ij,ij->i', np.cross(delta, ac), normal)/nn
        c = np.einsum('ij,ij->i', np.cross(ab, delta), normal)/nn
        hits = safe & (t > 0) & (b >= 0) & (c >= 0) & (b+c <= 1)
        mismatch += int(bool(hits.any()) != bool(valid[row, col]))
        if hits.any() and valid[row, col]:
            error = float(np.linalg.norm(C+t[hits].min()*direction-positions[row, col]))
            maximum = max(maximum, error)
    if mismatch or maximum > 1e-6:
        raise ValueError(f'Saved geometry fails independent plane-ray check: {mismatch}, {maximum}')
    return {'sampled_rays': len(chosen), 'hit_miss_disagreements': mismatch, 'max_position_error_meter': maximum}


def audit_geometry(vertices, faces, R, C, intrinsics, dense):
    shape = (intrinsics['height'], intrinsics['width'])
    valid, ids, bary, depth = (dense[k] for k in ('valid', 'triangle_id', 'material_barycentric', 'depth_z'))
    if (valid.shape != shape or valid.dtype != np.bool_ or ids.shape != shape
            or not np.issubdtype(ids.dtype, np.integer) or bary.shape != shape+(3,)
            or depth.shape != shape or not np.array_equal(valid, ids >= 0)
            or np.any(ids[~valid] != -1) or np.any(ids[valid] >= len(faces)) or not valid.any()):
        raise ValueError('Invalid saved support/identity shapes')
    weights = bary[valid]
    if (not np.isfinite(weights).all() or np.any(weights < -2e-7) or np.any(weights > 1+2e-7)
            or np.any(np.abs(weights.sum(axis=1)-1) > 2e-7)
            or not np.isnan(bary[~valid]).all() or not np.isnan(depth[~valid]).all()
            or not np.isfinite(depth[valid]).all() or np.any(depth[valid] <= 0)):
        raise ValueError('Invalid saved barycentrics/depth/background')
    tri = vertices[faces[ids[valid]]]
    xyz = weights[:, 0, None]*tri[:, 0]+weights[:, 1, None]*tri[:, 1]+weights[:, 2, None]*tri[:, 2]
    pc = np.einsum('ij,nj->ni', R.T, xyz-C)
    yy, xx = np.nonzero(valid)
    u, v = intrinsics['fx']*pc[:, 0]/pc[:, 2]+intrinsics['cx'], intrinsics['cy']-intrinsics['fy']*pc[:, 1]/pc[:, 2]
    pixel_error = float(np.hypot(u-xx-.5, v-yy-.5).max())
    depth_error = float(np.abs(pc[:, 2]-depth[valid]).max())
    if pixel_error > .01 or depth_error > 1e-12:
        raise ValueError(f'Saved projection/depth mismatch: {pixel_error}, {depth_error}')
    positions = np.full(shape+(3,), np.nan); positions[valid] = xyz
    rays = plane_ray_audit(vertices, faces, R, C, intrinsics, positions, valid)
    topology = b'material_triangle_topology_v1\0'+np.asarray([len(vertices), len(faces)], dtype='<u8').tobytes()
    topology += np.asarray(faces, dtype='<u8', order='C').tobytes()
    if str(dense['mesh_topology_sha256']) != hashlib.sha256(topology).hexdigest():
        raise ValueError('Material topology digest differs')
    return positions, {'max_reprojection_pixels': pixel_error, 'max_depth_difference_meter': depth_error,
                       'valid_fraction': float(valid.mean()), 'plane_ray_check': rays}


def frozen_package(root, frozen):
    folder = root/'provenance/source/backend/src'
    for name, expected in frozen['sources'].items():
        if digest(folder/name) != expected:
            raise ValueError('Archived producer source changed: '+name)
    if (folder/'__init__.py').read_bytes() != b'':
        raise ValueError('Unexpected code in the frozen namespace initializer')
    name = '_appearance_frozen_'+digest(root/'FROZEN.json')[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, folder/'__init__.py', submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return name


def expected_frame_paths(frame):
    tag = f'{frame:06d}'
    return {f'evaluation/dense/{tag}.npz', f'quality/{tag}.json',
        *(f'evaluation/radiance/{c}/{tag}.npz' for c in TRACED),
        *(f'evaluation/render_metadata/{c}/{tag}.json' for c in TRACED),
        *(f'public/{c}/rgb/{tag}.png' for c in CONDITIONS)}


def audit_metadata(meta, condition, frame, spec, camera, provenance, albedo_hash):
    s = meta['settings']
    expected = {'material': 'roughplastic' if condition == 'coated_symmetric' else 'diffuse',
        'roughness': .18, 'interior_ior': 1.38, 'exterior_ior': 1.,
        'samples_per_pixel': spec['samples_per_pixel'], 'threads': spec['threads'],
        'seed': spec['seed']+frame, 'max_depth': 4, 'ray_batch_pixels': 8192,
        'exposure': 1., 'smooth_shading': False, 'uv_mode': 'periodic_u',
        'light_positions_camera': [[-.003, 0., 0.], [.003, 0., 0.]],
        'light_intensities_rgb': [[.00075]*3, [.00025]*3] if condition == 'diffuse_asymmetric' else [[.0005]*3, [.0005]*3]}
    if s != expected or meta['albedo_sha256'] != albedo_hash or meta['camera'] != provenance['camera']:
        raise ValueError('Rendering metadata does not describe fixed matched settings')
    if meta['scene_length_unit'] != 'meter' or meta['variant'] != 'llvm_ad_rgb':
        raise ValueError('Unexpected rendering units/variant')
    adapter = meta['baseline_adapter']
    if any(adapter[k] != value for k, value in provenance.items()) or adapter['source_frame'] != frame:
        raise ValueError('Per-frame source provenance changed')


def replay_and_relocate(root, family, row, frozen, package, frame):
    adapter = importlib.import_module(package+'.benchmark_appearance')
    render = importlib.import_module(package+'.endoscopic_renderer')
    readers = importlib.import_module(package+'.appearance_reader')
    baseline = (root/frozen['baseline_relative_path']).resolve()
    spec = json.loads((root/'SPECIFICATION.json').read_text())
    sequence = adapter.BaselineAppearanceSequence(baseline/row['package'],
        resolution_divisor=spec['resolution_divisor'], median_radius_meter=spec['median_radius_meter'])
    original = readers.AppearanceEvaluation(family)
    dense = original.dense(frame)
    replays = []
    for condition in TRACED:
        meta = json.loads((family/f'evaluation/render_metadata/{condition}/{frame:06d}.json').read_text())
        rgb, truth, _ = sequence.render(frame, render.AppearanceSettings(**meta['settings']))
        adapter.assert_shared_geometry(dense, truth)
        if not np.array_equal(rgb, original.rgb(frame, condition)) or not np.array_equal(
                truth['linear_radiance_rgb'], original.radiance(frame, condition)):
            raise ValueError('Frozen representative replay differs from saved data')
        replays.append({'condition': condition, 'rgb_exact': True, 'radiance_exact': True, 'geometry_exact': True})
    # Same-filesystem hardlinks make a temporary reader fixture without another
    # payload copy. Only the fixture's directory is moved; source files are read-only.
    with tempfile.TemporaryDirectory(prefix='appearance_relocation_', dir=root.parent) as temp:
        dest = Path(temp)/'before'
        files = ['public/frames.json', 'public/intrinsics.json',
                 *(f'public/{c}/rgb/{frame:06d}.png' for c in CONDITIONS)]
        for relative in files:
            p = dest/relative; p.parent.mkdir(parents=True, exist_ok=True); os.link(family/relative, p)
        public = readers.AppearancePublic(dest)
        if not np.array_equal(public.rgb(frame), original.rgb(frame)) or (dest/'evaluation').exists():
            raise ValueError('Public-only reader fixture failed')
        for relative in ['evaluation/states.npz', 'evaluation/geometry.npz', f'evaluation/dense/{frame:06d}.npz']:
            p = dest/relative; p.parent.mkdir(parents=True, exist_ok=True); os.link(family/relative, p)
        moved = dest.with_name('after'); dest.rename(moved)
        other = readers.AppearanceEvaluation(moved)
        np.testing.assert_array_equal(other.dense(frame)['position_world'], dense['position_world'])
        np.testing.assert_array_equal(other.rgb(frame), original.rgb(frame))
    return {'frame': frame, 'replays': replays, 'public_reader_without_truth_passed': True,
            'relocated_rgb_and_material_geometry_exact': True, 'temporary_payload_copies': 0}


def validate(root, output, *, full=True, prefix_frames=3):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite a validation result')
    if prefix_frames < 1 or (not full and output.name == 'RELEASE_VALIDATION.json'):
        raise ValueError('Prefix checks need positive sample count and a distinct non-release filename')
    start = time.perf_counter()
    frozen = json.loads((root/'FROZEN.json').read_text())
    spec = json.loads((root/'SPECIFICATION.json').read_text())
    if digest(root/'SPECIFICATION.json') != frozen['specification_sha256'] or digest(root/'provenance/PRODUCTION_PROTOCOL.md') != frozen['protocol_sha256']:
        raise ValueError('Production freeze does not match specification/protocol')
    status = json.loads((root/'STATUS.json').read_text())
    mutex = None
    if full:
        if status['state'] != 'generation_complete' or not (root/'GENERATION_COMPLETE.json').is_file():
            raise ValueError('Full independent validation requires terminal complete generation')
        finished = json.loads((root/'GENERATION_COMPLETE.json').read_text())
        expected_count = len(spec['families'])*spec['frame_count']
        if (finished['shared_truth_frames'] != expected_count or finished['rgb_images'] != 5*expected_count
                or finished['families'] != len(spec['families']) or finished['frozen_sha256'] != digest(root/'FROZEN.json')):
            raise ValueError('Generation summary differs from actual frozen design')
        mutex = (root/'execution.lock').open('rb')
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
    package = frozen_package(root, frozen)
    baseline = (root/frozen['baseline_relative_path']).resolve()
    verified, rows, replay = {}, [], []

    def check(path, expected=None):
        path = Path(path)
        relative = str(path.relative_to(root))
        if '..' in Path(relative).parts:
            raise ValueError('Manifest path escapes dataset')
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Saved artifact hash mismatch: '+relative)
        verified[relative] = value
        return value

    try:
        for name in ['FROZEN.json', 'SPECIFICATION.json', 'provenance/PRODUCTION_PROTOCOL.md']:
            check(root/name)
        for row in spec['families']:
            if not full and row != spec['families'][0]:
                continue
            family = root/'families'/row['family']
            prepared = json.loads((family/'PREPARED.json').read_text())
            check(family/'PREPARED.json')
            for name, value in prepared['files'].items():
                check(family/name, value)
            provenance = frozen['inputs'][row['family']]
            for item in provenance['source_files'].values():
                if digest(baseline/item['path']) != item['sha256']:
                    raise ValueError('Consumed baseline input changed')
            source_geometry = load_npz(baseline/provenance['source_files']['geometry']['path'])
            source_states = load_npz(baseline/provenance['source_files']['states']['path'])
            states = load_npz(family/'evaluation/states.npz')
            geometry = load_npz(family/'evaluation/geometry.npz')
            camera = json.loads((family/'public/intrinsics.json').read_text())
            if camera != {k: provenance['camera'][k] for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy')}:
                raise ValueError('Public calibration differs from frozen converted calibration')
            manifest = json.loads((family/'public/frames.json').read_text())
            expected_conditions = {name: {'radiance_condition': pair[0], 'exposure': pair[1]} for name, pair in CONDITIONS.items()}
            if manifest['conditions'] != expected_conditions or len(manifest['frames']) != spec['frame_count']:
                raise ValueError('Public frame/condition manifest changed')
            scale = spec['median_radius_meter']/float(np.median(source_geometry['base_radius']))
            if scale != provenance['source_to_meter_scale']:
                raise ValueError('Assigned scene scale changed')
            np.testing.assert_array_equal(states['vertices_meter'], (source_states['vertices'].astype(float)*scale).astype(np.float32))
            np.testing.assert_array_equal(states['C_world_meter'], source_states['C_world'].astype(float)*scale)
            np.testing.assert_array_equal(states['time_seconds'], source_states['time_seconds'])
            np.testing.assert_array_equal(geometry['triangles'], source_geometry['triangles'])
            np.testing.assert_array_equal(geometry['uv'], source_geometry['uv'])
            np.testing.assert_array_equal(geometry['reference_vertices_meter'], (source_geometry['vertices'].astype(float)*scale).astype(np.float32))
            for i, R in enumerate(states['R_wc']):
                u, _, vt = np.linalg.svd(source_states['R_wc'][i].astype(float))
                np.testing.assert_array_equal(R, u@vt)
                if not np.allclose(R.T@R, np.eye(3), atol=1e-10, rtol=0) or np.linalg.det(R) <= 0:
                    raise ValueError('Improper converted pose')
                if manifest['frames'][i] != {'frame': i, 'image': f'{i:06d}.png', 'time_seconds': float(states['time_seconds'][i])}:
                    raise ValueError('Frame/timestamp identity changed')
            texture = saved_rgb(baseline/provenance['source_files']['texture']['path']).astype(float)/255
            albedo = np.where(texture <= .04045, texture/12.92, ((texture+.055)/1.055)**2.4).astype('<f4')
            albedo_hash = hashlib.sha256(albedo.tobytes()).hexdigest()
            commits = sorted((family/'commits').glob('*.json'))
            expected_inventory = {'PREPARED.json', *prepared['files']}
            expected_names = [f'{i:06d}.json' for i in range(spec['frame_count'])]
            if full and [p.name for p in commits] != expected_names:
                raise ValueError('Missing or extra committed production frames')
            if not full:
                commits = commits[:prefix_frames]
                if [p.name for p in commits] != expected_names[:prefix_frames] or len(commits) != min(prefix_frames, spec['frame_count']):
                    raise ValueError('Requested development prefix is not completely committed')
            reader = importlib.import_module(package+'.appearance_reader').AppearanceEvaluation(family)
            for commit in commits:
                frame = int(commit.stem)
                record = json.loads(commit.read_text())
                if record['frame'] != frame or set(record['files']) != expected_frame_paths(frame):
                    raise ValueError('Frame commit does not contain the exact declared modalities')
                check(commit)
                expected_inventory.update({'commits/'+commit.name, *record['files']})
                for name, value in record['files'].items():
                    check(family/name, value)
                dense = load_npz(family/f'evaluation/dense/{frame:06d}.npz')
                positions, qa = audit_geometry(states['vertices_meter'][frame].astype(float), geometry['triangles'],
                    states['R_wc'][frame], states['C_world_meter'][frame], camera, dense)
                good_ids = np.flatnonzero(dense['valid'])
                chosen = good_ids[np.unique(np.linspace(0, len(good_ids)-1, min(121, len(good_ids)), dtype=int))]
                material_ids = dense['triangle_id'].reshape(-1)[chosen]
                weights = dense['material_barycentric'].reshape(-1, 3)[chosen]
                target = min(frame+1, spec['frame_count']-1)
                tri = states['vertices_meter'][target].astype(float)[geometry['triangles'][material_ids]]
                expected_point = weights[:, 0, None]*tri[:, 0]+weights[:, 1, None]*tri[:, 1]+weights[:, 2, None]*tri[:, 2]
                actual_point = reader.material_points(target, material_ids, weights)
                np.testing.assert_allclose(actual_point, expected_point, rtol=0, atol=1e-12)
                pc = np.einsum('ij,nj->ni', states['R_wc'][target].T, expected_point-states['C_world_meter'][target])
                with np.errstate(divide='ignore', invalid='ignore'):
                    uv = np.c_[camera['fx']*pc[:, 0]/pc[:, 2]+camera['cx'], camera['cy']-camera['fy']*pc[:, 1]/pc[:, 2]]
                forward = (pc[:, 2] > 0) & np.isfinite(uv).all(axis=1)
                projected = reader.project_material(target, material_ids, weights)
                np.testing.assert_array_equal(projected['forward'], forward)
                np.testing.assert_allclose(projected['uv'][forward], uv[forward], rtol=1e-10, atol=1e-6)
                for condition in TRACED:
                    meta = json.loads((family/f'evaluation/render_metadata/{condition}/{frame:06d}.json').read_text())
                    audit_metadata(meta, condition, frame, spec, camera, provenance, albedo_hash)
                    conversion = meta['baseline_adapter']
                    expected_vertex_error = float(np.linalg.norm(states['vertices_meter'][frame].astype(float)-
                        source_states['vertices'][frame].astype(float)*scale, axis=1).max())
                    expected_rotation_error = float(np.abs(states['R_wc'][frame]-source_states['R_wc'][frame]).max())
                    if (conversion['time_seconds'] != float(states['time_seconds'][frame])
                            or conversion['vertex_quantization_max_meter'] != expected_vertex_error
                            or conversion['rotation_max_change'] != expected_rotation_error):
                        raise ValueError('Per-frame conversion/time metadata differs from actual source conversion')
                    if meta['mitsuba'] != frozen['environment']['mitsuba'] or meta['drjit'] != frozen['environment']['drjit']:
                        raise ValueError('Per-frame rendering environment differs from freeze')
                    radiance = load_npz(family/f'evaluation/radiance/{condition}/{frame:06d}.npz')['linear_radiance_rgb']
                    if radiance.shape != positions.shape or np.any(radiance[~dense['valid']] != 0):
                        raise ValueError('Radiance support/shape differs from geometry')
                    for name, (source, gain) in CONDITIONS.items():
                        if source == condition and not np.array_equal(encode_radiance(radiance, gain), saved_rgb(family/f'public/{name}/rgb/{frame:06d}.png')):
                            raise ValueError('Saved image differs from declared radiance/exposure')
                rows.append({'family': row['family'], 'frame': frame, **qa, 'radiance_and_rgb_exact': True,
                             'cross_time_material_queries': len(chosen), 'target_frame': target})
                if full and frame % 15 == 0:
                    print(json.dumps({'verified_family': row['family'], 'verified_frame': frame}), flush=True)
            if full:
                if any(p.is_file() for p in (family/'staging').rglob('*')):
                    raise ValueError('Uncommitted staging payload remains after generation')
                if {str(p.relative_to(family)) for p in family.rglob('*') if p.is_file()} != expected_inventory:
                    raise ValueError('Unexpected or missing family payload outside committed manifests')
                for name in CONDITIONS:
                    if sorted(p.name for p in (family/'public'/name/'rgb').glob('*')) != [f'{i:06d}.png' for i in range(spec['frame_count'])]:
                        raise ValueError('Unexpected public frame file')
                replay.append({'family': row['family'], **replay_and_relocate(root, family, row, frozen, package, min(67, spec['frame_count']-1))})
        total = len(spec['families'])*spec['frame_count']
        if full and (len(rows) != total or len(replay) != len(spec['families'])):
            raise ValueError('Full validation did not cover the frozen population')
        result = {'schema': 'appearance_release_validation_v1', 'mode': 'full' if full else 'committed_prefix_only',
            'all_data_checks_passed': full, 'checked_subset_passed': True,
            'verified_frames': len(rows), 'expected_frames': total, 'rows': rows, 'frozen_replay_and_portability': replay,
            'verified_file_sha256': verified, 'generation_freeze_sha256': digest(root/'FROZEN.json'),
            'validator_source_sha256': digest(__file__), 'seconds': time.perf_counter()-start,
            'scope': 'Independent saved-data formulas/rays plus frozen replay and relocation; no clinical or reconstruction-performance claim.'}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
        return result
    finally:
        if mutex is not None:
            mutex.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('full', 'prefix'), default='full')
    parser.add_argument('--prefix-frames', type=int, default=3)
    args = parser.parse_args()
    validate(args.root, args.output, full=args.mode == 'full', prefix_frames=args.prefix_frames)
