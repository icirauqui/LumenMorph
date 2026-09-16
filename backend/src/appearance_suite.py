"""Frozen, bounded appearance add-on producer with shared truth and frame commits.

CLI: ``python -m backend.src.appearance_suite freeze|run --help``.
Generation and producer QA are distinct from independent release validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import time

import cv2
import numpy as np

from .benchmark_appearance import BaselineAppearanceSequence, assert_shared_geometry, exposed_rgb, sha256
from .endoscopic_renderer import AppearanceSettings
from .appearance_reader import AppearanceEvaluation
from .appearance_validation import validate_dense


CONDITIONS = {
    'diffuse_symmetric': {'radiance_condition': 'diffuse_symmetric', 'exposure': 1.},
    'diffuse_asymmetric': {'radiance_condition': 'diffuse_asymmetric', 'exposure': 1.},
    'coated_symmetric': {'radiance_condition': 'coated_symmetric', 'exposure': 1.},
    'diffuse_exposure_half': {'radiance_condition': 'diffuse_symmetric', 'exposure': .5},
    'diffuse_exposure_double': {'radiance_condition': 'diffuse_symmetric', 'exposure': 2.},
}


def write_json(path, value, *, replace_existing=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace_existing:
        raise FileExistsError(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
    tmp.replace(path)


def source_paths():
    directory = Path(__file__).resolve().parent
    return [directory/name for name in ('appearance_suite.py', 'appearance_reader.py',
        'appearance_validation.py', 'benchmark_appearance.py', 'camera_models.py',
        'endoscopic_renderer.py', 'continuous_material.py')]


def verify_sources(root):
    lock = json.loads((root/'FROZEN.json').read_text())
    for p in source_paths():
        if sha256(p) != lock['sources'][p.name]:
            raise ValueError(f'Source differs from freeze: {p.name}')
    if sha256(root/'SPECIFICATION.json') != lock['specification_sha256']:
        raise ValueError('Specification differs from freeze')
    return lock


def freeze(root, baseline, spec_path, protocol):
    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    if root == baseline or baseline in root.parents:
        raise ValueError('Appearance output must be outside the sealed baseline')
    spec = json.loads(Path(spec_path).read_text())
    if spec['schema'] != 'tubular_appearance_spec_v1' or not spec['families']:
        raise ValueError('Unsupported/empty appearance specification')
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/'provenance/source/backend/src'
    snapshot.mkdir(parents=True)
    for p in source_paths():
        shutil.copy2(p, snapshot/p.name)
    for p in [snapshot/'__init__.py', snapshot.parent/'__init__.py']:
        p.write_text('')
    shutil.copy2(spec_path, root/'SPECIFICATION.json')
    shutil.copy2(protocol, root/'provenance/PRODUCTION_PROTOCOL.md')
    import mitsuba, drjit
    env = {'python': platform.python_version(), 'platform': platform.platform(),
           'numpy': np.__version__, 'opencv': cv2.__version__,
           'mitsuba': mitsuba.__version__, 'drjit': drjit.__version__}
    write_json(root/'provenance/ENVIRONMENT.json', env)
    inputs = {}
    for row in spec['families']:
        sequence = BaselineAppearanceSequence(baseline/row['package'],
            resolution_divisor=spec['resolution_divisor'], median_radius_meter=spec['median_radius_meter'])
        if (sequence.descriptor['family'] != row['family'] or len(sequence) != spec['frame_count']
                or sequence.descriptor['split'] != row['split']):
            raise ValueError('Baseline family/split/frame contract differs')
        inputs[row['family']] = sequence.provenance
    write_json(root/'FROZEN.json', {'schema': 'tubular_appearance_freeze_v1',
        'sources': {p.name: sha256(p) for p in source_paths()},
        'specification_sha256': sha256(root/'SPECIFICATION.json'),
        'protocol_sha256': sha256(root/'provenance/PRODUCTION_PROTOCOL.md'),
        'baseline_relative_path': os.path.relpath(baseline, root), 'inputs': inputs,
        'environment': env, 'conditions': CONDITIONS,
        'seal_scope': 'Source/input contract; not generation or scientific completion'})
    write_json(root/'STATUS.json', {'state': 'frozen', 'completed_frames': 0, 'elapsed_seconds': 0.})


def prepare_family(path, sequence, frame_count):
    if (path/'PREPARED.json').is_file():
        saved = json.loads((path/'PREPARED.json').read_text())
        for name, digest in saved['files'].items():
            if sha256(path/name) != digest:
                raise ValueError('Prepared family changed')
        return
    path.mkdir(parents=True, exist_ok=False)
    public, private = path/'public', path/'evaluation'
    public.mkdir(); private.mkdir()
    rows = [sequence.state(i) for i in range(frame_count)]
    np.savez_compressed(private/'states.npz', **{
        k: np.stack([s[k] for s in rows]) for k in ('vertices_meter', 'R_wc', 'C_world_meter', 'time_seconds')})
    np.savez_compressed(private/'geometry.npz', triangles=sequence.geometry['triangles'],
        uv=sequence.geometry['uv'], reference_vertices_meter=sequence.reference_vertices_meter)
    write_json(private/'BASELINE_PROVENANCE.json', sequence.provenance)
    camera = sequence.camera.to_dict()
    write_json(public/'intrinsics.json', {k: camera[k] for k in ('width', 'height', 'fx', 'fy', 'cx', 'cy')})
    write_json(public/'frames.json', {'schema': 'tubular_appearance_frames_v1',
        'conditions': CONDITIONS, 'rgb_channels': 'RGB', 'pixel_centers': 'column+.5,row+.5',
        'frames': [{'frame': i, 'image': f'{i:06d}.png', 'time_seconds': rows[i]['time_seconds']}
                   for i in range(frame_count)]})
    write_json(path/'PREPARED.json', {'files': {str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob('*')) if p.is_file()}})


def publish_frame(family, frame, stage):
    manifest = json.loads((stage/'READY.json').read_text())
    for name, digest in manifest['files'].items():
        source, dest = stage/name, family/name
        if dest.exists():
            if sha256(dest) != digest:
                raise ValueError(f'Existing output differs during frame publish: {name}')
        else:
            if sha256(source) != digest:
                raise ValueError(f'Staged file changed: {name}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.replace(dest)
    commit = family/'commits'/f'{frame:06d}.json'
    write_json(commit, manifest)
    # Only remove empty stage directories and the now-duplicated transaction
    # manifest; all scientific output has moved to its final declared location.
    (stage/'READY.json').unlink()
    for directory in sorted((p for p in stage.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()
    stage.rmdir()


def render_frame(family, sequence, frame, spec):
    commit = family/'commits'/f'{frame:06d}.json'
    if commit.is_file():
        manifest = json.loads(commit.read_text())
        for name, digest in manifest['files'].items():
            if sha256(family/name) != digest:
                raise ValueError(f'Committed frame changed: {frame}/{name}')
        return False
    stage = family/'staging'/f'{frame:06d}'
    if (stage/'READY.json').is_file():
        publish_frame(family, frame, stage)
        return True
    if stage.exists():
        raise RuntimeError(f'Incomplete uncommitted rendering retained at {stage}; inspect before any rerun')
    stage.mkdir(parents=True)
    base = AppearanceSettings(material='diffuse', samples_per_pixel=spec['samples_per_pixel'],
        threads=spec['threads'], seed=spec['seed']+frame, max_depth=4)
    settings = {'diffuse_symmetric': base,
        'diffuse_asymmetric': replace(base, light_intensities_rgb=((.00075,)*3, (.00025,)*3)),
        'coated_symmetric': replace(base, material='roughplastic')}
    first, reports, times = None, {}, {}
    state = sequence.state(frame)
    for name, setting in settings.items():
        begin = time.perf_counter()
        rgb, truth, meta = sequence.render(frame, setting)
        times[name] = time.perf_counter()-begin
        if first is None:
            first = truth
            qa = validate_dense(state['vertices_meter'], sequence.geometry['triangles'], sequence.camera,
                state['R_wc'], state['C_world_meter'], truth)
            dense_path = stage/'evaluation/dense'/f'{frame:06d}.npz'
            dense_path.parent.mkdir(parents=True)
            np.savez_compressed(dense_path, **{k: truth[k] for k in
                ('depth_z', 'triangle_id', 'material_barycentric', 'valid', 'mesh_topology_sha256')})
            with np.load(dense_path, allow_pickle=False) as a:
                reloaded = AppearanceEvaluation(family).expand_dense(dict(a), frame)
            assert_shared_geometry(truth, reloaded)
        else:
            assert_shared_geometry(first, truth)
        radiance_path = stage/'evaluation/radiance'/name/f'{frame:06d}.npz'
        radiance_path.parent.mkdir(parents=True)
        np.savez_compressed(radiance_path, linear_radiance_rgb=truth['linear_radiance_rgb'])
        with np.load(radiance_path, allow_pickle=False) as a:
            if not np.array_equal(a['linear_radiance_rgb'], truth['linear_radiance_rgb']):
                raise ValueError('Radiance changed on serialization')
        if not np.array_equal(rgb, exposed_rgb(truth['linear_radiance_rgb'], setting.exposure)):
            raise ValueError('Renderer and independent exposure adapter disagree')
        for condition, parameters in CONDITIONS.items():
            if parameters['radiance_condition'] != name:
                continue
            output = exposed_rgb(truth['linear_radiance_rgb'], parameters['exposure'])
            p = stage/'public'/condition/'rgb'/f'{frame:06d}.png'
            p.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(p), cv2.cvtColor(output, cv2.COLOR_RGB2BGR)):
                raise OSError(f'Unable to save {p}')
            if not np.array_equal(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), output):
                raise ValueError('PNG save/readback changed RGB')
            good = truth['valid']
            luma = output[good].astype(float)@np.array([.2126, .7152, .0722])/255
            reports[condition] = {'saturated_wall_fraction': float(np.any(output[good] == 255, axis=1).mean()),
                'black_wall_fraction': float(np.all(output[good] == 0, axis=1).mean()),
                'wall_luma_quantiles': np.quantile(luma, [.05, .5, .95]).tolist()}
        write_json(stage/'evaluation/render_metadata'/name/f'{frame:06d}.json', meta)
    write_json(stage/'quality'/f'{frame:06d}.json', {'frame': frame, 'geometry': qa,
        'appearance': reports, 'render_seconds': times, 'shared_label_readback_exact': True,
        'all_radiance_and_png_readbacks_exact': True, 'seed': base.seed})
    write_json(stage/'READY.json', {'frame': frame, 'files': {str(p.relative_to(stage)): sha256(p)
        for p in sorted(stage.rglob('*')) if p.is_file()}})
    publish_frame(family, frame, stage)
    return True


def run(root):
    root = Path(root).resolve()
    lock = verify_sources(root)
    spec = json.loads((root/'SPECIFICATION.json').read_text())
    baseline = (root/lock['baseline_relative_path']).resolve()
    with (root/'execution.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((root/'STATUS.json').read_text())
        if previous['state'] == 'generation_complete':
            raise ValueError('Generation already complete; run independent validation')
        elapsed_before = previous['elapsed_seconds']
        start = time.perf_counter()
        status = {**previous, 'state': 'running'}
        try:
            for row in spec['families']:
                sequence = BaselineAppearanceSequence(baseline/row['package'],
                    resolution_divisor=spec['resolution_divisor'], median_radius_meter=spec['median_radius_meter'])
                if sequence.provenance != lock['inputs'][row['family']]:
                    raise ValueError('Baseline inputs changed since freeze')
                family = root/'families'/row['family']
                prepare_family(family, sequence, spec['frame_count'])
                for frame in range(spec['frame_count']):
                    verify_sources(root)
                    status.update(family=row['family'], frame=frame,
                        elapsed_seconds=elapsed_before+time.perf_counter()-start)
                    if status['elapsed_seconds'] >= spec['wall_cap_seconds']:
                        raise RuntimeError('Registered generation wall-time cap reached')
                    write_json(root/'STATUS.json', status, replace_existing=True)
                    rendered = render_frame(family, sequence, frame, spec)
                    status['completed_frames'] = len(list((root/'families').glob('*/commits/*.json')))
                    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                    status.update(bytes=size, elapsed_seconds=elapsed_before+time.perf_counter()-start)
                    if size > spec['storage_cap_bytes']:
                        raise RuntimeError('Registered payload cap reached')
                    if status['elapsed_seconds'] > spec['wall_cap_seconds']:
                        raise RuntimeError('Registered generation wall-time cap reached')
                    write_json(root/'STATUS.json', status, replace_existing=True)
                    print(json.dumps({'family': row['family'], 'frame': frame, 'new_frame': rendered,
                        'completed_frames': status['completed_frames'], 'seconds': status['elapsed_seconds'], 'bytes': size}), flush=True)
            status['state'] = 'generation_complete'
            write_json(root/'GENERATION_COMPLETE.json', {'scope': 'Generation and producer QA; independent release validation still required',
                'families': len(spec['families']), 'shared_truth_frames': status['completed_frames'],
                'rgb_images': status['completed_frames']*len(CONDITIONS),
                'frozen_sha256': sha256(root/'FROZEN.json')})
        except Exception as error:
            status.update(state='stopped', error=repr(error))
            raise
        finally:
            status['elapsed_seconds'] = elapsed_before+time.perf_counter()-start
            write_json(root/'STATUS.json', status, replace_existing=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze')
    for key in ('root', 'baseline', 'spec', 'protocol'):
        p.add_argument('--'+key, type=Path, required=True)
    sub.add_parser('run').add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    if a.action == 'freeze':
        freeze(a.root, a.baseline, a.spec, a.protocol)
    else:
        import signal
        def terminated(signum, frame):
            raise TimeoutError('Generation process received termination signal; partial stage retained')
        signal.signal(signal.SIGTERM, terminated)
        run(a.root)


if __name__ == '__main__':
    main()
