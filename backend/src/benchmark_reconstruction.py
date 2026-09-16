"""Image-only COLMAP benchmark reconstruction with retained failure evidence.

Run as ``python -m backend.src.benchmark_reconstruction --help`` from the generator
root. Stationary captures intentionally have no SfM mode: they require RGB
associations to the independently reconstructed moving reference acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


GLOBAL_OPTIONS = {
    'Mapper.multiple_models': '0', 'Mapper.ba_refine_focal_length': '0',
    'Mapper.ba_refine_principal_point': '0', 'Mapper.ba_refine_extra_params': '0',
    'Mapper.ba_global_max_num_iterations': '75', 'Mapper.ba_global_max_refinements': '5',
    'Mapper.init_min_num_inliers': '100', 'Mapper.init_max_error': '2.0',
    'Mapper.init_max_forward_motion': '0.999', 'Mapper.init_min_tri_angle': '4.0',
    'Mapper.abs_pose_max_error': '4.0', 'Mapper.abs_pose_min_num_inliers': '30',
    'Mapper.filter_max_reproj_error': '2.0', 'Mapper.filter_min_tri_angle': '1.0',
    'Mapper.local_ba_min_tri_angle': '2.0', 'Mapper.tri_ignore_two_view_tracks': '1',
}
LOCAL_OPTIONS = {**GLOBAL_OPTIONS, 'Mapper.ba_global_max_num_iterations': '35',
                 'Mapper.ba_global_max_refinements': '2', 'Mapper.init_min_num_inliers': '75',
                 'Mapper.init_min_tri_angle': '3.0', 'Mapper.abs_pose_min_num_inliers': '40'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def options(values):
    return [part for key, value in values.items() for part in ('--' + key, str(value))]


def window_names(names, size=15, stride=10):
    if size < 2 or stride < 1 or len(names) < size:
        raise ValueError('Require frame count >= window size >= 2 and stride >= 1')
    starts = list(range(0, len(names) - size + 1, stride))
    if starts[-1] != len(names) - size:
        starts.append(len(names) - size)
    return [names[start:start + size] for start in starts]


def read_names(path):
    """Keep empty observation lines; image rows alternate with observation rows."""
    rows = [line for line in Path(path).read_text().splitlines()
            if not line.lstrip().startswith('#')]
    # Header comments may be followed by empty lines, but an observations row may
    # itself be empty. Only skip blanks when an image header is expected.
    names, index = [], 0
    while index < len(rows):
        if not rows[index].strip():
            index += 1
            continue
        fields = rows[index].split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError('Malformed COLMAP image header')
        int(fields[0])
        if index + 1 >= len(rows):
            raise ValueError('Missing COLMAP observations row')
        names.append(fields[9])
        index += 2
    if len(names) != len(set(names)):
        raise ValueError('Duplicate registered filename')
    return sorted(names)


def input_record(images, intrinsics):
    import math
    images, intrinsics = Path(images).resolve(), Path(intrinsics).resolve()
    files = sorted(images.glob('*.png'))
    if len(files) < 2 or any(not p.is_file() or any(c.isspace() for c in p.name) for p in files):
        raise ValueError('Need at least two PNG frames with whitespace-free filenames')
    data = json.loads(intrinsics.read_text())
    keys = ('width', 'height', 'fx', 'fy', 'cx', 'cy')
    values = {key: float(data[key]) for key in keys}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite intrinsics')
    if min(values[k] for k in ('width', 'height', 'fx', 'fy')) <= 0:
        raise ValueError('Image size and focal lengths must be positive')
    if not 0 <= values['cx'] <= values['width'] or not 0 <= values['cy'] <= values['height']:
        raise ValueError('Principal point outside image')
    return {'image_directory': str(images), 'rgb_sha256': {p.name: sha(p) for p in files},
            'intrinsics': {'path': str(intrinsics), 'sha256': sha(intrinsics)},
            'camera_model': 'PINHOLE',
            'camera_params': ','.join(format(values[k], '.17g') for k in ('fx', 'fy', 'cx', 'cy'))}


def provenance(executable, fork_source, output):
    executable, fork_source = Path(executable).resolve(), Path(fork_source).resolve()
    target = output / 'provenance'
    target.mkdir()
    shutil.copy2(executable, target / 'colmap')
    shutil.copy2(Path(__file__), target / 'benchmark_reconstruction.py')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(fork_source), *args])
    revision = git('rev-parse', 'HEAD').decode().strip()
    status = git('status', '--porcelain').decode()
    with (target / 'fork_head.tar').open('wb') as archive:
        subprocess.run(['git', '-C', str(fork_source), 'archive', 'HEAD'], stdout=archive, check=True)
    (target / 'fork_dirty.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    for relative in filter(None, untracked):
        path = fork_source / relative
        if path.is_file():
            dest = target / 'fork_untracked' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    build = executable.parents[2]
    if (build / 'CMakeCache.txt').is_file():
        shutil.copy2(build / 'CMakeCache.txt', target / 'CMakeCache.txt')
    version = subprocess.run([str(executable), '-h'], capture_output=True, text=True, check=True)
    (target / 'executable_help.txt').write_text(version.stdout + version.stderr)
    deps = subprocess.run(['ldd', str(executable)], capture_output=True, text=True)
    (target / 'ldd.txt').write_text(deps.stdout + deps.stderr)
    result = {'source_directory': str(fork_source), 'source_commit': revision, 'source_status': status,
              'executable_original': str(executable), 'executable_sha256': sha(executable),
              'executable_version': (version.stdout + version.stderr).splitlines()[:3],
              'source_binary_correspondence': 'Recorded separately; historical binary is not claimed built from current source.',
              'files_sha256': {str(p.relative_to(target)): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}}
    write_json(target / 'manifest.json', result)
    return result


def run_reconstruction(images, intrinsics, output, executable, fork_source, *,
                       mode='moving', threads=2, bundle_size=15, stride=10, timeout=3600):
    if mode not in ('reference', 'moving'):
        raise ValueError('Stationary action has no independent triangulation baseline; use reference associations')
    if threads < 1:
        raise ValueError('threads must be positive')
    inputs = input_record(images, intrinsics)
    names = sorted(inputs['rgb_sha256'])
    windows = window_names(names, bundle_size, stride) if mode == 'moving' else []
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    (output / 'window_lists').mkdir()
    images, executable = str(Path(images).resolve()), str(Path(executable).resolve())
    database = output / 'database.db'
    meta = provenance(executable, fork_source, output)
    config = {'mode': mode, 'threads': threads, 'bundle_size': bundle_size, 'stride': stride,
              'timeout_seconds_per_command': timeout, 'global_options': GLOBAL_OPTIONS,
              'local_options': LOCAL_OPTIONS, 'gpu': False, 'random_seed': 0,
              'classification': 'RGB-estimated calibrated global SfM and independently estimated local SfM',
              'truth_input': False, 'output': str(output)}
    write_json(output / 'inputs.json', inputs)
    write_json(output / 'configuration.json', config)
    env = {**os.environ, 'OMP_NUM_THREADS': str(threads), 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'QT_QPA_PLATFORM': 'offscreen'}
    report = {'format': 'tubular_benchmark_reconstruction_v1', 'configuration': config,
              'executable_sha256': meta['executable_sha256'], 'commands': [], 'global': None,
              'locals': [], 'complete': False, 'started_unix': time.time()}
    def command(label, argv):
        log = output / 'logs' / (label + '.log')
        start = time.time()
        with log.open('w') as stream:
            stream.write(json.dumps(argv) + '\n'); stream.flush()
            try:
                process = subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                         timeout=timeout or None)
                code, timed_out = process.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        row = {'label': label, 'argv': argv, 'returncode': code, 'timed_out': timed_out,
               'elapsed_seconds': time.time() - start, 'log_sha256': sha(log)}
        report['commands'].append(row)
        write_json(output / 'report.json', report)
        return code == 0
    def reconstruct(label, expected, mapper_options):
        dest = output / 'models' / label
        dest.mkdir(parents=True)
        image_list = output / 'window_lists' / (label + '.txt')
        image_list.write_text('\n'.join(expected) + '\n')
        good = command(label + '_mapper', [executable, 'mapper', '--random_seed', '0', '--database_path', str(database),
                       '--image_path', images, '--output_path', str(dest), '--image_list_path', str(image_list),
                       '--Mapper.num_threads', str(threads), *options(mapper_options)])
        candidates = []
        for candidate in sorted(p for p in dest.iterdir() if p.is_dir()):
            converted = command(label + '_' + candidate.name + '_convert', [executable, 'model_converter',
                                '--input_path', str(candidate), '--output_path', str(candidate), '--output_type', 'TXT'])
            registered, error, point_count = [], None, 0
            if converted:
                try:
                    registered = read_names(candidate / 'images.txt')
                    point_count = sum(bool(line.strip()) and not line.lstrip().startswith('#')
                                      for line in (candidate / 'points3D.txt').read_text().splitlines())
                except (OSError, ValueError) as exc:
                    error = str(exc)
            candidates.append({'path': str(candidate.relative_to(output)), 'registered_names': registered,
                               'missing_names': sorted(set(expected) - set(registered)),
                               'unexpected_names': sorted(set(registered) - set(expected)), 'parse_error': error, 'point_count': point_count,
                               'files_sha256': {p.name: sha(p) for p in sorted(candidate.iterdir()) if p.is_file()},
                               'complete': converted and error is None and point_count > 0 and registered == sorted(expected)})
        return {'name': label, 'expected_names': expected, 'mapper_success': good,
                'candidates': candidates, 'complete': good and len(candidates) == 1 and candidates[0]['complete']}
    try:
        extracted = command('01_extract', [executable, 'feature_extractor', '--random_seed', '0', '--database_path', str(database),
            '--image_path', images, '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE',
            '--ImageReader.camera_params', inputs['camera_params'], '--SiftExtraction.use_gpu', '0',
            '--SiftExtraction.num_threads', str(threads), '--SiftExtraction.max_image_size', '800',
            '--SiftExtraction.max_num_features', '12000', '--SiftExtraction.peak_threshold', '0.004',
            '--SiftExtraction.edge_threshold', '10', '--SiftExtraction.first_octave', '-1',
            '--SiftExtraction.num_octaves', '4', '--SiftExtraction.octave_resolution', '3',
            '--SiftExtraction.max_num_orientations', '2'])
        if not extracted:
            report['failure'] = 'feature_extraction_failed'
            return report
        matched = command('02_match', [executable, 'sequential_matcher', '--random_seed', '0', '--database_path', str(database),
            '--SiftMatching.use_gpu', '0', '--SiftMatching.num_threads', str(threads),
            '--SiftMatching.guided_matching', '1', '--SiftMatching.cross_check', '1',
            '--SiftMatching.max_ratio', '0.78', '--SiftMatching.max_distance', '0.7', '--SiftMatching.max_error', '2',
            '--SiftMatching.min_num_inliers', '25', '--SiftMatching.min_inlier_ratio', '0.25',
            '--SiftMatching.confidence', '0.999', '--SiftMatching.max_num_matches', '32768',
            '--SiftMatching.max_num_trials', '10000', '--SequentialMatching.overlap', '20',
            '--SequentialMatching.quadratic_overlap', '1', '--SequentialMatching.loop_detection', '0'])
        if not matched:
            report['failure'] = 'matching_failed'
            return report
        report['matched_database_sha256'] = sha(database)
        report['global'] = reconstruct('global', names, GLOBAL_OPTIONS)
        for index, expected in enumerate(windows):
            report['locals'].append(reconstruct(f'local_{index:03d}', expected, LOCAL_OPTIONS))
            write_json(output / 'report.json', report)
        report['complete'] = report['global']['complete'] and all(row['complete'] for row in report['locals'])
        report['local_complete_count'] = sum(row['complete'] for row in report['locals'])
        report['local_expected_count'] = len(windows)
        report['final_database_sha256'] = sha(database)
        return report
    finally:
        report['finished_unix'] = time.time()
        report['elapsed_seconds'] = report['finished_unix'] - report['started_unix']
        report['inputs_unchanged'] = input_record(images, intrinsics) == inputs
        report['complete'] = report['complete'] and report['inputs_unchanged']
        write_json(output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('images', 'intrinsics', 'output', 'executable', 'fork-source'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--mode', choices=('reference', 'moving'), default='moving')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--bundle-size', type=int, default=15)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=3600)
    args = vars(parser.parse_args())
    return 0 if run_reconstruction(**args)['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
