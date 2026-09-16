"""Fixed image-only appearance reconstruction study, with one shared provenance.

This module imports no ground-truth reader or numerical evaluator. Each job
receives public RGB and calibration; evaluation is a separate later command.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

from .benchmark_reconstruction import GLOBAL_OPTIONS, LOCAL_OPTIONS, input_record, options, read_names, sha, write_json


EXTRACTION = {
    'ImageReader.single_camera': 1, 'ImageReader.camera_model': 'PINHOLE',
    'SiftExtraction.use_gpu': 0, 'SiftExtraction.num_threads': 8,
    'SiftExtraction.max_image_size': 800, 'SiftExtraction.max_num_features': 12000,
    'SiftExtraction.peak_threshold': .004, 'SiftExtraction.edge_threshold': 10,
    'SiftExtraction.first_octave': -1, 'SiftExtraction.num_octaves': 4,
    'SiftExtraction.octave_resolution': 3, 'SiftExtraction.max_num_orientations': 2,
}
MATCHING = {
    'SiftMatching.use_gpu': 0, 'SiftMatching.num_threads': 8,
    'SiftMatching.guided_matching': 1, 'SiftMatching.cross_check': 1,
    'SiftMatching.max_ratio': .78, 'SiftMatching.max_distance': .7,
    'SiftMatching.max_error': 2, 'SiftMatching.min_num_inliers': 25,
    'SiftMatching.min_inlier_ratio': .25, 'SiftMatching.confidence': .999,
    'SiftMatching.max_num_matches': 32768, 'SiftMatching.max_num_trials': 10000,
    'SequentialMatching.overlap': 20, 'SequentialMatching.quadratic_overlap': 1,
    'SequentialMatching.loop_detection': 0,
}
METHOD = {'threads': 8, 'random_seed': 0, 'expected_frames': 90,
    'primary_jobs': 20, 'repeat_job': 'f01__diffuse_symmetric_repeat',
    'local_frames': list(range(60, 75)), 'extraction': EXTRACTION, 'matching': MATCHING,
    'global_mapper': GLOBAL_OPTIONS, 'local_mapper': LOCAL_OPTIONS,
    'timeouts_seconds': {'extract': 300, 'match': 3600, 'global': 3600, 'local': 1200, 'convert': 60},
    'study_wall_cap_seconds': 86400, 'study_storage_cap_bytes': 24*2**30,
    'scope': 'Image-estimated calibrated global and local SfM; no ground truth at inference.'}


def method_sources():
    folder = Path(__file__).resolve().parent
    return [folder/name for name in ('appearance_reconstruction.py',
        'appearance_reconstruction_metrics.py', 'benchmark_reconstruction.py')]


def prepare(output, baseline, protocol):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = output/'provenance'
    provenance.mkdir()
    prior = baseline/'v4/reconstruction/moving/provenance'
    old = json.loads((prior/'manifest.json').read_text())
    old_report = json.loads((baseline/'v4/reconstruction/moving/report.json').read_text())
    option_audit = []
    for label, chosen in [('01_extract', EXTRACTION), ('02_match', MATCHING),
                          ('global_mapper', GLOBAL_OPTIONS), ('local_000_mapper', LOCAL_OPTIONS)]:
        argv = next(row['argv'] for row in old_report['commands'] if row['label'] == label)
        recorded = dict(zip(argv[2::2], argv[3::2]))
        for key, value in chosen.items():
            previous = recorded['--'+key]
            changed_threads = key.endswith('.num_threads')
            try:
                same = float(previous) == float(value)
            except ValueError:
                same = str(previous) == str(value)
            if not same and not changed_threads:
                raise ValueError('Unregistered change from baseline CLI option: '+key)
            option_audit.append({'key': key, 'baseline': previous, 'chosen': value,
                                 'thread_change_only': not same and changed_threads})
    write_json(provenance/'BASELINE_OPTION_AUDIT.json', option_audit)
    # The same qualified binary/source/build assets are referenced once. A
    # relocatable combined release keeps the baseline and extension together.
    for name in ('colmap', 'fork_head.tar', 'CMakeCache.txt', 'fork_dirty.patch'):
        if sha(prior/name) != old['files_sha256'][name]:
            raise ValueError(f'Baseline reconstruction provenance changed: {name}')
        (provenance/name).symlink_to(os.path.relpath(prior/name, provenance))
    shutil.copy2(prior/'manifest.json', provenance/'BASELINE_PROVENANCE.json')
    shutil.copy2(protocol, provenance/'PROTOCOL.md')
    source = provenance/'source/backend/src'
    source.mkdir(parents=True)
    for p in method_sources():
        shutil.copy2(p, source/p.name)
    (source/'__init__.py').write_text('')
    (source.parent/'__init__.py').write_text('')
    executable = str(provenance/'colmap')
    for label, args in [('executable_help', [executable, '-h']), ('ldd', ['ldd', executable])]:
        p = subprocess.run(args, capture_output=True, text=True, check=True)
        (provenance/(label+'.txt')).write_text(p.stdout+p.stderr)
    write_json(output/'METHOD.json', METHOD)
    write_json(output/'METHOD_FREEZE.json', {'method_sha256': sha(output/'METHOD.json'),
        'sources': {p.name: sha(p) for p in method_sources()},
        'binary_sha256': sha(provenance/'colmap'), 'protocol_sha256': sha(provenance/'PROTOCOL.md'),
        'source_binary_correspondence': old['source_binary_correspondence'],
        'baseline_source_commit': old['source_commit'], 'dataset_inputs': 'Frozen separately before inference; no RGB outcomes yet.'})


def command(argv, log, timeout):
    env = {**os.environ, 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'QT_QPA_PLATFORM': 'offscreen'}
    start = time.perf_counter()
    with Path(log).open('x') as stream:
        stream.write(json.dumps(argv)+'\n'); stream.flush()
        process = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return {'argv': argv, 'returncode': process.returncode, 'timed_out': timed_out,
            'seconds': time.perf_counter()-start, 'log_sha256': sha(log)}


def database_inputs(path):
    """Hash SQL inputs, allowing only fork-owned keypoints_det values to change.

    SQLite file headers/layout may change on open/close/VACUUM. All schema,
    image/calibration, feature and match values remain part of this contract.
    The auxiliary flag table's identity and array shape also remain fixed.
    """
    result = {}
    with sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True) as db:
        schema = db.execute('SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name').fetchall()
        result['schema'] = hashlib.sha256(json.dumps(schema).encode()).hexdigest()
        for _, name, _, _ in schema:
            if not any(row[0] == 'table' and row[1] == name for row in schema):
                continue
            quoted = '"'+name.replace('"', '""')+'"'
            columns = 'image_id,rows,cols,length(data)' if name == 'keypoints_det' else '*'
            digest = hashlib.sha256()
            for row in db.execute(f'SELECT {columns} FROM {quoted} ORDER BY rowid'):
                for value in row:
                    payload = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
                    digest.update(type(value).__name__.encode()+b':'+str(len(payload)).encode()+b':'+payload)
                digest.update(b'\n')
            result[name] = digest.hexdigest()
    return result


def run_job(images, intrinsics, output, executable, *, remaining_seconds=None):
    images, output, executable = Path(images).resolve(), Path(output).resolve(), str(Path(executable).resolve())
    inputs = input_record(images, intrinsics)
    names = [f'{i:06d}.png' for i in range(90)]
    if sorted(inputs['rgb_sha256']) != names:
        raise ValueError('Expected all 90 fixed RGB inputs')
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    (output/'lists').mkdir()
    database = output/'database.db'
    write_json(output/'INPUTS.json', inputs)
    report = {'scope': METHOD['scope'], 'commands': [], 'models': {}, 'execution_finished': False}
    start = time.perf_counter()

    def call(label, argv, budget):
        if remaining_seconds is not None:
            remaining = remaining_seconds-(time.perf_counter()-start)
            if remaining <= 0:
                report['commands'].append({'label': label, 'argv': argv, 'returncode': None,
                    'timed_out': True, 'not_started': True, 'reason': 'aggregate_study_budget_exhausted'})
                write_json(output/'REPORT.json', report)
                return False
            budget = min(budget, remaining)
        row = command(argv, output/'logs'/f'{label}.log', budget)
        report['commands'].append({'label': label, **row})
        write_json(output/'REPORT.json', report)
        return row['returncode'] == 0 and not row['timed_out']

    try:
        if not call('extract', [executable, 'feature_extractor', '--random_seed', '0',
            '--database_path', str(database), '--image_path', str(images),
            '--ImageReader.camera_params', inputs['camera_params'], *options(EXTRACTION)], 300):
            report['pipeline_stop'] = 'feature_extraction_failed_or_timed_out'
            report['execution_finished'] = True
            return report
        if not call('match', [executable, 'sequential_matcher', '--random_seed', '0',
            '--database_path', str(database), *options(MATCHING)], 3600):
            report['pipeline_stop'] = 'matching_failed_or_timed_out'
            report['execution_finished'] = True
            return report
        frozen_database = sha(database)
        report['matched_database_sha256'] = frozen_database
        frozen_inputs = database_inputs(database)
        report['matched_database_inputs'] = frozen_inputs
        report['mapper_databases'] = {}
        for label, expected, mapper, budget in [
            ('global', names, GLOBAL_OPTIONS, 3600),
            ('local', [names[i] for i in range(60, 75)], LOCAL_OPTIONS, 1200)]:
            model = output/'models'/label
            model.mkdir(parents=True)
            listing = output/'lists'/f'{label}.txt'
            listing.write_text('\n'.join(expected)+'\n')
            working_database = output/(label+'_database.db')
            shutil.copyfile(database, working_database)
            if sha(working_database) != frozen_database:
                raise RuntimeError('Mapper working copy differs from frozen matches')
            success = call(label, [executable, 'mapper', '--random_seed', '0',
                '--database_path', str(working_database), '--image_path', str(images), '--output_path', str(model),
                '--image_list_path', str(listing), '--Mapper.num_threads', '8', *options(mapper)], budget)
            candidates = []
            for candidate in sorted(p for p in model.iterdir() if p.is_dir()):
                converted = call(label+'_'+candidate.name+'_convert', [executable, 'model_converter',
                    '--input_path', str(candidate), '--output_path', str(candidate), '--output_type', 'TXT'], 60)
                registered, error = [], None
                if converted:
                    try:
                        registered = read_names(candidate/'images.txt')
                    except (OSError, ValueError) as exc:
                        error = repr(exc)
                candidates.append({'path': str(candidate.relative_to(output)), 'registered_names': registered,
                    'parse_error': error, 'converted': converted,
                    'files': {p.name: sha(p) for p in candidate.iterdir() if p.is_file()}})
            report['models'][label] = {'expected_names': expected, 'mapper_completed': success,
                'candidates': candidates, 'complete_registration': success and len(candidates) == 1
                and candidates[0]['converted'] and candidates[0]['parse_error'] is None
                and sorted(candidates[0]['registered_names']) == expected}
            if sha(database) != frozen_database:
                raise RuntimeError('Mapper changed the shared immutable match database')
            after_inputs = database_inputs(working_database)
            report['mapper_databases'][label] = {'path': working_database.name,
                'initial_sha256': frozen_database, 'final_sha256': sha(working_database),
                'input_tables_unchanged': after_inputs == frozen_inputs,
                'final_input_tables': after_inputs}
            if after_inputs != frozen_inputs:
                raise RuntimeError('Mapper changed calibration/features/matches or auxiliary table shape')
            write_json(output/'REPORT.json', report)
        report['execution_finished'] = True
    finally:
        report['seconds'] = time.perf_counter()-start
        report['inputs_unchanged'] = inputs == input_record(images, intrinsics)
        report['execution_finished'] = report['execution_finished'] and report['inputs_unchanged']
        write_json(output/'REPORT.json', report)
    return report


def run_study(dataset, output):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    lock = json.loads((output/'METHOD_FREEZE.json').read_text())
    if sha(output/'METHOD.json') != lock['method_sha256'] or METHOD != json.loads((output/'METHOD.json').read_text()):
        raise ValueError('Method options changed since freeze')
    for p in method_sources():
        if sha(p) != lock['sources'][p.name]:
            raise ValueError('Method source changed since freeze')
    if sha(output/'provenance/colmap') != lock['binary_sha256']:
        raise ValueError('COLMAP executable changed')
    prior_attempt_seconds = 0.0
    for prior in lock.get('preserved_interrupted_attempts', []):
        path = output/prior['report']
        if sha(path) != prior['sha256']:
            raise ValueError('Preserved interrupted-attempt report changed')
        prior_attempt_seconds += json.loads(path.read_text())['seconds']
    if not (dataset/'GENERATION_COMPLETE.json').is_file():
        raise ValueError('Full production must finish before the fixed study')
    validation = json.loads((dataset/'RELEASE_VALIDATION.json').read_text())
    if not validation.get('all_data_checks_passed'):
        raise ValueError('Independent data checks must pass before inference')
    with (output/'execution.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        public = []
        for family in ['f01', 'f04', 'f05', 'f06']:
            folder = dataset/'families'/family/'public'
            manifest = json.loads((folder/'frames.json').read_text())
            expected_conditions = ['diffuse_symmetric', 'diffuse_asymmetric', 'coated_symmetric',
                                   'diffuse_exposure_half', 'diffuse_exposure_double']
            if list(manifest['conditions']) != expected_conditions:
                raise ValueError('Appearance conditions differ from the fixed design')
            for condition in manifest['conditions']:
                public.append({'family': family, 'condition': condition,
                    'inputs': input_record(folder/condition/'rgb', folder/'intrinsics.json')})
        public.append({**public[0], 'job_id': METHOD['repeat_job'], 'repeat_of': 'f01__diffuse_symmetric'})
        for job in public:
            record = job['inputs']
            files = {Path(record['image_directory'])/name: value for name, value in record['rgb_sha256'].items()}
            files[Path(record['intrinsics']['path'])] = record['intrinsics']['sha256']
            for path, digest in files.items():
                if validation['verified_file_sha256'][str(path.relative_to(dataset))] != digest:
                    raise ValueError('Public input changed after independent data validation')
        inputs = {'jobs': public, 'validation_sha256': sha(dataset/'RELEASE_VALIDATION.json'),
                  'generation_freeze_sha256': sha(dataset/'FROZEN.json')}
        if (output/'INPUT_FREEZE.json').exists():
            if inputs != json.loads((output/'INPUT_FREEZE.json').read_text()):
                raise ValueError('Study inputs changed since initial run')
        else:
            write_json(output/'INPUT_FREEZE.json', inputs)
        start = time.perf_counter()
        completed = []
        for job in public:
            identifier = job.get('job_id', job['family']+'__'+job['condition'])
            dest = output/'jobs'/identifier
            if dest.exists():
                saved = json.loads((dest/'REPORT.json').read_text())
                if not saved.get('execution_finished'):
                    raise RuntimeError('Interrupted study job requires inspection before resumption')
                if json.loads((dest/'INPUTS.json').read_text()) != job['inputs']:
                    raise ValueError('Previously executed study inputs differ')
                completed.append(identifier)
                continue
            prior_seconds = prior_attempt_seconds+sum(json.loads((output/'jobs'/name/'REPORT.json').read_text())['seconds'] for name in completed)
            if prior_seconds >= METHOD['study_wall_cap_seconds']:
                raise RuntimeError('Registered study time budget exhausted')
            inputs_one = job['inputs']
            report = run_job(inputs_one['image_directory'], inputs_one['intrinsics']['path'], dest,
                output/'provenance/colmap', remaining_seconds=METHOD['study_wall_cap_seconds']-prior_seconds)
            if not report['execution_finished']:
                raise RuntimeError('Study job did not finish with unchanged public inputs')
            completed.append(identifier)
            if prior_seconds+report['seconds'] > METHOD['study_wall_cap_seconds']:
                raise RuntimeError('Aggregate study time budget exhausted; terminal job retained')
            size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size > METHOD['study_storage_cap_bytes']:
                raise RuntimeError('Registered study storage budget exhausted')
            write_json(output/'STATUS.json', {'completed_jobs': completed, 'current_run_seconds': time.perf_counter()-start, 'bytes': size})
            print(json.dumps({'job': identifier, 'finished_jobs': len(completed), 'seconds': report['seconds']}), flush=True)
        write_json(output/'EXECUTION_COMPLETE.json', {'jobs': completed,
            'all_20_conditions_accounted_for': len(completed) == 21,
            'repeat_control_completed': METHOD['repeat_job'] in completed,
            'scope': 'Fixed inference execution, including failures; evaluator results are separate.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    for name in ('output', 'baseline', 'protocol'):
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output, args.baseline, args.protocol)
    else:
        def terminated(signum, frame):
            raise TimeoutError('Study termination requested; incomplete job retained')
        signal.signal(signal.SIGTERM, terminated)
        run_study(args.dataset, args.output)


if __name__ == '__main__':
    main()
