import json
from pathlib import Path
import sys
import sqlite3
import pytest

from backend.src import appearance_reconstruction as study


def test_matching_failure_is_retained_without_running_mappers_or_opening_truth(tmp_path, monkeypatch):
    images = tmp_path/'public/rgb'
    images.mkdir(parents=True)
    for i in range(90):
        (images/f'{i:06d}.png').write_bytes(b'control fixture; extraction is mocked')
    intrinsics = images.parent/'intrinsics.json'
    intrinsics.write_text(json.dumps(dict(width=640, height=480, fx=455, fy=452, cx=320, cy=240)))
    calls = []

    def command(argv, log, timeout):
        calls.append(argv)
        assert not any('evaluation' in part or part.endswith('.npz') for part in argv)
        Path(log).write_text('Preserved simulated matcher failure\n')
        return {'argv': argv, 'returncode': 3 if argv[1] == 'sequential_matcher' else 0,
                'timed_out': False, 'seconds': .01, 'log_sha256': study.sha(log)}

    monkeypatch.setattr(study, 'command', command)
    output = tmp_path/'job'
    result = study.run_job(images, intrinsics, output, tmp_path/'colmap')
    assert [x[1] for x in calls] == ['feature_extractor', 'sequential_matcher']
    assert result['execution_finished'] and result['inputs_unchanged']
    assert result['pipeline_stop'] == 'matching_failed_or_timed_out'
    assert result['models'] == {}
    assert json.loads((output/'REPORT.json').read_text()) == result
    assert 'simulated matcher failure' in (output/'logs/match.log').read_text()


def test_command_timeout_is_terminal_and_preserves_log(tmp_path):
    log = tmp_path/'timeout.log'
    result = study.command([sys.executable, '-c', 'import time; time.sleep(10)'], log, .05)
    assert result['timed_out'] and result['returncode'] != 0
    assert result['seconds'] < 3 and log.is_file()


def test_exhausted_aggregate_budget_does_not_launch_another_command(tmp_path, monkeypatch):
    images = tmp_path/'public/rgb'; images.mkdir(parents=True)
    for i in range(90):
        (images/f'{i:06d}.png').write_bytes(b'budget control')
    intrinsics = images.parent/'intrinsics.json'
    intrinsics.write_text(json.dumps(dict(width=640, height=480, fx=455, fy=452, cx=320, cy=240)))
    monkeypatch.setattr(study, 'command', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Command started after budget exhaustion')))
    result = study.run_job(images, intrinsics, tmp_path/'job', tmp_path/'colmap', remaining_seconds=0)
    assert result['commands'][0]['not_started']
    assert result['commands'][0]['reason'] == 'aggregate_study_budget_exhausted'


@pytest.mark.parametrize('corrupt_matches', [False, True])
def test_mapper_flag_writes_are_isolated_but_match_changes_rejected(tmp_path, monkeypatch, corrupt_matches):
    images = tmp_path/'public/rgb'; images.mkdir(parents=True)
    for i in range(90):
        (images/f'{i:06d}.png').write_bytes(b'isolated database control')
    intrinsics = images.parent/'intrinsics.json'
    intrinsics.write_text(json.dumps(dict(width=640, height=480, fx=455, fy=452, cx=320, cy=240)))
    mapper_inputs = []

    def command(argv, log, timeout):
        path = Path(argv[argv.index('--database_path')+1])
        with sqlite3.connect(path) as db:
            if argv[1] == 'feature_extractor':
                db.executescript('CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,data BLOB);'
                    "INSERT INTO matches VALUES(1,X'0102');"
                    'CREATE TABLE keypoints_det(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);'
                    "INSERT INTO keypoints_det VALUES(1,1,1,X'00000000');")
            elif argv[1] == 'mapper':
                assert db.execute('SELECT data FROM keypoints_det').fetchone()[0] == bytes(4)
                mapper_inputs.append(path)
                db.execute("UPDATE keypoints_det SET data=X'0000803f'")
                if corrupt_matches:
                    db.execute("UPDATE matches SET data=X'0103'")
        Path(log).write_text('simulated mapper auxiliary write\n')
        return {'argv': argv, 'returncode': 0, 'timed_out': False,
                'seconds': .01, 'log_sha256': study.sha(log)}

    monkeypatch.setattr(study, 'command', command)
    output = tmp_path/'job'
    if corrupt_matches:
        with pytest.raises(RuntimeError, match='Mapper changed calibration/features/matches'):
            study.run_job(images, intrinsics, output, tmp_path/'colmap')
        assert len(mapper_inputs) == 1
        assert not json.loads((output/'REPORT.json').read_text())['execution_finished']
    else:
        result = study.run_job(images, intrinsics, output, tmp_path/'colmap')
        assert result['execution_finished']
        assert len(set(mapper_inputs)) == 2
        assert study.sha(output/'database.db') == result['matched_database_sha256']
        assert all(x['input_tables_unchanged'] for x in result['mapper_databases'].values())
        assert all(x['initial_sha256'] != x['final_sha256'] for x in result['mapper_databases'].values())
