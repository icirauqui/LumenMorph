"""Freeze the actually selected sources and replay outside a Git checkout."""
import json
from pathlib import Path
import tarfile

import pytest

from backend.src import benchmark


def source_tree(tmp_path):
    generator = tmp_path/'generator'
    solver = tmp_path/'alternate_solver/src'
    (generator/'backend/src').mkdir(parents=True)
    solver.mkdir(parents=True)
    (generator/'backend/src/example.py').write_text('value = 1\n')
    (generator/'backend/requirements.txt').write_text('numpy\n')
    (solver/'solver.py').write_text('value = 2\n')
    return {benchmark.GENERATOR_LOGICAL_ROOT: generator,
            'third_party/continuum-fem/src': solver}


def test_freeze_relocated_sources_without_git(tmp_path, monkeypatch):
    roots = source_tree(tmp_path)
    monkeypatch.setattr(benchmark, '_source_roots', lambda: roots)
    spec=tmp_path/'spec.json'; spec.write_text('{}')
    suite=tmp_path/'suite'
    benchmark.freeze(spec, suite)
    record=json.loads((suite/'provenance/SOURCE_FREEZE.json').read_text())
    assert record['git']['generator']['available'] is False
    assert record['git']['continuum_fem']['available'] is False
    solver_key='third_party/continuum-fem/src/solver.py'
    assert solver_key in record['sources']
    assert 'third_party/LumenMorph/backend/requirements.txt' in record['sources']
    with tarfile.open(suite/'provenance/sources.tar.gz') as archive:
        assert archive.extractfile(solver_key).read() == b'value = 2\n'
    # Replay must not depend on how deeply the generator is nested.
    monkeypatch.setattr(benchmark, '__file__', '/generator/backend/src/benchmark.py')
    benchmark.verify_sources(suite)
    (roots['third_party/continuum-fem/src']/'solver.py').write_text('value = 3\n')
    with pytest.raises(ValueError, match='Frozen sources changed'):
        benchmark.verify_sources(suite)


def test_resolved_solver_environment_is_used(monkeypatch, tmp_path):
    alternate=tmp_path/'solver/src'
    alternate.mkdir(parents=True)
    monkeypatch.setenv('CONTINUUM_FEM_SRC', str(alternate))
    assert benchmark._source_roots()['third_party/continuum-fem/src'] == alternate


def test_missing_solver_fails_before_creating_stage(tmp_path, monkeypatch):
    roots=source_tree(tmp_path)
    roots['third_party/continuum-fem/src']=tmp_path/'missing'
    monkeypatch.setattr(benchmark, '_source_roots', lambda: roots)
    spec=tmp_path/'spec.json'; spec.write_text('{}')
    with pytest.raises(RuntimeError, match='source directory not found'):
        benchmark.freeze(spec, tmp_path/'suite')
    assert not (tmp_path/'suite').exists()
