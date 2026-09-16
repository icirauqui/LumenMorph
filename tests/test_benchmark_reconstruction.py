import json
from pathlib import Path
import subprocess

import pytest

from backend.src import benchmark_reconstruction as br


def test_explicit_windows_preserve_filename_order_and_include_tail():
    names = [f'{i:06d}.png' for i in range(90)]
    windows = br.window_names(names)
    assert len(windows) == 9
    assert windows[-1] == names[75:90]
    assert windows[-2] == names[70:85]
    assert br.window_names(names[:35])[-1] == names[20:35]


def test_colmap_parser_preserves_empty_observation_rows(tmp_path):
    path = tmp_path / 'images.txt'
    path.write_text('# comment\n1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 0 0 0 1 b.png\n2 3 -1 -1\n')
    assert br.read_names(path) == ['a.png', 'b.png']
    path.write_text('1 1 0 0 0 0 0 0 1 a.png\n')
    with pytest.raises(ValueError, match='Missing'):
        br.read_names(path)


def inputs(tmp_path, n=16):
    rgb = tmp_path / 'public' / 'moving' / 'rgb'
    rgb.mkdir(parents=True)
    for i in range(n):
        (rgb / f'{i:06d}.png').write_bytes(b'png fixture')
    intr = tmp_path / 'public' / 'intrinsics.json'
    intr.write_text(json.dumps(dict(width=640, height=480, fx=450, fy=450, cx=320, cy=240)))
    return rgb, intr


def test_invalid_intrinsics_and_stationary_rejected_before_outputs(tmp_path):
    rgb, intr = inputs(tmp_path)
    with pytest.raises(ValueError, match='Stationary'):
        br.run_reconstruction(rgb, intr, tmp_path / 'out', '/no/executable', '/no/source', mode='stationary')
    assert not (tmp_path / 'out').exists()
    intr.write_text(json.dumps(dict(width=640, height=480, fx=-1, fy=450, cx=320, cy=240)))
    with pytest.raises(ValueError, match='positive'):
        br.input_record(rgb, intr)


def install_fake(monkeypatch, *, incomplete=None, failure=None):
    commands = []
    monkeypatch.setattr(br, 'provenance', lambda *a: {'executable_sha256': 'fixture'})
    def run(argv, **kwargs):
        commands.append(argv)
        mode = argv[1]
        if mode == 'feature_extractor':
            Path(argv[argv.index('--database_path') + 1]).write_bytes(b'measured-only DB')
        elif mode == 'mapper':
            output = Path(argv[argv.index('--output_path') + 1])
            expected = Path(argv[argv.index('--image_list_path') + 1]).read_text().splitlines()
            if output.name == failure:
                return subprocess.CompletedProcess(argv, 1)
            candidate = output / '0'
            candidate.mkdir()
            if output.name == incomplete:
                expected = expected[:-1]
            (candidate / 'images.txt').write_text(''.join(f'{i+1} 1 0 0 0 0 0 0 1 {name}\n1 2 3 -1\n' for i, name in enumerate(expected)))
            (candidate / 'points3D.txt').write_text('1 0 0 1 255 0 0 0 1 0 2 0\n')
            (candidate / 'images.bin').write_bytes(b'binary retained')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(br.subprocess, 'run', run)
    return commands


def test_incomplete_local_preserved_and_next_window_runs(tmp_path, monkeypatch):
    rgb, intr = inputs(tmp_path)
    commands = install_fake(monkeypatch, incomplete='local_000')
    out = tmp_path / 'reconstruction'
    result = br.run_reconstruction(rgb, intr, out, '/fake/colmap', '/fake/fork')
    assert not result['complete'] and result['global']['complete']
    assert result['local_complete_count'] == 1 and result['local_expected_count'] == 2
    assert result['locals'][0]['candidates'][0]['missing_names'] == ['000014.png']
    assert (out / 'models/local_000/0/images.bin').is_file()
    assert (out / 'models/local_001/0/images.bin').is_file()
    assert result['inputs_unchanged']
    mapper_commands = [cmd for cmd in commands if cmd[1] == 'mapper']
    assert len(mapper_commands) == 3
    assert all('--image_list_path' in cmd and '--input_path' not in cmd for cmd in mapper_commands)
    assert all('truth' not in ' '.join(cmd) for cmd in commands)
    with pytest.raises(FileExistsError):
        br.run_reconstruction(rgb, intr, out, '/fake/colmap', '/fake/fork')


def test_global_mapper_failure_does_not_hide_local_outcomes(tmp_path, monkeypatch):
    rgb, intr = inputs(tmp_path)
    install_fake(monkeypatch, failure='global')
    result = br.run_reconstruction(rgb, intr, tmp_path / 'out', '/fake/colmap', '/fake/fork')
    assert not result['complete']
    assert result['global']['candidates'] == []
    assert result['local_complete_count'] == 2


def test_reference_has_no_local_reconstruction(tmp_path, monkeypatch):
    rgb, intr = inputs(tmp_path, 3)
    commands = install_fake(monkeypatch)
    result = br.run_reconstruction(rgb, intr, tmp_path / 'out', '/fake/colmap', '/fake/fork', mode='reference')
    assert result['complete'] and result['locals'] == []
    assert len([cmd for cmd in commands if cmd[1] == 'mapper']) == 1
