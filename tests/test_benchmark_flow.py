import numpy as np
import pytest

from backend.src.benchmark_flow import score, development_packages


def test_endpoint_error_and_nonfinite_coverage():
    target = np.zeros((2, 2, 2), dtype=np.float32)
    prediction = np.array([[[3, 4], [0, 0]], [[np.nan, 0], [99, 99]]])
    mask = np.array([[True, True], [True, False]])
    r = score(prediction, target, mask)
    assert r['eligible'] == 3 and r['predicted'] == 2
    assert r['prediction_coverage'] == pytest.approx(2 / 3)
    assert r['mean_epe'] == 2.5
    assert r['median_epe'] == 2.5
    assert r['fraction_gt3px'] == .5


def test_empty_support_is_not_zero_error():
    flow = np.zeros((2, 3, 2), dtype=np.float32)
    r = score(flow, flow, np.zeros((2, 3), dtype=bool))
    assert r['eligible'] == 0
    assert r['mean_epe'] is None and r['prediction_coverage'] is None


def test_truth_magnitude_does_not_depend_on_prediction_failures():
    target = np.array([[[3., 4.], [0., 0.]]])
    prediction = np.array([[[np.nan, 0.], [0., 0.]]])
    r = score(prediction, target, np.ones((1, 2), dtype=bool))
    assert r['mean_truth_magnitude'] == 2.5
    assert r['prediction_coverage'] == .5


def test_malformed_arrays_rejected():
    flow = np.zeros((2, 3, 2))
    with pytest.raises(ValueError):
        score(flow, flow, np.ones((2, 3)))
    with pytest.raises(ValueError):
        score(flow, flow, np.ones((3, 3), dtype=bool))


def indexed_bank(tmp_path, first_version=1):
    import json
    entries = []
    for offset, (family, condition) in enumerate(
            (f, c) for f in ('f01', 'f02', 'f03')
            for c in ('static', 'fem', 'nonfem')):
        version = first_version + offset
        row = dict(version=version, path=f'v{version}', family=family,
                   condition=condition, split='development')
        package = tmp_path / row['path']; package.mkdir()
        (package / 'DATASET.json').write_text(json.dumps(row))
        entries.append(row)
    # Its descriptor is deliberately absent: selection must not inspect it.
    entries.append(dict(version=999, path='v999', family='f05',
                        condition='fem', split='final_test'))
    index = tmp_path / 'SUITE_INDEX.json'
    index.write_text(json.dumps(dict(packages=entries)))
    return index, entries


@pytest.mark.parametrize('first_version', [1, 103])
def test_indexed_development_selection_and_reserved_exclusion(tmp_path, first_version):
    indexed_bank(tmp_path, first_version)
    selected = development_packages(tmp_path)
    assert [version for version, _, _ in selected] == list(range(first_version, first_version+9))
    assert {(d['family'], d['condition']) for _, _, d in selected} == {
        (f, c) for f in ('f01', 'f02', 'f03') for c in ('static', 'fem', 'nonfem')}


def test_duplicate_or_incomplete_development_identity_refused(tmp_path):
    import json
    index, entries = indexed_bank(tmp_path)
    entries[1]['family'] = entries[0]['family']
    entries[1]['condition'] = entries[0]['condition']
    index.write_text(json.dumps(dict(packages=entries)))
    with pytest.raises(ValueError, match='exactly three'):
        development_packages(tmp_path)


def test_descriptor_disagreement_refused_before_images_or_truth(tmp_path):
    import json
    _, entries = indexed_bank(tmp_path)
    first = entries[0].copy(); first['condition'] = 'nonfem'
    (tmp_path / 'v1/DATASET.json').write_text(json.dumps(first))
    with pytest.raises(ValueError, match='identity mismatch'):
        development_packages(tmp_path)
