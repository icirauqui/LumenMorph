import json

import cv2
import numpy as np
import pytest

pytest.importorskip('mitsuba')
from test_appearance_suite import fixture
from backend.src.appearance_suite import run
from backend.src.appearance_release_validation import validate, digest


def test_complete_saved_fixture_replays_and_relocates_but_prefix_cannot_approve(tmp_path):
    _, data, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='terminal complete'):
        validate(data, tmp_path/'premature.json')
    run(data)
    prefix = validate(data, tmp_path/'prefix.json', full=False, prefix_frames=1)
    assert prefix['checked_subset_passed'] and not prefix['all_data_checks_passed']
    assert prefix['verified_frames'] == 1 and prefix['expected_frames'] == 2
    full = validate(data, tmp_path/'RELEASE_VALIDATION.json')
    assert full['all_data_checks_passed'] and full['verified_frames'] == 2
    checks = full['frozen_replay_and_portability'][0]
    assert len(checks['replays']) == 3 and checks['public_reader_without_truth_passed']
    assert checks['relocated_rgb_and_material_geometry_exact']
    assert all(r['plane_ray_check']['hit_miss_disagreements'] == 0 for r in full['rows'])


def test_rehashed_wrong_rgb_is_rejected_by_semantic_check(tmp_path):
    _, data, _ = fixture(tmp_path); run(data)
    family = data/'families/f01'
    relative = 'public/diffuse_exposure_double/rgb/000000.png'
    image = cv2.imread(str(family/relative))
    image[0, 0] = [0, 0, 0] if image[0, 0].any() else [255, 255, 255]
    cv2.imwrite(str(family/relative), image)
    commit = family/'commits/000000.json'
    record = json.loads(commit.read_text()); record['files'][relative] = digest(family/relative)
    commit.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='radiance/exposure'):
        validate(data, tmp_path/'invalid.json')
    assert not (tmp_path/'invalid.json').exists()


def test_release_validation_refuses_undocumented_files_and_prefix_approval_name(tmp_path):
    _, data, _ = fixture(tmp_path); run(data)
    with pytest.raises(ValueError, match='distinct non-release'):
        validate(data, tmp_path/'RELEASE_VALIDATION.json', full=False)
    (data/'families/f01/evaluation/extra.npy').write_bytes(b'not in any frame or preparation manifest')
    with pytest.raises(ValueError, match='Unexpected or missing family payload'):
        validate(data, tmp_path/'invalid.json')
