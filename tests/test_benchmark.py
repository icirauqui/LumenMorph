"""Benchmark producer contracts; synthetic fixtures are not research evidence."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.src import benchmark as producer
from backend.src.benchmark_reader import PublicDataset, EvaluationDataset


@pytest.fixture(scope="module")
def tiny_benchmark(tmp_path_factory):
    root = tmp_path_factory.mktemp("benchmark_contract") / "dataset"
    spec = json.loads((Path(__file__).parents[1] / "configs/tubular_benchmark_v1.json").read_text())
    spec["families"] = [copy.deepcopy(spec["families"][0])]
    family = spec["families"][0]
    cfg = family["generator_config"]
    cfg["intrinsics"] = dict(width=24, height=18, fx=17., fy=17., cx=12., cy=9.)
    cfg["tube"]["dimensions"].update(radial_samples=12, longitudinal_samples=20)
    cfg["tube"]["surface"]["bump_count"] = 2
    cfg["texture"].update(width=32, height=32, min_orb_features=0)
    cfg["photometric"].update(gaussian_sigma=0., gain_jitter=0.)
    spec_path = root.parent / "tiny_spec.json"
    spec_path.write_text(json.dumps(spec))
    patch = pytest.MonkeyPatch()

    def cheap_fields(geom, tube, patches, peaks):
        fields = producer.prescribed_fields(geom, patches, peaks)
        # Only the output contract is tested; this intentionally avoids claiming
        # equilibrium verification from this inexpensive fake solver.
        return fields, {"basis_gauss_strain": np.zeros((3, 1, 6))}, [{"test_fixture": True}]

    patch.setattr(producer, "solve_benchmark_fields", cheap_fields)
    producer.freeze(spec_path, root)
    producer.prepare(root, family["id"])
    producer.render(root, family["id"], "static", "reference")
    static = producer.create_package(root, family["id"], "static")
    fem = producer.create_package(root, family["id"], "fem")
    producer.render(root, family["id"], "static", "stationary")
    producer.render(root, family["id"], "fem", "stationary")
    producer.render(root, family["id"], "fem", "moving")
    yield root, spec, family, static, fem
    patch.undo()


def test_phase_schedule_declares_rest_low_nominal_return():
    weights = producer.waveform()
    assert weights.shape == (90, 3)
    np.testing.assert_array_equal(weights[:15], 0.)
    np.testing.assert_allclose(weights[30:45], .02)
    np.testing.assert_allclose(weights[60:75], 1.)
    np.testing.assert_allclose(weights[-1], 0., atol=1e-15)
    assert np.all(np.diff(weights[15:30], axis=0) >= 0)
    assert np.all(np.diff(weights[45:60], axis=0) >= 0)
    assert not np.array_equal(weights[15:30, 0], weights[15:30, 1])
    with pytest.raises(ValueError):
        producer.waveform(20)


def test_freeze_records_sources_and_prevents_in_place_overwrite(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    frozen = json.loads((root / "FROZEN.json").read_text())
    assert frozen["spec_sha256"] == producer.sha(root / "SPECIFICATION.json")
    provenance = json.loads((root / "provenance/SOURCE_FREEZE.json").read_text())
    assert provenance["source_archive_sha256"] == producer.sha(root / "provenance/sources.tar.gz")
    assert any(p.endswith("backend/src/dense_truth.py") for p in provenance["sources"])
    assert frozen["recovery_evaluated"] is False
    with pytest.raises(FileExistsError):
        producer.prepare(root, family["id"])
    with pytest.raises(FileExistsError):
        producer.render(root, family["id"], "fem", "stationary")


def test_reference_is_static_and_stationary_action_uses_terminal_pose(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    reference = np.load(root / "families" / family["id"] / "reference_truth/states.npz")
    action = np.load(fem / "evaluation/stationary/states.npz")
    moving = np.load(fem / "evaluation/moving/states.npz")
    np.testing.assert_array_equal(reference["displacement"], 0.)
    np.testing.assert_allclose(action["R_wc"], np.broadcast_to(reference["R_wc"][-1], action["R_wc"].shape))
    np.testing.assert_allclose(action["C_world"], np.broadcast_to(reference["C_world"][-1], action["C_world"].shape))
    np.testing.assert_array_equal(action["displacement"][:15], 0.)
    np.testing.assert_array_equal(action["vertices"], moving["vertices"])
    assert np.max(np.linalg.norm(moving["C_world"] - action["C_world"], axis=1)) > 0.1
    bridge = cv2.imread(str(fem / "public/reference/rgb/000089.png"))
    first = cv2.imread(str(fem / "public/stationary/rgb/000000.png"))
    np.testing.assert_array_equal(bridge, first)  # noise disabled in test fixture


def test_phase_displacements_follow_declared_amplitudes(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    state = np.load(fem / "evaluation/stationary/states.npz")
    delta = state["displacement"]
    np.testing.assert_allclose(delta[30:45], np.broadcast_to(delta[30], delta[30:45].shape))
    np.testing.assert_allclose(delta[60:75], np.broadcast_to(delta[60], delta[60:75].shape))
    # Float32 stored mesh inputs introduce subtraction roundoff.
    np.testing.assert_allclose(delta[30], .02 * delta[60], atol=3e-6)
    assert np.max(np.linalg.norm(delta[60], axis=1)) > 0
    static_state = np.load(static / "evaluation/stationary/states.npz")
    np.testing.assert_array_equal(static_state["displacement"], 0.)


def test_public_inputs_resolve_and_exclude_dense_truth(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    descriptor = json.loads((fem / "DATASET.json").read_text())
    assert descriptor["inference_allowlist"] == ["public", "reconstruction"]
    assert descriptor["ground_truth_at_inference"] is False
    assert (fem / "public/acquisition.json").is_file()
    for stream in ("reference", "moving", "stationary"):
        public = fem / "public" / stream
        manifest = json.loads((public / "frames.json").read_text())
        assert (public / manifest["intrinsics"]).is_file()
        assert len(manifest["frames"]) == 90
        for frame in manifest["frames"]:
            assert (public / "rgb" / frame["image"]).is_file()
            assert "surface_fraction" not in frame
        assert not list(public.rglob("*.npz"))
        assert not list(public.rglob("*poses*"))


def test_timestamps_and_dense_flow_frame_contracts(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    for stream in ("moving", "stationary"):
        private = fem / "evaluation" / stream
        state = np.load(private / "states.npz")
        manifest = json.loads((fem / "public" / stream / "frames.json").read_text())
        public_time = np.array([x["timestamp_s"] for x in manifest["frames"]])
        np.testing.assert_allclose(np.diff(public_time), 1 / spec["fps"])
        np.testing.assert_allclose(state["time_seconds"], public_time)
        assert len(list((private / "dense").glob("*.npz"))) == 90
        assert len(list((private / "flow_forward").glob("*.npz"))) == 89
        dense = np.load(private / "dense/000060.npz")
        assert dense["depth_z"].shape == (18, 24)
        assert dense["material_barycentric"].shape == (18, 24, 3)
        assert dense["valid"].dtype == bool
        valid = dense["valid"]
        assert np.any(valid)
        np.testing.assert_allclose(dense["material_barycentric"][valid].sum(axis=1), 1., atol=1e-6)
        flow = np.load(private / "flow_forward/000060.npz")
        assert flow["flow_uv"].shape == (18, 24, 2)
        masks = [flow[x] for x in ("visible", "occluded", "out_of_frame", "behind_camera", "unresolved")]
        np.testing.assert_array_equal(sum(x.astype(int) for x in masks), valid.astype(int))


def test_completed_stage_hashes_match_outputs(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    for stage in (fem / "public/stationary", fem / "evaluation/stationary"):
        complete = json.loads((stage / "COMPLETE.json").read_text())
        for path, expected in complete["files"].items():
            assert producer.sha(stage / path) == expected


def test_reader_material_queries_match_exact_dense_state_and_reject_background(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    public = PublicDataset(fem)
    assert public.rgb(0).shape == (18, 24, 3)
    assert public.public_paths()["reference_rgb"].is_dir()
    assert not hasattr(public, "dense")
    reader = EvaluationDataset(fem)
    dense = reader.dense(60)
    valid = dense["valid"]
    ids, bary = dense["triangle_id"][valid], dense["material_barycentric"][valid]
    geometry = reader.geometry()
    state = reader.state(60)
    points = reader.material_points(ids, bary, 60)
    pc = (points - state["C_world"]) @ state["R_wc"]
    np.testing.assert_allclose(pc[:, 2], dense["depth_z"][valid], atol=5e-6)
    expected_delta = np.sum(bary[..., None] * state["displacement"][geometry["triangles"][ids]], axis=1)
    np.testing.assert_allclose(reader.material_displacement(ids, bary, 60), expected_delta, atol=1e-12)
    np.testing.assert_array_equal(reader.state(0, "reference")["displacement"], 0.)
    with pytest.raises(ValueError):
        reader.material_points(np.array([-1]), np.array([[1., 0., 0.]]))
    with pytest.raises(ValueError):
        reader.material_points(np.array([0]), np.array([[1., 1., 0.]]))
    with pytest.raises(IndexError):
        reader.rgb(-1)


def test_reader_derives_backward_flow_without_writing_or_using_inference(tiny_benchmark):
    root, spec, family, static, fem = tiny_benchmark
    reader = EvaluationDataset(fem)
    forward = reader.compute_flow(60, 61)
    with np.load(fem / "evaluation/stationary/flow_forward/000060.npz") as stored:
        for key in stored.files:
            np.testing.assert_array_equal(forward[key], stored[key])
    backward = reader.compute_flow(61, 60)
    valid = reader.dense(61)["valid"]
    assert np.any(backward["visible"])
    np.testing.assert_allclose(backward["flow_uv"][valid], 0., atol=1e-4)
    assert not (fem / "evaluation/stationary/flow_backward").exists()
