import numpy as np

from backend.src.benchmark_image_quality import audit_flow_samples
from backend.src.dense_truth import flow_to_frame, render_rgb_and_truth
from backend.src.types import CameraIntrinsics


def test_independent_flow_checks_projection_occlusion_and_corruption():
    intr = CameraIntrinsics(width=8, height=6, fx=8., fy=8., cx=4., cy=3.)
    front = np.array([[-3., -3., 2.], [3., -3., 2.], [3., 3., 2.], [-3., 3., 2.]], dtype=np.float32)
    back = front + [0, 0, 2]
    vertices = np.vstack((front, back)).astype(np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    _, truth = render_rgb_and_truth(vertices, triangles, np.zeros((8, 2)), np.zeros((2, 2, 3), np.uint8), intr, np.eye(3), np.zeros(3))
    target = vertices.copy()
    target[:4, 2] += 3
    flow = flow_to_frame(truth, target, triangles, intr, np.eye(3), np.zeros(3))
    report = audit_flow_samples(truth, target, triangles, np.eye(3), np.zeros(3), intr, flow, sample_count=48)
    assert report["passed"]
    assert report["source_foreground_count"] == 48
    assert flow["occluded"].all()
    bad = {k: v.copy() for k, v in flow.items()}
    bad["flow_uv"][3, 4, 0] += .02
    assert not audit_flow_samples(truth, target, triangles, np.eye(3), np.zeros(3), intr, bad, sample_count=48)["passed"]
    bad = {k: v.copy() for k, v in flow.items()}
    bad["visible"][3, 4] = True
    assert not audit_flow_samples(truth, target, triangles, np.eye(3), np.zeros(3), intr, bad, sample_count=48)["all_pixel_mask_partition_passed"]


def test_independent_flow_keeps_background_absent():
    intr = CameraIntrinsics(width=8, height=6, fx=8., fy=8., cx=4., cy=3.)
    vertices = np.array([[-.25, -.25, 2.], [.25, -.25, 2.], [0., .25, 2.]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]])
    _, truth = render_rgb_and_truth(vertices, triangles, np.zeros((3, 2)), np.zeros((2, 2, 3), np.uint8), intr, np.eye(3), np.zeros(3))
    flow = flow_to_frame(truth, vertices, triangles, intr, np.eye(3), np.zeros(3))
    report = audit_flow_samples(truth, vertices, triangles, np.eye(3), np.zeros(3), intr, flow, sample_count=48)
    assert report["passed"]
    assert report["source_foreground_count"] < 48
    flow["flow_uv"][0, 0] = 0
    assert not audit_flow_samples(truth, vertices, triangles, np.eye(3), np.zeros(3), intr, flow, sample_count=48)["passed"]
