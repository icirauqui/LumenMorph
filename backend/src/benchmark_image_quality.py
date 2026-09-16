"""Independent sampled physical audits of sealed benchmark images and dense truth.

Does not rerender images, reconstruct cameras, or run a deformation estimator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from .benchmark_quality import _sha, _write
from .benchmark_validation import _camera, _directions, _project, _ray_hits, audit_raster_samples


def audit_flow_samples(source_truth, target_vertices, triangles, R_wc, t_wc, intr,
                       saved_flow, *, sample_count=32, seed=0, pixel_tolerance=1e-3,
                       depth_atol=1e-5, depth_rtol=1e-5):
    """Independent material advection and physical occlusion at actual endpoints."""
    valid = np.asarray(source_truth["valid"])
    h, w = valid.shape
    keys = ("visible", "occluded", "out_of_frame", "behind_camera", "unresolved")
    mask_partition_passed = bool(np.array_equal(sum(np.asarray(saved_flow[k], int) for k in keys), valid.astype(int)))
    partition = np.asarray(saved_flow["visible"]) | np.asarray(saved_flow["occluded"]) | np.asarray(saved_flow["out_of_frame"]) | np.asarray(saved_flow["unresolved"])
    mask_partition_passed &= bool(np.array_equal(partition, saved_flow["flow_valid"]))
    rng = np.random.default_rng(seed)
    flat = rng.choice(h*w, min(sample_count, h*w), replace=False)
    rows, cols = np.unravel_index(flat, (h, w))
    selected_valid = valid[rows, cols]
    selected = np.flatnonzero(selected_valid)
    tri_ids = source_truth["triangle_id"][rows[selected], cols[selected]]
    bary = np.asarray(source_truth["material_barycentric"])[rows[selected], cols[selected]].astype(float)
    points = np.einsum("qi,qij->qj", bary, np.asarray(target_vertices, float)[np.asarray(triangles)[tri_ids]])
    R, t, inverse_Rt = _camera(R_wc, t_wc)
    projected, z = _project(points, R, t, intr)
    front = np.isfinite(projected).all(axis=1) & np.isfinite(z) & (z > 1e-6)
    inside = front & (projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h)
    first = np.full(len(points), np.inf)
    if inside.any():
        _, first[inside], _ = _ray_hits(np.asarray(target_vertices, float), np.asarray(triangles), t, _directions(projected[inside], intr, inverse_Rt))
    tol = depth_atol + depth_rtol*np.abs(z)
    visible = inside & np.isfinite(first) & (np.abs(first-z) <= tol)
    occluded = inside & np.isfinite(first) & (z > first+tol)
    expected = {"flow_valid": front, "visible": visible, "occluded": occluded,
                "out_of_frame": front & ~inside, "behind_camera": ~front,
                "unresolved": inside & ~visible & ~occluded}
    failures = []
    max_error = 0.
    for j, i in enumerate(selected):
        row, col = rows[i], cols[i]
        reasons = []
        for key, values in expected.items():
            if bool(saved_flow[key][row, col]) != bool(values[j]):
                reasons.append(key+"_mismatch")
        if front[j]:
            predicted = saved_flow["flow_uv"][row, col]
            actual = projected[j] - [col+.5, row+.5]
            error = float(np.linalg.norm(predicted-actual))
            max_error = max(max_error, error)
            if not np.isfinite(predicted).all() or error >= pixel_tolerance:
                reasons.append("material_flow_reprojection")
        elif np.isfinite(saved_flow["flow_uv"][row, col]).any():
            reasons.append("behind_camera_finite_flow")
        if reasons:
            failures.append({"row": int(row), "column": int(col), "reasons": reasons})
    for i in np.flatnonzero(~selected_valid):
        row, col = rows[i], cols[i]
        if any(saved_flow[k][row, col] for k in (*keys, "flow_valid")) or np.isfinite(saved_flow["flow_uv"][row, col]).any():
            failures.append({"row": int(row), "column": int(col), "reasons": ["invalid_source_has_flow"]})
    return {"passed": mask_partition_passed and not failures, "all_pixel_mask_partition_passed": mask_partition_passed,
            "sample_count": len(flat), "source_foreground_count": len(selected), "seed": int(seed),
            "max_flow_error_px": max_error, "failures": failures,
            "pixel_tolerance": pixel_tolerance, "depth_atol": depth_atol, "depth_rtol": depth_rtol}


def _verify_seal(path):
    complete = path/"COMPLETE.json"
    record = json.loads(complete.read_text())
    for name, expected in record["files"].items():
        if _sha(path/name) != expected:
            raise ValueError(f"seal mismatch: {path/name}")
    return str(complete.resolve()), _sha(complete)


def images(root: Path, family_id: str, condition: str, stream: str, run_name="images_r0"):
    spec = json.loads((root/"SPECIFICATION.json").read_text())
    if _sha(root/"SPECIFICATION.json") != json.loads((root/"FROZEN.json").read_text())["spec_sha256"]:
        raise ValueError("specification changed")
    family = next(f for f in spec["families"] if f["id"] == family_id)
    common = root/"families"/family_id
    if stream == "reference":
        public, truth = common/"public/reference", common/"reference_truth"
        condition = "static"
    else:
        version = 3+spec["families"].index(family)*3+spec["conditions"].index(condition)
        public, truth = root/f"v{version}"/"public"/stream, root/f"v{version}"/"evaluation"/stream
    out = common/"quality"/run_name/(condition+"_"+stream)
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("benchmark_validation.py"), Path(__file__).with_name("benchmark_quality.py")]
    input_seals = dict(_verify_seal(p) for p in (common/"evaluation", public, truth))
    _write(out/"INPUT_SOURCE_FREEZE.json", {"sources": {str(p.resolve()): _sha(p) for p in sources}, "input_seals": input_seals,
           "spec_sha256": _sha(root/"SPECIFICATION.json"), "command": sys.argv, "python": sys.executable,
           "started_unix": time.time(), "raster_samples_per_frame": 128, "flow_samples_per_pair": 32,
           "inference_performance_inspected": False})
    with np.load(common/"evaluation/geometry.npz", allow_pickle=False) as data:
        geom = dict(data)
    with np.load(truth/"states.npz", allow_pickle=False) as data:
        states = dict(data)
    intr = json.loads((common/"public/intrinsics.json").read_text())
    n = len(states["vertices"])
    summaries = []
    seed = int(family["generator_config"]["seed"])
    start = time.monotonic()
    for index in range(n):
        with np.load(truth/"dense"/f"{index:06d}.npz", allow_pickle=False) as data:
            dense = dict(data)
        image = cv2.imread(str(public/"rgb"/f"{index:06d}.png"))
        if image is None or image.shape != (intr["height"], intr["width"], 3):
            raise ValueError(f"invalid RGB frame {public} {index}")
        valid = dense["valid"]
        labels_ok = bool(np.array_equal(valid, dense["triangle_id"] >= 0)
                         and np.isfinite(dense["depth_z"][valid]).all()
                         and np.isfinite(dense["material_barycentric"][valid]).all()
                         and np.isfinite(dense["normal_world"][valid]).all()
                         and not np.isfinite(dense["depth_z"][~valid]).any())
        raster = audit_raster_samples(states["vertices"][index], geom["triangles"], states["R_wc"][index], states["C_world"][index], intr,
                  dense["triangle_id"], dense["material_barycentric"], dense["depth_z"], sample_count=128, seed=seed+index)
        rng = np.random.default_rng(seed+index)
        flat = rng.choice(valid.size, min(128, valid.size), replace=False)
        rows, cols = np.unravel_index(flat, valid.shape)
        foreground = valid[rows, cols]
        ids = dense["triangle_id"][rows[foreground], cols[foreground]]
        triangle_points = states["vertices"][index][geom["triangles"][ids]].astype(float)
        normal = np.cross(triangle_points[:, 1]-triangle_points[:, 0], triangle_points[:, 2]-triangle_points[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
        saved = dense["normal_world"][rows[foreground], cols[foreground]]
        normal_error = float(np.linalg.norm(normal-saved, axis=1).max()) if len(saved) else 0.
        flow = None
        if index+1 < n:
            with np.load(truth/"flow_forward"/f"{index:06d}.npz", allow_pickle=False) as data:
                saved_flow = dict(data)
            flow = audit_flow_samples(dense, states["vertices"][index+1], geom["triangles"], states["R_wc"][index+1], states["C_world"][index+1], intr, saved_flow, sample_count=32, seed=seed+index)
        report = {"frame": index, "passed": labels_ok and raster["passed"] and normal_error < 1e-5 and (flow is None or flow["passed"]),
                  "dense_finite_and_mask_passed": labels_ok, "raster": raster, "normal_max_error": normal_error,
                  "flow": flow, "rgb_channel_mean": image.mean(axis=(0, 1)).tolist(), "rgb_standard_deviation": float(image.std())}
        _write(out/f"{index:06d}.json", report)
        summaries.append({"frame": index, "passed": report["passed"], "max_pixel_error": raster["max_reprojection_error_px"],
                          "max_relative_depth_error": raster["max_relative_depth_error"], "max_flow_error_px": flow["max_flow_error_px"] if flow else None,
                          "raster_failure_count": len(raster["failures"]), "flow_failure_count": len(flow["failures"]) if flow else 0})
        if index % 15 == 0:
            print(json.dumps({"family": family_id, "condition": condition, "stream": stream, "audit_frame": index, "passed": report["passed"], "elapsed_s": time.monotonic()-start}), flush=True)
    result = {"family": family_id, "condition": condition, "stream": stream, "passed": all(r["passed"] for r in summaries),
              "frames": summaries, "frame_count": n, "raster_sample_count": n*128, "flow_sample_count": (n-1)*32,
              "scope": "sampled independent physical label/flow/normal checks; RGB decoding and photometric summaries; no estimator outcomes or exhaustive ray audit"}
    _write(out/"RESULT.json", result)
    _write(out/"COMPLETE.json", {"files": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()}, "completed_unix": time.time()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition", choices=("static", "fem", "nonfem"), default="static")
    parser.add_argument("--stream", choices=("reference", "moving", "stationary"), required=True)
    parser.add_argument("--run-name", default="images_r0")
    args = parser.parse_args()
    result = images(args.root, args.family, args.condition, args.stream, args.run_name)
    print(json.dumps({"family": args.family, "condition": args.condition, "stream": args.stream, "passed": result["passed"]}), flush=True)


if __name__ == "__main__":
    main()
