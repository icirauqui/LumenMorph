"""Reproducible independent QA entry points for frozen tubular benchmark assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .benchmark_validation import audit_bridge_signal, audit_c3d6_mechanics, audit_mesh_fields


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _frozen_inputs(root, family):
    private = root / "families" / family / "evaluation"
    complete = json.loads((private/"COMPLETE.json").read_text())
    for name, expected in complete["files"].items():
        if _sha(private/name) != expected:
            raise ValueError(f"preparation seal mismatch: {private/name}")
    paths = [root/"SPECIFICATION.json", root/"FROZEN.json", private/"COMPLETE.json",
             private/"geometry.npz", private/"fields.npz", private/"mechanics.npz", private/"poses.npz",
             root/"families"/family/"public/intrinsics.json"]
    return {str(p.resolve()): _sha(p) for p in paths}


def geometry(root: Path, family_id: str, run_name="geometry_r0"):
    """No renderer or inverse estimator is called; all outputs are private QA."""
    spec = json.loads((root/"SPECIFICATION.json").read_text())
    if _sha(root/"SPECIFICATION.json") != json.loads((root/"FROZEN.json").read_text())["spec_sha256"]:
        raise ValueError("specification freeze mismatch")
    family = next(f for f in spec["families"] if f["id"] == family_id)
    out = root/"families"/family_id/"quality"/run_name
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("benchmark_validation.py")]
    freeze = {"sources": {str(p.resolve()): _sha(p) for p in sources},
              "inputs": _frozen_inputs(root, family_id), "command": sys.argv,
              "python": sys.executable, "numpy_version": np.__version__,
              "started_unix": time.time(), "geometry_only": True, "inference_performance_inspected": False}
    _write(out/"INPUT_SOURCE_FREEZE.json", freeze)
    private = root/"families"/family_id/"evaluation"
    with np.load(private/"geometry.npz", allow_pickle=False) as data:
        geom = dict(data)
    with np.load(private/"fields.npz", allow_pickle=False) as data:
        fields = dict(data)
    with np.load(private/"poses.npz", allow_pickle=False) as data:
        poses = dict(data)
    with np.load(private/"mechanics.npz", allow_pickle=False) as data:
        mechanics = dict(data)
    intr = json.loads((root/"families"/family_id/"public/intrinsics.json").read_text())
    coefficients = fields["time_coefficients"]
    fea = audit_c3d6_mechanics(mechanics, coefficients=coefficients)
    _write(out/"MECHANICS_AUDIT.json", fea)
    base, triangles = geom["vertices"], geom["triangles"]
    radius = float(np.median(geom["base_radius"]))
    seed = int(family["generator_config"]["seed"])+77
    criteria = spec["signal_design"]
    reports = {}
    for condition in spec["conditions"]:
        if condition == "static":
            states = np.repeat(base[None], len(coefficients), axis=0)
        else:
            displacement = np.einsum("tk,kijc->tijc", coefficients, fields[condition+"_basis"]).reshape(len(coefficients), -1, 3).astype(np.float32)
            states = (base[None]+displacement).astype(np.float32)
        field_report = audit_mesh_fields(base, triangles, states, radius)
        _write(out/f"{condition}_FIELD_AUDIT.json", field_report)
        condition_report = {"field_geometry_passed": field_report["passed"], "contrasts": {},
                            "max_vertex_displacement_over_radius": max(f["peak_displacement_over_radius"] for f in field_report["frames"]),
                            "max_absolute_edge_length_change": max(f["max_absolute_edge_length_change"] for f in field_report["frames"])}
        for name, groups in spec["contrasts"].items():
            signal = audit_bridge_signal(base, triangles, states, poses["reference_R"][-1], poses["reference_C"][-1], intr,
                groups["a"], groups["b"], radius=radius, seed=seed, sample_count=2048,
                nominal_mean_floor=criteria["nominal_mean_over_radius_min"],
                nominal_p95_floor=criteria["nominal_p95_over_radius_min"],
                visibility_floor=criteria["visibility_fraction_min"])
            _write(out/f"{condition}_{name}_SIGNAL_AUDIT.json", signal)
            condition_report["contrasts"][name] = {k: v for k, v in signal.items() if k not in ("cohort", "visibility_by_frame")}
            print(json.dumps({"family": family_id, "condition": condition, "contrast": name,
                              "mean_over_radius": signal["mean_target_over_radius"], "p95_over_radius": signal["p95_target_over_radius"],
                              "visible_fraction": signal["common_visible_fraction"], "nominal_signal_qualified": signal["nominal_signal_qualified"]}), flush=True)
        reports[condition] = condition_report
    max_engineering = max(max(s["maximum_absolute_infinitesimal_tensor_strain"], s["maximum_absolute_engineering_shear_strain"]) for s in fea["states"])
    result = {"family": family_id, "role": family["split"], "radius": radius,
              "mechanics_arithmetic_passed": fea["passed"], "maximum_absolute_engineering_strain": max_engineering,
              "small_strain_screen_0_10": max_engineering <= .10,
              "small_strain_screen_scope": "engineering screen only; not a constitutive-validation claim",
              "minimum_deformation_jacobian_at_gauss_points": min(s["minimum_deformation_jacobian_at_gauss_points"] for s in fea["states"]),
              "conditions": reports, "static_exact_zero": reports["static"]["max_vertex_displacement_over_radius"] == 0,
              "all_dynamic_nominal_signal_qualified": all(reports[c]["contrasts"]["nominal"]["nominal_signal_qualified"] for c in spec["conditions"] if c != "static"),
              "all_surface_geometry_passed": all(r["field_geometry_passed"] for r in reports.values()),
              "inference_performance_inspected": False,
              "scope": "geometry-only generation QA; low-signal/static contrasts are not required to meet nominal motion thresholds; no reconstruction/recovery performance"}
    _write(out/"RESULT.json", result)
    _write(out/"COMPLETE.json", {"files": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()}, "completed_unix": time.time()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    geometric = sub.add_parser("geometry")
    geometric.add_argument("--root", type=Path, required=True)
    geometric.add_argument("--family")
    geometric.add_argument("--run-name", default="geometry_r0")
    args = parser.parse_args()
    spec = json.loads((args.root/"SPECIFICATION.json").read_text())
    families = [args.family] if args.family else [f["id"] for f in spec["families"]]
    for family in families:
        result = geometry(args.root, family, args.run_name)
        print(json.dumps({"family": family, "mechanics_passed": result["mechanics_arithmetic_passed"],
                          "dynamic_nominal_signal_qualified": result["all_dynamic_nominal_signal_qualified"],
                          "maximum_engineering_strain": result["maximum_absolute_engineering_strain"]}), flush=True)


if __name__ == "__main__":
    main()
