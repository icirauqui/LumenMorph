from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .texture import orb_feature_count
from .types import ValidationResult

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("OpenCV (cv2) is required for validation") from exc


REQUIRED_SEQUENCE_FILES = [
    "intrinsics.json",
    "poses.csv",
    "deformation_params.csv",
    "mesh_static.npz",
    "sequence_meta.json",
]


POSE_COLUMNS = ["frame", "timestamp_s", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
DEFORM_COLUMNS = ["frame", "timestamp_s", "preset", "global_amp", "global_freq", "phase_shift"]


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        return fieldnames, list(reader)


def _parse_float(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def _load_config_used(dataset_root: Path) -> dict[str, Any]:
    cfg_path = dataset_root / "config_used.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text())


def _load_validation_limits(cfg_used: dict[str, Any]) -> dict[str, float]:
    val = cfg_used.get("validation", {})
    return {
        "min_orb_features_mean": float(val.get("min_orb_features_mean", 100.0)),
        "min_orb_features_frame": float(val.get("min_orb_features_frame", 1.0)),
        "min_frame_motion": float(val.get("min_frame_motion", 1e-3)),
        "min_camera_clearance": float(val.get("min_camera_clearance", 0.1)),
        "max_abs_mesh_z": float(val.get("max_abs_mesh_z", 100.0)),
    }


def _sequence_frame_paths(seq_dir: Path) -> list[Path]:
    frames_dir = seq_dir / "frames"
    if not frames_dir.exists():
        return []
    return sorted(frames_dir.glob("*.png"))


def _sequence_scene_type(seq_dir: Path, default_scene_type: str) -> str:
    meta_path = seq_dir / "sequence_meta.json"
    if not meta_path.exists():
        return default_scene_type
    try:
        data = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return default_scene_type
    scene = str(data.get("scene_type", default_scene_type)).strip().lower()
    return scene if scene in {"plane", "tube"} else default_scene_type


def _validate_plane_clearance(
    mesh_paths: list[Path],
    tz: np.ndarray,
    limits: dict[str, float],
    seq_dir: Path,
    errors: list[str],
    min_clearance: float,
) -> float:
    # Count mismatches are already reported by validate_dataset. Limit this
    # deeper check to paired samples so malformed/stale datasets produce a
    # validation result rather than an IndexError.
    paired_count = min(len(mesh_paths), int(tz.shape[0]))
    sample_count = min(paired_count, 20)
    if sample_count == 0:
        return min_clearance
    sample_idx = np.linspace(0, paired_count - 1, sample_count, dtype=np.int32)

    for idx in sample_idx:
        mesh = np.load(mesh_paths[idx])
        if not np.isfinite(mesh).all():
            errors.append(f"Non-finite mesh values in {mesh_paths[idx]}")
            continue

        max_abs = float(np.max(np.abs(mesh)))
        if max_abs > limits["max_abs_mesh_z"]:
            errors.append(f"Mesh z exceeds bounds in {mesh_paths[idx]}: {max_abs:.3f} > {limits['max_abs_mesh_z']:.3f}")

        clearance = float(tz[idx] - np.max(mesh))
        min_clearance = min(min_clearance, clearance)
        if clearance < limits["min_camera_clearance"]:
            errors.append(
                f"Camera clearance too small in {seq_dir} frame {idx}: {clearance:.4f} < {limits['min_camera_clearance']:.4f}"
            )
    return min_clearance


def _validate_tube_clearance(
    mesh_paths: list[Path],
    pose_vec: np.ndarray,
    mesh_static_path: Path,
    limits: dict[str, float],
    seq_dir: Path,
    errors: list[str],
    min_clearance: float,
) -> float:
    try:
        static = np.load(mesh_static_path)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"Failed to read mesh_static for tube scene {mesh_static_path}: {exc}")
        return min_clearance

    required = ["centerline", "frame_right", "frame_up", "theta"]
    for k in required:
        if k not in static:
            errors.append(f"tube mesh_static missing key '{k}' in {mesh_static_path}")
            return min_clearance

    centerline = static["centerline"]
    frame_right = static["frame_right"]
    frame_up = static["frame_up"]
    theta = static["theta"]
    base_radius = static["base_radius"] if "base_radius" in static else None

    # Count mismatches are already reported by validate_dataset. Limit this
    # deeper check to paired samples so malformed/stale datasets produce a
    # validation result rather than an IndexError.
    paired_count = min(len(mesh_paths), int(pose_vec.shape[0]))
    sample_count = min(paired_count, 20)
    if sample_count == 0:
        return min_clearance
    sample_idx = np.linspace(0, paired_count - 1, sample_count, dtype=np.int32)

    n_long = int(centerline.shape[0])
    if n_long <= 0:
        errors.append(f"Invalid centerline in {mesh_static_path}")
        return min_clearance

    for idx in sample_idx:
        mesh = np.load(mesh_paths[idx])
        if not np.isfinite(mesh).all():
            errors.append(f"Non-finite mesh values in {mesh_paths[idx]}")
            continue

        max_abs = float(np.max(np.abs(mesh)))
        if max_abs > limits["max_abs_mesh_z"]:
            errors.append(f"Mesh z exceeds bounds in {mesh_paths[idx]}: {max_abs:.3f} > {limits['max_abs_mesh_z']:.3f}")

        cam = pose_vec[idx]
        # Camera progress need not be linear in frame index (the tube camera
        # reserves room for its look-ahead target). Select the geometrically
        # local cross-section instead of assuming raw time equals arc progress.
        ring_idx = int(np.argmin(np.linalg.norm(centerline - cam[None, :], axis=1)))

        center = centerline[ring_idx]
        right = frame_right[ring_idx]
        up = frame_up[ring_idx]

        off = cam - center
        radial = float(np.sqrt(np.dot(off, right) ** 2 + np.dot(off, up) ** 2))

        if mesh.ndim == 3 and mesh.shape[-1] == 3:
            if base_radius is None:
                errors.append(f"tube mesh_static missing key 'base_radius' for displacement mesh state in {mesh_static_path}")
                continue
            if ring_idx >= mesh.shape[0]:
                errors.append(f"Unexpected tube displacement frame shape in {mesh_paths[idx]}: {mesh.shape}")
                continue
            radial_dirs = (
                np.cos(theta)[:, None] * right[None, :]
                + np.sin(theta)[:, None] * up[None, :]
            )
            radial_disp = np.sum(mesh[ring_idx] * radial_dirs, axis=1)
            local_min_radius = float(np.min(base_radius[ring_idx] + radial_disp))
        elif mesh.ndim == 2 and ring_idx < mesh.shape[0]:
            local_min_radius = float(np.min(mesh[ring_idx]))
        else:
            errors.append(f"Unexpected tube mesh frame shape in {mesh_paths[idx]}: {mesh.shape}")
            continue
        clearance = local_min_radius - radial
        min_clearance = min(min_clearance, clearance)

        if clearance < limits["min_camera_clearance"]:
            errors.append(
                f"Tube camera clearance too small in {seq_dir} frame {idx}: {clearance:.4f} < {limits['min_camera_clearance']:.4f}"
            )

    return min_clearance


def validate_dataset(dataset_root: str | Path) -> ValidationResult:
    dataset_root = Path(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {
        "sequence_count": 0,
        "avg_orb_features": 0.0,
        "min_orb_features": 0.0,
        "p05_orb_features": 0.0,
        "avg_motion_norm": 0.0,
        "min_clearance": float("inf"),
        "checked_mp4": False,
    }

    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.exists():
        return ValidationResult(False, [f"Missing manifest: {manifest_path}"], warnings, stats)

    manifest = json.loads(manifest_path.read_text())
    sequences = manifest.get("sequences", [])
    if not sequences:
        return ValidationResult(False, ["Manifest has no sequences"], warnings, stats)

    cfg_used = _load_config_used(dataset_root)
    limits = _load_validation_limits(cfg_used)
    default_scene_type = str(cfg_used.get("scene_type", manifest.get("scene_type", "plane"))).strip().lower()
    if default_scene_type not in {"plane", "tube"}:
        default_scene_type = "plane"

    all_orb_counts: list[float] = []
    all_motion: list[float] = []
    min_clearance = float("inf")

    for seq in sequences:
        rel = seq.get("relative_path")
        if rel is None:
            errors.append("Manifest entry missing relative_path")
            continue

        seq_dir = dataset_root / rel
        if not seq_dir.exists():
            errors.append(f"Sequence folder missing: {seq_dir}")
            continue

        scene_type = _sequence_scene_type(seq_dir, default_scene_type)
        stats["sequence_count"] += 1

        for name in REQUIRED_SEQUENCE_FILES:
            p = seq_dir / name
            if not p.exists():
                errors.append(f"Missing sequence file: {p}")

        frame_paths = _sequence_frame_paths(seq_dir)
        mesh_paths = sorted((seq_dir / "mesh_z").glob("*.npy")) if (seq_dir / "mesh_z").exists() else []

        poses_path = seq_dir / "poses.csv"
        deform_path = seq_dir / "deformation_params.csv"

        if poses_path.exists():
            pose_cols, pose_rows = _read_csv_rows(poses_path)
            if pose_cols != POSE_COLUMNS:
                errors.append(f"Unexpected pose columns in {poses_path}: {pose_cols}")
                pose_rows = []
        else:
            pose_rows = []

        if deform_path.exists():
            deform_cols, deform_rows = _read_csv_rows(deform_path)
            if deform_cols != DEFORM_COLUMNS:
                errors.append(f"Unexpected deformation columns in {deform_path}: {deform_cols}")
                deform_rows = []
        else:
            deform_rows = []

        if pose_rows and deform_rows and len(pose_rows) != len(deform_rows):
            errors.append(f"Row mismatch between poses/deformation in {seq_dir}")

        if frame_paths and pose_rows and len(frame_paths) != len(pose_rows):
            errors.append(f"Frame count mismatch in {seq_dir}: {len(frame_paths)} frames vs {len(pose_rows)} poses")

        if mesh_paths and pose_rows and len(mesh_paths) != len(pose_rows):
            errors.append(f"Mesh frame count mismatch in {seq_dir}")

        pose_vec = np.empty((0, 3), dtype=np.float64)
        if pose_rows:
            tx = _parse_float(pose_rows, "tx")
            ty = _parse_float(pose_rows, "ty")
            tz = _parse_float(pose_rows, "tz")
            pose_vec = np.stack([tx, ty, tz], axis=1)
            if not np.isfinite(pose_vec).all():
                errors.append(f"Non-finite pose values in {poses_path}")

            if pose_vec.shape[0] > 1:
                motion = np.linalg.norm(np.diff(pose_vec, axis=0), axis=1)
                all_motion.extend(motion.tolist())

        if deform_rows:
            amp = _parse_float(deform_rows, "global_amp")
            frq = _parse_float(deform_rows, "global_freq")
            phs = _parse_float(deform_rows, "phase_shift")
            vals = np.stack([amp, frq, phs], axis=1)
            if not np.isfinite(vals).all():
                errors.append(f"Non-finite deformation values in {deform_path}")

        sample_count = min(len(frame_paths), 30)
        if sample_count > 0:
            sample_idx = np.linspace(0, len(frame_paths) - 1, sample_count, dtype=np.int32)
            for idx in sample_idx:
                frame = cv2.imread(str(frame_paths[idx]))
                if frame is None:
                    errors.append(f"Unreadable frame: {frame_paths[idx]}")
                    continue
                all_orb_counts.append(float(orb_feature_count(frame)))

        if mesh_paths and pose_rows:
            if scene_type == "tube":
                min_clearance = _validate_tube_clearance(
                    mesh_paths=mesh_paths,
                    pose_vec=pose_vec,
                    mesh_static_path=seq_dir / "mesh_static.npz",
                    limits=limits,
                    seq_dir=seq_dir,
                    errors=errors,
                    min_clearance=min_clearance,
                )
            else:
                tz = pose_vec[:, 2]
                min_clearance = _validate_plane_clearance(
                    mesh_paths=mesh_paths,
                    tz=tz,
                    limits=limits,
                    seq_dir=seq_dir,
                    errors=errors,
                    min_clearance=min_clearance,
                )

    if all_orb_counts:
        mean_orb = float(np.mean(all_orb_counts))
        min_orb = float(np.min(all_orb_counts))
        p05_orb = float(np.percentile(all_orb_counts, 5.0))
        stats["avg_orb_features"] = mean_orb
        stats["min_orb_features"] = min_orb
        stats["p05_orb_features"] = p05_orb
        if mean_orb < limits["min_orb_features_mean"]:
            errors.append(f"Average ORB feature count too low: {mean_orb:.2f} < {limits['min_orb_features_mean']:.2f}")
        if min_orb < limits["min_orb_features_frame"]:
            errors.append(f"Minimum sampled-frame ORB feature count too low: {min_orb:.2f} < {limits['min_orb_features_frame']:.2f}")
    else:
        warnings.append("No frames found for ORB feature checks")

    if all_motion:
        mean_motion = float(np.mean(all_motion))
        stats["avg_motion_norm"] = mean_motion
        if mean_motion < limits["min_frame_motion"]:
            errors.append(f"Average frame-to-frame motion too small: {mean_motion:.6f} < {limits['min_frame_motion']:.6f}")
    else:
        warnings.append("No motion data found")

    if min_clearance < float("inf"):
        stats["min_clearance"] = min_clearance

    for seq in sequences:
        rel = seq.get("relative_path")
        if not rel:
            continue
        mp4 = dataset_root / rel / "preview.mp4"
        if not mp4.exists():
            continue
        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            errors.append(f"Failed to open MP4: {mp4}")
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                errors.append(f"Failed to read first frame from MP4: {mp4}")
            stats["checked_mp4"] = True
        cap.release()
        break

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, stats=stats)
