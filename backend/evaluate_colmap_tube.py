#!/usr/bin/env python3
"""Evaluate an estimated COLMAP model against a generated tube sequence.

The model must be in COLMAP text format. Camera centres determine the Sim(3)
alignment, so the reported surface distances remain independent of known poses
during reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import cKDTree


def _rotation_from_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_images(path: Path) -> dict[int, dict]:
    images: dict[int, dict] = {}
    with path.open(encoding="utf-8") as stream:
        lines = iter(stream)
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            image_id = int(fields[0])
            qvec = np.asarray(fields[1:5], dtype=np.float64)
            tvec = np.asarray(fields[5:8], dtype=np.float64)
            rotation_cw = _rotation_from_quaternion_wxyz(qvec)
            images[image_id] = {
                "name": fields[9],
                "center": -(rotation_cw.T @ tvec),
                "rotation_wc": rotation_cw.T,
            }
            next(lines, None)  # POINTS2D line.
    return images


def _read_points(path: Path) -> list[dict]:
    points: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            points.append(
                {
                    "xyz": np.asarray(fields[1:4], dtype=np.float64),
                    "error_px": float(fields[7]),
                    "image_ids": np.asarray(fields[8::2], dtype=np.int64),
                }
            )
    return points


def _read_ground_truth_poses(path: Path) -> dict[str, dict]:
    poses: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = f"{int(row['frame']):06d}.png"
            q_xyzw = np.array(
                [row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64
            )
            poses[name] = {
                "center": np.array([row["tx"], row["ty"], row["tz"]], dtype=np.float64),
                "rotation_wc": _rotation_from_quaternion_wxyz(q_xyzw[[3, 0, 1, 2]]),
                "frame": int(row["frame"]),
            }
    return poses


def _fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    variance = np.mean(np.sum(source_zero * source_zero, axis=1))
    scale = float(np.sum(singular * sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _fit_orientation_similarity(
    source: np.ndarray,
    target: np.ndarray,
    source_rotations: np.ndarray,
    target_rotations: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit scale/translation after fixing rotation from camera orientations.

    A tube trajectory can be nearly collinear, leaving roll about its axis
    weakly constrained by camera centres alone. Averaging the measured camera
    orientation mappings removes that gauge ambiguity without using poses in
    the COLMAP reconstruction itself.
    """
    aggregate = np.zeros((3, 3), dtype=np.float64)
    for source_rotation, target_rotation in zip(source_rotations, target_rotations):
        aggregate += target_rotation @ source_rotation.T
    u, _, vt = np.linalg.svd(aggregate)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    rotated_source = (rotation @ source_zero.T).T
    denominator = float(np.sum(rotated_source * rotated_source))
    scale = float(np.sum(target_zero * rotated_source) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Orientation-constrained similarity produced an invalid scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _select_similarity(
    source: np.ndarray,
    target: np.ndarray,
    source_rotations: np.ndarray,
    target_rotations: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, str, np.ndarray]:
    """Choose a well-constrained Sim(3) without biasing curved trajectories.

    Camera centres are the primary alignment signal. Collinear and planar
    windows leave at least one rotational/reflection degree of freedom weakly
    constrained; in that case camera orientations supply the missing gauge
    constraint. Applying the orientation constraint to a genuinely 3-D curved
    trajectory can hide or amplify rotational SfM drift, so those trajectories
    use the position-only optimum.
    """
    singular = np.linalg.svd(source - source.mean(axis=0), compute_uv=False)
    ratios = singular / max(float(singular[0]), 1e-12)
    if float(ratios[2]) < 0.02:
        scale, rotation, translation = _fit_orientation_similarity(
            source, target, source_rotations, target_rotations
        )
        source_name = "camera_orientations_for_rank_deficient_position_gauge"
    else:
        scale, rotation, translation = _fit_similarity(source, target)
        source_name = "registered_camera_centres"
    return scale, rotation, translation, source_name, ratios


def _tube_vertices(mesh: np.lib.npyio.NpzFile, displacement: np.ndarray) -> np.ndarray:
    theta = mesh["theta"].astype(np.float64)
    radius = mesh["base_radius"].astype(np.float64)
    centerline = mesh["centerline"].astype(np.float64)
    right = mesh["frame_right"].astype(np.float64)
    up = mesh["frame_up"].astype(np.float64)
    vertices = (
        centerline[:, None, :]
        + radius[:, :, None] * np.cos(theta)[None, :, None] * right[:, None, :]
        + radius[:, :, None] * np.sin(theta)[None, :, None] * up[:, None, :]
    )
    return vertices + displacement.astype(np.float64)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else math.nan


def evaluate(sequence: Path, model: Path) -> dict:
    images = _read_images(model / "images.txt")
    points = _read_points(model / "points3D.txt")
    truth = _read_ground_truth_poses(sequence / "poses.csv")
    matched = [(image_id, image, truth[image["name"]]) for image_id, image in images.items() if image["name"] in truth]
    if len(matched) < 3:
        raise RuntimeError("At least three registered images with ground-truth poses are required")

    estimated_centres = np.stack([item[1]["center"] for item in matched])
    truth_centres = np.stack([item[2]["center"] for item in matched])
    estimated_rotations = np.stack([item[1]["rotation_wc"] for item in matched])
    truth_rotations = np.stack([item[2]["rotation_wc"] for item in matched])
    position_scale, position_rotation, position_translation = _fit_similarity(
        estimated_centres, truth_centres
    )
    position_aligned_centres = (
        position_scale * (position_rotation @ estimated_centres.T)
    ).T + position_translation
    position_centre_errors = np.linalg.norm(position_aligned_centres - truth_centres, axis=1)
    scale, rotation, translation, sim3_source, centre_spectrum_ratios = _select_similarity(
        estimated_centres,
        truth_centres,
        estimated_rotations,
        truth_rotations,
    )
    aligned_centres = (scale * (rotation @ estimated_centres.T)).T + translation
    centre_errors = np.linalg.norm(aligned_centres - truth_centres, axis=1)

    rotation_errors = []
    for _, estimated, gt in matched:
        aligned_rotation = rotation @ estimated["rotation_wc"]
        delta = gt["rotation_wc"].T @ aligned_rotation
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        rotation_errors.append(math.degrees(math.acos(cosine)))
    rotation_errors = np.asarray(rotation_errors)

    mesh = np.load(sequence / "mesh_static.npz")
    n_long = int(mesh["centerline"].shape[0])
    n_radial = int(mesh["theta"].shape[0])
    image_frames = {image_id: truth[image["name"]]["frame"] for image_id, image in images.items() if image["name"] in truth}
    point_xyz = np.stack([point["xyz"] for point in points]) if points else np.empty((0, 3))
    aligned_xyz = (scale * (rotation @ point_xyz.T)).T + translation if points else point_xyz

    by_frame: dict[int, list[int]] = {}
    for point_index, point in enumerate(points):
        frames = sorted(image_frames[image_id] for image_id in point["image_ids"] if image_id in image_frames)
        if frames:
            by_frame.setdefault(frames[len(frames) // 2], []).append(point_index)

    registered_frames = sorted(item[2]["frame"] for item in matched)
    registered_first = registered_frames[0]
    registered_last = registered_frames[-1]
    registered_span = registered_last - registered_first + 1
    temporal_bins = min(30, registered_span)
    temporal_occupancy = np.zeros(temporal_bins, dtype=bool)
    for frame, indices in by_frame.items():
        if indices and registered_first <= frame <= registered_last:
            bin_index = min(
                ((frame - registered_first) * temporal_bins) // registered_span,
                temporal_bins - 1,
            )
            temporal_occupancy[bin_index] = True

    distances = np.full(len(points), np.nan, dtype=np.float64)
    nearest_vertices = np.full(len(points), -1, dtype=np.int64)
    for frame, indices in by_frame.items():
        displacement = np.load(sequence / "mesh_z" / f"{frame:06d}.npy")
        vertices = _tube_vertices(mesh, displacement).reshape(-1, 3)
        frame_distances, frame_nearest = cKDTree(vertices).query(aligned_xyz[indices], workers=-1)
        distances[indices] = frame_distances
        nearest_vertices[indices] = frame_nearest

    valid = np.isfinite(distances)
    surface_distances = distances[valid]
    tube_radius = float(np.median(mesh["base_radius"]))
    inlier_limit = 0.25 * tube_radius
    surface_inliers = valid & (distances <= inlier_limit)

    centreline = mesh["centerline"].astype(np.float64)
    gt_camera_rings = cKDTree(centreline).query(truth_centres, workers=-1)[1]
    with (sequence / "sequence_meta.json").open(encoding="utf-8") as stream:
        sequence_meta = json.load(stream)
    with (sequence / "intrinsics.json").open(encoding="utf-8") as stream:
        intrinsics = json.load(stream)
    spacing = float(np.median(np.linalg.norm(np.diff(centreline, axis=0), axis=1)))
    horizontal_half_fov = math.atan(
        float(intrinsics["width"]) / (2.0 * float(intrinsics["fx"]))
    )
    vertical_half_fov = math.atan(
        float(intrinsics["height"]) / (2.0 * float(intrinsics["fy"]))
    )
    # Approximate where the widest image ray first intersects a cylindrical
    # wall. Shifting the traversed camera interval by this depth is more honest
    # than assuming the wall beside the camera or the whole look-ahead region
    # is visible.
    wall_depth = tube_radius / math.tan(max(horizontal_half_fov, vertical_half_fov))
    wall_shift_rings = int(round(wall_depth / spacing))
    visible_first = min(n_long - 1, int(gt_camera_rings.min()) + wall_shift_rings)
    visible_last = min(n_long - 1, int(gt_camera_rings.max()) + wall_shift_rings)

    longitudinal_bins = 40
    angular_bins = 24
    occupied = np.zeros((longitudinal_bins, angular_bins), dtype=bool)
    vertex_indices = nearest_vertices[surface_inliers]
    if vertex_indices.size:
        rings = vertex_indices // n_radial
        angular = vertex_indices % n_radial
        long_bin = np.minimum((rings * longitudinal_bins) // n_long, longitudinal_bins - 1)
        angular_bin = np.minimum((angular * angular_bins) // n_radial, angular_bins - 1)
        occupied[long_bin, angular_bin] = True
    expected_bins = np.zeros(longitudinal_bins, dtype=bool)
    expected_bins[
        min((visible_first * longitudinal_bins) // n_long, longitudinal_bins - 1) :
        min((visible_last * longitudinal_bins) // n_long + 1, longitudinal_bins)
    ] = True
    expected_occupancy = occupied[expected_bins]
    ring_coverage = expected_occupancy.any(axis=1) if expected_occupancy.size else np.array([], dtype=bool)
    occupied_longitudinal = occupied.any(axis=1)
    occupied_indices = np.flatnonzero(occupied_longitudinal)
    if occupied_indices.size:
        reconstructed_span = occupied_longitudinal[occupied_indices[0] : occupied_indices[-1] + 1]
        reconstructed_internal_holes = int((~reconstructed_span).sum())
    else:
        reconstructed_internal_holes = longitudinal_bins

    reprojection_errors = np.asarray([point["error_px"] for point in points], dtype=np.float64)
    track_lengths = np.asarray([len(point["image_ids"]) for point in points], dtype=np.int64)
    result = {
        "sequence": str(sequence),
        "model": str(model),
        "registered_images": len(images),
        "total_ground_truth_images": len(truth),
        "registered_fraction": len(images) / len(truth),
        "points3D": len(points),
        "sim3": {
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "source": sim3_source,
            "camera_centre_singular_value_ratios": centre_spectrum_ratios.tolist(),
        },
        "position_only_sim3_diagnostic": {
            "scale": position_scale,
            "rotation": position_rotation.tolist(),
            "translation": position_translation.tolist(),
            "camera_position_rmse": float(
                np.sqrt(np.mean(position_centre_errors * position_centre_errors))
            ),
            "note": "Diagnostic only; centre-only roll is weakly constrained for tubular trajectories.",
        },
        "camera_position_error": {
            "rmse": float(np.sqrt(np.mean(centre_errors * centre_errors))),
            "median": _percentile(centre_errors, 50),
            "p95": _percentile(centre_errors, 95),
            "max": float(centre_errors.max()),
            "normalized_rmse_by_tube_radius": float(np.sqrt(np.mean(centre_errors * centre_errors)) / tube_radius),
        },
        "camera_rotation_error_deg": {
            "median": _percentile(rotation_errors, 50),
            "p95": _percentile(rotation_errors, 95),
            "max": float(rotation_errors.max()),
        },
        "point_reprojection_error_px": {
            "mean": float(reprojection_errors.mean()) if reprojection_errors.size else math.nan,
            "median": _percentile(reprojection_errors, 50),
            "p95": _percentile(reprojection_errors, 95),
        },
        "track_length": {
            "mean": float(track_lengths.mean()) if track_lengths.size else math.nan,
            "median": _percentile(track_lengths, 50),
            "p05": _percentile(track_lengths, 5),
        },
        "observation_time_coverage": {
            "registered_frame_range": [registered_first, registered_last],
            "bins": temporal_bins,
            "bins_occupied": int(temporal_occupancy.sum()),
            "internal_holes": int((~temporal_occupancy).sum()),
            "coverage_fraction": float(temporal_occupancy.mean()),
            "note": "Track-median observation-time coverage; complements but does not prove spatial surface continuity.",
        },
        "dynamic_surface_nearest_vertex_error": {
            "evaluated_points": int(valid.sum()),
            "median": _percentile(surface_distances, 50),
            "p95": _percentile(surface_distances, 95),
            "normalized_median_by_tube_radius": _percentile(surface_distances, 50) / tube_radius,
            "inlier_limit": inlier_limit,
            "inlier_fraction": float(surface_inliers.sum() / max(valid.sum(), 1)),
            "note": "Nearest sampled vertex at the median observation frame; includes mesh-discretization error.",
        },
        "visible_surface_coverage": {
            "expected_ring_index_range": [visible_first, visible_last],
            "estimated_first_wall_depth": wall_depth,
            "longitudinal_bins": int(expected_bins.sum()),
            "longitudinal_bins_occupied": int(ring_coverage.sum()),
            "longitudinal_holes": int((~ring_coverage).sum()),
            "longitudinal_coverage_fraction": float(ring_coverage.mean()) if ring_coverage.size else math.nan,
            "angular_longitudinal_bin_fraction": float(expected_occupancy.mean()) if expected_occupancy.size else math.nan,
            "surface_inlier_points": int(surface_inliers.sum()),
            "reconstructed_longitudinal_bin_span": (
                [int(occupied_indices[0]), int(occupied_indices[-1])]
                if occupied_indices.size
                else None
            ),
            "reconstructed_internal_longitudinal_holes": reconstructed_internal_holes,
        },
        "ground_truth": {
            "tube_length": float(sequence_meta["tube_length"]),
            "median_radius": tube_radius,
            "min_camera_clearance": float(sequence_meta["min_camera_clearance"]),
        },
    }
    return result


def render_overview(sequence: Path, model: Path, output: Path) -> None:
    """Render three orthographic views for a quick geometry sanity check."""
    images = _read_images(model / "images.txt")
    points = _read_points(model / "points3D.txt")
    truth = _read_ground_truth_poses(sequence / "poses.csv")
    matched = [image for image in images.values() if image["name"] in truth]
    estimated_centres = np.stack([image["center"] for image in matched])
    truth_centres = np.stack([truth[image["name"]]["center"] for image in matched])
    estimated_rotations = np.stack([image["rotation_wc"] for image in matched])
    truth_rotations = np.stack([truth[image["name"]]["rotation_wc"] for image in matched])
    scale, rotation, translation, _, _ = _select_similarity(
        estimated_centres, truth_centres, estimated_rotations, truth_rotations
    )
    xyz = np.stack([point["xyz"] for point in points])
    xyz = (scale * (rotation @ xyz.T)).T + translation
    cameras = (scale * (rotation @ estimated_centres.T)).T + translation
    mesh = np.load(sequence / "mesh_static.npz")
    centerline = mesh["centerline"].astype(np.float64)
    static_surface = _tube_vertices(
        mesh,
        np.zeros(
            (centerline.shape[0], mesh["theta"].shape[0], 3), dtype=np.float64
        ),
    ).reshape(-1, 3)

    panel_width, panel_height = 700, 620
    canvas = np.full((panel_height, panel_width * 3, 3), 248, dtype=np.uint8)
    views = ((0, 1, "X / Y"), (0, 2, "X / Z"), (1, 2, "Y / Z"))
    # Use known finite scene bounds so a few COLMAP outliers cannot collapse
    # the useful geometry to a speck in the diagnostic image.
    all_geometry = np.concatenate([static_surface, cameras], axis=0)
    for panel, (axis_x, axis_y, label) in enumerate(views):
        values = all_geometry[:, [axis_x, axis_y]]
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        extent = np.maximum(upper - lower, 1e-6)
        padding = 35
        scale_px = min(
            (panel_width - 2 * padding) / extent[0],
            (panel_height - 2 * padding) / extent[1],
        )

        def project(values_3d: np.ndarray) -> np.ndarray:
            values_2d = values_3d[:, [axis_x, axis_y]]
            pixels = (values_2d - lower) * scale_px
            pixels[:, 0] += panel * panel_width + padding
            pixels[:, 1] = panel_height - padding - pixels[:, 1]
            return np.rint(pixels).astype(np.int32)

        for px, py in project(xyz):
            if panel * panel_width <= px < (panel + 1) * panel_width and 0 <= py < panel_height:
                canvas[py, px] = (175, 110, 45)
        centre_pixels = project(centerline).reshape(-1, 1, 2)
        cv2.polylines(canvas, [centre_pixels], False, (45, 165, 70), 2, cv2.LINE_AA)
        for px, py in project(cameras)[:: max(1, len(cameras) // 45)]:
            cv2.circle(canvas, (int(px), int(py)), 3, (40, 75, 220), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (panel * panel_width + 18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (35, 35, 35),
            2,
            cv2.LINE_AA,
        )
        if panel:
            cv2.line(
                canvas,
                (panel * panel_width, 0),
                (panel * panel_width, panel_height),
                (215, 215, 215),
                1,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write overview image to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overview", type=Path)
    args = parser.parse_args()
    result = evaluate(args.sequence, args.model)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.overview:
        render_overview(args.sequence, args.model, args.overview)
    print(payload, end="")


if __name__ == "__main__":
    main()
