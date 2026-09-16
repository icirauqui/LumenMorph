#!/usr/bin/env python3
"""Triangulate verified COLMAP tracks at generator ground-truth camera poses.

This produces a geometry/alignment control. It is deliberately stored apart
from independently estimated SfM and must never be reported as trajectory
estimation performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import numpy as np

try:
    from backend.evaluate_colmap_tube import _rotation_from_quaternion_wxyz
except ModuleNotFoundError:  # Direct ``python backend/script.py`` execution.
    from evaluate_colmap_tube import _rotation_from_quaternion_wxyz


# The renderer uses a right/up/forward camera frame and maps image rows with
# v = cy - fy * Y/Z. COLMAP uses v = cy + fy * Y/Z and requires proper camera
# rotations. A camera-Y reflection alone would be improper, so pair it with a
# fixed reflection of the exported world frame. Geometry written by this
# control therefore lives in this explicitly documented COLMAP-compatible
# reflected world frame.
GENERATOR_TO_COLMAP_WORLD = np.diag([1.0, -1.0, 1.0])
GENERATOR_TO_COLMAP_CAMERA = np.diag([1.0, -1.0, 1.0])


def _quaternion_wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _ground_truth_colmap_poses(poses_csv: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with poses_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = f"{int(row['frame']):06d}.png"
            q_wc_xyzw = np.array(
                [row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64
            )
            rotation_wc = _rotation_from_quaternion_wxyz(q_wc_xyzw[[3, 0, 1, 2]])
            centre_generator = np.array(
                [row["tx"], row["ty"], row["tz"]], dtype=np.float64
            )
            centre = GENERATOR_TO_COLMAP_WORLD @ centre_generator
            rotation_wc = (
                GENERATOR_TO_COLMAP_WORLD
                @ rotation_wc
                @ GENERATOR_TO_COLMAP_CAMERA
            )
            rotation_cw = rotation_wc.T
            translation_cw = -(rotation_cw @ centre)
            result[name] = (_quaternion_wxyz_from_rotation(rotation_cw), translation_cw)
    return result


def _database_keypoints(connection: sqlite3.Connection) -> dict[int, np.ndarray]:
    """Load COLMAP keypoint coordinates indexed by database image id."""
    result: dict[int, np.ndarray] = {}
    for image_id, rows, columns, data in connection.execute(
        "SELECT image_id, rows, cols, data FROM keypoints ORDER BY image_id"
    ):
        keypoints = np.frombuffer(data, dtype=np.float32)
        expected = int(rows) * int(columns)
        if keypoints.size != expected:
            raise RuntimeError(
                f"Malformed keypoint blob for image {image_id}: "
                f"expected {expected} floats, found {keypoints.size}"
            )
        result[int(image_id)] = keypoints.reshape(int(rows), int(columns))[:, :2].copy()
    return result


def _format_fork_observations(keypoints: np.ndarray) -> str:
    """Write X Y POINT3D_ID POINT3DD_ID for the retained COLMAP fork."""
    return " ".join(
        f"{float(x):.9g} {float(y):.9g} -1 -1" for x, y in keypoints
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _database_bundle_hashes(database: Path) -> dict[str, str]:
    paths = [database, Path(f"{database}-wal"), Path(f"{database}-shm")]
    return {path.name: _sha256(path) for path in paths if path.exists()}


def _copy_database_bundle(source: Path, destination: Path) -> None:
    """Byte-copy a SQLite database and any WAL companions without opening it."""
    shutil.copy2(source, destination)
    for suffix in ("-wal", "-shm"):
        companion = Path(f"{source}{suffix}")
        if companion.exists():
            shutil.copy2(companion, Path(f"{destination}{suffix}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--image-list",
        type=Path,
        help="Optional newline-delimited filename list for a temporal window.",
    )
    parser.add_argument("--min-angle-deg", type=float, default=1.5)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes_before = _database_bundle_hashes(args.database)

    intrinsics = json.loads((args.sequence / "intrinsics.json").read_text(encoding="utf-8"))
    poses = _ground_truth_colmap_poses(args.sequence / "poses.csv")
    with tempfile.TemporaryDirectory(prefix="known_pose_inspect_") as inspect_name:
        inspection_database = Path(inspect_name) / "database.db"
        _copy_database_bundle(args.database, inspection_database)
        with sqlite3.connect(inspection_database) as connection:
            database_images = connection.execute(
                "SELECT image_id, name FROM images ORDER BY image_id"
            ).fetchall()
            keypoints_by_image = _database_keypoints(connection)
    database_names = {name for _, name in database_images}
    if database_names != set(poses):
        raise RuntimeError("Database image names and ground-truth pose names differ")
    if {int(image_id) for image_id, _ in database_images} != set(keypoints_by_image):
        raise RuntimeError("Database images and keypoint rows differ")

    selected_names: set[str] | None = None
    if args.image_list is not None:
        image_list = [
            line.strip()
            for line in args.image_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not image_list:
            raise RuntimeError(f"Empty image list: {args.image_list}")
        if len(image_list) != len(set(image_list)):
            raise RuntimeError(f"Duplicate filenames in image list: {args.image_list}")
        selected_names = set(image_list)
        missing = selected_names - database_names
        if missing:
            raise RuntimeError(
                "Image-list filenames absent from database: " + ", ".join(sorted(missing))
            )
        database_images = [
            (image_id, name)
            for image_id, name in database_images
            if name in selected_names
        ]

    with tempfile.TemporaryDirectory(prefix="known_pose_control_") as work_name:
        work = Path(work_name)
        seed = work / "seed"
        seed.mkdir()
        working_database = work / "database.db"
        # The retained colonoscopy fork updates its custom keypoint-detection
        # metadata while triangulating. Byte-copy the complete SQLite bundle
        # before opening it so even WAL/SHM creation or checkpointing is
        # confined to the temporary workspace.
        _copy_database_bundle(args.database, working_database)
        (seed / "cameras.txt").write_text(
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"1 PINHOLE {intrinsics['width']} {intrinsics['height']} "
            f"{intrinsics['fx']} {intrinsics['fy']} {intrinsics['cx']} {intrinsics['cy']}\n",
            encoding="utf-8",
        )
        image_lines = ["# Ground-truth camera poses; no points are injected."]
        for image_id, name in database_images:
            quaternion, translation = poses[name]
            values = [*quaternion.tolist(), *translation.tolist()]
            image_lines.append(
                f"{image_id} " + " ".join(f"{value:.17g}" for value in values)
                + f" 1 {name}"
            )
            observations = _format_fork_observations(
                keypoints_by_image[int(image_id)]
            )
            image_lines.append(observations)
        (seed / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        (seed / "points3D.txt").write_text(
            "# Empty seed: verified database tracks are triangulated below.\n",
            encoding="utf-8",
        )
        # The thesis-era COLMAP fork requires its companion deformable-point
        # file even when the seed contains no 3D points.
        (seed / "points3Dd.txt").write_text(
            "# Empty deformable-point seed.\n",
            encoding="utf-8",
        )

        command = [
            str(args.colmap), "point_triangulator",
            "--database_path", str(working_database),
            "--image_path", str(args.images),
            "--input_path", str(seed),
            "--output_path", str(args.output),
            "--clear_points", "1",
            "--Mapper.fix_existing_images", "1",
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
            "--Mapper.tri_min_angle", str(args.min_angle_deg),
            "--Mapper.filter_min_tri_angle", str(args.min_angle_deg),
            "--Mapper.filter_max_reproj_error", "2.0",
            "--Mapper.tri_ignore_two_view_tracks", "1",
        ]
        subprocess.run(command, check=True)

    subprocess.run(
        [
            str(args.colmap), "model_converter",
            "--input_path", str(args.output),
            "--output_path", str(args.output),
            "--output_type", "TXT",
        ],
        check=True,
    )
    (args.output / "CONTROL_CLASSIFICATION.txt").write_text(
        "Known-pose geometry/alignment control. Camera poses come from generator "
        "ground truth after the documented generator-to-COLMAP paired "
        "camera/world reflection; this is not independently estimated SfM. "
        + (
            "Triangulation is restricted to an explicit temporal window.\n"
            if selected_names is not None
            else "Triangulation uses the complete sequence.\n"
        ),
        encoding="utf-8",
    )
    source_hashes_after = _database_bundle_hashes(args.database)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError(
            "Source database files changed while building the control: "
            f"before={source_hashes_before}, after={source_hashes_after}"
        )
    recorded_command = [
        "{temporary_database_snapshot}" if value == str(working_database) else value
        for value in command
    ]
    (args.output / "triangulation_command.json").write_text(
        json.dumps(
            {
                "command": recorded_command,
                "source_database": str(args.database),
                "source_database_files_sha256": source_hashes_before,
                "min_angle_deg": args.min_angle_deg,
                "image_list": str(args.image_list) if args.image_list else None,
                "selected_image_count": len(database_images),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
