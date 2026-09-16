from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .types import CameraIntrinsics, FrameState

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("OpenCV (cv2) is required for writing frames/videos") from exc


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_intrinsics(path: Path, intr: CameraIntrinsics) -> None:
    payload = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
    }
    path.write_text(json.dumps(payload, indent=2))


def write_mesh_static(path: Path, **arrays) -> None:
    import numpy as np

    np.savez_compressed(path, **arrays)


def write_poses_csv(path: Path, frame_states: Iterable[FrameState]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_s", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        for s in frame_states:
            w.writerow([s.frame_idx, s.timestamp_s, s.tx, s.ty, s.tz, s.qx, s.qy, s.qz, s.qw])


def write_deformation_csv(path: Path, frame_states: Iterable[FrameState], preset: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_s", "preset", "global_amp", "global_freq", "phase_shift"])
        for s in frame_states:
            w.writerow([s.frame_idx, s.timestamp_s, preset, s.global_amp, s.global_freq, s.phase_shift])


def write_sequence_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def open_video_writer(path: Path, width: int, height: int, fps: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def write_frame_png(path: Path, image_bgr) -> None:
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write frame: {path}")


def frame_state_to_dict(s: FrameState) -> dict[str, Any]:
    return asdict(s)
