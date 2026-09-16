from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config_dict
from .pipeline import generate_dataset
from .texture import orb_feature_count
from .types import LocalizedWaveComponent
from .validator import validate_dataset

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("OpenCV (cv2) is required for CLI operations") from exc


def _cmd_generate(args: argparse.Namespace) -> int:
    raw_payload = json.loads(Path(args.config).read_text())
    if not isinstance(raw_payload, dict):
        raise ValueError(f"Config root must be an object: {args.config}")
    cfg = load_config_dict(raw_payload)
    localized_override = (
        _localized_components_from_raw(raw_payload) if cfg.scene_type == "tube" else None
    )
    max_sequences = args.max_sequences
    if args.single_sequence:
        max_sequences = 1
    manifest = generate_dataset(
        cfg=cfg,
        out_dir=args.out,
        max_sequences=max_sequences,
        override_num_frames=args.override_num_frames,
        override_fps=args.override_fps,
        show_progress=not args.no_progress,
        localized_components_override=localized_override,
    )
    print(json.dumps({"status": "ok", "manifest": manifest}, indent=2))
    return 0


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(default)


def _localized_components_from_raw(raw_payload: dict) -> list[LocalizedWaveComponent] | None:
    items = raw_payload.get("localized_components")
    if not isinstance(items, list):
        return None
    out: list[LocalizedWaveComponent] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        legacy_amp = _to_float(item.get("amp"), 0.0)
        transversal_only = _to_bool(item.get("transversal_only"), False)
        flow_dir_s = 0.0 if transversal_only else _to_float(item.get("flow_dir_s"), 0.0)
        flow_dir_theta = 0.0 if transversal_only else _to_float(item.get("flow_dir_theta"), 0.0)
        out.append(
            LocalizedWaveComponent(
                amp=legacy_amp,
                amp_out=_to_float(item.get("amp_out"), legacy_amp),
                amp_in=_to_float(item.get("amp_in"), legacy_amp),
                fx=_to_float(item.get("fx"), 0.0),
                fy=_to_float(item.get("fy"), 0.0),
                omega=_to_float(item.get("omega"), 0.0),
                phase=_to_float(item.get("phase"), 0.0),
                center_s=_to_float(item.get("center_s"), 0.5),
                center_theta=_to_float(item.get("center_theta"), 0.0),
                sigma_s=max(_to_float(item.get("sigma_s"), 0.1), 1e-4),
                sigma_theta=max(_to_float(item.get("sigma_theta"), 0.1), 1e-4),
                transversal_only=transversal_only,
                flow_dir_s=flow_dir_s,
                flow_dir_theta=flow_dir_theta,
            )
        )
    return out


def _cmd_validate(args: argparse.Namespace) -> int:
    result = validate_dataset(args.dataset)
    payload = {
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "stats": result.stats,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


def _cmd_preview(args: argparse.Namespace) -> int:
    seq_dir = Path(args.sequence)
    if not seq_dir.exists():
        print(json.dumps({"ok": False, "error": f"Sequence not found: {seq_dir}"}, indent=2))
        return 1

    frames = sorted((seq_dir / "frames").glob("*.png"))
    if not frames:
        print(json.dumps({"ok": False, "error": "No frames found"}, indent=2))
        return 1

    sample_count = min(len(frames), 30)
    sample_idx = np.linspace(0, len(frames) - 1, sample_count, dtype=np.int32)
    orb_counts = []
    shape = None
    for idx in sample_idx:
        im = cv2.imread(str(frames[idx]))
        if im is None:
            continue
        if shape is None:
            shape = [int(im.shape[1]), int(im.shape[0])]
        orb_counts.append(orb_feature_count(im))

    mp4 = seq_dir / "preview.mp4"
    mp4_ok = False
    if mp4.exists():
        cap = cv2.VideoCapture(str(mp4))
        if cap.isOpened():
            ret, frm = cap.read()
            mp4_ok = bool(ret and frm is not None)
        cap.release()

    payload = {
        "ok": True,
        "sequence": str(seq_dir),
        "frame_count": len(frames),
        "resolution": shape,
        "mean_orb_features": float(np.mean(orb_counts)) if orb_counts else 0.0,
        "mp4_exists": mp4.exists(),
        "mp4_readable": mp4_ok,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_api(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - dependency optional in some setups
        raise RuntimeError("API mode requires 'uvicorn' (install requirements.txt)") from exc

    from .api.app import create_app

    uvicorn.run(
        create_app(),
        host=str(args.host),
        port=int(args.port),
        log_level=str(args.log_level),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LumenMorph: synthetic deformable endoscopic sequence generator"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate dataset")
    g.add_argument("--config", type=str, required=True)
    g.add_argument("--out", type=str, required=True)
    g.add_argument("--max-sequences", type=int, default=None)
    g.add_argument("--single-sequence", action="store_true", help="Generate one sequence only")
    g.add_argument("--override-num-frames", type=int, default=None)
    g.add_argument("--override-fps", type=int, default=None, help="Override sequence FPS")
    g.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    g.set_defaults(func=_cmd_generate)

    v = sub.add_parser("validate", help="Validate dataset")
    v.add_argument("--dataset", type=str, required=True)
    v.set_defaults(func=_cmd_validate)

    pr = sub.add_parser("preview", help="Preview sequence statistics")
    pr.add_argument("--sequence", type=str, required=True)
    pr.set_defaults(func=_cmd_preview)

    api = sub.add_parser("api", help="Run local generator API service")
    api.add_argument("--host", type=str, default="127.0.0.1")
    api.add_argument("--port", type=int, default=8765)
    api.add_argument("--log-level", type=str, default="info")
    api.set_defaults(func=_cmd_api)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
