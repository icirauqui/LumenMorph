#!/usr/bin/env python3
"""Build reproducible temporal COLMAP local models from explicit frame names.

The colonoscopy COLMAP fork's ``mapper_bundle`` selects windows by database
image identifier. Parallel feature extraction does not guarantee that those
identifiers follow filename order, so a numerically contiguous ID interval can
silently mix temporal frames. This utility gives the regular mapper an explicit
filename list for every window and validates the registered filenames before
accepting the result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


MAPPER_OPTIONS = [
    "--Mapper.multiple_models", "0",
    "--Mapper.ba_refine_focal_length", "0",
    "--Mapper.ba_refine_principal_point", "0",
    "--Mapper.ba_refine_extra_params", "0",
    "--Mapper.ba_global_max_num_iterations", "35",
    "--Mapper.ba_global_max_refinements", "2",
    "--Mapper.init_min_num_inliers", "75",
    "--Mapper.init_max_error", "2.0",
    "--Mapper.init_max_forward_motion", "0.999",
    "--Mapper.init_min_tri_angle", "3.0",
    "--Mapper.abs_pose_max_error", "4.0",
    "--Mapper.abs_pose_min_num_inliers", "40",
    "--Mapper.filter_max_reproj_error", "2.0",
    "--Mapper.filter_min_tri_angle", "1.0",
    "--Mapper.local_ba_min_tri_angle", "2.0",
    "--Mapper.tri_ignore_two_view_tracks", "1",
]


def _registered_names(images_txt: Path) -> list[str]:
    names: list[str] = []
    data_line = 0
    for raw_line in images_txt.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if data_line % 2 == 0:
            names.append(line.split()[9])
        data_line += 1
    return sorted(names)


def _window_starts(num_frames: int, bundle_size: int, stride: int) -> list[int]:
    if num_frames < bundle_size:
        raise ValueError("num_frames must be at least bundle_size")
    starts = list(range(0, num_frames - bundle_size + 1, stride))
    final_start = num_frames - bundle_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-frames", required=True, type=int)
    parser.add_argument("--bundle-size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    lists_dir = args.output / "window_lists"
    lists_dir.mkdir()

    summaries: list[dict] = []
    for model_index, start in enumerate(
        _window_starts(args.num_frames, args.bundle_size, args.stride)
    ):
        end = start + args.bundle_size - 1
        expected_names = [f"{frame:06d}.png" for frame in range(start, end + 1)]
        list_path = lists_dir / f"window_{start:06d}_{end:06d}.txt"
        list_path.write_text("\n".join(expected_names) + "\n", encoding="utf-8")

        with tempfile.TemporaryDirectory(
            prefix=f"colmap_local_{model_index:03d}_", dir=args.output
        ) as temp_name:
            temp_output = Path(temp_name)
            command = [
                str(args.colmap),
                "mapper",
                "--database_path", str(args.database),
                "--image_path", str(args.images),
                "--output_path", str(temp_output),
                "--image_list_path", str(list_path),
                "--Mapper.num_threads", str(args.threads),
                *MAPPER_OPTIONS,
            ]
            subprocess.run(command, check=True)
            candidates = sorted(path for path in temp_output.iterdir() if path.is_dir())
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Window {start}:{end} produced {len(candidates)} models, expected one"
                )

            final_model = args.output / str(model_index)
            final_model.mkdir()
            for source in candidates[0].iterdir():
                if source.is_file():
                    shutil.copy2(source, final_model / source.name)
            subprocess.run(
                [
                    str(args.colmap), "model_converter",
                    "--input_path", str(candidates[0]),
                    "--output_path", str(final_model),
                    "--output_type", "TXT",
                ],
                check=True,
            )

        registered_names = _registered_names(final_model / "images.txt")
        if registered_names != expected_names:
            raise RuntimeError(
                f"Window {start}:{end} registered {registered_names}, expected {expected_names}"
            )
        analyzer = subprocess.run(
            [str(args.colmap), "model_analyzer", "--path", str(final_model)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
        summaries.append(
            {
                "model": model_index,
                "first_frame": start,
                "last_frame": end,
                "registered_images": len(registered_names),
                "image_list": str(list_path),
                "model_analyzer": analyzer.strip().splitlines(),
                "mapper_command": command,
            }
        )
        print(f"Accepted local {model_index}: frames {start}-{end}", flush=True)

    (args.output / "reconstruction_summary.json").write_text(
        json.dumps(
            {
                "bundle_size": args.bundle_size,
                "stride": args.stride,
                "num_frames": args.num_frames,
                "models": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
