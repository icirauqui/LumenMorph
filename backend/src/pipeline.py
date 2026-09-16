from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from .camera import CameraTrajectory
from .config import canonical_config_dict, config_hash
from .deformation import (
    compute_localized_weight,
    deformed_z,
    sample_localized_wave_components,
    sample_wave_components,
)
from .noise import apply_photometric_noise
from .plane import add_landmarks_and_rigid_regions
from .renderer import render_mesh
from .terrain import build_static_mesh, flatten_vertices
from .texture import generate_texture_with_qa
from .tube import TubeCameraTrajectory, build_tube_static_geometry, deformed_tube_mesh, tube_deformation_weight
from .tube_fea import build_tube_fea_responses
from .types import DatasetConfig, FrameState, LocalizedWaveComponent, SequenceDescriptor
from .writer import (
    ensure_dir,
    open_video_writer,
    write_deformation_csv,
    write_frame_png,
    write_intrinsics,
    write_manifest,
    write_mesh_static,
    write_poses_csv,
    write_sequence_meta,
)


class GenerationCanceledError(RuntimeError):
    """Raised when generation is canceled via cancellation token."""


def _clear_owned_sequence_artifacts(seq_dir: Path) -> None:
    """Remove stale files from an earlier generation of the same sequence.

    Only generator-owned files are touched; unrelated files and directories in
    the selected output location are preserved.
    """
    for directory, pattern in ((seq_dir / "frames", "*.png"),
                               (seq_dir / "mesh_z", "*.npy")):
        if directory.exists():
            for path in directory.glob(pattern):
                path.unlink()
    for name in (
        "intrinsics.json",
        "preview.mp4",
        "poses.csv",
        "deformation_params.csv",
        "mesh_static.npz",
        "sequence_meta.json",
    ):
        path = seq_dir / name
        if path.exists():
            path.unlink()


def _check_canceled(cancel_event: Any | None) -> None:
    if cancel_event is not None and bool(cancel_event.is_set()):
        raise GenerationCanceledError("Generation canceled")


def _emit_progress(
    progress_callback: Callable[[str, float, str], None] | None,
    *,
    stage: str,
    percent: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(stage, float(np.clip(percent, 0.0, 1.0)), message)


def sequence_descriptors(cfg: DatasetConfig) -> list[SequenceDescriptor]:
    seqs: list[SequenceDescriptor] = []
    global_idx = 0
    cycle = cfg.default_preset_cycle

    for split in ("train", "val", "test"):
        split_count = cfg.sequence_count[split]
        for seq_idx in range(split_count):
            preset = cycle[global_idx % len(cycle)]
            seed = cfg.seed + 1009 * global_idx + (11 if split == "train" else 29 if split == "val" else 47)
            seqs.append(SequenceDescriptor(split=split, seq_idx=seq_idx, seed=seed, preset=preset))
            global_idx += 1
    return seqs


def _wave_component_dicts(components) -> list[dict[str, float]]:
    return [asdict(c) for c in components]


def _localized_wave_component_dicts(components) -> list[dict[str, float]]:
    return [asdict(c) for c in components]


def _run_plane_sequence(
    cfg: DatasetConfig,
    preset,
    rng: np.random.Generator,
    texture,
    meshz_dir: Path,
    frames_dir: Path,
    video_writer,
    frame_progress: bool,
    frame_desc: str,
    cancel_event: Any | None = None,
    on_frame: Callable[[int], None] | None = None,
) -> tuple[list[FrameState], list[Any], dict[str, Any], dict[str, np.ndarray]]:
    x_grid, y_grid, base_z, triangles, uv = build_static_mesh(cfg.sequence, rng)
    assert cfg.plane is not None
    base_z, rigid_mask, landmarks, rigid_fraction, rigid_pattern = add_landmarks_and_rigid_regions(
        x_grid=x_grid,
        y_grid=y_grid,
        base_z=base_z,
        cfg=cfg.plane,
        rng=rng,
    )
    rigid_weight_vertices = rigid_mask.reshape(-1).astype(np.float32)
    components = sample_wave_components(rng, preset)
    camera = CameraTrajectory(cfg.camera, preset, rng)

    states: list[FrameState] = []
    min_clearance = float("inf")

    frame_iter = range(cfg.sequence.num_frames)
    if frame_progress and tqdm is not None:
        frame_iter = tqdm(
            frame_iter,
            total=cfg.sequence.num_frames,
            desc=frame_desc,
            dynamic_ncols=True,
            leave=False,
        )

    for frame_idx in frame_iter:
        _check_canceled(cancel_event)
        t_s = frame_idx / float(cfg.sequence.fps)
        z_candidate, global_amp, global_freq, phase_shift = deformed_z(base_z, x_grid, y_grid, t_s, components)
        z_grid = base_z + (z_candidate - base_z) * (1.0 - rigid_mask)
        verts = flatten_vertices(x_grid, y_grid, z_grid)
        pose = camera.pose_at(t_s)

        image = render_mesh(
            vertices_world=verts,
            triangles=triangles,
            uv_vertices=uv,
            texture_bgr=texture,
            intr=cfg.intrinsics,
            R_wc=pose.R_wc,
            t_wc=pose.t_wc,
            rigid_weight_vertices=rigid_weight_vertices,
            rigid_tint_bgr=tuple(cfg.plane.rigid_tint_bgr) if cfg.plane.rigid_colorize else None,
            rigid_tint_strength=cfg.plane.rigid_tint_strength if cfg.plane.rigid_colorize else 0.0,
        )
        if cfg.plane.flip_vertical_view:
            image = np.ascontiguousarray(image[::-1, :, :])
        image = apply_photometric_noise(image, rng, cfg.photometric)

        if cfg.output.write_frames:
            write_frame_png(frames_dir / f"{frame_idx:06d}.png", image)
        if video_writer is not None:
            video_writer.write(image)
        if cfg.output.write_mesh_state:
            np.save(meshz_dir / f"{frame_idx:06d}.npy", z_grid)

        clearance = float(pose.t_wc[2] - np.max(z_grid))
        min_clearance = min(min_clearance, clearance)

        states.append(
            FrameState(
                frame_idx=frame_idx,
                timestamp_s=t_s,
                tx=float(pose.t_wc[0]),
                ty=float(pose.t_wc[1]),
                tz=float(pose.t_wc[2]),
                qx=pose.q_xyzw[0],
                qy=pose.q_xyzw[1],
                qz=pose.q_xyzw[2],
                qw=pose.q_xyzw[3],
                global_amp=global_amp,
                global_freq=global_freq,
                phase_shift=phase_shift,
            )
        )
        if on_frame is not None:
            on_frame(frame_idx + 1)

    static_arrays = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "triangles": triangles,
        "uv": uv,
        "rigid_mask": rigid_mask,
    }
    extra = {
        "scene_type": "plane",
        "mesh_state_type": "z_grid",
        "min_camera_clearance": float(min_clearance),
        "rigid_fraction": rigid_fraction,
        "rigid_pattern": rigid_pattern,
        "landmarks": landmarks,
        "rigid_colorize": bool(cfg.plane.rigid_colorize),
    }
    return states, components, extra, static_arrays


def _run_tube_sequence(
    cfg: DatasetConfig,
    preset,
    rng: np.random.Generator,
    texture,
    meshz_dir: Path,
    frames_dir: Path,
    video_writer,
    frame_progress: bool,
    frame_desc: str,
    cancel_event: Any | None = None,
    on_frame: Callable[[int], None] | None = None,
    localized_components_override: list[LocalizedWaveComponent] | None = None,
) -> tuple[list[FrameState], tuple[list[Any], list[Any]], dict[str, Any], dict[str, np.ndarray]]:
    assert cfg.tube is not None

    geom = build_tube_static_geometry(cfg.tube, preset, rng)
    components = sample_wave_components(rng, preset)
    localized_components = list(localized_components_override) if localized_components_override is not None else None
    if localized_components_override is None and not cfg.tube.deformation.areas and cfg.tube.localized_deform_count > 0:
        localized_components = sample_localized_wave_components(rng, preset, cfg.tube)
    fea_responses = build_tube_fea_responses(geom, cfg.tube)

    camera = TubeCameraTrajectory(
        geom=geom,
        tube_cfg=cfg.tube,
        cam_cfg=cfg.camera,
        preset=preset,
        rng=rng,
        fps=cfg.sequence.fps,
        num_frames=cfg.sequence.num_frames,
    )

    states: list[FrameState] = []
    min_clearance = float("inf")

    frame_iter = range(cfg.sequence.num_frames)
    if frame_progress and tqdm is not None:
        frame_iter = tqdm(
            frame_iter,
            total=cfg.sequence.num_frames,
            desc=frame_desc,
            dynamic_ncols=True,
            leave=False,
        )

    for frame_idx in frame_iter:
        _check_canceled(cancel_event)
        t_s = frame_idx / float(cfg.sequence.fps)
        t_norm = frame_idx / float(max(cfg.sequence.num_frames - 1, 1))
        verts, radius, displacement_grid, global_amp, global_freq, phase_shift = deformed_tube_mesh(
            geom=geom,
            tube_cfg=cfg.tube,
            t_s=t_s,
            components=components,
            localized_components=localized_components,
            fea_responses=fea_responses,
            t_norm=t_norm,
        )
        pose = camera.pose_at(t_s)
        localized_tint_weight_vertices = None
        if cfg.tube.localized_tint_enabled:
            areas = cfg.tube.deformation.areas
            if localized_components:
                from .tube import tube_areas_from_localized_components

                areas = tube_areas_from_localized_components(localized_components)
            if areas:
                localized_tint_weight_vertices = tube_deformation_weight(
                    geom,
                    areas,
                    flow_mode=cfg.tube.deformation.flow_mode,
                    flow_cycles=cfg.tube.deformation.flow_cycles,
                    t_norm=t_norm,
                ).reshape(-1).astype(np.float32)

        image = render_mesh(
            vertices_world=verts,
            triangles=geom.triangles,
            uv_vertices=geom.uv,
            texture_bgr=texture,
            intr=cfg.intrinsics,
            R_wc=pose.R_wc,
            t_wc=pose.t_wc,
            rigid_weight_vertices=localized_tint_weight_vertices,
            rigid_tint_bgr=tuple(cfg.tube.localized_tint_bgr) if cfg.tube.localized_tint_enabled else None,
            rigid_tint_strength=cfg.tube.localized_tint_strength if cfg.tube.localized_tint_enabled else 0.0,
        )
        image = apply_photometric_noise(image, rng, cfg.photometric)

        if cfg.output.write_frames:
            write_frame_png(frames_dir / f"{frame_idx:06d}.png", image)
        if video_writer is not None:
            video_writer.write(image)
        if cfg.output.write_mesh_state:
            np.save(meshz_dir / f"{frame_idx:06d}.npy", displacement_grid)

        clearance = camera.clearance_at(t_s=t_s, radius_grid=radius, cam_pos=pose.t_wc)
        min_clearance = min(min_clearance, clearance)

        states.append(
            FrameState(
                frame_idx=frame_idx,
                timestamp_s=t_s,
                tx=float(pose.t_wc[0]),
                ty=float(pose.t_wc[1]),
                tz=float(pose.t_wc[2]),
                qx=pose.q_xyzw[0],
                qy=pose.q_xyzw[1],
                qz=pose.q_xyzw[2],
                qw=pose.q_xyzw[3],
                global_amp=global_amp,
                global_freq=global_freq,
                phase_shift=phase_shift,
            )
        )
        if on_frame is not None:
            on_frame(frame_idx + 1)

    static_arrays = {
        "centerline": geom.centerline,
        "tangent": geom.tangent,
        "frame_right": geom.frame_right,
        "frame_up": geom.frame_up,
        "theta": geom.theta,
        "s_norm": geom.s_norm,
        "base_radius": geom.base_radius,
        "triangles": geom.triangles,
        "uv": geom.uv,
    }
    extra = {
        "scene_type": "tube",
        "mesh_state_type": "tube_displacement_xyz_grid",
        "tube_length": float(cfg.tube.length),
        "tube_min_radius": float(cfg.tube.min_radius),
        "localized_deformation_count": (
            len(localized_components)
            if localized_components is not None
            else sum(1 for area in cfg.tube.deformation.areas if area.kind == "localized")
        ),
        "localized_tint_enabled": bool(cfg.tube.localized_tint_enabled),
        "localized_tint_strength": float(cfg.tube.localized_tint_strength),
        "fea_deformation_count": len(fea_responses),
        "fea_deformation_responses": [
            {
                "area_id": r.area_id,
                "applied_s_norm": r.applied_s_norm,
                "applied_theta_norm": r.applied_theta_norm,
                "applied_node": r.applied_node,
                "applied_axial_index": r.applied_axial_index,
                "applied_theta_index": r.applied_theta_index,
                "force_vector_local": r.force_vector_local.tolist(),
                "load_distribution": r.load_distribution,
                "load_node_count": int(r.load_node_count),
            }
            for r in fea_responses
        ],
        "min_camera_clearance": float(min_clearance),
    }
    localized_meta = localized_components if localized_components is not None else list(cfg.tube.deformation.areas)
    return states, (components, localized_meta), extra, static_arrays


def generate_dataset(
    cfg: DatasetConfig,
    out_dir: str | Path,
    max_sequences: int | None = None,
    override_num_frames: int | None = None,
    override_fps: int | None = None,
    show_progress: bool = True,
    localized_components_override: list[LocalizedWaveComponent] | None = None,
    flatten_single_sequence: bool = False,
    cancel_event: Any | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    cfg_runtime = cfg
    if override_num_frames is not None:
        cfg_runtime = replace(cfg, sequence=replace(cfg.sequence, num_frames=int(override_num_frames)))
    if override_fps is not None:
        cfg_runtime = replace(cfg_runtime, sequence=replace(cfg_runtime.sequence, fps=int(override_fps)))

    (out_dir / "config_used.json").write_text(json.dumps(canonical_config_dict(cfg_runtime), indent=2))

    descriptors = sequence_descriptors(cfg_runtime)
    if max_sequences is not None:
        descriptors = descriptors[: int(max_sequences)]
    use_flat_layout = bool(flatten_single_sequence and len(descriptors) == 1)

    if not use_flat_layout:
        for split in ("train", "val", "test"):
            ensure_dir(out_dir / split)

    total_frames = max(1, len(descriptors) * max(cfg_runtime.sequence.num_frames, 1))
    completed_frames = 0

    manifest: dict[str, Any] = {
        "generator_version": cfg_runtime.version,
        "scene_type": cfg_runtime.scene_type,
        "config_hash": config_hash(cfg_runtime),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_sequences": len(descriptors),
        "layout": "flat_single_sequence" if use_flat_layout else "split_sequence",
        "splits": {"train": [], "val": [], "test": []},
        "sequences": [],
    }

    seq_iter = descriptors
    if show_progress and tqdm is not None:
        seq_iter = tqdm(descriptors, total=len(descriptors), desc="Sequences", dynamic_ncols=True)

    _emit_progress(progress_callback, stage="setup", percent=0.0, message="Preparing generation")
    for desc in seq_iter:
        _check_canceled(cancel_event)
        preset = cfg_runtime.difficulty_presets[desc.preset]
        localized_components: list[Any] = []
        seq_name = f"seq_{desc.seq_idx:03d}"
        if use_flat_layout:
            seq_rel = Path(".")
            seq_dir = out_dir
            run_label = "single_sequence"
        else:
            seq_rel = Path(desc.split) / seq_name
            seq_dir = out_dir / seq_rel
            run_label = f"{desc.split}/{seq_name}"
        ensure_dir(seq_dir)
        _clear_owned_sequence_artifacts(seq_dir)
        frames_dir = seq_dir / "frames"
        meshz_dir = seq_dir / "mesh_z"
        if cfg_runtime.output.write_frames:
            ensure_dir(frames_dir)
        if cfg_runtime.output.write_mesh_state:
            ensure_dir(meshz_dir)

        rng = np.random.default_rng(desc.seed)
        texture, tex_feature_score = generate_texture_with_qa(cfg_runtime.texture, rng)

        write_intrinsics(seq_dir / "intrinsics.json", cfg_runtime.intrinsics)

        video_writer = None
        if cfg_runtime.output.write_mp4:
            video_writer = open_video_writer(
                seq_dir / "preview.mp4",
                cfg_runtime.intrinsics.width,
                cfg_runtime.intrinsics.height,
                cfg_runtime.sequence.fps,
            )

        _emit_progress(
            progress_callback,
            stage="sequence",
            percent=completed_frames / float(total_frames),
            message=f"Running {run_label}",
        )

        def _on_frame(_frame_done: int) -> None:
            nonlocal completed_frames
            completed_frames += 1
            _emit_progress(
                progress_callback,
                stage="frames",
                percent=completed_frames / float(total_frames),
                message=f"Rendering {run_label}",
            )

        try:
            if cfg_runtime.scene_type == "plane":
                states, components, scene_meta, static_arrays = _run_plane_sequence(
                    cfg=cfg_runtime,
                    preset=preset,
                    rng=rng,
                    texture=texture,
                    meshz_dir=meshz_dir,
                    frames_dir=frames_dir,
                    video_writer=video_writer,
                    frame_progress=(show_progress and tqdm is not None),
                    frame_desc=f"Frames {run_label}",
                    cancel_event=cancel_event,
                    on_frame=_on_frame,
                )
                write_mesh_static(seq_dir / "mesh_static.npz", **static_arrays)
            else:
                states, tube_components, scene_meta, static_arrays = _run_tube_sequence(
                    cfg=cfg_runtime,
                    preset=preset,
                    rng=rng,
                    texture=texture,
                    meshz_dir=meshz_dir,
                    frames_dir=frames_dir,
                    video_writer=video_writer,
                    frame_progress=(show_progress and tqdm is not None),
                    frame_desc=f"Frames {run_label}",
                    cancel_event=cancel_event,
                    on_frame=_on_frame,
                    localized_components_override=localized_components_override,
                )
                write_mesh_static(seq_dir / "mesh_static.npz", **static_arrays)
                components, localized_components = tube_components
        finally:
            if video_writer is not None:
                video_writer.release()

        write_poses_csv(seq_dir / "poses.csv", states)
        write_deformation_csv(seq_dir / "deformation_params.csv", states, desc.preset)

        sequence_meta = {
            "split": desc.split,
            "seq_idx": desc.seq_idx,
            "seed": desc.seed,
            "preset": desc.preset,
            "scene_type": cfg_runtime.scene_type,
            "num_frames": cfg_runtime.sequence.num_frames,
            "fps": cfg_runtime.sequence.fps,
            "texture_orb_score": int(tex_feature_score),
            "wave_components": _wave_component_dicts(components),
            "localized_wave_components": _localized_wave_component_dicts(localized_components)
            if cfg_runtime.scene_type == "tube"
            else [],
            **scene_meta,
        }
        write_sequence_meta(seq_dir / "sequence_meta.json", sequence_meta)

        manifest["splits"][desc.split].append(str(seq_rel))
        manifest["sequences"].append(
            {
                "split": desc.split,
                "seq_idx": desc.seq_idx,
                "relative_path": str(seq_rel),
                "preset": desc.preset,
                "seed": desc.seed,
                "scene_type": cfg_runtime.scene_type,
                "texture_orb_score": int(tex_feature_score),
            }
        )

    _check_canceled(cancel_event)
    write_manifest(out_dir / "dataset_manifest.json", manifest)
    _emit_progress(progress_callback, stage="done", percent=1.0, message="Generation completed")
    return manifest
