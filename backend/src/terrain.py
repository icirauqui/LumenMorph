from __future__ import annotations

import numpy as np

from .types import SequenceConfig


def _upsample_nearest(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    rep_h = max(1, int(np.ceil(target_h / arr.shape[0])))
    rep_w = max(1, int(np.ceil(target_w / arr.shape[1])))
    up = np.repeat(np.repeat(arr, rep_h, axis=0), rep_w, axis=1)
    return up[:target_h, :target_w]


def _fractal_noise(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for octave in range(octaves):
        scale = 2 ** octave
        low_h = max(2, h // scale)
        low_w = max(2, w // scale)
        coarse = rng.normal(0.0, 1.0, size=(low_h, low_w)).astype(np.float32)
        up = _upsample_nearest(coarse, h, w)
        noise += amp * up
        total += amp
        amp *= 0.5
    noise /= max(total, 1e-8)

    # Two rounds of simple box blur to smooth block artifacts.
    for _ in range(2):
        padded = np.pad(noise, ((1, 1), (1, 1)), mode="edge")
        noise = (
            padded[0:-2, 0:-2]
            + padded[0:-2, 1:-1]
            + padded[0:-2, 2:]
            + padded[1:-1, 0:-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, 0:-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0

    noise -= noise.min()
    ptp = np.ptp(noise)
    if ptp > 1e-8:
        noise /= ptp
    return noise


def build_static_mesh(seq_cfg: SequenceConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(seq_cfg.x_min, seq_cfg.x_max, seq_cfg.mesh_nx, dtype=np.float32)
    ys = np.linspace(seq_cfg.y_min, seq_cfg.y_max, seq_cfg.mesh_ny, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)

    base = _fractal_noise((seq_cfg.mesh_ny, seq_cfg.mesh_nx), rng)
    mountain = (base - 0.5) * 3.0

    # Add broad global undulation so the shape has terrain-like slopes.
    broad = 0.6 * np.sin(0.18 * x_grid + 0.11 * y_grid) + 0.4 * np.cos(0.13 * x_grid - 0.17 * y_grid)
    z_grid = mountain + broad.astype(np.float32)

    tris = []
    for j in range(seq_cfg.mesh_ny - 1):
        for i in range(seq_cfg.mesh_nx - 1):
            idx0 = j * seq_cfg.mesh_nx + i
            idx1 = idx0 + 1
            idx2 = idx0 + seq_cfg.mesh_nx
            idx3 = idx2 + 1
            tris.append((idx0, idx2, idx1))
            tris.append((idx1, idx2, idx3))
    triangles = np.asarray(tris, dtype=np.int32)

    u = (x_grid - seq_cfg.x_min) / max(seq_cfg.x_max - seq_cfg.x_min, 1e-8)
    v = (y_grid - seq_cfg.y_min) / max(seq_cfg.y_max - seq_cfg.y_min, 1e-8)
    uv = np.stack([u, v], axis=-1).reshape(-1, 2).astype(np.float32)

    return x_grid.astype(np.float32), y_grid.astype(np.float32), z_grid.astype(np.float32), triangles, uv


def flatten_vertices(x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    return np.stack([x_grid, y_grid, z_grid], axis=-1).reshape(-1, 3).astype(np.float32)
