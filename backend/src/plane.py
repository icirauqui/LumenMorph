from __future__ import annotations

from typing import Any

import numpy as np

from .types import PlaneSceneConfig


def _box_blur(arr: np.ndarray, passes: int) -> np.ndarray:
    out = arr.astype(np.float32)
    for _ in range(max(0, int(passes))):
        p = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            p[0:-2, 0:-2]
            + p[0:-2, 1:-1]
            + p[0:-2, 2:]
            + p[1:-1, 0:-2]
            + p[1:-1, 1:-1]
            + p[1:-1, 2:]
            + p[2:, 0:-2]
            + p[2:, 1:-1]
            + p[2:, 2:]
        ) / 9.0
    return out.astype(np.float32)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def add_landmarks_and_rigid_regions(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    base_z: np.ndarray,
    cfg: PlaneSceneConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float, str]:
    z = base_z.astype(np.float32).copy()
    rigid_landmark_mask = np.zeros_like(z, dtype=np.float32)

    x_min, x_max = float(np.min(x_grid)), float(np.max(x_grid))
    y_min, y_max = float(np.min(y_grid)), float(np.max(y_grid))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    landmarks: list[dict[str, Any]] = []
    for i in range(max(0, int(cfg.landmark_count))):
        cx = float(rng.uniform(x_min + 0.1 * x_span, x_max - 0.1 * x_span))
        cy = float(rng.uniform(y_min + 0.08 * y_span, y_max - 0.08 * y_span))
        h = float(rng.uniform(cfg.landmark_height_min, cfg.landmark_height_max))
        sigma = float(rng.uniform(cfg.landmark_sigma_min, cfg.landmark_sigma_max))

        r2 = ((x_grid - cx) / max(sigma, 1e-4)) ** 2 + ((y_grid - cy) / max(sigma, 1e-4)) ** 2
        hill = h * np.exp(-0.5 * r2)
        z += hill.astype(np.float32)

        landmark_rigid = np.exp(-0.5 * (r2 / 1.6**2)).astype(np.float32)
        rigid_landmark_mask = np.maximum(rigid_landmark_mask, _clip01(landmark_rigid))

        landmarks.append(
            {
                "id": int(i),
                "cx": cx,
                "cy": cy,
                "height": h,
                "sigma": sigma,
            }
        )

    rigid_area = np.zeros_like(z, dtype=np.float32)
    rigid_pattern = "none"
    rigid_ratio = float(np.clip(cfg.rigid_ratio, 0.0, 1.0))
    if rigid_ratio > 0.0:
        u = (x_grid - x_min) / x_span
        v = (y_grid - y_min) / y_span

        rigid_pattern = str(rng.choice(["corner", "side", "band"]))
        if rigid_pattern == "corner":
            cx, cy = tuple(rng.choice(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])))
            sx = float(rng.uniform(0.18, 0.42))
            sy = float(rng.uniform(0.18, 0.42))
            d2 = ((u - cx) / max(sx, 1e-6)) ** 2 + ((v - cy) / max(sy, 1e-6)) ** 2
            field = np.exp(-0.5 * d2).astype(np.float32)
        elif rigid_pattern == "side":
            side = str(rng.choice(["left", "right", "top", "bottom"]))
            width = float(rng.uniform(0.14, 0.30))
            if side == "left":
                d = u
                mod = 0.75 + 0.25 * np.sin(3.0 * np.pi * v + float(rng.uniform(0.0, 2.0 * np.pi)))
            elif side == "right":
                d = 1.0 - u
                mod = 0.75 + 0.25 * np.sin(3.0 * np.pi * v + float(rng.uniform(0.0, 2.0 * np.pi)))
            elif side == "top":
                d = v
                mod = 0.75 + 0.25 * np.sin(3.0 * np.pi * u + float(rng.uniform(0.0, 2.0 * np.pi)))
            else:
                d = 1.0 - v
                mod = 0.75 + 0.25 * np.sin(3.0 * np.pi * u + float(rng.uniform(0.0, 2.0 * np.pi)))
            field = np.exp(-0.5 * (d / max(width, 1e-6)) ** 2).astype(np.float32) * mod.astype(np.float32)
        else:
            orientation = str(rng.choice(["horizontal", "vertical", "diag_pos", "diag_neg"]))
            width = float(rng.uniform(0.10, 0.22))
            offset = float(rng.uniform(-0.10, 0.10))
            if orientation == "horizontal":
                d = np.abs(v - (0.5 + offset))
            elif orientation == "vertical":
                d = np.abs(u - (0.5 + offset))
            elif orientation == "diag_pos":
                d = np.abs((v - u) - offset) / np.sqrt(2.0)
            else:
                d = np.abs((v - (1.0 - u)) - offset) / np.sqrt(2.0)
            field = np.exp(-0.5 * (d / max(width, 1e-6)) ** 2).astype(np.float32)
            rigid_pattern = f"band_{orientation}"

        if rigid_ratio >= 1.0:
            rigid_area = np.ones_like(z, dtype=np.float32)
        else:
            thr = float(np.quantile(field, 1.0 - rigid_ratio))
            rigid_area = (field >= thr).astype(np.float32)

    rigid = rigid_area
    if cfg.rigid_include_landmarks:
        rigid = np.maximum(rigid, (rigid_landmark_mask >= 0.24).astype(np.float32))

    if cfg.rigid_edge_softness > 0:
        rigid = _box_blur(rigid, passes=int(cfg.rigid_edge_softness))

    if cfg.rigid_include_landmarks:
        rigid = np.maximum(rigid, rigid_landmark_mask)

    rigid = _clip01(rigid)
    rigid_fraction = float(np.mean(rigid >= 0.5))
    return z.astype(np.float32), rigid.astype(np.float32), landmarks, rigid_fraction, rigid_pattern
