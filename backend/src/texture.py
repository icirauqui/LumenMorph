from __future__ import annotations

import numpy as np

from .types import TextureConfig

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError("OpenCV (cv2) is required for texture generation") from exc


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    ptp = np.ptp(x)
    if ptp > 1e-8:
        x /= ptp
    return x


def _multiscale_noise(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    weight = 1.0
    total = 0.0
    for scale in [1, 2, 4, 8, 16]:
        hh = max(2, h // scale)
        ww = max(2, w // scale)
        base = rng.normal(0.0, 1.0, size=(hh, ww)).astype(np.float32)
        up = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
        acc += weight * up
        total += weight
        weight *= 0.5
    return _normalize01(acc / max(total, 1e-8))


def _feature_count(gray: np.ndarray) -> int:
    orb = cv2.ORB_create(nfeatures=1200)
    kps = orb.detect(gray, None)
    return len(kps)


def generate_procedural_texture(cfg: TextureConfig, rng: np.random.Generator) -> np.ndarray:
    h, w = cfg.height, cfg.width
    n0 = _multiscale_noise(h, w, rng)
    n1 = _multiscale_noise(h, w, rng)

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, h, dtype=np.float32),
        np.linspace(0.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    stripes = 0.5 + 0.5 * np.sin(42.0 * xx + 23.0 * yy + float(rng.uniform(0.0, 2.0 * np.pi)))
    checker = ((np.floor(xx * 28) + np.floor(yy * 18)) % 2).astype(np.float32)

    r = _normalize01(0.55 * n0 + 0.25 * stripes + 0.20 * checker)
    g = _normalize01(0.45 * n1 + 0.35 * (1.0 - stripes) + 0.20 * checker)
    b = _normalize01(0.65 * n0 + 0.20 * n1 + 0.15 * (1.0 - checker))

    tex = np.stack([b, g, r], axis=-1)
    tex = (255.0 * tex).astype(np.uint8)

    # Sharpen edges slightly to increase keypoint density.
    tex = cv2.GaussianBlur(tex, (0, 0), 0.8)
    tex = cv2.addWeighted(tex, 1.5, cv2.GaussianBlur(tex, (0, 0), 2.2), -0.5, 0)
    return tex


def generate_texture_with_qa(cfg: TextureConfig, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    best_img = None
    best_score = -1
    for _ in range(cfg.max_attempts):
        tex = generate_procedural_texture(cfg, rng)
        gray = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)
        score = _feature_count(gray)
        if score > best_score:
            best_img = tex
            best_score = score
        if score >= cfg.min_orb_features:
            return tex, score
    assert best_img is not None
    return best_img, best_score


def orb_feature_count(image_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return _feature_count(gray)
