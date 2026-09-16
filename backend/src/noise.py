from __future__ import annotations

import numpy as np

from .types import PhotometricConfig


def apply_photometric_noise(image_bgr: np.ndarray, rng: np.random.Generator, cfg: PhotometricConfig) -> np.ndarray:
    out = image_bgr.astype(np.float32)
    gain = float(rng.uniform(1.0 - cfg.gain_jitter, 1.0 + cfg.gain_jitter))
    out *= gain

    if cfg.gaussian_sigma > 0.0:
        noise = rng.normal(0.0, cfg.gaussian_sigma, size=out.shape).astype(np.float32)
        out += noise

    out = np.clip(out, 0.0, 255.0)
    return out.astype(np.uint8)
