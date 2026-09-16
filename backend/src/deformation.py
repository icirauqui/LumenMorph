from __future__ import annotations

import numpy as np

from .types import DifficultyPreset, LocalizedWaveComponent, TubeSceneConfig, WaveComponent

_NON_TRANSVERSAL_RADIAL_MIX = 0.35


def sample_wave_components(rng: np.random.Generator, preset: DifficultyPreset) -> list[WaveComponent]:
    components: list[WaveComponent] = []
    for _ in range(preset.wave_count):
        amp = float(rng.uniform(preset.amp_min, preset.amp_max))
        freq = float(rng.uniform(preset.freq_min, preset.freq_max))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        fx = float(np.cos(theta) * freq)
        fy = float(np.sin(theta) * freq)
        omega = float(rng.uniform(preset.omega_min, preset.omega_max))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        components.append(WaveComponent(amp=amp, fx=fx, fy=fy, omega=omega, phase=phase))
    return components


def sample_localized_wave_components(
    rng: np.random.Generator,
    preset: DifficultyPreset,
    tube_cfg: TubeSceneConfig,
) -> list[LocalizedWaveComponent]:
    count = max(0, int(tube_cfg.localized_deform_count))
    if count == 0:
        return []

    s_min = float(np.clip(tube_cfg.localized_longitudinal_min, 0.0, 1.0))
    s_max = float(np.clip(tube_cfg.localized_longitudinal_max, 0.0, 1.0))
    if s_max < s_min:
        s_min, s_max = s_max, s_min

    theta_center = float(tube_cfg.localized_theta_center) % 1.0
    theta_jitter = float(np.clip(tube_cfg.localized_theta_jitter, 0.0, 0.5))

    sigma_s_min = float(max(tube_cfg.localized_sigma_long_min, 1e-4))
    sigma_s_max = float(max(tube_cfg.localized_sigma_long_max, sigma_s_min))
    sigma_t_min = float(max(tube_cfg.localized_sigma_theta_min, 1e-4))
    sigma_t_max = float(max(tube_cfg.localized_sigma_theta_max, sigma_t_min))

    amp_scale = float(max(tube_cfg.localized_amp_scale, 0.0))
    negative_ratio = float(np.clip(tube_cfg.localized_negative_ratio, 0.0, 1.0))

    components: list[LocalizedWaveComponent] = []
    for _ in range(count):
        amp = float(rng.uniform(preset.amp_min, preset.amp_max)) * amp_scale
        if float(rng.uniform(0.0, 1.0)) < negative_ratio:
            amp = -amp

        freq = float(rng.uniform(preset.freq_min, preset.freq_max))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        fx = float(np.cos(theta) * freq)
        fy = float(np.sin(theta) * freq)
        omega = float(rng.uniform(preset.omega_min, preset.omega_max))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        center_s = float(rng.uniform(s_min, s_max))
        center_theta = (theta_center + float(rng.uniform(-theta_jitter, theta_jitter))) % 1.0
        sigma_s = float(rng.uniform(sigma_s_min, sigma_s_max))
        sigma_theta = float(rng.uniform(sigma_t_min, sigma_t_max))
        flow_scale = float(max(0.0, 0.5 + 0.5 * preset.curve_scale))
        flow_dir_s = float(rng.uniform(-0.20, 0.20) * flow_scale)
        flow_dir_theta = float(rng.uniform(-0.20, 0.20) * flow_scale)

        components.append(
            LocalizedWaveComponent(
                amp=amp,
                fx=fx,
                fy=fy,
                omega=omega,
                phase=phase,
                center_s=center_s,
                center_theta=center_theta,
                sigma_s=sigma_s,
                sigma_theta=sigma_theta,
                flow_dir_s=flow_dir_s,
                flow_dir_theta=flow_dir_theta,
            )
        )
    return components


def compute_displacement(x_grid: np.ndarray, y_grid: np.ndarray, t_s: float, components: list[WaveComponent]) -> np.ndarray:
    disp = np.zeros_like(x_grid, dtype=np.float32)
    for c in components:
        arg = 2.0 * np.pi * (c.fx * x_grid + c.fy * y_grid) + c.omega * t_s + c.phase
        disp += (c.amp * np.sin(arg)).astype(np.float32)
    return disp


def compute_localized_displacement(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    t_s: float,
    components: list[LocalizedWaveComponent],
    flow_mode: str = "cyclic",
    flow_cycles: float = 1.0,
    t_norm: float | None = None,
) -> np.ndarray:
    disp = np.zeros_like(x_grid, dtype=np.float32)
    flow_progress = _flow_progress(
        t_norm=t_s if t_norm is None else t_norm,
        flow_mode=flow_mode,
        flow_cycles=flow_cycles,
    )
    for c in components:
        comp_progress = 0.0 if c.transversal_only else flow_progress
        envelope = _localized_envelope(x_grid, y_grid, c, progress=comp_progress)
        if c.transversal_only:
            arg = c.omega * t_s + c.phase
            signal = _localized_signal(c, arg)
        else:
            radial_arg = c.omega * t_s + c.phase
            spatial_arg = 2.0 * np.pi * (c.fx * x_grid + c.fy * y_grid) + c.omega * t_s + c.phase
            radial_signal = _localized_signal(c, radial_arg)
            spatial_signal = _localized_signal(c, spatial_arg)
            k = float(np.clip(_NON_TRANSVERSAL_RADIAL_MIX, 0.0, 1.0))
            signal = ((1.0 - k) * spatial_signal + k * radial_signal).astype(np.float32)
        disp += signal * envelope
    return disp


def compute_localized_weight(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    components: list[LocalizedWaveComponent],
    flow_mode: str = "cyclic",
    flow_cycles: float = 1.0,
    t_norm: float | None = None,
) -> np.ndarray:
    weight = np.zeros_like(x_grid, dtype=np.float32)
    flow_progress = _flow_progress(
        t_norm=0.0 if t_norm is None else t_norm,
        flow_mode=flow_mode,
        flow_cycles=flow_cycles,
    )
    for c in components:
        comp_progress = 0.0 if c.transversal_only else flow_progress
        weight = np.maximum(weight, _localized_envelope(x_grid, y_grid, c, progress=comp_progress))
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def _localized_envelope(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    component: LocalizedWaveComponent,
    progress: float = 0.0,
) -> np.ndarray:
    center_s_t = float(np.clip(component.center_s + component.flow_dir_s * progress, 0.0, 1.0))
    center_theta_t = float((component.center_theta + component.flow_dir_theta * progress) % 1.0)
    ds = y_grid - center_s_t
    dt = np.abs(x_grid - center_theta_t)
    dt = np.minimum(dt, 1.0 - dt)
    return np.exp(
        -0.5
        * ((ds / max(component.sigma_s, 1e-4)) ** 2 + (dt / max(component.sigma_theta, 1e-4)) ** 2)
    ).astype(np.float32)


def _localized_signal(component: LocalizedWaveComponent, arg: np.ndarray | float) -> np.ndarray:
    wave = np.asarray(np.sin(arg), dtype=np.float32)
    if component.amp_out is None and component.amp_in is None:
        return np.asarray(component.amp * wave, dtype=np.float32)

    amp_out = float(abs(component.amp_out if component.amp_out is not None else component.amp))
    amp_in = float(abs(component.amp_in if component.amp_in is not None else component.amp))
    out_part = amp_out * np.maximum(wave, 0.0)
    in_part = amp_in * np.minimum(wave, 0.0)
    return (out_part + in_part).astype(np.float32)


def _flow_progress(t_norm: float, flow_mode: str, flow_cycles: float) -> float:
    tn = float(np.clip(t_norm, 0.0, 1.0))
    mode = str(flow_mode).strip().lower()
    if mode == "linear":
        return tn
    cycles = float(max(flow_cycles, 0.0))
    return float(np.sin(2.0 * np.pi * cycles * tn))


def deformed_z(base_z: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, t_s: float, components: list[WaveComponent]) -> tuple[np.ndarray, float, float, float]:
    disp = compute_displacement(x_grid, y_grid, t_s, components)
    z = base_z + disp

    amps = np.asarray([c.amp for c in components], dtype=np.float32)
    freqs = np.asarray([np.hypot(c.fx, c.fy) for c in components], dtype=np.float32)
    phase_shift = float(np.mean([c.omega * t_s + c.phase for c in components])) if components else 0.0
    global_amp = float(np.mean(amps)) if amps.size else 0.0
    global_freq = float(np.mean(freqs)) if freqs.size else 0.0
    return z.astype(np.float32), global_amp, global_freq, phase_shift
