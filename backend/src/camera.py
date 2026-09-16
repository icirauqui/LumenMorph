from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import CameraIntrinsics, CameraTrajectoryConfig, DifficultyPreset


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _rotation_from_lookat(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    right = _normalize(np.cross(up_hint, forward))
    if np.linalg.norm(right) < 1e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = _normalize(np.cross(forward, right))

    # Camera-to-world rotation. Columns are camera axes in world coords.
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _quat_from_rot(R: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


@dataclass
class CameraPose:
    t_wc: np.ndarray
    R_wc: np.ndarray
    q_xyzw: tuple[float, float, float, float]


class CameraTrajectory:
    def __init__(
        self,
        cfg: CameraTrajectoryConfig,
        preset: DifficultyPreset,
        rng: np.random.Generator,
    ) -> None:
        self.cfg = cfg
        self.preset = preset

        self.phase_orbit = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_lateral = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_roll = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_pitch = float(rng.uniform(0.0, 2.0 * np.pi))
        self.phase_yaw = float(rng.uniform(0.0, 2.0 * np.pi))

    def pose_at(self, t_s: float) -> CameraPose:
        c = self.cfg
        p = self.preset

        drift = c.forward_drift_per_s * p.drift_scale * t_s
        theta = 2.0 * np.pi * c.orbit_frequency_hz * t_s + self.phase_orbit

        x = (
            c.base_x
            + c.orbit_radius * np.cos(theta)
            + c.lateral_amplitude * np.sin(2.0 * np.pi * c.lateral_frequency_hz * t_s + self.phase_lateral)
        )
        y = c.base_y + drift + c.orbit_radius * np.sin(theta)
        z = c.base_z

        eye = np.array([x, y, z], dtype=np.float32)
        target = np.array([x, y + c.lookahead_distance, c.lookat_height], dtype=np.float32)

        R_wc = _rotation_from_lookat(eye, target, up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float32))

        roll = np.deg2rad(c.roll_jitter_deg * p.jitter_scale) * np.sin(0.7 * t_s + self.phase_roll)
        pitch = np.deg2rad(c.pitch_jitter_deg * p.jitter_scale) * np.sin(0.9 * t_s + self.phase_pitch)
        yaw = np.deg2rad(c.yaw_jitter_deg * p.jitter_scale) * np.sin(0.8 * t_s + self.phase_yaw)

        R_jitter = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
        R_wc = (R_wc @ R_jitter).astype(np.float32)

        q = _quat_from_rot(R_wc.astype(np.float64))
        return CameraPose(t_wc=eye, R_wc=R_wc, q_xyzw=q)


def world_to_camera(points_world: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    # p_c = R_cw * (p_w - t_wc), with R_cw = R_wc^T
    return ((R_wc.T @ (points_world - t_wc).T).T).astype(np.float32)


def project_points(points_cam: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = z > 1e-6

    uv = np.full((points_cam.shape[0], 2), -1.0, dtype=np.float32)
    if np.any(valid):
        x = points_cam[valid, 0] / z[valid]
        y = points_cam[valid, 1] / z[valid]
        uv[valid, 0] = intr.fx * x + intr.cx
        uv[valid, 1] = intr.cy - intr.fy * y
    return uv, valid
