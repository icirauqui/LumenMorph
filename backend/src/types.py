from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


DeformationFlowMode = Literal["linear", "cyclic"]
TubeSectionKind = Literal["straight", "turn"]
TubeTurnDirection = Literal["up", "down", "left", "right"]
TubeDeformationKind = Literal["localized", "fea_c3d6"]
TubeAreaMode = Literal["radial", "travelling"]


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class SequenceConfig:
    num_frames: int
    fps: int
    mesh_nx: int
    mesh_ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class WaveComponent:
    amp: float
    fx: float
    fy: float
    omega: float
    phase: float


@dataclass(frozen=True)
class LocalizedWaveComponent:
    amp: float
    fx: float
    fy: float
    omega: float
    phase: float
    center_s: float
    center_theta: float
    sigma_s: float
    sigma_theta: float
    amp_out: float | None = None
    amp_in: float | None = None
    transversal_only: bool = False
    flow_dir_s: float = 0.0
    flow_dir_theta: float = 0.0


@dataclass(frozen=True)
class DifficultyPreset:
    name: str
    wave_count: int
    amp_min: float
    amp_max: float
    freq_min: float
    freq_max: float
    omega_min: float
    omega_max: float
    drift_scale: float
    jitter_scale: float
    curve_scale: float = 1.0


@dataclass(frozen=True)
class CameraTrajectoryConfig:
    base_x: float
    base_y: float
    base_z: float
    forward_drift_per_s: float
    orbit_radius: float
    orbit_frequency_hz: float
    lateral_amplitude: float
    lateral_frequency_hz: float
    lookahead_distance: float
    lookat_height: float
    roll_jitter_deg: float
    pitch_jitter_deg: float
    yaw_jitter_deg: float


@dataclass(frozen=True)
class TextureConfig:
    width: int
    height: int
    min_orb_features: int
    max_attempts: int


@dataclass(frozen=True)
class PlaneSceneConfig:
    landmark_count: int
    landmark_height_min: float
    landmark_height_max: float
    landmark_sigma_min: float
    landmark_sigma_max: float
    rigid_ratio: float
    rigid_include_landmarks: bool
    rigid_edge_softness: int
    rigid_colorize: bool
    rigid_tint_bgr: list[int]
    rigid_tint_strength: float
    flip_vertical_view: bool


@dataclass(frozen=True)
class TubeAxisSection:
    kind: TubeSectionKind
    length: float = 4.0
    direction: TubeTurnDirection = "right"
    angle_deg: float = 45.0
    radius: float = 5.0


@dataclass(frozen=True)
class TubeAxisConfig:
    sections: List[TubeAxisSection]


@dataclass(frozen=True)
class TubeDimensionsConfig:
    radial_samples: int
    longitudinal_samples: int
    base_radius: float
    min_radius: float
    shell_thickness: float
    cap_ends: bool = False


@dataclass(frozen=True)
class TubeSurfaceConfig:
    bump_count: int
    bump_amp_min: float
    bump_amp_max: float
    bump_sigma_long: float
    bump_sigma_theta: float


@dataclass(frozen=True)
class TubeCameraConfig:
    offset_ratio: float
    wobble_ratio: float
    wobble_hz: float
    lookahead_distance: float
    end_margin_distance: float = 0.0


@dataclass(frozen=True)
class TubeDeformationDirection:
    radial: float = 1.0
    tangent: float = 0.0
    circumferential: float = 0.0


@dataclass(frozen=True)
class TubeDeformationArea:
    id: str
    enabled: bool
    center_s: float
    center_theta: float
    sigma_s: float
    sigma_theta: float
    kind: TubeDeformationKind = "localized"
    direction: TubeDeformationDirection = field(default_factory=TubeDeformationDirection)
    amplitude_out: float = 0.2
    amplitude_in: float = 0.2
    force_n: float = 800.0
    frequency_hz: float = 0.2
    phase_rad: float = 0.0
    mode: TubeAreaMode = "radial"
    spatial_cycles_s: float = 0.0
    spatial_cycles_theta: float = 0.0
    flow_dir_s: float = 0.0
    flow_dir_theta: float = 0.0


@dataclass(frozen=True)
class TubeFEAConfig:
    young_modulus_pa: float = 1.0e6
    poisson_ratio: float = 0.30
    axial_elements: int = 8
    theta_elements: int = 16
    # "point" applies each area's whole force to a single outer node (original
    # behaviour, kept so earlier datasets stay reproducible). "gaussian_patch"
    # spreads it over the area's configured sigma_s/sigma_theta footprint,
    # which removes the point-load strain singularity and makes the area
    # extent declared in the config actually affect the FEA solve.
    load_distribution: str = "point"
    load_sigma_cutoff: float = 3.0


@dataclass(frozen=True)
class TubeDeformationConfig:
    areas: List[TubeDeformationArea] = field(default_factory=list)
    fea: TubeFEAConfig = field(default_factory=TubeFEAConfig)
    flow_mode: DeformationFlowMode = "cyclic"
    flow_cycles: float = 1.0
    localized_tint_enabled: bool = False
    localized_tint_bgr: tuple[int, int, int] = (48, 185, 255)
    localized_tint_strength: float = 0.18


@dataclass(frozen=True)
class TubeSceneConfig:
    axis: TubeAxisConfig
    dimensions: TubeDimensionsConfig
    surface: TubeSurfaceConfig
    camera: TubeCameraConfig
    deformation: TubeDeformationConfig
    length: float
    radial_samples: int
    longitudinal_samples: int
    base_radius: float
    min_radius: float
    shell_thickness: float
    bump_count: int
    bump_amp_min: float
    bump_amp_max: float
    bump_sigma_long: float
    bump_sigma_theta: float
    curve_amplitude_x: float
    curve_amplitude_z: float
    curve_frequency_x: float
    curve_frequency_z: float
    camera_offset_ratio: float
    camera_wobble_ratio: float
    camera_wobble_hz: float
    lookahead_distance: float
    localized_deform_count: int = 0
    localized_longitudinal_min: float = 0.15
    localized_longitudinal_max: float = 0.85
    localized_theta_center: float = 0.25
    localized_theta_jitter: float = 0.08
    localized_sigma_long_min: float = 0.04
    localized_sigma_long_max: float = 0.10
    localized_sigma_theta_min: float = 0.05
    localized_sigma_theta_max: float = 0.12
    localized_amp_scale: float = 1.0
    localized_negative_ratio: float = 0.35
    localized_tint_enabled: bool = False
    localized_tint_bgr: tuple[int, int, int] = (48, 185, 255)
    localized_tint_strength: float = 0.18
    deformation_flow_mode: DeformationFlowMode = "cyclic"
    deformation_flow_cycles: float = 1.0


@dataclass(frozen=True)
class PhotometricConfig:
    gaussian_sigma: float
    gain_jitter: float


@dataclass(frozen=True)
class OutputToggles:
    write_frames: bool
    write_mp4: bool
    write_mesh_state: bool


@dataclass(frozen=True)
class ValidationConfig:
    min_orb_features_mean: float
    min_orb_features_frame: float
    min_frame_motion: float
    min_camera_clearance: float
    max_abs_mesh_z: float


@dataclass(frozen=True)
class DatasetConfig:
    version: str
    seed: int
    scene_type: str
    sequence_count: Dict[str, int]
    default_preset_cycle: List[str]
    intrinsics: CameraIntrinsics
    sequence: SequenceConfig
    camera: CameraTrajectoryConfig
    texture: TextureConfig
    photometric: PhotometricConfig
    output: OutputToggles
    validation: ValidationConfig
    plane: PlaneSceneConfig | None = None
    tube: TubeSceneConfig | None = None
    difficulty_presets: Dict[str, DifficultyPreset] = field(default_factory=dict)

    @property
    def total_sequences(self) -> int:
        return sum(self.sequence_count.values())


@dataclass(frozen=True)
class FrameState:
    frame_idx: int
    timestamp_s: float
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float
    global_amp: float
    global_freq: float
    phase_shift: float


@dataclass(frozen=True)
class SequenceDescriptor:
    split: str
    seq_idx: int
    seed: int
    preset: str


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]
