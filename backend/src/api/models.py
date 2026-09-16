from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..types import LocalizedWaveComponent


class LocalizedAreaRequest(BaseModel):
    amp: float = 0.2
    amp_out: float | None = None
    amp_in: float | None = None
    fx: float = 0.0
    fy: float = 0.0
    omega: float = 1.0
    phase: float = 0.0
    center_s: float = 0.5
    center_theta: float = 0.2
    sigma_s: float = 0.1
    sigma_theta: float = 0.1
    transversal_only: bool = True
    flow_dir_s: float = 0.0
    flow_dir_theta: float = 0.0

    def to_component(self) -> LocalizedWaveComponent:
        transversal_only = bool(self.transversal_only)
        return LocalizedWaveComponent(
            amp=float(self.amp),
            amp_out=float(self.amp_out) if self.amp_out is not None else None,
            amp_in=float(self.amp_in) if self.amp_in is not None else None,
            fx=float(self.fx),
            fy=float(self.fy),
            omega=float(self.omega),
            phase=float(self.phase),
            center_s=float(self.center_s),
            center_theta=float(self.center_theta),
            sigma_s=float(self.sigma_s),
            sigma_theta=float(self.sigma_theta),
            transversal_only=transversal_only,
            flow_dir_s=0.0 if transversal_only else float(self.flow_dir_s),
            flow_dir_theta=0.0 if transversal_only else float(self.flow_dir_theta),
        )


class PreviewCameraRequest(BaseModel):
    yaw_deg: float = 42.0
    pitch_deg: float = 18.0
    distance_scale: float = 1.45
    fov_deg: float = 55.0
    width: int = 384
    height: int = 216
    background_bgr: list[int] = Field(default_factory=lambda: [14, 14, 14])
    jpeg_quality: int = 82
    texture_enabled: bool = False


class PreviewJobRequest(BaseModel):
    config: dict[str, Any]
    frame_count: int = 90
    fps: int = 30
    preview_camera: PreviewCameraRequest = Field(default_factory=PreviewCameraRequest)
    localized_components: list[LocalizedAreaRequest] | None = None


class RealtimePlaybackRequest(BaseModel):
    playing: bool = True
    fps: int = 30
    direction: int = 1
    frame_count: int = 90
    t_norm: float = 0.0


class PreviewSessionCreateRequest(BaseModel):
    config: dict[str, Any]
    preview_camera: PreviewCameraRequest = Field(default_factory=PreviewCameraRequest)
    playback: RealtimePlaybackRequest = Field(default_factory=RealtimePlaybackRequest)
    localized_components: list[LocalizedAreaRequest] | None = None
    draft_profile: bool = True


class PreviewSessionConfigPatchRequest(BaseModel):
    config: dict[str, Any] | None = None
    localized_components: list[LocalizedAreaRequest] | None = None


class PreviewSessionCameraPatchRequest(BaseModel):
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    distance_scale: float | None = None
    fov_deg: float | None = None
    width: int | None = None
    height: int | None = None
    background_bgr: list[int] | None = None
    jpeg_quality: int | None = None
    texture_enabled: bool | None = None


class PreviewSessionPlaybackPatchRequest(BaseModel):
    playing: bool | None = None
    fps: int | None = None
    direction: int | None = None
    frame_count: int | None = None
    t_norm: float | None = None


class FullPreviewFromSessionRequest(BaseModel):
    frame_count: int = 90
    fps: int = 30
    preview_camera: PreviewCameraRequest | None = None
    localized_components: list[LocalizedAreaRequest] | None = None


class GenerateQuickOptions(BaseModel):
    max_sequences: int | None = None
    override_num_frames: int | None = None
    override_fps: int | None = None


class GenerateJobRequest(BaseModel):
    config: dict[str, Any]
    output_dir: str | None = None
    mode: Literal["quick", "full", "static_triplet"] = "quick"
    quick: GenerateQuickOptions = Field(default_factory=GenerateQuickOptions)
    show_progress: bool = False
    localized_components: list[LocalizedAreaRequest] | None = None


class NormalizeConfigRequest(BaseModel):
    config: dict[str, Any]


class TubeFeaPreviewRequest(BaseModel):
    config: dict[str, Any]
    longitudinal_samples: int | None = None
    radial_samples: int | None = None


def localized_components_or_none(
    items: list[LocalizedAreaRequest] | None,
) -> list[LocalizedWaveComponent] | None:
    if items is None:
        return None
    return [x.to_component() for x in items]
