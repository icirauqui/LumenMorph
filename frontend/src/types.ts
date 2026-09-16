export type JsonObject = Record<string, unknown>

export type TubeTurnDirection = 'up' | 'down' | 'left' | 'right'
export type TubeSection =
  | { kind: 'straight'; length: number }
  | { kind: 'turn'; direction: TubeTurnDirection; angle_deg: number; radius: number }

export type TubeDeformationArea = {
  id: string
  enabled: boolean
  kind?: 'localized' | 'fea_c3d6'
  center_s: number
  center_theta: number
  sigma_s: number
  sigma_theta: number
  direction: {
    radial: number
    tangent: number
    circumferential: number
  }
  amplitude_out: number
  amplitude_in: number
  force_n?: number
  frequency_hz: number
  phase_rad: number
  mode: 'radial' | 'travelling'
  spatial_cycles_s: number
  spatial_cycles_theta: number
  flow_dir_s: number
  flow_dir_theta: number
}

export type TubeConfig = {
  axis: { sections: TubeSection[] }
  dimensions: {
    radial_samples: number
    longitudinal_samples: number
    base_radius: number
    min_radius: number
    shell_thickness: number
    cap_ends?: boolean
  }
  surface: {
    bump_count: number
    bump_amp_min: number
    bump_amp_max: number
    bump_sigma_long: number
    bump_sigma_theta: number
  }
  camera: {
    offset_ratio: number
    wobble_ratio: number
    wobble_hz: number
    lookahead_distance: number
    end_margin_distance?: number
  }
  deformation: {
    flow_mode: 'linear' | 'cyclic'
    flow_cycles: number
    fea?: {
      young_modulus_pa: number
      poisson_ratio: number
      axial_elements: number
      theta_elements: number
    }
    localized_tint_enabled: boolean
    localized_tint_bgr: number[]
    localized_tint_strength: number
    areas: TubeDeformationArea[]
  }
}

export type DatasetConfig = JsonObject & {
  scene_type?: string
  tube?: TubeConfig
  sequence?: { num_frames?: number; fps?: number }
  difficulty_presets?: Record<string, Record<string, number>>
}

export type Preset = {
  name: string
  path: string
  scene_type: string
  config: DatasetConfig
  warnings?: string[]
}

export type PreviewSettings = {
  yaw_deg: number
  pitch_deg: number
  distance_scale: number
  fov_deg: number
  width: number
  height: number
  jpeg_quality: number
  texture_enabled: boolean
  frame_count: number
  fps: number
}

export type JobStatus = {
  job_id: string
  kind: string
  status: string
  stage: string
  percent: number
  message: string
  error?: string | null
  result_ready: boolean
  frame_count: number
}

export type PreviewSessionState = {
  session_id: string
  status: string
  error?: string | null
  seq: number
  t_norm: number
  render_ms: number
  camera?: Record<string, unknown>
  playback?: Record<string, unknown>
}

export type TubeFeaPreviewArea = {
  area_id: string
  applied_s_norm: number
  applied_theta_norm: number
  applied_node?: number
  applied_axial_index?: number
  applied_theta_index?: number
  force_vector_local: number[]
  displacement_grid: number[][][]
}

export type TubeFeaPreview = {
  longitudinal_samples: number
  radial_samples: number
  responses: TubeFeaPreviewArea[]
}

export const defaultPreviewSettings: PreviewSettings = {
  yaw_deg: 42,
  pitch_deg: 18,
  distance_scale: 1.45,
  fov_deg: 55,
  width: 480,
  height: 270,
  jpeg_quality: 88,
  texture_enabled: false,
  frame_count: 90,
  fps: 24,
}

export const defaultArea = (index: number): TubeDeformationArea => ({
  id: `area_${index + 1}`,
  enabled: true,
  kind: 'localized',
  center_s: 0.5,
  center_theta: 0.2,
  sigma_s: 0.1,
  sigma_theta: 0.12,
  direction: { radial: 1, tangent: 0, circumferential: 0 },
  amplitude_out: 0.2,
  amplitude_in: 0.2,
  force_n: 800,
  frequency_hz: 0.2,
  phase_rad: 0,
  mode: 'radial',
  spatial_cycles_s: 0,
  spatial_cycles_theta: 0,
  flow_dir_s: 0,
  flow_dir_theta: 0,
})
