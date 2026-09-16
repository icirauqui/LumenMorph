import type { DatasetConfig, JobStatus, Preset, PreviewSettings, PreviewSessionState, TubeFeaPreview } from './types'

const jsonHeaders = { 'Content-Type': 'application/json' }

const makeUrl = (baseUrl: string, path: string) =>
  `${baseUrl.replace(/\/$/, '')}${path}`

const decodeJson = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`)
  }
  return (await response.json()) as T
}

export const previewCameraPayload = (settings: PreviewSettings) => ({
  yaw_deg: settings.yaw_deg,
  pitch_deg: settings.pitch_deg,
  distance_scale: settings.distance_scale,
  fov_deg: settings.fov_deg,
  width: settings.width,
  height: settings.height,
  jpeg_quality: settings.jpeg_quality,
  texture_enabled: settings.texture_enabled,
  background_bgr: [14, 14, 14],
})

export const wsPreviewUrl = (baseUrl: string, sessionId: string) => {
  const url = new URL(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}/stream`))
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

export async function getHealth(baseUrl: string) {
  return decodeJson<Record<string, unknown>>(await fetch(makeUrl(baseUrl, '/v1/health')))
}

export async function getSchema(baseUrl: string) {
  return decodeJson<Record<string, unknown>>(await fetch(makeUrl(baseUrl, '/v1/schema')))
}

export async function getPresets(baseUrl: string) {
  const payload = await decodeJson<{ presets: Preset[] }>(await fetch(makeUrl(baseUrl, '/v1/presets')))
  return payload.presets
}

export async function getOutputDefaults(baseUrl: string) {
  return decodeJson<{ repo_root: string; default_output_dir: string; exists: boolean }>(
    await fetch(makeUrl(baseUrl, '/v1/output/defaults')),
  )
}

export async function selectOutputDirectory(baseUrl: string) {
  return decodeJson<{ selected: boolean; output_dir: string }>(
    await fetch(makeUrl(baseUrl, '/v1/output/select-directory'), { method: 'POST' }),
  )
}

export async function normalizeConfig(baseUrl: string, config: DatasetConfig) {
  return decodeJson<{
    ok: boolean
    errors: string[]
    warnings: string[]
    normalized_config?: DatasetConfig
    config_hash?: string
  }>(
    await fetch(makeUrl(baseUrl, '/v1/config/normalize'), {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({ config }),
    }),
  )
}

export async function fetchTubeFeaPreview(
  baseUrl: string,
  config: DatasetConfig,
  longitudinalSamples: number,
  radialSamples: number,
) {
  return decodeJson<TubeFeaPreview>(
    await fetch(makeUrl(baseUrl, '/v1/tube/fea-preview'), {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({
        config,
        longitudinal_samples: longitudinalSamples,
        radial_samples: radialSamples,
      }),
    }),
  )
}

export async function createPreviewSession(
  baseUrl: string,
  config: DatasetConfig,
  settings: PreviewSettings,
  playing: boolean,
  tNorm: number,
) {
  return decodeJson<PreviewSessionState>(
    await fetch(makeUrl(baseUrl, '/v1/preview/sessions'), {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({
        config,
        preview_camera: previewCameraPayload(settings),
        playback: {
          playing,
          fps: settings.fps,
          frame_count: settings.frame_count,
          t_norm: tNorm,
          direction: 1,
        },
        draft_profile: true,
      }),
    }),
  )
}

export async function deletePreviewSession(baseUrl: string, sessionId: string) {
  await decodeJson<Record<string, unknown>>(
    await fetch(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}`), { method: 'DELETE' }),
  )
}

export async function patchPreviewConfig(baseUrl: string, sessionId: string, config: DatasetConfig) {
  return decodeJson<PreviewSessionState>(
    await fetch(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}/config`), {
      method: 'PATCH',
      headers: jsonHeaders,
      body: JSON.stringify({ config }),
    }),
  )
}

export async function patchPreviewCamera(
  baseUrl: string,
  sessionId: string,
  settings: PreviewSettings,
) {
  return decodeJson<PreviewSessionState>(
    await fetch(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}/camera`), {
      method: 'PATCH',
      headers: jsonHeaders,
      body: JSON.stringify(previewCameraPayload(settings)),
    }),
  )
}

export async function patchPreviewPlayback(
  baseUrl: string,
  sessionId: string,
  settings: PreviewSettings,
  playing: boolean,
  tNorm: number,
) {
  return decodeJson<PreviewSessionState>(
    await fetch(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}/playback`), {
      method: 'PATCH',
      headers: jsonHeaders,
      body: JSON.stringify({
        playing,
        fps: settings.fps,
        frame_count: settings.frame_count,
        t_norm: tNorm,
        direction: 1,
      }),
    }),
  )
}

export async function latestPreviewFrame(baseUrl: string, sessionId: string) {
  const response = await fetch(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}/frame/latest`))
  if (response.status === 202) {
    return { blob: null, meta: await response.json() }
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`)
  }
  return {
    blob: await response.blob(),
    meta: {
      status: response.headers.get('x-preview-status') ?? 'ok',
      seq: Number(response.headers.get('x-preview-seq') ?? 0),
      t_norm: Number(response.headers.get('x-preview-tnorm') ?? 0),
      render_ms: Number(response.headers.get('x-preview-render-ms') ?? 0),
      server_ts: Number(response.headers.get('x-preview-server-ts') ?? 0),
      playing: response.headers.get('x-preview-playing') === '1',
    },
  }
}

export async function startFullPreview(
  baseUrl: string,
  sessionId: string,
  settings: PreviewSettings,
) {
  const payload = await decodeJson<{ job_id: string }>(
    await fetch(makeUrl(baseUrl, `/v1/preview/sessions/${sessionId}/full-preview/jobs`), {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({
        frame_count: settings.frame_count,
        fps: settings.fps,
        preview_camera: previewCameraPayload(settings),
      }),
    }),
  )
  return payload.job_id
}

export async function startGenerateJob(
  baseUrl: string,
  config: DatasetConfig,
  outputDir: string,
  mode: string,
  quickMaxSequences: number,
  quickOverrideFrames?: number,
  quickOverrideFps?: number,
) {
  const payload = await decodeJson<{ job_id: string }>(
    await fetch(makeUrl(baseUrl, '/v1/generate/jobs'), {
      method: 'POST',
      headers: jsonHeaders,
      body: JSON.stringify({
        config,
        output_dir: outputDir.trim() || null,
        mode,
        quick: {
          max_sequences: quickMaxSequences,
          override_num_frames: quickOverrideFrames ?? null,
          override_fps: quickOverrideFps ?? null,
        },
        show_progress: false,
      }),
    }),
  )
  return payload.job_id
}

export async function getJob(baseUrl: string, jobId: string) {
  return decodeJson<JobStatus>(await fetch(makeUrl(baseUrl, `/v1/jobs/${jobId}`)))
}

export async function cancelJob(baseUrl: string, jobId: string) {
  await decodeJson<Record<string, unknown>>(
    await fetch(makeUrl(baseUrl, `/v1/jobs/${jobId}/cancel`), { method: 'POST' }),
  )
}

export async function getJobResult(baseUrl: string, jobId: string) {
  return decodeJson<Record<string, unknown>>(
    await fetch(makeUrl(baseUrl, `/v1/jobs/${jobId}/result`)),
  )
}

export async function getPreviewFrame(baseUrl: string, jobId: string, frameIndex: number) {
  const response = await fetch(makeUrl(baseUrl, `/v1/jobs/${jobId}/frames/${frameIndex}`))
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`)
  }
  return response.blob()
}
