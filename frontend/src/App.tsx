import {
  Activity,
  BookOpen,
  ChevronDown,
  ChevronRight,
  Download,
  FileUp,
  FolderOpen,
  PauseCircle,
  PlayCircle,
  Plus,
  RotateCcw,
  Save,
  Sparkles,
  Trash2,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react'
import { type ChangeEvent, type PointerEvent, type WheelEvent, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  cancelJob,
  createPreviewSession,
  deletePreviewSession,
  fetchTubeFeaPreview,
  getHealth,
  getJob,
  getJobResult,
  getOutputDefaults,
  getPresets,
  getPreviewFrame,
  latestPreviewFrame,
  normalizeConfig,
  patchPreviewCamera,
  patchPreviewConfig,
  patchPreviewPlayback,
  selectOutputDirectory,
  startFullPreview,
  startGenerateJob,
  wsPreviewUrl,
} from './api'
import { buildTubeDesignMesh, renderTubeDesignPreview, tubeDesignSampleCounts } from './designPreview'
import {
  applyViewportOrbit,
  applyViewportWheel,
  cameraFacingThetaFromYawDeg,
  clamp,
  cyclesForFrequency,
  frequencyForCycles,
  oppositeTheta,
  resetViewportCamera,
  sectionSummary,
  sequenceDurationSeconds,
  thetaFromCrossSectionPoint,
  tubeAxisLength,
} from './tubeMath'
import {
  defaultArea,
  defaultPreviewSettings,
  type DatasetConfig,
  type JobStatus,
  type Preset,
  type PreviewSettings,
  type TubeConfig,
  type TubeDeformationArea,
  type TubeFeaPreview,
  type TubeSection,
} from './types'

type Message = { tone: 'ok' | 'error' | 'info'; text: string } | null
type CachedFeaPreview = TubeFeaPreview & { signature: string }

const cloneConfig = (config: DatasetConfig): DatasetConfig => structuredClone(config)
const PREVIEW_MAX_PIXELS = 900_000
const VIEW_PRESETS = [
  { label: 'Top', title: 'View from above', yaw_deg: 0, pitch_deg: 85 },
  { label: 'Bottom', title: 'View from below', yaw_deg: 0, pitch_deg: -85 },
  { label: 'Left', title: 'View from the left', yaw_deg: 90, pitch_deg: 0 },
  { label: 'Right', title: 'View from the right', yaw_deg: -90, pitch_deg: 0 },
  { label: 'Front', title: 'View from the tube end', yaw_deg: 0, pitch_deg: 0 },
  { label: 'Persp', title: 'Perspective view', yaw_deg: 42, pitch_deg: 18 },
]

const isTubeConfig = (config: DatasetConfig | null): config is DatasetConfig & { tube: NonNullable<DatasetConfig['tube']> } =>
  config?.scene_type === 'tube' && Boolean(config.tube)

const defaultFeaConfig = () => ({
  young_modulus_pa: 1e6,
  poisson_ratio: 0.3,
  axial_elements: 8,
  theta_elements: 16,
})

const tubeHasFeaAreas = (tube: TubeConfig | null) =>
  Boolean(tube?.deformation.areas.some((area) => area.enabled && (area.kind ?? 'localized') === 'fea_c3d6'))

const maxFeaPreviewDisplacement = (preview: TubeFeaPreview | null | undefined) => {
  let maxValue = 0
  for (const response of preview?.responses ?? []) {
    for (const row of response.displacement_grid) {
      for (const point of row) {
        const norm = Math.hypot(Number(point[0] ?? 0), Number(point[1] ?? 0), Number(point[2] ?? 0))
        maxValue = Math.max(maxValue, norm)
      }
    }
  }
  return maxValue
}

const suggestedFeaViewportGain = (preview: TubeFeaPreview, tube: TubeConfig) => {
  const maxDisplacement = maxFeaPreviewDisplacement(preview)
  if (maxDisplacement <= 1e-12) return 1
  const targetDisplacement = Math.max(Number(tube.dimensions.base_radius), 0.01) * 0.08
  return clamp(targetDisplacement / maxDisplacement, 1, 1_000_000)
}

const formatLength = (value: number) => {
  const magnitude = Math.abs(value)
  if (magnitude === 0) return '0 m'
  if (magnitude < 1e-6) return `${(value * 1e9).toFixed(2)} nm`
  if (magnitude < 1e-3) return `${(value * 1e6).toFixed(2)} um`
  if (magnitude < 1) return `${(value * 1e3).toFixed(2)} mm`
  return `${value.toFixed(3)} m`
}

const sequenceGenerationDefaults = (config: DatasetConfig | null) => ({
  frames: Math.max(1, Math.round(Number(config?.sequence?.num_frames ?? defaultPreviewSettings.frame_count))),
  fps: Math.max(1, Math.round(Number(config?.sequence?.fps ?? defaultPreviewSettings.fps))),
})

const releasePointerCaptureSafe = (element: Element, pointerId: number) => {
  const pointerElement = element as Element & {
    hasPointerCapture?: (id: number) => boolean
    releasePointerCapture?: (id: number) => void
  }
  if (pointerElement.hasPointerCapture?.(pointerId)) pointerElement.releasePointerCapture?.(pointerId)
}

const previewRenderSizeForElement = (width: number, height: number) => {
  let nextWidth = Math.max(320, Math.round(width))
  let nextHeight = Math.max(180, Math.round(height))
  const pixels = nextWidth * nextHeight
  if (pixels > PREVIEW_MAX_PIXELS) {
    const scale = Math.sqrt(PREVIEW_MAX_PIXELS / pixels)
    nextWidth = Math.max(320, Math.round(nextWidth * scale))
    nextHeight = Math.max(180, Math.round(nextHeight * scale))
  }
  return {
    width: Math.max(2, nextWidth - (nextWidth % 2)),
    height: Math.max(2, nextHeight - (nextHeight % 2)),
  }
}

function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState('http://127.0.0.1:8765')
  const [connected, setConnected] = useState(false)
  const [initializing, setInitializing] = useState(true)
  const [presets, setPresets] = useState<Preset[]>([])
  const [selectedPreset, setSelectedPreset] = useState('')
  const [config, setConfig] = useState<DatasetConfig | null>(null)
  const [settings, setSettings] = useState<PreviewSettings>(defaultPreviewSettings)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessionStatus, setSessionStatus] = useState('idle')
  const [streamConnected, setStreamConnected] = useState(false)
  const [liveImageUrl, setLiveImageUrl] = useState<string | null>(null)
  const [liveSeq, setLiveSeq] = useState(0)
  const [liveTNorm, setLiveTNorm] = useState(0)
  const [liveRenderMs, setLiveRenderMs] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [viewMode, setViewMode] = useState<'live' | 'full'>('live')
  const [fullFrames, setFullFrames] = useState<string[]>([])
  const [fullFrameIndex, setFullFrameIndex] = useState(0)
  const [fullPlaying, setFullPlaying] = useState(true)
  const [selectedArea, setSelectedArea] = useState(0)
  const [previewJobId, setPreviewJobId] = useState<string | null>(null)
  const [previewJob, setPreviewJob] = useState<JobStatus | null>(null)
  const [generateJobId, setGenerateJobId] = useState<string | null>(null)
  const [generateJob, setGenerateJob] = useState<JobStatus | null>(null)
  const [generationMode, setGenerationMode] = useState('quick')
  const [outputDir, setOutputDir] = useState('')
  const [quickMaxSequences, setQuickMaxSequences] = useState(1)
  const [quickOverrideFrames, setQuickOverrideFrames] = useState(defaultPreviewSettings.frame_count)
  const [quickOverrideFps, setQuickOverrideFps] = useState(defaultPreviewSettings.fps)
  const [generationLocked, setGenerationLocked] = useState(false)
  const [jobsOpen, setJobsOpen] = useState(false)
  const [guideOpen, setGuideOpen] = useState(true)
  const [message, setMessage] = useState<Message>(null)
  const [feaPreview, setFeaPreview] = useState<CachedFeaPreview | null>(null)
  const [feaPreviewStatus, setFeaPreviewStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [feaPreviewError, setFeaPreviewError] = useState('')
  const [feaViewportGain, setFeaViewportGain] = useState(1)

  const wsRef = useRef<WebSocket | null>(null)
  const activeStreamIdRef = useRef<string | null>(null)
  const configPatchTimer = useRef<number | null>(null)
  const cameraPatchTimer = useRef<number | null>(null)
  const playbackPatchTimer = useRef<number | null>(null)
  const framePollTimer = useRef<number | null>(null)
  const jobPollTimer = useRef<number | null>(null)
  const designClockTimer = useRef<number | null>(null)
  const dragStart = useRef<{ x: number; y: number } | null>(null)
  const settingsRef = useRef(settings)
  const localDesignPreviewActiveRef = useRef(false)
  const previewJobIdRef = useRef<string | null>(null)
  const generateJobIdRef = useRef<string | null>(null)
  const pollJobsRef = useRef<() => Promise<void>>(async () => undefined)
  const generationOutputDirRef = useRef('')

  const tube = isTubeConfig(config) ? config.tube : null
  const areas = tube?.deformation.areas ?? []
  const hasFeaAreas = tubeHasFeaAreas(tube)
  const axisLength = useMemo(() => tubeAxisLength(tube?.axis.sections ?? []), [tube])
  const sequenceFrames = Math.max(1, Number(config?.sequence?.num_frames ?? settings.frame_count))
  const sequenceFps = Math.max(1, Number(config?.sequence?.fps ?? settings.fps))
  const previewDurationSeconds = sequenceDurationSeconds(sequenceFrames, sequenceFps)
  const feaPreviewSignature = useMemo(
    () => JSON.stringify({ tube, sequence: config?.sequence ?? null }),
    [tube, config?.sequence],
  )
  const activeFeaPreview = feaPreview?.signature === feaPreviewSignature ? feaPreview : null
  const activeFeaPreviewMaxDisplacement = useMemo(
    () => maxFeaPreviewDisplacement(activeFeaPreview),
    [activeFeaPreview],
  )
  const feaPreviewStale = hasFeaAreas && Boolean(feaPreview && feaPreview.signature !== feaPreviewSignature)
  const currentImage = viewMode === 'full' && fullFrames.length > 0 ? fullFrames[fullFrameIndex] : liveImageUrl
  const localDesignPreviewActive = viewMode === 'live' && Boolean(tube) && !settings.texture_enabled
  const jobsActive = generationLocked || previewJobId !== null || generateJobId !== null

  const clearTimers = () => {
    ;[configPatchTimer, cameraPatchTimer, playbackPatchTimer, framePollTimer, jobPollTimer, designClockTimer].forEach((timerRef) => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current)
      timerRef.current = null
    })
  }

  const revokeFrames = () => {
    if (liveImageUrl) URL.revokeObjectURL(liveImageUrl)
    fullFrames.forEach((url) => URL.revokeObjectURL(url))
  }

  const applySequenceDefaults = (nextConfig: DatasetConfig | null) => {
    const defaults = sequenceGenerationDefaults(nextConfig)
    setQuickOverrideFrames(defaults.frames)
    setQuickOverrideFps(defaults.fps)
    setSettings((old) => ({ ...old, frame_count: defaults.frames, fps: defaults.fps }))
  }

  const bootstrap = async () => {
    setInitializing(true)
    setMessage({ tone: 'info', text: 'Connecting to backend' })
    try {
      await getHealth(apiBaseUrl)
      const outputDefaults = await getOutputDefaults(apiBaseUrl)
      setOutputDir((current) => current.trim() || outputDefaults.default_output_dir)
      await getPresetsAndSelect()
      setConnected(true)
      setMessage({ tone: 'ok', text: 'Connected' })
    } catch (err) {
      setConnected(false)
      setMessage({ tone: 'error', text: String(err) })
    } finally {
      setInitializing(false)
    }
  }

  const getPresetsAndSelect = async () => {
    const loaded = await getPresets(apiBaseUrl)
    setPresets(loaded)
    const initial =
      loaded.find((p) => p.scene_type === 'tube' && p.name.includes('straight_localized_one_area')) ??
      loaded.find((p) => p.scene_type === 'tube') ??
      loaded[0]
    if (initial) {
      setSelectedPreset(initial.name)
      setConfig(initial.config)
      applySequenceDefaults(initial.config)
      setSelectedArea(0)
      await restartPreview(initial.config)
    }
  }

  const closeStream = () => {
    const socket = wsRef.current
    wsRef.current = null
    activeStreamIdRef.current = null
    if (framePollTimer.current !== null) window.clearInterval(framePollTimer.current)
    framePollTimer.current = null
    setStreamConnected(false)
    socket?.close()
  }

  const applySession = (payload: Record<string, unknown>) => {
    setSessionId(String(payload.session_id ?? sessionId ?? ''))
    setSessionStatus(String(payload.status ?? 'ok'))
    setLiveSeq(Number(payload.seq ?? liveSeq))
    if (!localDesignPreviewActiveRef.current) {
      setLiveTNorm(clamp(Number(payload.t_norm ?? liveTNorm), 0, 1))
    }
    setLiveRenderMs(Number(payload.render_ms ?? liveRenderMs))
  }

  const restartPreview = async (nextConfig = config) => {
    if (!nextConfig || generationLocked) return
    closeStream()
    if (sessionId) {
      try {
        await deletePreviewSession(apiBaseUrl, sessionId)
      } catch {
        // best effort cleanup
      }
    }
    setSessionStatus('connecting')
    const session = await createPreviewSession(apiBaseUrl, nextConfig, settings, playing, liveTNorm)
    applySession(session)
    openStream(session.session_id)
    setViewMode('live')
  }

  const openStream = (id: string) => {
    closeStream()
    activeStreamIdRef.current = id
    try {
      const socket = new WebSocket(wsPreviewUrl(apiBaseUrl, id))
      socket.binaryType = 'blob'
      wsRef.current = socket
      socket.onopen = () => {
        if (activeStreamIdRef.current !== id || wsRef.current !== socket) return
        setStreamConnected(true)
        setSessionStatus('ok')
      }
      socket.onmessage = (event) => {
        if (activeStreamIdRef.current !== id || wsRef.current !== socket) return
        if (typeof event.data === 'string') {
          applySession(JSON.parse(event.data))
          return
        }
        const blob = event.data instanceof Blob ? event.data : new Blob([event.data])
        const url = URL.createObjectURL(blob)
        setLiveImageUrl((old) => {
          if (old) URL.revokeObjectURL(old)
          return url
        })
      }
      socket.onerror = () => {
        if (activeStreamIdRef.current === id && wsRef.current === socket) startHttpPolling(id)
      }
      socket.onclose = () => {
        if (activeStreamIdRef.current === id && wsRef.current === socket) startHttpPolling(id)
      }
    } catch {
      if (activeStreamIdRef.current === id) startHttpPolling(id)
    }
  }

  const startHttpPolling = (id: string) => {
    if (activeStreamIdRef.current !== id) return
    wsRef.current = null
    setStreamConnected(true)
    setSessionStatus('polling')
    if (framePollTimer.current !== null) window.clearInterval(framePollTimer.current)
    const poll = async () => {
      if (activeStreamIdRef.current !== id) return
      try {
        const latest = await latestPreviewFrame(apiBaseUrl, id)
        if (activeStreamIdRef.current !== id) return
        applySession(latest.meta as Record<string, unknown>)
        if (latest.blob) {
          const url = URL.createObjectURL(latest.blob)
          setLiveImageUrl((old) => {
            if (old) URL.revokeObjectURL(old)
            return url
          })
        }
      } catch (err) {
        if (activeStreamIdRef.current !== id) return
        setSessionStatus('error')
        setMessage({ tone: 'error', text: String(err) })
      }
    }
    void poll()
    framePollTimer.current = window.setInterval(poll, 80)
  }

  const scheduleConfigPatch = (nextConfig: DatasetConfig) => {
    if (!connected || !sessionId || generationLocked) return
    if (configPatchTimer.current !== null) window.clearTimeout(configPatchTimer.current)
    configPatchTimer.current = window.setTimeout(async () => {
      try {
        applySession(await patchPreviewConfig(apiBaseUrl, sessionId, nextConfig))
      } catch (err) {
        setMessage({ tone: 'error', text: String(err) })
      }
    }, 60)
  }

  const scheduleCameraPatch = (nextSettings: PreviewSettings) => {
    if (!connected || !sessionId || generationLocked) return
    if (cameraPatchTimer.current !== null) window.clearTimeout(cameraPatchTimer.current)
    cameraPatchTimer.current = window.setTimeout(async () => {
      try {
        applySession(await patchPreviewCamera(apiBaseUrl, sessionId, nextSettings))
      } catch (err) {
        setMessage({ tone: 'error', text: String(err) })
      }
    }, 40)
  }

  const schedulePlaybackPatch = (nextPlaying = playing, nextTNorm = liveTNorm, nextSettings = settings) => {
    if (!connected || !sessionId || generationLocked) return
    if (playbackPatchTimer.current !== null) window.clearTimeout(playbackPatchTimer.current)
    playbackPatchTimer.current = window.setTimeout(async () => {
      try {
        applySession(await patchPreviewPlayback(apiBaseUrl, sessionId, nextSettings, nextPlaying, nextTNorm))
      } catch (err) {
        setMessage({ tone: 'error', text: String(err) })
      }
    }, 40)
  }

  const updateConfig = (recipe: (draft: DatasetConfig) => void) => {
    if (!config) return
    const next = cloneConfig(config)
    recipe(next)
    setConfig(next)
    scheduleConfigPatch(next)
  }

  const updateTube = (recipe: (draft: NonNullable<DatasetConfig['tube']>) => void) =>
    updateConfig((draft) => {
      if (draft.tube) recipe(draft.tube)
    })

  const updateSettings = (next: PreviewSettings) => {
    const wasTextureEnabled = settingsRef.current.texture_enabled
    const backendBackedLivePreview = hasFeaAreas || next.texture_enabled || wasTextureEnabled
    settingsRef.current = next
    setSettings(next)
    if (backendBackedLivePreview || viewMode === 'full') {
      scheduleCameraPatch(next)
      schedulePlaybackPatch(playing, liveTNorm, next)
    }
  }

  const refreshFeaRender = async () => {
    if (!config || !tube || !hasFeaAreas) return
    const signature = feaPreviewSignature
    const counts = tubeDesignSampleCounts(tube)
    setFeaPreviewStatus('loading')
    setFeaPreviewError('')
    try {
      const response = await fetchTubeFeaPreview(
        apiBaseUrl,
        config,
        counts.longitudinalSamples,
        counts.radialSamples,
      )
      setFeaPreview({ ...response, signature })
      setFeaViewportGain(suggestedFeaViewportGain(response, tube))
      setFeaPreviewStatus('ready')
      setMessage({
        tone: 'ok',
        text: `FEA render refreshed (${response.responses.length} response${response.responses.length === 1 ? '' : 's'})`,
      })
      setViewMode('live')
    } catch (err) {
      const text = String(err)
      setFeaPreviewStatus('error')
      setFeaPreviewError(text)
      setMessage({ tone: 'error', text: `FEA render refresh failed: ${text}` })
    }
  }

  const loadPreset = async (name: string) => {
    const preset = presets.find((item) => item.name === name)
    if (!preset) return
    setSelectedPreset(name)
    setConfig(preset.config)
    applySequenceDefaults(preset.config)
    setSelectedArea(0)
    await restartPreview(preset.config)
  }

  const importConfig = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      const raw = JSON.parse(await file.text()) as DatasetConfig
      const normalized = await normalizeConfig(apiBaseUrl, raw)
      if (!normalized.ok || !normalized.normalized_config) {
        throw new Error(normalized.errors.join(', '))
      }
      setConfig(normalized.normalized_config)
      applySequenceDefaults(normalized.normalized_config)
      setSelectedPreset('')
      setSelectedArea(0)
      setMessage({
        tone: normalized.warnings.length ? 'info' : 'ok',
        text: normalized.warnings[0] ?? `Imported ${file.name}`,
      })
      await restartPreview(normalized.normalized_config)
    } catch (err) {
      setMessage({ tone: 'error', text: `Import failed: ${String(err)}` })
    } finally {
      event.target.value = ''
    }
  }

  const exportConfig = () => {
    if (!config) return
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selectedPreset || 'tube_config'}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
  }

  const addSection = (kind: 'straight' | 'turn') =>
    updateTube((draft) => {
      draft.axis.sections.push(
        kind === 'straight'
          ? { kind: 'straight', length: 4 }
          : { kind: 'turn', direction: 'right', angle_deg: 45, radius: 5 },
      )
    })

  const updateSection = (index: number, next: TubeSection) =>
    updateTube((draft) => {
      draft.axis.sections[index] = next
    })

  const removeSection = (index: number) =>
    updateTube((draft) => {
      if (draft.axis.sections.length > 1) draft.axis.sections.splice(index, 1)
    })

  const addArea = () =>
    updateTube((draft) => {
      draft.deformation.areas.push(defaultArea(draft.deformation.areas.length))
      setSelectedArea(draft.deformation.areas.length - 1)
    })

  const updateArea = (index: number, recipe: (draft: TubeDeformationArea) => void) =>
    updateTube((draft) => {
      const area = draft.deformation.areas[index]
      if (!area) return
      recipe(area)
    })

  const removeArea = (index: number) =>
    updateTube((draft) => {
      draft.deformation.areas.splice(index, 1)
      setSelectedArea((old) => Math.max(0, Math.min(old, draft.deformation.areas.length - 1)))
    })

  const handleFullPreview = async () => {
    if (!sessionId) return
    setJobsOpen(true)
    const jobId = await startFullPreview(apiBaseUrl, sessionId, settings)
    previewJobIdRef.current = jobId
    setPreviewJobId(jobId)
    setPreviewJob({ job_id: jobId, kind: 'preview', status: 'running', stage: 'queued', percent: 0, message: 'Rendering full preview', result_ready: false, frame_count: 0 })
    ensureJobPolling()
  }

  const handleGenerate = async () => {
    setJobsOpen(true)
    if (!config) {
      setMessage({ tone: 'error', text: 'No config loaded' })
      return
    }
    setGenerationLocked(true)
    closeStream()
    try {
      const targetOutputDir = outputDir.trim()
      generationOutputDirRef.current = targetOutputDir || 'backend default output directory'
      const jobId = await startGenerateJob(
        apiBaseUrl,
        config,
        targetOutputDir,
        generationMode,
        quickMaxSequences,
        quickOverrideFrames,
        quickOverrideFps,
      )
      generateJobIdRef.current = jobId
      setGenerateJobId(jobId)
      setGenerateJob({ job_id: jobId, kind: 'generate', status: 'running', stage: 'queued', percent: 0, message: 'Generating dataset', result_ready: false, frame_count: 0 })
      ensureJobPolling()
    } catch (err) {
      setGenerationLocked(false)
      setMessage({ tone: 'error', text: String(err) })
      await restartPreview()
    }
  }

  const handleSelectOutputDir = async () => {
    setJobsOpen(true)
    try {
      const selected = await selectOutputDirectory(apiBaseUrl)
      if (selected.output_dir) setOutputDir(selected.output_dir)
      setMessage({
        tone: selected.selected ? 'ok' : 'info',
        text: selected.selected ? 'Output directory selected' : 'Directory selection canceled',
      })
    } catch (err) {
      setMessage({ tone: 'error', text: `Directory dialog unavailable: ${String(err)}` })
    }
  }

  const ensureJobPolling = () => {
    if (jobPollTimer.current !== null) return
    jobPollTimer.current = window.setInterval(() => void pollJobsRef.current(), 350)
  }

  const pollJobs = async () => {
    let active = false
    const activePreviewJobId = previewJobIdRef.current
    if (activePreviewJobId) {
      active = true
      const job = await getJob(apiBaseUrl, activePreviewJobId)
      setPreviewJob(job)
      if (job.status === 'succeeded') {
        const urls: string[] = []
        for (let i = 0; i < job.frame_count; i += 1) {
          urls.push(URL.createObjectURL(await getPreviewFrame(apiBaseUrl, activePreviewJobId, i)))
        }
        setFullFrames((old) => {
          old.forEach((url) => URL.revokeObjectURL(url))
          return urls
        })
        setFullFrameIndex(0)
        setViewMode('full')
        previewJobIdRef.current = null
        setPreviewJobId(null)
        setMessage({ tone: 'ok', text: `Full preview loaded (${urls.length} frames)` })
      } else if (['failed', 'canceled'].includes(job.status)) {
        previewJobIdRef.current = null
        setPreviewJobId(null)
      }
    }
    const activeGenerateJobId = generateJobIdRef.current
    if (activeGenerateJobId) {
      active = true
      const job = await getJob(apiBaseUrl, activeGenerateJobId)
      setGenerateJob(job)
      if (job.status === 'succeeded') {
        const result = await getJobResult(apiBaseUrl, activeGenerateJobId)
        generateJobIdRef.current = null
        setGenerateJobId(null)
        setGenerationLocked(false)
        setMessage({ tone: 'ok', text: `Generation completed: ${String((result.result as Record<string, unknown>)?.output_dir ?? generationOutputDirRef.current)}` })
        await restartPreview()
      } else if (['failed', 'canceled'].includes(job.status)) {
        generateJobIdRef.current = null
        setGenerateJobId(null)
        setGenerationLocked(false)
        setMessage({ tone: job.status === 'canceled' ? 'info' : 'error', text: `Generation ${job.status}` })
        await restartPreview()
      }
    }
    if (!active && jobPollTimer.current !== null) {
      window.clearInterval(jobPollTimer.current)
      jobPollTimer.current = null
    }
  }

  const onViewportPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const designViewportAvailable = Boolean(tube && !settings.texture_enabled && viewMode === 'live')
    const backendViewportAvailable = Boolean(tube && hasFeaAreas && viewMode === 'live')
    if ((!currentImage && !designViewportAvailable && !backendViewportAvailable) || !dragStart.current) return
    const dx = event.clientX - dragStart.current.x
    const dy = event.clientY - dragStart.current.y
    dragStart.current = { x: event.clientX, y: event.clientY }
    setViewMode('live')
    updateSettings(applyViewportOrbit(settingsRef.current, dx, dy))
  }

  useEffect(() => {
    settingsRef.current = settings
  }, [settings])

  useEffect(() => {
    localDesignPreviewActiveRef.current = localDesignPreviewActive
  }, [localDesignPreviewActive])

  useEffect(() => {
    if (designClockTimer.current !== null) {
      window.clearInterval(designClockTimer.current)
      designClockTimer.current = null
    }
    if (!localDesignPreviewActive || !playing) return
    let lastTick = window.performance.now()
    const frameSpan = Math.max(settings.frame_count - 1, 1)
    const fps = Math.max(settings.fps, 1)
    const delay = Math.max(16, Math.round(1000 / Math.min(fps, 30)))
    designClockTimer.current = window.setInterval(() => {
      const now = window.performance.now()
      const dt = Math.max(0, now - lastTick) / 1000
      lastTick = now
      setLiveTNorm((current) => (current + (dt * fps) / frameSpan) % 1)
    }, delay)
    return () => {
      if (designClockTimer.current !== null) {
        window.clearInterval(designClockTimer.current)
        designClockTimer.current = null
      }
    }
  }, [localDesignPreviewActive, playing, settings.fps, settings.frame_count])

  useEffect(() => {
    pollJobsRef.current = pollJobs
  })

  useEffect(() => {
    const startupTimer = window.setTimeout(() => void bootstrap(), 0)
    return () => {
      window.clearTimeout(startupTimer)
      closeStream()
      clearTimers()
      revokeFrames()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!fullPlaying || viewMode !== 'full' || fullFrames.length === 0) return
    const delay = Math.max(16, Math.round(1000 / Math.max(settings.fps, 1)))
    const timer = window.setInterval(() => {
      setFullFrameIndex((idx) => (idx + 1) % fullFrames.length)
    }, delay)
    return () => window.clearInterval(timer)
  }, [fullPlaying, fullFrames.length, settings.fps, viewMode])

  if (initializing) {
    return <main className="loading">Connecting to generator backend...</main>
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>LumenMorph</h1>
          <p>Deformable endoscopic sequence generator</p>
        </div>
        <div className="topbar-actions">
          <button type="button" className="topbar-action" onClick={() => setGuideOpen((open) => !open)} aria-pressed={guideOpen}>
            <BookOpen size={16} /> Guide
          </button>
          <button type="button" className={jobsActive ? 'topbar-action primary-action active' : 'topbar-action primary-action'} onClick={() => setJobsOpen(true)}>
            <Sparkles size={16} /> Generate dataset
            {jobsActive && <span className="job-dot" />}
          </button>
          <span className={`status ${connected ? 'ok' : 'error'}`} role="status">
            {connected ? <Wifi size={16} /> : <WifiOff size={16} />}
            {connected ? 'online' : 'offline'}
          </span>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel left-panel">
          <GettingStarted
            open={guideOpen}
            onToggle={() => setGuideOpen((open) => !open)}
            onGenerate={() => {
              setGenerationMode('quick')
              setJobsOpen(true)
            }}
          />
          <BackendPanel
            apiBaseUrl={apiBaseUrl}
            presets={presets}
            selectedPreset={selectedPreset}
            onApiBaseUrl={setApiBaseUrl}
            onReconnect={bootstrap}
            onPreset={loadPreset}
            onImport={importConfig}
            onExport={exportConfig}
          />
          {tube && (
            <>
              <TubeBasics tube={tube} axisLength={axisLength} onUpdate={updateTube} />
              <SectionBuilder
                sections={tube.axis.sections}
                onAdd={addSection}
                onRemove={removeSection}
                onUpdate={updateSection}
              />
            </>
          )}
        </aside>

        <section className="panel preview-panel">
          <PreviewPanel
            tube={tube}
            imageUrl={currentImage}
            viewMode={viewMode}
            fullFrames={fullFrames}
            fullFrameIndex={fullFrameIndex}
            status={sessionStatus}
            streamConnected={streamConnected}
            liveSeq={liveSeq}
            liveTNorm={liveTNorm}
            liveRenderMs={liveRenderMs}
            playing={viewMode === 'full' ? fullPlaying : playing}
            feaPreview={activeFeaPreview}
            feaPreviewStatus={feaPreviewStatus}
            feaPreviewStale={feaPreviewStale}
            feaPreviewMaxDisplacement={activeFeaPreviewMaxDisplacement}
            feaViewportGain={feaViewportGain}
            sequenceDurationSeconds={previewDurationSeconds}
            onRestart={() => void restartPreview()}
            onFullPreview={() => void handleFullPreview()}
            onCancelFull={() => previewJobId && void cancelJob(apiBaseUrl, previewJobId)}
            onViewMode={setViewMode}
            onTogglePlay={(next) => {
              if (viewMode === 'full') setFullPlaying(next)
              else {
                setPlaying(next)
                schedulePlaybackPatch(next, liveTNorm)
              }
            }}
            onScrub={(value) => {
              if (viewMode === 'full') setFullFrameIndex(Math.round(value))
              else {
                setLiveTNorm(value)
                schedulePlaybackPatch(playing, value)
              }
            }}
            onPointerDown={(event) => {
              dragStart.current = { x: event.clientX, y: event.clientY }
              event.currentTarget.setPointerCapture(event.pointerId)
            }}
            onPointerMove={onViewportPointerMove}
            onPointerUp={(event) => {
              dragStart.current = null
              releasePointerCaptureSafe(event.currentTarget, event.pointerId)
            }}
            onPointerCancel={(event) => {
              dragStart.current = null
              releasePointerCaptureSafe(event.currentTarget, event.pointerId)
            }}
            onWheel={(event) => {
              const designViewportAvailable = Boolean(tube && !settings.texture_enabled && viewMode === 'live')
              const backendViewportAvailable = Boolean(tube && hasFeaAreas && viewMode === 'live')
              if (!currentImage && !designViewportAvailable && !backendViewportAvailable) return
              event.preventDefault()
              setViewMode('live')
              updateSettings(applyViewportWheel(settingsRef.current, event.deltaY, event.shiftKey))
            }}
            onReset={() => updateSettings(resetViewportCamera(settingsRef.current))}
            settings={settings}
            onSettings={updateSettings}
          />
        </section>

        <aside className="panel right-panel">
          {tube && (
            <DeformationEditor
              tube={tube}
              areas={areas}
              selectedArea={selectedArea}
              settings={settings}
              sequenceFrames={sequenceFrames}
              sequenceFps={sequenceFps}
              previewTNorm={liveTNorm}
              feaPreviewStatus={feaPreviewStatus}
              feaPreviewStale={feaPreviewStale}
              feaPreviewError={feaPreviewError}
              hasFeaPreview={Boolean(activeFeaPreview)}
              feaPreview={activeFeaPreview}
              feaViewportGain={feaViewportGain}
              feaPreviewMaxDisplacement={activeFeaPreviewMaxDisplacement}
              onFeaViewportGain={setFeaViewportGain}
              onSelect={setSelectedArea}
              onAdd={addArea}
              onRemove={removeArea}
              onUpdate={updateArea}
              onInspectPhase={(tNorm) => {
                setViewMode('live')
                setPlaying(false)
                setLiveTNorm(tNorm)
                schedulePlaybackPatch(false, tNorm)
              }}
              onRefreshFeaRender={() => void refreshFeaRender()}
            />
          )}
        </aside>
      </section>

      {jobsOpen && (
        <JobsModal
          outputDir={outputDir}
          generationMode={generationMode}
          quickMaxSequences={quickMaxSequences}
          quickOverrideFrames={quickOverrideFrames}
          quickOverrideFps={quickOverrideFps}
          previewJob={previewJob}
          generateJob={generateJob}
          generationLocked={generationLocked}
          message={message}
          onOutputDir={setOutputDir}
          onSelectOutputDir={() => void handleSelectOutputDir()}
          onMode={setGenerationMode}
          onQuickMax={setQuickMaxSequences}
          onQuickFrames={setQuickOverrideFrames}
          onQuickFps={setQuickOverrideFps}
          onGenerate={() => void handleGenerate()}
          onCancelGenerate={() => generateJobId && void cancelJob(apiBaseUrl, generateJobId)}
          onClose={() => setJobsOpen(false)}
        />
      )}
    </main>
  )
}

function GettingStarted({ open, onToggle, onGenerate }: { open: boolean; onToggle: () => void; onGenerate: () => void }) {
  return (
    <section className="getting-started" aria-labelledby="getting-started-title">
      <button type="button" className="getting-started-heading" onClick={onToggle} aria-expanded={open}>
        <span className="getting-started-mark"><Sparkles size={16} /></span>
        <span>
          <strong id="getting-started-title">Start here</strong>
          <small>Design, preview, then export a synthetic sequence.</small>
        </span>
        {open ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
      </button>
      {open && (
        <div className="getting-started-body">
          <ol className="workflow-list">
            <li><span>1</span><div><strong>Choose a preset</strong><small>Start from the tube configuration already loaded below.</small></div></li>
            <li><span>2</span><div><strong>Adjust shape and motion</strong><small>Changes update the design preview as you work.</small></div></li>
            <li><span>3</span><div><strong>Export a dataset</strong><small>Begin with one quick sequence, then use full generation when ready.</small></div></li>
          </ol>
          <button type="button" className="primary-button getting-started-action" onClick={onGenerate}>
            <Sparkles size={16} /> Generate a quick sample
          </button>
        </div>
      )}
    </section>
  )
}

function BackendPanel(props: {
  apiBaseUrl: string
  presets: Preset[]
  selectedPreset: string
  onApiBaseUrl: (value: string) => void
  onReconnect: () => void
  onPreset: (name: string) => void
  onImport: (event: ChangeEvent<HTMLInputElement>) => void
  onExport: () => void
}) {
  return (
    <section className="section connection-section">
      <h2>Connection</h2>
      <label>
        API base URL
        <input value={props.apiBaseUrl} onChange={(event) => props.onApiBaseUrl(event.target.value)} />
      </label>
      <button type="button" className="wide-button" onClick={props.onReconnect}>
        <Activity size={16} /> Reconnect
      </button>
      <label>
        Tube preset
        <select value={props.selectedPreset} onChange={(event) => void props.onPreset(event.target.value)}>
          {props.presets
            .filter((preset) => preset.scene_type === 'tube')
            .map((preset) => (
              <option key={preset.name} value={preset.name}>
                {preset.name}
              </option>
          ))}
        </select>
      </label>
      <div className="action-grid">
        <label className="button file-button">
          <FileUp size={16} /> Import JSON
          <input type="file" accept="application/json,.json" onChange={props.onImport} />
        </label>
        <button type="button" onClick={props.onExport}>
          <Download size={16} /> Export JSON
        </button>
      </div>
    </section>
  )
}

function TubeBasics({
  tube,
  axisLength,
  onUpdate,
}: {
  tube: NonNullable<DatasetConfig['tube']>
  axisLength: number
  onUpdate: (recipe: (draft: NonNullable<DatasetConfig['tube']>) => void) => void
}) {
  const [collapsed, setCollapsed] = useState(false)
  const areaLabel = tube.deformation.areas.length === 1 ? 'area' : 'areas'
  const fea = tube.deformation.fea ?? defaultFeaConfig()
  const updateFea = (recipe: (draft: NonNullable<TubeConfig['deformation']['fea']>) => void) =>
    onUpdate((draft) => {
      draft.deformation.fea = { ...(draft.deformation.fea ?? defaultFeaConfig()) }
      recipe(draft.deformation.fea)
    })

  return (
    <section className="section tube-section">
      <button type="button" className="section-collapse-toggle" onClick={() => setCollapsed((value) => !value)}>
        {collapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
        <span>Tube</span>
        <small>{axisLength.toFixed(2)} length · {tube.deformation.areas.length} {areaLabel}</small>
      </button>
      {!collapsed && (
        <div className="collapsible-body">
          <div className="metric-grid">
            <Metric label="Axis length" value={axisLength.toFixed(2)} />
            <Metric label="Areas" value={String(tube.deformation.areas.length)} />
          </div>
          <NumberField label="Base radius" value={tube.dimensions.base_radius} onChange={(value) => onUpdate((draft) => void (draft.dimensions.base_radius = value))} />
          <NumberField label="Minimum radius" value={tube.dimensions.min_radius} onChange={(value) => onUpdate((draft) => void (draft.dimensions.min_radius = value))} />
          <NumberField label="Shell thickness" value={tube.dimensions.shell_thickness ?? 0.25} onChange={(value) => onUpdate((draft) => void (draft.dimensions.shell_thickness = value))} />
          <NumberField label="Radial samples" value={tube.dimensions.radial_samples} integer onChange={(value) => onUpdate((draft) => void (draft.dimensions.radial_samples = Math.round(value)))} />
          <NumberField label="Longitudinal samples" value={tube.dimensions.longitudinal_samples} integer onChange={(value) => onUpdate((draft) => void (draft.dimensions.longitudinal_samples = Math.round(value)))} />
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={tube.deformation.localized_tint_enabled}
              onChange={(event) => onUpdate((draft) => void (draft.deformation.localized_tint_enabled = event.target.checked))}
            />
            Tint deformation areas
          </label>
          <label>
            Flow mode
            <select
              value={tube.deformation.flow_mode}
              onChange={(event) => onUpdate((draft) => void (draft.deformation.flow_mode = event.target.value as 'linear' | 'cyclic'))}
            >
              <option value="cyclic">cyclic</option>
              <option value="linear">linear</option>
            </select>
          </label>
          <NumberField label="Flow cycles" value={tube.deformation.flow_cycles} onChange={(value) => onUpdate((draft) => void (draft.deformation.flow_cycles = value))} />
          <details open>
            <summary>FEA material and mesh</summary>
            <div className="grid-2">
              <NumberField label="Young modulus Pa" value={fea.young_modulus_pa} onChange={(value) => updateFea((draft) => void (draft.young_modulus_pa = Math.max(1, value)))} />
              <NumberField label="Poisson ratio" value={fea.poisson_ratio} onChange={(value) => updateFea((draft) => void (draft.poisson_ratio = clamp(value, -0.99, 0.489)))} />
              <NumberField label="Axial elements" value={fea.axial_elements} integer onChange={(value) => updateFea((draft) => void (draft.axial_elements = Math.max(2, Math.round(value))))} />
              <NumberField label="Theta elements" value={fea.theta_elements} integer onChange={(value) => updateFea((draft) => void (draft.theta_elements = Math.max(6, Math.round(value))))} />
            </div>
          </details>
        </div>
      )}
    </section>
  )
}

function SectionBuilder({
  sections,
  onAdd,
  onRemove,
  onUpdate,
}: {
  sections: TubeSection[]
  onAdd: (kind: 'straight' | 'turn') => void
  onRemove: (index: number) => void
  onUpdate: (index: number, next: TubeSection) => void
}) {
  const [collapsed, setCollapsed] = useState<Record<number, boolean>>({})

  return (
    <section className="section">
      <div className="section-heading">
        <h2>Axis Sections</h2>
        <div className="icon-row">
          <button type="button" title="Add straight section" onClick={() => onAdd('straight')}>
            <Plus size={16} /> Straight
          </button>
          <button type="button" title="Add turn section" onClick={() => onAdd('turn')}>
            <Plus size={16} /> Turn
          </button>
        </div>
      </div>
      <div className="stack">
        {sections.map((section, index) => (
          <div className="item section-card" key={`${section.kind}-${index}`}>
            <div className="item-title section-card-title">
              <button
                type="button"
                className="section-card-toggle"
                onClick={() => setCollapsed((old) => ({ ...old, [index]: !old[index] }))}
              >
                {collapsed[index] ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                <strong>{index + 1}. {sectionSummary(section)}</strong>
              </button>
              <button type="button" title="Remove section" onClick={() => onRemove(index)}>
                <Trash2 size={15} />
              </button>
            </div>
            {!collapsed[index] && (
              <div className="section-card-body">
                {section.kind === 'straight' ? (
                  <NumberField label="Length" value={section.length} onChange={(value) => onUpdate(index, { ...section, length: value })} />
                ) : (
                  <>
                    <label>
                      Direction
                      <select value={section.direction} onChange={(event) => onUpdate(index, { ...section, direction: event.target.value as 'up' | 'down' | 'left' | 'right' })}>
                        <option value="right">right</option>
                        <option value="left">left</option>
                        <option value="up">up</option>
                        <option value="down">down</option>
                      </select>
                    </label>
                    <NumberField label="Angle degrees" value={section.angle_deg} onChange={(value) => onUpdate(index, { ...section, angle_deg: value })} />
                    <NumberField label="Turn radius" value={section.radius} onChange={(value) => onUpdate(index, { ...section, radius: value })} />
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  )
}

function PreviewPanel(props: {
  tube: TubeConfig | null
  imageUrl: string | null
  viewMode: 'live' | 'full'
  fullFrames: string[]
  fullFrameIndex: number
  status: string
  streamConnected: boolean
  liveSeq: number
  liveTNorm: number
  liveRenderMs: number
  playing: boolean
  settings: PreviewSettings
  feaPreview: TubeFeaPreview | null
  feaPreviewStatus: 'idle' | 'loading' | 'ready' | 'error'
  feaPreviewStale: boolean
  feaPreviewMaxDisplacement: number
  feaViewportGain: number
  sequenceDurationSeconds: number
  onRestart: () => void
  onFullPreview: () => void
  onCancelFull: () => void
  onViewMode: (mode: 'live' | 'full') => void
  onTogglePlay: (playing: boolean) => void
  onScrub: (value: number) => void
  onPointerDown: (event: PointerEvent<HTMLDivElement>) => void
  onPointerMove: (event: PointerEvent<HTMLDivElement>) => void
  onPointerUp: (event: PointerEvent<HTMLDivElement>) => void
  onPointerCancel: (event: PointerEvent<HTMLDivElement>) => void
  onWheel: (event: WheelEvent<HTMLDivElement>) => void
  onReset: () => void
  onSettings: (settings: PreviewSettings) => void
}) {
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const latestSettings = useRef(props.settings)
  const latestOnSettings = useRef(props.onSettings)
  const maxScrub = props.viewMode === 'full' ? Math.max(props.fullFrames.length - 1, 1) : 1
  const scrubValue = props.viewMode === 'full' ? props.fullFrameIndex : props.liveTNorm
  const useDesignViewport = props.viewMode === 'live' && Boolean(props.tube) && !props.settings.texture_enabled
  const applyPreset = (preset: (typeof VIEW_PRESETS)[number]) => {
    props.onSettings({
      ...props.settings,
      yaw_deg: preset.yaw_deg,
      pitch_deg: preset.pitch_deg,
      distance_scale: preset.label === 'Persp' ? defaultPreviewSettings.distance_scale : props.settings.distance_scale,
      fov_deg: preset.label === 'Persp' ? defaultPreviewSettings.fov_deg : props.settings.fov_deg,
    })
  }

  useEffect(() => {
    latestSettings.current = props.settings
    latestOnSettings.current = props.onSettings
  })

  useEffect(() => {
    const element = viewportRef.current
    if (!element || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      const { width, height } = previewRenderSizeForElement(entry.contentRect.width, entry.contentRect.height)
      const current = latestSettings.current
      if (Math.abs(current.width - width) < 16 && Math.abs(current.height - height) < 16) return
      latestOnSettings.current({ ...current, width, height })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  return (
    <>
      <div className="preview-toolbar">
        <div>
          <h2>Design preview</h2>
          <p>{props.viewMode === 'full' ? `Full preview · frame ${props.fullFrameIndex + 1} of ${props.fullFrames.length}` : `Live preview · ${props.status}`}</p>
        </div>
        <div className="icon-row">
          <button type="button" title="Rebuild the live preview from the current configuration" onClick={props.onRestart}>
            <RotateCcw size={16} /> Restart
          </button>
          <button type="button" title="Render the complete preview sequence without writing a dataset" onClick={props.onFullPreview}>
            <Save size={16} /> Render full preview
          </button>
          <button type="button" className="quiet-button" title="Cancel the current full preview render" onClick={props.onCancelFull}>Cancel render</button>
        </div>
      </div>
      {props.fullFrames.length > 0 && (
        <div className="segmented">
          <button type="button" className={props.viewMode === 'live' ? 'active' : ''} onClick={() => props.onViewMode('live')}>Live</button>
          <button type="button" className={props.viewMode === 'full' ? 'active' : ''} onClick={() => props.onViewMode('full')}>Full</button>
        </div>
      )}
      <div
        className="viewport"
        ref={viewportRef}
        onPointerDown={props.onPointerDown}
        onPointerMove={props.onPointerMove}
        onPointerUp={props.onPointerUp}
        onPointerCancel={props.onPointerCancel}
        onWheel={props.onWheel}
        onDoubleClick={props.onReset}
      >
        <div className="viewport-view-controls" onPointerDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
          {VIEW_PRESETS.map((preset) => (
            <button type="button" key={preset.label} title={preset.title} onClick={() => applyPreset(preset)}>
              {preset.label}
            </button>
          ))}
          <button
            type="button"
            className={props.settings.texture_enabled ? 'active' : ''}
            title="Toggle preview texture. Dataset generation always uses texture."
            onClick={() => props.onSettings({ ...props.settings, texture_enabled: !props.settings.texture_enabled })}
          >
            Texture
          </button>
        </div>
        {useDesignViewport && props.tube ? (
          <DesignViewport
            tube={props.tube}
            settings={props.settings}
            tNorm={props.liveTNorm}
            feaPreview={props.feaPreview}
            sequenceDurationSeconds={props.sequenceDurationSeconds}
            feaDisplacementGain={props.feaViewportGain}
          />
        ) : props.imageUrl ? (
          <img src={props.imageUrl} alt="Tube preview" />
        ) : (
          <span>Waiting for preview frame...</span>
        )}
      </div>
      <div className="transport">
        <button type="button" title={props.playing ? 'Pause' : 'Play'} onClick={() => props.onTogglePlay(!props.playing)}>
          {props.playing ? <PauseCircle size={24} /> : <PlayCircle size={24} />}
        </button>
        <input type="range" min={0} max={maxScrub} step={props.viewMode === 'full' ? 1 : 0.001} value={scrubValue} onChange={(event) => props.onScrub(Number(event.target.value))} />
        <span>{props.viewMode === 'full' ? `${props.fullFrameIndex + 1}/${props.fullFrames.length}` : props.liveTNorm.toFixed(3)}</span>
      </div>
      <div className="telemetry">
        <span>
          {useDesignViewport
            ? tubeHasFeaAreas(props.tube)
              ? props.feaPreview
                ? props.feaPreviewStale
                  ? 'FEA render stale'
                  : 'FEA design viewport'
                : props.feaPreviewStatus === 'loading'
                  ? 'FEA render loading'
                  : 'FEA render not refreshed'
              : 'local design viewport'
            : props.streamConnected ? 'stream connected' : 'stream offline'}
        </span>
        <span>render {props.liveRenderMs.toFixed(1)} ms</span>
        <span>seq {props.liveSeq}</span>
        {props.feaPreview && <span>FEA max {formatLength(props.feaPreviewMaxDisplacement)} · view {props.feaViewportGain.toFixed(1)}x</span>}
      </div>
      <details>
        <summary>Camera</summary>
        <div className="grid-2">
          <NumberField label="Yaw" value={props.settings.yaw_deg} onChange={(value) => props.onSettings({ ...props.settings, yaw_deg: value })} />
          <NumberField label="Pitch" value={props.settings.pitch_deg} onChange={(value) => props.onSettings({ ...props.settings, pitch_deg: value })} />
          <NumberField label="Fit scale" value={props.settings.distance_scale} onChange={(value) => props.onSettings({ ...props.settings, distance_scale: value })} />
          <NumberField label="FOV" value={props.settings.fov_deg} onChange={(value) => props.onSettings({ ...props.settings, fov_deg: value })} />
          <NumberField label="Live width" value={props.settings.width} integer onChange={(value) => props.onSettings({ ...props.settings, width: Math.round(value) })} />
          <NumberField label="Live height" value={props.settings.height} integer onChange={(value) => props.onSettings({ ...props.settings, height: Math.round(value) })} />
          <label className="checkbox-row">
            <input type="checkbox" checked={props.settings.texture_enabled} onChange={(event) => props.onSettings({ ...props.settings, texture_enabled: event.target.checked })} />
            Preview texture
          </label>
          <NumberField label="Full frames" value={props.settings.frame_count} integer onChange={(value) => props.onSettings({ ...props.settings, frame_count: Math.round(value) })} />
          <NumberField label="FPS" value={props.settings.fps} integer onChange={(value) => props.onSettings({ ...props.settings, fps: Math.round(value) })} />
        </div>
      </details>
    </>
  )
}

function DesignViewport({
  tube,
  settings,
  tNorm,
  feaPreview,
  sequenceDurationSeconds,
  feaDisplacementGain,
}: {
  tube: TubeConfig
  settings: PreviewSettings
  tNorm: number
  feaPreview: TubeFeaPreview | null
  sequenceDurationSeconds: number
  feaDisplacementGain: number
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const mesh = useMemo(
    () => buildTubeDesignMesh(tube, tNorm, feaPreview, sequenceDurationSeconds, feaDisplacementGain),
    [tube, tNorm, feaPreview, sequenceDurationSeconds, feaDisplacementGain],
  )

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || typeof CanvasRenderingContext2D === 'undefined') return
    let frame = window.requestAnimationFrame(() => renderTubeDesignPreview(canvas, mesh, settings))
    const redraw = () => {
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => renderTubeDesignPreview(canvas, mesh, settings))
    }
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(redraw)
    observer?.observe(canvas)
    return () => {
      window.cancelAnimationFrame(frame)
      observer?.disconnect()
    }
  }, [mesh, settings])

  return <canvas ref={canvasRef} aria-label="Tube design preview" />
}

function DeformationEditor(props: {
  tube: TubeConfig
  areas: TubeDeformationArea[]
  selectedArea: number
  settings: PreviewSettings
  sequenceFrames: number
  sequenceFps: number
  previewTNorm: number
  feaPreviewStatus: 'idle' | 'loading' | 'ready' | 'error'
  feaPreviewStale: boolean
  feaPreviewError: string
  hasFeaPreview: boolean
  feaPreview: TubeFeaPreview | null
  feaViewportGain: number
  feaPreviewMaxDisplacement: number
  onFeaViewportGain: (value: number) => void
  onSelect: (index: number) => void
  onAdd: () => void
  onRemove: (index: number) => void
  onUpdate: (index: number, recipe: (draft: TubeDeformationArea) => void) => void
  onInspectPhase: (tNorm: number) => void
  onRefreshFeaRender: () => void
}) {
  const area = props.areas[props.selectedArea]
  return (
    <section className="section deformation-editor">
      <div className="section-heading">
        <h2>Deformation Areas</h2>
        <button type="button" onClick={props.onAdd}>
          <Plus size={16} /> Add
        </button>
      </div>
      <div className="chip-row">
        {props.areas.map((item, index) => (
          <button
            type="button"
            className={index === props.selectedArea ? 'chip active' : 'chip'}
            key={item.id}
            onClick={() => props.onSelect(index)}
          >
            {item.id || `Area ${index + 1}`}
          </button>
        ))}
      </div>
      {!area ? (
        <p>No localized areas yet.</p>
      ) : (
        <>
          <button type="button" className="danger" onClick={() => props.onRemove(props.selectedArea)}>
            <Trash2 size={16} /> Remove selected
          </button>
          <CrossSectionEditor
            theta={area.center_theta}
            cameraTheta={cameraFacingThetaFromYawDeg(props.settings.yaw_deg)}
            onTheta={(theta) => props.onUpdate(props.selectedArea, (draft) => void (draft.center_theta = theta))}
          />
          <MapEditor area={area} onArea={(recipe) => props.onUpdate(props.selectedArea, recipe)} />
          <AreaControls
            tube={props.tube}
            area={area}
            sequenceFrames={props.sequenceFrames}
            sequenceFps={props.sequenceFps}
            previewTNorm={props.previewTNorm}
            feaPreviewStatus={props.feaPreviewStatus}
            feaPreviewStale={props.feaPreviewStale}
            feaPreviewError={props.feaPreviewError}
            hasFeaPreview={props.hasFeaPreview}
            feaPreview={props.feaPreview}
            feaViewportGain={props.feaViewportGain}
            feaPreviewMaxDisplacement={props.feaPreviewMaxDisplacement}
            onFeaViewportGain={props.onFeaViewportGain}
            onArea={(recipe) => props.onUpdate(props.selectedArea, recipe)}
            onInspectPhase={props.onInspectPhase}
            onRefreshFeaRender={props.onRefreshFeaRender}
          />
        </>
      )}
    </section>
  )
}

function CrossSectionEditor({ theta, cameraTheta, onTheta }: { theta: number; cameraTheta: number; onTheta: (theta: number) => void }) {
  const [draftTheta, setDraftTheta] = useState(theta)
  const dragging = useRef(false)
  const thetaFromPointer = (event: PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return thetaFromCrossSectionPoint(event.clientX - rect.left, event.clientY - rect.top, rect.width, rect.height)
  }

  useEffect(() => {
    if (!dragging.current) setDraftTheta(theta)
  }, [theta])

  return (
    <div className="cross-section">
      <svg
        role="img"
        aria-label="Cross-section theta editor"
        viewBox="0 0 200 200"
        onPointerDown={(event) => {
          const next = thetaFromPointer(event)
          dragging.current = true
          setDraftTheta(next)
          event.currentTarget.setPointerCapture(event.pointerId)
        }}
        onPointerMove={(event) => {
          if (!dragging.current) return
          setDraftTheta(thetaFromPointer(event))
        }}
        onPointerUp={(event) => {
          if (!dragging.current) return
          const next = thetaFromPointer(event)
          dragging.current = false
          setDraftTheta(next)
          onTheta(next)
          releasePointerCaptureSafe(event.currentTarget, event.pointerId)
        }}
        onPointerCancel={(event) => {
          dragging.current = false
          setDraftTheta(theta)
          releasePointerCaptureSafe(event.currentTarget, event.pointerId)
        }}
      >
        <circle cx="100" cy="100" r="78" />
        <line x1="22" y1="100" x2="178" y2="100" />
        <line x1="100" y1="22" x2="100" y2="178" />
        <line x1="100" y1="100" x2={100 + Math.cos(draftTheta * Math.PI * 2) * 78} y2={100 - Math.sin(draftTheta * Math.PI * 2) * 78} className="marker-line" />
        <circle cx={100 + Math.cos(draftTheta * Math.PI * 2) * 78} cy={100 - Math.sin(draftTheta * Math.PI * 2) * 78} r="7" className="marker" />
      </svg>
      <Metric label="Theta" value={draftTheta.toFixed(3)} />
      <div className="button-row wrap">
        {[
          ['Right', 0],
          ['Top', 0.25],
          ['Left', 0.5],
          ['Bottom', 0.75],
          ['Camera', cameraTheta],
          ['Opposite', oppositeTheta(cameraTheta)],
        ].map(([label, value]) => (
          <button type="button" key={label} onClick={() => onTheta(Number(value))}>{label}</button>
        ))}
      </div>
    </div>
  )
}

function MapEditor({ area, onArea }: { area: TubeDeformationArea; onArea: (recipe: (draft: TubeDeformationArea) => void) => void }) {
  const dragging = useRef(false)
  const updateFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const theta = clamp((event.clientX - rect.left) / rect.width, 0, 1)
    const s = clamp(1 - (event.clientY - rect.top) / rect.height, 0, 1)
    onArea((draft) => {
      draft.center_theta = theta
      draft.center_s = s
    })
  }

  return (
    <div
      className="map-editor"
      onPointerDown={(event) => {
        dragging.current = true
        event.currentTarget.setPointerCapture(event.pointerId)
        updateFromPointer(event)
      }}
      onPointerMove={(event) => {
        if (dragging.current) updateFromPointer(event)
      }}
      onPointerUp={(event) => {
        if (!dragging.current) return
        updateFromPointer(event)
        dragging.current = false
        releasePointerCaptureSafe(event.currentTarget, event.pointerId)
      }}
      onPointerCancel={(event) => {
        dragging.current = false
        releasePointerCaptureSafe(event.currentTarget, event.pointerId)
      }}
    >
      <div
        className="area-ellipse"
        style={{
          left: `${area.center_theta * 100}%`,
          top: `${(1 - area.center_s) * 100}%`,
          width: `${area.sigma_theta * 200}%`,
          height: `${area.sigma_s * 200}%`,
        }}
      />
      <span className="axis-label top">s 1.0</span>
      <span className="axis-label bottom">theta wraps 0..1</span>
    </div>
  )
}

function AreaControls({
  tube,
  area,
  sequenceFrames,
  sequenceFps,
  previewTNorm,
  feaPreviewStatus,
  feaPreviewStale,
  feaPreviewError,
  hasFeaPreview,
  feaPreview,
  feaViewportGain,
  feaPreviewMaxDisplacement,
  onFeaViewportGain,
  onArea,
  onInspectPhase,
  onRefreshFeaRender,
}: {
  tube: TubeConfig
  area: TubeDeformationArea
  sequenceFrames: number
  sequenceFps: number
  previewTNorm: number
  feaPreviewStatus: 'idle' | 'loading' | 'ready' | 'error'
  feaPreviewStale: boolean
  feaPreviewError: string
  hasFeaPreview: boolean
  feaPreview: TubeFeaPreview | null
  feaViewportGain: number
  feaPreviewMaxDisplacement: number
  onFeaViewportGain: (value: number) => void
  onArea: (recipe: (draft: TubeDeformationArea) => void) => void
  onInspectPhase: (tNorm: number) => void
  onRefreshFeaRender: () => void
}) {
  const areaKind = area.kind ?? 'localized'
  const cycles = cyclesForFrequency(area.frequency_hz, sequenceFrames, sequenceFps)
  const feaAreaResponse = feaPreview?.responses.find((response) => response.area_id === area.id) ?? null
  const baseRadius = Math.max(Number(tube.dimensions.base_radius), 0.01)
  const minRadius = Math.max(Number(tube.dimensions.min_radius), 0)
  const inwardRoom = Math.max(baseRadius - minRadius, baseRadius * 0.5, 0.25)
  const outwardMax = Math.max(baseRadius * 2, 1, area.amplitude_out)
  const inwardMax = Math.max(inwardRoom, 1, area.amplitude_in)
  const pureRadial = Math.abs(area.direction.radial) > 1e-9 && Math.abs(area.direction.tangent) <= 1e-9 && Math.abs(area.direction.circumferential) <= 1e-9
  const positiveDirection = area.direction.radial >= 0 ? 'outward' : 'inward'
  const negativeDirection = positiveDirection === 'outward' ? 'inward' : 'outward'
  const positiveAmplitudeLabel = pureRadial ? `Peak ${positiveDirection} amplitude` : 'Positive-phase amplitude'
  const negativeAmplitudeLabel = pureRadial ? `Peak ${negativeDirection} amplitude` : 'Negative-phase amplitude'
  const phase = Math.sin(Math.PI * 2 * cycles * previewTNorm + area.phase_rad)
  const activeAmplitude = phase >= 0 ? area.amplitude_out : area.amplitude_in
  const activeDirection = phase >= 0 ? positiveDirection : negativeDirection
  const peakTime = (positive: boolean) => {
    if (cycles <= 1e-9) return null
    const target = positive ? Math.PI / 2 : (3 * Math.PI) / 2
    let best: number | null = null
    for (let turn = -12; turn <= 12; turn += 1) {
      const candidate = (target + 2 * Math.PI * turn - area.phase_rad) / (2 * Math.PI * cycles)
      if (candidate >= 0 && candidate <= 1 && (best === null || Math.abs(candidate - previewTNorm) < Math.abs(best - previewTNorm))) best = candidate
    }
    return best
  }
  const positivePeakTime = peakTime(true)
  const negativePeakTime = peakTime(false)
  return (
    <div className="stack">
      <label className="checkbox-row">
        <input type="checkbox" checked={area.enabled} onChange={(event) => onArea((draft) => void (draft.enabled = event.target.checked))} />
        Enabled
      </label>
      <label>
        Deformation kind
        <select
          value={areaKind}
          onChange={(event) => onArea((draft) => {
            draft.kind = event.target.value as NonNullable<TubeDeformationArea['kind']>
            if (draft.kind === 'fea_c3d6') {
              draft.force_n = Math.max(Number(draft.force_n ?? 800), 0)
              draft.direction = { radial: draft.direction.radial < 0 ? -1 : 1, tangent: 0, circumferential: 0 }
              draft.mode = 'radial'
              draft.spatial_cycles_s = 0
              draft.spatial_cycles_theta = 0
              draft.flow_dir_s = 0
              draft.flow_dir_theta = 0
            }
          })}
        >
          <option value="localized">Localized</option>
          <option value="fea_c3d6">FEA C3D6</option>
        </select>
      </label>
      <RangeNumberField label="Longitudinal position" value={area.center_s} min={0} max={1} step={0.001} onChange={(value) => onArea((draft) => void (draft.center_s = clamp(value, 0, 1)))} />
      {areaKind === 'localized' ? (
        <>
          <div className="grid-2">
            <RangeNumberField label="Area length" value={area.sigma_s} min={0.01} max={0.5} step={0.005} onChange={(value) => onArea((draft) => void (draft.sigma_s = clamp(value, 0.01, 0.5)))} />
            <RangeNumberField label="Area width" value={area.sigma_theta} min={0.01} max={0.5} step={0.005} onChange={(value) => onArea((draft) => void (draft.sigma_theta = clamp(value, 0.01, 0.5)))} />
          </div>
          <div className="grid-2">
            <RangeNumberField label={positiveAmplitudeLabel} value={area.amplitude_out} min={0} max={outwardMax} step={0.01} onChange={(value) => onArea((draft) => void (draft.amplitude_out = Math.max(0, value)))} />
            <RangeNumberField label={negativeAmplitudeLabel} value={area.amplitude_in} min={0} max={inwardMax} step={0.01} onChange={(value) => onArea((draft) => void (draft.amplitude_in = Math.max(0, value)))} />
          </div>
          <small className="field-note">
            {pureRadial
              ? `Preview is currently in the ${activeDirection} phase and uses ${formatLength(activeAmplitude)}.`
              : `Preview is currently in the ${phase >= 0 ? 'positive' : 'negative'} waveform phase and uses ${formatLength(activeAmplitude)}.`}
          </small>
          <div className="button-row wrap">
            <button type="button" className="quiet-button" disabled={positivePeakTime === null} onClick={() => positivePeakTime !== null && onInspectPhase(positivePeakTime)}>
              Inspect {pureRadial ? positiveDirection : 'positive'} peak
            </button>
            <button type="button" className="quiet-button" disabled={negativePeakTime === null} onClick={() => negativePeakTime !== null && onInspectPhase(negativePeakTime)}>
              Inspect {pureRadial ? negativeDirection : 'negative'} peak
            </button>
          </div>
        </>
      ) : (
        <>
          <RangeNumberField label="Force N" value={Number(area.force_n ?? 800)} min={0} max={Math.max(Number(area.force_n ?? 800), 5000)} step={10} onChange={(value) => onArea((draft) => void (draft.force_n = Math.max(0, value)))} />
          <label>
            Force direction
            <select
              value={area.direction.radial >= 0 ? 'outward' : 'inward'}
              onChange={(event) => onArea((draft) => {
                draft.direction = {
                  radial: event.target.value === 'inward' ? -1 : 1,
                  tangent: 0,
                  circumferential: 0,
                }
              })}
            >
              <option value="outward">radially outward</option>
              <option value="inward">radially inward</option>
            </select>
          </label>
          <div className="button-row wrap">
            <button type="button" onClick={onRefreshFeaRender} disabled={feaPreviewStatus === 'loading'}>
              <RotateCcw size={16} /> {feaPreviewStatus === 'loading' ? 'Refreshing FEA render' : 'Refresh FEA render'}
            </button>
          </div>
          {hasFeaPreview && (
            <div className="metric-grid">
              <Metric label="Actual max displacement" value={formatLength(feaPreviewMaxDisplacement)} />
              <Metric label="Viewport gain" value={`${feaViewportGain.toFixed(1)}x`} />
              {feaAreaResponse && (
                <>
                  <Metric label="Applied node" value={String(feaAreaResponse.applied_node ?? 'n/a')} />
                  <Metric
                    label="Applied indices"
                    value={`${feaAreaResponse.applied_axial_index ?? 'n/a'}, ${feaAreaResponse.applied_theta_index ?? 'n/a'}`}
                  />
                </>
              )}
            </div>
          )}
          {hasFeaPreview && (
            <NumberField
              label="FEA viewport gain"
              value={feaViewportGain}
              min={1}
              max={1000000}
              onChange={(value) => onFeaViewportGain(clamp(value, 1, 1000000))}
            />
          )}
          <small className={feaPreviewStatus === 'error' ? 'field-note error' : 'field-note'}>
            {feaPreviewStatus === 'error'
              ? feaPreviewError || 'FEA render failed'
              : feaPreviewStale
                ? 'FEA render is stale after edits'
                : hasFeaPreview
                  ? 'FEA render cache ready. Viewport gain affects only this editor view; exported displacements stay physical.'
                  : 'Refresh once to animate the physical deformation in the mesh viewport'}
          </small>
        </>
      )}
      <RangeNumberField label="Frequency Hz" value={area.frequency_hz} min={0} max={5} step={0.01} onChange={(value) => onArea((draft) => void (draft.frequency_hz = Math.max(0, value)))} />
      <RangeNumberField label={`Cycles over sequence (${cycles.toFixed(2)})`} value={cycles} min={0} max={10} step={0.01} onChange={(value) => onArea((draft) => void (draft.frequency_hz = frequencyForCycles(value, sequenceFrames, sequenceFps)))} />
      <RangeNumberField label="Phase radians" value={area.phase_rad} min={-Math.PI} max={Math.PI} step={0.01} onChange={(value) => onArea((draft) => void (draft.phase_rad = value))} />
      {areaKind === 'localized' && (
        <label>
          Mode
          <select value={area.mode} onChange={(event) => onArea((draft) => void (draft.mode = event.target.value as TubeDeformationArea['mode']))}>
            <option value="radial">radial</option>
            <option value="travelling">travelling</option>
          </select>
        </label>
      )}
      {areaKind === 'localized' && (
        <>
          <div className="button-row wrap">
            {[
              ['Outward', { radial: 1, tangent: 0, circumferential: 0 }],
              ['Inward', { radial: -1, tangent: 0, circumferential: 0 }],
              ['Forward', { radial: 0, tangent: 1, circumferential: 0 }],
              ['Backward', { radial: 0, tangent: -1, circumferential: 0 }],
              ['Around +', { radial: 0, tangent: 0, circumferential: 1 }],
              ['Around -', { radial: 0, tangent: 0, circumferential: -1 }],
            ].map(([label, direction]) => (
              <button
                type="button"
                key={String(label)}
                onClick={() => onArea((draft) => {
                  draft.direction = direction as TubeDeformationArea['direction']
                })}
              >
                {String(label)}
              </button>
            ))}
          </div>
          <div className="grid-3">
            <NumberField label="Radial dir" value={area.direction.radial} onChange={(value) => onArea((draft) => void (draft.direction.radial = value))} />
            <NumberField label="Tangent dir" value={area.direction.tangent} onChange={(value) => onArea((draft) => void (draft.direction.tangent = value))} />
            <NumberField label="Circ dir" value={area.direction.circumferential} onChange={(value) => onArea((draft) => void (draft.direction.circumferential = value))} />
          </div>
        </>
      )}
      {areaKind === 'localized' && area.mode === 'travelling' && (
        <div className="grid-2">
          <NumberField label="Spatial cycles s" value={area.spatial_cycles_s} onChange={(value) => onArea((draft) => void (draft.spatial_cycles_s = value))} />
          <NumberField label="Spatial cycles theta" value={area.spatial_cycles_theta} onChange={(value) => onArea((draft) => void (draft.spatial_cycles_theta = value))} />
          <NumberField label="Flow s" value={area.flow_dir_s} onChange={(value) => onArea((draft) => void (draft.flow_dir_s = value))} />
          <NumberField label="Flow theta" value={area.flow_dir_theta} onChange={(value) => onArea((draft) => void (draft.flow_dir_theta = value))} />
        </div>
      )}
    </div>
  )
}

type JobPanelProps = {
  outputDir: string
  generationMode: string
  quickMaxSequences: number
  quickOverrideFrames: number
  quickOverrideFps: number
  previewJob: JobStatus | null
  generateJob: JobStatus | null
  generationLocked: boolean
  message: Message
  onOutputDir: (value: string) => void
  onSelectOutputDir: () => void
  onMode: (value: string) => void
  onQuickMax: (value: number) => void
  onQuickFrames: (value: number) => void
  onQuickFps: (value: number) => void
  onGenerate: () => void
  onCancelGenerate: () => void
}

function JobsModal({ onClose, ...panelProps }: JobPanelProps & { onClose: () => void }) {
  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="jobs-modal" role="dialog" aria-modal="true" aria-labelledby="jobs-title">
        <div className="modal-header">
          <div>
            <h2 id="jobs-title">Generate dataset</h2>
            <p>Review the output and generation scope before starting.</p>
          </div>
          <button type="button" title="Close generation panel" onClick={onClose}>
            <X size={16} /> Close
          </button>
        </div>
        <JobPanel {...panelProps} embedded />
      </div>
    </div>
  )
}

function JobPanel(props: JobPanelProps & { embedded?: boolean }) {
  return (
    <section className={props.embedded ? 'jobs-panel embedded' : 'section jobs-panel'}>
      {!props.embedded && <h2>Generate dataset</h2>}
      <label>
        Generation scope
        <select value={props.generationMode} onChange={(event) => props.onMode(event.target.value)}>
          <option value="quick">Quick sample</option>
          <option value="full">Full configuration</option>
          <option value="static_triplet">Static control triplet</option>
        </select>
      </label>
      <p className="mode-explainer">
        {props.generationMode === 'quick'
          ? 'Creates a small, editable test run. This is the recommended first export.'
          : props.generationMode === 'full'
            ? 'Uses the full selected configuration and may take substantially longer.'
            : 'Creates matched rigid, inward and outward static variants from one tube setup.'}
      </p>
      <label>
        Output directory
        <div className="output-dir-row">
          <input value={props.outputDir} onChange={(event) => props.onOutputDir(event.target.value)} placeholder="outputs" />
          <button type="button" title="Select output directory on the backend host" onClick={props.onSelectOutputDir}>
            <FolderOpen size={16} /> Browse
          </button>
        </div>
      </label>
      {props.generationMode === 'quick' && (
        <div className="grid-3">
          <NumberField label="Max seq" value={props.quickMaxSequences} integer onChange={(value) => props.onQuickMax(Math.max(1, Math.round(value)))} />
          <NumberField label="Frames" value={props.quickOverrideFrames} integer onChange={(value) => props.onQuickFrames(Math.max(1, Math.round(value)))} />
          <NumberField label="FPS" value={props.quickOverrideFps} integer onChange={(value) => props.onQuickFps(Math.max(1, Math.round(value)))} />
        </div>
      )}
      <div className="button-row">
        <button type="button" className="primary-button" disabled={props.generationLocked} onClick={props.onGenerate}>
          <Save size={16} /> Generate dataset
        </button>
        <button type="button" onClick={props.onCancelGenerate}>Cancel</button>
      </div>
      <JobStatusLine label="Full preview" job={props.previewJob} />
      <JobStatusLine label="Generate" job={props.generateJob} />
      {props.message && <p className={`message ${props.message.tone}`}>{props.message.text}</p>}
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  integer,
  min,
  max,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  integer?: boolean
  min?: number
  max?: number
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={integer ? 1 : 0.01}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  )
}

function RangeNumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  const clamped = clamp(Number.isFinite(value) ? value : min, min, max)
  return (
    <label className="range-number-field">
      <span>{label}</span>
      <div>
        <input type="range" value={clamped} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
        <input type="number" value={Number.isFinite(value) ? value : min} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} />
      </div>
    </label>
  )
}

function JobStatusLine({ label, job }: { label: string; job: JobStatus | null }) {
  const percent = job ? clamp(job.percent, 0, 1) : 0
  const progressValue = job?.status === 'succeeded' ? 1 : percent
  return (
    <div className="job-line">
      <span>{label}: {job?.status ?? 'idle'}</span>
      {job && <progress max={1} value={progressValue} />}
      {job?.message && <small>{job.message}</small>}
    </div>
  )
}

export default App
