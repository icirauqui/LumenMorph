import type { PreviewSettings, TubeSection } from './types'

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

export const wrap01 = (value: number) => {
  const wrapped = value % 1
  return wrapped < 0 ? wrapped + 1 : wrapped
}

export const wrapYawDegrees = (value: number) => {
  let yaw = value
  while (yaw > 180) yaw -= 360
  while (yaw < -180) yaw += 360
  return yaw
}

export const cameraFacingThetaFromYawDeg = (yawDeg: number) => wrap01(yawDeg / 360)
export const oppositeTheta = (theta: number) => wrap01(theta + 0.5)

export const thetaFromCrossSectionPoint = (
  x: number,
  y: number,
  width: number,
  height: number,
) => {
  const cx = width * 0.5
  const cy = height * 0.5
  const dx = x - cx
  const dy = y - cy
  if (dx * dx + dy * dy < 1e-6) return 0
  return wrap01(Math.atan2(-dy, dx) / (2 * Math.PI))
}

export const sequenceDurationSeconds = (frames: number, fps: number) => {
  const safeFrames = Math.max(frames, 1)
  const safeFps = Math.max(fps, 1)
  return Math.max((safeFrames - 1) / safeFps, 1 / safeFps)
}

export const frequencyForCycles = (cycles: number, frames: number, fps: number) =>
  cycles / sequenceDurationSeconds(frames, fps)

export const cyclesForFrequency = (frequencyHz: number, frames: number, fps: number) =>
  Math.abs(frequencyHz) * sequenceDurationSeconds(frames, fps)

export const sectionLength = (section: TubeSection) =>
  section.kind === 'straight'
    ? Math.max(section.length, 0)
    : Math.abs((section.angle_deg * Math.PI) / 180) * Math.max(section.radius, 0)

export const tubeAxisLength = (sections: TubeSection[]) =>
  sections.reduce((sum, section) => sum + sectionLength(section), 0)

export const compactNumber = (value: number, digits = 2) =>
  Number.isFinite(value) ? value.toFixed(digits).replace(/\.?0+$/, '') : '0'

export const sectionSummary = (section: TubeSection) =>
  section.kind === 'straight'
    ? `straight ${compactNumber(section.length)}`
    : `turn ${section.direction} ${compactNumber(section.angle_deg, 1)}-${compactNumber(section.radius)}`

export const applyViewportOrbit = (
  settings: PreviewSettings,
  dx: number,
  dy: number,
): PreviewSettings => ({
  ...settings,
  yaw_deg: wrapYawDegrees(settings.yaw_deg + dx * 0.25),
  pitch_deg: clamp(settings.pitch_deg - dy * 0.2, -60, 60),
})

export const applyViewportWheel = (
  settings: PreviewSettings,
  scrollDy: number,
  shiftPressed: boolean,
): PreviewSettings =>
  shiftPressed
    ? { ...settings, fov_deg: clamp(settings.fov_deg + (scrollDy / 120) * 2, 25, 110) }
    : {
        ...settings,
        distance_scale: clamp(settings.distance_scale + (scrollDy / 120) * 0.18, 0.4, 5),
      }

export const resetViewportCamera = (settings: PreviewSettings): PreviewSettings => ({
  ...settings,
  yaw_deg: 42,
  pitch_deg: 18,
  distance_scale: 1.45,
  fov_deg: 55,
})
