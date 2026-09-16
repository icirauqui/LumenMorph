import type { PreviewSettings, TubeConfig, TubeFeaPreview, TubeFeaPreviewArea, TubeSection } from './types'

type Vec3 = [number, number, number]
type TubeLayer = 'outer' | 'inner' | 'rim'

export type TubeShellRadii = {
  outer: number
  inner: number
  thickness: number
}

export type TubeAxisSample = {
  position: Vec3
  tangent: Vec3
  right: Vec3
  up: Vec3
  sNorm: number
}

type TubeFace = {
  indices: [number, number, number] | [number, number, number, number]
  normal: Vec3
  layer: TubeLayer
  areaWeight: number
}

export type TubeDesignMesh = {
  vertices: Vec3[]
  faces: TubeFace[]
  bounds: {
    min: Vec3
    max: Vec3
    center: Vec3
    radius: number
  }
  radii: TubeShellRadii
  longitudinalSamples: number
  radialSamples: number
}

const EPS = 1e-8
const TWO_PI = Math.PI * 2
const MAX_DESIGN_LONGITUDINAL = 96
const MAX_DESIGN_RADIAL = 48

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const length = (a: Vec3) => Math.sqrt(dot(a, a))
const normalize = (a: Vec3, fallback: Vec3 = [1, 0, 0]): Vec3 => {
  const n = length(a)
  return n > EPS ? scale(a, 1 / n) : fallback
}
const mix = (a: Vec3, b: Vec3, t: number): Vec3 => [
  a[0] * (1 - t) + b[0] * t,
  a[1] * (1 - t) + b[1] * t,
  a[2] * (1 - t) + b[2] * t,
]
const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
const wrap01 = (value: number) => {
  const wrapped = value % 1
  return wrapped < 0 ? wrapped + 1 : wrapped
}
const wrappedDistance = (a: number, b: number) => {
  const d = Math.abs(wrap01(a) - wrap01(b))
  return Math.min(d, 1 - d)
}

const tubeHasEnabledFeaArea = (tube: TubeConfig) =>
  (tube.deformation?.areas ?? []).some((area) => area.enabled && (area.kind ?? 'localized') === 'fea_c3d6')

const areaGaussianWeight = (
  area: NonNullable<TubeConfig['deformation']['areas']>[number],
  sNorm: number,
  thetaNorm: number,
) => {
  const sigmaS = Math.max(area.sigma_s, EPS)
  const sigmaTheta = Math.max(area.sigma_theta, EPS)
  const ds = (sNorm - area.center_s) / sigmaS
  const dt = wrappedDistance(thetaNorm, area.center_theta) / sigmaTheta
  return Math.exp(-0.5 * (ds * ds + dt * dt))
}

export const deformationAreaWeight = (tube: TubeConfig, sNorm: number, thetaNorm: number) => {
  let weight = 0
  for (const area of tube.deformation?.areas ?? []) {
    if (!area.enabled) continue
    weight = Math.max(weight, areaGaussianWeight(area, sNorm, thetaNorm))
  }
  return clamp(weight, 0, 1)
}

const sectionArcLength = (section: TubeSection) =>
  section.kind === 'turn'
    ? Math.abs((section.angle_deg * Math.PI) / 180) * Math.max(section.radius, EPS)
    : Math.max(section.length, EPS)

const advanceAxisState = (
  position: Vec3,
  tangent: Vec3,
  right: Vec3,
  up: Vec3,
  section: TubeSection,
  distance: number,
): Omit<TubeAxisSample, 'sNorm'> => {
  const dist = clamp(distance, 0, sectionArcLength(section))
  if (section.kind === 'straight') {
    return { position: add(position, scale(tangent, dist)), tangent, right, up }
  }

  const totalAngle = (section.angle_deg * Math.PI) / 180
  const sign = section.direction === 'left' || section.direction === 'down' ? -1 : 1
  const totalPhi = sign * totalAngle
  const totalArc = Math.max(Math.abs(totalPhi) * Math.max(section.radius, EPS), EPS)
  const phi = totalPhi * (dist / totalArc)
  const radius = Math.max(section.radius, EPS)
  const c = Math.cos(phi)
  const s = Math.sin(phi)

  if (section.direction === 'left' || section.direction === 'right') {
    const bend = right
    const nextPosition = add(position, scale(add(scale(tangent, s), scale(bend, 1 - c)), radius))
    return {
      position: nextPosition,
      tangent: normalize(add(scale(tangent, c), scale(bend, s)), tangent),
      right: normalize(add(scale(tangent, -s), scale(bend, c)), right),
      up,
    }
  }

  const bend = up
  const nextPosition = add(position, scale(add(scale(tangent, s), scale(bend, 1 - c)), radius))
  return {
    position: nextPosition,
    tangent: normalize(add(scale(tangent, c), scale(bend, s)), tangent),
    right,
    up: normalize(add(scale(tangent, -s), scale(bend, c)), up),
  }
}

export const tubeShellRadii = (tube: TubeConfig): TubeShellRadii => {
  const outer = Math.max(Number(tube.dimensions.base_radius), EPS)
  const requestedThickness = Math.max(Number(tube.dimensions.shell_thickness ?? 0.25), 0)
  const minRadius = Math.max(Number(tube.dimensions.min_radius ?? 0), 0)
  const inner = Math.max(minRadius, outer - requestedThickness)
  return { outer, inner, thickness: Math.max(outer - inner, 0) }
}

export const tubeDesignSampleCounts = (tube: TubeConfig) => ({
  longitudinalSamples: Math.max(2, Math.min(tube.dimensions.longitudinal_samples, MAX_DESIGN_LONGITUDINAL)),
  radialSamples: Math.max(8, Math.min(tube.dimensions.radial_samples, MAX_DESIGN_RADIAL)),
})

export const buildTubeAxisSamples = (tube: TubeConfig, sampleCount?: number): TubeAxisSample[] => {
  const sections = tube.axis.sections.length > 0 ? tube.axis.sections : [{ kind: 'straight' as const, length: 4 }]
  const nLong = Math.max(2, Math.min(sampleCount ?? tube.dimensions.longitudinal_samples, MAX_DESIGN_LONGITUDINAL))
  const starts: Array<{
    arcStart: number
    section: TubeSection
    position: Vec3
    tangent: Vec3
    right: Vec3
    up: Vec3
  }> = []

  let arcCursor = 0
  let position: Vec3 = [0, 0, 0]
  let tangent: Vec3 = [0, 1, 0]
  let right: Vec3 = [1, 0, 0]
  let up: Vec3 = [0, 0, 1]

  for (const section of sections) {
    starts.push({ arcStart: arcCursor, section, position, tangent, right, up })
    const advanced = advanceAxisState(position, tangent, right, up, section, sectionArcLength(section))
    position = advanced.position
    tangent = advanced.tangent
    right = advanced.right
    up = advanced.up
    arcCursor += sectionArcLength(section)
  }

  const totalLength = Math.max(arcCursor, EPS)
  const samples: TubeAxisSample[] = []
  for (let index = 0; index < nLong; index += 1) {
    const sNorm = nLong === 1 ? 0 : index / (nLong - 1)
    const absoluteS = totalLength * sNorm
    let start = starts[starts.length - 1]
    for (const candidate of starts) {
      if (candidate.arcStart <= absoluteS + EPS) start = candidate
      else break
    }
    const advanced = advanceAxisState(
      start.position,
      start.tangent,
      start.right,
      start.up,
      start.section,
      absoluteS - start.arcStart,
    )
    samples.push({
      position: advanced.position,
      tangent: advanced.tangent,
      right: advanced.right,
      up: advanced.up,
      sNorm,
    })
  }
  return samples
}

const interpolateFeaResponse = (response: TubeFeaPreviewArea, sNorm: number, thetaNorm: number): Vec3 => {
  const grid = response.displacement_grid
  const nLong = grid.length
  const nRad = grid[0]?.length ?? 0
  if (nLong < 2 || nRad < 1) return [0, 0, 0]

  const sPos = clamp(sNorm, 0, 1) * (nLong - 1)
  const s0 = Math.min(Math.floor(sPos), nLong - 2)
  const s1 = s0 + 1
  const sa = sPos - s0

  const thetaPos = wrap01(thetaNorm) * nRad
  const t0 = Math.floor(thetaPos) % nRad
  const t1 = (t0 + 1) % nRad
  const ta = thetaPos - Math.floor(thetaPos)

  const v00 = grid[s0][t0]
  const v01 = grid[s0][t1]
  const v10 = grid[s1][t0]
  const v11 = grid[s1][t1]
  return [0, 1, 2].map((axis) => {
    const lo = Number(v00?.[axis] ?? 0) * (1 - ta) + Number(v01?.[axis] ?? 0) * ta
    const hi = Number(v10?.[axis] ?? 0) * (1 - ta) + Number(v11?.[axis] ?? 0) * ta
    return lo * (1 - sa) + hi * sa
  }) as Vec3
}

const feaCyclicScale = (
  area: NonNullable<TubeConfig['deformation']['areas']>[number],
  tNorm: number,
  sequenceDurationSeconds: number,
) => 0.5 * (1 - Math.cos(TWO_PI * Math.max(area.frequency_hz, 0) * Math.max(sequenceDurationSeconds, EPS) * tNorm + area.phase_rad))

const feaSurfaceDisplacement = (
  tube: TubeConfig,
  preview: TubeFeaPreview | null | undefined,
  sNorm: number,
  thetaNorm: number,
  tNorm: number,
  sequenceDurationSeconds: number,
  displacementGain: number,
): Vec3 => {
  if (!preview || preview.responses.length === 0) return [0, 0, 0]
  let displacement: Vec3 = [0, 0, 0]
  for (const response of preview.responses) {
    const area = tube.deformation.areas.find((item) => item.id === response.area_id)
    if (!area || !area.enabled || (area.kind ?? 'localized') !== 'fea_c3d6') continue
    const scaleValue = feaCyclicScale(area, tNorm, sequenceDurationSeconds) * Math.max(displacementGain, 0)
    displacement = add(displacement, scale(interpolateFeaResponse(response, sNorm, thetaNorm), scaleValue))
  }
  return displacement
}

const localDeformation = (
  tube: TubeConfig,
  sNorm: number,
  thetaNorm: number,
  radial: Vec3,
  tangent: Vec3,
  circumferential: Vec3,
  tNorm: number,
  outerRadius: number,
  sequenceDurationSeconds: number,
): Vec3 => {
  const areas = tube.deformation?.areas ?? []
  let displacement: Vec3 = [0, 0, 0]
  for (const area of areas) {
    if (!area.enabled) continue
    if ((area.kind ?? 'localized') !== 'localized') continue
    const weight = areaGaussianWeight(area, sNorm, thetaNorm)
    if (weight < 0.002) continue

    // tNorm is a position within the sequence, whereas frequency is expressed
    // in Hz. Convert it to elapsed seconds so the editor matches generated and
    // backend-preview motion, including both outward and inward half-cycles.
    const temporalCycles = Math.abs(area.frequency_hz) > EPS
      ? area.frequency_hz * Math.max(sequenceDurationSeconds, 0)
      : tube.deformation.flow_cycles
    let phase = TWO_PI * temporalCycles * tNorm + area.phase_rad
    if (area.mode === 'travelling') {
      const spatial =
        area.spatial_cycles_s * sNorm +
        area.spatial_cycles_theta * thetaNorm -
        (area.flow_dir_s + area.flow_dir_theta) * tNorm
      phase += TWO_PI * spatial
    }
    const wave = Math.sin(phase)
    const amplitude = wave >= 0 ? area.amplitude_out : area.amplitude_in
    const amount = wave * amplitude * weight
    const direction = normalize(
      add(
        add(scale(radial, area.direction.radial), scale(tangent, area.direction.tangent)),
        scale(circumferential, area.direction.circumferential),
      ),
      radial,
    )
    displacement = add(displacement, scale(direction, amount))
  }

  const minRadius = Math.max(Number(tube.dimensions.min_radius ?? 0), 0)
  const radialDisplacement = dot(displacement, radial)
  if (outerRadius + radialDisplacement < minRadius) {
    displacement = add(displacement, scale(radial, minRadius - outerRadius - radialDisplacement))
  }
  return displacement
}

export const buildTubeDesignMesh = (
  tube: TubeConfig,
  tNorm = 0,
  feaPreview?: TubeFeaPreview | null,
  sequenceDurationSeconds = 1,
  feaDisplacementGain = 1,
): TubeDesignMesh => {
  const radii = tubeShellRadii(tube)
  const { longitudinalSamples: nLong, radialSamples: nRad } = tubeDesignSampleCounts(tube)
  const axis = buildTubeAxisSamples(tube, nLong)
  const vertices: Vec3[] = []
  const normals: Vec3[] = []
  const outerIndex = (j: number, i: number) => j * nRad + (i % nRad)
  const innerOffset = nLong * nRad
  const innerIndex = (j: number, i: number) => innerOffset + j * nRad + (i % nRad)

  for (let j = 0; j < nLong; j += 1) {
    const sample = axis[j]
    for (let i = 0; i < nRad; i += 1) {
      const theta = (i / nRad) * TWO_PI
      const thetaNorm = i / nRad
      const radial = normalize(add(scale(sample.right, Math.cos(theta)), scale(sample.up, Math.sin(theta))), sample.right)
      const circumferential = normalize(add(scale(sample.right, -Math.sin(theta)), scale(sample.up, Math.cos(theta))), sample.up)
      const displacement = add(
        localDeformation(tube, sample.sNorm, thetaNorm, radial, sample.tangent, circumferential, tNorm, radii.outer, sequenceDurationSeconds),
        feaSurfaceDisplacement(tube, feaPreview, sample.sNorm, thetaNorm, tNorm, sequenceDurationSeconds, feaDisplacementGain),
      )
      vertices.push(add(add(sample.position, scale(radial, radii.outer)), displacement))
      normals.push(radial)
    }
  }

  for (let j = 0; j < nLong; j += 1) {
    const sample = axis[j]
    for (let i = 0; i < nRad; i += 1) {
      const theta = (i / nRad) * TWO_PI
      const thetaNorm = i / nRad
      const radial = normalize(add(scale(sample.right, Math.cos(theta)), scale(sample.up, Math.sin(theta))), sample.right)
      const circumferential = normalize(add(scale(sample.right, -Math.sin(theta)), scale(sample.up, Math.cos(theta))), sample.up)
      const displacement = add(
        localDeformation(tube, sample.sNorm, thetaNorm, radial, sample.tangent, circumferential, tNorm, radii.outer, sequenceDurationSeconds),
        feaSurfaceDisplacement(tube, feaPreview, sample.sNorm, thetaNorm, tNorm, sequenceDurationSeconds, feaDisplacementGain),
      )
      vertices.push(add(add(sample.position, scale(radial, radii.inner)), displacement))
      normals.push(scale(radial, -1))
    }
  }

  const faces: TubeFace[] = []
  const useC3d6SurfacePreview = tubeHasEnabledFeaArea(tube)
  const pushFace = (
    indices: TubeFace['indices'],
    normal: Vec3,
    layer: TubeLayer,
    areaWeight: number,
  ) => faces.push({ indices, normal, layer, areaWeight })
  const pushSplitQuad = (
    indices: [number, number, number, number],
    normal: Vec3,
    layer: TubeLayer,
    areaWeight: number,
    flipDiagonal: boolean,
  ) => {
    if (!useC3d6SurfacePreview) {
      pushFace(indices, normal, layer, areaWeight)
      return
    }
    const [a, b, c, d] = indices
    if (flipDiagonal) {
      pushFace([a, b, d], normal, layer, areaWeight)
      pushFace([b, c, d], normal, layer, areaWeight)
    } else {
      pushFace([a, b, c], normal, layer, areaWeight)
      pushFace([a, c, d], normal, layer, areaWeight)
    }
  }

  for (let j = 0; j < nLong - 1; j += 1) {
    for (let i = 0; i < nRad; i += 1) {
      const nextI = (i + 1) % nRad
      const areaWeight = deformationAreaWeight(tube, (axis[j].sNorm + axis[j + 1].sNorm) * 0.5, (i + 0.5) / nRad)
      const flipDiagonal = (j + i) % 2 === 1
      pushSplitQuad(
        [outerIndex(j, i), outerIndex(j + 1, i), outerIndex(j + 1, nextI), outerIndex(j, nextI)],
        normalize(add(normals[outerIndex(j, i)], normals[outerIndex(j, nextI)])),
        'outer',
        areaWeight,
        flipDiagonal,
      )
      pushSplitQuad(
        [innerIndex(j, nextI), innerIndex(j + 1, nextI), innerIndex(j + 1, i), innerIndex(j, i)],
        normalize(add(normals[innerIndex(j, i)], normals[innerIndex(j, nextI)]), scale(normals[innerIndex(j, i)], -1)),
        'inner',
        areaWeight * 0.6,
        flipDiagonal,
      )
    }
  }

  for (let i = 0; i < nRad; i += 1) {
    const nextI = (i + 1) % nRad
    pushSplitQuad(
      [outerIndex(0, i), outerIndex(0, nextI), innerIndex(0, nextI), innerIndex(0, i)],
      scale(axis[0].tangent, -1),
      'rim',
      0,
      i % 2 === 1,
    )
    pushSplitQuad(
      [outerIndex(nLong - 1, nextI), outerIndex(nLong - 1, i), innerIndex(nLong - 1, i), innerIndex(nLong - 1, nextI)],
      axis[nLong - 1].tangent,
      'rim',
      0,
      i % 2 === 1,
    )
  }

  const min: Vec3 = [Infinity, Infinity, Infinity]
  const max: Vec3 = [-Infinity, -Infinity, -Infinity]
  for (const vertex of vertices) {
    for (let axisIndex = 0; axisIndex < 3; axisIndex += 1) {
      min[axisIndex] = Math.min(min[axisIndex], vertex[axisIndex])
      max[axisIndex] = Math.max(max[axisIndex], vertex[axisIndex])
    }
  }
  const center = mix(min, max, 0.5)
  const radius = Math.max(...vertices.map((vertex) => length(sub(vertex, center))), radii.outer)

  return { vertices, faces, bounds: { min, max, center, radius }, radii, longitudinalSamples: nLong, radialSamples: nRad }
}

const layerBaseColor = (layer: TubeLayer): Vec3 => {
  if (layer === 'inner') return [82, 101, 113]
  if (layer === 'rim') return [139, 150, 158]
  return [189, 195, 199]
}

const colorForFace = (normal: Vec3, layer: TubeLayer, areaWeight: number) => {
  const light = normalize([-0.35, -0.4, 0.85])
  const shade = 0.74 + Math.max(0, dot(normalize(normal), light)) * 0.28
  const base = layerBaseColor(layer)
  const highlighted = mix(base, [255, 186, 74], clamp(areaWeight * 0.88, 0, 0.88))
  const rgb = highlighted.map((value) => Math.round(clamp(value * shade, 0, 255)))
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
}

export const renderTubeDesignPreview = (
  canvas: HTMLCanvasElement,
  mesh: TubeDesignMesh,
  settings: PreviewSettings,
) => {
  if (typeof CanvasRenderingContext2D === 'undefined') return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const cssWidth = Math.max(1, canvas.clientWidth)
  const cssHeight = Math.max(1, canvas.clientHeight)
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2)
  const targetWidth = Math.round(cssWidth * pixelRatio)
  const targetHeight = Math.round(cssHeight * pixelRatio)
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth
    canvas.height = targetHeight
  }
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  ctx.clearRect(0, 0, cssWidth, cssHeight)
  ctx.fillStyle = '#101417'
  ctx.fillRect(0, 0, cssWidth, cssHeight)

  const yaw = (settings.yaw_deg * Math.PI) / 180
  const pitch = (settings.pitch_deg * Math.PI) / 180
  const horizontal = Math.cos(pitch)
  const cameraDirection = normalize([Math.sin(yaw) * horizontal, -Math.cos(yaw) * horizontal, Math.sin(pitch)])
  const forward = scale(cameraDirection, -1)
  const upHint = Math.abs(dot(forward, [0, 0, 1])) > 0.96 ? [0, 1, 0] as Vec3 : [0, 0, 1] as Vec3
  const right = normalize(cross(forward, upHint), [1, 0, 0])
  const up = normalize(cross(right, forward), [0, 0, 1])
  const verticalFov = clamp(settings.fov_deg, 20, 115) * Math.PI / 180
  const tanY = Math.tan(verticalFov / 2)
  const tanX = tanY * (cssWidth / cssHeight)
  const center = mesh.bounds.center

  let requiredDistance = Math.max(mesh.bounds.radius, EPS)
  for (const vertex of mesh.vertices) {
    const offset = sub(vertex, center)
    const zOffset = dot(offset, forward)
    requiredDistance = Math.max(requiredDistance, Math.abs(dot(offset, right)) / Math.max(tanX * 0.82, EPS) - zOffset)
    requiredDistance = Math.max(requiredDistance, Math.abs(dot(offset, up)) / Math.max(tanY * 0.82, EPS) - zOffset)
    requiredDistance = Math.max(requiredDistance, -zOffset + mesh.radii.outer * 0.5)
  }
  const distance = Math.max(requiredDistance, mesh.bounds.radius) * clamp(settings.distance_scale, 0.25, 8) + mesh.radii.outer * 0.4
  const eye = add(center, scale(cameraDirection, distance))
  const projected = mesh.vertices.map((vertex) => {
    const rel = sub(vertex, eye)
    const z = dot(rel, forward)
    if (z <= EPS) return null
    return {
      x: cssWidth * 0.5 + (dot(rel, right) / z / tanX) * cssWidth * 0.5,
      y: cssHeight * 0.5 - (dot(rel, up) / z / tanY) * cssHeight * 0.5,
      z,
    }
  })

  const drawable = mesh.faces
    .map((face) => {
      const points = face.indices.map((index) => projected[index])
      if (points.some((point) => point === null)) return null
      const safePoints = points as Array<{ x: number; y: number; z: number }>
      return {
        points: safePoints,
        z: safePoints.reduce((sum, point) => sum + point.z, 0) / safePoints.length,
        color: colorForFace(face.normal, face.layer, face.areaWeight),
        stroke: face.areaWeight > 0.22 ? 'rgba(255, 221, 129, 0.7)' : face.layer === 'inner' ? 'rgba(23, 31, 37, 0.28)' : 'rgba(18, 23, 27, 0.18)',
      }
    })
    .filter((face): face is NonNullable<typeof face> => face !== null)
    .sort((a, b) => b.z - a.z)

  ctx.lineWidth = 0.6
  for (const face of drawable) {
    ctx.beginPath()
    ctx.moveTo(face.points[0].x, face.points[0].y)
    for (let index = 1; index < face.points.length; index += 1) {
      ctx.lineTo(face.points[index].x, face.points[index].y)
    }
    ctx.closePath()
    ctx.fillStyle = face.color
    ctx.fill()
    ctx.strokeStyle = face.stroke
    ctx.stroke()
  }
}
