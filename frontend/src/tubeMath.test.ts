import { describe, expect, it } from 'vitest'
import {
  applyViewportOrbit,
  applyViewportWheel,
  cameraFacingThetaFromYawDeg,
  frequencyForCycles,
  oppositeTheta,
  sectionLength,
  sectionSummary,
  thetaFromCrossSectionPoint,
  tubeAxisLength,
} from './tubeMath'
import { defaultPreviewSettings } from './types'

describe('tubeMath', () => {
  it('maps cross-section cardinal points to theta', () => {
    expect(thetaFromCrossSectionPoint(180, 100, 200, 200)).toBeCloseTo(0)
    expect(thetaFromCrossSectionPoint(100, 20, 200, 200)).toBeCloseTo(0.25)
    expect(thetaFromCrossSectionPoint(20, 100, 200, 200)).toBeCloseTo(0.5)
    expect(thetaFromCrossSectionPoint(100, 180, 200, 200)).toBeCloseTo(0.75)
  })

  it('computes axis length from straight and turn sections', () => {
    expect(
      tubeAxisLength([
        { kind: 'straight', length: 2 },
        { kind: 'turn', direction: 'right', angle_deg: 90, radius: 4 },
      ]),
    ).toBeCloseTo(2 + sectionLength({ kind: 'turn', direction: 'right', angle_deg: 90, radius: 4 }))
  })

  it('formats concise axis section summaries', () => {
    expect(sectionSummary({ kind: 'straight', length: 5 })).toBe('straight 5')
    expect(sectionSummary({ kind: 'turn', direction: 'right', angle_deg: 30, radius: 5 })).toBe('turn right 30-5')
  })

  it('converts viewport gestures into camera settings', () => {
    const orbit = applyViewportOrbit({ ...defaultPreviewSettings, yaw_deg: 179, pitch_deg: 59 }, 20, -20)
    expect(orbit.yaw_deg).toBeCloseTo(-176)
    expect(orbit.pitch_deg).toBe(60)

    const zoom = applyViewportWheel({ ...defaultPreviewSettings, distance_scale: 4.95 }, 120, false)
    expect(zoom.distance_scale).toBe(5)

    const fov = applyViewportWheel({ ...defaultPreviewSettings, fov_deg: 109 }, 120, true)
    expect(fov.fov_deg).toBe(110)
  })

  it('normalizes camera-facing theta helpers and frequency cycles', () => {
    expect(cameraFacingThetaFromYawDeg(450)).toBeCloseTo(0.25)
    expect(oppositeTheta(0.9)).toBeCloseTo(0.4)
    expect(frequencyForCycles(2, 241, 24)).toBeCloseTo(0.2)
  })
})
