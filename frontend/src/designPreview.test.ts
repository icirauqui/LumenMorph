import { describe, expect, it } from 'vitest'
import { buildTubeAxisSamples, buildTubeDesignMesh, deformationAreaWeight, tubeShellRadii } from './designPreview'
import type { TubeConfig } from './types'

const baseTube = (overrides: Partial<TubeConfig> = {}): TubeConfig => ({
  axis: { sections: [{ kind: 'straight', length: 4 }] },
  dimensions: {
    radial_samples: 16,
    longitudinal_samples: 24,
    base_radius: 1,
    min_radius: 0.35,
    shell_thickness: 0.25,
  },
  surface: { bump_count: 0, bump_amp_min: 0, bump_amp_max: 0, bump_sigma_long: 0.05, bump_sigma_theta: 0.05 },
  camera: { offset_ratio: 0, wobble_ratio: 0, wobble_hz: 0, lookahead_distance: 1 },
  deformation: {
    flow_mode: 'cyclic',
    flow_cycles: 1,
    localized_tint_enabled: false,
    localized_tint_bgr: [48, 185, 255],
    localized_tint_strength: 0.18,
    areas: [],
  },
  ...overrides,
})

describe('designPreview', () => {
  it('computes shell radii from outer radius, thickness, and minimum radius', () => {
    expect(tubeShellRadii(baseTube())).toEqual({ outer: 1, inner: 0.75, thickness: 0.25 })
    const clamped = tubeShellRadii(
      baseTube({
        dimensions: { radial_samples: 16, longitudinal_samples: 24, base_radius: 1, min_radius: 0.55, shell_thickness: 0.8 },
      }),
    )
    expect(clamped.outer).toBe(1)
    expect(clamped.inner).toBe(0.55)
    expect(clamped.thickness).toBeCloseTo(0.45)
  })

  it('samples right and upward turns with propagated frames', () => {
    const rightTurn = buildTubeAxisSamples(
      baseTube({
        axis: {
          sections: [
            { kind: 'straight', length: 2 },
            { kind: 'turn', direction: 'right', angle_deg: 90, radius: 2 },
            { kind: 'straight', length: 2 },
          ],
        },
      }),
      36,
    )
    const rightEnd = rightTurn[rightTurn.length - 1]
    expect(rightEnd.position[0]).toBeGreaterThan(2)
    expect(rightEnd.position[1]).toBeGreaterThan(2)
    expect(Math.abs(rightEnd.tangent[0] * rightEnd.right[0] + rightEnd.tangent[1] * rightEnd.right[1] + rightEnd.tangent[2] * rightEnd.right[2])).toBeLessThan(1e-6)

    const upTurn = buildTubeAxisSamples(
      baseTube({
        axis: {
          sections: [
            { kind: 'straight', length: 2 },
            { kind: 'turn', direction: 'up', angle_deg: 90, radius: 2 },
          ],
        },
      }),
      24,
    )
    expect(upTurn[upTurn.length - 1].position[2]).toBeGreaterThan(1.9)
  })

  it('builds a capped outer and inner editing mesh', () => {
    const mesh = buildTubeDesignMesh(baseTube(), 0)
    expect(mesh.longitudinalSamples).toBe(24)
    expect(mesh.radialSamples).toBe(16)
    expect(mesh.vertices).toHaveLength(24 * 16 * 2)
    expect(mesh.faces.length).toBeGreaterThan(24 * 16)
    expect(mesh.faces.every((face) => face.indices.length === 4)).toBe(true)
    expect(mesh.bounds.radius).toBeGreaterThan(2)
  })

  it('caps the editing mesh density for fluent local preview interaction', () => {
    const mesh = buildTubeDesignMesh(
      baseTube({
        dimensions: { radial_samples: 256, longitudinal_samples: 512, base_radius: 1, min_radius: 0.35, shell_thickness: 0.25 },
      }),
      0,
    )
    expect(mesh.longitudinalSamples).toBe(96)
    expect(mesh.radialSamples).toBe(48)
  })

  it('uses sequence duration for localized deformation timing, including the inward half-cycle', () => {
    const tube = baseTube({
      dimensions: { radial_samples: 8, longitudinal_samples: 5, base_radius: 1, min_radius: 0.35, shell_thickness: 0.25 },
      deformation: {
        ...baseTube().deformation,
        areas: [{
          id: 'radial',
          enabled: true,
          kind: 'localized',
          center_s: 0.5,
          center_theta: 0.25,
          sigma_s: 0.1,
          sigma_theta: 0.12,
          direction: { radial: 1, tangent: 0, circumferential: 0 },
          amplitude_out: 0.2,
          amplitude_in: 0.2,
          force_n: 0,
          frequency_hz: 0.2,
          phase_rad: 0,
          mode: 'radial',
          spatial_cycles_s: 0,
          spatial_cycles_theta: 0,
          flow_dir_s: 0,
          flow_dir_theta: 0,
        }],
      },
    })
    const atStart = buildTubeDesignMesh(tube, 0, undefined, 4)
    const atOutwardPeak = buildTubeDesignMesh(tube, 0.25, undefined, 4)
    const atInwardPeak = buildTubeDesignMesh(tube, 0.75, undefined, 4)
    const centreTopVertex = 2 * 8 + 2

    expect(atOutwardPeak.vertices[centreTopVertex][2]).toBeGreaterThan(atStart.vertices[centreTopVertex][2])
    expect(atInwardPeak.vertices[centreTopVertex][2]).toBeLessThan(atStart.vertices[centreTopVertex][2])
  })

  it('applies cached FEA displacement responses to the design mesh cyclically', () => {
    const tube = baseTube({
      dimensions: { radial_samples: 8, longitudinal_samples: 5, base_radius: 1, min_radius: 0.35, shell_thickness: 0.25 },
      deformation: {
        ...baseTube().deformation,
        areas: [
          {
            id: 'fea_1',
            enabled: true,
            kind: 'fea_c3d6',
            center_s: 0.5,
            center_theta: 0.25,
            sigma_s: 0.1,
            sigma_theta: 0.12,
            direction: { radial: 1, tangent: 0, circumferential: 0 },
            amplitude_out: 0,
            amplitude_in: 0,
            force_n: 100,
            frequency_hz: 1,
            phase_rad: 0,
            mode: 'radial',
            spatial_cycles_s: 0,
            spatial_cycles_theta: 0,
            flow_dir_s: 0,
            flow_dir_theta: 0,
          },
        ],
      },
    })
    const zero = buildTubeDesignMesh(tube, 0, {
      longitudinal_samples: 5,
      radial_samples: 8,
      responses: [
        {
          area_id: 'fea_1',
          applied_s_norm: 0.5,
          applied_theta_norm: 0.25,
          force_vector_local: [0, 0, 100],
          displacement_grid: Array.from({ length: 5 }, () =>
            Array.from({ length: 8 }, () => [0, 0, 0.1]),
          ),
        },
      ],
    }, 1)
    const peak = buildTubeDesignMesh(tube, 0.5, {
      longitudinal_samples: 5,
      radial_samples: 8,
      responses: [
        {
          area_id: 'fea_1',
          applied_s_norm: 0.5,
          applied_theta_norm: 0.25,
          force_vector_local: [0, 0, 100],
          displacement_grid: Array.from({ length: 5 }, () =>
            Array.from({ length: 8 }, () => [0, 0, 0.1]),
          ),
        },
      ],
    }, 1)
    const amplified = buildTubeDesignMesh(tube, 0.5, {
      longitudinal_samples: 5,
      radial_samples: 8,
      responses: [
        {
          area_id: 'fea_1',
          applied_s_norm: 0.5,
          applied_theta_norm: 0.25,
          force_vector_local: [0, 0, 100],
          displacement_grid: Array.from({ length: 5 }, () =>
            Array.from({ length: 8 }, () => [0, 0, 0.1]),
          ),
        },
      ],
    }, 1, 3)

    expect(zero.vertices[0][2]).toBeCloseTo(peak.vertices[0][2] - 0.1)
    expect(amplified.vertices[0][2]).toBeCloseTo(peak.vertices[0][2] + 0.2)
    expect(peak.faces.every((face) => face.indices.length === 3)).toBe(true)
  })

  it('weights enabled deformation areas for local design viewport highlighting', () => {
    const tube = baseTube({
      deformation: {
        ...baseTube().deformation,
        areas: [
          {
            id: 'a',
            enabled: true,
            center_s: 0.5,
            center_theta: 0.25,
            sigma_s: 0.1,
            sigma_theta: 0.12,
            direction: { radial: 1, tangent: 0, circumferential: 0 },
            amplitude_out: 0.2,
            amplitude_in: 0.2,
            frequency_hz: 0.2,
            phase_rad: 0,
            mode: 'radial',
            spatial_cycles_s: 0,
            spatial_cycles_theta: 0,
            flow_dir_s: 0,
            flow_dir_theta: 0,
          },
        ],
      },
    })
    expect(deformationAreaWeight(tube, 0.5, 0.25)).toBeCloseTo(1)
    expect(deformationAreaWeight(tube, 0.95, 0.75)).toBeLessThan(0.01)
    const disabled = { ...tube, deformation: { ...tube.deformation, areas: [{ ...tube.deformation.areas[0], enabled: false }] } }
    expect(deformationAreaWeight(disabled, 0.5, 0.25)).toBe(0)
  })
})
