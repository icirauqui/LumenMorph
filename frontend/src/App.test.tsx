import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const presetConfig = {
  version: '0.2.2',
  scene_type: 'tube',
  sequence: { num_frames: 90, fps: 24 },
  tube: {
    axis: { sections: [{ kind: 'straight', length: 4 }] },
    dimensions: { radial_samples: 16, longitudinal_samples: 24, base_radius: 1, min_radius: 0.4, shell_thickness: 0.2 },
    surface: { bump_count: 0, bump_amp_min: 0, bump_amp_max: 0, bump_sigma_long: 0.05, bump_sigma_theta: 0.05 },
    camera: { offset_ratio: 0, wobble_ratio: 0, wobble_hz: 0, lookahead_distance: 1 },
    deformation: {
      flow_mode: 'cyclic',
      flow_cycles: 1,
      localized_tint_enabled: true,
      localized_tint_bgr: [180, 70, 200],
      localized_tint_strength: 0.2,
      areas: [],
    },
  },
}

class MockWebSocket {
  static instances: MockWebSocket[] = []
  binaryType = 'blob'
  onopen: (() => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: (() => void) | null = null
  onclose: (() => void) | null = null

  constructor() {
    MockWebSocket.instances.push(this)
    setTimeout(() => this.onopen?.(), 0)
  }

  close() {
    this.onclose?.()
  }
}

describe('App', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket)
    Object.defineProperty(HTMLElement.prototype, 'setPointerCapture', { configurable: true, value: vi.fn() })
    Object.defineProperty(HTMLElement.prototype, 'releasePointerCapture', { configurable: true, value: vi.fn() })
    Object.defineProperty(HTMLElement.prototype, 'hasPointerCapture', { configurable: true, value: vi.fn(() => true) })
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        if (url.endsWith('/v1/health')) return Response.json({ ok: true })
        if (url.endsWith('/v1/output/defaults')) {
          return Response.json({ repo_root: '/repo', default_output_dir: '/repo/outputs', exists: false })
        }
        if (url.endsWith('/v1/output/select-directory') && init?.method === 'POST') {
          return Response.json({ selected: true, output_dir: '/tmp/chosen-output' })
        }
        if (url.endsWith('/v1/presets')) {
          return Response.json({
            presets: [{ name: 'tube_v2', scene_type: 'tube', path: 'configs/tube_v2.json', config: presetConfig }],
          })
        }
        if (url.endsWith('/v1/preview/sessions') && init?.method === 'POST') {
          return Response.json({
            session_id: 'session-1',
            status: 'ok',
            seq: 1,
            t_norm: 0,
            render_ms: 4,
            camera: {},
            playback: {},
          })
        }
        if (url.endsWith('/v1/tube/fea-preview') && init?.method === 'POST') {
          return Response.json({
            longitudinal_samples: 4,
            radial_samples: 4,
            responses: [
              {
                area_id: 'area_1',
                applied_s_norm: 0.5,
                applied_theta_norm: 0.25,
                applied_node: 21,
                applied_axial_index: 2,
                applied_theta_index: 1,
                force_vector_local: [0, 0, 800],
                displacement_grid: [
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0.1], [0, 0, 0.05], [0, 0, 0], [0, 0, 0.05]],
                  [[0, 0, 0.1], [0, 0, 0.05], [0, 0, 0], [0, 0, 0.05]],
                  [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ],
              },
            ],
          })
        }
        if (url.endsWith('/v1/preview/sessions/session-1/config') && init?.method === 'PATCH') {
          return Response.json({ session_id: 'session-1', status: 'ok', seq: 2, t_norm: 0, render_ms: 5 })
        }
        if (url.endsWith('/v1/preview/sessions/session-1/camera') && init?.method === 'PATCH') {
          return Response.json({ session_id: 'session-1', status: 'ok', seq: 3, t_norm: 0, render_ms: 5 })
        }
        if (url.endsWith('/v1/preview/sessions/session-1/playback') && init?.method === 'PATCH') {
          return Response.json({ session_id: 'session-1', status: 'ok', seq: 4, t_norm: 0, render_ms: 5 })
        }
        if (url.includes('/v1/preview/sessions/session-1') && init?.method === 'DELETE') {
          return Response.json({ ok: true })
        }
        if (url.endsWith('/v1/generate/jobs') && init?.method === 'POST') {
          return Response.json({ job_id: 'generate-1' })
        }
        if (url.endsWith('/v1/jobs/generate-1/result')) {
          return Response.json({ result: { mode: 'quick', output_dir: '/repo/outputs' } })
        }
        if (url.endsWith('/v1/jobs/generate-1')) {
          return Response.json({
            job_id: 'generate-1',
            kind: 'generate',
            status: 'succeeded',
            stage: 'done',
            percent: 1,
            message: 'Done',
            result_ready: true,
            frame_count: 0,
          })
        }
        return Response.json({ ok: true })
      }),
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
    MockWebSocket.instances = []
  })

  it('boots into the tube-first workspace from presets', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())
    expect(screen.getByText('Axis Sections')).toBeInTheDocument()
    expect(screen.getByText('Deformation Areas')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Top' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Persp' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Texture' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Generate dataset' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Start here' })).toBeInTheDocument()
  })

  it('opens quick generation from the guided first-run action', async () => {
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: 'Generate a quick sample' }))
    const dialog = screen.getByRole('dialog', { name: 'Generate dataset' })
    expect(dialog).toBeInTheDocument()
    expect(within(dialog).getByLabelText('Generation scope')).toHaveValue('quick')
    expect(within(dialog).getByText('Creates a small, editable test run. This is the recommended first export.')).toBeInTheDocument()
  })

  it('collapses axis section cards with compact section titles', async () => {
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    const sectionToggle = screen.getByRole('button', { name: /1\. straight 4/ })
    expect(screen.getByLabelText('Length')).toBeInTheDocument()
    await user.click(sectionToggle)
    expect(screen.queryByLabelText('Length')).not.toBeInTheDocument()
    await user.click(sectionToggle)
    expect(screen.getByLabelText('Length')).toBeInTheDocument()
  })

  it('collapses the tube definition block', async () => {
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    const tubeToggle = screen.getByRole('button', { name: /Tube.*4\.00 length/ })
    expect(screen.getByLabelText('Base radius')).toBeInTheDocument()
    await user.click(tubeToggle)
    expect(screen.queryByLabelText('Base radius')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Tube.*4\.00 length · 0 areas/ })).toBeInTheDocument()
  })

  it('starts generation and polls the job to completion', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.mocked(fetch)
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    await user.click(screen.getByRole('button', { name: 'Generate dataset' }))
    const dialog = screen.getByRole('dialog', { name: 'Generate dataset' })
    expect(dialog).toBeInTheDocument()
    expect(within(dialog).getByLabelText('Output directory')).toHaveValue('/repo/outputs')
    expect(within(dialog).getByLabelText('Frames')).toHaveValue(90)
    expect(within(dialog).getByLabelText('FPS')).toHaveValue(24)
    await user.click(within(dialog).getByRole('button', { name: /Generate/ }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/v1/generate/jobs'), expect.objectContaining({ method: 'POST' })))
    const generateCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/v1/generate/jobs'))
    const body = JSON.parse(String(generateCall?.[1]?.body))
    expect(body.output_dir).toBe('/repo/outputs')
    expect(body.quick.override_num_frames).toBe(90)
    expect(body.quick.override_fps).toBe(24)
    await waitFor(() => expect(screen.getByText('Generation completed: /repo/outputs')).toBeInTheDocument())
    expect(within(dialog).getByRole('progressbar')).toHaveAttribute('value', '1')
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes('/frame/latest'))).toBe(false)
  })

  it('switches a deformation area to FEA and sends FEA config for generation', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.mocked(fetch)
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    const deformationSection = screen.getByText('Deformation Areas').closest('section')
    expect(deformationSection).not.toBeNull()
    await user.click(within(deformationSection as HTMLElement).getByRole('button', { name: /Add/ }))
    await user.selectOptions(screen.getByLabelText('Deformation kind'), 'fea_c3d6')
    expect(screen.getByText('Force N')).toBeInTheDocument()
    expect(screen.getByLabelText('Force direction')).toHaveValue('outward')

    await user.click(screen.getByRole('button', { name: 'Generate dataset' }))
    const dialog = screen.getByRole('dialog', { name: 'Generate dataset' })
    await user.click(within(dialog).getByRole('button', { name: /Generate/ }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/v1/generate/jobs'), expect.objectContaining({ method: 'POST' })))
    const generateCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/v1/generate/jobs'))
    const body = JSON.parse(String(generateCall?.[1]?.body))
    expect(body.config.tube.deformation.areas[0].kind).toBe('fea_c3d6')
    expect(body.config.tube.deformation.areas[0].force_n).toBe(800)
  })

  it('sends backend camera updates when orbiting the FEA live viewport', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.mocked(fetch)
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    const deformationSection = screen.getByText('Deformation Areas').closest('section')
    expect(deformationSection).not.toBeNull()
    await user.click(within(deformationSection as HTMLElement).getByRole('button', { name: /Add/ }))
    await user.selectOptions(screen.getByLabelText('Deformation kind'), 'fea_c3d6')

    const viewport = screen.getByLabelText('Tube design preview').closest('.viewport')
    expect(viewport).not.toBeNull()
    fireEvent.pointerDown(viewport as HTMLElement, { clientX: 120, clientY: 120, pointerId: 1 })
    fireEvent.pointerMove(viewport as HTMLElement, { clientX: 150, clientY: 95, pointerId: 1 })
    fireEvent.pointerUp(viewport as HTMLElement, { pointerId: 1 })

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/v1/preview/sessions/session-1/camera'),
        expect.objectContaining({ method: 'PATCH' }),
      ),
    )
    const cameraCall = fetchMock.mock.calls.find(([url, init]) =>
      String(url).endsWith('/v1/preview/sessions/session-1/camera') && init?.method === 'PATCH',
    )
    const body = JSON.parse(String(cameraCall?.[1]?.body))
    expect(body.yaw_deg).not.toBe(42)
    expect(body.pitch_deg).not.toBe(18)
  })

  it('refreshes the FEA render cache from the deformation controls', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.mocked(fetch)
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    const deformationSection = screen.getByText('Deformation Areas').closest('section')
    expect(deformationSection).not.toBeNull()
    await user.click(within(deformationSection as HTMLElement).getByRole('button', { name: /Add/ }))
    await user.selectOptions(screen.getByLabelText('Deformation kind'), 'fea_c3d6')
    expect(screen.getByText('FEA render not refreshed')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Refresh FEA render/ }))

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/v1/tube/fea-preview'),
        expect.objectContaining({ method: 'POST' }),
      ),
    )
    expect(await screen.findByText(/FEA render cache ready/)).toBeInTheDocument()
    expect(screen.getByText('Actual max displacement')).toBeInTheDocument()
    expect(screen.getByText('Applied node')).toBeInTheDocument()
    const feaCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/v1/tube/fea-preview'))
    const body = JSON.parse(String(feaCall?.[1]?.body))
    expect(body.config.tube.deformation.areas[0].kind).toBe('fea_c3d6')
    expect(body.longitudinal_samples).toBeGreaterThan(0)
    expect(body.radial_samples).toBeGreaterThan(0)
  })

  it('opens jobs in a modal from the top bar', async () => {
    const user = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())

    expect(screen.queryByPlaceholderText('/tmp/tube_dataset')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate dataset' }))

    expect(screen.getByRole('dialog', { name: 'Generate dataset' })).toBeInTheDocument()
    expect(screen.getByPlaceholderText('outputs')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Browse/ }))
    expect(within(screen.getByRole('dialog', { name: 'Generate dataset' })).getByLabelText('Output directory')).toHaveValue('/tmp/chosen-output')
    await user.click(screen.getByRole('button', { name: /Close/ }))
    expect(screen.queryByRole('dialog', { name: 'Generate dataset' })).not.toBeInTheDocument()
  })

  it('exports the current tube config as canonical JSON', async () => {
    const user = userEvent.setup()
    const createObjectURL = vi.fn((blob: Blob | MediaSource) => {
      void blob
      return 'blob:tube-config'
    })
    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createObjectURL })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeObjectURL })

    render(<App />)
    await waitFor(() => expect(screen.getByText('LumenMorph')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /Export JSON/ }))

    await waitFor(() => expect(createObjectURL).toHaveBeenCalled())
    const exportedBlob = createObjectURL.mock.calls[0]?.[0]
    expect(exportedBlob).toBeInstanceOf(Blob)
    const exported = JSON.parse(await (exportedBlob as Blob).text())
    expect(exported.scene_type).toBe('tube')
    expect(exported.tube.dimensions.shell_thickness).toBe(0.2)
  })
})
