import numpy as np

from backend.src.deformation import compute_localized_displacement, compute_localized_weight, deformed_z, sample_wave_components
from backend.src.types import DifficultyPreset, LocalizedWaveComponent


def test_deformation_is_deterministic_for_fixed_seed():
    preset = DifficultyPreset(
        name="test",
        wave_count=3,
        amp_min=0.1,
        amp_max=0.3,
        freq_min=0.02,
        freq_max=0.08,
        omega_min=0.5,
        omega_max=1.2,
        drift_scale=1.0,
        jitter_scale=1.0,
    )

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    c1 = sample_wave_components(rng1, preset)
    c2 = sample_wave_components(rng2, preset)

    x, y = np.meshgrid(np.linspace(-1, 1, 8), np.linspace(0, 2, 8))
    base_z = np.zeros_like(x, dtype=np.float32)

    z1, a1, f1, p1 = deformed_z(base_z, x, y, 0.7, c1)
    z2, a2, f2, p2 = deformed_z(base_z, x, y, 0.7, c2)

    assert np.allclose(z1, z2)
    assert a1 == a2
    assert f1 == f2
    assert p1 == p2


def test_localized_displacement_is_spatially_concentrated():
    theta = np.linspace(0.0, 1.0, 96, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")

    components = [
        LocalizedWaveComponent(
            amp=1.0,
            fx=0.0,
            fy=0.0,
            omega=0.0,
            phase=np.pi / 2.0,
            center_s=0.55,
            center_theta=0.22,
            sigma_s=0.06,
            sigma_theta=0.06,
        )
    ]
    disp = compute_localized_displacement(theta_grid, s_grid, t_s=0.0, components=components)

    near_idx = (int(round(0.55 * (disp.shape[0] - 1))), int(round(0.22 * disp.shape[1])))
    far_idx = (int(round(0.90 * (disp.shape[0] - 1))), int(round(0.72 * disp.shape[1])))
    near_val = abs(float(disp[near_idx]))
    far_val = abs(float(disp[far_idx]))
    assert near_val > 0.8
    assert far_val < 0.01


def test_localized_weight_is_bounded_and_localized():
    theta = np.linspace(0.0, 1.0, 96, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")

    components = [
        LocalizedWaveComponent(
            amp=0.5,
            fx=0.0,
            fy=0.0,
            omega=0.0,
            phase=0.0,
            center_s=0.58,
            center_theta=0.18,
            sigma_s=0.07,
            sigma_theta=0.07,
        )
    ]
    weight = compute_localized_weight(theta_grid, s_grid, components=components)

    near_idx = (int(round(0.58 * (weight.shape[0] - 1))), int(round(0.18 * weight.shape[1])))
    far_idx = (int(round(0.92 * (weight.shape[0] - 1))), int(round(0.70 * weight.shape[1])))
    near_val = float(weight[near_idx])
    far_val = float(weight[far_idx])
    assert 0.95 <= near_val <= 1.0
    assert 0.0 <= float(np.min(weight)) <= float(np.max(weight)) <= 1.0
    assert far_val < 0.02


def test_localized_flow_linear_moves_area_center():
    theta = np.linspace(0.0, 1.0, 128, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")
    comp = LocalizedWaveComponent(
        amp=1.0,
        fx=0.0,
        fy=0.0,
        omega=0.0,
        phase=0.0,
        center_s=0.2,
        center_theta=0.3,
        sigma_s=0.06,
        sigma_theta=0.06,
        flow_dir_s=0.45,
        flow_dir_theta=0.0,
    )

    w0 = compute_localized_weight(theta_grid, s_grid, [comp], flow_mode="linear", t_norm=0.0)
    w1 = compute_localized_weight(theta_grid, s_grid, [comp], flow_mode="linear", t_norm=1.0)

    idx0 = np.unravel_index(int(np.argmax(w0)), w0.shape)
    idx1 = np.unravel_index(int(np.argmax(w1)), w1.shape)
    s0 = s[idx0[0]]
    s1 = s[idx1[0]]
    assert s0 < 0.3
    assert s1 > 0.55


def test_localized_flow_cyclic_reverses_direction():
    theta = np.linspace(0.0, 1.0, 128, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")
    comp = LocalizedWaveComponent(
        amp=1.0,
        fx=0.0,
        fy=0.0,
        omega=0.0,
        phase=np.pi / 2.0,
        center_s=0.5,
        center_theta=0.3,
        sigma_s=0.05,
        sigma_theta=0.05,
        flow_dir_s=0.2,
        flow_dir_theta=0.0,
    )

    w_quarter = compute_localized_weight(theta_grid, s_grid, [comp], flow_mode="cyclic", flow_cycles=1.0, t_norm=0.25)
    w_three_quarters = compute_localized_weight(
        theta_grid,
        s_grid,
        [comp],
        flow_mode="cyclic",
        flow_cycles=1.0,
        t_norm=0.75,
    )
    i_q = np.unravel_index(int(np.argmax(w_quarter)), w_quarter.shape)
    i_tq = np.unravel_index(int(np.argmax(w_three_quarters)), w_three_quarters.shape)
    s_q = s[i_q[0]]
    s_tq = s[i_tq[0]]
    assert s_q > 0.6
    assert s_tq < 0.4


def test_localized_transversal_only_ignores_spatial_wave_axes():
    theta = np.linspace(0.0, 1.0, 96, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")

    comp_a = LocalizedWaveComponent(
        amp=0.9,
        fx=4.0,
        fy=-3.5,
        omega=1.4,
        phase=0.2,
        center_s=0.5,
        center_theta=0.5,
        sigma_s=2.0,
        sigma_theta=2.0,
        transversal_only=True,
    )
    comp_b = LocalizedWaveComponent(
        amp=0.9,
        fx=-1.2,
        fy=2.3,
        omega=1.4,
        phase=0.2,
        center_s=0.5,
        center_theta=0.5,
        sigma_s=2.0,
        sigma_theta=2.0,
        transversal_only=True,
    )
    disp_a = compute_localized_displacement(theta_grid, s_grid, t_s=0.8, components=[comp_a])
    disp_b = compute_localized_displacement(theta_grid, s_grid, t_s=0.8, components=[comp_b])
    assert np.allclose(disp_a, disp_b)


def test_localized_asymmetric_inside_outside_amplitudes():
    theta = np.linspace(0.0, 1.0, 96, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 96, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")
    center_idx = (int(round(0.5 * (len(s) - 1))), int(round(0.2 * len(theta))))

    comp = LocalizedWaveComponent(
        amp=0.1,
        amp_out=0.9,
        amp_in=0.25,
        fx=0.0,
        fy=0.0,
        omega=1.0,
        phase=0.0,
        center_s=0.5,
        center_theta=0.2,
        sigma_s=0.08,
        sigma_theta=0.08,
        transversal_only=True,
    )

    disp_out = compute_localized_displacement(theta_grid, s_grid, t_s=np.pi / 2.0, components=[comp])
    disp_in = compute_localized_displacement(theta_grid, s_grid, t_s=3.0 * np.pi / 2.0, components=[comp])
    assert float(disp_out[center_idx]) > 0.75
    assert float(disp_in[center_idx]) < -0.20


def test_transversal_only_ignores_flow_translation():
    theta = np.linspace(0.0, 1.0, 128, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")
    comp = LocalizedWaveComponent(
        amp=0.6,
        fx=0.0,
        fy=0.0,
        omega=0.0,
        phase=np.pi / 2.0,
        center_s=0.45,
        center_theta=0.3,
        sigma_s=0.06,
        sigma_theta=0.06,
        transversal_only=True,
        flow_dir_s=0.45,
        flow_dir_theta=-0.35,
    )

    w0 = compute_localized_weight(theta_grid, s_grid, [comp], flow_mode="linear", t_norm=0.0)
    w1 = compute_localized_weight(theta_grid, s_grid, [comp], flow_mode="linear", t_norm=1.0)
    assert np.allclose(w0, w1)


def test_non_transversal_preserves_radial_component_in_mix():
    theta = np.linspace(0.0, 1.0, 128, endpoint=False, dtype=np.float32)
    s = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    s_grid, theta_grid = np.meshgrid(s, theta, indexing="ij")

    comp = LocalizedWaveComponent(
        amp=1.0,
        fx=3.0,
        fy=2.0,
        omega=1.0,
        phase=0.0,
        center_s=0.5,
        center_theta=0.5,
        sigma_s=2.0,
        sigma_theta=2.0,
        transversal_only=False,
    )
    disp = compute_localized_displacement(theta_grid, s_grid, t_s=0.9, components=[comp])
    # If there were no guaranteed radial term, this mean would be near zero due to spatial cancellation.
    assert abs(float(np.mean(disp))) > 0.12
