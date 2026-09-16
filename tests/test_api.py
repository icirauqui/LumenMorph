from dataclasses import replace
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from backend.api import create_app
import backend.src.api.app as api_app
from backend.src.config import canonical_config_dict, load_config


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _sleep(seconds: float) -> None:
    import anyio

    await anyio.sleep(seconds)


async def _poll_status(client, job_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = await client.get(f"/v1/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        if payload["status"] in {"succeeded", "failed", "canceled"}:
            return payload
        await _sleep(0.1)
    raise AssertionError(f"job did not finish in time: {job_id}")


@pytest.mark.anyio
async def test_api_health_schema_presets_preview_and_generate(tmp_path):
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True
        assert health.json()["service"] == "lumenmorph-local-api"
        assert health.json()["version"] == "0.1.0"

        schema = await client.get("/v1/schema")
        assert schema.status_code == 200
        assert "tube.deformation.areas" in schema.json()["groups"]["tube"]

        presets = await client.get("/v1/presets")
        assert presets.status_code == 200
        assert len(presets.json()["presets"]) > 0

        output_defaults = await client.get("/v1/output/defaults")
        assert output_defaults.status_code == 200
        assert output_defaults.json()["default_output_dir"].endswith("/outputs")

        normalize_bad = await client.post("/v1/config/normalize", json={"config": {"seed": 1}})
        assert normalize_bad.status_code == 200
        assert normalize_bad.json()["ok"] is False

        tube_cfg = load_config("configs/tube_default.json")
        assert tube_cfg.tube is not None
        tube_cfg = replace(
            tube_cfg,
            sequence_count={"train": 1, "val": 0, "test": 0},
            intrinsics=replace(tube_cfg.intrinsics, width=160, height=120, fx=130.0, fy=130.0, cx=80.0, cy=60.0),
            sequence=replace(tube_cfg.sequence, num_frames=6, fps=8),
            tube=replace(tube_cfg.tube, radial_samples=24, longitudinal_samples=28, bump_count=10),
        )
        preview_start = await client.post(
            "/v1/preview/jobs",
            json={
                "config": canonical_config_dict(tube_cfg),
                "frame_count": 4,
                "fps": 8,
                "preview_camera": {"yaw_deg": 40.0, "pitch_deg": 18.0, "width": 192, "height": 128},
            },
        )
        assert preview_start.status_code == 200, preview_start.text
        preview_job_id = preview_start.json()["job_id"]
        preview_status = await _poll_status(client, preview_job_id)
        assert preview_status["status"] == "succeeded", preview_status
        assert preview_status["frame_count"] == 4

        frame0 = await client.get(f"/v1/jobs/{preview_job_id}/frames/0")
        assert frame0.status_code == 200
        assert frame0.headers["content-type"] == "image/png"
        assert len(frame0.content) > 100

        plane_cfg = load_config("configs/default.json")
        plane_cfg = replace(
            plane_cfg,
            sequence_count={"train": 1, "val": 0, "test": 0},
            intrinsics=replace(plane_cfg.intrinsics, width=160, height=120, fx=130.0, fy=130.0, cx=80.0, cy=60.0),
            sequence=replace(plane_cfg.sequence, num_frames=6, fps=8, mesh_nx=20, mesh_ny=20),
        )
        out_dir = tmp_path / "api_gen_out"
        gen_start = await client.post(
            "/v1/generate/jobs",
            json={
                "config": canonical_config_dict(plane_cfg),
                "output_dir": str(out_dir),
                "mode": "quick",
                "quick": {"max_sequences": 1, "override_num_frames": 4, "override_fps": 8},
                "show_progress": False,
            },
        )
        assert gen_start.status_code == 200, gen_start.text
        gen_job_id = gen_start.json()["job_id"]
        gen_status = await _poll_status(client, gen_job_id, timeout_s=60.0)
        assert gen_status["status"] == "succeeded", gen_status

        gen_result = await client.get(f"/v1/jobs/{gen_job_id}/result")
        assert gen_result.status_code == 200
        assert gen_result.json()["result"]["mode"] == "quick"
        assert (out_dir / "dataset_manifest.json").exists()


@pytest.mark.anyio
async def test_generate_job_uses_default_outputs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api_app, "ROOT_DIR", tmp_path)
    transport = httpx.ASGITransport(app=api_app.create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        defaults = await client.get("/v1/output/defaults")
        assert defaults.status_code == 200
        expected_out = tmp_path / "outputs"
        assert defaults.json()["default_output_dir"] == str(expected_out)

        plane_cfg = load_config("configs/default.json")
        plane_cfg = replace(
            plane_cfg,
            sequence_count={"train": 1, "val": 0, "test": 0},
            intrinsics=replace(plane_cfg.intrinsics, width=96, height=72, fx=80.0, fy=80.0, cx=48.0, cy=36.0),
            sequence=replace(plane_cfg.sequence, num_frames=3, fps=6, mesh_nx=12, mesh_ny=12),
        )
        gen_start = await client.post(
            "/v1/generate/jobs",
            json={
                "config": canonical_config_dict(plane_cfg),
                "mode": "quick",
                "quick": {"max_sequences": 1, "override_num_frames": 2, "override_fps": 6},
                "show_progress": False,
            },
        )
        assert gen_start.status_code == 200, gen_start.text
        gen_job_id = gen_start.json()["job_id"]
        gen_status = await _poll_status(client, gen_job_id, timeout_s=60.0)
        assert gen_status["status"] == "succeeded", gen_status

        gen_result = await client.get(f"/v1/jobs/{gen_job_id}/result")
        assert gen_result.status_code == 200
        assert gen_result.json()["result"]["output_dir"] == str(expected_out)
        assert (expected_out / "dataset_manifest.json").exists()


@pytest.mark.anyio
async def test_realtime_preview_session_lifecycle_and_full_preview():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tube_cfg = load_config("configs/tube_straight_localized_one_area.json")
        tube_cfg = replace(
            tube_cfg,
            sequence_count={"train": 1, "val": 0, "test": 0},
            sequence=replace(tube_cfg.sequence, num_frames=12, fps=10),
        )
        payload = canonical_config_dict(tube_cfg)

        created = await client.post(
            "/v1/preview/sessions",
            json={
                "config": payload,
                "preview_camera": {"width": 256, "height": 144, "yaw_deg": 40.0, "pitch_deg": 18.0},
                "playback": {"playing": False, "fps": 20, "frame_count": 90, "t_norm": 0.0},
                "draft_profile": True,
            },
        )
        assert created.status_code == 200, created.text
        session_id = created.json()["session_id"]

        got = await client.get(f"/v1/preview/sessions/{session_id}")
        assert got.status_code == 200
        assert got.json()["session_id"] == session_id

        latest = await client.get(f"/v1/preview/sessions/{session_id}/frame/latest")
        assert latest.status_code in {200, 202}
        if latest.status_code == 200:
            assert latest.headers.get("content-type", "").startswith("image/jpeg")
            assert len(latest.content) > 50

        patched = await client.patch(
            f"/v1/preview/sessions/{session_id}/config",
            json={"config": payload},
        )
        assert patched.status_code == 200, patched.text

        playback = await client.patch(
            f"/v1/preview/sessions/{session_id}/playback",
            json={"playing": True, "fps": 24, "direction": 1},
        )
        assert playback.status_code == 200, playback.text

        full_job = await client.post(
            f"/v1/preview/sessions/{session_id}/full-preview/jobs",
            json={"frame_count": 3, "fps": 8, "preview_camera": {"width": 192, "height": 108}},
        )
        assert full_job.status_code == 200, full_job.text
        full_job_id = full_job.json()["job_id"]
        status = await _poll_status(client, full_job_id, timeout_s=60.0)
        assert status["status"] == "succeeded", status

        frame0 = await client.get(f"/v1/jobs/{full_job_id}/frames/0")
        assert frame0.status_code == 200
        assert frame0.headers["content-type"] == "image/png"

        deleted = await client.delete(f"/v1/preview/sessions/{session_id}")
        assert deleted.status_code == 200


@pytest.mark.anyio
async def test_tube_fea_preview_endpoint_returns_surface_displacement():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cfg = load_config("configs/tube_fea_c3d6_one_area.json")
        payload = canonical_config_dict(cfg)
        payload["tube"]["dimensions"]["longitudinal_samples"] = 12
        payload["tube"]["dimensions"]["radial_samples"] = 12
        payload["tube"]["deformation"]["fea"] = {
            "young_modulus_pa": 1.0e6,
            "poisson_ratio": 0.30,
            "axial_elements": 4,
            "theta_elements": 8,
        }
        payload["tube"]["deformation"]["areas"][0]["force_n"] = 100.0

        response = await client.post(
            "/v1/tube/fea-preview",
            json={"config": payload, "longitudinal_samples": 10, "radial_samples": 8},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["longitudinal_samples"] == 10
        assert body["radial_samples"] == 8
        assert len(body["responses"]) == 1
        assert body["responses"][0]["applied_node"] >= 0
        assert body["responses"][0]["applied_axial_index"] >= 1
        assert body["responses"][0]["applied_theta_index"] >= 0
        grid = body["responses"][0]["displacement_grid"]
        assert len(grid) == 10
        assert len(grid[0]) == 8
        assert len(grid[0][0]) == 3
        assert max(abs(value) for row in grid for point in row for value in point) > 0.0


@pytest.mark.anyio
async def test_api_job_cancel_preview():
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tube_cfg = load_config("configs/tube_default.json")
        tube_cfg = replace(
            tube_cfg,
            sequence_count={"train": 1, "val": 0, "test": 0},
            intrinsics=replace(tube_cfg.intrinsics, width=128, height=96, fx=100.0, fy=100.0, cx=64.0, cy=48.0),
            sequence=replace(tube_cfg.sequence, num_frames=8, fps=8),
        )
        start = await client.post(
            "/v1/preview/jobs",
            json={
                "config": canonical_config_dict(tube_cfg),
                "frame_count": 48,
                "fps": 8,
                "preview_camera": {"width": 160, "height": 100},
            },
        )
        assert start.status_code == 200, start.text
        job_id = start.json()["job_id"]

        cancel = await client.post(f"/v1/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        status = await _poll_status(client, job_id, timeout_s=20.0)
        assert status["status"] in {"canceled", "succeeded"}
