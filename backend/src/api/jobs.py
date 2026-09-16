from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..pipeline import GenerationCanceledError


@dataclass
class JobRecord:
    id: str
    kind: str
    status: str = "queued"
    stage: str = "queued"
    percent: float = 0.0
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    result: dict[str, Any] | None = None
    preview_frames: list[bytes] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str) -> None:
        self._manager = manager
        self.job_id = job_id

    @property
    def cancel_event(self) -> threading.Event:
        return self._manager._get_cancel_event(self.job_id)

    def is_canceled(self) -> bool:
        return bool(self.cancel_event.is_set())

    def report(self, stage: str, percent: float, message: str) -> None:
        self._manager._update_progress(
            self.job_id,
            stage=stage,
            percent=percent,
            message=message,
        )
        if self.is_canceled():
            raise GenerationCanceledError("Job canceled")


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._active_job_id: str | None = None

    def create_job(
        self,
        *,
        kind: str,
        runner: Callable[[JobContext], dict[str, Any] | None],
    ) -> str:
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active is not None and active.status in {"queued", "running"}:
                    if kind == "preview" and active.kind == "preview":
                        active.cancel_event.set()
                    else:
                        raise RuntimeError(f"Another job is already active: {active.id} ({active.kind})")

            job_id = uuid.uuid4().hex
            job = JobRecord(id=job_id, kind=kind, status="queued", stage="queued", message="Queued")
            self._jobs[job_id] = job
            self._active_job_id = job_id

        thread = threading.Thread(target=self._run_job_delayed, args=(job_id, runner), daemon=True)
        with self._lock:
            job = self._jobs[job_id]
            job.thread = thread
            job.status = "running"
            job.stage = "starting"
            job.message = "Starting"
            job.updated_at = time.time()
        thread.start()
        return job_id

    def _run_job_delayed(
        self,
        job_id: str,
        runner: Callable[[JobContext], dict[str, Any] | None],
    ) -> None:
        # Let the request handler finish before CPU-heavy render loops compete
        # for the GIL under in-process clients such as FastAPI TestClient.
        time.sleep(0.01)
        self._run_job(job_id, runner)

    def _run_job(
        self,
        job_id: str,
        runner: Callable[[JobContext], dict[str, Any] | None],
    ) -> None:
        ctx = JobContext(self, job_id)
        try:
            output = runner(ctx)
            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_event.is_set():
                    job.status = "canceled"
                    job.stage = "canceled"
                    job.message = "Canceled"
                    job.percent = min(job.percent, 0.999)
                else:
                    job.status = "succeeded"
                    job.stage = "done"
                    job.percent = 1.0
                    job.message = "Completed"
                    if output is not None:
                        if "frames" in output:
                            frames = output.get("frames") or []
                            job.preview_frames = list(frames)
                        if "result" in output:
                            job.result = dict(output.get("result") or {})
                        else:
                            job.result = dict(output)
                job.updated_at = time.time()
        except GenerationCanceledError:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "canceled"
                job.stage = "canceled"
                job.message = "Canceled"
                job.updated_at = time.time()
        except Exception as exc:  # pragma: no cover - defensive runtime behavior
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.stage = "failed"
                job.message = "Failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.updated_at = time.time()
        finally:
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _get_cancel_event(self, job_id: str) -> threading.Event:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.cancel_event

    def _update_progress(self, job_id: str, *, stage: str, percent: float, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.stage = stage
            job.percent = float(np.clip(percent, 0.0, 1.0))
            job.message = message
            job.updated_at = time.time()

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "job_id": job.id,
                "kind": job.kind,
                "status": job.status,
                "stage": job.stage,
                "percent": job.percent,
                "message": job.message,
                "error": job.error,
                "created_at_unix": job.created_at,
                "updated_at_unix": job.updated_at,
                "result_ready": job.result is not None,
                "frame_count": len(job.preview_frames or []),
            }

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job.cancel_event.set()
            job.message = "Cancel requested"
            job.updated_at = time.time()
            return {
                "job_id": job.id,
                "status": job.status,
                "message": "Cancel requested",
            }

    def get_frame(self, job_id: str, frame_idx: int) -> bytes:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.preview_frames is None:
                raise RuntimeError("Frames are not available for this job")
            if frame_idx < 0 or frame_idx >= len(job.preview_frames):
                raise IndexError(frame_idx)
            return job.preview_frames[frame_idx]

    def get_result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.result is None:
                raise RuntimeError("Result is not ready")
            return {
                "job_id": job.id,
                "kind": job.kind,
                "status": job.status,
                "result": job.result,
            }
