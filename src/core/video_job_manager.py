"""In-memory controller for background video processing jobs."""

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional


class JobCancelled(Exception):
    """Raised when a video job is cancelled at a safe checkpoint."""


@dataclass
class VideoJob:
    filename: str
    upload_path: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "queued"
    progress: int = 0
    stage: str = "Waiting for a worker"
    frame_count: int = 0
    processed_frames: int = 0
    processed_frames_dir: Optional[str] = field(default=None, repr=False)
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started: bool = False
    resume_event: threading.Event = field(default_factory=threading.Event, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self):
        self.resume_event.set()


JobProcessor = Callable[[VideoJob], Awaitable[None]]
JobCleanup = Callable[[VideoJob], Awaitable[None]]


class VideoJobManager:
    def __init__(self, worker_count: int = 10):
        self.worker_count = max(1, min(worker_count, 10))
        self.jobs: Dict[str, VideoJob] = {}
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.processor: Optional[JobProcessor] = None
        self.cleanup: Optional[JobCleanup] = None
        self.lock = asyncio.Lock()

    async def start(self, processor: JobProcessor, cleanup: JobCleanup):
        self.processor = processor
        self.cleanup = cleanup
        self.workers = [asyncio.create_task(self._worker())
                        for _ in range(self.worker_count)]

    async def stop(self):
        for job in self.jobs.values():
            job.cancel_event.set()
            job.resume_event.set()
        for worker in self.workers:
            worker.cancel()
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers = []

    async def submit(self, filename: str, upload_path: str) -> VideoJob:
        job = VideoJob(filename=filename, upload_path=upload_path)
        async with self.lock:
            self.jobs[job.id] = job
        await self.queue.put(job.id)
        return job

    async def _worker(self):
        while True:
            job_id = await self.queue.get()
            job = None
            try:
                job = self.jobs.get(job_id)
                if job is None:
                    continue
                if job.cancel_event.is_set():
                    await self._cleanup_job(job)
                    continue

                # A paused queued job should not occupy a worker slot while it
                # waits for the user to resume it.
                if not job.started and job.status == "paused":
                    await asyncio.sleep(0.1)
                    await self.queue.put(job.id)
                    continue

                await self.wait_if_paused(job)
                if job.cancel_event.is_set():
                    await self._cleanup_job(job)
                    continue

                job.started = True
                await self.update(job.id, status="running", stage="Starting")
                if self.processor is None:
                    raise RuntimeError("Video job processor is not initialized.")
                await self.processor(job)
            except JobCancelled:
                if job is not None:
                    await self.update(job.id, status="cancelled", progress=0,
                                      stage="Cancelled")
                    await self._cleanup_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if job is not None:
                    await self.update(job.id, status="failed", stage="Failed",
                                      error=str(error))
            finally:
                self.queue.task_done()

    async def _cleanup_job(self, job: VideoJob):
        if self.cleanup is not None:
            await self.cleanup(job)

    async def wait_if_paused(self, job: VideoJob):
        while not job.resume_event.is_set():
            if job.cancel_event.is_set():
                raise JobCancelled()
            await asyncio.sleep(0.2)
        if job.cancel_event.is_set():
            raise JobCancelled()

    async def update(self, job_id: str, **values):
        async with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return
            for key, value in values.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc).isoformat()

    async def pause(self, job_id: str) -> VideoJob:
        async with self.lock:
            job = self._get_job(job_id)
            if job.status in {"queued", "running"}:
                job.resume_event.clear()
                job.status = "paused"
                job.stage = "Paused"
                job.updated_at = datetime.now(timezone.utc).isoformat()
            return job

    async def resume(self, job_id: str) -> VideoJob:
        async with self.lock:
            job = self._get_job(job_id)
            if job.status == "paused":
                job.resume_event.set()
                job.status = "running" if job.started else "queued"
                job.stage = "Resuming"
                job.updated_at = datetime.now(timezone.utc).isoformat()
            return job

    async def cancel(self, job_id: str) -> VideoJob:
        async with self.lock:
            job = self._get_job(job_id)
            if job.status not in {"completed", "failed", "cancelled"}:
                job.cancel_event.set()
                job.resume_event.set()
                job.status = "cancelling" if job.started else "cancelled"
                job.stage = "Cancelling" if job.started else "Cancelled"
                job.updated_at = datetime.now(timezone.utc).isoformat()
            return job

    def _get_job(self, job_id: str) -> VideoJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    async def get(self, job_id: str) -> VideoJob:
        async with self.lock:
            return self._get_job(job_id)

    async def list(self) -> List[VideoJob]:
        async with self.lock:
            return list(self.jobs.values())
