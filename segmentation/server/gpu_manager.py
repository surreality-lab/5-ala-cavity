"""FastAPI GPU manager that centralises SAM2 inference across GUI users."""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..engine.sam2_engine import SAM2Engine, SAM2_AVAILABLE

try:
    import torch
except ImportError:  # pragma: no cover - PyTorch required for GPU workloads
    torch = None


# ── utility helpers ────────────────────────────────────────────────


def _default_checkpoint() -> Path | None:
    env = os.environ.get("SAM2_CHECKPOINT")
    if env and Path(env).exists():
        return Path(env)
    fallback = Path("/opt/5-ALA-Videos/weights/sam2_cavity_finetuned.pt")
    if fallback.exists():
        return fallback
    local = Path.home() / "Desktop/VScode/ICG FA/checkpoints/sam2_hiera_large.pt"
    return local if local.exists() else None


def _decode_jpeg(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid JPEG payload")
    return img


def _decode_mask(data: str) -> np.ndarray:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Invalid mask payload")
    return mask.astype(np.uint8)


def _encode_mask(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("Failed to encode mask result")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── configuration + state containers ───────────────────────────────


@dataclass
class ManagerConfig:
    checkpoint: Path | None
    heartbeat_interval: float = 30.0
    session_timeout: float = 120.0
    shutdown_grace: float = 45.0
    host: str = "127.0.0.1"
    port: int = 9777

    @classmethod
    def from_env(cls) -> "ManagerConfig":
        ckpt = _default_checkpoint()
        hb = float(os.environ.get("SAM2_MANAGER_HEARTBEAT", "30"))
        sess = float(os.environ.get("SAM2_MANAGER_SESSION_TIMEOUT", "120"))
        shutdown = float(os.environ.get("SAM2_MANAGER_SHUTDOWN_GRACE", "45"))
        host = os.environ.get("SAM2_MANAGER_HOST", "127.0.0.1")
        port = int(os.environ.get("SAM2_MANAGER_PORT", "9777"))
        return cls(checkpoint=ckpt, heartbeat_interval=hb, session_timeout=sess, shutdown_grace=shutdown, host=host, port=port)


@dataclass
class SessionRecord:
    session_id: str
    user: str | None
    host: str | None
    pid: int | None
    last_seen: float


class SessionRegistry:
    def __init__(self, heartbeat_timeout: float):
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()
        self._timeout = heartbeat_timeout

    async def register(self, user: str | None, host: str | None, pid: int | None) -> SessionRecord:
        session_id = os.urandom(6).hex()
        record = SessionRecord(session_id, user, host, pid, time.time())
        async with self._lock:
            self._sessions[session_id] = record
        return record

    async def heartbeat(self, session_id: str) -> bool:
        async with self._lock:
            rec = self._sessions.get(session_id)
            if not rec:
                return False
            rec.last_seen = time.time()
            return True

    async def release(self, session_id: str):
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def prune(self) -> list[str]:
        now = time.time()
        expired: list[str] = []
        async with self._lock:
            for sid, rec in list(self._sessions.items()):
                if now - rec.last_seen > self._timeout:
                    expired.append(sid)
                    self._sessions.pop(sid, None)
        return expired

    async def active_count(self) -> int:
        async with self._lock:
            return len(self._sessions)


class FairScheduler:
    """Ensures GPU work is interleaved by session."""

    def __init__(self):
        self._queues: dict[str, deque] = defaultdict(deque)
        self._rotation: deque[str] = deque()
        self._cv = asyncio.Condition()

    async def put(self, session_id: str, job: "Job"):
        async with self._cv:
            q = self._queues[session_id]
            q.append(job)
            if session_id not in self._rotation:
                self._rotation.append(session_id)
            self._cv.notify()

    async def get(self) -> "Job":
        async with self._cv:
            while True:
                for _ in range(len(self._rotation)):
                    session_id = self._rotation[0]
                    queue = self._queues.get(session_id)
                    if queue:
                        job = queue.popleft()
                        self._rotation.rotate(-1)
                        return job
                    else:
                        self._rotation.popleft()
                await self._cv.wait()

    def remove_session(self, session_id: str):
        queue = self._queues.pop(session_id, None)
        if queue:
            for job in queue:
                if not job.future.done():
                    job.future.set_exception(HTTPException(status_code=410, detail="Session expired"))
        if session_id in self._rotation:
            self._rotation.remove(session_id)

    def pending_jobs(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def is_idle(self) -> bool:
        return self.pending_jobs() == 0


class GPUEnginePool:
    def __init__(self, engines: list[SAM2Engine], devices: list[str]):
        self.engines = engines
        self.devices = devices
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for idx in range(len(engines)):
            self._available.put_nowait(idx)

    @classmethod
    async def create(cls, checkpoint: Path | None) -> "GPUEnginePool":
        if not SAM2_AVAILABLE:
            raise RuntimeError("SAM2 not installed in GPU manager environment")
        devices = _discover_devices()
        if not devices:
            raise RuntimeError("No GPU devices detected for SAM2 manager")
        loop = asyncio.get_running_loop()
        engines: list[SAM2Engine] = []
        for dev in devices:
            engine = await loop.run_in_executor(None, lambda d=dev: SAM2Engine(checkpoint, device=d))
            if not engine.available:
                raise RuntimeError(f"Failed to load SAM2 on {dev}")
            engines.append(engine)
        return cls(engines, devices)

    async def acquire(self) -> tuple[int, SAM2Engine]:
        idx = await self._available.get()
        return idx, self.engines[idx]

    def release(self, idx: int):
        self._available.put_nowait(idx)

    def shutdown(self):
        for eng in self.engines:
            eng.shutdown()


def _discover_devices() -> list[str]:
    if torch is None or not torch.cuda.is_available():
        return ["cpu"]
    count = torch.cuda.device_count()
    return [f"cuda:{i}" for i in range(count)] if count else ["cuda"]


# ── job representation ─────────────────────────────────────────────


class Job:
    def __init__(self, session_id: str, kind: str, payload: dict[str, Any]):
        self.session_id = session_id
        self.kind = kind
        self.payload = payload
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()

    def run(self, engine: SAM2Engine):
        if self.kind == "predict_click":
            frame = _decode_jpeg(self.payload["frame_jpeg"])
            points_pos = [tuple(pt) for pt in self.payload.get("points_pos", [])]
            points_neg = [tuple(pt) for pt in self.payload.get("points_neg", [])]
            engine.set_image(frame)
            mask = engine.predict_mask(points_pos, points_neg)
            return {"mask": _encode_mask(mask)} if mask is not None else {"mask": None}
        if self.kind == "propagate_single":
            src = _decode_jpeg(self.payload["src_frame"])
            dst = _decode_jpeg(self.payload["dst_frame"])
            masks = [
                (int(entry["obj_id"]), _decode_mask(entry["mask"]))
                for entry in self.payload.get("masks", [])
            ]
            results = engine.propagate_single(src, dst, masks)
            return {
                "masks": [
                    {"obj_id": oid, "mask": _encode_mask(mask)}
                    for oid, mask in results.items()
                ]
            }
        if self.kind == "propagate_batch":
            frames = [_decode_jpeg(blob) for blob in self.payload.get("frames", [])]
            masks = [
                (int(entry["obj_id"]), _decode_mask(entry["mask"]))
                for entry in self.payload.get("masks", [])
            ]
            results = engine.propagate_batch(frames, masks)
            serialised: list[dict[str, Any]] = []
            for sam_idx, frame_masks in results.items():
                serialised.append({
                    "sam_index": sam_idx,
                    "masks": [
                        {"obj_id": oid, "mask": _encode_mask(mask)}
                        for oid, mask in frame_masks.items()
                    ],
                })
            return {"results": serialised}
        raise ValueError(f"Unknown job kind {self.kind}")


# ── FastAPI wiring ─────────────────────────────────────────────────


class PredictPayload(BaseModel):
    session_id: str = Field(..., description="Active session token")
    points_pos: list[list[int]] = Field(default_factory=list)
    points_neg: list[list[int]] = Field(default_factory=list)
    frame_jpeg: str


class PropagateSinglePayload(BaseModel):
    session_id: str
    src_frame: str
    dst_frame: str
    masks: list[dict[str, Any]]


class PropagateBatchPayload(BaseModel):
    session_id: str
    frames: list[str]
    masks: list[dict[str, Any]]


class SessionMeta(BaseModel):
    user: str | None = None
    host: str | None = None
    pid: int | None = None


class GPUManagerState:
    def __init__(self, config: ManagerConfig):
        self.config = config
        self.registry = SessionRegistry(config.session_timeout)
        self.scheduler = FairScheduler()
        self.pool: GPUEnginePool | None = None
        self._workers: list[asyncio.Task] = []
        self._shutdown_task: asyncio.Task | None = None
        self._prune_task: asyncio.Task | None = None

    async def startup(self):
        self.pool = await GPUEnginePool.create(self.config.checkpoint)
        for _ in range(len(self.pool.engines)):
            self._workers.append(asyncio.create_task(self._worker()))
        self._prune_task = asyncio.create_task(self._prune_loop())
        print("GPU manager ready on devices:", ", ".join(self.pool.devices))

    async def shutdown(self):
        for task in self._workers:
            task.cancel()
        if self._prune_task:
            self._prune_task.cancel()
        if self.pool:
            self.pool.shutdown()

    async def _worker(self):
        while True:
            job = await self.scheduler.get()
            idx, engine = await self.pool.acquire()
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, job.run, engine)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:  # pragma: no cover - defensive path
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self.pool.release(idx)

    async def _prune_loop(self):
        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            expired = await self.registry.prune()
            for sid in expired:
                self.scheduler.remove_session(sid)
            await self._maybe_schedule_shutdown()

    async def enqueue(self, job: Job) -> dict[str, Any]:
        await self.scheduler.put(job.session_id, job)
        return await job.future

    async def _maybe_schedule_shutdown(self):
        if self._shutdown_task and not self._shutdown_task.done():
            return
        if await self.registry.active_count() > 0 or not self.scheduler.is_idle():
            return
        async def _delayed():
            await asyncio.sleep(self.config.shutdown_grace)
            if await self.registry.active_count() == 0 and self.scheduler.is_idle():
                os.kill(os.getpid(), signal.SIGTERM)
        self._shutdown_task = asyncio.create_task(_delayed())

    def cancel_shutdown(self):
        if self._shutdown_task and not self._shutdown_task.done():
            self._shutdown_task.cancel()
            self._shutdown_task = None


def create_app(config: ManagerConfig) -> FastAPI:
    manager = GPUManagerState(config)
    app = FastAPI(title="SAM2 GPU Manager", version="1.0")

    @app.on_event("startup")
    async def _on_startup():
        await manager.startup()

    @app.on_event("shutdown")
    async def _on_shutdown():
        await manager.shutdown()

    @app.get("/health")
    async def health():
        active = await manager.registry.active_count()
        return {
            "sessions": active,
            "queue": manager.scheduler.pending_jobs(),
            "devices": manager.pool.devices if manager.pool else [],
            "weights_label": manager.pool.engines[0].weights_label if manager.pool else "unknown",
        }

    @app.post("/sessions/register")
    async def register(meta: SessionMeta):
        manager.cancel_shutdown()
        record = await manager.registry.register(meta.user, meta.host, meta.pid)
        if not manager.pool:
            raise HTTPException(status_code=503, detail="GPU pool unavailable")
        return {
            "session_id": record.session_id,
            "device_label": ", ".join(manager.pool.devices),
            "weights_label": manager.pool.engines[0].weights_label,
            "heartbeat_interval": manager.config.heartbeat_interval,
        }

    @app.post("/sessions/heartbeat")
    async def heartbeat(meta: dict[str, str]):
        sid = meta.get("session_id")
        if not sid:
            raise HTTPException(status_code=400, detail="Missing session_id")
        ok = await manager.registry.heartbeat(sid)
        if not ok:
            raise HTTPException(status_code=404, detail="Unknown session")
        return {"status": "ok"}

    @app.post("/sessions/release")
    async def release(meta: dict[str, str]):
        sid = meta.get("session_id")
        if sid:
            await manager.registry.release(sid)
            manager.scheduler.remove_session(sid)
            await manager._maybe_schedule_shutdown()
        return {"status": "released"}

    async def _validate_session(session_id: str):
        ok = await manager.registry.heartbeat(session_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Unknown session")

    @app.post("/predict_click")
    async def predict(payload: PredictPayload):
        await _validate_session(payload.session_id)
        job = Job(payload.session_id, "predict_click", payload.dict())
        return await manager.enqueue(job)

    @app.post("/propagate_single")
    async def propagate_single(payload: PropagateSinglePayload):
        await _validate_session(payload.session_id)
        job = Job(payload.session_id, "propagate_single", payload.dict())
        return await manager.enqueue(job)

    @app.post("/propagate_batch")
    async def propagate_batch(payload: PropagateBatchPayload):
        await _validate_session(payload.session_id)
        job = Job(payload.session_id, "propagate_batch", payload.dict())
        return await manager.enqueue(job)

    return app


DEFAULT_CONFIG = ManagerConfig.from_env()
app = create_app(DEFAULT_CONFIG)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="SAM2 GPU Manager")
    parser.add_argument("--host", default=DEFAULT_CONFIG.host)
    parser.add_argument("--port", default=DEFAULT_CONFIG.port, type=int)
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CONFIG.checkpoint) if DEFAULT_CONFIG.checkpoint else None)
    args = parser.parse_args(argv)

    cfg = ManagerConfig(
        checkpoint=Path(args.checkpoint) if args.checkpoint else DEFAULT_CONFIG.checkpoint,
        heartbeat_interval=DEFAULT_CONFIG.heartbeat_interval,
        session_timeout=DEFAULT_CONFIG.session_timeout,
        shutdown_grace=DEFAULT_CONFIG.shutdown_grace,
        host=args.host,
        port=args.port,
    )

    import uvicorn

    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()
