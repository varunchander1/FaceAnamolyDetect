"""Immutable per-frame records shared by the whole pipeline.

Kept separate from `detector` so the pure modules — scene_context, facetrack,
alerts — and their tests never have to import insightface or onnxruntime.
Same reasoning as `events` staying free of LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATUS_MATCHED = "matched"
STATUS_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Pose:
    """Head orientation in degrees, plus a phrase a language model can use."""

    yaw: float
    pitch: float
    roll: float
    facing: str  # "frontal" | "three-quarter left|right" | "profile left|right"


@dataclass(frozen=True)
class Face:
    """One detected face in a frame, with whatever identity we know of it.

    `track_id` comes from our own IoU tracker, not from the models — it is what
    lets the buffer say "newly appeared" and "moved right", and what identity is
    cached against between recognition cycles.
    """

    track_id: int
    xyxy: tuple[float, float, float, float]
    xyxy_norm: tuple[float, float, float, float]
    det_score: float
    position: str  # e.g. "middle-center"
    relative_size: float  # box area as a fraction of the frame
    sharpness: float  # variance of Laplacian; <30 is unusable
    status: str  # STATUS_MATCHED | STATUS_UNKNOWN
    identity: str | None = None
    confidence: float | None = None
    nearest: str = ""  # closest gallery entry, e.g. "Alice 0.31"
    best_similarity: float | None = None
    pose: Pose | None = None
    index: int = 0  # left-to-right rank within the frame

    @property
    def is_known(self) -> bool:
        return self.status == STATUS_MATCHED and self.identity is not None


@dataclass(frozen=True)
class FrameRecord:
    """Immutable result of running detection on a single frame."""

    timestamp: float  # time.monotonic() — a duration basis, never a clock
    faces: tuple[Face, ...]
    annotated: Any  # numpy BGR image with boxes drawn — for DISPLAY only
    frame: Any = None  # the clean BGR frame — this is what the VLM must see
    wall_time: float = 0.0  # time.time() at capture, for human-readable stamps
    media_time: float | None = None  # seconds into an uploaded file; None if live
    frame_index: int = 0

    @property
    def face_count(self) -> int:
        return len(self.faces)
