"""The immutable records the pipeline emits: VLM probes and unknown-face alerts.

Kept in its own module so `eventlog` can stay free of LangChain/LangGraph
imports — the log is pure data handling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Column order for CSV export and the on-screen table.
FIELDS = (
    "iso_time",
    "answer",
    "error",
    "model",
    "question",
    "scene_context",
    "frame_count",
    "timestamp",
)


@dataclass(frozen=True)
class ProbeEvent:
    """One probe result. `error` is non-empty only when the probe failed."""

    timestamp: float  # time.time() at completion
    iso_time: str
    scene_context: str
    question: str
    answer: str
    model: str
    frame_count: int
    error: str = ""

    def as_dict(self) -> dict:
        """Plain dict in FIELDS order, safe for JSON and CSV."""
        raw = asdict(self)
        return {name: raw[name] for name in FIELDS}


ALERT_FIELDS = (
    "iso_time",
    "media_clock",
    "track_id",
    "detail",
    "position",
    "nearest",
    "best_similarity",
    "checks",
    "source_label",
    "media_time",
    "frame_index",
    "timestamp",
)


@dataclass(frozen=True)
class AlertEvent:
    """An unidentified face that stayed unidentified long enough to matter.

    Raised by the detector thread without waiting on the VLM, so the alert
    latency is detection latency, not model latency.
    """

    timestamp: float  # time.time() when the alert fired
    iso_time: str
    track_id: int
    source_label: str
    position: str  # e.g. "middle-right"
    checks: int  # consecutive recognition checks that came back unknown
    nearest: str = ""  # closest gallery entry, e.g. "Alice 0.31"
    best_similarity: float | None = None  # that entry's score — tune threshold by it
    media_time: float | None = None  # seconds into an uploaded file; None if live
    frame_index: int = 0

    @property
    def media_clock(self) -> str:
        """Offset into the media as h:mm:ss.s, or "live" for a camera."""
        if self.media_time is None:
            return "live"
        hours, rest = divmod(max(0.0, self.media_time), 3600)
        minutes, seconds = divmod(rest, 60)
        return f"{int(hours)}:{int(minutes):02d}:{seconds:04.1f}"

    @property
    def detail(self) -> str:
        return f"Unknown person (track {self.track_id}) at {self.position}"

    def as_dict(self) -> dict:
        """Plain dict in ALERT_FIELDS order, safe for JSON and CSV."""
        raw = asdict(self) | {"media_clock": self.media_clock, "detail": self.detail}
        return {name: raw[name] for name in ALERT_FIELDS}
