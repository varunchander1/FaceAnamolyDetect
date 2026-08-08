"""Raise a timestamped alert when an unidentified face persists.

Pure — the caller owns the clock and the "already fired" map, so the whole
policy is testable without threads or models.

Two gates, both necessary. Debounce: a single soft or profile frame drops a
known face below threshold (visible in the POC's own capture, where a
`profile right` face fell to unknown for one frame), so one unknown check is
noise. Cooldown: without it a stranger standing in view would emit an alert
every recognition cycle for as long as they stayed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from vision.events import AlertEvent
from vision.facetrack import Track, TrackState
from vision.records import FrameRecord


@dataclass(frozen=True)
class AlertConfig:
    """How stubborn an unknown face must be before it is worth interrupting for."""

    # Consecutive recognition checks, not frames — recognition runs on a
    # cadence, so counting frames would change the meaning of this number every
    # time `recognize_every` moved. At the default cadence of every frame and
    # ~20fps detection, 10 checks is about half a second of sustained doubt.
    min_unknown_checks: int = 10
    cooldown_s: float = 30.0  # per track ID


def unknown_alerts(
    record: FrameRecord,
    state: TrackState,
    fired: Mapping[int, float],
    source_label: str,
    config: AlertConfig = AlertConfig(),
) -> tuple[AlertEvent, ...]:
    """Alerts owed for this frame, newest state and past firings considered.

    Args:
        record: The frame just detected — supplies the stamps and the on-screen
            position of each face.
        state: Tracker state after `with_identity` for this cycle.
        fired: track_id -> wall time of that track's last alert.
        source_label: Verbatim from the sink, so a file is never reported as live.
    """
    if config.min_unknown_checks <= 0:
        raise ValueError("min_unknown_checks must be positive")

    by_id = {t.track_id: t for t in state.tracks}
    now = record.wall_time

    return tuple(
        _alert(face, by_id[face.track_id], record, source_label, now)
        for face in record.faces
        if face.track_id in by_id
        and _is_due(by_id[face.track_id], fired.get(face.track_id), now, config)
    )


def record_fired(
    fired: Mapping[int, float], alerts: tuple[AlertEvent, ...]
) -> dict[int, float]:
    """New map with each alert's track stamped. Never mutates the input."""
    return dict(fired) | {a.track_id: a.timestamp for a in alerts}


def _is_due(
    track: Track, last_fired: float | None, now: float, config: AlertConfig
) -> bool:
    if track.is_known or track.unknown_streak < config.min_unknown_checks:
        return False
    return last_fired is None or (now - last_fired) >= config.cooldown_s


def _alert(
    face, track: Track, record: FrameRecord, source_label: str, now: float
) -> AlertEvent:
    return AlertEvent(
        timestamp=now,
        iso_time=_iso(now),
        track_id=track.track_id,
        source_label=source_label,
        position=face.position,
        checks=track.unknown_streak,
        nearest=track.nearest,
        best_similarity=track.best_similarity,
        media_time=record.media_time,
        frame_index=record.frame_index,
    )


def _iso(wall_time: float) -> str:
    return datetime.fromtimestamp(wall_time).isoformat(timespec="seconds")
