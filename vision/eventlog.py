"""Thread-safe, capped logs with JSON/CSV export.

Written by worker threads (the scheduler for probes, the detector for alerts),
read by the Streamlit script thread, so every accessor returns an immutable
snapshot rather than a live view.
"""

from __future__ import annotations

import csv
import io
import json
import threading
from collections import deque
from typing import Generic, Protocol, TypeVar

from vision.events import ALERT_FIELDS, FIELDS, AlertEvent, ProbeEvent

DEFAULT_MAXLEN = 200
DEFAULT_ALERT_MAXLEN = 500


class Exportable(Protocol):
    """Any record that can flatten itself into one export row."""

    def as_dict(self) -> dict: ...


T = TypeVar("T", bound=Exportable)


class RecordLog(Generic[T]):
    """Rolling window of the most recent records (oldest dropped)."""

    def __init__(self, fields: tuple[str, ...], maxlen: int = DEFAULT_MAXLEN) -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._fields = fields
        self._events: deque[T] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, event: T) -> None:
        with self._lock:
            self._events.append(event)

    def extend(self, events: tuple[T, ...]) -> None:
        if not events:
            return
        with self._lock:
            self._events.extend(events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def oldest_first(self) -> tuple[T, ...]:
        with self._lock:
            return tuple(self._events)

    def newest_first(self) -> tuple[T, ...]:
        return tuple(reversed(self.oldest_first()))

    def to_json(self) -> str:
        rows = [e.as_dict() for e in self.oldest_first()]
        return json.dumps(rows, indent=2, ensure_ascii=False)

    def to_csv(self) -> str:
        """CSV with proper quoting — VLM answers contain commas and newlines."""
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(self._fields), lineterminator="\n")
        writer.writeheader()
        for event in self.oldest_first():
            writer.writerow(event.as_dict())
        return out.getvalue()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class EventLog(RecordLog[ProbeEvent]):
    """Probe results, plus the sliding-window memory the next probe reads."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        super().__init__(FIELDS, maxlen)

    def recent_answers(self) -> tuple[str, ...]:
        """Answers of successful probes, oldest first — feeds sliding-window memory."""
        return tuple(e.answer for e in self.oldest_first() if not e.error and e.answer)


class AlertLog(RecordLog[AlertEvent]):
    """Unknown-face alerts. Written by the detector thread, not the scheduler."""

    def __init__(self, maxlen: int = DEFAULT_ALERT_MAXLEN) -> None:
        super().__init__(ALERT_FIELDS, maxlen)
