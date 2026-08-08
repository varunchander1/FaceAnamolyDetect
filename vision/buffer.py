"""Rolling buffer of recent frame records."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from vision.detector import FrameRecord

DEFAULT_MAXLEN = 45  # ~3s at 15 fps


class FrameBuffer:
    """Fixed-size rolling window of FrameRecords (oldest dropped)."""

    def __init__(self, maxlen: int = DEFAULT_MAXLEN) -> None:
        self._records: deque[FrameRecord] = deque(maxlen=maxlen)

    def append(self, record: FrameRecord) -> None:
        self._records.append(record)

    def snapshot(self) -> tuple[FrameRecord, ...]:
        """Immutable copy of current contents (oldest first)."""
        return tuple(self._records)

    def latest(self) -> FrameRecord | None:
        return self._records[-1] if self._records else None

    def sample(self, n: int) -> tuple[FrameRecord, ...]:
        """Up to n records evenly spaced across the window (oldest first).

        Always includes the newest frame; fewer than n if the buffer is small.
        """
        records = tuple(self._records)
        if n <= 0 or not records:
            return ()
        if len(records) <= n:
            return records
        step = (len(records) - 1) / (n - 1)
        idxs = sorted({round(i * step) for i in range(n)})
        return tuple(records[i] for i in idxs)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterable[FrameRecord]:
        return iter(self._records)
