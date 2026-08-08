"""Turn a buffer of frame records into grounded text for the VLM.

Pure functions only — no mutation of inputs, no side effects.

This is the *only* place detection reaches the model: one string, injected at
one point (`probe.run_probe`). The model sees the real frames as well, so this
text exists to say what pixels cannot — who a face belongs to, how sure we are,
and what changed over the window.
"""

from __future__ import annotations

from vision.records import Face, FrameRecord

MOTION_THRESHOLD_PX = 40.0  # min x-center shift to call it "moved"

DEFAULT_SOURCE_LABEL = "live camera"


def build_context(
    records: tuple[FrameRecord, ...],
    source_label: str = DEFAULT_SOURCE_LABEL,
) -> str:
    """Compact description of what the buffer window contains.

    Args:
        records: Buffer window, oldest first.
        source_label: What the frames came from, e.g. "live camera" or
            "video file 'clip.mp4'". Goes verbatim into the VLM prompt, so
            it must not claim a live camera when replaying a file.

    Returns a single text block; empty-window safe.
    """
    if not records:
        return "No frames captured yet."

    latest = records[-1]
    window_s = records[-1].timestamp - records[0].timestamp

    parts = [
        f"Over the last ~{window_s:.1f}s of {source_label}:",
        _count_line(latest),
        _known_line(latest),
        _unknown_line(latest),
    ]

    entered = _entered(records)
    if entered:
        parts.append(f"Newly appeared this window: {', '.join(entered)}.")

    motions = _motions(records)
    if motions:
        parts.append("Motion: " + "; ".join(motions) + ".")

    return " ".join(p for p in parts if p)


def _count_line(record: FrameRecord) -> str:
    if not record.faces:
        return "No faces currently visible."
    return f"{record.face_count} face(s) visible, ordered left to right."


def _known_line(record: FrameRecord) -> str:
    known = [f for f in record.faces if f.is_known]
    if not known:
        return ""
    return "Identified: " + ", ".join(_describe(f) for f in known) + "."


def _unknown_line(record: FrameRecord) -> str:
    """Unknowns are named as such deliberately — it is the thing being watched for."""
    unknown = [f for f in record.faces if not f.is_known]
    if not unknown:
        return ""
    return f"Unidentified: {len(unknown)} — " + ", ".join(
        _describe(f) for f in unknown
    ) + "."


def _describe(face: Face) -> str:
    """`[0] Alice (0.81, frontal, middle-center)` — index matches the video label."""
    who = face.identity if face.is_known else "UNKNOWN person"
    details = []
    if face.is_known and face.confidence is not None:
        details.append(f"{face.confidence:.2f}")
    if face.pose is not None:
        details.append(face.pose.facing)
    details.append(face.position)
    return f"[{face.index}] {who} ({', '.join(details)})"


def _entered(records: tuple[FrameRecord, ...]) -> list[str]:
    """Faces present in the latest frame but absent from the first."""
    first_ids = _ids(records[0])
    return [
        _label(f) for f in records[-1].faces if f.track_id not in first_ids
    ]


def _motions(records: tuple[FrameRecord, ...]) -> list[str]:
    """Rough left/right motion per track across the window."""
    first_by_id = _by_id(records[0])
    out = []
    for tid, last in _by_id(records[-1]).items():
        first = first_by_id.get(tid)
        if first is None:
            continue
        dx = _cx(last) - _cx(first)
        if abs(dx) < MOTION_THRESHOLD_PX:
            continue
        out.append(f"{_label(last)} moved {'right' if dx > 0 else 'left'}")
    return out


def _by_id(record: FrameRecord) -> dict[int, Face]:
    return {f.track_id: f for f in record.faces}


def _ids(record: FrameRecord) -> set[int]:
    return {f.track_id for f in record.faces}


def _cx(f: Face) -> float:
    x1, _, x2, _ = f.xyxy
    return (x1 + x2) / 2.0


def _label(f: Face) -> str:
    return f.identity if f.is_known else "an unidentified person"
