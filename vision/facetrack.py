"""Greedy IoU tracking of face boxes across frames.

Pure: every function returns a new `TrackState` and never touches the one it
was given.

Why this exists at all: the POC analyses every frame independently, so it can
never say "someone new walked in" or "that face moved right", and it would have
to re-run ArcFace on every face of every frame. A track ID fixes both — it is
the temporal thread the scene context reads, and the slot the recognition
result is cached in between cadence ticks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from vision.identity import Match
from vision.records import STATUS_MATCHED, STATUS_UNKNOWN, Pose

DEFAULT_IOU_MIN = 0.3
DEFAULT_MAX_MISSED = 5  # cycles a face may vanish for before its ID retires

Box = tuple[float, float, float, float]

NEVER_RECOGNIZED = -1


@dataclass(frozen=True)
class Track:
    """One face followed across frames, carrying its last known identity."""

    track_id: int
    xyxy: Box
    first_seen: float
    last_seen: float
    seen: int = 1  # frames this track was matched in
    missed: int = 0  # consecutive frames it was absent from
    det_score: float = 0.0  # detector confidence in the most recent frame
    status: str = STATUS_UNKNOWN
    identity: str | None = None
    confidence: float | None = None
    nearest: str = ""  # closest gallery entry, e.g. "Alice 0.31"
    best_similarity: float | None = None
    unknown_streak: int = 0  # consecutive recognition checks that said unknown
    recognized_cycle: int = NEVER_RECOGNIZED
    pose: Pose | None = None  # last measured head orientation, cached like identity

    @property
    def is_visible(self) -> bool:
        return self.missed == 0

    @property
    def is_known(self) -> bool:
        return self.status == STATUS_MATCHED and self.identity is not None


@dataclass(frozen=True)
class TrackState:
    """All live tracks plus the counters that make IDs and cadence work."""

    tracks: tuple[Track, ...] = ()
    next_id: int = 1
    cycle: int = 0

    @property
    def visible(self) -> tuple[Track, ...]:
        return tuple(t for t in self.tracks if t.is_visible)


def iou(a: Box, b: Box) -> float:
    """Intersection over union. Degenerate boxes score 0 rather than dividing by 0."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    inter = inter_w * inter_h
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def update(
    state: TrackState,
    boxes: tuple[Box, ...],
    timestamp: float,
    scores: tuple[float, ...] | None = None,
    iou_min: float = DEFAULT_IOU_MIN,
    max_missed: int = DEFAULT_MAX_MISSED,
) -> TrackState:
    """Associate this frame's boxes with existing tracks; return a new state.

    Greedy highest-IoU-first, which is enough for faces: they are few, roughly
    the same size, and do not cross each other at detector cadence.
    """
    if scores is not None and len(scores) != len(boxes):
        raise ValueError(f"{len(scores)} scores for {len(boxes)} boxes")

    def score_of(j: int) -> float:
        return 0.0 if scores is None else scores[j]

    pairs = _greedy_pairs(state.tracks, boxes, iou_min)
    matched_boxes = set(pairs.values())

    kept = []
    for i, track in enumerate(state.tracks):
        if i in pairs:
            kept.append(
                replace(
                    track,
                    xyxy=boxes[pairs[i]],
                    det_score=score_of(pairs[i]),
                    last_seen=timestamp,
                    seen=track.seen + 1,
                    missed=0,
                )
            )
        elif track.missed + 1 <= max_missed:
            kept.append(replace(track, missed=track.missed + 1))

    next_id = state.next_id
    for j, box in enumerate(boxes):
        if j in matched_boxes:
            continue
        kept.append(
            Track(
                track_id=next_id,
                xyxy=box,
                det_score=score_of(j),
                first_seen=timestamp,
                last_seen=timestamp,
            )
        )
        next_id += 1

    return TrackState(tracks=tuple(kept), next_id=next_id, cycle=state.cycle + 1)


def with_identity(
    state: TrackState,
    results: Mapping[int, Match],
    poses: Mapping[int, Pose] | None = None,
) -> TrackState:
    """Fold recognition results onto their tracks; return a new state.

    `unknown_streak` counts consecutive *checks*, not frames — recognition runs
    on a cadence, so a frame count here would depend on `recognize_every` and
    make the alert debounce mean something different at every setting.

    Pose rides along because it is measured by the same expensive per-face pass
    and must survive the frames in between, exactly like identity.
    """
    poses = poses or {}
    if not results and not poses:
        return state

    updated = tuple(
        _apply(track, results.get(track.track_id), poses.get(track.track_id), state.cycle)
        for track in state.tracks
    )
    return replace(state, tracks=updated)


def due_for_recognition(state: TrackState, every: int) -> tuple[int, ...]:
    """Visible tracks needing ArcFace this cycle.

    A face never seen before is always included: waiting up to `every` cycles to
    learn whether a new arrival is a stranger would delay the alert we exist to
    raise.
    """
    step = max(1, every)
    return tuple(
        t.track_id
        for t in state.visible
        if t.recognized_cycle == NEVER_RECOGNIZED
        or state.cycle - t.recognized_cycle >= step
    )


def _apply(
    track: Track, match: Match | None, pose: Pose | None, cycle: int
) -> Track:
    updated = track if pose is None else replace(track, pose=pose)
    if match is None:
        return updated

    known = match.status == STATUS_MATCHED
    return replace(
        updated,
        status=match.status,
        identity=match.identity,
        confidence=match.confidence,
        nearest=match.nearest,
        best_similarity=match.best_similarity,
        unknown_streak=0 if known else track.unknown_streak + 1,
        recognized_cycle=cycle,
    )


def _greedy_pairs(
    tracks: tuple[Track, ...], boxes: tuple[Box, ...], iou_min: float
) -> dict[int, int]:
    """Map track index -> box index, best overlap first."""
    candidates = sorted(
        (
            (iou(track.xyxy, box), i, j)
            for i, track in enumerate(tracks)
            for j, box in enumerate(boxes)
            if iou(track.xyxy, box) >= iou_min
        ),
        reverse=True,
    )

    pairs: dict[int, int] = {}
    taken_boxes: set[int] = set()
    for _, i, j in candidates:
        if i in pairs or j in taken_boxes:
            continue
        pairs[i] = j
        taken_boxes.add(j)
    return pairs


def _area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
