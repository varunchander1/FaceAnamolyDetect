"""Descriptors that turn a tracked box into something a VLM can reason about.

Pure. Ported from the VideoRAG POC (`extract.py`) — position, pose wording and
sharpness are exactly the fields that let a prompt say "the second person from
the left, turned away" instead of "a face".
"""

from __future__ import annotations

import cv2
import numpy as np

from vision.facetrack import Track
from vision.records import Face, Pose

FRONTAL_YAW = 15.0  # below this, the face reads as looking at the camera
PROFILE_YAW = 40.0  # above this, only one side of the face is visible


def position_label(cx: float, cy: float, width: int, height: int) -> str:
    """Which third of the frame a point falls in, e.g. "middle-center"."""
    if width <= 0 or height <= 0:
        return "middle-center"
    col = "left" if cx < width / 3 else ("right" if cx > 2 * width / 3 else "center")
    row = "top" if cy < height / 3 else ("bottom" if cy > 2 * height / 3 else "middle")
    return f"{row}-{col}"


def blur_score(crop: np.ndarray) -> float:
    """Variance of Laplacian. Under ~100 is visibly soft; under ~30 unusable.

    A soft face is the usual reason a known person scores as unknown, so this
    is what tells an operator whether to distrust an alert.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def describe_pose(pose: np.ndarray | None) -> Pose | None:
    """Turn the model's pitch/yaw/roll into words a language model can use."""
    if pose is None:
        return None
    pitch, yaw, roll = (float(v) for v in pose)

    if abs(yaw) < FRONTAL_YAW:
        facing = "frontal"
    elif abs(yaw) < PROFILE_YAW:
        facing = "three-quarter " + ("right" if yaw > 0 else "left")
    else:
        facing = "profile " + ("right" if yaw > 0 else "left")

    return Pose(
        yaw=round(yaw, 1), pitch=round(pitch, 1), roll=round(roll, 1), facing=facing
    )


def build_face(
    track: Track,
    width: int,
    height: int,
    sharpness: float,
    pose: Pose | None = None,
    index: int = 0,
) -> Face:
    """Assemble the frame-facing view of a track. Never mutates the track."""
    x1, y1, x2, y2 = track.xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    frame_area = float(width * height) or 1.0

    return Face(
        track_id=track.track_id,
        xyxy=track.xyxy,
        xyxy_norm=(
            _ratio(x1, width),
            _ratio(y1, height),
            _ratio(x2, width),
            _ratio(y2, height),
        ),
        det_score=track.det_score,
        position=position_label((x1 + x2) / 2, (y1 + y2) / 2, width, height),
        relative_size=round(area / frame_area, 4),
        sharpness=round(sharpness, 1),
        status=track.status,
        identity=track.identity,
        confidence=track.confidence,
        nearest=track.nearest,
        best_similarity=track.best_similarity,
        pose=pose,
        index=index,
    )


def crop(frame: np.ndarray, xyxy: tuple[float, float, float, float]) -> np.ndarray:
    """Box contents, clipped to the frame. Empty if the box falls outside it."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def _ratio(value: float, extent: int) -> float:
    return round(value / extent, 4) if extent > 0 else 0.0
