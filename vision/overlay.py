"""Draw identity boxes on a frame. Display only — the VLM never sees this.

Replaces ultralytics' `Results.plot()`. Ported from the VideoRAG POC's
`annotate()`, with the colour carrying the thing the operator is watching for:
green means we know who that is, red means we do not.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision.records import Face

KNOWN_COLOR = (0, 200, 0)  # BGR
UNKNOWN_COLOR = (0, 0, 220)
BOX_PX = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_PX = 2
UNKNOWN_TEXT = "UNKNOWN"


def draw_faces(frame: np.ndarray, faces: tuple[Face, ...]) -> np.ndarray:
    """New frame with a labelled box per face. The input is never mutated."""
    out = frame.copy()
    for face in faces:
        color = KNOWN_COLOR if face.is_known else UNKNOWN_COLOR
        x1, y1, x2, y2 = (int(v) for v in face.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_PX)
        cv2.putText(
            out,
            label_for(face),
            (x1, max(y1 - 8, 14)),
            FONT,
            FONT_SCALE,
            color,
            FONT_PX,
            cv2.LINE_AA,
        )
    return out


def label_for(face: Face) -> str:
    """`[0] Alice 0.81` or `[1] UNKNOWN` — the index matches the prompt's ordering."""
    if not face.is_known:
        return f"[{face.index}] {UNKNOWN_TEXT}"
    if face.confidence is None:
        return f"[{face.index}] {face.identity}"
    return f"[{face.index}] {face.identity} {face.confidence:.2f}"
