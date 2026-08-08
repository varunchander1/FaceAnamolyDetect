"""Face detection + identity, returning immutable per-frame records.

Replaces the two YOLO models with InsightFace `buffalo_l`, loaded model-by-model
rather than through `FaceAnalysis` — that wrapper runs every sub-model on every
face of every frame, which measured 801ms for 6 faces here.

The split that makes this affordable:

    detection (det_10g)      ~126ms  per FRAME  -> every cycle, so boxes stay fresh
    recognition (w600k_r50)   ~69ms  per FACE   -> every Nth cycle
    pose (1k3d68)             ~34ms  per FACE   -> every Nth cycle

Identity and pose are cached on a track ID by `facetrack`, so the frames in
between still carry a name. A face never seen before is always recognised
immediately — waiting on the cadence would delay the stranger alert.

Threading: `Detector` owns mutable tracker state and is NOT thread-safe. One
detector per stream, called from one thread — the same contract the YOLO
tracker needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from vision import facemeta, overlay
from vision.facetrack import (
    DEFAULT_IOU_MIN,
    DEFAULT_MAX_MISSED,
    NEVER_RECOGNIZED,
    Box,
    Track,
    TrackState,
    due_for_recognition,
    update,
    with_identity,
)
from vision.identity import (
    DEFAULT_MARGIN,
    DEFAULT_THRESHOLD,
    Gallery,
    Match,
    load_gallery,
    match,
)
from vision.models import (
    APPLE_PROVIDERS,
    CTX_ID,
    DEFAULT_DET_SIZE,
    DEFAULT_DET_THRESH,
    DET_FILE,
    MODEL_ROOT,
    POSE_FILE,
    REC_FILE,
    load_onnx_model,
    model_pack_dir,
)
from vision.records import Face, FrameRecord, Pose
from vision.settings import DEFAULT_GALLERY_PATH

_log = logging.getLogger(__name__)

# Measured 640x480, CoreML, warm: 50ms/frame (20fps) recognising every face on
# every frame with 1-2 faces, 68ms (15fps) with 6. every=4 only saves ~25%, so
# the default identifies every frame and the knob exists for crowded scenes.
# Note it also stretches the alert debounce, which counts recognition checks.
DEFAULT_RECOGNIZE_EVERY = 1
ALIGNED_FACE_PX = 112


@dataclass(frozen=True)
class DetectorConfig:
    """Everything that changes how a frame is turned into a FrameRecord."""

    gallery_path: str = DEFAULT_GALLERY_PATH
    recognize_every: int = DEFAULT_RECOGNIZE_EVERY
    det_size: tuple[int, int] = DEFAULT_DET_SIZE
    det_thresh: float = DEFAULT_DET_THRESH
    threshold: float = DEFAULT_THRESHOLD  # identity: min cosine similarity
    margin: float = DEFAULT_MARGIN  # identity: min lead over the next person
    with_pose: bool = True
    iou_min: float = DEFAULT_IOU_MIN
    max_missed: int = DEFAULT_MAX_MISSED
    providers: tuple[str, ...] = APPLE_PROVIDERS
    model_root: str = MODEL_ROOT


class Detector:
    """Detects faces every frame, identifies them on a cadence."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        gallery: Gallery | None = None,
    ) -> None:
        self._config = config or DetectorConfig()
        self._gallery = (
            gallery if gallery is not None else load_gallery(self._config.gallery_path)
        )
        self._state = TrackState()

        pack = model_pack_dir(self._config.model_root)
        self._det = load_onnx_model(pack / DET_FILE, self._config.providers)
        self._det.prepare(
            CTX_ID,
            input_size=self._config.det_size,
            det_thresh=self._config.det_thresh,
        )
        self._rec = load_onnx_model(pack / REC_FILE, self._config.providers)
        self._rec.prepare(CTX_ID)

        self._pose = None
        if self._config.with_pose:
            self._pose = load_onnx_model(pack / POSE_FILE, self._config.providers)
            self._pose.prepare(CTX_ID)

    @property
    def config(self) -> DetectorConfig:
        return self._config

    @property
    def gallery(self) -> Gallery:
        return self._gallery

    @property
    def state(self) -> TrackState:
        """Tracker state after the most recent `detect` — what alerting reads."""
        return self._state

    def set_matching(
        self, threshold: float, margin: float, recognize_every: int
    ) -> None:
        """Retune identity without rebuilding the ONNX sessions.

        Called from the script thread while the detector thread is running: the
        swap is a single frozen-dataclass rebind, so a cycle reads either the
        old config or the new one, never a half-applied mix.
        """
        self._config = replace(
            self._config,
            threshold=threshold,
            margin=margin,
            recognize_every=max(1, recognize_every),
        )

    def detect(
        self,
        frame,
        timestamp: float,
        wall_time: float = 0.0,
        media_time: float | None = None,
        frame_index: int = 0,
    ) -> FrameRecord:
        """Run detection (and, on cadence, recognition) on one frame.

        Returns a new FrameRecord; never mutates the input frame.
        """
        boxes, scores, kps_by_box = self._locate(frame)

        self._state = update(
            self._state,
            boxes,
            timestamp=timestamp,
            scores=scores,
            iou_min=self._config.iou_min,
            max_missed=self._config.max_missed,
        )
        self._recognize(frame, kps_by_box)

        faces = self._faces(frame)
        return FrameRecord(
            timestamp=timestamp,
            faces=faces,
            annotated=overlay.draw_faces(frame, faces),
            frame=frame,
            wall_time=wall_time,
            media_time=media_time,
            frame_index=frame_index,
        )

    def reload_gallery(self, gallery: Gallery) -> None:
        """Swap the enrolled faces without rebuilding the ONNX sessions."""
        self._gallery = gallery
        self._state = replace(
            self._state,
            tracks=tuple(_forget_identity(t) for t in self._state.tracks),
        )

    # --- stages -----------------------------------------------------------

    def _locate(self, frame) -> tuple[tuple[Box, ...], tuple[float, ...], dict]:
        """Per-frame detection. `kps_by_box` keys on the box, to rejoin tracks."""
        det, kpss = self._det.detect(frame, max_num=0)
        if det is None or len(det) == 0:
            return (), (), {}

        boxes = tuple(tuple(float(v) for v in row[:4]) for row in det)
        scores = tuple(float(row[4]) for row in det)
        kps_by_box = {} if kpss is None else dict(zip(boxes, kpss))
        return boxes, scores, kps_by_box

    def _recognize(self, frame, kps_by_box: dict) -> None:
        """ArcFace (and pose) for the tracks whose cadence is up."""
        due = due_for_recognition(self._state, self._config.recognize_every)
        if not due:
            return

        tracks = [t for t in self._state.tracks if t.track_id in set(due)]
        crops = [
            (t, kps_by_box[t.xyxy]) for t in tracks if t.xyxy in kps_by_box
        ]
        if not crops:
            return

        matches = self._identify(frame, crops)
        poses = self._poses(frame, [t for t, _ in crops])
        self._state = with_identity(self._state, matches, poses)

    def _identify(self, frame, crops: list) -> dict[int, Match]:
        """One batched session.run for every face due this cycle.

        w600k_r50 declares a batch of 1, so onnxruntime logs "Expected shape
        {1,512} does not match actual shape {N,512}". It is noise: batched and
        one-at-a-time embeddings were verified identical (cosine 1.0, max abs
        diff 1.5e-5). Do not "fix" it by looping — that costs a session.run per
        face.
        """
        from insightface.utils import face_align

        aligned = [
            face_align.norm_crop(frame, landmark=kps, image_size=ALIGNED_FACE_PX)
            for _, kps in crops
        ]
        try:
            embeddings = self._rec.get_feat(aligned)
        except Exception:  # noqa: BLE001 - a bad crop must not kill the stream
            _log.exception("recognition failed for %d face(s)", len(aligned))
            return {}

        return {
            track.track_id: match(
                np.asarray(embeddings[i]).ravel(),
                self._gallery,
                threshold=self._config.threshold,
                margin=self._config.margin,
            )
            for i, (track, _) in enumerate(crops)
        }

    def _poses(self, frame, tracks: list[Track]) -> dict[int, Pose]:
        if self._pose is None:
            return {}

        from insightface.app.common import Face as _ModelFace

        out = {}
        for track in tracks:
            face = _ModelFace(bbox=np.array(track.xyxy, dtype=np.float32))
            try:
                self._pose.get(frame, face)
            except Exception:  # noqa: BLE001 - pose is a nicety, identity is not
                _log.exception("pose failed for track %d", track.track_id)
                continue
            described = facemeta.describe_pose(face.pose)
            if described is not None:
                out[track.track_id] = described
        return out

    def _faces(self, frame) -> tuple[Face, ...]:
        """Visible tracks as frame-facing records, ordered left to right.

        The ordering is what lets a prompt say "the second person from the left"
        and have it mean the same thing as the `[1]` drawn on the video.
        """
        height, width = frame.shape[:2]
        visible = sorted(self._state.visible, key=lambda t: t.xyxy[0])
        return tuple(
            facemeta.build_face(
                track,
                width=width,
                height=height,
                sharpness=facemeta.blur_score(facemeta.crop(frame, track.xyxy)),
                pose=track.pose,
                index=i,
            )
            for i, track in enumerate(visible)
        )


WARMUP_FRAMES = 5


def warmup(detector: Detector, size: tuple[int, int] = (480, 640)) -> None:
    """Run throwaway frames so the first real ones are not the slow ones.

    More than one on purpose: CoreML compiles its partitions lazily and needs
    several inferences to reach steady state. Benchmarking after a single frame
    made CoreML look 3-4x slower than CPU, which is the reverse of the truth.
    """
    blank = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(WARMUP_FRAMES):
        detector.detect(blank, timestamp=float(i))


def _forget_identity(track: Track) -> Track:
    """Force re-recognition against the new gallery on the next cycle."""
    return replace(
        track,
        recognized_cycle=NEVER_RECOGNIZED,
        identity=None,
        confidence=None,
        nearest="",
        best_similarity=None,
        unknown_streak=0,
    )
