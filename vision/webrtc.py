"""WebRTC frame sink: the browser owns the camera, we only process frames.

Replaces the OpenCV capture path for display. `streamlit-webrtc` negotiates a
peer connection and hands us each frame on its own worker thread; we return a
frame to send back, which the browser renders in a real `<video>` element.

Why this instead of reading the camera in Python:
  - the browser owns the device, so a stale Python process can no longer hold
    the camera hostage and leave the next run with a black picture;
  - playback is a genuine video element, not a poll-and-repaint loop.

Detection still costs tens of milliseconds per frame, so it must not run inside
the callback — that would throttle the video to detector speed. The callback
hands the frame to a one-slot mailbox and returns immediately; a worker thread
detects on whatever the newest frame is and publishes the annotated result,
which the next callback sends back in place of the raw frame. Only the detected
view is streamed — the browser already has the raw picture, so shipping it back
alongside doubled the frame height for nothing.

Unknown-face alerts are raised here, on the detector thread, rather than in the
probe path — an intruder must not wait on a VLM round trip to be reported.

Threading contract: nothing here may touch Streamlit.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import cv2

from vision.alerts import AlertConfig, record_fired, unknown_alerts
from vision.buffer import FrameBuffer
from vision.detector import Detector, warmup
from vision.eventlog import AlertLog
from vision.records import FrameRecord

_log = logging.getLogger(__name__)

DETECT_IDLE_WAIT_S = 0.05


@dataclass(frozen=True)
class PendingFrame:
    """A frame plus the stamps an alert needs to say *when* it happened.

    `media_time` is what makes an alert on an uploaded file point at an offset
    into the clip rather than at the wall clock of the replay.
    """

    frame: Any
    wall_time: float
    media_time: float | None
    frame_index: int


class _LatestFrame:
    """One-slot mailbox: the newest frame wins, older ones are dropped."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self._pending: PendingFrame | None = None

    def put(self, pending: PendingFrame) -> None:
        with self._lock:
            self._pending = pending
        self._arrived.set()

    def take(self, timeout: float) -> PendingFrame | None:
        if not self._arrived.wait(timeout):
            return None
        with self._lock:
            pending, self._pending = self._pending, None
            self._arrived.clear()
        return pending

    def wake(self) -> None:
        self._arrived.set()


class FrameSink:
    """Consumes WebRTC frames, runs detection off-thread, feeds the buffer.

    Satisfies `probe.SceneSource`, so the scheduler treats it exactly like the
    old CaptureWorker.
    """

    def __init__(
        self,
        source_label: str,
        detector_factory: Callable[[], Detector] = Detector,
        buffer: FrameBuffer | None = None,
        alert_log: AlertLog | None = None,
        alert_config: AlertConfig | None = None,
    ) -> None:
        self.source_label = source_label
        self._detector_factory = detector_factory
        self._buffer = buffer or FrameBuffer()
        self._alert_log = alert_log if alert_log is not None else AlertLog()
        self._alert_config = alert_config or AlertConfig()

        self._pending = _LatestFrame()
        self._stopping = threading.Event()
        self._lock = threading.Lock()

        self._frames_seen = 0
        self._detections = 0
        self._error = ""
        self._annotated: Any = None
        self._fired: dict[int, float] = {}  # detector thread only
        self._detector: Detector | None = None

        self._thread = threading.Thread(
            target=self._detect_loop, name="webrtc-detect", daemon=True
        )
        self._thread.start()

    # --- called from the WebRTC callback thread ---------------------------

    def submit(
        self, frame_bgr, media_time: float | None = None, frame_index: int | None = None
    ) -> None:
        """Hand over the newest frame. Returns immediately."""
        with self._lock:
            self._frames_seen += 1
            index = self._frames_seen if frame_index is None else frame_index
        self._pending.put(
            PendingFrame(
                frame=frame_bgr,
                wall_time=time.time(),
                media_time=media_time,
                frame_index=index,
            )
        )

    def annotated(self) -> Any:
        with self._lock:
            return self._annotated

    # --- read by the script thread / probe scheduler ----------------------

    @property
    def frames_seen(self) -> int:
        with self._lock:
            return self._frames_seen

    @property
    def detections(self) -> int:
        with self._lock:
            return self._detections

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def alerts(self) -> AlertLog:
        return self._alert_log

    def set_alert_config(self, config: AlertConfig) -> None:
        """Publish new alert thresholds. Immutable swap, read next detect cycle."""
        self._alert_config = config

    def set_matching(self, threshold: float, margin: float, recognize_every: int) -> None:
        """Retune identity live. No-op until the detector has been constructed."""
        if self._detector is not None:
            self._detector.set_matching(threshold, margin, recognize_every)

    def reload_gallery(self, gallery) -> None:
        """Swap enrolled faces mid-stream; every track is re-identified next cycle."""
        if self._detector is not None:
            self._detector.reload_gallery(gallery)

    def snapshot(self) -> tuple[FrameRecord, ...]:
        return self._buffer.snapshot()

    def sample(self, n: int) -> tuple[FrameRecord, ...]:
        return self._buffer.sample(n)

    def stop(self) -> None:
        self._stopping.set()
        self._pending.wake()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # --- detector thread --------------------------------------------------

    def _detect_loop(self) -> None:
        try:
            detector = self._detector_factory()
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        self._detector = detector
        # On this thread, never the script thread. CoreML compiles lazily, so
        # without this the first seconds of real video run several times slower.
        try:
            warmup(detector)
        except Exception:  # noqa: BLE001 - a failed warmup is not fatal
            _log.exception("detector warmup failed")

        while not self._stopping.is_set():
            pending = self._pending.take(DETECT_IDLE_WAIT_S)
            if pending is None:
                continue
            try:
                record = detector.detect(
                    pending.frame,
                    timestamp=time.monotonic(),
                    wall_time=pending.wall_time,
                    media_time=pending.media_time,
                    frame_index=pending.frame_index,
                )
            except Exception as exc:  # noqa: BLE001 - one bad frame isn't fatal
                _log.exception("detection failed")
                self._fail(f"{type(exc).__name__}: {exc}")
                continue

            # The annotated copy is for display only; keeping it in the buffer
            # would double memory and risk it reaching the VLM, which would then
            # describe the boxes instead of the scene.
            self._buffer.append(replace(record, annotated=None))
            self._raise_alerts(record, detector)

            with self._lock:
                self._detections += 1
                self._annotated = record.annotated

    def _raise_alerts(self, record: FrameRecord, detector: Detector) -> None:
        """Fire and remember alerts. Detector-thread only, so `_fired` needs no lock."""
        try:
            alerts = unknown_alerts(
                record,
                detector.state,
                self._fired,
                self.source_label,
                self._alert_config,
            )
        except Exception as exc:  # noqa: BLE001 - alerting must not kill detection
            _log.exception("alerting failed")
            self._fail(f"{type(exc).__name__}: {exc}")
            return

        if alerts:
            self._alert_log.extend(alerts)
            self._fired = record_fired(self._fired, alerts)

    def _fail(self, message: str) -> None:
        with self._lock:
            self._error = message


def detected_view(raw, annotated) -> Any:
    """The annotated frame alone, at the raw frame's size.

    Falls back to `raw` until the first detection lands, so the video starts
    immediately rather than showing nothing for the first ~50ms. `annotated`
    lags the live frame by one detection.
    """
    if annotated is None:
        return raw
    if annotated.shape != raw.shape:
        return cv2.resize(annotated, (raw.shape[1], raw.shape[0]))
    return annotated


def make_video_frame_callback(sink: FrameSink, is_live: bool = True):
    """Build the `video_frame_callback` streamlit-webrtc will call per frame.

    Must return fast: anything slow here throttles the video itself.

    `is_live` decides what an alert timestamp means. A camera has no meaningful
    offset, so it stays None and the alert reads as wall-clock; a file replays
    from its own timeline, so we pass the frame's presentation time and an
    alert can point at the second of the clip it came from.
    """
    import av  # imported lazily so the pure modules stay import-light

    def callback(frame: "av.VideoFrame") -> "av.VideoFrame":
        raw = frame.to_ndarray(format="bgr24")
        sink.submit(raw, media_time=None if is_live else media_time_of(frame))
        out = av.VideoFrame.from_ndarray(
            detected_view(raw, sink.annotated()), format="bgr24"
        )
        # Preserve timing or the browser cannot pace playback.
        out.pts = frame.pts
        out.time_base = frame.time_base
        return out

    return callback


def media_time_of(frame) -> float | None:
    """Seconds into the media, or None when the frame carries no timing."""
    if frame.pts is None or frame.time_base is None:
        return None
    return float(frame.pts * frame.time_base)
