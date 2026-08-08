"""Scheduled VLM probes: config, one-shot execution, and the scheduler thread.

A "probe" is one automatic question put to the VLM about what the camera (or
uploaded video) is currently showing. The scheduler fires probes on an interval
the user controls, and hands each result to a callback (the event log).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable, Iterable, Protocol

from vision.events import ProbeEvent
from vision.memory import build_probe_question
from vision.records import FrameRecord
from vision.scene_context import build_context
from vision.vlm_client import MODEL, VLMError

_log = logging.getLogger(__name__)

DEFAULT_PROMPT = "Describe what is happening in the scene right now."
MIN_INTERVAL_S = 2.0
MAX_INTERVAL_S = 3600.0
MIN_SAMPLE_FRAMES = 1
MAX_SAMPLE_FRAMES = 8
MAX_MEMORY_K = 20

# How often the scheduler wakes to re-check its deadline. Small enough that an
# interval edit or a manual trigger lands promptly, large enough to stay idle.
TICK_S = 0.25


class SceneSource(Protocol):
    """What a probe needs from the frame source (see `vision.webrtc`)."""

    source_label: str

    def sample(self, n: int) -> tuple[FrameRecord, ...]: ...

    def snapshot(self) -> tuple[FrameRecord, ...]: ...


@dataclass(frozen=True)
class ProbeConfig:
    """User-controlled probe settings. Immutable: swap the whole object."""

    interval_s: float = 15.0
    prompt: str = DEFAULT_PROMPT
    memory_k: int = 3
    sample_frames: int = 3
    paused: bool = False
    model: str = MODEL
    fallback_chain: tuple[str, ...] = ()  # empty -> graph's env-configured default

    def __post_init__(self) -> None:
        if not MIN_INTERVAL_S <= self.interval_s <= MAX_INTERVAL_S:
            raise ValueError(
                f"interval_s must be in [{MIN_INTERVAL_S}, {MAX_INTERVAL_S}], "
                f"got {self.interval_s}"
            )
        if not MIN_SAMPLE_FRAMES <= self.sample_frames <= MAX_SAMPLE_FRAMES:
            raise ValueError(
                f"sample_frames must be in "
                f"[{MIN_SAMPLE_FRAMES}, {MAX_SAMPLE_FRAMES}], got {self.sample_frames}"
            )
        if not 0 <= self.memory_k <= MAX_MEMORY_K:
            raise ValueError(f"memory_k must be in [0, {MAX_MEMORY_K}], got {self.memory_k}")
        if not self.prompt.strip():
            raise ValueError("prompt must not be blank")
        if not self.model.strip():
            raise ValueError("model must not be blank")

    @classmethod
    def create(cls, **kwargs) -> ProbeConfig:
        """Build a config, clamping numeric fields into range instead of raising."""
        clamped = dict(kwargs)
        if "interval_s" in clamped:
            clamped["interval_s"] = _clamp(
                float(clamped["interval_s"]), MIN_INTERVAL_S, MAX_INTERVAL_S
            )
        if "sample_frames" in clamped:
            clamped["sample_frames"] = int(
                _clamp(int(clamped["sample_frames"]), MIN_SAMPLE_FRAMES, MAX_SAMPLE_FRAMES)
            )
        if "memory_k" in clamped:
            clamped["memory_k"] = int(_clamp(int(clamped["memory_k"]), 0, MAX_MEMORY_K))
        prompt = clamped.get("prompt")
        if prompt is not None and not str(prompt).strip():
            clamped["prompt"] = DEFAULT_PROMPT
        model = clamped.get("model")
        if model is not None and not str(model).strip():
            clamped["model"] = MODEL
        if "fallback_chain" in clamped:
            clamped["fallback_chain"] = parse_fallback_chain(clamped["fallback_chain"])
        return cls(**clamped)

    def with_paused(self, paused: bool) -> ProbeConfig:
        return replace(self, paused=paused)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_fallback_chain(value: str | Iterable[str]) -> tuple[str, ...]:
    """Comma-separated text (or any iterable) to a clean tuple of model names."""
    names = value.split(",") if isinstance(value, str) else value
    return tuple(name.strip() for name in names if name and name.strip())


def run_probe(
    graph,
    source: SceneSource,
    config: ProbeConfig,
    recent_answers: tuple[str, ...],
) -> ProbeEvent:
    """Sample the window, ask the VLM once, and return an immutable event.

    Never raises: VLM failures come back as an event with `error` set, so the
    scheduler thread can't die on a transient API problem.
    """
    model = config.model
    sampled = source.sample(config.sample_frames)
    context = build_context(source.snapshot(), source_label=source.source_label)
    question = build_probe_question(config.prompt, recent_answers, config.memory_k)

    if not sampled:
        return _event(context, question, "", model, 0, "No frames captured yet.")

    state = {
        "messages": [],
        # Clean frames, never the annotated ones: boxes and labels drawn on the
        # image make the model narrate the detector instead of the scene.
        "frames": [record.frame for record in sampled],
        "context": context,
        "question": question,
        "model": model,
        "answer": "",
        "attempts": 0,
        "fallback_chain": list(config.fallback_chain),
    }
    try:
        final = graph.invoke(state)
    except VLMError as exc:
        return _event(context, question, "", model, len(sampled), str(exc))
    except Exception as exc:  # noqa: BLE001 - scheduler thread must survive
        return _event(
            context, question, "", model, len(sampled), f"{type(exc).__name__}: {exc}"
        )

    answer = (final.get("answer") or "").strip()
    error = "" if answer else "All models returned an empty answer."
    return _event(
        context, question, answer, final.get("model", model), len(sampled), error
    )


def _event(
    context: str, question: str, answer: str, model: str, frames: int, error: str
) -> ProbeEvent:
    now = time.time()
    return ProbeEvent(
        timestamp=now,
        iso_time=datetime.fromtimestamp(now).isoformat(timespec="seconds"),
        scene_context=context,
        question=question,
        answer=answer,
        model=model,
        frame_count=frames,
        error=error,
    )


class ProbeScheduler:
    """Daemon thread firing `run_probe` on an interval, plus manual triggers.

    Waits on an Event rather than sleeping the full interval, so config edits,
    manual triggers and shutdown all take effect within one TICK_S.
    """

    def __init__(
        self,
        graph,
        source: SceneSource,
        on_event: Callable[[ProbeEvent], None],
        recent_answers: Callable[[], tuple[str, ...]],
        config: ProbeConfig | None = None,
    ) -> None:
        self._graph = graph
        self._source = source
        self._on_event = on_event
        self._recent_answers = recent_answers
        self._lock = threading.Lock()
        self._config = config or ProbeConfig()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._trigger = threading.Event()
        self._busy = False
        self._next_due = time.monotonic() + self._config.interval_s
        self._thread = threading.Thread(
            target=self._run, name="probe-scheduler", daemon=True
        )

    # --- script-thread API -------------------------------------------------

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def config(self) -> ProbeConfig:
        with self._lock:
            return self._config

    @property
    def source(self) -> SceneSource:
        with self._lock:
            return self._source

    @property
    def busy(self) -> bool:
        return self._busy

    def seconds_until_next(self) -> float:
        return max(0.0, self._next_due - time.monotonic())

    def update(self, config: ProbeConfig) -> None:
        """Publish new settings. No-op when nothing changed."""
        with self._lock:
            if config == self._config:
                return
            interval_changed = config.interval_s != self._config.interval_s
            # Resuming restarts the clock. Without this, a scheduler that sat
            # paused longer than one interval has a deadline in the past and
            # fires the instant it resumes — against a buffer that has not
            # refilled yet, which is an error event every time you press start.
            resumed = self._config.paused and not config.paused
            self._config = config
        if interval_changed or resumed:
            self._next_due = time.monotonic() + config.interval_s
        self._wake.set()

    def set_source(self, source: SceneSource) -> None:
        """Point the scheduler at a new frame sink (camera/file switch)."""
        with self._lock:
            self._source = source

    def trigger_now(self) -> None:
        self._trigger.set()
        self._wake.set()

    # --- scheduler thread --------------------------------------------------

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._wake.wait(timeout=TICK_S)
            self._wake.clear()
            if self._stopping.is_set():
                return

            config = self.config
            manual = self._trigger.is_set()
            if not manual and (config.paused or time.monotonic() < self._next_due):
                continue
            self._trigger.clear()
            self._probe_once(config)

    def _probe_once(self, config: ProbeConfig) -> None:
        self._busy = True
        try:
            event = run_probe(self._graph, self.source, config, self._recent_answers())
            self._on_event(event)
        except Exception:  # noqa: BLE001 - a bad sink must not kill the thread
            _log.exception("probe sink failed; scheduler continuing")
        finally:
            self._busy = False
            # Measure the next deadline from completion, not from start, so a
            # slow model can't queue back-to-back probes with zero idle time.
            self._next_due = time.monotonic() + self.config.interval_s
