"""Streamlit front end: live camera, identity overlay, VLM probes, alerts.

Threading contract: the detector and scheduler threads never touch Streamlit.
This module resolves the cached singletons on the script thread and passes them
in, then polls them from auto-rerunning fragments.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, replace
from pathlib import Path

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from vision.alerts import AlertConfig
from vision.detector import DEFAULT_RECOGNIZE_EVERY, Detector, DetectorConfig
from vision.enroll import (
    GOOD_PHOTOS,
    IMAGE_EXTS,
    MIN_PHOTOS,
    BuildReport,
    EnrolmentError,
    build_gallery,
    save_photos,
)
from vision.events import AlertEvent, ProbeEvent
from vision.eventlog import AlertLog, EventLog
from vision.graph import FALLBACK_CHAIN, build_graph
from vision.identity import DEFAULT_MARGIN, DEFAULT_THRESHOLD, Gallery, load_gallery
from vision.labelstudio import SeedReport, seed_from_zip
from vision.models import load_face_models
from vision.probe import (
    DEFAULT_PROMPT,
    MAX_INTERVAL_S,
    MAX_MEMORY_K,
    MAX_SAMPLE_FRAMES,
    MIN_INTERVAL_S,
    MIN_SAMPLE_FRAMES,
    ProbeConfig,
    ProbeScheduler,
    parse_fallback_chain,
)
from vision.scene_context import DEFAULT_SOURCE_LABEL, build_context
from vision.settings import settings
from vision.vlm_client import API_KEY_ENV, MODEL, has_api_key
from vision.webrtc import FrameSink, make_video_frame_callback

APP_TITLE = "Live Streaming Anomaly Detection"
STAGE_TO_FEED_RATIO = (2, 1)
HEADING_TO_ACTIONS_RATIO = (2, 1)  # feed title left, Actions menu right
ALERT_HEIGHT = 240
FEED_HEIGHT = 460  # scroll rail; keeps the video from being pushed off-screen
STATUS_REFRESH = "1s"
SCENE_REFRESH = "1s"
LOG_REFRESH = "2s"
ALERT_REFRESH = "1s"
MAX_RECOGNIZE_EVERY = 10
MAX_UNKNOWN_CHECKS = 60
MAX_COOLDOWN_S = 600.0

# Whether the WebRTC stream is playing. Written once per script run by
# `video_stream`, read by the status fragment on its own timer.
STREAMING_KEY = "_streaming"

TRAIN_UPLOAD_KEY = "_train_photos"
TRAIN_NAME_KEY = "_train_name"
TRAIN_EXPORT_KEY = "_train_export"

# Loopback-only app: a public STUN server is unnecessary and would just add a
# startup round trip, but keep the list explicit so it is easy to add one if
# this is ever served beyond localhost.
RTC_CONFIGURATION = {"iceServers": []}

# Streamlit's default top padding wastes most of a screen height before any
# content; the title is also h1-sized, which is too loud for a single-page tool.
PAGE_CSS = """
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 1.5rem; }
  header[data-testid="stHeader"] { height: 0; background: transparent; }
  .lac-title { font-size: 1.45rem; font-weight: 650; margin: 0 0 .15rem 0; }
  /* The sidebar reserves a tall strip for its collapse control. Reclaim it so
     the Train button sits in that space instead of below an empty gap. */
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1rem;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
  /* Streamlit pads every heading from above. The first heading in the feed
     column would sit well below the page title because of it, so drop the pad
     and let the two columns start on the same line. */
  [data-testid="stColumn"] [data-testid="stHeading"] { padding-top: 0; }
  [data-testid="stColumn"] [data-testid="stHeading"] h3 { padding-top: 0; }
</style>
"""


# --- cached singletons (script thread only) --------------------------------
# Bodies stay one-liners: the cache key hashes each function's own source, so
# editing a body here orphans the live worker/scheduler on the next rerun.


@st.cache_resource
def get_graph():
    return build_graph()  # no checkpointer: probes are stateless, memory is explicit


@st.cache_resource
def get_log() -> EventLog:
    return EventLog()


@st.cache_resource
def get_alert_log() -> AlertLog:
    return AlertLog()  # outlives the sink, so alerts survive a gallery change


@st.cache_resource(max_entries=1, on_release=lambda sink: _release_sink(sink))
def get_sink(gallery_path: str) -> FrameSink:
    return _start_sink(gallery_path)


@st.cache_resource
def get_enrolment_models():
    return load_face_models()  # own sessions: sharing the detector's would race it


@st.cache_resource
def get_scheduler(_sink: FrameSink, _config: ProbeConfig) -> ProbeScheduler:
    return _start_scheduler(_sink, _config)


# The sink the cache is currently holding, if any. Train needs to push a
# rebuilt gallery into a running stream, but must not *create* one — it is
# reachable on a first run where there is no gallery to build a Detector from.
_LIVE_SINK: FrameSink | None = None


def _running_sink() -> FrameSink | None:
    return _LIVE_SINK


def _release_sink(sink: FrameSink) -> None:
    global _LIVE_SINK
    if _LIVE_SINK is sink:
        _LIVE_SINK = None
    sink.stop()


def _start_sink(gallery_path: str) -> FrameSink:
    # max_entries=1 above: one Detector per stream, since track IDs are stateful.
    global _LIVE_SINK
    config = DetectorConfig(gallery_path=gallery_path)
    sink = FrameSink(
        DEFAULT_SOURCE_LABEL,
        detector_factory=lambda: Detector(config),
        alert_log=get_alert_log(),
    )
    atexit.register(sink.stop)
    _LIVE_SINK = sink
    return sink


def _start_scheduler(sink: FrameSink, config: ProbeConfig) -> ProbeScheduler:
    log = get_log()
    scheduler = ProbeScheduler(
        graph=get_graph(),
        source=sink,
        on_event=log.append,
        recent_answers=log.recent_answers,
        config=config,
    )
    scheduler.start()
    atexit.register(scheduler.stop)
    return scheduler


# --- sidebar ---------------------------------------------------------------


@dataclass(frozen=True)
class IdentitySettings:
    """Live-tunable identity and alerting knobs (no stream restart).

    The paths are not sidebar controls — they come from `vision.settings`
    (`GALLERY_PATH`, `FACES_DIR`) and are the same for the whole run, so
    enrolment and detection can never point at different files.
    """

    gallery_path: str
    faces_dir: str
    threshold: float
    margin: float
    recognize_every: int
    alerts: AlertConfig


def _identity_sidebar() -> IdentitySettings:
    st.sidebar.subheader("Identity")  # first section now: no leading divider
    threshold = st.sidebar.slider(
        "Match threshold", 0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.01,
        help="Minimum cosine similarity to claim a name. Higher = stricter.",
    )
    margin = st.sidebar.slider(
        "Margin over next person", 0.0, 0.5, value=DEFAULT_MARGIN, step=0.01,
        help="How far ahead the best match must be before it counts. Guards look-alikes.",
    )
    recognize_every = st.sidebar.slider(
        "Recognise every Nth frame", 1, MAX_RECOGNIZE_EVERY,
        value=DEFAULT_RECOGNIZE_EVERY,
        help="Boxes are drawn every frame regardless; this only paces the naming.",
    )

    st.sidebar.subheader("Alerts")
    checks = st.sidebar.slider(
        "Unknown for N checks", 1, MAX_UNKNOWN_CHECKS,
        value=AlertConfig().min_unknown_checks,
        help="How long a face must stay unidentified before it is worth interrupting for.",
    )
    cooldown = st.sidebar.number_input(
        "Re-alert after (seconds)", min_value=1.0, max_value=MAX_COOLDOWN_S,
        value=AlertConfig().cooldown_s, step=5.0,
    )
    return IdentitySettings(
        gallery_path=settings.gallery_path,
        faces_dir=settings.faces_dir,
        threshold=threshold,
        margin=margin,
        recognize_every=recognize_every,
        alerts=AlertConfig(min_unknown_checks=checks, cooldown_s=cooldown),
    )


def _require_gallery(path: str) -> Gallery:
    """Fail loudly. A missing gallery would silently mark everyone unknown.

    Stops the page rather than running blind. Train lives in the sidebar, which
    renders before this check, so it stays reachable — that is the only way out
    of a first run with no gallery at all.
    """
    try:
        gallery = load_gallery(path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Cannot load the face gallery: {exc}")
        st.info("Use **Train a face**, top of the sidebar, to enrol the first person.")
        st.stop()

    if gallery.is_empty:
        st.warning(
            f"The gallery at {path} has no enrolled faces — everyone will be "
            "reported as unknown. Use **Train a face** in the sidebar, or `make gallery`."
        )
    return gallery


def _require_api_key() -> None:
    """Warn once, up front, rather than letting every probe fail 15s apart.

    Does not stop the page: detection, identity and alerts are entirely local,
    so the app is still useful without a key — only the scene descriptions are
    lost. Reports whether the variable is set and never reads the key itself.
    """
    if has_api_key():
        return
    st.warning(
        f"{API_KEY_ENV} is not set, so scene descriptions will fail. "
        "Face recognition and alerts are unaffected.",
        icon="🔑",
    )
    st.caption(f"Set it in `.env` or the environment:  `{API_KEY_ENV}=sk-...`")


def _train_button(faces_dir: str, gallery_path: str, sink: FrameSink | None) -> None:
    """Open the enrolment dialog. Lives in the slot at the top of the sidebar."""
    if st.button(
        "🧑‍🏫 Train a face",
        width="stretch",
        help="Add reference photos for someone and rebuild the gallery",
    ):
        train_dialog(faces_dir, gallery_path, sink)


def _probe_sidebar() -> ProbeConfig:
    st.sidebar.divider()
    st.sidebar.subheader("Probe")
    prompt = st.sidebar.text_area("Prompt", value=DEFAULT_PROMPT, height=90)
    interval = st.sidebar.number_input(
        "Every (seconds)", min_value=MIN_INTERVAL_S, max_value=MAX_INTERVAL_S,
        value=15.0, step=1.0,
    )
    memory_k = st.sidebar.slider(
        "Remember last N answers", 0, MAX_MEMORY_K, value=3,
        help="Prior answers are prepended to the prompt so the model sees continuity.",
    )
    frames = st.sidebar.slider(
        "Frames per probe", MIN_SAMPLE_FRAMES, MAX_SAMPLE_FRAMES, value=3
    )
    st.sidebar.caption(
        "The scheduler follows the stream: it runs while the video is playing "
        "and pauses when you stop it."
    )

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    model = st.sidebar.text_input(
        "VLM model", value=MODEL,
        help="Any OpenAI model with image input. gpt-5-mini is the cheapest; "
             "gpt-5.4-mini is newer at roughly 3x the price.",
    )
    fallback = st.sidebar.text_input(
        "Fallback chain", value=",".join(FALLBACK_CHAIN),
        help="Comma-separated models tried in order when an answer comes back empty.",
    )
    # `paused` is not a sidebar control: the stream owns it (see `main`).
    return ProbeConfig.create(
        interval_s=interval, prompt=prompt, memory_k=memory_k, sample_frames=frames,
        paused=True, model=model, fallback_chain=parse_fallback_chain(fallback),
    )


# --- fragments -------------------------------------------------------------


def video_stream(sink: FrameSink) -> bool:
    """One WebRTC video element showing the detected view only.

    SENDRECV: the browser captures, we process, it plays the result back. The
    picture is a real <video> element rather than a repaint loop, and the camera
    device belongs to the browser — a stale Python process cannot hold it.

    Returns whether the stream is playing. That single flag drives the probe
    scheduler and the status line — probing a stream nobody started would burn
    a VLM call every interval on a stale buffer.
    """
    context = webrtc_streamer(
        key="stream-cam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=make_video_frame_callback(sink),
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={"style": {"width": "100%"}, "controls": False, "autoPlay": True},
    )

    playing = bool(context.state.playing)
    # Fragments rerun on their own timer, long after this function returned, so
    # they cannot read a local. Session state is how they see the live value.
    st.session_state[STREAMING_KEY] = playing
    return playing


@st.fragment(run_every=STATUS_REFRESH)
def status_bar(sink: FrameSink, scheduler: ProbeScheduler) -> None:
    # Read live, not from an argument: this reruns on its own timer, so a value
    # captured when the script last ran would be stale the moment you hit stop.
    playing = st.session_state.get(STREAMING_KEY, False)

    if scheduler.busy:
        probe = "🧠 probing…"
    elif scheduler.config.paused:
        probe = "⏸️ probes paused"
    else:
        probe = f"next probe in ~{scheduler.seconds_until_next():.0f}s"

    if playing:
        live = f"🟢 Streaming {sink.frames_seen} frames"
    elif sink.frames_seen:
        live = f"⏹️ Stopped after {sink.frames_seen} frames"
    else:
        live = "⏳ Press START to begin"

    st.caption(f"{live} · {sink.detections} detected · {probe}")
    if sink.error:
        st.error(sink.error)


@st.fragment(run_every=SCENE_REFRESH)
def scene_panel(sink: FrameSink) -> None:
    st.markdown(f"**Scene now** · {build_context(sink.snapshot(), sink.source_label)}")


@st.fragment(run_every=ALERT_REFRESH)
def alert_feed(log: AlertLog) -> None:
    """Unknown faces, newest first. Refreshed faster than the probe feed —
    these come straight off the detector and do not wait for the VLM."""
    alerts = log.newest_first()
    with st.container(height=ALERT_HEIGHT, border=False):
        if not alerts:
            st.success("No unknown faces seen.")
            return
        for alert in alerts:
            _alert_card(alert)


def _alert_card(alert: AlertEvent) -> None:
    with st.container(border=True):
        clock = alert.iso_time.split("T")[-1]
        st.markdown(f":red[⚠️ **Unknown person** · {clock} · {alert.media_clock}]")
        nearest = f" · nearest {alert.nearest}" if alert.nearest else ""
        st.caption(
            f"{alert.position} · track {alert.track_id} · "
            f"{alert.checks} checks{nearest} · {alert.source_label}"
        )


@st.fragment(run_every=LOG_REFRESH)
def event_feed(log: EventLog) -> None:
    """Newest-first card feed, scrolled inside a fixed-height rail."""
    events = log.newest_first()
    with st.container(height=FEED_HEIGHT, border=False):
        if not events:
            st.info("No probes yet.")
            return
        for event in events:
            _event_card(event)


def _event_card(event: ProbeEvent) -> None:
    with st.container(border=True):
        clock = event.iso_time.split("T")[-1]
        st.caption(f"🕑 {clock} · {event.model} · {event.frame_count} frames")
        if event.error:
            st.markdown(f":red[⚠️ {event.error}]")
        if event.answer:
            st.markdown(event.answer)
        with st.expander("scene", expanded=False):
            st.caption(event.scene_context)


# --- enrolment ---------------------------------------------------------------


@st.dialog("Train a face", width="large")
def train_dialog(faces_dir: str, gallery_path: str, sink: FrameSink | None) -> None:
    """Enrol one person from photos, or many at once from a Label Studio export.

    Both routes write into `faces/<person>/` and then rebuild the whole gallery
    — the file is written whole, and a partial write would silently drop
    everyone else.
    """
    one, bulk = st.tabs(["One person", "Bulk from Label Studio"])
    with one:
        _enrol_one_person(faces_dir, gallery_path, sink)
    with bulk:
        _enrol_from_export(faces_dir, gallery_path, sink)


def _enrol_one_person(
    faces_dir: str, gallery_path: str, sink: FrameSink | None
) -> None:
    st.caption(
        f"{GOOD_PHOTOS}-20 varied photos work best: different angles, lighting "
        "and expressions, one clear face per photo. Photos are saved under "
        f"`{faces_dir}/<name>/` and can be added to later."
    )

    name = st.text_input("Name", key=TRAIN_NAME_KEY, placeholder="Pratik Panchal")
    photos = st.file_uploader(
        "Photos",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=TRAIN_UPLOAD_KEY,
    )

    if photos:
        st.image([p.getvalue() for p in photos[:8]], width=90)
        if len(photos) < MIN_PHOTOS:
            st.warning(f"{len(photos)} photo(s) — {GOOD_PHOTOS} or more is much better.")

    if not st.button("Save and rebuild gallery", type="primary", width="stretch"):
        return

    try:
        saved = save_photos(faces_dir, name, _as_payloads(photos))
    except EnrolmentError as exc:
        st.error(str(exc))
        return

    st.success(f"Saved {len(saved)} photo(s) for {saved[0].parent.name}.")
    _rebuild_and_apply(faces_dir, gallery_path, sink)


def _enrol_from_export(
    faces_dir: str, gallery_path: str, sink: FrameSink | None
) -> None:
    """Seed every labelled person at once from a labels-only export archive."""
    images_dir = Path(settings.training_images_dir)
    st.caption(
        "Upload the Label Studio **YOLO** export (`classes.txt` + `labels/`). "
        "Images are not read from the zip — they are read from "
        f"`{images_dir}/`, so export without the media."
    )

    available = _count_images(images_dir)
    if available == 0:
        st.error(
            f"No images found in `{images_dir}/`. Put the source images there "
            "first, or point `TRAINING_IMAGES_DIR` somewhere else."
        )
    else:
        st.caption(f"{available} image(s) available in `{images_dir}/`.")

    archive = st.file_uploader("Export (.zip)", type=["zip"], key=TRAIN_EXPORT_KEY)

    if not st.button(
        "Seed and rebuild gallery",
        type="primary",
        width="stretch",
        disabled=archive is None or available == 0,
    ):
        return

    with st.spinner("Cropping labelled faces…"):
        try:
            seeded = seed_from_zip(archive.getvalue(), faces_dir, images_dir)
        except EnrolmentError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface, never crash the page
            st.error(f"Seeding failed: {type(exc).__name__}: {exc}")
            return

    _report_seeding(seeded)
    if seeded.total_crops == 0:
        return
    _rebuild_and_apply(faces_dir, gallery_path, sink)


def _report_seeding(seeded: SeedReport) -> None:
    """What the export produced, before anything is embedded."""
    st.success(
        f"{seeded.total_crops} crop(s) written for "
        f"{len(seeded.people) - len(seeded.empty_classes)} person(s)."
    )
    for person in sorted(seeded.people, key=lambda p: p.name):
        if person.is_empty:
            st.error(
                f"**{person.name}** — labelled in the export but produced no "
                "usable crop, so they stay unenrolled and will be alerted on."
            )
        elif person.crops < MIN_PHOTOS:
            st.warning(f"**{person.name}** — only {person.crops} crop(s).")

    notes = []
    if seeded.skipped_small:
        notes.append(f"{seeded.skipped_small} crop(s) too small")
    if seeded.skipped_malformed:
        notes.append(f"{seeded.skipped_malformed} malformed label line(s)")
    if seeded.unreadable_images:
        notes.append(f"{len(seeded.unreadable_images)} unreadable image(s)")
    if notes:
        st.caption("Skipped: " + ", ".join(notes) + ".")

    if seeded.missing_images:
        # Name them: "5 labels had no image" is not actionable, but seeing the
        # filenames next to what is in the folder makes the mismatch obvious.
        with st.expander(
            f"⚠️ {len(seeded.missing_images)} label(s) had no matching image",
            expanded=True,
        ):
            st.caption(
                "These annotations were labelled but their image is not in "
                f"`{settings.training_images_dir}/`. Match is by filename, "
                "ignoring Label Studio's upload prefix and letter case."
            )
            for name in seeded.missing_images[:15]:
                st.caption(f"· {name}")
            if len(seeded.missing_images) > 15:
                st.caption(f"· …and {len(seeded.missing_images) - 15} more")


def _count_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _rebuild_and_apply(
    faces_dir: str, gallery_path: str, sink: FrameSink | None
) -> None:
    """Re-embed everyone and push the result to a running stream."""
    with st.spinner("Re-embedding every enrolled face…"):
        try:
            detector, recognizer = get_enrolment_models()
            report = build_gallery(faces_dir, gallery_path, detector, recognizer)
        except EnrolmentError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface, never crash the page
            st.error(f"Rebuild failed: {type(exc).__name__}: {exc}")
            return

    _apply_new_gallery(report, gallery_path, sink)


def _apply_new_gallery(
    report: BuildReport, gallery_path: str, sink: FrameSink | None
) -> None:
    """Show what the rebuild did and push it to a running stream."""
    st.success(
        f"Gallery rebuilt: {len(report.enrolled)} "
        f"{'person' if len(report.enrolled) == 1 else 'people'}, "
        f"{report.total_embeddings} embeddings."
    )

    for person in report.people:
        if not person.is_enrolled:
            st.error(
                f"**{person.name}** — no usable face in any photo, so they are "
                "NOT enrolled and will be alerted on as a stranger."
            )
        elif person.is_thin:
            st.warning(f"**{person.name}** — only {person.accepted} photo(s). Add more.")
        if person.is_incoherent:
            st.warning(
                f"**{person.name}** — photos disagree with each other "
                f"(cohesion {person.cohesion:.2f}). Is that really all one person?"
            )
        if person.skipped:
            with st.expander(f"{person.name}: {len(person.skipped)} photo(s) skipped"):
                for reason in person.skipped:
                    st.caption(reason)

    for a, b, score in report.confusable:
        st.warning(
            f"**{a}** and **{b}** look alike to the matcher ({score:.2f}). "
            "Raise the margin, or add photos that tell them apart."
        )

    if sink is not None:
        # Live swap: every track is re-identified next cycle, so a person you
        # just enrolled stops being an unknown without restarting the stream.
        sink.reload_gallery(load_gallery(gallery_path))
        st.caption("Applied to the running stream.")

    st.button("Done", width="stretch", on_click=_clear_train_form)


def _clear_train_form() -> None:
    for key in (TRAIN_NAME_KEY, TRAIN_UPLOAD_KEY, TRAIN_EXPORT_KEY):
        st.session_state.pop(key, None)


def _as_payloads(photos) -> tuple[tuple[str, bytes], ...]:
    """Streamlit uploads -> the (filename, bytes) pairs `enroll` expects."""
    return tuple((p.name, p.getvalue()) for p in (photos or ()))


# --- page ------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.html(PAGE_CSS)

    # Default to "not streaming" so a path that `st.stop()`s before the video
    # renders (a missing gallery) cannot leave a stale True behind.
    st.session_state[STREAMING_KEY] = False

    # Reserve the top of the sidebar before anything fills it: Train belongs
    # there, but its arguments only exist once the identity section has run.
    train_slot = st.sidebar.container()

    identity = _identity_sidebar()
    config = _probe_sidebar()

    # Rendered before the gallery check below, which stops the page when
    # nothing is enrolled yet — Train is the only way out of that state.
    with train_slot:
        _train_button(identity.faces_dir, identity.gallery_path, _running_sink())

    gallery = _require_gallery(identity.gallery_path)
    st.sidebar.caption(f"Enrolled: {', '.join(gallery.people) or 'nobody'}")

    _require_api_key()
    sink = get_sink(identity.gallery_path)
    sink.set_matching(identity.threshold, identity.margin, identity.recognize_every)
    sink.set_alert_config(identity.alerts)

    scheduler = get_scheduler(sink, config)
    scheduler.set_source(sink)  # follows a gallery-driven sink rebuild

    st.sidebar.divider()
    if st.sidebar.button("Probe now", width="stretch", disabled=scheduler.busy):
        scheduler.trigger_now()

    # Video left, feeds pinned right. The columns open *before* the title so
    # both sides start at the same y — otherwise the title and status line push
    # the Alerts heading down and the two columns read as misaligned.
    stage, feed = st.columns(STAGE_TO_FEED_RATIO, gap="large")
    with stage:
        st.html(f'<div class="lac-title">{APP_TITLE}</div>')
        status_bar(sink, scheduler)
        playing = video_stream(sink)
        scene_panel(sink)

    # After the video, because only rendering it reveals whether it is playing.
    # The scheduler publishes only on a real change, so this is a no-op most runs.
    scheduler.update(replace(config, paused=not playing))

    with feed:
        _feed_heading("Alerts", get_alert_log(), "alerts")
        alert_feed(get_alert_log())

        _feed_heading("Event log", get_log(), "probe_events")
        event_feed(get_log())


def _feed_heading(title: str, log, stem: str) -> None:
    """A feed's title with its Actions menu tucked to the right of it."""
    heading, actions = st.columns(HEADING_TO_ACTIONS_RATIO, vertical_alignment="bottom")
    with heading:
        st.subheader(title)
    with actions:
        _log_actions(log, stem)


def _log_actions(log, stem: str) -> None:
    """Export and clear, collapsed into one menu.

    Rendered from the script body, never inside an auto-rerunning fragment: a
    widget rebuilt on a 1s timer loses clicks. `data=` takes the bound method
    rather than its result, so the export is only serialised when the popover
    is actually opened.
    """
    with st.popover("Actions", icon=":material/more_horiz:", width="stretch"):
        st.download_button(
            "Download JSON", data=log.to_json, file_name=f"{stem}.json",
            key=f"{stem}-json", mime="application/json", on_click="ignore",
            icon=":material/download:", width="stretch",
        )
        st.download_button(
            "Download CSV", data=log.to_csv, file_name=f"{stem}.csv",
            key=f"{stem}-csv", mime="text/csv", on_click="ignore",
            icon=":material/download:", width="stretch",
        )
        if st.button(
            "Clear", key=f"{stem}-clear", icon=":material/delete:", width="stretch"
        ):
            log.clear()
            st.rerun()


if __name__ == "__main__":  # `streamlit run` executes this file as __main__
    main()
