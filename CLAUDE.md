# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Streamlit app that watches a live camera, recognises **who** is in frame, alerts on strangers, and narrates "what is happening". InsightFace `buffalo_l` detects and identifies faces per frame against an enrolled gallery; an unidentified face that persists raises a timestamped alert immediately, and on a user-set interval an OpenAI vision model (VLM) reasons over a burst of recent frames plus a text summary of who is present. Both alerts and answers land in downloadable logs.

YOLO was removed entirely (see "Why no YOLO" below) — there is no object detection, no body detection, faces only.

## Commands

```bash
uv sync                       # install deps
make run                      # == uv run streamlit run app.py --server.runOnSave=false
make seed                     # Label Studio YOLO export -> faces/<person>/ crops
make gallery                  # enrol faces: tools/build_gallery.py --faces-dir faces

# VLM prereq: scene descriptions call OpenAI. Detection/alerting need no key.
export OPENAI_API_KEY=sk-...   # or: cp .env.example .env && edit
```

Face models (~340MB) self-download to `~/.insightface/models/buffalo_l/` on first run.

## Architecture

**The browser owns the camera.** `streamlit-webrtc` negotiates a peer connection;
frames arrive on an aiortc worker thread and the frame we return is played back
in a real `<video>` element. Python never opens the camera, so a stale process
can no longer hold the device and leave the next run with a black picture — that
failure mode cost most of a debugging session.

```
aiortc callback thread        detector thread         script thread      scheduler thread
──────────────────────        ───────────────         ─────────────      ────────────────
video_frame_callback                                  fragments poll ↓   waits on Event
  ├─> sink.submit() ──> _LatestFrame ──> Detector.detect()
  │                                        ├─> annotated ──┐
  │                                        └─> FrameRecord ─┼> FrameBuffer ──> sample(n)
  └─> detected_view(raw, annotated) ──> back to browser <───┘  sidebar ─> Config
                                                    event feed <──────  graph.invoke
```

**The callback must return immediately.** A full detect+identify costs ~50ms
(CoreML, 640x480, 1-2 faces) and ~68ms with six; running it inline would
throttle the video to that. It goes into `_LatestFrame`, a one-slot mailbox — a busy detector drops
skipped frames rather than building a backlog — and the callback sends back
whatever annotation is ready, so the picture stays at camera rate while the
boxes refresh at detector rate.

**Only the detected view is streamed back.** `detected_view()` returns the
annotated frame alone (falling back to `raw` until the first detection lands).
An earlier `stacked()` shipped Detected above Original in one frame; the raw
half was dropped because the browser already has that picture locally, so
returning it doubled the frame height for nothing.

- **`vision/records.py`** — **pure**, model-free: frozen `Face`, `FrameRecord`, `Pose`. Lives apart from `detector` so every pure module imports zero insightface/onnxruntime (same reasoning as `events` staying LangChain-free). `FrameRecord.face_count` is a **property**, not a field, so it cannot desync from `faces`.
- **`vision/identity.py`** — **pure**: `Gallery` (names + L2-normed rows, names may **repeat**), `load_gallery`, `match`. Two gates: similarity ≥ threshold **and** a margin over the best *other person*. The runner-up deliberately excludes other photos of the winner — gating alice against alice would reject every well-enrolled face. A refusal still reports `best_name`/`best_similarity`, which is the number you tune the threshold with.
- **`vision/facetrack.py`** — **pure**: greedy IoU association, `Track`/`TrackState`, `with_identity`, `due_for_recognition`. The POC had **no tracker**; this is what restores "newly appeared"/"moved right" and what caches identity + pose between recognition cycles. `unknown_streak` counts **checks, not frames** — recognition runs on a cadence, so a frame count would change meaning whenever `recognize_every` moved.
- **`vision/facemeta.py`** — **pure**: `position_label`, `blur_score`, `describe_pose`, `build_face`, `crop`. Ported from the POC's `extract.py`.
- **`vision/alerts.py`** — **pure**: `AlertConfig` + `unknown_alerts` + `record_fired`. Debounce (a single blurry/profile frame flips a known face to unknown) and per-track cooldown (else a stranger standing still alerts every cycle). The caller owns the clock and the fired-map, so the whole policy is testable without threads.
- **`vision/detector.py`** — the only module that runs models *per frame*. Loads `det_10g` + `w600k_r50` (+ `1k3d68` for pose) individually via `vision/models.py`, **not** `FaceAnalysis` — that wrapper runs every sub-model on every face (measured 801ms for 6 faces). Detection runs every frame (boxes stay fresh); recognition + pose run on `recognize_every`, except a **brand-new track is always recognised immediately** or the stranger alert would be late. Owns mutable `TrackState`, so **one Detector per stream**.
- **`vision/overlay.py`** — draws the boxes (green = named, red = `UNKNOWN`). Replaces ultralytics' `Results.plot()`.
- **`vision/buffer.py`** — `FrameBuffer`, a `deque(maxlen=45)`. `sample(n)` returns up to n evenly-spaced records, always including the newest.
- **`vision/scene_context.py`** — **pure**: buffer → grounding text (who is identified with confidence/pose/position, how many are not, appeared, left/right motion). `source_label` must say what the frames really are; it goes verbatim into the prompt.
- **`vision/memory.py`** — **pure**: last K probe answers → prompt preamble. Folded into the **question** rather than the context; both reach the model now, but keeping memory in the question means it survives any future prompt-shape change to `context`.
- **`vision/probe.py`** — frozen `ProbeConfig` (interval, prompt, memory K, frames, paused, model, fallback chain), `run_probe` (never raises — failures come back as `ProbeEvent.error`), and `ProbeScheduler` (daemon thread waiting on an Event with a 0.25s tick, so interval edits and manual triggers land immediately; next deadline measured from probe *completion*). `update()` restarts the deadline on an interval change **and on resume** — `paused` is driven by the stream, so without that a scheduler paused longer than one interval fires the instant you press start, against a buffer that has not refilled.
- **`vision/events.py` / `vision/eventlog.py`** — frozen `ProbeEvent` **and** `AlertEvent`, each with its own field tuple; `RecordLog` is generic over anything with `.as_dict()`, subclassed as `EventLog` / `AlertLog`. Lock-guarded capped deques with JSON/CSV export (`csv.writer` quoting matters: answers contain commas and newlines). `AlertEvent.media_clock` renders `media_time` as `0:01:23.4`, or `"live"` when there is no media offset.
- **`vision/settings.py`** — **the only module that reads the environment**. A pydantic-settings `Settings`: field name maps to its upper-case env var, `.env` is picked up automatically (and gitignored), blank overrides are rejected rather than silently accepted. `openai_api_key` is a `SecretStr`, so a repr/log of the settings object prints `**********`; `require_api_key()` is the single place it becomes plain text. Read attributes off the module-level `settings`; do not reach for `os.environ` anywhere else.
- **`vision/vlm_client.py`** — what we send: `MODEL`, `INSTRUCTIONS`, frame downscale+JPEG encode, and `api_key()`/`has_api_key()`, both thin wrappers over `settings`. **No network calls and no key held in a module constant** — `api_key()` pulls it out of the `SecretStr` per call and raises a `VLMError` naming the variable, so a missing key becomes a visible probe event instead of an import crash. The key is never logged, echoed to the page, or stored in an event.
- **`vision/llm.py`** — LangChain `ChatOpenAI` factory (`make_chat`) and multimodal `build_human_message` (base64 data-URL image parts with an explicit `detail`). Sets **no `temperature`** — the GPT-5 family rejects anything but the default.
- **`vision/graph.py`** — LangGraph `StateGraph`: `vlm → (check) → END | fallback → vlm`. `build_graph(checkpointer=None)` — the app passes **no** checkpointer because probes are one-shot (`InMemorySaver` never evicts, so a fresh `thread_id` per probe would leak). `state["fallback_chain"]` overrides the env-driven `FALLBACK_CHAIN` per call.

- **`vision/webrtc.py`** — `FrameSink` (satisfies `probe.SceneSource`), `PendingFrame`, the `video_frame_callback` factory, and `detected_view()`. The camera is `SENDRECV` and is the only source the app offers. `PendingFrame.media_time` / `AlertEvent.media_clock` still carry an offset-into-the-media for a `RECVONLY` file player, — the plumbing was kept when the file-source picker was removed, so re-adding replay is a UI change rather than a re-architecture. Nothing sets `make_video_frame_callback(is_live=False)` today, so every alert stamps wall-clock. **Alerts are raised here**, on the detector thread, so an intruder never waits on a VLM round trip; `_fired` is detector-thread-only and needs no lock. `_detect_loop` calls `detector.warmup()` right after constructing it — on that thread, never the script thread — and those frames reach neither the buffer nor alerting. Nothing here may touch Streamlit — it all runs on aiortc/worker threads.
- **`app.py`** — Streamlit UI only. Singletons via `@st.cache_resource`; the sink is keyed by `gallery_path` (fixed per run, from `GALLERY_PATH`) with `max_entries=1` + `on_release=_release_sink`, so changing the gallery evicts and tears down the old one (track IDs are per-stream state). The `AlertLog` is cached **separately** from the sink, so alerts survive that rebuild. Threshold/margin/cadence and the alert config apply live via `sink.set_matching` / `set_alert_config` — no stream restart for a slider. Only the status bar, scene line, alert feed and event feed use `@st.fragment(run_every=…)`; the video is a WebRTC component and does not rerun. **The stream owns `ProbeConfig.paused`**: `video_stream` returns `ctx.state.playing`, `main` applies it via `replace(config, paused=not playing)` *after* the columns render (nothing before them knows it), and it is mirrored into `st.session_state[STREAMING_KEY]` because the status fragment reruns on its own timer and cannot read a local. `main` sets that key False first, so a path that `st.stop()`s before the video renders cannot leave a stale True. The **Train** dialog (`train_dialog`) is rendered by `_title_row` **before** the gallery check, because that check `st.stop()`s when nothing is enrolled and Train is the only way out of it. It gets its own ONNX sessions via `get_enrolment_models` — reusing the detector's would race the detector thread — and pushes the rebuilt gallery into a live stream through `_running_sink()`, a module-level handle on the cached sink that must **not** create one (there may be no gallery to build a `Detector` from yet).
- **`vision/enroll.py`** — enrolment, shared by the app's **Train** dialog and the CLI so both write byte-identical galleries. `safe_person_name` **rejects** rather than mangles (a person silently enrolled under the wrong folder is worse than an error) and strips path separators, so a crafted name cannot write outside `faces/`; `_write_unique` never overwrites. `build_gallery` rebuilds **every** person, because the file is written whole — a partial write would drop everyone else. Fixes two POC defects: a folder with no usable photo is **loudly** reported (silently dropping them means they get flagged as an intruder), and every accepted photo keeps its own row instead of being averaged into one centroid — the averaging is what made profile faces fall to unknown.
- **`vision/models.py`** — loads the `buffalo_l` ONNX models. Separate from `detector` because enrolment needs the same two models without the tracker, gallery, or per-stream state.
- **`vision/labelstudio.py`** — Label Studio YOLO export -> `faces/<person>/` crops (`make seed`), ported from the POC's `seed.py`. Class names go through `safe_person_name` (an export is external input; a class named `../x` would escape the faces dir) and every skip is counted, not silent. **`index_images` scans the directory once and matches on a lower-cased stem** rather than probing `stem + ext`: the real export is 96% `.JPG` against a lower-case extension list, so name-probing only works on a case-insensitive filesystem and would silently lose ~all images on Linux. `find_image` also strips Label Studio's `<hash>__` prefix and url-decodes `%20`. `seed_from_zip` backs the Train dialog's bulk tab: the archive is **untrusted**, so only `classes.txt` and `labels/*.txt` are extracted (images are deliberately ignored — they come from `TRAINING_IMAGES_DIR`), every entry is checked for zip-slip via `is_relative_to`, and member count and total size are capped. `images_dir` defaults to `export/images` but is overridable, which is what makes a labels-only export work at all.
- **`tools/seed_faces.py` / `tools/build_gallery.py`** — thin CLIs over `vision/labelstudio.py` and `vision/enroll.py`.

## Conventions that matter here

- **Immutability:** `FrameRecord`, `Face`, `Pose`, `Track`, `TrackState`, `Gallery`, `Match`, `DetectorConfig`, `AlertConfig`, `ProbeConfig`, `ProbeEvent`, `AlertEvent` are `@dataclass(frozen=True)`. Config changes swap the whole object; the scheduler publishes only when `!=` the current one. `facetrack.update` / `with_identity` return new state and never touch the state handed in.
- **Threading rule:** worker and scheduler threads must never call `st.*`, touch `st.session_state`, or invoke a `@st.cache_resource` function — resolve singletons on the script thread and pass them in. Never call `st.rerun()` from a thread; fragments poll instead.
- **No `while True` + `st.empty()` render loop:** it pins the single ScriptRunner thread and every sidebar widget stops responding. Fragments only.
- **Don't put video through Streamlit.** Repainting `st.image` on a fragment timer is choppy and costs a rerun per frame; a hand-rolled MJPEG side-server fixed the smoothness but leaked connections until the panels went permanently blank (browsers cap ~6 per host). WebRTC is what actually holds up.
- **Cached factory bodies stay one-liners:** the cache key hashes each function's own source, so editing a body orphans the live worker holding the camera. (Same reason `--server.runOnSave=false`.)
- **Grounding, not guessing:** the VLM prompt fuses the real frames **and** the scene text (fusion B). Preserve both inputs when changing the prompt. `vlm_client.INSTRUCTIONS` tells the model to **use the names** but never reveal where they came from, and to render UNKNOWN as "someone I don't recognise".
- **One string, one injection point:** all recognition reaches the model as `build_context(...)` → `probe.py` `state["context"]` → `llm.build_human_message`. Nothing else about detection is visible to the VLM. Buffered records carry `annotated=None` deliberately (`webrtc._detect_loop`) — if boxes reached the model it would narrate the boxes.
- **Reasoning tokens are billed as output and count against `max_tokens`.** `MAX_OUTPUT_TOKENS = 2000` is deliberately generous: cap it near the visible answer length and the reasoning budget eats the whole allowance and the model returns `""`. Brevity is enforced by `INSTRUCTIONS` + `answer.tidy`, never by the token ceiling.
- `FrameBuffer` needs no lock: `tuple(deque)` runs in C without releasing the GIL. That guarantee disappears under free-threaded CPython.
- `Chat_Interface.py` / `Chat_Interface_No_Feed_Play.py` are an unrelated legacy OpenAI prototype, kept deliberately; they do not import `vision/` and their deps are not in `pyproject.toml`. The `detection.ipynb` notebook is the original prototype — the model-load lines and the `result.plot()` face-layering originate there.

## Known environment notes

- Default VLM is `gpt-5-mini` — cheapest OpenAI model with image input ($0.25/$2.00 per 1M, ~$0.20/hour at 15s probes). `gpt-5.4-mini` is newer at ~3x. `gpt-5-nano` is cheaper still but has **no vision** — unusable here.
- Camera permission is now the **browser's** to ask for, not the terminal's.
- PyAV and opencv-python both ship `libavdevice`, so macOS logs a duplicate-ObjC-class warning at import. Harmless so far, but it is the first thing to suspect on a mysterious native crash.
- **CoreML needs several inferences to reach steady state** (it compiles partitions lazily). Timing it after a single warm-up frame makes it look 3-4x *slower* than CPU — the reverse of the truth. `detector.warmup()` runs `WARMUP_FRAMES = 3` for this reason; use it before any benchmark, and interleave provider comparisons or thermal drift will decide the result for you.
- **insightface never picks CoreML by itself.** `model_zoo.get_default_providers()` is hardcoded to CUDA/CPU, so `providers=` must be passed explicitly — **and** `prepare(ctx_id=0)`, because `ctx_id < 0` calls `set_providers(['CPUExecutionProvider'])` and silently undoes it. `CTX_ID = 0` in `detector.py` exists for exactly this.
- **In insightface 1.0.1 `det_10g.onnx` routes to `SCRFD`, not `RetinaFace`** (0.7.3 routed the other way). Same `detect(img, input_size, max_num, metric) -> (det (N,5), kpss (N,5,2))` signature either way.
- **`w600k_r50.onnx` declares batch=1**, so batched recognition logs `Expected shape {1,512} does not match actual shape {N,512}`. Verified harmless: batched vs one-at-a-time embeddings agree to cosine 1.0 (max abs diff 1.5e-5). Do **not** "fix" it with a loop — that is a `session.run` per face.
- `onnxruntime < 1.19` pins `numpy<2` and cannot coexist with this project's numpy 2.x / opencv 5. The floor is `onnxruntime>=1.20`.
- Face weights (~340MB) live in `~/.insightface/models/buffalo_l/`, **outside** the repo, and self-download. `faces/` and `gallery.npz` are gitignored — they are biometric data.

## Why no YOLO

`yolo11l.pt` (objects + track IDs) and `model/yolov11m-face.pt` (face boxes) were both removed. YOLO can say *a face is here* but not *whose*, and naming people via one YOLO class per person needs retraining per person and has no calibrated "unknown". ArcFace embeddings give all of it: enrol from a handful of photos, no retraining, a real refusal threshold.

**What was given up, deliberately:** object classes (bag, vehicle, weapon) and body detection. A back-turned, masked or distant intruder produces no face, no track, and therefore **no alert**. If that matters, the way back is a small person-only detector re-added purely as a presence signal — not the old two-model stack.
