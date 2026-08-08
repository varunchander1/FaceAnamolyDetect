# Live Streaming Anomaly Detection

A Streamlit app that watches a live camera, recognises who is in frame, raises
a timestamped alert when it sees someone it does not know, and asks an OpenAI
vision model — on a schedule you control — what is happening.

## How it works

```
browser camera ──WebRTC──> detect + identify ──> annotated ──> browser
     │                                    │
     │                                    └─> unknown for N checks ──> ALERT
     rolling buffer (recent frames + identified faces)
                     │  every N seconds, or "Probe now"
       scene-context builder (text summary of the window)
                     │  + last K probe answers (sliding-window memory)
   LangGraph:  vlm ──(empty?)──> fallback ──> vlm
       │
       └─> ChatOpenAI (frames + scene notes + prompt) ──> event log
```

- **InsightFace `buffalo_l`** detects faces (`det_10g`), embeds them with
  ArcFace (`w600k_r50`) and measures head pose (`1k3d68`). A face is named by
  cosine similarity against an enrolled gallery, with a runner-up margin gate
  so "unknown" means *refused*, not *closest row wins*.
- **Alerts do not wait on the VLM.** They are raised on the detector thread the
  moment a face has been unidentified for N consecutive checks, and carry a
  wall-clock time for a camera or an offset into the clip for a file.
- An **IoU tracker** gives each face a persistent ID, which is what turns
  per-frame detections into "what is happening" (appeared, left/right motion)
  and what identity is cached against between recognition cycles.
- The **VLM** reasons over the actual frame images **plus** the scene text
  (fusion B), which cuts hallucination. It is told the names and told never to
  reveal where they came from.
- A **LangGraph** state graph auto-retries a fallback model when the primary
  returns an empty answer.

## Setup

```bash
uv sync
cp .env.example .env      # then put your OpenAI key in it
```

`.env` is gitignored. Every setting can also be an ordinary environment
variable (`OPENAI_API_KEY`, `VLM_MODEL`, `VLM_FALLBACK`, `GALLERY_PATH`,
`FACES_DIR`), which takes precedence over `.env`. They are declared and
validated in one place, `vision/settings.py`.

Face recognition and alerts run entirely locally and need no key; only the
scene descriptions call OpenAI. Without a key the app still detects, identifies
and alerts — each probe just lands in the log as an error saying the key is
missing.

The face models (~340MB) download themselves to `~/.insightface` on first run.

### Enrol faces

Anyone not enrolled is reported as an intruder, so do this first. Two routes,
same code and same result:

**In the app** — the **Train** button, top right, with two tabs:

- *One person* — type a name, drop in photos; they are saved under
  `faces/<name>/` and the gallery is rebuilt.
- *Bulk from Label Studio* — upload the YOLO export **zip without media**
  (`classes.txt` + `labels/`) and every labelled person is seeded at once. The
  images are read from `labeldata/training/` (set `TRAINING_IMAGES_DIR` to move
  it), not from the zip — exports with media attached are huge and the pixels
  are already on disk.

Either way the new gallery is pushed straight into a running stream, so someone
you just enrolled stops being an unknown without restarting anything.

**From a Label Studio export** — if you already have bounding-box annotations
whose class names are people's names, they seed the gallery for free:

```bash
make seed         # == tools/seed_faces.py --export-dir export --faces-dir faces
#  ...prune the blurry and near-duplicate crops by hand...
make gallery
```

Add `--images-dir` when the export carries labels only — the same split the
Train dialog's bulk tab uses.

It expects the YOLO export layout (`classes.txt`, `images/`, `labels/`), crops
each labelled box with 25% padding, and skips crops under 80px. A class that
yields no usable crop is named in the output rather than dropped quietly — it
would otherwise be alerted on as a stranger.

**From the shell** — arrange photos yourself and build:

```
faces/
    Pratik Panchal/     5-20 varied photos
    Karishma Panchal/   5-20 varied photos
```

```bash
make gallery      # == uv run python tools/build_gallery.py --faces-dir faces
```

Every accepted photo becomes its own row and matching takes the best score
across a person's rows, so a frontal and a profile reference both count. The
build prints per-person *cohesion* (low means the folder mixes two people) and
cross-identity similarity (high means two enrolled people are confusable).

> `faces/` and `gallery.npz` are gitignored — they are biometric data.

`VLM_MODEL` / `VLM_FALLBACK` seed the sidebar defaults (see the `Makefile`);
both are editable live in the app.

### Model and cost

| model | input /1M | output /1M | ~cost per hour at 15s probes |
|-------|-----------|------------|------------------------------|
| `gpt-5-mini` (default) | $0.25 | $2.00 | ~$0.20 |
| `gpt-5.4-mini` | $0.75 | $4.50 | ~$0.63 |

Each probe sends up to 3 frames at 512px longest edge (~255 tokens each) plus a
short prompt, and caps output at 2000 tokens — generous because reasoning
tokens are billed as output and counted against that cap, so a tight limit
starves the answer and returns an empty string. Brevity comes from the prompt
rules and `answer.tidy`, not the token ceiling.

## Run

```bash
make run          # == uv run streamlit run app.py --server.runOnSave=false
```

In the browser:

- **Video** — a WebRTC stream in a real `<video>` element; the **browser** opens
  the camera, so no Python process can hold it hostage. Only the detected view
  is shown — a green box names a known face, a red `UNKNOWN` box marks one we
  refused to name.
- **Identity** — match threshold, runner-up margin, and how often to
  re-identify. The gallery and photo paths come from `GALLERY_PATH` /
  `FACES_DIR` (defaults `gallery.npz` and `faces/`), not the sidebar, so
  enrolment and detection can never point at different files. Boxes are drawn every frame regardless; the cadence only paces
  the naming. Threshold and margin apply live, without restarting the stream.
- **Alerts** — how many consecutive unknown checks before raising, and how long
  before the same face may alert again. Newest-first feed, exportable.
- **Probe** — the prompt, the interval, how many prior answers to remember, and
  how many frames each probe sends. The scheduler follows the stream: it runs
  while the video is playing and pauses the moment you stop it, so a stopped
  stream never burns VLM calls on a stale buffer. **Probe now** works either
  way, for one-off questions.
- **Event log** — probe answers, newest first, capped at 200, JSON or CSV.

Press **START** on the video component to begin; the browser will ask for camera
permission.

All browser tabs share one camera, one scheduler and one log.

## Performance

Measured at 640×480 on an M4, warm, median of interleaved runs:

| providers | faces | recognise every | ms/frame | detect fps |
|-----------|-------|-----------------|----------|-----------|
| CoreML    | 1-2   | 1               | 50       | 20.2      |
| CoreML    | 1-2   | 4               | 41       | 24.5      |
| CoreML    | 6     | 1               | 68       | 14.8      |
| CoreML    | 6     | 4               | 50       | 20.1      |
| CPU       | 1-2   | 1               | 187      | 5.3       |
| CPU       | 6     | 1               | 351      | 2.9       |

CoreML is ~4x CPU here, so it is the default. Raising the cadence saves only
~25% on CoreML and costs alert latency (the debounce counts recognition
checks), which is why the default identifies on every frame.

In the shape the app actually runs — one long-lived detector — steady state is
reached straight after warm-up: 45ms (22fps) with 1-2 faces, 61ms (16fps) with
six. At 22fps the default debounce of 10 unknown checks means a stranger is
alerted in roughly half a second.

> Benchmark honestly: CoreML compiles its partitions lazily and needs several
> inferences to reach steady state. Timing it after a single warm-up frame makes
> it look *slower* than CPU. `warmup()` runs five frames for this reason. Also
> measure one detector at a time — several alive at once contend and read 3-4x
> slow.

## Layout

| File | Role |
|------|------|
| `vision/records.py` | pure: frozen `Face` / `FrameRecord` / `Pose` |
| `vision/identity.py` | pure: gallery loading + open-set cosine matching |
| `vision/facetrack.py` | pure: IoU tracking, identity/pose cache, cadence |
| `vision/facemeta.py` | pure: position, pose wording, sharpness, assembly |
| `vision/alerts.py` | pure: unknown-face debounce + cooldown policy |
| `vision/detector.py` | InsightFace models → `FrameRecord` (the only model code) |
| `vision/overlay.py` | identity boxes drawn on the display frame |
| `vision/buffer.py` | rolling window of recent frames |
| `vision/scene_context.py` | pure: buffer → grounding text |
| `vision/memory.py` | pure: last K answers → prompt preamble |
| `vision/webrtc.py` | WebRTC frame sink, detector thread, alerting, detected view |
| `vision/probe.py` | probe config, one-shot probe, scheduler thread |
| `vision/events.py` / `vision/eventlog.py` | probe + alert records, capped JSON/CSV logs |
| `vision/settings.py` | pydantic-settings: every env var, validated, key as `SecretStr` |
| `vision/vlm_client.py` | model defaults, prompt rules, frame encoding, API-key access |
| `vision/llm.py` | ChatOpenAI factory + multimodal message builder |
| `vision/graph.py` | LangGraph state graph (fallback, optional memory) |
| `vision/enroll.py` | pure-ish: name/photo safety, rebuild, quality reports |
| `vision/models.py` | loading the buffalo_l ONNX models |
| `vision/labelstudio.py` | Label Studio YOLO export (dir or zip) -> `faces/<person>/` crops |
| `tools/seed_faces.py` | thin CLI over `vision/labelstudio.py` (`make seed`) |
| `tools/build_gallery.py` | thin CLI over `vision/enroll.py` (`make gallery`) |
| `app.py` | Streamlit UI |

## Known limits

Faces only. Dropping YOLO removed object classes and body detection, so a
back-turned, masked or distant intruder produces no face, no track and no
alert. Re-adding a small person-only detector as a pure presence signal is the
way back if that matters.
