"""
Live Face-ID + Activity Chat — Streamlit app

What it does
------------
- Runs your existing YOLO person model + trained YOLO face-ID model on a live
  webcam feed in a background thread.
- Every few seconds, sends the current frame to OpenAI's vision model to get a
  short natural-language description of what's happening.
- Keeps a rolling log of (time, names present, activity caption).
- Lets you chat with the app; questions are answered using that rolling log
  (not by "watching" the live video directly).

Setup
-----
    pip install -r requirements.txt
    setx OPENAI_API_KEY "sk-..."      # Windows (restart terminal after)
    # or: export OPENAI_API_KEY="sk-..."   # macOS/Linux

Run
---
    streamlit run app.py

Notes
-----
- Update PERSON_MODEL_PATH / FACE_MODEL_PATH / CAMERA_INDEX below to match
  your setup (same paths you were already using in your notebook).
- The face model's class names are assumed to be the person names you
  labeled during training (i.e. face_model.names -> {0: "Ashish", 1: "Riya", ...}).
"""

import base64
import os
import tempfile
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import streamlit as st
import torch
from openai import OpenAI
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — adjust to match your environment
# ---------------------------------------------------------------------------
PERSON_MODEL_PATH = "yolo11l.pt"
FACE_MODEL_PATH = "model/my_model.pt"
CAMERA_INDEX = 0
CAPTION_INTERVAL_SECONDS = 4       # how often we ask the vision model for a caption
LOG_MAX_ENTRIES = 200              # rolling log cap
CHAT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"

# Inference performance settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # run on GPU automatically if available
INFERENCE_IMG_SIZE = 480                                   # lower than YOLO's default 640 for faster inference

client = OpenAI()  # reads OPENAI_API_KEY from the environment


# ---------------------------------------------------------------------------
# Shared state between the background capture thread and the Streamlit UI
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None                       # annotated BGR frame
        self.current_names = []                        # names in most recent frame
        self.log = deque(maxlen=LOG_MAX_ENTRIES)        # dicts: ts, names, caption
        self.running = False
        self.error = None
        self.source_type = None                       # "camera" or "video"
        self.video_path = None
        self.source_name = None

    def append_log(self, names, caption):
        with self.lock:
            self.log.append({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "names": names,
                "caption": caption,
            })

    def snapshot_log_text(self, max_entries=15):
        with self.lock:
            entries = list(self.log)[-max_entries:]
        if not entries:
            return "No observations logged yet."
        lines = [
            f"[{e['ts']}] Present: {', '.join(e['names']) or 'no one identified'}. "
            f"Activity: {e['caption']}"
            for e in entries
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------
def frame_to_b64_jpeg(frame, quality=70):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8") if ok else None


def get_activity_caption(frame, names):
    b64 = frame_to_b64_jpeg(frame)
    if b64 is None:
        return "unable to capture frame"

    who = ", ".join(names) if names else "the person(s) visible"
    prompt = (
        f"This is a frame from a live webcam/uploaded video. {who} may be visible. "
        "In one short sentence, describe what the person/people are doing "
        "(posture, action, what they're interacting with). Be concise and factual."
    )
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"(caption failed: {exc})"


def answer_question(state: "SharedState", question: str):
    context = state.snapshot_log_text()
    with state.lock:
        current = ", ".join(state.current_names) if state.current_names else "no one identified right now"

    system_prompt = (
        "You are an assistant answering questions about a live webcam/uploaded video feed, based "
        "only on a rolling observation log you're given. If the answer isn't in "
        "the log, say you don't have that information yet rather than guessing. "
        "Be concise and conversational."
    )
    user_prompt = (
        f"Currently visible: {current}\n\n"
        f"Recent observation log (most recent last):\n{context}\n\n"
        f"Question: {question}"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"Sorry, I couldn't get an answer right now ({exc})."


# ---------------------------------------------------------------------------
# Helpers for uploaded video support
# ---------------------------------------------------------------------------
def save_uploaded_video(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


# ---------------------------------------------------------------------------
# Background capture / detection loop
# ---------------------------------------------------------------------------
def process_feed_loop(state: SharedState, person_model, face_model):
    source_type = state.source_type or "camera"
    if source_type == "video":
        video_path = state.video_path
        cap = cv2.VideoCapture(video_path) if video_path else None
        if cap is None or not cap.isOpened():
            with state.lock:
                state.error = "Could not open uploaded video."
                state.running = False
            return
        source_name = state.source_name or "uploaded video"
    else:
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            with state.lock:
                state.error = "Could not open camera."
                state.running = False
            return
        source_name = "camera"

    last_caption_time = 0.0
    frame_count = 0

    while state.running:
        try:
            ret, frame = cap.read()
            if not ret:
                if source_type == "video":
                    break
                time.sleep(0.05)
                continue

            frame_count += 1
            person_result = person_model(
                frame, verbose=False, device=DEVICE, imgsz=INFERENCE_IMG_SIZE
            )[0]
            face_result = face_model(
                frame, verbose=False, device=DEVICE, imgsz=INFERENCE_IMG_SIZE
            )[0]

            has_person = any(int(b.cls[0]) == 0 for b in person_result.boxes)

            names_present = set()
            for box in face_result.boxes:
                cls_id = int(box.cls[0])
                names_present.add(face_model.names.get(cls_id, "unknown"))

            annotated = person_result.plot()
            annotated = face_result.plot(img=annotated)

            with state.lock:
                state.latest_frame = annotated
                state.current_names = sorted(names_present)

            now = time.time()
            if has_person and (now - last_caption_time >= CAPTION_INTERVAL_SECONDS):
                caption = get_activity_caption(frame, sorted(names_present))
                state.append_log(sorted(names_present), caption)
                last_caption_time = now

            if source_type == "video" and frame_count % 5 == 0:
                time.sleep(0.01)

        except Exception as exc:
            with state.lock:
                state.error = f"Processing stopped due to an error: {exc}"
            break

    cap.release()
    with state.lock:
        if source_type == "video" and not state.error:
            state.error = f"Finished processing {source_name}."
        state.running = False


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Live Face-ID & Activity Chat", layout="centered")
st.title("Live Face-ID & Activity Chat")

if "state" not in st.session_state:
    st.session_state.state = SharedState()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_video" not in st.session_state:
    st.session_state.uploaded_video = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "uploaded_video_name" not in st.session_state:
    st.session_state.uploaded_video_name = None
if "initial_caption_posted" not in st.session_state:
    st.session_state.initial_caption_posted = False

state: SharedState = st.session_state.state


@st.cache_resource
def load_models():
    person_model = YOLO(PERSON_MODEL_PATH)
    face_model = YOLO(FACE_MODEL_PATH)
    return person_model, face_model


with st.sidebar:
    st.header("Input source")

    st.caption("Choose a live camera feed or upload a video.")

    if st.button("Use live camera", use_container_width=True, type="primary"):
        state.source_type = "camera"
        state.video_path = None
        state.source_name = "camera"
        st.session_state.uploaded_video_path = None
        st.session_state.uploaded_video_name = None
        st.success("Camera selected")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload a video clip for analysis. Please keep the video under 2 minutes.",
    )
    if uploaded_video is not None:
        file_id = getattr(uploaded_video, "file_id", None) or (uploaded_video.name, uploaded_video.size)
        if st.session_state.get("uploaded_video_file_id") != file_id:
            temp_path = save_uploaded_video(uploaded_video)
            st.session_state.uploaded_video = uploaded_video
            st.session_state.uploaded_video_path = temp_path
            st.session_state.uploaded_video_name = uploaded_video.name
            st.session_state.uploaded_video_file_id = file_id
            state.source_type = "video"
            state.video_path = temp_path
            state.source_name = uploaded_video.name
            st.success(f"Uploaded: {uploaded_video.name}")
    elif st.session_state.get("uploaded_video_path"):
        st.info("Using the previously uploaded video.")

    st.caption("Disclaimer: Please keep uploaded videos under 2 minutes for best performance.")

    st.header("Processing control")
    if not state.running:
        if st.button("▶ Start processing", use_container_width=True):
            if st.session_state.get("uploaded_video_path"):
                state.source_type = "video"
                state.video_path = st.session_state.uploaded_video_path
                state.source_name = st.session_state.get("uploaded_video_name", "uploaded video")
            else:
                state.source_type = "camera"
                state.video_path = None
                state.source_name = "camera"

            try:
                person_model, face_model = load_models()
            except Exception as exc:
                state.error = f"Failed to load models: {exc}"
            else:
                state.running = True
                state.error = None
                st.session_state.initial_caption_posted = False
                threading.Thread(
                    target=process_feed_loop,
                    args=(state, person_model, face_model),
                    daemon=True,
                ).start()
            st.rerun()
    else:
        if st.button("⏹ Stop processing", use_container_width=True):
            state.running = False
            st.rerun()

    st.caption(f"Caption interval: {CAPTION_INTERVAL_SECONDS}s")
    st.caption(f"Inference device: {DEVICE} | Image size: {INFERENCE_IMG_SIZE}px")
    if state.error:
        st.error(state.error)

st.subheader("Ask about the feed")

if st.button("Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.initial_caption_posted = False
    st.rerun()


@st.fragment(run_every=0.5)
def watch_for_first_observation():
    """Polls the shared log while processing is running. As soon as the
    first observation lands, posts it into chat as the assistant's first
    message and triggers a full rerun so it renders in the message list."""
    with state.lock:
        running = state.running
        has_entries = len(state.log) > 0
        first_entry = state.log[0] if has_entries else None

    if running and has_entries and not st.session_state.initial_caption_posted:
        with state.lock:
            names = ", ".join(first_entry["names"]) if first_entry["names"] else "no one identified yet"
        intro = f"Here's what I'm seeing so far — {names}. {first_entry['caption']}"
        st.session_state.chat_history.append(("assistant", intro))
        st.session_state.initial_caption_posted = True
        st.rerun()
    elif running and not has_entries:
        st.caption("Watching the feed... I'll describe what's happening as soon as I see something.")


if state.running and not st.session_state.initial_caption_posted:
    watch_for_first_observation()

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Who's there and what are they doing?")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = answer_question(state, question)
        st.markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))