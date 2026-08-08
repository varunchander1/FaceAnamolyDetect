"""OpenAI vision layer: model defaults, prompt rules, and frame encoding.

Holds everything about *what* we send, so `llm` can stay about *how* it is
wrapped for LangChain. No network calls happen here.

Model choice: `gpt-5-mini` is the cheapest OpenAI model with image input
($0.25/1M in, $2.00/1M out), which at a 15s probe interval is roughly $0.20 an
hour. `gpt-5.4-mini` is newer and stronger at ~3x the price; both are vision
capable, and the model is a sidebar field, so switching is one edit.
"""

from __future__ import annotations

import cv2

from vision.settings import API_KEY_ENV, settings

# Override at runtime:  VLM_MODEL=gpt-5.4-mini make run
MODEL = settings.vlm_model

__all__ = [
    "API_KEY_ENV",
    "IMAGE_DETAIL",
    "INSTRUCTIONS",
    "MAX_FRAMES",
    "MAX_OUTPUT_TOKENS",
    "MAX_RETRIES",
    "MAX_SIDE",
    "MODEL",
    "REASONING_EFFORT",
    "REQUEST_TIMEOUT_S",
    "VLMError",
    "api_key",
    "encode_frame",
    "has_api_key",
]

# Reasoning tokens are billed as output and counted against this cap, so a tight
# budget starves the visible answer and the model returns "". That is exactly
# the failure the old local reasoning model had; the ceiling is generous and
# brevity is enforced by INSTRUCTIONS + `answer.tidy` instead.
MAX_OUTPUT_TOKENS = 2000
REASONING_EFFORT = "low"  # a scene description needs no deliberation
REQUEST_TIMEOUT_S = 30.0
MAX_RETRIES = 2

MAX_FRAMES = 5  # newest N frames per probe
MAX_SIDE = 512  # longest edge before JPEG encoding
# 512px longest edge is one tile, so "high" costs ~255 tokens per image against
# "low"'s 85 — under $0.0002 a probe either way. Fidelity is worth more.
IMAGE_DETAIL = "high"

INSTRUCTIONS = (
    "You are watching a scene through a camera and reporting what happens. "
    "You receive images sampled over the last few seconds, oldest first.\n"
    "You also receive private notes from a face-recognition system, including "
    "people's names. Use the names — they are the point — but NEVER quote the "
    "notes or reveal where the names came from. Do not use the words detection, "
    "detector, recognition, gallery, embedding, bounding box, label, "
    "confidence, ID, track, tracked, frame, or image, and never write things "
    "like '[0]', 'track 7', 'face count', '0.89', or 'the system detects'. "
    "Someone the notes call UNKNOWN or unidentified is simply a person you do "
    "not recognise: say so plainly, e.g. 'someone I don't recognise'. Describe "
    "what a person watching the room would say — who is there, what they are "
    "doing, what changed.\n"
    "Answer in 2-3 short sentences, maximum 50 words. No lists, no headings, "
    "no per-image breakdown.\n"
    "Start directly with what is happening. Never open with a preamble such as "
    '"Based on the provided frames", "According to the context", '
    '"Here is what is happening" — those are forbidden.'
)


class VLMError(RuntimeError):
    """Raised when the vision call fails (no API key, rate limit, etc.)."""


def api_key() -> str:
    """The configured key, or a VLMError naming the variable to set.

    Fetched at call time rather than held in a module constant, so the key
    lives in the `SecretStr` on `settings` and only materialises as plain text
    at the point it is handed to the client.
    """
    try:
        return settings.require_api_key()
    except ValueError as exc:
        raise VLMError(str(exc)) from exc


def has_api_key() -> bool:
    """Whether a key is configured, without raising. Never returns the key."""
    return settings.has_api_key


def _downscale(frame, max_side: int):
    """Shrink so the longest side is <= max_side (keeps aspect ratio)."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def encode_frame(frame, max_side: int = MAX_SIDE) -> bytes:
    """BGR numpy frame -> downscaled JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", _downscale(frame, max_side))
    if not ok:
        raise VLMError("Failed to JPEG-encode frame.")
    return buf.tobytes()
