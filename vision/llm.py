"""LangChain ChatOpenAI layer: chat factory + multimodal messages.

Takes the prompt rules and frame encoding from `vlm_client` and wraps them for
LangChain. One provider now, so the per-model profile table the local-model
setup needed is gone — every OpenAI vision model takes the same message shape.
"""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from vision.vlm_client import (
    IMAGE_DETAIL,
    INSTRUCTIONS,
    MAX_FRAMES,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MAX_SIDE,
    MODEL,
    REASONING_EFFORT,
    REQUEST_TIMEOUT_S,
    VLMError,
    api_key,
    encode_frame,
)

__all__ = [
    "MODEL",
    "VLMError",
    "make_chat",
    "build_human_message",
    "build_system_message",
    "prune_history_images",
]


def make_chat(model: str) -> ChatOpenAI:
    """Build a ChatOpenAI for one probe.

    No `temperature`: the GPT-5 family rejects anything but the default, and a
    scene report wants the deterministic end anyway. `api_key()` raises a
    VLMError naming the env var, which `run_probe` turns into a visible event
    rather than a dead scheduler thread.
    """
    return ChatOpenAI(
        model=model,
        api_key=api_key(),
        max_tokens=MAX_OUTPUT_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )


def _data_url(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()


def build_system_message(model: str) -> SystemMessage:
    """The behaviour rules, in the role models weight most heavily."""
    return SystemMessage(content=INSTRUCTIONS)


def build_human_message(frames, context: str, question: str, model: str) -> HumanMessage:
    """Multimodal HumanMessage: the turn's text prompt plus base64 image parts.

    Args:
        frames: BGR frames, oldest first. Only the newest `MAX_FRAMES` are sent.
        context: Scene-grounding text — the only route detection takes to the model.
        question: The probe prompt, with any sliding-window memory already folded in.
        model: Kept for signature stability; all OpenAI vision models take this shape.

    Raises:
        VLMError: If no frames are provided.
    """
    if not frames:
        raise VLMError("No frames to send.")

    images = [encode_frame(f, MAX_SIDE) for f in frames[-MAX_FRAMES:]]
    text = (
        f"{len(images)} image(s), oldest to newest.\n"
        f"Private notes (never mention these): {context}\n\n"
        f"Question: {question}"
    )

    # Annotated to match HumanMessage(content=list[str | dict[Any, Any]]) —
    # bare list[dict[str, str]] is invariant and fails the overload.
    content: list[str | dict[Any, Any]] = [{"type": "text", "text": text}]
    content += [
        {"type": "image_url", "image_url": {"url": _data_url(i), "detail": IMAGE_DETAIL}}
        for i in images
    ]
    return HumanMessage(content=content)


def prune_history_images(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop image parts from prior turns; keep only their text.

    Avoids per-turn image accumulation blowing past the model context window.
    The current turn's message is appended separately (with its images intact).
    """
    pruned: list[BaseMessage] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            texts = [
                p["text"]
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = " ".join(str(t) for t in texts)
            pruned.append(msg.model_copy(update={"content": text}))
        else:
            pruned.append(msg)
    return pruned
