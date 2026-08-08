"""Sliding-window probe memory.

Pure functions only. The last K probe answers are folded into the *question*
text rather than the scene context. Both reach the model today, but the context
is the detector's channel and gets reshaped whenever grounding changes; the
question is the user's, so memory riding there cannot be dropped by an edit
aimed at something else.
"""

from __future__ import annotations

PREAMBLE = "Previously observed (oldest first):"


def recent_answers(answers: tuple[str, ...], k: int) -> tuple[str, ...]:
    """Last k non-blank answers, oldest first. Empty when k <= 0."""
    if k <= 0:
        return ()
    kept = tuple(a.strip() for a in answers if a and a.strip())
    return kept[-k:]


def build_probe_question(prompt: str, answers: tuple[str, ...], k: int) -> str:
    """Prefix `prompt` with up to k prior answers; unchanged when there are none."""
    recent = recent_answers(answers, k)
    if not recent:
        return prompt
    bullets = "\n".join(f"- {a}" for a in recent)
    return f"{PREAMBLE}\n{bullets}\n\n{prompt}"
