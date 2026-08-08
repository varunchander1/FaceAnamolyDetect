"""Length guard for VLM output.

Tone, preamble and detector-jargon are handled by the SystemMessage in
`vlm_client.INSTRUCTIONS`, so there is no regex scrubbing here. Length is the
rule models most reliably drift on (a 50-word instruction routinely came back
at 28-73 words), so it gets a deterministic backstop — which also means the
output token ceiling never has to double as a brevity control.

Pure functions, no dependencies.
"""

from __future__ import annotations

import re

MAX_SENTENCES = 3
MAX_WORDS = 60  # a little above the 50 the prompt asks for: only catch overruns

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def clamp_sentences(text: str, max_sentences: int = MAX_SENTENCES) -> str:
    """Keep at most `max_sentences` sentences, on one paragraph."""
    if max_sentences <= 0:
        return text.strip()
    flat = " ".join(line.strip() for line in text.splitlines() if line.strip())
    sentences = _SENTENCE_END_RE.split(flat)
    return " ".join(sentences[:max_sentences]).strip()


def clamp_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Hard word budget: the model often answers in one long run-on sentence.

    Cuts back to the last sentence end inside the budget when there is one;
    otherwise truncates and marks the cut with an ellipsis.
    """
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text.strip()

    head = " ".join(words[:max_words])
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", head)]
    if ends and ends[-1] >= len(head) // 2:
        return head[: ends[-1]].strip()
    return head.rstrip(" ,;:-") + "…"


def tidy(
    text: str, max_sentences: int = MAX_SENTENCES, max_words: int = MAX_WORDS
) -> str:
    """Clamp an answer to something that fits an event card."""
    return clamp_words(clamp_sentences(text, max_sentences), max_words)
