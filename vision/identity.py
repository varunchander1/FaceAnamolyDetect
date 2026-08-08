"""Open-set face identity: cosine match against an enrolled gallery.

Pure — no models, no I/O beyond reading the gallery file. Ported from the
VideoRAG POC (`extract.py`), with one deliberate change: a person may own
several embeddings, so the runner-up used by the margin gate is the best score
belonging to a *different* person. Gating alice against another photo of alice
would reject every well-enrolled face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vision.records import STATUS_MATCHED, STATUS_UNKNOWN

DEFAULT_THRESHOLD = 0.40  # min cosine similarity to claim a match
DEFAULT_MARGIN = 0.05  # min lead over the best other person

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Gallery:
    """Enrolled faces. `names[i]` owns `embeddings[i]`; names may repeat."""

    names: tuple[str, ...]
    embeddings: np.ndarray  # (N, D) float32, rows L2-normalised

    @property
    def is_empty(self) -> bool:
        return len(self.names) == 0

    @property
    def people(self) -> tuple[str, ...]:
        """Distinct enrolled people, in first-seen order."""
        return tuple(dict.fromkeys(self.names))


@dataclass(frozen=True)
class Match:
    """Outcome of matching one probe embedding against a gallery.

    `identity`/`confidence` are set only on a match, but `best_*` is filled in
    either way: on a refusal, "closest was Alice at 0.31" is exactly what an
    operator needs to judge whether the threshold is wrong or the stranger real.
    """

    status: str  # STATUS_MATCHED | STATUS_UNKNOWN
    identity: str | None = None
    confidence: float | None = None
    best_name: str | None = None
    best_similarity: float | None = None
    runner_up: tuple[str, float] | None = None

    @property
    def nearest(self) -> str:
        """Closest gallery entry as "name 0.31", or "" if the gallery is empty."""
        if self.best_name is None or self.best_similarity is None:
            return ""
        return f"{self.best_name} {self.best_similarity:.2f}"


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, returning a new array. A zero vector is returned as-is."""
    vec = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.copy()
    return vec / norm


def match(
    embedding: np.ndarray,
    gallery: Gallery,
    threshold: float = DEFAULT_THRESHOLD,
    margin: float = DEFAULT_MARGIN,
) -> Match:
    """Identify one face, or refuse to.

    A match requires both a high enough similarity *and* a clear enough lead
    over the next-best person — the second gate is what makes "unknown" mean
    something rather than "closest row wins".
    """
    if gallery.is_empty:
        return Match(status=STATUS_UNKNOWN)

    probe = normalize(embedding)
    if probe.shape[0] != gallery.embeddings.shape[1]:
        raise ValueError(
            f"embedding dimension {probe.shape[0]} does not match "
            f"gallery dimension {gallery.embeddings.shape[1]}"
        )

    sims = gallery.embeddings @ probe
    best = int(np.argmax(sims))
    best_name = gallery.names[best]
    best_sim = float(sims[best])

    runner_up = _best_other_person(sims, gallery.names, best_name)
    lead = best_sim - runner_up[1] if runner_up is not None else 1.0

    nearest = {
        "best_name": best_name,
        "best_similarity": best_sim,
        "runner_up": runner_up,
    }

    if best_sim < threshold or lead < margin:
        return Match(status=STATUS_UNKNOWN, **nearest)

    return Match(
        status=STATUS_MATCHED, identity=best_name, confidence=best_sim, **nearest
    )


def load_gallery(path: str | Path) -> Gallery:
    """Read an enrolment file written by `tools/build_gallery.py`.

    Raises FileNotFoundError or ValueError with a message the UI can show —
    a silently empty gallery would mark every person unknown.
    """
    gallery_path = Path(path)
    if not gallery_path.is_file():
        raise FileNotFoundError(
            f"Gallery not found at {gallery_path}. Enrol faces first: "
            "`make gallery` (see tools/build_gallery.py)."
        )

    data = np.load(gallery_path, allow_pickle=True)
    for key in ("names", "embeddings"):
        if key not in data:
            raise ValueError(f"{gallery_path} is missing the '{key}' array.")

    names = tuple(str(n) for n in data["names"])
    raw = np.asarray(data["embeddings"], dtype=np.float32)

    if raw.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {raw.shape}.")
    if len(names) != raw.shape[0]:
        raise ValueError(
            f"{len(names)} names but {raw.shape[0]} embeddings in {gallery_path}."
        )

    return Gallery(names=names, embeddings=_normalize_rows(raw))


def _best_other_person(
    sims: np.ndarray, names: tuple[str, ...], winner: str
) -> tuple[str, float] | None:
    """Highest-scoring row belonging to anyone but `winner`."""
    others = [(float(s), n) for s, n in zip(sims, names) if n != winner]
    if not others:
        return None
    score, name = max(others)
    return (name, score)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
