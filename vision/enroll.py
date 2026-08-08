"""Enrol faces: save reference photos, then rebuild the gallery from them.

Shared by the `Train` dialog in the app and `tools/build_gallery.py`, so both
produce byte-identical galleries. Nothing here imports Streamlit.

Every accepted photo keeps its own row rather than being averaged into one
centroid per person: averaging collapses appearance variation, which is what
made profile and low-light faces fall to "unknown" in the POC this replaced.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vision.identity import normalize

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_PHOTOS = 3  # below this, recognition gets brittle
GOOD_PHOTOS = 5
LOW_COHESION = 0.45  # below this, the folder probably mixes two people
CONFUSABLE = 0.5  # above this, two enrolled people look alike
DEFAULT_MIN_DET_SCORE = 0.60
ALIGNED_FACE_PX = 112

MAX_NAME_LEN = 64
_UNSAFE = re.compile(r"[^\w \-'.]", re.UNICODE)
_COLLAPSE = re.compile(r"\s+")


@dataclass(frozen=True)
class PersonReport:
    """What happened to one person's folder during a rebuild."""

    name: str
    accepted: int
    skipped: tuple[str, ...]  # human-readable reasons, one per rejected photo
    cohesion: float | None = None  # None when there are too few photos to judge

    @property
    def is_enrolled(self) -> bool:
        return self.accepted > 0

    @property
    def is_thin(self) -> bool:
        return 0 < self.accepted < MIN_PHOTOS

    @property
    def is_incoherent(self) -> bool:
        return self.cohesion is not None and self.cohesion < LOW_COHESION


@dataclass(frozen=True)
class BuildReport:
    """Outcome of a whole rebuild — everything the UI or CLI needs to report."""

    people: tuple[PersonReport, ...]
    confusable: tuple[tuple[str, str, float], ...]  # (a, b, worst similarity)
    total_embeddings: int
    out_path: str

    @property
    def enrolled(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.people if p.is_enrolled)

    @property
    def skipped_people(self) -> tuple[str, ...]:
        """Folders that produced no usable face — they will read as intruders."""
        return tuple(p.name for p in self.people if not p.is_enrolled)


class EnrolmentError(ValueError):
    """A problem the user can fix, phrased for them."""


def safe_person_name(name: str) -> str:
    """Normalise a typed name into something safe to use as a folder.

    Rejects rather than silently mangles: a person quietly enrolled under the
    wrong folder is worse than an error message. Path separators and `..` are
    stripped, so a crafted name cannot write outside the faces directory.
    """
    cleaned = _COLLAPSE.sub(" ", _UNSAFE.sub(" ", unicodedata.normalize("NFC", name)))
    cleaned = cleaned.strip(" .")

    if not cleaned:
        raise EnrolmentError("Enter a name for this person.")
    if len(cleaned) > MAX_NAME_LEN:
        raise EnrolmentError(f"Name is too long (max {MAX_NAME_LEN} characters).")
    return cleaned


def save_photos(
    faces_dir: str | Path, name: str, photos: tuple[tuple[str, bytes], ...]
) -> tuple[Path, ...]:
    """Write `(filename, bytes)` pairs into `faces_dir/<name>/`.

    Returns the paths written. Existing photos for that person are kept —
    enrolling twice adds references rather than replacing them.
    """
    if not photos:
        raise EnrolmentError("Add at least one photo.")

    person = Path(faces_dir) / safe_person_name(name)
    person.mkdir(parents=True, exist_ok=True)

    written = []
    for filename, data in photos:
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise EnrolmentError(
                f"{filename}: unsupported file type. Use "
                f"{', '.join(sorted(IMAGE_EXTS))}."
            )
        if not data:
            raise EnrolmentError(f"{filename}: file is empty.")
        written.append(_write_unique(person, Path(filename).name, data))

    return tuple(written)


def build_gallery(
    faces_dir: str | Path,
    out_path: str | Path,
    detector,
    recognizer,
    min_det_score: float = DEFAULT_MIN_DET_SCORE,
) -> BuildReport:
    """Re-embed every person folder and write the gallery file.

    `detector` / `recognizer` are the insightface models — injected so this is
    testable with fakes and so the caller decides where they come from.
    """
    root = Path(faces_dir)
    if not root.is_dir():
        raise EnrolmentError(
            f"No faces directory at {root}. Add photos before building."
        )

    person_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not person_dirs:
        raise EnrolmentError(f"No person folders inside {root}.")

    reports, names, rows = [], [], []
    for person_dir in person_dirs:
        embeddings, skipped = embed_person(
            detector, recognizer, person_dir, min_det_score
        )
        stacked = np.stack(embeddings) if embeddings else None
        reports.append(
            PersonReport(
                name=person_dir.name,
                accepted=len(embeddings),
                skipped=tuple(skipped),
                cohesion=None if stacked is None else cohesion(stacked),
            )
        )
        names += [person_dir.name] * len(embeddings)
        rows += embeddings

    if not rows:
        raise EnrolmentError(
            "No usable faces found in any folder. Photos need a clear, "
            "front-facing face."
        )

    embeddings_arr = np.stack(rows).astype(np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, names=np.array(names), embeddings=embeddings_arr)

    return BuildReport(
        people=tuple(reports),
        confusable=confusable_pairs(names, embeddings_arr),
        total_embeddings=len(names),
        out_path=str(out_path),
    )


def embed_person(
    detector, recognizer, person_dir: Path, min_det_score: float
) -> tuple[list[np.ndarray], list[str]]:
    """Unit-norm embedding per usable photo, plus a reason for each rejection."""
    from insightface.utils import face_align

    embeddings: list[np.ndarray] = []
    skipped: list[str] = []

    for path in sorted(person_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        image = cv2.imread(str(path))
        if image is None:
            skipped.append(f"{path.name}: unreadable")
            continue

        boxes, kpss = detector.detect(image, max_num=0)
        if boxes is None or len(boxes) == 0:
            skipped.append(f"{path.name}: no face found")
            continue

        # A reference photo should hold one subject; if several, take the biggest.
        best = max(range(len(boxes)), key=lambda i: _area(boxes[i]))
        score = float(boxes[best][4])
        if score < min_det_score:
            skipped.append(f"{path.name}: face too unclear ({score:.2f})")
            continue

        aligned = face_align.norm_crop(
            image, landmark=kpss[best], image_size=ALIGNED_FACE_PX
        )
        embeddings.append(normalize(recognizer.get_feat([aligned])[0].ravel()))

    return embeddings, skipped


def cohesion(embeddings: np.ndarray) -> float | None:
    """Mean pairwise cosine similarity — low means the folder mixes people."""
    n = len(embeddings)
    if n < 2:
        return None
    sims = embeddings @ embeddings.T
    return float((sims.sum() - n) / (n * (n - 1)))


def confusable_pairs(
    names: list[str], embeddings: np.ndarray, threshold: float = CONFUSABLE
) -> tuple[tuple[str, str, float], ...]:
    """People whose closest rows are near enough to be mistaken for each other."""
    people = sorted(set(names))
    index = {p: [i for i, n in enumerate(names) if n == p] for p in people}

    out = []
    for i, a in enumerate(people):
        for b in people[i + 1 :]:
            worst = float((embeddings[index[a]] @ embeddings[index[b]].T).max())
            if worst > threshold:
                out.append((a, b, worst))
    return tuple(out)


def _write_unique(person_dir: Path, filename: str, data: bytes) -> Path:
    """Never overwrite: two photos of the same name are two references."""
    stem, suffix = Path(filename).stem or "photo", Path(filename).suffix.lower()
    stem = _UNSAFE.sub("_", stem)[:MAX_NAME_LEN] or "photo"

    target = person_dir / f"{stem}{suffix}"
    counter = 1
    while target.exists():
        target = person_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    target.write_bytes(data)
    return target


def _area(box) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))
