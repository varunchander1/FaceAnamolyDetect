"""Rebuild the face gallery from a folder-per-person layout.

Expected input:

    faces/
        Pratik Panchal/    5-20 varied photos

Output: gallery.npz -> {"names": [N], "embeddings": (N, 512) float32}

The same code backs the app's **Train** dialog (`vision/enroll.py`), so both
routes produce identical galleries.

Usage:
    uv run python tools/build_gallery.py --faces-dir faces --out gallery.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.enroll import (  # noqa: E402
    DEFAULT_MIN_DET_SCORE,
    GOOD_PHOTOS,
    MIN_PHOTOS,
    BuildReport,
    EnrolmentError,
    build_gallery,
)
from vision.models import load_face_models  # noqa: E402


def report(result: BuildReport) -> None:
    for person in result.people:
        print(f"\n{person.name}:")
        for reason in person.skipped:
            print(f"    [skip] {reason}")
        if not person.is_enrolled:
            # Say this loudly: a person missing from the gallery is reported
            # as an intruder every time they walk past the camera.
            print("    [WARN] no usable faces — NOT enrolled")
            continue
        print(f"    [ok]   {person.accepted} photo(s)")
        if person.is_thin:
            print(f"    [warn] under {MIN_PHOTOS}; {GOOD_PHOTOS}-20 varied is best")
        if person.cohesion is not None:
            warn = "  <-- LOW: is this really all one person?" if person.is_incoherent else ""
            print(f"    cohesion: {person.cohesion:.3f}{warn}")

    print(
        f"\nSaved {len(result.enrolled)} identities "
        f"({result.total_embeddings} embeddings) to {result.out_path}: "
        f"{', '.join(result.enrolled)}"
    )
    if result.skipped_people:
        print(f"NOT enrolled: {', '.join(result.skipped_people)}")
    for a, b, score in result.confusable:
        print(f"CONFUSABLE: {a} vs {b} at {score:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces-dir", default="faces")
    parser.add_argument("--out", default="gallery.npz")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--min-det-score", type=float, default=DEFAULT_MIN_DET_SCORE)
    parser.add_argument("--cpu", action="store_true", help="skip CoreML")
    args = parser.parse_args()

    print("Loading buffalo_l (first run downloads ~300 MB of models)...")
    detector, recognizer = load_face_models(
        det_size=(args.det_size, args.det_size), cpu_only=args.cpu
    )

    try:
        result = build_gallery(
            args.faces_dir, args.out, detector, recognizer, args.min_det_score
        )
    except EnrolmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
