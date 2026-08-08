"""Seed `faces/` from a Label Studio YOLO export, so existing box annotations
become recognition references for free.

Expected export layout (Label Studio -> Export -> YOLO):

    export/
        classes.txt
        images/   img001.jpg ...
        labels/   img001.txt ...   # <class_id> <cx> <cy> <w> <h>, normalised

If the export carries labels only, point --images-dir at the folder holding the
source images.

This only writes crops. Prune by hand, then build the gallery:

    uv run python tools/seed_faces.py --export-dir export
    #  ...delete the blurry and near-duplicate crops...
    make gallery

The app's Train dialog does the same thing from an uploaded zip.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.enroll import GOOD_PHOTOS, MIN_PHOTOS, EnrolmentError  # noqa: E402
from vision.labelstudio import (  # noqa: E402
    DEFAULT_MIN_SIZE,
    DEFAULT_PAD,
    SeedReport,
    seed_from_export,
)
from vision.settings import settings  # noqa: E402


def report(result: SeedReport) -> None:
    for person in sorted(result.people, key=lambda p: p.name):
        if person.is_empty:
            # Loud on purpose: they are labelled in the export, so you expect
            # them enrolled, but they will be alerted on as strangers.
            print(f"  {person.name}: 0 crops  <-- NOT seeded")
            continue
        thin = f"  <-- thin, {GOOD_PHOTOS}-20 is best" if person.crops < MIN_PHOTOS else ""
        print(f"  {person.name}: {person.crops} crops{thin}")

    print(f"\n{result.total_crops} crop(s) written to {Path(result.out_dir).resolve()}")

    if result.skipped_small:
        print(f"  skipped {result.skipped_small} crop(s) below --min-size")
    if result.skipped_malformed:
        print(f"  skipped {result.skipped_malformed} malformed label line(s)")
    if result.missing_images:
        print(f"  {len(result.missing_images)} label file(s) had no image: "
              f"{', '.join(result.missing_images[:5])}")
    if result.unreadable_images:
        print(f"  {len(result.unreadable_images)} unreadable image(s): "
              f"{', '.join(result.unreadable_images[:5])}")
    if result.empty_classes:
        print(f"  NOT seeded: {', '.join(result.empty_classes)}")

    print(
        "\nNow prune by hand — keep sharp, varied shots per person and delete "
        "the rest — then run `make gallery`."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="export")
    parser.add_argument(
        "--faces-dir",
        default=settings.faces_dir,
        help="where crops land, one folder per person (default: FACES_DIR)",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help=(
            "where the source images live (default: <export-dir>/images). Use "
            f"this when the export has labels only, e.g. {settings.training_images_dir}"
        ),
    )
    parser.add_argument(
        "--pad", type=float, default=DEFAULT_PAD,
        help="expand each box by this fraction; ArcFace likes some context",
    )
    parser.add_argument(
        "--min-size", type=int, default=DEFAULT_MIN_SIZE,
        help="skip crops smaller than this many pixels on the short side",
    )
    args = parser.parse_args()

    try:
        result = seed_from_export(
            args.export_dir, args.faces_dir, args.pad, args.min_size, args.images_dir
        )
    except EnrolmentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
