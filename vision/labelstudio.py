"""Turn a Label Studio YOLO export into `faces/<person>/` reference crops.

Existing bounding-box annotations seed the recognition gallery for free: the
class names are already people's names, so each labelled box is a usable
reference photo once cropped out.

Expected export layout (Label Studio -> Export -> YOLO):

    export/
        classes.txt
        images/   img001.jpg ...
        labels/   img001.txt ...   # <class_id> <cx> <cy> <w> <h>, normalised

Output feeds `enroll.build_gallery` unchanged — this only writes crops, it does
not embed anything. Prune by hand before building: 5-20 sharp, varied shots per
person beats a hundred near-identical frames.

Ported from the VideoRAG POC's `seed.py`, with class names run through
`safe_person_name` (an export is external input, and a class named `../x` would
otherwise write outside the faces directory) and every silent skip counted and
reported.
"""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from urllib.parse import unquote

import cv2
import numpy as np

from vision.enroll import IMAGE_EXTS, EnrolmentError, safe_person_name, _write_unique

DEFAULT_PAD = 0.25  # ArcFace likes some context around the face
DEFAULT_MIN_SIZE = 80  # px on the short side; below this a crop is unusable
LABEL_FIELDS = 5  # class_id cx cy w h

# Caps for an uploaded archive. Labels are tiny text files, so these are far
# above any real export while still refusing a decompression bomb.
MAX_ZIP_MEMBERS = 50_000
MAX_ZIP_BYTES = 100 * 1024 * 1024

# Label Studio's upload prefix: 6+ hex characters then `-` or `__`. Six is the
# floor so ordinary names like `2024-photo` are not mistaken for a hash.
_UPLOAD_HASH = re.compile(r"^[0-9a-fA-F]{6,}(?:__|-)")

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PersonCrops:
    """How many usable crops one class produced."""

    name: str
    crops: int

    @property
    def is_empty(self) -> bool:
        return self.crops == 0


@dataclass(frozen=True)
class SeedReport:
    """Outcome of one export conversion — everything the CLI needs to report."""

    people: tuple[PersonCrops, ...]
    out_dir: str
    skipped_small: int = 0
    skipped_malformed: int = 0
    missing_images: tuple[str, ...] = ()
    unreadable_images: tuple[str, ...] = ()

    @property
    def total_crops(self) -> int:
        return sum(p.crops for p in self.people)

    @property
    def empty_classes(self) -> tuple[str, ...]:
        """Labelled people who ended up with no usable crop at all.

        Worth naming: they are in the export, so you expect them enrolled, but
        they will be alerted on as strangers until they have photos.
        """
        return tuple(p.name for p in self.people if p.is_empty)


def load_classes(classes_file: Path) -> tuple[str, ...]:
    """Class names in id order, exactly as the export lists them."""
    lines = classes_file.read_text().splitlines()
    names = tuple(line.strip() for line in lines if line.strip())
    if not names:
        raise EnrolmentError(f"{classes_file} lists no classes.")
    return names


def padded_box(
    cx: float, cy: float, bw: float, bh: float, width: int, height: int, pad: float
) -> Box:
    """Normalised centre/size -> padded pixel corners, clamped to the image."""
    bw *= 1 + pad
    bh *= 1 + pad
    x1 = max(int((cx - bw / 2) * width), 0)
    y1 = max(int((cy - bh / 2) * height), 0)
    x2 = min(int((cx + bw / 2) * width), width)
    y2 = min(int((cy + bh / 2) * height), height)
    return x1, y1, x2, y2


def index_images(images_dir: Path) -> dict[str, Path]:
    """Map lower-cased stem -> image path, built once per run.

    Not `images_dir / f"{stem}{ext}"` per label: exports here are 96% `.JPG`
    while the extension list is lower-case, so probing by name only works on a
    case-insensitive filesystem. On Linux that silently loses almost every
    image and looks like "hardly anything was labelled".
    """
    return {
        path.stem.lower(): path
        for path in sorted(images_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTS
    }


def find_image(index: dict[str, Path], stem: str) -> Path | None:
    """The image for a label stem, allowing for how Label Studio renames files.

    Exports carry an upload hash (`a4dd4281__name`) and percent-encode spaces
    (`IMG_4716%202`), neither of which appears in the image filename.
    """
    for candidate in _stem_variants(stem):
        found = index.get(candidate.lower())
        if found is not None:
            return found
    return None


def _stem_variants(stem: str) -> tuple[str, ...]:
    """The stem as written, url-decoded, and with any upload hash stripped.

    The unmodified stem is always tried first, so stripping only ever *adds* a
    fallback — a real file that happens to look hash-prefixed still matches
    itself.
    """
    seen: list[str] = []
    for value in (stem, unquote(stem)):
        for candidate in (value, _strip_upload_hash(value)):
            if candidate and candidate not in seen:
                seen.append(candidate)
    return tuple(seen)


def _strip_upload_hash(stem: str) -> str:
    """Drop Label Studio's upload prefix.

    It uses two formats depending on how the file arrived:
    `a4dd4281__name` for one import path, `e88c5c36-name` for UI uploads.
    Neither prefix appears in the image filename on disk.
    """
    return _UPLOAD_HASH.sub("", stem, count=1)


def seed_from_export(
    export_dir: str | Path,
    faces_dir: str | Path,
    pad: float = DEFAULT_PAD,
    min_size: int = DEFAULT_MIN_SIZE,
    images_dir: str | Path | None = None,
) -> SeedReport:
    """Crop every labelled box into `faces_dir/<class name>/`.

    Args:
        export_dir: A Label Studio YOLO export — `classes.txt` and `labels/`.
        faces_dir: Where crops land, one folder per person.
        pad: Fraction to expand each box by; ArcFace likes some context.
        min_size: Skip crops below this many pixels on the short side.
        images_dir: Where the source images live. Defaults to `export_dir/images`.
            Point it elsewhere when the export carries labels only and the
            images sit in a local folder — Label Studio exports can be
            enormous, and the pixels are usually already on disk.

    Existing photos are kept — seeding twice adds crops rather than replacing
    them, same as enrolling twice through the Train dialog.
    """
    export = Path(export_dir)
    labels_dir, classes_file = export / "labels", export / "classes.txt"
    images = Path(images_dir) if images_dir is not None else export / "images"

    for path in (labels_dir, classes_file):
        if not path.exists():
            raise EnrolmentError(
                f"Not a Label Studio YOLO export — missing {path.name} in {export}."
            )
    if not images.is_dir():
        raise EnrolmentError(
            f"No images directory at {images}. The export carries labels only, "
            "so point this at the folder holding the source images."
        )
    if pad < 0:
        raise EnrolmentError(f"pad must not be negative, got {pad}")

    classes = load_classes(classes_file)
    # Sanitised up front so a hand-edited classes.txt cannot escape faces_dir.
    folders = tuple(safe_person_name(name) for name in classes)

    image_index = index_images(images)
    out_root = Path(faces_dir)
    counts = dict.fromkeys(folders, 0)
    skipped_small = skipped_malformed = 0
    missing: list[str] = []
    unreadable: list[str] = []

    for label_file in sorted(labels_dir.glob("*.txt")):
        image_path = find_image(image_index, label_file.stem)
        if image_path is None:
            missing.append(label_file.name)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            unreadable.append(image_path.name)
            continue

        height, width = image.shape[:2]
        for i, line in enumerate(label_file.read_text().splitlines()):
            parsed = _parse_line(line)
            if parsed is None:
                if line.strip():
                    skipped_malformed += 1
                continue

            cls_id, cx, cy, bw, bh = parsed
            x1, y1, x2, y2 = padded_box(cx, cy, bw, bh, width, height, pad)
            if min(x2 - x1, y2 - y1) < min_size:
                skipped_small += 1
                continue

            name = folders[cls_id] if cls_id < len(folders) else f"class_{cls_id}"
            person_dir = out_root / name
            person_dir.mkdir(parents=True, exist_ok=True)
            _write_crop(person_dir, f"{label_file.stem}_{i}.jpg", image[y1:y2, x1:x2])
            counts[name] = counts.get(name, 0) + 1

    return SeedReport(
        people=tuple(PersonCrops(name=n, crops=c) for n, c in counts.items()),
        out_dir=str(out_root),
        skipped_small=skipped_small,
        skipped_malformed=skipped_malformed,
        missing_images=tuple(missing),
        unreadable_images=tuple(unreadable),
    )


def seed_from_zip(
    archive: bytes | BinaryIO,
    faces_dir: str | Path,
    images_dir: str | Path,
    pad: float = DEFAULT_PAD,
    min_size: int = DEFAULT_MIN_SIZE,
) -> SeedReport:
    """Seed from an uploaded export archive holding labels but no images.

    The zip is treated as untrusted: only `classes.txt` and `labels/*.txt` are
    extracted, each is size-capped, and any entry that would resolve outside
    the temporary directory is refused. Images are never read from the archive
    — they come from `images_dir` on disk, which is the whole point of
    uploading a labels-only export.
    """
    with tempfile.TemporaryDirectory(prefix="ls-export-") as tmp:
        root = _extract_labels(archive, Path(tmp))
        return seed_from_export(root, faces_dir, pad, min_size, images_dir)


def _extract_labels(archive: bytes | BinaryIO, target: Path) -> Path:
    """Unpack just the label files, and return the export root inside `target`."""
    source = io.BytesIO(archive) if isinstance(archive, bytes) else archive
    try:
        with zipfile.ZipFile(source) as zf:
            members = [m for m in zf.infolist() if _is_label_member(m)]
            if not members:
                raise EnrolmentError(
                    "That zip has no classes.txt or labels/*.txt — export from "
                    "Label Studio as YOLO, not JSON."
                )
            if len(members) > MAX_ZIP_MEMBERS:
                raise EnrolmentError(
                    f"That zip holds {len(members)} label files, over the "
                    f"{MAX_ZIP_MEMBERS} limit."
                )
            total = sum(m.file_size for m in members)
            if total > MAX_ZIP_BYTES:
                raise EnrolmentError(
                    f"Label files total {total // 1_000_000}MB, over the "
                    f"{MAX_ZIP_BYTES // 1_000_000}MB limit."
                )
            for member in members:
                _extract_one(zf, member, target)
    except zipfile.BadZipFile as exc:
        raise EnrolmentError(f"Not a readable zip file: {exc}") from exc

    return _export_root(target)


def _is_label_member(member: zipfile.ZipInfo) -> bool:
    """Only the two things we need: classes.txt and the label text files."""
    if member.is_dir():
        return False
    name = PurePosixPath(member.filename).name
    return name == "classes.txt" or member.filename.lower().endswith(".txt")


def _extract_one(zf: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path) -> None:
    """Write one member, refusing anything that escapes `target` (zip slip)."""
    relative = PurePosixPath(member.filename)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise EnrolmentError(f"Refusing unsafe path in zip: {member.filename}")

    destination = (target / Path(*relative.parts)).resolve()
    if not destination.is_relative_to(target.resolve()):
        raise EnrolmentError(f"Refusing unsafe path in zip: {member.filename}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(destination, "wb") as dst:
        # Bounded read: file_size is what the header *claims*, not what the
        # archive necessarily delivers.
        shutil.copyfileobj(src, dst, length=64 * 1024)


def _export_root(target: Path) -> Path:
    """The directory holding classes.txt — exports often nest one level deep."""
    for classes_file in sorted(target.rglob("classes.txt")):
        return classes_file.parent
    raise EnrolmentError("That zip has no classes.txt at any level.")


def _parse_line(line: str) -> tuple[int, float, float, float, float] | None:
    """`<class_id> <cx> <cy> <w> <h>` -> parsed, or None if it is not that."""
    parts = line.split()
    if len(parts) != LABEL_FIELDS:
        return None
    try:
        return (int(parts[0]), *(float(v) for v in parts[1:]))  # type: ignore[return-value]
    except ValueError:
        return None


def _write_crop(person_dir: Path, filename: str, crop: np.ndarray) -> Path:
    """Encode in memory, then hand the bytes to enrol's non-overwriting writer."""
    ok, buf = cv2.imencode(".jpg", crop)
    if not ok:
        raise EnrolmentError(f"Failed to JPEG-encode a crop for {person_dir.name}.")
    return _write_unique(person_dir, filename, buf.tobytes())
