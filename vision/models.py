"""Loading the InsightFace `buffalo_l` ONNX models.

Separate from `detector` because enrolment needs the same two models without
the tracker, the gallery, or any per-stream state.

Models are loaded individually rather than through `FaceAnalysis`: that wrapper
runs every sub-model on every face of every frame, which measured 801ms for six
faces here.
"""

from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)

MODEL_PACK = "buffalo_l"
MODEL_ROOT = "~/.insightface"
DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"
POSE_FILE = "1k3d68.onnx"

DEFAULT_DET_SIZE = (640, 640)
DEFAULT_DET_THRESH = 0.5

# ctx_id must be >= 0: insightface forces CPUExecutionProvider when it is
# negative, silently undoing whatever `providers` we asked for.
CTX_ID = 0
APPLE_PROVIDERS = ("CoreMLExecutionProvider", "CPUExecutionProvider")
CPU_PROVIDERS = ("CPUExecutionProvider",)


def model_pack_dir(root: str = MODEL_ROOT) -> Path:
    """Resolve the model pack, downloading it on first run if need be."""
    path = Path(root).expanduser() / "models" / MODEL_PACK
    if not (path / DET_FILE).is_file():
        from insightface.utils import storage

        _log.info("downloading %s (~300MB, one time)", MODEL_PACK)
        storage.ensure_available("models", MODEL_PACK, root=str(Path(root).expanduser()))
    return path


def load_onnx_model(path: Path, providers: tuple[str, ...]):
    from insightface.model_zoo import get_model

    model = get_model(str(path), providers=list(providers))
    if model is None:
        raise RuntimeError(f"insightface could not load {path}")
    return model


def load_face_models(
    det_size: tuple[int, int] = DEFAULT_DET_SIZE,
    det_thresh: float = DEFAULT_DET_THRESH,
    cpu_only: bool = False,
    root: str = MODEL_ROOT,
):
    """The detector + recogniser pair enrolment needs, ready to use."""
    providers = CPU_PROVIDERS if cpu_only else APPLE_PROVIDERS
    pack = model_pack_dir(root)

    detector = load_onnx_model(pack / DET_FILE, providers)
    detector.prepare(CTX_ID, input_size=det_size, det_thresh=det_thresh)
    recognizer = load_onnx_model(pack / REC_FILE, providers)
    recognizer.prepare(CTX_ID)
    return detector, recognizer
