"""Decode and validate an uploaded leaf image before inference.

This is a focused mirror of the soil_image helper. It is duplicated on purpose:
the two image modules are separate deployable signals, and neither may break
because the other package has an import problem. When common/ is editable again,
both should collapse into one agri.common.vision module.

The same two input shapes are supported: base64 bytes in the request body (what the
HTTP layer forwards) or a server-side path for CLI and offline scripts. A bad
upload fails here with a DataValidationError naming the real problem, rather than
later as an opaque tensor error.
"""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from agri.common.config import get_settings
from agri.common.errors import DataValidationError
from agri.common.logging import get_logger
from agri.common.paths import PROJECT_ROOT

_log = get_logger("modules.disease.imaging")

MIN_SIDE = 32
MAX_SIDE = 8192
_CONVERTIBLE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"})


def decode_request_image(image_base64: str | None, image_path: str | None) -> Image.Image:
    """Return a validated, upright, RGB PIL image for one request."""
    if bool(image_base64) == bool(image_path):
        raise DataValidationError(
            "Supply exactly one of image_base64 or image_path",
            context={"module": "disease"},
        )
    if image_base64:
        raw, source = _decode_base64(image_base64)
    else:
        raw, source = _read_path(str(image_path))
    return _validate_and_normalise(raw, source)


def _max_upload_bytes() -> int:
    try:
        return int(get_settings().max_upload_bytes)
    except Exception:  # pragma: no cover - settings always load in practice
        return 5 * 1024 * 1024


def _decode_base64(payload: str) -> tuple[bytes, str]:
    text = payload.strip()
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DataValidationError(
            "image_base64 is not valid base64",
            context={"module": "disease", "error": str(exc)},
        ) from exc
    if not raw:
        raise DataValidationError("image_base64 decoded to zero bytes", context={"module": "disease"})
    limit = _max_upload_bytes()
    if len(raw) > limit:
        raise DataValidationError(
            f"Image is {len(raw)} bytes, above the {limit} byte upload limit",
            context={"module": "disease", "bytes": len(raw), "limit": limit},
        )
    return raw, "image_base64"


def _read_path(raw_path: str) -> tuple[bytes, str]:
    candidate = Path(raw_path).expanduser()
    resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not resolved.is_file():
        raise DataValidationError(
            "image_path does not point at a readable file",
            context={"module": "disease", "path": str(resolved)},
        )
    limit = _max_upload_bytes()
    size = resolved.stat().st_size
    if size == 0:
        raise DataValidationError(
            "image_path points at an empty file",
            context={"module": "disease", "path": str(resolved)},
        )
    if size > limit:
        raise DataValidationError(
            f"Image is {size} bytes, above the {limit} byte upload limit",
            context={"module": "disease", "path": str(resolved), "limit": limit},
        )
    return resolved.read_bytes(), str(resolved)


def _validate_and_normalise(raw: bytes, source: str) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            width, height = opened.size
            mode = opened.mode
            if width < MIN_SIDE or height < MIN_SIDE:
                raise DataValidationError(
                    f"Image is {width}x{height}, smaller than the {MIN_SIDE}x{MIN_SIDE} minimum",
                    context={"module": "disease", "source": source, "width": width, "height": height},
                )
            if width > MAX_SIDE or height > MAX_SIDE:
                raise DataValidationError(
                    f"Image is {width}x{height}, larger than the {MAX_SIDE}x{MAX_SIDE} maximum",
                    context={"module": "disease", "source": source, "width": width, "height": height},
                )
            if mode not in _CONVERTIBLE_MODES:
                raise DataValidationError(
                    f"Unsupported image mode '{mode}'; expected one of {sorted(_CONVERTIBLE_MODES)}",
                    context={"module": "disease", "source": source, "mode": mode},
                )
            oriented = ImageOps.exif_transpose(opened)
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DataValidationError(
            "Uploaded bytes are not a decodable image",
            context={"module": "disease", "source": source, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
