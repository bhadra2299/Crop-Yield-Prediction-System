"""Decode and validate an uploaded image before any tensor is built.

Images reach this module in one of two shapes:

  * image_base64 - base64-encoded JPEG or PNG bytes, what the HTTP layer forwards
    after reading an upload. A data-URL prefix such as "data:image/jpeg;base64,"
    is tolerated because browsers and notebooks produce it routinely.
  * image_path - a path on the serving host, used by the CLI sample and by offline
    scripts. It is not exposed by the public API.

Validation here is deliberate rather than defensive: a truncated upload, a PDF
renamed to .jpg, a 30000x30000 decompression bomb or a 60 MB TIFF all fail with a
DataValidationError naming the actual problem, instead of surfacing later as an
opaque torch shape error or an out-of-memory kill.

This helper is duplicated in agri.modules.disease.imaging because the write
boundary for this change is the module package. When common/ is editable again it
should be promoted to a single agri.common.vision helper.
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

_log = get_logger("modules.soil_image.imaging")

#: Refuse anything smaller than this; a 16x16 thumbnail carries no texture.
MIN_SIDE = 32
#: Refuse pathological dimensions before allocating the decoded bitmap.
MAX_SIDE = 8192
#: PIL modes we can convert to RGB. Anything else is rejected explicitly.
_CONVERTIBLE_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "I", "F"})


def decode_request_image(image_base64: str | None, image_path: str | None) -> Image.Image:
    """Return a validated, upright, RGB PIL image for one request.

    Raises DataValidationError with a caller-readable message and a context block
    naming the offending input. Callers supply exactly one of the two inputs; the
    pydantic request model already enforces that, and this function re-checks it
    so the helper is safe to reuse directly.
    """
    if bool(image_base64) == bool(image_path):
        raise DataValidationError(
            "Supply exactly one of image_base64 or image_path",
            context={"module": "soil_image"},
        )

    if image_base64:
        raw, source = _decode_base64(image_base64)
    else:
        raw, source = _read_path(str(image_path))

    return _validate_and_normalise(raw, source)


def _max_upload_bytes() -> int:
    """Upload ceiling from the shared settings, so HTTP and module agree."""
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
            context={"module": "soil_image", "error": str(exc)},
        ) from exc
    if not raw:
        raise DataValidationError(
            "image_base64 decoded to zero bytes",
            context={"module": "soil_image"},
        )
    limit = _max_upload_bytes()
    if len(raw) > limit:
        raise DataValidationError(
            f"Image is {len(raw)} bytes, above the {limit} byte upload limit",
            context={"module": "soil_image", "bytes": len(raw), "limit": limit},
        )
    return raw, "image_base64"


def _read_path(raw_path: str) -> tuple[bytes, str]:
    candidate = Path(raw_path).expanduser()
    resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    if not resolved.is_file():
        raise DataValidationError(
            "image_path does not point at a readable file",
            context={"module": "soil_image", "path": str(resolved)},
        )
    limit = _max_upload_bytes()
    size = resolved.stat().st_size
    if size == 0:
        raise DataValidationError(
            "image_path points at an empty file",
            context={"module": "soil_image", "path": str(resolved)},
        )
    if size > limit:
        raise DataValidationError(
            f"Image is {size} bytes, above the {limit} byte upload limit",
            context={"module": "soil_image", "path": str(resolved), "limit": limit},
        )
    return resolved.read_bytes(), str(resolved)


def _validate_and_normalise(raw: bytes, source: str) -> Image.Image:
    """Open, bound-check, orient and convert one image to RGB."""
    Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE  # explicit cap instead of PIL's warning-only default
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            width, height = opened.size
            mode = opened.mode
            if width < MIN_SIDE or height < MIN_SIDE:
                raise DataValidationError(
                    f"Image is {width}x{height}, smaller than the {MIN_SIDE}x{MIN_SIDE} minimum",
                    context={"module": "soil_image", "source": source, "width": width, "height": height},
                )
            if width > MAX_SIDE or height > MAX_SIDE:
                raise DataValidationError(
                    f"Image is {width}x{height}, larger than the {MAX_SIDE}x{MAX_SIDE} maximum",
                    context={"module": "soil_image", "source": source, "width": width, "height": height},
                )
            if mode not in _CONVERTIBLE_MODES:
                raise DataValidationError(
                    f"Unsupported image mode '{mode}'; expected one of {sorted(_CONVERTIBLE_MODES)}",
                    context={"module": "soil_image", "source": source, "mode": mode},
                )
            # Phone cameras store orientation in EXIF; without this a portrait
            # photo is classified sideways.
            oriented = ImageOps.exif_transpose(opened)
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DataValidationError(
            "Uploaded bytes are not a decodable image",
            context={"module": "soil_image", "source": source, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
