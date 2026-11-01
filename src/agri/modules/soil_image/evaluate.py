"""Evaluate the registered soil_image artefact.

Usage: python -m agri.modules.soil_image.evaluate [--version VERSION]

Rebuilds the held-out test partition from the split parameters recorded in the
manifest, scores the registered state dict on it, rewrites the metrics report and
exits non-zero when the configured gate fails. Nothing is trained here.

The reported macro_f1 measures separability of procedurally generated synthetic
textures. It is not a field-accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import sys

from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging

from .training import evaluate_soil_image

_log = get_logger("modules.soil_image.evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the soil_image classifier")
    parser.add_argument("--version", default=None, help="registry version; defaults to LATEST")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        result = evaluate_soil_image(version=args.version)
    except AgriError as exc:
        _log.error(
            "evaluation failed", extra={"module": "soil_image", "code": exc.code, "error": exc.message}
        )
        print(
            json.dumps({"ok": False, "module": "soil_image", "error": exc.to_dict()}, indent=2, default=str)
        )
        return 1
    print(json.dumps({"ok": True, **result}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
