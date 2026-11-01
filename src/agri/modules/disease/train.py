"""disease has no training step.

The module serves third-party PlantVillage weights. Training or fine-tuning them
here would either (a) require a labelled regional dataset this repository does not
have, or (b) quietly reproduce the upstream training run and call the result ours.
Neither is honest, so the command is kept for contract uniformity and does what is
actually meaningful: it verifies that the weights resolve and the inference stack
loads, and exits non-zero when they do not.

Usage: python -m agri.modules.disease.train [--no-download]

--no-download forces a cache-only check, which is what an air-gapped deployment
should run in CI before shipping the image.
"""

from __future__ import annotations

import argparse
import json
import sys

from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging

from .predict import Predictor

_log = get_logger("modules.disease.train")


def verify(allow_download: bool = True) -> dict[str, object]:
    """Load the predictor and report whether inference is actually available."""
    predictor = Predictor()
    # allow_download is a config value; the CLI can only narrow it to cache-only.
    if not allow_download:
        params = dict(predictor.config.model.params)
        params["allow_download"] = False
        predictor.config.model.params = params
    predictor.load()
    health = predictor.health()
    payload: dict[str, object] = {
        "module": "disease",
        "kind": "inference_availability_check",
        "ready": predictor.ready,
        "detail": health.detail,
        "version": predictor.version,
        "trained_here": False,
        "note": (
            "disease is inference-only. This command verifies that third-party weights "
            "resolve; it does not train anything and reports no accuracy."
        ),
    }
    if predictor.ready:
        payload["weights_source"] = getattr(predictor, "weights_source", {})
        payload["num_labels"] = len(getattr(predictor, "id2label", {}))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify disease inference availability")
    parser.add_argument(
        "--no-download", action="store_true", help="check the cache only; never contact the Hub"
    )
    args = parser.parse_args(argv)
    setup_logging()
    try:
        payload = verify(allow_download=not args.no_download)
    except AgriError as exc:
        _log.error(
            "disease availability check failed",
            extra={"module": "disease", "code": exc.code, "error": exc.message},
        )
        print(json.dumps({"ok": False, "module": "disease", "error": exc.to_dict()}, indent=2, default=str))
        return 1
    print(json.dumps({"ok": bool(payload["ready"]), **payload}, indent=2, default=str))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
