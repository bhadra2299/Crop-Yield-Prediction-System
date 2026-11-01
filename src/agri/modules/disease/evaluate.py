"""Evaluate the disease module.

Usage: python -m agri.modules.disease.evaluate

There is no labelled regional dataset in this repository, so there is nothing to
compute an accuracy against, and the PlantVillage score the upstream authors
published is not a property of this deployment. Inventing a number here would be
worse than reporting none.

What this command does instead:
  * loads the weights (cache first, Hub second),
  * proves the whole path runs on one bundled image (a smoke test, not a metric),
  * writes the metrics report with accuracy_claimed=false and the limitations,
  * exits non-zero only when inference is unavailable at all.

The gate for this module is inference_available, which is the only claim it makes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agri.common.config import load_module_config
from agri.common.errors import AgriError
from agri.common.logging import get_logger, setup_logging
from agri.common.metrics import evaluate_gate
from agri.common.paths import PROJECT_ROOT
from agri.common.tabular import write_metrics_report

from .predict import NO_ACCURACY_CLAIM, SAMPLE, THIRD_PARTY_LIMITATION, Predictor

_log = get_logger("modules.disease.evaluate")

MODULE_NAME = "disease"


def _smoke_test(predictor: Predictor) -> dict[str, Any]:
    """Run one real prediction to prove the path works. Not a metric."""
    record: dict[str, Any] = {"attempted": False}
    path_value = SAMPLE.get("image_path")
    resolved = PROJECT_ROOT / str(path_value) if path_value else None
    if resolved is None or not resolved.is_file():
        record["reason"] = "no bundled smoke-test image is available in this checkout"
        return record
    record["attempted"] = True
    record["input"] = str(resolved.relative_to(PROJECT_ROOT))
    try:
        request = predictor.request_model(**SAMPLE)
        prediction = predictor.predict(request)
        record["ok"] = True
        record["predicted_label"] = prediction.value
        record["confidence_score"] = prediction.confidence_score
        record["note"] = (
            "The bundled image is a soil texture, not a leaf. This proves the inference "
            "path runs; it says nothing about disease detection quality."
        )
    except AgriError as exc:
        record["ok"] = False
        record["error"] = exc.to_dict()
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the disease module")
    parser.add_argument(
        "--no-download", action="store_true", help="evaluate from the local weight cache only"
    )
    args = parser.parse_args(argv)
    setup_logging()
    config = load_module_config(MODULE_NAME)

    if args.no_download:
        params = dict(config.model.params)
        params["allow_download"] = False
        config.model.params = params

    predictor = Predictor(config=config)
    predictor.load()

    if not predictor.ready:
        payload = {
            "ok": False,
            "module": MODULE_NAME,
            "inference_available": False,
            "detail": predictor.health().detail,
            "limitations": [THIRD_PARTY_LIMITATION, NO_ACCURACY_CLAIM],
        }
        print(json.dumps(payload, indent=2, default=str))
        return 1

    smoke = _smoke_test(predictor)
    metrics = {"inference_available": 1.0 if smoke.get("ok") else 0.0}
    gate = evaluate_gate(metrics, config.gates)
    detail: dict[str, Any] = {
        "num_labels": len(getattr(predictor, "id2label", {})),
        "labels": list(getattr(predictor, "id2label", {}).values())[:40],
        "smoke_test": smoke,
    }
    write_metrics_report(
        config,
        metrics,
        detail,
        gate.to_dict(),
        manifest={
            "weights_source": getattr(predictor, "weights_source", {}),
            "evaluated_from_registry": False,
            "third_party_weights": True,
        },
        extra={
            "accuracy_claimed": False,
            "accuracy_note": NO_ACCURACY_CLAIM,
            "third_party_limitation": THIRD_PARTY_LIMITATION,
            "metric_scope": "inference availability only",
        },
    )

    payload = {
        "ok": bool(smoke.get("ok")),
        "module": MODULE_NAME,
        "inference_available": True,
        "weights_source": getattr(predictor, "weights_source", {}),
        "smoke_test": smoke,
        "metrics": metrics,
        "gate": gate.to_dict(),
        "accuracy_claimed": False,
        "accuracy_note": NO_ACCURACY_CLAIM,
        "limitations": [THIRD_PARTY_LIMITATION, NO_ACCURACY_CLAIM],
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0 if smoke.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
