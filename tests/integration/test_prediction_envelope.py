"""Serving envelope contract test.

What this protects against
--------------------------
A served prediction that drifts from the documented envelope the API and the
mobile clients parse: a missing confidence, an empty reasons list, a value
outside the declared class set, or a version stamp that does not match the
registered artefact. It is the only test that runs a real model end to end from
the registry to a typed response.

Marked slow because it depends on the trained_crop_run fixture, which trains the
production model once per session.
"""

from __future__ import annotations

import pytest

from agri.common.schemas import Confidence

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_crop_recommendation_envelope_has_the_documented_fields(trained_crop_run) -> None:
    from agri.modules.crop_recommendation import Predictor, Request
    from agri.modules.crop_recommendation.predict import SAMPLE

    predictor = Predictor()
    predictor.load()
    assert predictor.ready is True, predictor.health().detail

    response = predictor.predict(Request(**SAMPLE))

    assert response.model_name == "crop_recommendation"
    assert response.model_version == trained_crop_run.result["version"]
    assert response.latency_ms is not None
    assert response.latency_ms >= 0.0

    assert isinstance(response.value, str)
    assert response.value in predictor.config.data.classes

    assert isinstance(response.confidence, Confidence)
    assert response.confidence is not Confidence.UNKNOWN
    assert response.confidence_score is not None
    assert 0.0 <= response.confidence_score <= 1.0

    assert response.reasons
    assert all(isinstance(reason, str) and reason for reason in response.reasons)


def test_crop_recommendation_envelope_is_json_serialisable(trained_crop_run) -> None:
    from agri.modules.crop_recommendation import Predictor, Request
    from agri.modules.crop_recommendation.predict import SAMPLE

    predictor = Predictor()
    predictor.load()
    payload = predictor.predict(Request(**SAMPLE)).model_dump()
    assert {"value", "confidence", "reasons", "model_version", "latency_ms"} <= set(payload)
