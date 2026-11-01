"""Irrigation API contract test.

What this protects against
--------------------------
A physics module whose served figures stop matching the FAO-56 arithmetic it
documents. The module needs no trained artefact, so this runs as a plain
integration test: it builds a real Request, runs the real predictor and checks
the reference total available water for the default loam, which is the number
the agronomy documentation quotes.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


def test_default_soil_reports_taw_and_the_arithmetic_steps() -> None:
    from agri.modules.irrigation import Predictor, Request

    predictor = Predictor()
    predictor.load()
    assert predictor.ready is True, predictor.health().detail

    response = predictor.predict(Request(depletion_mm=50.0))

    assert response.steps, "the recommendation must expose its arithmetic"
    assert response.taw_mm == pytest.approx(108.0, rel=1e-9)
    assert response.raw_mm == pytest.approx(59.4, rel=1e-9)
    assert response.depletion_mm == pytest.approx(50.0)
    assert response.should_irrigate is False
    assert response.value == pytest.approx(0.0)
    assert response.model_name == "irrigation"


def test_depletion_beyond_raw_irrigates_and_caps_the_depth() -> None:
    from agri.modules.irrigation import Predictor, Request

    predictor = Predictor()
    predictor.load()

    response = predictor.predict(Request(depletion_mm=200.0))

    assert response.should_irrigate is True
    assert response.value == pytest.approx(40.0)
    assert response.urgency == "now"
    assert response.steps
