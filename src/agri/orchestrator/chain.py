"""The full-recommendation chain.

One request in, one advisory out, assembled from several modules. The design rule
is that a partial answer beats no answer, as long as the caller can see which
parts are missing and why.

So the chain never raises on a module failure. It records the failure against that
signal, continues with the rest, and sets complete=False with the reasons. The one
exception is the anchor: without a crop and an irrigation decision there is no
advisory to give, and the response says that plainly.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agri.api.registry_cache import get_cache
from agri.common.errors import AgriError, DataValidationError
from agri.common.logging import get_logger

_log = get_logger("orchestrator.chain")

#: Modules worth running in parallel: they are independent and network-bound.
PARALLEL_MODULES = ("weather", "market_price", "risk")


def _run_one(name: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    """Run a single module, converting any failure into a reportable reason."""
    started = time.perf_counter()
    try:
        predictor = get_cache().get(name)
        if predictor is None:
            return name, None, get_cache().load_failure(name) or "module unavailable"
        request = predictor.request_model(**payload)
        prediction = predictor.predict(request)
        elapsed = (time.perf_counter() - started) * 1000
        _log.info("module completed", extra={"module": name, "ms": round(elapsed, 1)})
        return name, prediction.model_dump(), None
    except AgriError as exc:
        _log.warning("module failed", extra={"module": name, "code": exc.code, "error": exc.message})
        return name, None, f"{exc.code}: {exc.message}"
    except TypeError as exc:
        # Usually a request field the module does not accept; that is a caller error worth naming.
        return name, None, f"invalid_request: {exc}"
    except Exception as exc:
        _log.exception("module raised unexpectedly", extra={"module": name})
        return name, None, f"{type(exc).__name__}: {exc}"


def full_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    """Assemble the composite advisory.

    payload may contain a nested "crop_recommendation", "irrigation", "weather"
    and "risk" block, each forwarded verbatim to that module. Keys not addressed
    to a module are ignored rather than guessed at.
    """
    started = time.perf_counter()
    blocks = {key: payload.get(key) or {} for key in ("crop_recommendation", "irrigation", "weather", "risk")}
    blocks["market_price"] = payload.get("market_price") or {}

    addressed = sorted(key for key, block in blocks.items() if block)
    if not addressed:
        # A flat payload silently produced a 200 with complete=False and every signal
        # missing, which is indistinguishable from "all ten modules are broken". Name the
        # expected shape instead: this is a caller error, so it must be a 4xx.
        raise DataValidationError(
            "No module block was addressed. This endpoint takes one nested object per "
            "module, each forwarded verbatim to that module's own request contract.",
            context={
                "accepted_blocks": sorted(blocks),
                "example": {
                    "crop_recommendation": {
                        "N": 90,
                        "P": 42,
                        "K": 43,
                        "temperature": 26.0,
                        "humidity": 82.0,
                        "ph": 6.5,
                        "rainfall": 200.0,
                    },
                    "irrigation": {"crop": "Rice", "latitude": 17.385, "forecast": []},
                },
            },
        )

    signals: dict[str, Any] = {}
    problems: dict[str, str] = {}

    # Parallel: independent and dominated by network latency.
    with ThreadPoolExecutor(max_workers=len(PARALLEL_MODULES)) as pool:
        futures = {pool.submit(_run_one, name, blocks.get(name, {})): name for name in PARALLEL_MODULES}
        for future, _name in futures.items():
            module, result, error = future.result()
            if error:
                problems[module] = error
            else:
                signals[module] = result

    # Sequential: the crop recommendation gates which crops are worth advising on.
    module, result, error = _run_one("crop_recommendation", blocks.get("crop_recommendation", {}))
    if error:
        problems["crop_recommendation"] = error
    else:
        signals["crop_recommendation"] = result

    # Irrigation is the operational core, so it is always attempted.
    irrigation_payload = dict(blocks.get("irrigation", {}))
    # Carry yesterday's rainfall forward when the caller supplied weather but not irrigation.
    weather_signal = signals.get("weather") or {}
    if weather_signal and not irrigation_payload.get("forecast_rain_mm"):
        forecast = weather_signal.get("forecast") or []
        if forecast:
            irrigation_payload["forecast_rain_mm"] = float(
                sum(day.get("rainfall", 0.0) for day in forecast[:2])
            )
            irrigation_payload.setdefault(
                "forecast",
                [
                    {
                        "t_max": day.get("t_max", 30.0),
                        "t_min": day.get("t_min", 20.0),
                        "humidity_mean": day.get("humidity_mean"),
                        "rainfall": day.get("rainfall", 0.0),
                    }
                    for day in forecast[:7]
                ],
            )

    module, result, error = _run_one("irrigation", irrigation_payload)
    if error:
        problems["irrigation"] = error
    else:
        signals["irrigation"] = result

    anchor_present = "irrigation" in signals or "crop_recommendation" in signals
    complete = anchor_present and not problems

    advisory = _compose(signals, problems)
    elapsed_ms = (time.perf_counter() - started) * 1000

    _log.info(
        "full recommendation assembled",
        extra={
            "complete": complete,
            "signals": len(signals),
            "problems": len(problems),
            "ms": round(elapsed_ms, 1),
        },
    )
    return {
        "complete": complete,
        "signals_available": sorted(signals.keys()),
        "signals_missing": sorted(problems.keys()),
        "problems": problems,
        "signals": signals,
        "advisory": advisory,
        "latency_ms": round(elapsed_ms, 2),
        "note": (
            "Partial results are returned deliberately: each signal is independent, so a failed "
            "module is reported in problems rather than failing the whole request."
        ),
    }


def _compose(signals: dict[str, Any], problems: dict[str, str]) -> dict[str, Any]:
    """Turn raw signals into the sentences a user actually acts on."""
    actions: list[str] = []

    irrigation = signals.get("irrigation")
    if irrigation:
        # The irrigation Response carries the recommended depth in the envelope's value
        # field; there is no depth_mm key. Reading depth_mm raised KeyError and turned the
        # whole advisory into a 500 whenever irrigation was actually recommended.
        depth = irrigation.get("value") or 0.0
        if irrigation.get("should_irrigate"):
            actions.append(
                f"Irrigate {depth:.0f} mm ({irrigation.get('urgency', 'unknown')} urgency). "
                f"{(irrigation.get('reasons') or [''])[0]}"
            )
        else:
            actions.append(f"No irrigation needed. {(irrigation.get('reasons') or [''])[0]}")

    crop = signals.get("crop_recommendation")
    if crop:
        confidence = crop.get("confidence", "unknown")
        actions.append(
            f"{crop.get('value')} is the best-suited crop in the candidate set ({confidence} confidence)."
        )

    market = signals.get("market_price")
    if market:
        change = market.get("value", 0.0)
        signal = market.get("signal", "unknown")
        actions.append(
            f"{market.get('commodity', 'This commodity')} is forecast at {change:+.1%} over "
            f"{market.get('horizon_days', 7)} days (signal: {signal}). "
            f"Backtested error {market.get('backtested_mape')}% versus "
            f"{market.get('baseline_mape')}% for repeating the last price."
        )

    risk = signals.get("risk")
    if risk:
        actions.append(
            f"Field risk band: {risk.get('value')} ({risk.get('confidence', 'unknown')} confidence)."
        )

    weather = signals.get("weather")
    if weather:
        actions.append(
            f"Reference evapotranspiration today is {weather.get('value')} mm/day "
            f"from {weather.get('source')}."
        )

    caveats: list[str] = []
    for module, reason in sorted(problems.items()):
        caveats.append(f"{module} could not be included: {reason}")
    if irrigation is None:
        caveats.append(
            "No irrigation guidance is available for this request, so the advice below is incomplete."
        )

    return {"actions": actions, "caveats": caveats}
