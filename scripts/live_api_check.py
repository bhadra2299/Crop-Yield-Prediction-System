"""Boot the real API on a scratch port and exercise every endpoint over HTTP.

This is an end-to-end check, not an in-process one: it starts uvicorn as a child
process, talks to it over a socket, and shuts it down. That catches wiring problems
(mounting, lifespan, serialisation, status codes) that an ASGITransport test cannot.

The child's stdout/stderr go to a file rather than a pipe. Redirecting a long-lived
server's output into an unread pipe fills the buffer and deadlocks the parent, which is
exactly what made an earlier version of this script appear to hang.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8077
BASE = f"http://127.0.0.1:{PORT}"

failures: list[str] = []
checks = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


def free_port(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict | list | str]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        status = exc.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw


def main() -> int:
    if not free_port(PORT):
        print(f"port {PORT} is already in use; refusing to test against an unknown service")
        return 2

    log = Path(tempfile.mkstemp(prefix="agri_api_", suffix=".log")[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("AGRI_LOG_LEVEL", "WARNING")

    with log.open("wb") as sink:
        proc = subprocess.Popen(
            [
                str(ROOT / ".venv" / "Scripts" / "python.exe"),
                "-m",
                "uvicorn",
                "agri.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
        )

    try:
        deadline = time.time() + 180
        status = None
        while time.time() < deadline:
            if proc.poll() is not None:
                print("server exited before becoming ready; log tail:")
                print(log.read_text(errors="replace")[-3000:])
                return 1
            try:
                status, _ = call("GET", "/v1/health")
                if status in (200, 503):
                    break
            except Exception:
                pass
            time.sleep(2)

        check("server became reachable", status in (200, 503), f"status={status}")
        if status not in (200, 503):
            print(log.read_text(errors="replace")[-3000:])
            return 1
        print(f"       (cold start took {180 - max(0, deadline - time.time()):.0f}s)", flush=True)

        # ---- system ------------------------------------------------------
        code, body = call("GET", "/v1/health")
        check("/v1/health is 200", code == 200, f"status={code}")
        check("/v1/health reports 10 modules", isinstance(body, dict) and len(body.get("modules", [])) == 10)
        ready = [m["name"] for m in body.get("modules", []) if m.get("ready")]
        print(f"       ready: {ready}", flush=True)
        for required in body.get("required_modules", []):
            check(f"required module {required} is ready", required in ready)

        code, body = call("GET", "/v1/version")
        check("/v1/version is 200", code == 200, str(body)[:80] if code != 200 else "")
        code, body = call("GET", "/v1/models")
        check("/v1/models is 200", code == 200, f"status={code}")
        code, body = call("GET", "/v1/ready/irrigation")
        check("/v1/ready/irrigation is 200", code == 200, f"status={code}")

        # ---- predictions -------------------------------------------------
        # Field names come from each module's request model, not from intuition: the
        # first draft of this script used crop/state/season and every call returned 500
        # because the payload did not match the contract.
        payloads: dict[str, dict] = {
            "crop-recommendation": {
                "N": 90,
                "P": 42,
                "K": 43,
                "temperature": 26.0,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 200.0,
            },
            "yield-prediction": {
                "Crop": "Rice",
                "Crop_Year": 2019,
                "Season": "Kharif",
                "State": "Punjab",
                "Area": 1000.0,
                "Annual_Rainfall": 800.0,
                "Fertilizer": 5000.0,
                "Pesticide": 200.0,
            },
            "fertilizer": {
                "Temparature": 26,
                "Humidity": 52,
                "Moisture": 38,
                "Soil_Type": "Loamy",
                "Crop_Type": "Rice",
                "Nitrogen": 37,
                "Potassium": 0,
                "Phosphorous": 0,
            },
            "soil-nutrient": {"temperature": 20.88, "humidity": 82, "ph": 6.5, "rainfall": 202.9},
            "risk": {
                "Crop": "Rice",
                "Season": "Kharif",
                "State": "Punjab",
                "Crop_Year": 2019,
                "Area": 1000.0,
                "Annual_Rainfall": 800.0,
                "Fertilizer": 5000.0,
                "Pesticide": 200.0,
            },
            "market-price": {"commodity": "Wheat", "state": "Madhya Pradesh", "horizon_days": 7},
            "irrigation": {
                "crop": "Rice",
                "latitude": 17.385,
                "elevation_m": 542,
                "soil": {
                    "field_capacity": 0.32,
                    "wilting_point": 0.14,
                    "root_depth_m": 0.6,
                    "depletion_fraction": 0.55,
                },
                "current_moisture": 0.18,
                "forecast_rain_mm": 0.0,
                "forecast": [
                    {"t_max": 34.0, "t_min": 24.0, "humidity_mean": 45.0, "rainfall": 0.0, "kc": 1.05},
                    {"t_max": 35.0, "t_min": 25.0, "humidity_mean": 40.0, "rainfall": 0.0, "kc": 1.05},
                ],
            },
        }
        for path, payload in payloads.items():
            code, body = call("POST", f"/v1/predict/{path}", payload)
            ok = code == 200 and isinstance(body, dict) and "value" in body
            check(f"POST /v1/predict/{path}", ok, f"status={code} " + ("" if ok else str(body)[:150]))
            if ok and path in ("market-price", "irrigation"):
                print(
                    f"       {path}: value={body.get('value')!r} confidence={body.get('confidence')}",
                    flush=True,
                )

        # weather needs an API key; without one it must fail as a configuration problem,
        # not as an internal error.
        code, body = call("POST", "/v1/predict/weather", {"latitude": 17.4, "longitude": 78.5})
        check(
            "POST /v1/predict/weather is honest without a key",
            code in (200, 503),
            f"status={code} " + str(body)[:120],
        )

        # The vision modules take an image path or base64, not an empty string. A bad
        # image must be a clean 4xx, not a 500.
        for path, sample_path in (
            ("soil-image", "data/raw/soil_images/loamy_soil/loamy_soil_0001.jpg"),
            ("disease", "data/raw/soil_images/loamy_soil/loamy_soil_0001.jpg"),
        ):
            code, body = call("POST", f"/v1/predict/{path}", {"image_path": sample_path, "top_k": 3})
            check(
                f"POST /v1/predict/{path} runs or declines cleanly", code in (200, 422, 503), f"status={code}"
            )
            if code == 200:
                print(f"       {path}: value={body.get('value')!r}", flush=True)

        code, body = call("POST", "/v1/predict/soil-image", {"image_base64": "not-an-image"})
        check(
            "POST /v1/predict/soil-image rejects bad input with 4xx",
            code in (400, 422, 503),
            f"status={code}",
        )

        # ---- orchestration ----------------------------------------------
        code, body = call(
            "POST",
            "/v1/predict/full-recommendation",
            {
                "crop_recommendation": {
                    "N": 90,
                    "P": 42,
                    "K": 43,
                    "temperature": 26.0,
                    "humidity": 82.0,
                    "ph": 6.5,
                    "rainfall": 200.0,
                },
                "irrigation": {
                    "crop": "Rice",
                    "latitude": 17.385,
                    "elevation_m": 542,
                    "soil": {
                        "field_capacity": 0.32,
                        "wilting_point": 0.14,
                        "root_depth_m": 0.6,
                        "depletion_fraction": 0.55,
                    },
                    "current_moisture": 0.18,
                    "forecast_rain_mm": 0.0,
                    "forecast": [
                        {"t_max": 34.0, "t_min": 24.0, "humidity_mean": 45.0, "rainfall": 0.0, "kc": 1.05}
                    ],
                },
                "risk": {
                    "Crop": "Rice",
                    "Season": "Kharif",
                    "State": "Punjab",
                    "Crop_Year": 2019,
                    "Area": 1000.0,
                    "Annual_Rainfall": 800.0,
                    "Fertilizer": 5000.0,
                    "Pesticide": 200.0,
                },
                "market_price": {"commodity": "Wheat", "state": "Madhya Pradesh", "horizon_days": 7},
            },
        )
        ok = code == 200 and isinstance(body, dict)
        check("POST /v1/predict/full-recommendation", ok, f"status={code} " + ("" if ok else str(body)[:150]))
        if ok:
            signals = body.get("signals") or {}
            print(
                f"       complete={body.get('complete')} signals={sorted(signals)} "
                f"missing={body.get('signals_missing')}",
                flush=True,
            )
            check("orchestration produced the anchor decision", "crop_recommendation" in signals)
            check("orchestration produced the irrigation decision", "irrigation" in signals)

        # A flat payload addresses no module and must be refused, not answered with an
        # advisory in which everything is missing.
        code, body = call("POST", "/v1/predict/full-recommendation", {"latitude": 17.4})
        check("flat orchestration payload is a 4xx", code in (400, 422), f"status={code}")

        # ---- ingest ------------------------------------------------------
        code, _ = call(
            "POST",
            "/v1/ingest/irrigation-event",
            {"field_id": "field-live-check", "occurred_on": "2026-10-28", "depth_mm": 25.0},
        )
        check("POST /v1/ingest/irrigation-event is 201", code == 201, f"status={code}")
        code, _ = call("GET", "/v1/ingest/irrigation-events")
        check("GET /v1/ingest/irrigation-events is 200", code == 200, f"status={code}")

        # ---- error contract ---------------------------------------------
        code, body = call("POST", "/v1/predict/irrigation", {"latitude": 999, "longitude": 0})
        check("invalid payload yields 422", code == 422, f"status={code}")
        code, body = call("GET", "/v1/health-not-a-route")
        check("unknown route yields 404", code == 404, f"status={code}")

        # ---- openapi -----------------------------------------------------
        code, body = call("GET", "/openapi.json")
        paths = len(body.get("paths", {})) if isinstance(body, dict) else 0
        check("openapi exposes every path", code == 200 and paths >= 29, f"paths={paths}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            log.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows keeps the handle briefly after the child exits; it is a temp file.

    print()
    print(f"{checks - len(failures)}/{checks} checks passed")
    print("LIVE API OK" if not failures else "FAILURES: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
