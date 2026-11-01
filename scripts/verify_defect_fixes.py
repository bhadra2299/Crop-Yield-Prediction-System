"""Directly verify the defects tests-writer reported, without the pytest suite."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from agri.features import CategoryIndexEncoder, pipeline_hash
from agri.features.physics import (
    DailyWeather,
    SoilProfile,
    compute_water_balance,
)

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        failures.append(name)


# --- defect 1: pipeline_hash raised NameError -------------------------------
class _Step:
    def __init__(self, params):
        self.params = params

    def to_params(self):
        return self.params


class _Pipeline:
    def __init__(self, steps):
        self.steps = steps


sample = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
h1 = pipeline_hash(_Pipeline([_Step({"k": 1})]), sample)
h2 = pipeline_hash(_Pipeline([_Step({"k": 1})]), sample)
h3 = pipeline_hash(_Pipeline([_Step({"k": 2})]), sample)
check("pipeline_hash computes", isinstance(h1, str) and len(h1) == 64, h1[:12])
check("pipeline_hash is stable", h1 == h2)
check("pipeline_hash changes with params", h1 != h3)
check("pipeline_hash handles empty steps", len(pipeline_hash(_Pipeline([]), sample)) == 64)

# --- defect 2: CategoryIndexEncoder crashed on NaN under pandas 3.x ---------
frame = pd.DataFrame({"Soil": ["Loam", None, "Clay", np.nan, "loam"]})
enc = CategoryIndexEncoder.fit(frame, ["Soil"])
check(
    "fit tolerates NaN",
    sorted(enc.vocabularies["Soil"]) == ["Clay", "Loam", "loam"],
    str(enc.vocabularies["Soil"]),
)
built = enc.build(frame)
check(
    "build maps unknown/NaN to -1",
    int(built["Soil"].iloc[1]) == -1 and int(built["Soil"].iloc[3]) == -1,
    built["Soil"].tolist(),
)
check("build maps known values", int(built["Soil"].iloc[0]) != -1)
check("build dtype is int32", str(built["Soil"].dtype) == "int32", str(built["Soil"].dtype))

# --- defect 3: water balance closure double-counted deep percolation --------
# A wet series that must generate percolation: heavy rain on a dry-ish soil.
soil = SoilProfile(field_capacity=0.32, wilting_point=0.14, root_depth=0.6, initial_depletion=5.0)
days = 10
weather = [
    DailyWeather(t_max=22.0, t_min=10.0, humidity_mean=60.0, solar_radiation=15.0, rainfall=60.0)
    for _ in range(days)
]
result = compute_water_balance(weather, soil, 17.4, date(2025, 6, 1))
percolated = result.total_deep_percolation
check("the wet series actually percolates", percolated > 0.0, f"deep_percolation={percolated:.3f} mm")
check(
    "closure identity holds with percolation",
    abs(result.closure_error_mm) < 1e-6,
    f"closure_error={result.closure_error_mm:.3e} mm",
)

# A dry series that should not percolate at all.
dry_weather = [
    DailyWeather(t_max=34.0, t_min=24.0, humidity_mean=30.0, solar_radiation=24.0, rainfall=0.0)
    for _ in range(days)
]
dry = compute_water_balance(dry_weather, soil, 17.4, date(2025, 6, 1))
check(
    "dry series does not percolate",
    dry.total_deep_percolation == 0.0,
    f"deep_percolation={dry.total_deep_percolation:.3f}",
)
check(
    "dry series closure holds", abs(dry.closure_error_mm) < 1e-6, f"closure_error={dry.closure_error_mm:.3e}"
)

# --- defect 4: weather predictor importable ---------------------------------
try:
    import agri.modules.weather.predict as wp

    check("weather.predict imports", callable(getattr(wp, "Predictor", None)))
except Exception as exc:
    check("weather.predict imports", False, f"{type(exc).__name__}: {exc}")

try:
    import agri.ingestion as ing

    check("ingestion imports", hasattr(ing, "Puller"))
except Exception as exc:
    check("ingestion imports", False, f"{type(exc).__name__}: {exc}")

print()
print(f"{'ALL FIXED' if not failures else 'STILL FAILING: ' + ', '.join(failures)}")
raise SystemExit(1 if failures else 0)
