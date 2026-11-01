# irrigation

**Question.** Should I irrigate today, and how many millimetres?

| | |
|---|---|
| Task | deterministic computation (FAO-56) |
| Algorithm | Penman-Monteith reference ET + dual-coefficient water balance |
| Artefact | **none** (`requires_artifact = False`) |
| Gate | 8 published FAO-56 reference checks must pass |
| Measured | **8/8 pass** |

This module does not use machine learning, and that is the point. The irrigation
decision is a physical water balance, and the equations are published. Replacing them
with a model fitted on three seasons of demo data would make the answer worse and
unauditable.

## What it computes

```
ET0     reference evapotranspiration          FAO-56 Eq. 6 (Penman-Monteith)
Kc      crop coefficient, interpolated        FAO-56 Table 12, stage-weighted
ETc     crop evapotranspiration = Kc * ET0
TAW     total available water = 1000 * (fc - wp) * Zr
RAW     readily available water = p * TAW     p from FAO-56 Table 22
Ks      water stress coefficient              FAO-56 Eq. 84
Dr      root-zone depletion, running balance
```

The response returns `should_irrigate`, `depth_mm`, `urgency`, the full `steps`
arithmetic, and a projection of depletion over the forecast horizon - which may
**upgrade the urgency to "soon"** when the projected depletion crosses RAW even
though it has not yet.

## Why there is no accuracy percentage

There is no field-truth series in this repository: no soil-moisture sensor readings
paired with known applied depths. Without that, any "irrigation accuracy: 94%" figure
would be fabricated.

What *is* verifiable is that the implementation matches the published equations. So
that is what is tested:

| Check | Expected | Source |
|---|---|---|
| `e0(20)` | 2.338 kPa | FAO-56 Table 2.3 |
| `e0(30)` | 4.243 kPa | FAO-56 Table 2.3 |
| `delta(20)` | 0.144 kPa/degC | FAO-56 Table 2.4 |
| `P(0)` | 101.3 kPa | FAO-56 Eq. 7 |
| `P(1000)` | 89.9 kPa | FAO-56 Eq. 7 |
| `gamma(0)` | 0.0674 | FAO-56 Eq. 8 |
| `gamma(1000)` | 0.0598 | FAO-56 Eq. 8 |
| `Ra(equator, day 81)` | 37.6 MJ/m2/day | FAO-56 Eq. 21 |
| `N(equator, day 81)` | 12.0 h | FAO-56 Eq. 34 |
| `TAW(0.32, 0.14, 0.6 m)` | 108.0 mm | FAO-56 Eq. 82 |
| `RAW(0.32, 0.14, 0.6 m, p=0.55)` | 59.4 mm | FAO-56 Eq. 83 |

**What has not been verified:** exact end-to-end reproduction of FAO-56 Chapter 4
Example 18. That needs the example's full daily weather record, which is not in this
repository. The tests check the component equations against published values. Saying
"FAO-56 verified" without that distinction would overstate it.

## Confidence means input quality

`confidence` here reflects **how good the inputs were**, not how sure the model is -
there is no model. Measured soil moisture gives high confidence; an estimate derived
from days-since-last-rain gives low confidence. The `reasons` field names which
occurred.

## Honest limits

* **No field validation.** See above. This is a calculator, not a calibrated predictor.
* **Single-layer soil.** One root zone with one field capacity. Real profiles are
  layered.
* **Kc from tables.** `Kc` is stage-interpolated from published values, not
  crop-calibrated locally, and assumes a standard crop calendar.
* **No capillary rise, no deep percolation losses, no runoff modelling.** The balance
  is inflow minus ET.
* **Depletion drifts without irrigation events.** The running balance is only as good
  as the events recorded. Use `POST /v1/ingest/irrigation-event` to keep it honest.
* **Events go to a JSONL file.** Correct for one node; not for a cluster.

## Inputs

`current_moisture` (volumetric fraction) is converted to depletion internally, so a
sensor reading can be supplied directly instead of a modelled balance.

`forecast` accepts up to 7 days of `t_max`, `t_min`, `humidity_mean`, `rainfall`
to project the balance forward.

## Usage

```bash
python -m agri.modules.irrigation.train     # runs the 8 reference checks
python -m agri.modules.irrigation.predict
```

```bash
curl -X POST localhost:8000/v1/predict/irrigation \
  -H 'Content-Type: application/json' \
  -d '{"crop":"Rice","latitude":17.385,"current_moisture":0.18,"forecast_rain_mm":0}'
```
