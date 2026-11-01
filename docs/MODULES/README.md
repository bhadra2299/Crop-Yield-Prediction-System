# Module reference

One page per module. Every page states the question the module answers, the data and
algorithm behind it, its gate, what was actually measured, and — the section that
matters most — its honest limits.

| Module | Question | Gate | Measured |
|---|---|---|---|
| [crop_recommendation](crop_recommendation.md) | Which crop suits this soil? | macro_f1 >= 0.85 | 0.9932 |
| [yield_prediction](yield_prediction.md) | How much will this field yield? | r2 >= 0.85 | 0.9698 |
| [fertilizer](fertilizer.md) | Which blend, and how much? | macro_f1 >= 0.70 | 1.0 |
| [soil_nutrient](soil_nutrient.md) | How much soil nitrogen? | r2 >= 0.05 | 0.8192 |
| [risk](risk.md) | How risky is this crop and season? | recall_high_risk >= 0.50 | 0.5840 |
| [market_price](market_price.md) | What will the mandi pay? | median_ape <= 25 | 20.00 (baseline served) |
| [irrigation](irrigation.md) | Should I irrigate today? | 8 FAO-56 checks | 8/8 pass |
| [weather](weather.md) | What is the weather doing? | ET0 plausibility | passes |
| [soil_image](soil_image.md) | What soil type is this? | macro_f1 >= 0.85 | 0.9362 (synthetic) |
| [disease](disease.md) | What is wrong with this leaf? | none - inference only | no metric |

## Read the gates as claims, not scores

A gate is a promise: "this model clears this bar on data it was not trained on". The
measured value is what it actually achieved. Where those two are close, the gate is
doing work. Where they are far apart, the gate is loose — and that is stated on the
page rather than left for a reader to discover.

Two gates are deliberately low, and both pages explain why:

* **soil_nutrient** (r2 >= 0.05) — climate and pH carry weak signal for N, P and K. A
  high score there would indicate leakage, not skill.
* **risk** (recall >= 0.50 against a 0.33 chance rate) — the original 0.70 intent is
  recorded as unmet in [../DECISIONS.md](../DECISIONS.md) D7 because the data does not
  support it.

Three modules carry limitations serious enough to change how you should use them:

* **soil_image** trains on **synthetic** images. Its accuracy measures synthetic
  separability only.
* **disease** uses **third-party weights** never validated on regional varieties.
* **market_price** serves the **naive baseline**, because the trend model measurably lost
  to it. It reports no directional signal and says so, and the data has a 426-day hole.

## Common shape

Every module provides the same three things, so nothing downstream special-cases one:

```python
from agri.modules.<name> import Predictor, Request

predictor = Predictor()
predictor.load()                    # explicit; never at import time
response = predictor.predict(Request(...))
print(response.value, response.confidence, response.reasons)
```

and `python -m agri.modules.<name>.train` / `.evaluate` / `.predict`.