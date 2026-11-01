# soil_nutrient

**Question.** How much available **nitrogen** does this soil carry, given its climate and pH?

**Scope note, stated up front.** This module predicts **one** nutrient, not three. An
earlier draft of this page and of the module description claimed N, P and K. That was
wrong: `configs/soil_nutrient.yaml` targets `N` and drops `P` and `K`. The response
envelope even declared a dict, which the regression head never produced, so calling the
predictor returned a validation error. It now declares `float` and returns nitrogen.

| | |
|---|---|
| Task | single-target regression |
| Target | `N` (kg/ha) |
| Algorithm | RandomForest, 400 trees, depth 12 |
| Features | `temperature`, `humidity`, `ph`, `rainfall` |
| Dropped | `P`, `K`, `label` |
| Gate | `r2 >= 0.05`, must beat the median baseline |
| Measured | **R2 = 0.8192**, RMSE 15.03, MAE 11.67 |

---

## Why the gate is 0.05 while the result is 0.82

The low floor looks like a typo. It is deliberate.

The obvious risk is **leakage**, and the file this comes from makes it easy to commit:
`crop_recommendation.csv` holds `N`, `P`, `K` and `label` side by side. The `label` is
the crop, and each crop has a characteristic nutrient profile - rice sits near N=80,
chickpea near N=40. So any feature that encodes the crop also encodes the nutrient.

The measured 0.82 is explainable without leakage: the dataset is 22 crops of 100 rows
each, and `humidity` plus `rainfall` separate those crops cleanly. The feature set
predicts *which crop this looks like*, and the nutrient follows from the crop. That is a
real pattern in the data, but it is **not** inference from soil chemistry, and it will
not transfer to a field growing a crop this dataset has never seen.

`P` and `K` are deliberately **not** modelled. Doing so would require separate targets
and would make the crop-label entanglement worse, not better. Predicting them from four
climate columns would be fitting the label rather than the soil, which is exactly what
`detect_leakage()` exists to refuse.

The floor of 0.05 says: a regression toward zero here is not automatically a bug, and a
high score deserves an investigation rather than applause. `must_beat_baseline: median`
remains the real requirement, because a model that cannot beat predicting the median is
not a model.

---

## How to use it honestly

Treat the output as **a prior for a crop this dataset covers**, never as a substitute for
a soil test. A lab measurement of available N is cheap; acting on a 15 kg/ha RMSE
estimate when the true field carries a different crop is not.

If you need P and K, measure them.

---

## Reproduce

```bash
agri train --module soil_nutrient
agri evaluate --module soil_nutrient
```

```python
from agri.modules.soil_nutrient import Predictor, Request

predictor = Predictor()
predictor.load()
response = predictor.predict(
    Request(temperature=20.88, humidity=82.0, ph=6.5, rainfall=202.9)
)
print(response.value)        # nitrogen, kg/ha
print(response.reasons)      # why, and what the uncertainty is
```
