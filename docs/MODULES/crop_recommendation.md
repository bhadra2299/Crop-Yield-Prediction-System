# crop_recommendation

**Question.** Given soil chemistry and climate, which crop is best suited?

| | |
|---|---|
| Task | multiclass classification |
| Algorithm | gradient boosting (configurable) |
| Dataset | `data/raw/crop_recommendation/crop_recommendation.csv`, 2,200 rows, 22 classes |
| Features | N, P, K, temperature, humidity, ph, rainfall |
| Split | random, stratified |
| Gate | `macro_f1 >= 0.85` |
| Measured | **macro_f1 = 0.9932** |

## What it does

Each row is a set of soil and climate measurements labelled with the crop that
thrives in them. The model learns the decision boundaries between 22 crop profiles.

The response returns the recommended crop as `value`, the full ranked list as
`alternatives`, and the top contributing features as `reasons` - so a user can see
*why* rice was recommended rather than just being told to grow rice.

## Why macro-F1 and not accuracy

With 22 classes of roughly equal size, accuracy is close to macro-F1, but macro-F1
weights every crop equally. A model that nails the abundant crops and fails the rare
ones should not score well, because a farmer growing an uncommon crop is exactly the
person who needs the recommendation.

## Confidence

`confidence_score` is the calibrated probability of the predicted class.
`high` requires >= 0.90, `medium` >= 0.65. A 0.55 top-1 with a 0.40 runner-up
correctly reports as low confidence.

## Honest limits

* **2,200 rows is small.** 100 rows per crop. The near-perfect F1 reflects a clean
  synthetic-style dataset; real field data is noisier and this number will fall.
* **22 crops is a fixed vocabulary.** It cannot recommend a crop absent from
  training, and does not know regional viability, market demand, water rights or
  rotation history.
* **No soil texture.** Only chemistry. Two soils with identical N-P-K but different
  structure are indistinguishable to this model.
* The recommendation is a **ranking over candidates**, not a statement that the crop
  will succeed.

## Usage

```bash
agri train --module crop_recommendation
agri evaluate --module crop_recommendation
python -m agri.modules.crop_recommendation.predict
```

```bash
curl -X POST localhost:8000/v1/predict/crop-recommendation \
  -H 'Content-Type: application/json' \
  -d '{"N":90,"P":42,"K":43,"temperature":20.88,"humidity":82,"ph":6.5,"rainfall":202.9}'
```
