# fertilizer

**Question.** Which fertilizer blend should be applied?

| | |
|---|---|
| Task | multiclass classification |
| Algorithm | gradient boosting |
| Dataset | `data/raw/fertilizer/fertilizer_prediction.csv`, **99 rows** |
| Features | Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous |
| Split | random, stratified |
| Gate | `macro_f1 >= 0.70` |
| Measured | **macro_f1 = 1.0** |

## Read the 99 rows before the 1.0

An F1 of 1.0 normally means a bug or leakage. Here it means the dataset is tiny and
clean: 99 rows, well-separated classes, few features. The model has effectively
memorised a lookup table, and a lookup table on well-separated data scores perfectly.

**This is the smallest dataset in the project.** Its gate is set at 0.70 rather than
0.95 partly because a small test split makes F1 noisy - a single misclassification
moves macro-F1 by several points.

## Confidence thresholds are deliberately conservative

`confidence_high: 0.90`, `confidence_medium: 0.70`.

The intent is that most real answers surface as **medium or low**. With 99 training
rows, a model that reports "high confidence" on a novel soil and crop combination is
overclaiming, and the thresholds are set so that it cannot.

## Honest limits

* **99 rows.** No meaningful validation split is possible; the test set is ~20 rows.
* **Fixed label vocabulary.** It can only ever return one of the blends it has seen.
* **No dose.** It recommends a blend, not a quantity. `risk` estimates a
  per-hectare application rate separately, and that is a different and weaker model.
* **No cost, availability or environmental constraint.** The cheapest adequate blend
  and the best agronomic blend are often not the same thing.
* **Advisory only.** This should not replace a soil test and a local agronomist.

## Usage

```bash
agri train --module fertilizer
python -m agri.modules.fertilizer.predict
```
