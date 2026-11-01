# disease

**Question.** What is wrong with this leaf?

| | |
|---|---|
| Task | image classification, inference only |
| Algorithm | pretrained CNN weights |
| Weights | third-party (PlantVillage-style) |
| Classes | ~38 |
| Gate | **none** - inference only |

## Why there is no metric

The weights were not trained in this repository and there is no validation set for
regional crop varieties here. Publishing a metric would mean either inventing one or
reporting a number measured on data this model was trained on.

So `agri evaluate` records disease as **skipped**, with the reason. An honest gap beats
a fabricated number.

## What it is good for

Rapid triage: pointing a farmer at a probable cause so they can confirm it. It is not
a diagnosis.

## Honest limits

* **Third-party weights, ~38 classes.** We did not train them and cannot vouch for
  their validation methodology.
* **Not validated on regional varieties.** PlantVillage-style datasets are dominated by
  a few crops under controlled conditions. A disease on a local variety, or a
  nutrient deficiency that *looks* like a disease, can be confidently misclassified.
* **Lab-style photographs.** Datasets like this are typically leaf-on-plain-background.
  A phone photo of a plant in a field has soil, other leaves, shadow and motion blur
  in it, and the model has not seen that distribution shift.
* **Nutrient deficiency versus pathogen is a known confusion.** Several deficiencies
  present as yellowing that resembles a viral symptom.
* **No severity estimate.** It names a class, not how far advanced it is, which is
  usually what determines whether treatment is worth it.
* **Confidence is the classifier's own, uncalibrated** against real field data.
* **Advisory only.** Confirm with an extension officer before spending on treatment.

## Usage

```bash
python -m agri.modules.disease.predict
```

```bash
curl -X POST localhost:8000/v1/predict/disease \
  -H 'Content-Type: application/json' \
  -d '{"image_base64":"<base64 jpeg>"}'
```
