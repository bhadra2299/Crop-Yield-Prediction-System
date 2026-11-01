# soil_image

**Question.** What soil type is in this photograph?

| | |
|---|---|
| Task | image classification |
| Algorithm | CNN transfer learning (torch/torchvision, optional `vision` extra) |
| Dataset | `data/raw/soil_images/`, 4,200 JPEGs, 10 classes |
| Split | **group-based** |
| Gate | `macro_f1 >= 0.85` |

## The dataset is synthetic, and the artefact says so

The training images are **synthetically generated**, not photographed in a field.

This means the accuracy measures **how well the model separates our synthetic
classes**. It is a demonstration that the pipeline works end to end. It is not
evidence that the model identifies real soil in real conditions.

That is enforced, not just documented:

* The manifest carries `synthetic_data: true`.
* The response includes a limitation string naming it.
* This document is the third place it appears.

A model trained on synthetic separability and presented as a field classifier would be
the single most misleading thing in this repository. So the number is reported with
its provenance attached, everywhere it appears.

## Why a group split

`split.strategy: group`. If augmented variants of the same source image land in both
train and test, the test score measures memorisation of that source, not
generalisation. Grouping keeps all variants of one source on one side of the split.

## Honest limits

* **Synthetic training data.** The headline limitation. Restated.
* **10 classes.** A fixed vocabulary; a real soil outside it is forced into the
  nearest one.
* **Photograph conditions matter enormously.** Wet versus dry soil looks completely
  different. Lighting, shadow, camera white balance and depth of field all shift the
  input distribution away from anything the model saw.
* **A photograph sees a few square centimetres.** Soil type varies across a field.
* **Soil *type* is not soil *quality*.** Knowing it is loam says nothing about its
  nutrients - that is `soil_nutrient`, and a lab.
* **Minimum resolution and size enforced.** Oversized uploads are rejected with a 413
  rather than downsampled silently.

## Usage

Requires the optional extra:

```bash
pip install -e ".[vision]"
agri train --module soil_image
agri evaluate --module soil_image
```

```bash
curl -X POST localhost:8000/v1/predict/soil-image \
  -H 'Content-Type: application/json' \
  -d '{"image_base64":"<base64 jpeg>"}'
```
