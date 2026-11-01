# Release process

## What "release" means here

There is no package to publish. A release is a **new model version**, and the
thing that must be reproducible is the artefact, not the wheel.

Every release produces, per trained module:

```
models/registry/<module>/<version>/bundle.joblib
models/registry/<module>/<version>/manifest.json
models/registry/<module>/LATEST
reports/metrics/<module>_v<version>.json
```

and the manifest is the contract. If it is not in the manifest, it did not happen.

---

## The gate

`agri evaluate --all` is the release gate. It exits 1 if any module fails its
configured floor. CI runs it on every pull request.

**A model does not ship because it finished training. It ships because it cleared its
floor on data the evaluator can verify it was trained on.**

`evaluate_module` checks the dataset digest recorded at training time against the file
on disk and refuses to score a mismatch. A metric computed on different data than the
model was fitted on is not a metric.

---

## Checklist

### Before training

- [ ] `agri doctor` is clean: python, datasets present, registry writable
- [ ] `agri data --all` passes: every input matches its declared schema and ranges
- [ ] `.env` is not committed and contains no value that also appears in a tracked file
- [ ] `configs/<module>.yaml` changed only where the change is intended; `git diff` them

### Training

- [ ] `git status` is clean, so `git_sha` in the manifest identifies the code
- [ ] `agri train --module <name>`
- [ ] The gate passed in the training output
- [ ] The recorded `baseline_metrics` are meaningfully beaten, not just cleared

### Evaluation

- [ ] `agri evaluate --all` exits 0
- [ ] `reports/metrics/<module>_evaluated.json` is committed for the modules that changed
- [ ] Any metric that *dropped* has an explanation in the pull request

### Serving

- [ ] `agri health` reports the required modules ready
- [ ] `agri serve` then `GET /v1/health` returns 200
- [ ] One live request per changed module returns a plausible answer
- [ ] `GET /v1/models` shows the new version against the module

### Cleaning up

- [ ] Temporary scripts under `scripts/` prefixed with `_` are removed
- [ ] No large file was added under `data/`, `models/` or `reports/`
- [ ] `.gitignore` covers any new cache directory

---

## Versioning

`version` in `configs/<module>.yaml` is the artefact version. Bump it whenever the
fitted artefact changes in a way a consumer could observe: features, algorithm,
hyperparameters, split, or the training data.

| Change | Version |
|---|---|
| Hyperparameter tweak, retrain on new data | patch, e.g. `1.0.0` to `1.0.1` |
| New feature, new algorithm, changed split | minor, e.g. `1.0.1` to `1.1.0` |
| Different task, or an output contract change | major, e.g. `1.1.0` to `2.0.0` |

The API reports `model_version` in every response, so a caller can detect a change
without being told.

---

## Rolling back

The registry is append-only by version, so rollback is a pointer change:

```bash
# Re-point LATEST at the previous good version.
printf '%s' "1.0.0" > models/registry/<module>/LATEST

# Cached predictors must be dropped; a running worker keeps serving the old artefact
# until it reloads. Restart the service, or call the cache's reload().
```

There is no hot-reload endpoint by design. Reloading a model in a live process while
requests are in flight is a correctness problem, not a feature, and the honest answer
for a single-node deployment is a restart.

---

## When a gate cannot be met

This will happen, and the correct response is not to lower the floor quietly.

The process used for `risk` is the template:

1. **Measure.** Establish what the model actually achieves, and against what chance
   rate.
2. **Check for a missing feature.** Adding `fertilizer_per_hectare` and
   `pesticide_per_hectare` made `risk` slightly *worse* (0.6196 to 0.5840), which is
   evidence the signal is absent rather than that features were missing.
3. **Set the floor to a level the data supports, and say so.** `risk` shipped at 0.50
   against a 0.33 chance rate, and the original 0.70 intent is recorded in
   [DECISIONS.md](DECISIONS.md) as unmet.
4. **Document the shortfall where a user will see it** - the module doc, the README,
   and the response reasons.

What is not acceptable: setting the floor from the result without recording what the
original intent was and why it changed. That is indistinguishable from moving the
goalposts, and the difference is entirely whether it is written down.

---

## Reproducing a released artefact

Given a manifest, a build is reproducible if:

| Field | Reproduces |
|---|---|
| `git_sha` | the code |
| `config_digest` | the configuration |
| `data_digest` | the training data |
| `python`, `platform`, `libraries` | the environment |
| `seed` | the random seed |

`seed` is set from config and applied by `set_global_seed`, so a retrain on the same
data and code reproduces the same split and the same fitted estimator for the
deterministic algorithms. XGBoost and HistGradientBoosting may differ in the last
decimal place across machines and library builds; the manifest records the library
versions so that difference is attributable rather than mysterious.

---

## Data hygiene

* `data/raw` holds inputs and is not committed.
* Duplicate files were SHA-256 verified before deletion. `market_mandi.csv` is
  canonical; `agriculture_prices.csv` is a genuinely different schema and was kept.
* Build output is not committed. If a build artefact is needed to reproduce a result,
  it belongs in a release asset, not in the tree.

## Secret hygiene

Two live credentials are present in the working tree and **must be rotated**:

| File | Contains |
|---|---|
| `smart_crop_ai_PRD.md` | an OpenWeather API key, in plaintext, currently authenticating |
| `intellicrop-ai-farm-main/.env` | Supabase URL and keys |

Both are listed in `.gitignore` and neither is committed. They must still be revoked:
a key that has been on a disk in plaintext is compromised regardless of whether it
reached git. Rotate first, then confirm `git log --all -- smart_crop_ai_PRD.md` is
empty.
