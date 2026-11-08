# kNN on the mapped scANVI embedding

PR 1 exposes `scanvi_knn`, distance-weighted Euclidean kNN (`k=15`) on the
existing scANVI reference and mapped query posterior means. It compares
classifiers while keeping the fitted representation and query adaptation fixed.
It is a classifier extension, not a complete ArchMap implementation or a claim
of algorithmic superiority. The original benchmark is by **Matin Gerami**.
**Aysan Neghabi** contributes the classifier and its verification.

## Method and data

The existing kNN helper supplies both class probabilities and the mean distance
to the 15 reference neighbours. Only labelled reference cells enter its index.
Probability columns are aligned to native scANVI classes; class-set mismatches
raise an error. Query order is preserved. Equal maximum votes resolve in the
declared class order. Zero-distance neighbours receive all the vote weight.
Equal distances at the neighbour-selection boundary inherit sklearn's ordering
behaviour; reference-row permutation invariance is not promised in that case.

HVG selection uses the reference only. Query annotations are removed before
mapping, and its registered label is Unknown. All query cells participate in
unlabelled adaptation, so this is a transductive cross-cohort comparison.
The new head never changes the native model.

The local experiment used SMC (63,689 cells, 23 patients) and KUL3 (27,414
cells, 6 patients), retaining the original 3,000-HVG configuration. There are
61,508 labelled reference cells and **34 prediction classes**. Fine-level
scoring uses **33 shared classes**, comprising 23,218 query cells.
Proliferative ECs has 44 reference cells but no shared query cells. Natural
novelty uses the shared cells plus 734 novel cells; 3,462 cells are excluded.
The original present-truth-class Macro F1 definition is unchanged.

## Archived benchmark and completed local run

The original `results/tables/` and `results/figures/` remain the historical
archive. Newly executed results are isolated in
[the PR 1 experiment report](../results/experiments/pr1_local_2026-11-08/README.md).

| Classifier | Archived fine Macro F1 | Local fine Macro F1 | Local accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.742016 | 0.741243 | 0.817125 |
| kNN + PCA | 0.728851 | 0.727043 | 0.813119 |
| scVI + kNN | 0.694419 | 0.694198 | 0.783272 |
| Native scANVI | 0.753714 | 0.750986 | 0.816349 |
| scANVI + kNN | Not previously reported | 0.747181 | 0.821647 |

Logistic Regression uses 3,000 scaled log-normalized expression features.
PCA belongs to the kNN baseline. The broader representation-versus-classifier
question is not resolved by this two-head comparison.

Aysan completed `make transfer` and `make patients` on Windows 11/WSL2,
reporting 32 GB physical RAM and CPU-only training. The log records seed 0,
200 scVI epochs, 40 scANVI epochs, both 100-epoch query adaptations, and
`TRANSFER_EXIT=0`. Training progress timers sum to about 2 h 14 min; this is
not total pipeline wall time or peak memory. The earlier 9.7 GiB Work import
failure and reduced smoke test are historical attempts, superseded for the
main transfer experiment by this completed full-data run.

The supplied provenance identifies commit `506442b42c37a00e99c046bf5ca44cea121cb2c3`.
Configuration and lock hashes, installed package versions, and core GEO hashes
match the reviewed inputs. The preserved training provenance and analysis
manifest distinguish the training environment from the later analysis runtime.
Raw counts were not independently reprocessed during the saved-output review.

## Paired uncertainty and classifier behaviour

Two thousand seed-0 patient-bootstrap draws resample all six query patients
with replacement, retaining every cell and repeated-patient multiplicity.
Each draw evaluates both classifiers and subtracts their pooled scores.
Classes present in that draw's truth determine Macro F1, as in the benchmark.

| kNN minus native | Estimate | Percentile 95% interval |
|---|---:|---:|
| Macro F1 | -0.003805 | [-0.022650, 0.007224] |
| Accuracy | 0.005298 | [-0.018333, 0.022453] |

Both intervals include zero. Neither superiority nor equivalence is established.
Five of six patients have lower Macro F1 with kNN. Equal-patient mean Macro F1
difference is -0.006904; leave-one-patient-out pooled differences range from
about -0.0130 to +0.0020. These intervals condition on this fitted model and
mapped cohort; they exclude training-seed and new-cohort variation.

The classifiers disagree on 1,750 scored cells. kNN corrects 782 native errors
and introduces 659 errors, yielding 123 additional correct predictions.
Higher accuracy coexists with lower Macro F1 and balanced accuracy.

| Subtype | Native precision / recall / F1 | kNN precision / recall / F1 |
|---|---|---|
| cDC | 0.595 / 0.944 / 0.730 | 0.708 / 0.914 / 0.798 |
| T follicular helper | 0.680 / 0.869 / 0.763 | 0.841 / 0.795 / 0.817 |
| T helper 17 | 0.472 / 0.782 / 0.589 | 0.613 / 0.615 / 0.614 |
| CMS1 | 0.622 / 0.327 / 0.429 | 0.770 / 0.150 / 0.252 |
| CMS2 | 0.779 / 0.705 / 0.740 | 0.514 / 0.906 / 0.656 |
| CMS4 | 0 / 0 / 0 | 0 / 0 / 0 |

These are exploratory precision-recall trade-offs, not validated subtype gains.
CMS2 false positives rise from 124 to 531, including 243 CMS1 and 211 CMS3
cells. CMS1 reference support is concentrated: 722/1,201 cells belong to SMC03.
KUL19 supplies 488/620 CMS2 and 228/259 T follicular helper query cells;
KUL31 supplies 168/312 T helper 17 cells. Donor presence is not donor balance.
Reference cell count versus subtype F1 change gives Spearman rho 0.178,
nominal p=0.322 across 33 subtypes, without a clear monotonic association.

The reconstructed neighbourhoods describe how the kNN decisions arise.
True CMS1 query cells receive mean vote weight 46.0% CMS3, 36.0% CMS2 and
17.3% CMS1. No true CMS4 query cell has a CMS4 reference among its 15 neighbours.
CMS4 has only 11 reference cells across 9 donors and 30 query cells, 21 from
KUL21. These observations do not establish why the neighbourhoods formed or
that reference imbalance caused the errors.

Native CMS1 F1 changed from **0.6126 archived to 0.4287 locally**, despite
unchanged support counts and close aggregate reproduction. The cause remains
unresolved. Historical per-cell predictions, HVGs, logs or checkpoints could
narrow it without training; none was available for this comparison.

## Novelty and remaining validation

The new head reuses the existing scANVI latent-distance novelty array exactly.
Its natural-novelty AUROC 0.710302 and average precision 0.048226 (prevalence
0.030645) are **not an independent novelty improvement**. Native confidence
AUROC is 0.508508. Weighted vote normalization does not establish calibration.

| Contribution requirement | Status |
|---|---|
| Transfer, registration, unit tests | Implemented; previous head passed locked CI |
| Full transfer and patient-paired analysis | Completed; saved outputs independently reconstructed |
| Reproducible analysis and isolated figures | Provided here, without training |
| Native class-order evidence | Reference registry and classifier shape verified statically; mapped-query runtime columns not directly captured |
| Deleted-subtype open-set evaluation | Integrated but new execution not supplied; needs saved deletion artifacts or training |
| New-head calibration/conformal | Incomplete; needs suitable 18-training/5-calibration artifacts or training/adaptation |
| Checklist exceptions | Require explicit maintainer agreement; none assumed |

The full-reference embedding cannot replace a model trained without the five
calibration patients. Existing native calibration thresholds cannot be reused
for kNN. Missing experiments remain acceptance gaps under
[CONTRIBUTING.md](../CONTRIBUTING.md). Keep the PR draft until completed or
explicitly deferred. Six patients, one training seed, one query cohort, shared
annotation provenance and exploratory subtype selection limit inference.

## Reproduce without training

```bash
uv sync --frozen
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest -q
uv run --frozen python scripts/analyze_scanvi_knn.py \
  --predictions /path/to/predictions.parquet \
  --embeddings /path/to/embeddings.h5ad \
  --reference-annotation data/raw/smc_annotation.txt.gz \
  --query-annotation data/raw/kul3_annotation.txt.gz \
  --historical-dir results/historical/63bcf65028fb \
  --native-classes results/verification/scanvi_reference_metadata.json \
  --provenance /path/to/provenance.txt \
  --output results/experiments/pr1_reproduction
```

The output directory must not exist. The script writes 14 tables, six vector PDF
figures, input/source hashes, versions and verification checks. Annotation IDs
must match the prepared cells and their labels must match the embeddings.
There is no training, adaptation, feature selection or threshold tuning.

Historical input must be a content-verified snapshot of the original Git blobs
at `63bcf65028fb2597a23742c1f52a21d963697681`. The included snapshot preserves
the two tables used for comparison. The script verifies its commit, manifest
and both CSVs against pinned SHA-256 values before loading experimental data.
Regenerated working-tree `results/tables/` is not a valid substitute, even
with a copied or altered manifest. To recreate the snapshot at a fresh path:

```bash
uv run --frozen python scripts/snapshot_historical_results.py --output /new/path/original_crc_results
```

This reads `git show` blobs and requires the baseline commit to be available
locally. It never copies the working-tree tables. The included verified
snapshot can be used without Git history beyond the current checkout.

The supplied reference checkpoint was inspected statically through pickle
opcodes, without `torch.load`, deserialization of objects, tensor loading or
model execution. Its 35-category registry ends with Unknown; its classifier
weight has shape 34 x 128. Pinned scvi-tools 1.5.1 uses the first 34 registry
entries for `predict(soft=True)` columns. The final two are **gamma delta T
cells, cDC**, opposite to their sorted-reference order. Reconstruction using
the recovered order still matches exactly. The registered reference labels
cover 63,689 cells, including 2,181 Unknown cells.

The [metadata report](../results/verification/scanvi_reference_metadata.json)
records the checkpoint and framework-source hashes. Label transfer in the
pinned version does not extend categories and retains Unknown last, supporting
the same mapped-query order. The original mapped-query DataFrame was not
captured directly; this is registry/source evidence, not a new execution of
native query prediction. To independently inspect a checkpoint:

```bash
uv run --frozen python scripts/inspect_scanvi_checkpoint.py \
  --checkpoint /path/to/model.pt --output /new/path/checkpoint_metadata.json
```

`--native-classes` accepts this report or an actual column-name JSON list.
Without either it explicitly records a sorted-reference fallback. Future
script-02 runs export the actual native column list automatically.
`--full-output` additionally emits bootstrap draws, derived group deltas and
PNG previews; these reproducible extras are omitted from the PR artifact set.

Reconstruction matched all 27,414 labels, maximum confidence values and mean
distances exactly. Its 27,414 x 34 probabilities normalize to floating-point
precision, with no maximum-vote ties or zero-distance neighbours. Synthetic
tests separately cover those edge cases and annotation leakage. Actual
interval endpoints are drawn independently of the point estimate.
Archived figures retain their original rendering; the correction applies to
the new report and future script-05 output.

ArchMap motivates the wider project ([Lotfollahi and colleagues, 2026](https://doi.org/10.1038/s41588-026-02756-y)).
This PR reuses the benchmark's sklearn helper; no ArchMap source was copied.
scPoli, Mahalanobis uncertainty, Presence Score and latent Logistic Regression
remain outside this PR. Source licensing must be checked before later reuse.
