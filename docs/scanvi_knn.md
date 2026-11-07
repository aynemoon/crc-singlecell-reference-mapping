# kNN on the mapped scANVI embedding

`scanvi_knn` fits the existing distance-weighted kNN classifier to labelled SMC
reference latents and predicts every mapped KUL3 cell. It uses
`baselines.knn_k` from `configs/default.yaml`, currently 15. The encoder,
query adaptation, feature selection and existing scoring definitions follow
the original benchmark.

The new row exposes a vote already computed for scANVI's latent-distance
novelty score. One call supplies both the new classifier and the original
`scanvi_latent_distance` score. Native scANVI probabilities continue to belong
to the native row and its distance-only variant. Probability columns for
`scanvi_knn` are explicitly aligned to native scANVI classes. A class-set
mismatch raises an error. Query rows retain their input order. Tied class
probabilities resolve in that declared class order.

Only the query sample key is passed to the mapping models. Query subtype,
lineage and label-role fields remain available to evaluation, after prediction.
The kNN head receives reference labels and latent arrays and accepts no query
labels. Unlabelled SMC cells are excluded from its neighbour index.

This is the first classifier comparison in an ArchMap-compatible CRC study.
ArchMap provides a wider workflow with reference mapping, classifiers,
uncertainty and reference presence measures. The current contribution
implements the scANVI latent kNN head.

## Integration and reproduction

The method is registered in `pipeline.METHODS` and the plotting classifier
list. Scripts 02 and 03 already iterate over the returned Transfer objects,
so the new row enters fine/coarse scoring and open-set deletion experiments.
Scripts 05 and 08 support both the original four-method archived tables and
new five-method output. The open-set distance panel retains its four existing
score series because the new head shares the native latent-distance score.

Run in a disposable checkout with Python 3.12 and the original lock.
Keep that checkout's new outputs apart from the committed reported tables.

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
make data prepare transfer patients openset figures
```

The data target also downloads Pelka. For the core reproduction, the four
SMC/KUL3 URLs in `scripts/download_data.sh` are sufficient. Record their hashes,
the checkout commit, device, package versions and elapsed time per stage.
Archive the full per-cell predictions and reference/query embedding IDs.
Compare the original four-method output with the original reported results
before interpreting a new score.

The current abstention and conformal scripts specifically evaluate the native
scANVI probabilities from their separate 18-training/5-calibration-donor model.
Their existing tables do not evaluate the new kNN head. A subsequent evaluation
contribution must fit its kNN index on those 18 training patients, choose its own
thresholds on the same five held-out patients and evaluate KUL3 after freezing
those choices. Native scANVI thresholds cannot be reused for the kNN vote.

## Phase 2 companion study

Preserve both scientific objectives with this predefined sequence.

| Stage | Controlled comparison | Scientific purpose |
|---|---|---|
| PR 1 | Native scANVI and kNN on the same frozen mapped embedding | Compare the classifiers with the representation held fixed |
| Companion | LR on expression, LR on frozen scANVI, kNN on frozen scANVI and native scANVI | Separate representation and classifier contributions |
| Next mapping PR | A pinned scPoli reference/query mapping with the same labelled SMC index and kNN rule | Compare integration backbones |
| Subsequent metrics PR | Euclidean and regularized Mahalanobis uncertainty, plus reference Presence Score | Test reliability and reference-state coverage under cohort shift |

For the companion, use the exact reference/query IDs from the reproduced run.
Fit the expression LR with the existing log-normalization, reference scaling
and C=0.1. Fit the latent LR with reference-only scaling and the same C.
Reuse the saved scANVI means and the new kNN/native predictions.
Keep all KUL3 cells during expression-only adaptation, then apply the original
shared-subtype mask only for scoring. Query labels never select features,
hyperparameters, patients, covariance estimates or thresholds.

Use paired patient bootstrap draws for differences in pooled fine Macro-F1
and accuracy. Recompute both methods on each identical draw under the
existing present-class scoring rule. Also report equal-patient mean differences
and all six per-patient results. The pooled Macro-F1 difference cannot be
reconstructed from per-patient F1 summaries alone. Repeated training seeds
measure a different source of variability and should be reported separately.

Report per-class precision, recall, F1, reference/query cell support and
reference donor support. Predefine rarity from SMC only. Inspect T helper 17,
CMS2 and CMS4 without tuning against their KUL3 outcomes. Report confusion
destinations and each class's donor distribution. CMS4 has only 11 reference
cells in the archived benchmark. A zero or uncertain result is informative.

## Validation status

The unchanged existing GitHub Actions workflow passed on the standard
`ubuntu-latest` runner for commit `0985ef0`.

- `uv sync --frozen` succeeded.
- `uv run ruff check .` passed.
- `uv run pytest -q` completed with **32 passed, 2 warnings in 18.87 seconds**.
- The warnings concern an upstream Pyro escape sequence and the deliberate
  test of predictions outside the truth-label set.
- The pipeline test uses real AnnData and deterministic model doubles.
  The plotting tests cover archived and expanded method lists.

[Locked-environment CI run](https://github.com/aynemoon/crc-singlecell-reference-mapping/actions/runs/42369804498)

The same complete locked suite also passed locally with 32 tests and two
warnings in 15.73 seconds. This supersedes the earlier dependency-isolated
checks. Local Ruff lint, formatting and compilation passed. The exact lock
was installed after moving the package cache from the limited home tmpfs
to workspace disk.

All four core GEO inputs were downloaded and passed gzip integrity checks.

| Input | Bytes | SHA-256 |
|---|---:|---|
| `smc_counts.txt.gz` | 128941455 | `deff856c7fe1d2c845af5b31e1c65a1dd95ea29541772a66c89bda3aaa8c0666` |
| `smc_annotation.txt.gz` | 434022 | `4bf2ebdadc88da53b0c46e654a6576e841925addc25c981d7cdbc0d988b4677c` |
| `kul3_counts.txt.gz` | 56598691 | `046cbbd6c7501a4ddbca6d09fc0bb923e62d4d9421d5781efdeacb2c0a5b3b31` |
| `kul3_annotation.txt.gz` | 194418 | `4eb9da239bef95b9003c0f6d0eeb41c6598ecc794cd25f57ad06d9bad81e881b` |

### Full-data reproduction attempt

The full default script 01 was executed in an isolated reproduction directory.
Ten original training, preprocessing, scoring and configuration files were
verified byte for byte against commit `63bcf65028fb`. The worker exposes
Python 3.12.14, nine CPU threads, approximately 9.7 GiB RAM and no visible GPU.

The full import terminated with exit 137 (SIGKILL) after sustained high memory
use. Reduced smoke-input preparation was also active near termination. No
kernel diagnostic confirmed the exact kill cause. No complete full-size
reference/query output, trained full model or new CRC performance estimate
was produced. The original reported result files remain the reported archive.

A full validation requires a successful complete import and the original
training schedules, followed by transfer, patient summaries, open-set runs
and regenerated figures. A further attempt should isolate the importer and
provide sufficient memory. The existing loader also exposes a smaller
`chunksize` option, which can be assessed as an I/O resource adjustment with
checks for identical cell/gene order and counts. Preprocessing changes belong
in a separate contribution. The current CI workflow runs software tests.

### Reduced real-data backend smoke

An explicitly reduced software probe selected 1,200 SMC columns and 300 KUL3
columns uniformly with seed 0, without using labels for selection. It retained
all 33,694 input genes while loading, then selected 500 HVGs from the reference.
It used eight latent dimensions, one hidden layer of width 32, batch size 128,
four CPU threads and one epoch for each training/adaptation stage.

Per-patient Seurat-v3 LOESS was numerically singular with the reduced donor
sample sizes. The probe was rerun with pooled reference-only HVG selection.
Original sample keys were retained for the models. This pooled selection is
a smoke-specific adaptation. Full benchmark preprocessing is unchanged.

Actual reference training, both query mappings and all six Transfer outputs
completed in 3.28 seconds, excluding input preparation and imports. Process
peak RSS was approximately 1,186 MiB. The smoke had 32 named reference classes.
Checks covered finite normalized probabilities, shapes, prediction lengths
and matching native/kNN class order.

This probe establishes backend execution on a reduced real input. It supplies
no CRC accuracy, rare-subtype result or statistical superiority claim. The
full four-way companion study and patient-paired performance intervals remain
pending full-size reproduction. Native calibration/conformal output also
requires a separate held-out-donor evaluation of the new head.

## Attribution and next extensions

The original benchmark is by Matin Gerami. Aysan Neghabi contributes the
mapped kNN classifier and its verification.

ArchMap's classifier comparison motivates this controlled experiment.
Its atlas ranking requires testing in the patient-based CRC design.
The implementation uses this repository's existing scikit-learn kNN helper.
The separate ArchMap ML repository's licensing still needs confirmation
before copying or redistributing its source or pretrained assets.

- Lotfollahi, Bright, Skorobogat and colleagues. ArchMap. Nature Genetics, 2026.
  [Paper](https://doi.org/10.1038/s41588-026-02756-y)
- Lotfollahi and colleagues. scArches. Nature Biotechnology, 2022.
  [Paper](https://doi.org/10.1038/s41587-021-01001-7)
- De Donno and colleagues. scPoli. Nature Methods, 2023.
  [Paper](https://doi.org/10.1038/s41592-023-02035-2)
- [ArchMap ML workflow](https://github.com/theislab/archmap_data/tree/mappingjob)
- [ArchMap platform](https://github.com/theislab/archmap)
- [scArches source](https://github.com/theislab/scarches)
- [scPoli mapping tutorial](https://docs.scarches.org/en/latest/scpoli_surgery_pipeline.html)
