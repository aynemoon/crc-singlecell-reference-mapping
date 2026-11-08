# PR 1 saved-output experiment

This directory contains newly executed analyses of Aysan Neghabi's completed
SMC-to-KUL3 run at commit `506442b42c37a00e99c046bf5ca44cea121cb2c3`.
The original benchmark by Matin Gerami remains unchanged in `results/tables/`
and `results/figures/`. This is a frozen-scANVI classifier comparison.

The saved inputs were supplied for review and are not duplicated in Git.
Reproduction requires `predictions.parquet`, `embeddings.h5ad`, and the two
SMC/KUL3 annotation files matching their prepared cell IDs. No model training
or adaptation was performed by this analysis. See
[method and commands](../../../docs/scanvi_knn.md).

## Verification and provenance

- All 27,414 reconstructed kNN predictions, maximum confidence values and
  mean neighbour distances match exactly; 34 reference probability classes
  are distinct from the 33 shared query classes scored on 23,218 cells.
- Two thousand paired patient-bootstrap resamples use seed 0 and preserve
  repeated-patient multiplicity. Macro F1 follows the existing present-class rule.
- Local native/kNN Macro F1 is 0.750986/0.747181, accuracy 0.816349/0.821647.
  Differences are -0.003805 [-0.022650, 0.007224] and
  +0.005298 [-0.018333, 0.022453], respectively. Neither superiority nor
  equivalence is established.
- Original training provenance is preserved verbatim in
  [training_provenance.txt](training_provenance.txt). It records the lock,
  configuration, raw-input hashes, installed versions and local working tree.
  [manifest.json](manifest.json) separately records analysis versions, source
  hashes, input hashes and reconstruction checks. Its commit identifies the
  checkout base; the analysed source changes were uncommitted during review.
- Analysis used the unchanged locked environment, Python 3.12.14 and four
  numerical threads. All 14 retained CSV tables are byte-identical to the
  previous verified analysis. Recovery of the class order did not change scores.
- Historical comparisons use the [verified Git snapshot](../../historical/63bcf65028fb/historical_snapshot.json),
  not working-tree `results/tables/`. Pinned commit and content hashes are checked
  before loading experimental data; regenerated or forged snapshots are rejected.
- Locked Ruff lint/format checks passed; the full suite passed with **56 tests,
  one expected sklearn warning, in 26.32 seconds**. The warning is the original
  test of predictions outside a truth-label set. No new GitHub CI run was
  triggered because these changes have not been pushed.

## Figures

Purple denotes native scANVI and orange denotes scANVI + kNN in classifier
comparisons. Subtype and donor analyses are exploratory, without significance
annotations. Each figure is retained once as an editable vector PDF. Bootstrap
draws, derived group differences and PNG previews can be regenerated with
`--full-output`; they are omitted from the PR to avoid redundant artifacts.

| Figure | Scientific purpose | Files |
|---|---|---|
| Performance | Pooled Macro F1 and accuracy; actual marginal 95% patient-bootstrap endpoints are drawn separately from point estimates | [PDF](figures/performance.pdf) |
| Paired differences | kNN minus native on identical draws; dashed zero line; intervals include zero | [PDF](figures/paired_differences.pdf) |
| Patient variation | Within-patient comparisons; lines connect the two classifiers, not confidence bounds | [PDF](figures/patients.pdf) |
| Subtype trade-offs | Precision, recall and F1 for selected CMS and immune subtypes; overlapping markers indicate identical values | [PDF](figures/subtype_tradeoffs.pdf) |
| CMS confusions | Numbers are cell counts; colours are within-truth-subtype fractions; Other includes every non-CMS prediction | [PDF](figures/cms_confusion.pdf) |
| Reference support | Labelled reference cell counts on a log scale and largest-donor shares; purple/orange denote reference/query here | [PDF](figures/reference_support.pdf) |

## Table inventory

All classification and confusion tables use shared query cells unless stated
otherwise. Reference support includes the reference-only class. Coarse predictions
are validated against their reference parent labels but these tables report fine metrics.

| File | Contents |
|---|---|
| [metrics.csv](metrics.csv) | Pooled Macro F1, accuracy and balanced accuracy for every saved method |
| [bootstrap.csv](bootstrap.csv) | Native/kNN pooled metrics and paired differences with percentile intervals |
| [groups.csv](groups.csv) | Patient and tissue scores, with cell counts |
| [patient_sensitivity.csv](patient_sensitivity.csv) | Equal-patient mean differences and leave-one-patient-out results |
| [subtypes.csv](subtypes.csv) | Precision, recall, F1, TP, FP, FN, query support and reference/donor support |
| [support.csv](support.csv) | Reference counts, donor counts, largest shares and effective donor counts, defined as 1/sum(donor share squared) |
| [donors.csv](donors.csv) | Cell counts by subtype and donor, for labelled reference and shared query |
| [support_association.csv](support_association.csv) | Exploratory Spearman association of reference abundance with F1 difference; nominal p-value |
| [confusions.csv](confusions.csv) | All nonzero prediction destinations for CMS1/2/3/4, cDC, T follicular helper and T helper 17 |
| [disagreements.csv](disagreements.csv) | Patient counts of disagreements, corrected errors and introduced errors |
| [neighbours.csv](neighbours.csv) | Neighbour-label fractions, mean normalized distance weights, support and donor diversity, using all query cells of each focus subtype |
| [novelty.csv](novelty.csv) | Natural novelty AUROC, average precision and prevalence on shared plus novel cells |
| [historical_metrics.csv](historical_metrics.csv) | Archived and local aggregate values, kept in separate columns |
| [historical_subtypes.csv](historical_subtypes.csv) | Archived/local subtype F1 and differences; no invented historical new-head result |

## Interpretation and outstanding requirements

Five patients have lower Macro F1 with kNN, despite improved pooled accuracy.
Precision-recall trade-offs, donor concentration and the CMS2 confusion shift
matter more than a ranked aggregate score. CMS4 has no same-label neighbour
among the 15 selected references for any of its 30 query cells. CMS1 mean vote
weights are dominated by CMS3 and CMS2. This describes the current voting
mechanism and does not establish a causal explanation of latent geometry.

Native CMS1 F1 drifted from 0.6126 archived to 0.4287 locally; its cause remains
unresolved. One training seed and one cohort cannot establish robustness.
Six-patient intervals condition on this fitted model and mapped query cohort.

The kNN novelty array is identical to the existing latent-distance array.
Its AUROC is not an independent improvement. Calibration is not established
by probability normalization. Deleted-subtype open-set, new-head calibration
and conformal evaluation are not supplied and require suitable saved artifacts
or additional training/adaptation, or explicit maintainer agreement to defer.

The [reference checkpoint metadata](../../verification/scanvi_reference_metadata.json)
was recovered by static pickle-opcode inspection, without loading tensors or
executing serialized objects. There are 35 registry categories including
Unknown last, with 34 classifier outputs. The recovered prediction order ends
with gamma delta T cells, cDC, unlike the earlier sorted fallback. The final
reconstruction uses this recovered order and still matches every saved label,
confidence and distance exactly. No vote ties occur in the real run.

scvi-tools 1.5.1 uses the registry prefix as soft-prediction columns and preserves
label categories during query transfer. This establishes reference order and
supports the expected mapped-query order. The original mapped-query DataFrame
was not captured directly; the reference checkpoint alone cannot independently
document that runtime artifact. The report records this limitation explicitly.

Keep PR 1 draft until the contribution-checklist requirements are completed or
explicitly deferred, and CI passes after an approved push. No commit, push,
merge or PR-state change was performed.
