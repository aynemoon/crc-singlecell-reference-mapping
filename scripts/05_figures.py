"""Figures for the README, from the saved tables."""

from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from scmap.config import load_config
from scmap.plotting import (
    CLASSIFIERS,
    DOUBLE,
    INK,
    INK_3,
    METHOD_LABELS,
    MM,
    ONE_HALF,
    SEQUENTIAL,
    SERIES,
    SINGLE,
    SKY,
    apply_style,
    panel_label,
    save,
)

cfg = load_config()
TAB, FIG = cfg.path("results") / "tables", cfg.path("results") / "figures"
NOVELTY_METHODS = ["logreg", "scvi_knn", "scanvi", "scanvi_latent_distance"]


def umap_embedding() -> pd.DataFrame:
    """2-D UMAP of the scANVI latent space, cached so figures can be redrawn without refitting."""
    path = TAB / "umap_scanvi.csv"
    if path.exists():
        return pd.read_csv(path)
    import scanpy as sc

    emb = ad.read_h5ad(cfg.path("processed") / "embeddings.h5ad")
    rng = np.random.default_rng(cfg.seed)
    sub = emb[np.sort(rng.choice(emb.n_obs, min(40_000, emb.n_obs), replace=False))].copy()
    sc.pp.neighbors(sub, use_rep="scANVI", n_neighbors=15, random_state=cfg.seed)
    sc.tl.umap(sub, random_state=cfg.seed)
    xy = sub.obsm["X_umap"]
    out = pd.DataFrame(
        {
            "cell": sub.obs_names,
            "umap_1": xy[:, 0],
            "umap_2": xy[:, 1],
            "cohort": sub.obs["cohort"].astype(str).to_numpy(),
            "cell_type": sub.obs[cfg["labels"]["coarse"]].astype(str).to_numpy(),
        }
    )
    out.to_csv(path, index=False)
    return out


def _spread_labels(xy: np.ndarray, box: tuple[float, float], n_iter: int = 500) -> np.ndarray:
    """Move label centres apart until their boxes (width, height in data units) do not overlap."""
    xy = xy.astype(float).copy()
    w, h = box
    for _ in range(n_iter):
        moved = False
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                dx, dy = xy[j] - xy[i]
                ox, oy = w - abs(dx), h - abs(dy)
                if ox > 0 and oy > 0:  # overlap: separate along the axis that needs less movement
                    if ox / w < oy / h:
                        step = np.array([np.sign(dx) or 1.0, 0.0]) * ox / 2
                    else:
                        step = np.array([0.0, np.sign(dy) or 1.0]) * oy / 2
                    xy[i] -= step
                    xy[j] += step
                    moved = True
        if not moved:
            break
    return xy


def fig_umap() -> None:
    emb = umap_embedding()
    xy = emb[["umap_1", "umap_2"]].to_numpy()
    order = np.random.default_rng(cfg.seed).permutation(len(emb))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.8, 70 * MM))
    ax = axes[0]
    for name, colour, role in zip(
        ["SMC", "KUL3"], SERIES[:2], ["reference, Korea", "query, Belgium"], strict=True
    ):
        m = (emb["cohort"].to_numpy() == name)[order]
        ax.scatter(
            xy[order][m, 0],
            xy[order][m, 1],
            s=0.4,
            c=colour,
            alpha=0.4,
            lw=0,
            rasterized=True,
            label=f"{name} ({role})",
        )
    leg = ax.legend(markerscale=5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    panel_label(ax, "a", x=-0.08, y=1.06)

    ax = axes[1]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.4, c=SKY, alpha=0.3, lw=0, rasterized=True)
    centres = emb.groupby("cell_type")[["umap_1", "umap_2"]].median()
    span = np.ptp(xy, axis=0).max()
    pos = _spread_labels(centres.to_numpy(), box=(0.2 * span, 0.05 * span))
    for name, (x, y) in zip(centres.index, pos, strict=True):
        ax.text(x, y, name, fontsize=6, ha="center", va="center", color=INK)
    panel_label(ax, "b", x=-0.08, y=1.06)
    for ax in axes:
        ax.set(xticks=[], yticks=[], xlabel="UMAP 1", ylabel="UMAP 2")
    fig.tight_layout(w_pad=2)
    save(fig, FIG / "fig1_umap_scanvi")


def fig_accuracy() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 55 * MM))
    for ax, letter, level, label in zip(
        axes,
        "ab",
        ["coarse", "fine"],
        ["major cell types (6)", "shared fine subtypes (33)"],
        strict=True,
    ):
        m = (
            pd.read_csv(TAB / f"metrics_{level}.csv")
            .query("metric == 'macro_f1'")
            .set_index("method")
        )
        m = m.loc[[method for method in CLASSIFIERS if method in m.index]]
        x = np.arange(len(m))
        # with 6 query patients the percentile interval can sit below the point estimate
        # (most patient-resamples omit a patient and score lower); clip bars at zero
        ax.errorbar(
            x,
            m.value,
            yerr=[np.clip(m.value - m.ci_low, 0, None), np.clip(m.ci_high - m.value, 0, None)],
            fmt="o",
            color=SERIES[0],
            elinewidth=0.8,
        )
        ax.set_xticks(x, [METHOD_LABELS[k] for k in m.index], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(m) - 0.4)
        ax.set_ylabel(f"Macro-F1 on KUL3\n{label}")
        panel_label(ax, letter, x=-0.3)
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fig.tight_layout(w_pad=2.5)
    save(fig, FIG / "fig2_accuracy")


def fig_per_class() -> None:
    f1 = pd.read_csv(TAB / "per_class_f1_fine.csv", index_col=0)
    f1 = f1[[method for method in CLASSIFIERS if method in f1]]
    roles = pd.read_csv(TAB / "label_roles.csv").set_index("cell_subtype")
    f1 = f1.assign(_ct=roles.loc[f1.index, "cell_type"].to_numpy()).sort_values(["_ct", "scanvi"])
    groups = f1.pop("_ct")
    fig, ax = plt.subplots(figsize=(ONE_HALF, 125 * MM))
    im = ax.imshow(f1.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(f1.shape[1]),
        [METHOD_LABELS[k] for k in f1.columns],
        rotation=30,
        ha="right",
        fontsize=5,
    )
    ax.set_yticks(
        range(len(f1)),
        [
            f"{s.replace('gamma delta', 'γδ')}  ({g.split()[0]})"
            for s, g in zip(f1.index, groups, strict=True)
        ],
        fontsize=5,
    )
    ax.tick_params(length=1.5, width=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, j in np.ndindex(f1.shape):
        v = f1.iat[i, j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=4,
            color="white" if v > 0.65 else INK,
        )
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("F1 on KUL3")
    cb.outline.set_linewidth(0.4)
    save(fig, FIG / "fig3_per_subtype_f1")


def fig_open_set() -> None:
    path = TAB / "open_set_auroc.csv"
    if not path.exists():
        return
    a = pd.read_csv(path)
    types = list(dict.fromkeys(a["held_out"]))
    fig, ax = plt.subplots(figsize=(ONE_HALF, 55 * MM))
    ax.axhline(0.5, color=INK_3, lw=0.6, ls="--")
    offsets = np.linspace(-0.24, 0.24, len(NOVELTY_METHODS))
    for off, method, colour in zip(offsets, NOVELTY_METHODS, SERIES[: len(offsets)], strict=True):
        d = a[a.method == method].set_index("held_out").reindex(types)
        ax.scatter(
            np.arange(len(types)) + off,
            d.auroc,
            s=10,
            color=colour,
            edgecolor="white",
            lw=0.4,
            zorder=3,
            label=METHOD_LABELS[method],
        )
    ax.set_xticks(
        range(len(types)), [t.replace("gamma delta", "γδ") for t in types], rotation=30, ha="right"
    )
    ax.set_xlim(-0.5, len(types) - 0.5)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("AUROC for flagging the held-out type")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    save(fig, FIG / "fig4_open_set_auroc")


def _query_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.read_parquet(TAB / "predictions.parquet")
    qry = ad.read_h5ad(cfg.path("processed") / "query.h5ad", backed="r")
    return preds, qry.obs[["cell_subtype", "label_role"]].astype(str)


def fig_natural_novelty(preds: pd.DataFrame, obs: pd.DataFrame) -> None:
    p = preds[preds.method == "scanvi_latent_distance"].set_index("cell").join(obs)
    fig, ax = plt.subplots(figsize=(SINGLE, 58 * MM))
    groups = [("Shared subtypes", p.label_role == "shared")] + [
        (t, p.cell_subtype == t) for t in ("Anti-inflammatory", "BEST4+ Enterocytes", "Tuft cells")
    ]
    for (name, m), colour in zip(groups, SERIES[: len(groups)], strict=True):
        v = np.sort(p.loc[m, "novelty"].to_numpy())
        ax.plot(v, np.linspace(0, 1, len(v)), color=colour, label=f"{name} (n = {m.sum():,})")
    ax.set(
        xlabel="Distance to nearest reference cells (scANVI)",
        ylabel="Cumulative fraction",
        ylim=(0, 1.01),
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    save(fig, FIG / "fig5_natural_novelty")


def fig_selective_prediction(preds: pd.DataFrame, obs: pd.DataFrame) -> None:
    """Coverage-accuracy trade-off: abstain on the least confident cells, score the rest."""
    fig, ax = plt.subplots(figsize=(SINGLE, 58 * MM))
    rows = []
    for method, colour in zip(CLASSIFIERS, SERIES, strict=True):
        p = preds[preds.method == method].set_index("cell").join(obs)
        if p.empty:  # archived predictions may predate a method
            continue
        p = p[p.label_role == "shared"]
        correct = (p.pred_fine == p.cell_subtype).to_numpy()
        order = np.argsort(-p.confidence.to_numpy())
        correct = correct[order]
        coverage = np.arange(1, len(correct) + 1) / len(correct)
        accuracy = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        keep = slice(int(0.05 * len(correct)), None)  # tiny-coverage noise
        ax.plot(coverage[keep], accuracy[keep], color=colour, label=METHOD_LABELS[method])
        for cov in (0.5, 0.8, 1.0):
            i = min(int(cov * len(correct)) - 1, len(correct) - 1)
            rows.append({"method": method, "coverage": cov, "accuracy": accuracy[i]})
    ax.set(
        xlabel="Coverage (fraction of cells not abstained)",
        ylabel="Accuracy on covered cells",
        xlim=(0, 1.02),
    )
    ax.legend(loc="lower left")
    save(fig, FIG / "fig6_selective_prediction")
    pd.DataFrame(rows).to_csv(TAB / "selective_prediction.csv", index=False)


def main() -> None:
    apply_style()
    FIG.mkdir(exist_ok=True)
    fig_accuracy()
    fig_per_class()
    fig_open_set()
    preds, obs = _query_predictions()
    fig_natural_novelty(preds, obs)
    fig_selective_prediction(preds, obs)
    fig_umap()
    print(sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
