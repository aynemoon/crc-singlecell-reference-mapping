"""Figures for a single saved-output classifier comparison, separate from the archive."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scmap.plotting import (
    DOUBLE,
    METHOD_LABELS,
    MM,
    SEQUENTIAL,
    SERIES,
    apply_style,
    interval_points,
    panel_label,
    save,
)
from scmap.scanvi_knn_analysis import FOCUS, PAIR

COLOURS = (SERIES[3], SERIES[4])
METRICS = (("macro_f1", "Macro F1"), ("accuracy", "Accuracy"))


def plot_review(tables: dict[str, pd.DataFrame], directory: Path, *, preview_png=False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    apply_style()

    def save_review(fig, path):
        if preview_png:
            save(fig, path)
        else:
            fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)

    # Absolute performance and paired differences have distinct statistical meanings.
    for difference, filename in ((False, "performance"), (True, "paired_differences")):
        fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 65 * MM))
        for ax, letter, (metric, label) in zip(axes, "ab", METRICS, strict=True):
            data = tables["bootstrap"].loc[tables["bootstrap"].metric == metric].set_index("method")
            if difference:
                row = data.loc["knn_minus_native"]
                interval_points(
                    ax,
                    [0],
                    [row.value],
                    [row.ci_low],
                    [row.ci_high],
                    color=COLOURS[1],
                    horizontal=True,
                )
                ax.axvline(0, color="#777777", lw=0.7, ls="--")
                ax.set_yticks([])
                ax.set_xlabel(f"Difference in {label}\nkNN minus native")
            else:
                for i, (method, colour) in enumerate(zip(PAIR, COLOURS, strict=True)):
                    row = data.loc[method]
                    interval_points(ax, [i], [row.value], [row.ci_low], [row.ci_high], color=colour)
                ax.set_xticks(range(2), ["Native scANVI", "scANVI + kNN"])
                ax.set_ylabel(label)
                ax.margins(x=0.5)
            panel_label(ax, letter)
        fig.tight_layout(w_pad=3)
        save_review(fig, directory / filename)

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 80 * MM), sharey=True)
    patient = tables["groups"].query("factor == 'patient'")
    for ax, letter, (metric, label) in zip(axes, "ab", METRICS, strict=True):
        wide = patient.pivot(index="level", columns="method", values=metric).loc[:, list(PAIR)]
        for i, (_, row) in enumerate(wide.iterrows()):
            ax.plot(row.to_numpy(), [i, i], color="#bbbbbb", lw=0.7)
        for i, colour in enumerate(COLOURS):
            ax.scatter(
                wide.iloc[:, i],
                np.arange(len(wide)),
                color=colour,
                s=13,
                zorder=3,
                label=METHOD_LABELS[PAIR[i]],
            )
        ax.set_yticks(range(len(wide)), wide.index)
        ax.set_xlabel(label)
        panel_label(ax, letter)
    axes[0].invert_yaxis()
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout(w_pad=3)
    save_review(fig, directory / "patients")

    focus = [s for s in FOCUS if s in set(tables["subtypes"].subtype)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 96 * MM), sharey=True)
    for ax, letter, metric in zip(axes, "abc", ("precision", "recall", "f1"), strict=True):
        wide = (
            tables["subtypes"]
            .pivot(index="subtype", columns="method", values=metric)
            .reindex(focus)
        )
        for i, label in enumerate(focus):
            ax.plot(wide.loc[label, list(PAIR)], [i, i], color="#bbbbbb", lw=0.7)
        for method, colour in zip(PAIR, COLOURS, strict=True):
            ax.scatter(
                wide[method],
                np.arange(len(focus)),
                color=colour,
                s=14,
                label=METHOD_LABELS[method],
                zorder=3,
            )
        ax.set_yticks(range(len(focus)), focus)
        ax.set_xlim(-0.025, 1.025)
        ax.set_xlabel({"precision": "Precision", "recall": "Recall", "f1": "F1"}[metric])
        panel_label(ax, letter)
    axes[0].invert_yaxis()
    axes[2].legend(loc="lower left", bbox_to_anchor=(-0.2, -0.4))
    fig.tight_layout(w_pad=2)
    save_review(fig, directory / "subtype_tradeoffs")

    cms = [s for s in ("CMS1", "CMS2", "CMS3", "CMS4") if s in focus]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 80 * MM), layout="constrained")
    images = []
    for ax, letter, method in zip(axes, "ab", PAIR, strict=True):
        data = tables["confusions"].query("method == @method and truth in @cms")
        counts = (
            data.pivot(index="truth", columns="prediction", values="count")
            .fillna(0)
            .reindex(cms, fill_value=0)
        )
        matrix = counts.reindex(columns=cms, fill_value=0)
        matrix["Other"] = counts.sum(1) - matrix.sum(1)
        fraction = matrix.div(matrix.sum(1).replace(0, np.nan), axis=0).fillna(0)
        images.append(ax.imshow(fraction, vmin=0, vmax=1, cmap=SEQUENTIAL, aspect="auto"))
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                ax.text(
                    j,
                    i,
                    str(int(matrix.iloc[i, j])),
                    ha="center",
                    va="center",
                    color="white" if fraction.iloc[i, j] > 0.6 else "black",
                    fontsize=6,
                )
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(cms)), cms)
        ax.set_xlabel(f"Predicted subtype\n{METHOD_LABELS[method]}")
        ax.set_ylabel("True subtype")
        panel_label(ax, letter)
    fig.colorbar(images[0], ax=axes, label="Fraction within true subtype", shrink=0.7)
    save_review(fig, directory / "cms_confusion")

    support = tables["support"].set_index("subtype").loc[focus]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 96 * MM), sharey=True)
    axes[0].scatter(support.reference_cells, np.arange(len(focus)), color=COLOURS[0], s=18)
    axes[0].set_yticks(range(len(focus)), focus)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Labelled reference cells (log scale)")
    for offset, col, label, colour in (
        (-0.15, "reference_largest_donor_share", "Reference", COLOURS[0]),
        (0.15, "query_largest_donor_share", "Query", COLOURS[1]),
    ):
        axes[1].scatter(
            support[col], np.arange(len(focus)) + offset, color=colour, label=label, s=14
        )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Fraction from largest donor")
    axes[1].legend(loc="lower right")
    axes[0].invert_yaxis()
    for ax, letter in zip(axes, "ab", strict=True):
        panel_label(ax, letter)
    fig.tight_layout(w_pad=3)
    save_review(fig, directory / "reference_support")
