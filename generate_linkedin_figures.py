"""Generate publication-ready figures for an EndoSignature-Net LinkedIn carousel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_DIR = Path("docs/social-media")
BACKGROUND = "#F7F7F2"
INK = "#152238"
MUTED = "#5D6B7A"
TEAL = "#1B998B"
BLUE = "#3973B8"
ORANGE = "#E9A23B"
RED = "#D95D5D"
PALE_TEAL = "#DDF1ED"
PALE_BLUE = "#E2ECF7"


def _canvas():
    figure, axis = plt.subplots(figsize=(10.8, 13.5), dpi=100)
    figure.patch.set_facecolor(BACKGROUND)
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_position([0, 0, 1, 1])
    return figure, axis


def _save(figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_DIR / name,
        dpi=100,
        facecolor=figure.get_facecolor(),
        bbox_inches=None,
    )
    plt.close(figure)


def _title(axis, headline: str, subtitle: str) -> None:
    axis.text(
        0.08,
        0.92,
        headline,
        color=INK,
        fontsize=28,
        fontweight="bold",
        va="top",
    )
    axis.text(
        0.08,
        0.815,
        subtitle,
        color=MUTED,
        fontsize=15,
        va="top",
    )


def _footer(axis, page: str) -> None:
    axis.text(
        0.08,
        0.045,
        "EndoSignature-Net",
        color=MUTED,
        fontsize=12,
        va="bottom",
    )
    axis.text(0.92, 0.045, page, color=MUTED, fontsize=12, ha="right", va="bottom")


def pipeline_figure() -> None:
    """Create the carousel pipeline overview."""

    figure, axis = _canvas()
    _title(
        axis,
        "From single cells to\nexternal validation",
        "A patient-level, multi-cohort transcriptomics workflow",
    )
    steps = [
        ("1", "Single-cell RNA-seq", "QC, doublets and cell-family annotation"),
        ("2", "Patient-level pseudobulk", "Cells aggregated within each patient"),
        ("3", "12-gene signature", "Evidence integrated across technologies"),
        ("4", "Architecture benchmark", "Linear, nonlinear and neural models"),
        ("5", "Locked external testing", "No external retuning or score flipping"),
        ("6", "Interpretability", "Gene stability and transfer heterogeneity"),
    ]
    y_positions = [0.70, 0.595, 0.49, 0.385, 0.28, 0.175]
    for index, ((number, label, detail), y) in enumerate(zip(steps, y_positions)):
        fill = PALE_TEAL if index in {0, 1, 2} else PALE_BLUE
        box = FancyBboxPatch(
            (0.13, y - 0.042),
            0.74,
            0.084,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0,
            facecolor=fill,
        )
        axis.add_patch(box)
        axis.text(
            0.18,
            y,
            number,
            fontsize=18,
            fontweight="bold",
            color=TEAL if index < 3 else BLUE,
            va="center",
            ha="center",
        )
        axis.text(0.24, y + 0.014, label, fontsize=17, fontweight="bold", color=INK)
        axis.text(0.24, y - 0.019, detail, fontsize=12.5, color=MUTED)
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.049),
                xytext=(0.5, y - 0.049),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.4},
            )
    _footer(axis, "1 / 3")
    _save(figure, "linkedin_01_pipeline.png")


def auc_comparison_figure() -> None:
    """Create a contextualized comparison of discrimination results."""

    figure, axis = _canvas()
    _title(
        axis,
        "Strong internally.\nHeterogeneous externally.",
        "ROC-AUC across patient-level evaluations",
    )
    labels = [
        "Internal repeated CV",
        "GSE212787 external",
        "Other-pathology stress test",
        "GSE153740 external",
    ]
    values = [0.860, 0.738, 0.560, 0.125]
    colors = [BLUE, TEAL, ORANGE, RED]
    notes = [
        "77 cases · 34 clean controls",
        "13 patients · 95% CI 0.429–0.976",
        "77 cases · 37 other pathologies",
        "8 patients · 95% CI 0.000–0.500",
    ]
    bar_axis = figure.add_axes([0.13, 0.20, 0.76, 0.53], facecolor=BACKGROUND)
    positions = list(range(len(labels)))[::-1]
    bar_axis.barh(positions, values, color=colors, height=0.48)
    bar_axis.axvline(0.5, color=MUTED, linewidth=1.2, linestyle="--")
    bar_axis.text(0.505, 3.58, "chance", color=MUTED, fontsize=11, va="bottom")
    for y, value, label, note in zip(positions, values, labels, notes):
        bar_axis.text(0.01, y + 0.14, label, color=INK, fontsize=14, fontweight="bold")
        bar_axis.text(0.01, y - 0.18, note, color=MUTED, fontsize=10.5)
        bar_axis.text(
            min(value + 0.025, 0.94),
            y,
            f"{value:.3f}",
            color=INK,
            fontsize=17,
            fontweight="bold",
            va="center",
        )
    bar_axis.set_xlim(0, 1)
    bar_axis.set_ylim(-0.6, 3.75)
    bar_axis.set_yticks([])
    bar_axis.set_xlabel("ROC-AUC", color=MUTED, fontsize=12)
    bar_axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bar_axis.tick_params(axis="x", colors=MUTED, labelsize=11)
    bar_axis.grid(axis="x", color="#D8DDE3", linewidth=0.8)
    bar_axis.set_axisbelow(True)
    for spine in bar_axis.spines.values():
        spine.set_visible(False)
    axis.text(
        0.08,
        0.12,
        "Takeaway: internal performance alone did not establish a robust biomarker.",
        color=INK,
        fontsize=13,
        fontweight="bold",
    )
    _footer(axis, "2 / 3")
    _save(figure, "linkedin_02_auc_comparison.png")


def stability_message_figure() -> None:
    """Create the central internal-versus-external stability message."""

    figure, axis = _canvas()
    _title(
        axis,
        "Stable model ≠\nuniversal biomarker",
        "The key interpretability result",
    )
    cards = [
        (
            0.10,
            0.47,
            PALE_BLUE,
            BLUE,
            "11 / 12",
            "internally stable",
            "Coefficient signs were stable in ≥95%\nof 1,000 patient bootstraps.",
        ),
        (
            0.53,
            0.47,
            PALE_TEAL,
            TEAL,
            "2 / 12",
            "externally consistent",
            "Only PDZD2 and ACSS2 matched the\nfrozen direction in both cohorts.",
        ),
    ]
    for x, y, fill, accent, value, label, detail in cards:
        card = FancyBboxPatch(
            (x, y),
            0.37,
            0.25,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=0,
            facecolor=fill,
        )
        axis.add_patch(card)
        axis.text(x + 0.185, y + 0.17, value, ha="center", color=accent, fontsize=34, fontweight="bold")
        axis.text(x + 0.185, y + 0.115, label, ha="center", color=INK, fontsize=16, fontweight="bold")
        axis.text(x + 0.185, y + 0.052, detail, ha="center", color=MUTED, fontsize=11.2, linespacing=1.4)
    axis.annotate(
        "",
        xy=(0.53, 0.595),
        xytext=(0.47, 0.595),
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.8},
    )
    axis.text(
        0.50,
        0.35,
        "Why did transfer break?",
        color=INK,
        fontsize=18,
        fontweight="bold",
        ha="center",
    )
    factors = [
        "Menstrual phase",
        "Tissue composition",
        "Disease phenotype",
        "Control definition",
        "Measurement platform",
    ]
    for index, factor in enumerate(factors):
        row = index // 2
        column = index % 2
        x = 0.17 + column * 0.37 if index < 4 else 0.355
        y = 0.285 - row * 0.06 if index < 4 else 0.165
        pill = FancyBboxPatch(
            (x, y - 0.022),
            0.29,
            0.044,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            linewidth=1,
            edgecolor="#C9D0D8",
            facecolor=BACKGROUND,
        )
        axis.add_patch(pill)
        axis.text(x + 0.145, y, factor, ha="center", va="center", color=MUTED, fontsize=11.5)
    axis.text(
        0.50,
        0.105,
        "The negative replication was analyzed—not tuned away.",
        color=INK,
        fontsize=13,
        fontweight="bold",
        ha="center",
    )
    _footer(axis, "3 / 3")
    _save(figure, "linkedin_03_stability_vs_transfer.png")


def main() -> None:
    pipeline_figure()
    auc_comparison_figure()
    stability_message_figure()
    print(f"Generated 3 LinkedIn figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
