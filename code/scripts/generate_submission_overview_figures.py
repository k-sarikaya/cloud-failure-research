from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless/non-GUI environments (e.g., CI or minimal Python installs).
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


FRAMEWORK_STEPS = [
    ("M/M/1 Queue\nTheory", "queue saturation\nand delay"),
    ("Proxy-Based\nGrounding", "358 source\nincidents"),
    ("Load-Dependent\nHazard", r"$\lambda(\rho)=\lambda_{base} e^{\alpha(\rho-1)}$"),
    ("Graph State\nEncoder (GAT)", "topology-aware\nnode embeddings"),
    ("SAC Actor /\nTwin Critics", "continuous-control\npolicy learning"),
    ("Continuous\nThrottling Policy", "graded load\nshedding"),
    ("Cascade\nSimulator", "redistribution,\nfailure, recovery"),
    ("Paired Eval.\n+ Latency", "40 paired\nscenarios"),
]


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT.parent
FIG_DIRS = [ROOT / "main_text" / "figures", PACKAGE_ROOT / "main_text" / "figures"]
for _fig_dir in FIG_DIRS:
    _fig_dir.mkdir(parents=True, exist_ok=True)


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 12,
            "axes.titlesize": 18,
            "figure.dpi": 160,
        }
    )


def make_framework_figure(out_base: Path) -> None:
    _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(18, 6.2))
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 6)
    ax.axis("off")

    palette = [
        "#4C78A8",
        "#5E8BC0",
        "#A05195",
        "#72B7B2",
        "#E45756",
        "#54A24B",
        "#F58518",
        "#9D755D",
    ]
    xs = [0.35, 2.45, 4.55, 6.65, 8.75, 10.85, 12.95, 15.05]
    y = 2.15
    width = 1.65
    height = 1.7

    ax.text(
        8.0,
        5.45,
        "Framework Overview: Topology-Aware Continuous Control for Cascade Prevention",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
    )
    ax.text(
        8.0,
        4.85,
        "From queueing-derived hazard estimation to graph-aware SAC control and reproducible benchmark evaluation",
        ha="center",
        va="center",
        fontsize=13,
        color="#444444",
    )

    for idx, ((title, subtitle), x) in enumerate(zip(FRAMEWORK_STEPS, xs)):
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.2,rounding_size=0.18",
            linewidth=2.0,
            facecolor=palette[idx],
            edgecolor="#2F2F2F",
            alpha=0.96,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + 1.08,
            title,
            ha="center",
            va="center",
            color="white",
            fontsize=14.2,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y + 0.38,
            subtitle,
            ha="center",
            va="center",
            color="white",
            fontsize=10.9,
        )
        if idx < len(xs) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.08, y + height / 2),
                (xs[idx + 1] - 0.08, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=22,
                linewidth=2.4,
                color="#444444",
            )
            ax.add_patch(arrow)

    foundation = FancyBboxPatch(
        (0.55, 1.0),
        5.65,
        0.65,
        boxstyle="round,pad=0.16,rounding_size=0.14",
        linewidth=1.3,
        facecolor="#EEF3F8",
        edgecolor="#8AA1B1",
    )
    control = FancyBboxPatch(
        (5.95, 1.0),
        5.9,
        0.65,
        boxstyle="round,pad=0.16,rounding_size=0.14",
        linewidth=1.3,
        facecolor="#F8EFEA",
        edgecolor="#C58F62",
    )
    evaluation = FancyBboxPatch(
        (11.85, 1.0),
        3.8,
        0.65,
        boxstyle="round,pad=0.16,rounding_size=0.14",
        linewidth=1.3,
        facecolor="#F6F1EB",
        edgecolor="#9D755D",
    )
    ax.add_patch(foundation)
    ax.add_patch(control)
    ax.add_patch(evaluation)
    ax.text(
        3.38,
        1.33,
        "Physics + data grounding",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#2E5676",
    )
    ax.text(
        8.9,
        1.33,
        "Topology-aware control stack",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#954F20",
    )
    ax.text(
        13.75,
        1.33,
        "Verification outputs",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#6A4D39",
    )

    fig.tight_layout(pad=0.6)
    for fig_dir in FIG_DIRS:
        target = fig_dir / out_base.name
        fig.savefig(target.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _draw_node(ax, xy, radius, facecolor, label, edgecolor="#333333", linewidth=2.2):
    circle = Circle(xy, radius=radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=16,
        fontweight="bold",
    )


def _draw_network_edges(ax, nodes, edges, color="#8C8C8C", linestyle="-", linewidth=3.0, alpha=1.0):
    for a, b in edges:
        ax.plot(
            [nodes[a][0], nodes[b][0]],
            [nodes[a][1], nodes[b][1]],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=0,
        )


def make_cascade_figure(out_base: Path) -> None:
    _setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 6.35))
    fig.subplots_adjust(wspace=0.14, bottom=0.22)

    nodes = {
        0: (0.7, 2.0),
        1: (1.7, 3.15),
        2: (1.75, 0.95),
        3: (3.1, 2.0),
        4: (4.2, 3.15),
        5: (4.1, 0.95),
        6: (5.55, 2.0),
        7: (6.75, 2.0),
    }
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (2, 5), (3, 5), (3, 6), (4, 6), (5, 6), (6, 7)]

    healthy = "#2ECC71"
    stressed = "#F39C12"
    failed = "#E74C3C"
    throttled = "#3498DB"

    panel_specs = [
        (
            "(a) Normal Operation",
            {0: "56%", 1: "47%", 2: "67%", 3: "41%", 4: "63%", 5: "42%", 6: "52%", 7: "55%"},
            {idx: healthy for idx in nodes},
            "All nodes operate below the overload region.",
        ),
        (
            "(b) Uncontrolled Cascade",
            {0: "FAIL", 1: "FAIL", 2: "FAIL", 3: "92%", 4: "88%", 5: "FAIL", 6: "FAIL", 7: "65%"},
            {0: failed, 1: failed, 2: failed, 3: stressed, 4: stressed, 5: failed, 6: failed, 7: healthy},
            "A single overload shock triggers secondary failures.",
        ),
        (
            "(c) Graceful Throttling /\nControlled Load Shedding",
            {0: "56%", 1: "50%", 2: "65%", 3: "T25%", 4: "63%", 5: "49%", 6: "58%", 7: "54%"},
            {0: healthy, 1: healthy, 2: healthy, 3: throttled, 4: healthy, 5: healthy, 6: healthy, 7: healthy},
            "Partial throttling absorbs the shock without hard removal.",
        ),
    ]

    for ax, (title, labels, colors, subtitle) in zip(axes, panel_specs):
        ax.set_xlim(0.0, 7.5)
        ax.set_ylim(-0.05, 4.25)
        ax.axis("off")
        title_y = 3.94 if title.startswith("(c)") else 3.78
        title_fs = 16 if title.startswith("(c)") else 18
        ax.text(
            3.75,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            linespacing=1.06,
        )
        _draw_network_edges(ax, nodes, edges)
        for idx, xy in nodes.items():
            facecolor = colors[idx]
            edgecolor = "#1F4E79" if (title.startswith("(c)") and idx == 3) else "#333333"
            linewidth = 3.2 if (title.startswith("(c)") and idx == 3) else 2.2
            _draw_node(ax, xy, 0.44, facecolor, labels[idx], edgecolor=edgecolor, linewidth=linewidth)

        if title.startswith("(b)"):
            ax.text(
                3.75,
                0.32,
                "5/8 nodes fail after abrupt load redistribution",
                ha="center",
                va="center",
                fontsize=14,
                color=failed,
                fontweight="bold",
            )
        elif title.startswith("(c)"):
            ax.annotate(
                "25% controlled shedding reduces neighbor exposure",
                xy=nodes[3],
                xytext=(3.75, 0.06),
                ha="center",
                va="center",
                fontsize=11.0,
                color="#1F4E79",
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#1F4E79",
                    "lw": 2.0,
                    "connectionstyle": "arc3,rad=-0.35",
                },
            )
            ax.text(
                4.15,
                0.20,
                "8/8 nodes remain operational",
                ha="center",
                va="center",
                fontsize=13,
                color="#239B56",
                fontweight="bold",
            )

        ax.text(3.75, -0.12, subtitle, ha="center", va="top", fontsize=12, color="#444444")

    legend_ax = fig.add_axes([0.21, 0.02, 0.58, 0.1])
    legend_ax.axis("off")
    legend_items = [
        ("Healthy", healthy),
        ("Overloaded", stressed),
        ("Failed", failed),
        ("Controlled throttle", throttled),
    ]
    for idx, (label, color) in enumerate(legend_items):
        x = 0.02 + idx * 0.25
        patch = FancyBboxPatch(
            (x, 0.25),
            0.055,
            0.28,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.6,
            transform=legend_ax.transAxes,
        )
        legend_ax.add_patch(patch)
        legend_ax.text(x + 0.075, 0.39, label, transform=legend_ax.transAxes, va="center", fontsize=13)

    for fig_dir in FIG_DIRS:
        target = fig_dir / out_base.name
        fig.savefig(target.with_suffix(".png"), bbox_inches="tight")
        fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_framework_figure(ROOT / "main_text" / "figures" / "fig1_framework")
    make_cascade_figure(ROOT / "main_text" / "figures" / "fig2_cascade")


if __name__ == "__main__":
    main()
