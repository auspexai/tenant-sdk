"""Render a DriftBenchmark report as the ladder-style PNG (the presentation
pattern the ratified standard uses: per-probe bars in envelope units, the
calibrated 1-EU threshold, per-feature dots, byte-divergence and within-run
divergence as neutral text — never folded into the bars).

Needs matplotlib — the `[analysis]` extra; the import is guarded so the core
benchmark stays dependency-free.
"""

from __future__ import annotations

from auspexai_tenant.benchmark import DriftBenchmark

# The validated categorical palette (dataviz reference instance; worst adjacent
# CVD dE 47.2 in the order used). Feature dots take slots in fixed order.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_BAR = "#86b6ef"  # sequential blue step 250 (ordinal-legal on the light surface)
_FEATURE_SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]


def plot_report(
    report: DriftBenchmark,
    out_path: str,
    *,
    title: str = "Drift Benchmark",
    subtitle: str | None = None,
) -> str:
    """Write the ladder PNG for one scored comparison; returns `out_path`."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover — exercised only without the extra
        raise ImportError(
            "plotting needs matplotlib — install the analysis extra: "
            "pip install 'auspexai-tenant[analysis]'"
        ) from e

    probes = list(report.probes)
    n = max(len(probes), 1)
    fig, ax = plt.subplots(figsize=(9.2, 1.5 + 0.9 * n), dpi=160)
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    # Fixed feature→slot assignment across the whole report (never cycled).
    feature_names = sorted({f.feature for p in probes for f in p.features if f.eu is not None})
    slot = {name: _FEATURE_SLOTS[i % len(_FEATURE_SLOTS)] for i, name in enumerate(feature_names)}

    max_eu = max(
        [p.peak_eu or 0.0 for p in probes]
        + [f.eu for p in probes for f in p.features if f.eu is not None]
        + [1.0]
    )
    xmax = max(max_eu * 1.25, 1.6)

    for y, p in enumerate(reversed(probes)):
        peak = p.peak_eu or 0.0
        ax.barh(y, peak, height=0.32, color=_BAR, zorder=2)
        label = f"{peak:.2f} EU" if p.peak_eu is not None else "n/a"
        ax.text(
            peak + xmax * 0.015,
            y + 0.28,
            label,
            va="center",
            ha="left",
            fontsize=10,
            color=_INK,
            fontweight="bold",
            zorder=4,
        )
        for f in p.features:
            if f.eu is None:
                continue
            ax.scatter(
                f.eu, y, s=80, color=slot[f.feature], edgecolors=_SURFACE, linewidths=2, zorder=3
            )
        annotations = []
        if p.byte_divergence_rate is not None:
            annotations.append(f"byte-div {p.byte_divergence_rate:.0%}")
        diverged = (report.diverged_by_key or {}).get(p.key)
        if diverged:
            annotations.append(f"diverged {diverged}")
        if annotations:
            ax.text(
                1.015,
                y,
                " · ".join(annotations),
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=9,
                color=_INK2,
            )

    ax.axvline(1.0, color=_INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        1.0 + xmax * 0.01,
        n - 0.42,
        "calibrated envelope (1 EU)",
        fontsize=8.5,
        color=_INK2,
        ha="left",
        va="bottom",
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels([p.key for p in reversed(probes)] or [""], fontsize=10, color=_INK)
    ax.set_ylim(-0.55, n - 0.3)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(
        "drift (envelope units — each feature's shift ÷ its calibrated tolerance; <1 = within noise)",
        fontsize=9.5,
        color=_INK2,
    )
    ax.tick_params(colors=_INK2, labelsize=9)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e6e5e1")

    if feature_names:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=8,
                markerfacecolor=slot[name],
                markeredgecolor=_SURFACE,
                markeredgewidth=1.5,
                label=name,
            )
            for name in feature_names
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            frameon=False,
            fontsize=8.5,
            title="per-feature (median EU)",
            title_fontsize=8.5,
            labelcolor=_INK,
        )

    fig.text(0.03, 0.97, title, fontsize=12.5, color=_INK, fontweight="bold", ha="left", va="top")
    parts = []
    if report.peak_eu is not None:
        parts.append(f"peak {report.peak_eu:.2f} EU")
    if report.breadth is not None:
        parts.append(f"{report.breadth:.0%} of probes beyond envelope")
    if report.byte_divergence_rate is not None:
        parts.append(f"byte-divergence {report.byte_divergence_rate:.0%} (separate overlay)")
    if report.diverged_units_total:
        parts.append(f"⚠ {report.diverged_units_total} diverged unit(s) — signed, not EU-scorable")
    line2 = subtitle or " · ".join(parts) or "no scored probes"
    fig.text(0.03, 0.905, line2, fontsize=9, color=_INK2, ha="left", va="top")

    fig.subplots_adjust(left=0.16, right=0.84, top=0.80, bottom=0.2)
    fig.savefig(out_path, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
