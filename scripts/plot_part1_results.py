"""
Render the Part 1 (Validation Experiments) results as publication-quality
seaborn figures for docs/RESULTS_PART1.md.

Reads:
  - results/metrics/activity_preference_results.json  (produced by run_activity_preference.py)
  - data/validation/activities.json                   (activity -> category map)

Writes three PNGs into results/figures/:
  - part1_baseline_emotion_correlations.png
  - part1_causal_steering.png
  - part1_category_preferences.png

Run (locally, no GPU / no torch needed — this only needs seaborn + pandas):
    uv run --no-project --with seaborn --with pandas --with matplotlib \
        python scripts/plot_part1_results.py

Or on a machine where the project env resolves:
    uv run --extra viz python scripts/plot_part1_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "metrics" / "activity_preference_results.json"
CONTROLLED = ROOT / "results" / "metrics" / "causal_steering_controlled.json"
CONTROLLED_LOW = ROOT / "results" / "metrics" / "causal_steering_controlled_lowalpha.json"
ACTIVITIES = ROOT / "data" / "validation" / "activities.json"
FIG_DIR = ROOT / "results" / "figures"

# Shared visual identity across all three figures.
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
ACCENT = "#4c6ef5"
GRID_KW = dict(alpha=0.35, linewidth=0.6)
CAVEAT_COLOR = "#8a8a99"


def _load() -> tuple[dict, dict]:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    activities = json.loads(ACTIVITIES.read_text(encoding="utf-8"))
    cat_by_id = {a["id"]: a["category"] for a in activities}
    return results, cat_by_id


def fig_baseline_correlations(results: dict) -> None:
    """All emotions ranked by how their probe activation correlates with the
    model's baseline activity preference (Elo). Negative = the model is averse
    to activities that evoke this emotion; positive = drawn to them."""
    rows = [r for r in results["correlations"] if r["correlation_with_elo"] is not None]
    df = pd.DataFrame(rows).sort_values("correlation_with_elo")
    df["sign"] = df["correlation_with_elo"].apply(lambda v: "aversive (−)" if v < 0 else "attractive (+)")

    fig, ax = plt.subplots(figsize=(9, 12))
    palette = {"aversive (−)": "#e8543f", "attractive (+)": "#2f9e6f"}
    sns.barplot(
        data=df, y="emotion", x="correlation_with_elo",
        hue="sign", palette=palette, dodge=False, ax=ax,
    )
    ax.axvline(0, color="#333", linewidth=1.0)
    ax.set_xlabel("Pearson r  (emotion-probe activation  vs.  activity Elo)")
    ax.set_ylabel("")
    ax.set_title("Which emotions predict the model's activity preferences?\n"
                 "Gemma-2-2B · 60 activities · full Elo round-robin", loc="left")
    ax.legend(title="", loc="upper right", frameon=True)
    ax.grid(axis="y", **GRID_KW)
    ax.margins(y=0.01)
    fig.tight_layout()
    out = FIG_DIR / "part1_baseline_emotion_correlations.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_causal_steering(results: dict) -> None:
    """Causal test: for each steered emotion, does turning its vector on shift
    preference toward the activities that evoke it (steering_effect) in line with
    that emotion's baseline preference-correlation? Carries an honest caveat."""
    pts = [
        e for e in results["steering_effects"]
        if e.get("baseline_correlation") is not None and e.get("steering_effect") is not None
    ]
    df = pd.DataFrame(pts)
    causal_r = results.get("causal_correlation")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.regplot(
        data=df, x="baseline_correlation", y="steering_effect",
        ax=ax, scatter=False, ci=None,
        line_kws=dict(color=CAVEAT_COLOR, linewidth=1.5, linestyle="--"),
    )
    sns.scatterplot(
        data=df, x="baseline_correlation", y="steering_effect",
        s=180, color=ACCENT, edgecolor="white", linewidth=1.5, ax=ax, zorder=5,
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["emotion"], (row["baseline_correlation"], row["steering_effect"]),
            xytext=(8, 6), textcoords="offset points", fontsize=12, fontweight="normal",
        )
    ax.axhline(0, color="#bbb", linewidth=0.8)
    ax.axvline(0, color="#bbb", linewidth=0.8)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 0.12)  # room for the right-most point's label (blissful)
    title = "Causal steering: does turning an emotion on shift preference?"
    if causal_r is not None:
        title += f"\ncross-emotion r = {causal_r:+.2f}   (paper reports +0.85 on Claude Sonnet 4.5)"
    ax.set_title(title, loc="left")
    ax.set_xlabel("Baseline preference-correlation of the emotion")
    ax.set_ylabel("Steering effect\n(Elo shift  vs.  emotion-probe score)")
    ax.grid(**GRID_KW)
    caveat = (
        "Caveat — not a clean +0.85 replication: Elo is zero-sum and comparisons are deterministic, so a\n"
        "steering-induced flip can only lower a high-Elo activity and raise a low one. That compression injects\n"
        "a term anti-correlated with the baseline correlation by construction, and n = 6 steered emotions is tiny.\n"
        "Read the strong negative slope as 'a real, large effect confounded by the round-robin design.'"
    )
    fig.text(0.01, -0.02, caveat, ha="left", va="top", fontsize=9, color=CAVEAT_COLOR)
    fig.tight_layout()
    out = FIG_DIR / "part1_causal_steering.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_category_preferences(results: dict, cat_by_id: dict) -> None:
    """Baseline Elo grouped by activity category — the model's revealed
    preference ordering over kinds of tasks."""
    rows = [
        {"category": cat_by_id[aid], "elo": elo}
        for aid, elo in results["baseline_elo"].items()
        if aid in cat_by_id
    ]
    df = pd.DataFrame(rows)
    order = df.groupby("category")["elo"].mean().sort_values(ascending=False).index.tolist()
    pretty = {
        "helpful": "helpful", "engaging": "engaging", "social": "social",
        "self_curiosity": "self-curiosity", "neutral": "neutral",
        "aversive": "aversive", "misaligned": "misaligned", "unsafe": "unsafe",
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    palette = sns.color_palette("vlag_r", n_colors=len(order))
    sns.boxplot(
        data=df, x="category", y="elo", order=order,
        hue="category", hue_order=order, palette=palette, legend=False,
        width=0.6, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=df, x="category", y="elo", order=order,
        color="#2a2a3a", size=5, alpha=0.55, jitter=0.18, ax=ax,
    )
    ax.axhline(1500, color="#888", linewidth=1.0, linestyle="--")
    ax.text(len(order) - 0.5, 1508, "start (1500)", color="#888", fontsize=10, ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([pretty[c] for c in order], rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Baseline Elo rating")
    ax.set_title("What kinds of activities does the model prefer?\n"
                 "higher Elo = chosen more often across the round-robin", loc="left")
    ax.grid(axis="x", **GRID_KW)
    fig.tight_layout()
    out = FIG_DIR / "part1_category_preferences.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_controlled_alpha_sweep() -> None:
    """Controlled (compression-immune) causal experiment across steering strength.
    Shows the negative result: as alpha rises, steering degenerates — every emotion
    produces the same generic preference flip (inter-emotion delta corr -> 1), so the
    causal correlation collapses to a mechanical -1 and is uninterpretable. No alpha
    reaches the 'emotion-specific' zone (delta corr < 0.3)."""
    if not (CONTROLLED.exists() and CONTROLLED_LOW.exists()):
        print("skip alpha-sweep figure (controlled JSON(s) not found)")
        return
    rows = []
    for path in (CONTROLLED_LOW, CONTROLLED):
        blocks = json.loads(path.read_text(encoding="utf-8"))["by_alpha"]
        for alpha, blk in blocks.items():
            rows.append({
                "alpha": float(alpha),
                "inter_emotion_delta_corr": blk["inter_emotion_delta_corr"],
                "causal_correlation": blk["causal_correlation"],
            })
    df = pd.DataFrame(rows).drop_duplicates("alpha").sort_values("alpha")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axhspan(-0.3, 0.3, color="#2f9e6f", alpha=0.10)
    ax.text(0.055, 0.18, "emotion-specific zone\n(|delta corr| < 0.3) — never reached",
            color="#2f9e6f", fontsize=11, va="top")
    sns.lineplot(data=df, x="alpha", y="inter_emotion_delta_corr", marker="o",
                 markersize=11, linewidth=2.5, color="#e8543f", ax=ax, label="inter-emotion delta corr\n(1 = every emotion identical = degenerate)")
    sns.lineplot(data=df, x="alpha", y="causal_correlation", marker="s",
                 markersize=10, linewidth=2.0, color=ACCENT, linestyle="--", ax=ax,
                 label="causal correlation\n(mechanically → −1 as steering degenerates)")
    ax.axhline(0, color="#bbb", linewidth=0.8)
    ax.set_xlabel("steering strength  α")
    ax.set_ylabel("correlation")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Controlled causal test (compression-immune): steering degenerates with α\n"
                 "Gemma-2-2B — no α produces an emotion-specific preference shift", loc="left")
    ax.legend(loc="center right", frameon=True, fontsize=9)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    out = FIG_DIR / "part1_causal_controlled_alpha_sweep.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results, cat_by_id = _load()
    fig_baseline_correlations(results)
    fig_causal_steering(results)
    fig_category_preferences(results, cat_by_id)
    fig_controlled_alpha_sweep()


if __name__ == "__main__":
    main()
