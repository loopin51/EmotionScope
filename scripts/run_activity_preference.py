"""
CLI: activity-preference / Elo experiment — reproduces paper Fig. 4.

Usage:
    uv run python scripts/run_activity_preference.py \
        --vectors results/vectors/google_gemma-2-2b-it.pt \
        --steer-emotions blissful hostile desperate calm loving

Causal metric (why not a mean Elo delta):
    Elo ratings are zero-sum — the 60 activities always sum to 60 * 1500 both
    before and after steering — so the MEAN of (steered - baseline) is
    identically 0 and carries no signal. The per-activity shifts are non-zero;
    they just cancel in the mean. So the per-emotion `steering_effect` is instead
    the Pearson correlation, across activities, between each activity's Elo shift
    under steering toward emotion e and that activity's baseline emotion-e probe
    score: "does steering toward e raise preference for the activities that
    intrinsically evoke e?" The cross-emotion `causal_correlation` then relates
    each emotion's baseline preference-correlation to its steering_effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from emotion_scope.config import DATA_DIR, FIGURES_DIR, METRICS_DIR
from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.preference import run_full_pairwise
from emotion_scope.probe import EmotionProbe
from emotion_scope.steer import Steerer, middle_third_layers
from emotion_scope.tracking import init_run


def load_activities(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_pearsonr(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r, or None when it is undefined (fewer than 2 points, or either
    series has zero variance — e.g. steering that left every rating untouched)."""
    if len(xs) < 2:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    r, _ = pearsonr(xs, ys)
    if r != r:  # NaN guard
        return None
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Activity-preference / Elo experiment")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--activities", default=str(DATA_DIR / "validation" / "activities.json"))
    parser.add_argument("--steer-emotions", nargs="+", default=["blissful", "hostile", "desperate", "calm"])
    parser.add_argument("--steer-alpha", type=float, default=0.5)
    parser.add_argument("--out", default=str(METRICS_DIR / "activity_preference_results.json"))
    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    vectors = saved["vectors"]
    model_name = saved["model_info"]["model_name"]
    activities = load_activities(Path(args.activities))

    model, tokenizer, backend, info = load_model(model_name=model_name, backend="huggingface", run_smoke_test=False)
    probe = EmotionProbe(model, tokenizer, backend, info, vectors)
    steerer = Steerer(model, tokenizer, backend, info)
    layers = middle_third_layers(info["n_layers"])
    avg_norms = steerer.compute_avg_norms(layers=layers)

    tracker = init_run(
        part="part1", job_type="activity-preference",
        config={"model": model_name, "n_activities": len(activities), "steer_emotions": args.steer_emotions},
    )

    print(f"[preference] Running baseline round-robin over {len(activities)} activities "
          f"({len(activities) * (len(activities) - 1)} comparisons)...")
    baseline = run_full_pairwise(model, tokenizer, activities)

    print("[preference] Measuring emotion-probe activation per activity...")
    activity_scores: dict[str, dict[str, float]] = {}
    for a in activities:
        state = probe.analyze(a["text"])
        activity_scores[a["id"]] = state.scores

    elo_values = [baseline.ratings[a["id"]] for a in activities]
    correlation_rows = []
    for emotion_name in vectors.keys():
        emotion_values = [activity_scores[a["id"]][emotion_name] for a in activities]
        r = safe_pearsonr(emotion_values, elo_values)
        correlation_rows.append({"emotion": emotion_name, "correlation_with_elo": r})
    correlation_rows.sort(key=lambda row: (row["correlation_with_elo"] is None, row["correlation_with_elo"]))

    print("\n[preference] Top 5 negatively correlated emotions (Elo drops when active):")
    for row in correlation_rows[:5]:
        print(f"  {row['emotion']:15s} r={row['correlation_with_elo']:+.3f}")
    print("[preference] Top 5 positively correlated emotions (Elo rises when active):")
    for row in correlation_rows[-5:]:
        print(f"  {row['emotion']:15s} r={row['correlation_with_elo']:+.3f}")

    print(f"\n[preference] Running steered round-robins for: {args.steer_emotions}")
    steering_effects = []
    steered_elo_by_emotion: dict[str, dict[str, float]] = {}
    for emotion_name in args.steer_emotions:
        if emotion_name not in vectors:
            print(f"  skipping '{emotion_name}' — not in vectors file")
            continue
        steered = run_full_pairwise(
            model, tokenizer, activities,
            steer_vector=vectors[emotion_name], steer_alpha=args.steer_alpha,
            steer_layers=layers, steer_avg_norms=avg_norms, backend=backend,
        )
        steered_elo_by_emotion[emotion_name] = steered.ratings
        # Per-activity Elo shift under steering toward this emotion. These sum to
        # zero (Elo is zero-sum) but vary per activity — that variation is the signal.
        deltas = [steered.ratings[a["id"]] - baseline.ratings[a["id"]] for a in activities]
        probe_e = [activity_scores[a["id"]][emotion_name] for a in activities]
        # steering_effect: does steering toward e lift the Elo of e-congruent activities?
        steering_effect = safe_pearsonr(probe_e, deltas)
        max_abs_delta = max(abs(d) for d in deltas)
        correlation = next((r["correlation_with_elo"] for r in correlation_rows if r["emotion"] == emotion_name), None)
        steering_effects.append({
            "emotion": emotion_name,
            "steering_effect": steering_effect,
            "max_abs_elo_delta": max_abs_delta,
            "baseline_correlation": correlation,
        })
        eff_str = f"{steering_effect:+.3f}" if steering_effect is not None else "  n/a"
        corr_str = f"{correlation:+.3f}" if correlation is not None else "n/a"
        print(f"  {emotion_name:15s} steering_effect r={eff_str}  "
              f"(max |Elo delta|={max_abs_delta:6.1f}, baseline corr r={corr_str})")

    # Cross-emotion causal test: emotions whose probe correlates with preference,
    # do they also steer preference toward their congruent activities?
    causal_pairs = [
        (e["baseline_correlation"], e["steering_effect"])
        for e in steering_effects
        if e["baseline_correlation"] is not None and e["steering_effect"] is not None
    ]
    causal_r = None
    if len(causal_pairs) >= 2:
        causal_r = safe_pearsonr([p[0] for p in causal_pairs], [p[1] for p in causal_pairs])
        if causal_r is not None:
            print(f"\n[preference] Correlation between baseline-correlation and steering_effect: r={causal_r:+.3f}")
            print("  (paper reports r=0.85 for the analogous relationship on Claude Sonnet 4.5)")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pts = [e for e in steering_effects
                if e["baseline_correlation"] is not None and e["steering_effect"] is not None]
    if plot_pts:
        ax.scatter([e["baseline_correlation"] for e in plot_pts], [e["steering_effect"] for e in plot_pts])
        for e in plot_pts:
            ax.annotate(e["emotion"], (e["baseline_correlation"], e["steering_effect"]))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Emotion-probe correlation with baseline Elo")
    ax.set_ylabel("Steering effect (Elo-shift vs. emotion-probe correlation)")
    ax.set_title("Steering effect vs. baseline preference correlation")
    fig_path = FIGURES_DIR / "activity_preference_scatter.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[preference] Saved scatter plot to {fig_path}")

    results = {
        "model": model_name,
        "n_activities": len(activities),
        "steer_alpha": args.steer_alpha,
        "baseline_elo": baseline.ratings,
        "correlations": correlation_rows,
        "steering_effects": steering_effects,
        "steered_elo": steered_elo_by_emotion,
        "causal_correlation": causal_r,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[preference] Saved full results to {args.out}")

    tracker.log_table(
        "elo_ratings",
        columns=["activity_id", "category", "elo"],
        data=[[a["id"], a["category"], baseline.ratings[a["id"]]] for a in activities],
    )
    tracker.log_table(
        "emotion_correlations",
        columns=["emotion", "correlation_with_elo"],
        data=[[r["emotion"], r["correlation_with_elo"]] for r in correlation_rows],
    )
    if steering_effects:
        tracker.log_table(
            "steering_effects",
            columns=["emotion", "steering_effect", "max_abs_elo_delta", "baseline_correlation"],
            data=[[e["emotion"], e["steering_effect"], e["max_abs_elo_delta"], e["baseline_correlation"]]
                  for e in steering_effects],
        )
    if causal_r is not None:
        tracker.log_metrics({"causal_correlation": causal_r})
    tracker.log_artifact(str(fig_path), name="activity_preference_scatter", type="figure")
    tracker.finish()


if __name__ == "__main__":
    main()
