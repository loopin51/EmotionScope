"""
Controlled causal-steering re-experiment — isolate "design confound" from
"model/method limitation" behind the round-robin's r = -0.82 (opposite to the
paper's +0.85).

Two changes vs. scripts/run_activity_preference.py, each targeting one suspect:

1. DESIGN (compression). The round-robin Elo is zero-sum, so a steering-induced
   flip can only lower a high-Elo activity and raise a low one — injecting a
   term anti-correlated with baseline preference by construction. Fix: score
   each activity's preference INDEPENDENTLY as its win-rate against a FIXED
   neutral reference set (the 8 'neutral' activities), instead of a closed
   round-robin. Per-activity win-rates are decoupled -> no zero-sum compression.

2. METHOD/MODEL (over-steering). At alpha=0.5 the 2B model's steered deltas were
   found to be IDENTICAL across opposite emotions (blissful == hostile == ...),
   i.e. steering was not emotion-specific. Fix: sweep alpha and, at each, report
   the mean inter-emotion delta correlation (near 1.0 = generic/degenerate;
   near 0 = emotion-specific).

For each alpha we report:
  - causal_correlation = corr over emotions of (baseline_correlation, steering_effect)
    where steering_effect(e) = corr over activities of (probe_e, win-rate delta)
  - inter_emotion_delta_corr: mean pairwise corr of per-activity deltas across
    emotions (the emotion-specificity diagnostic)
  - a non-zero-sum sanity number (sum of deltas; ~0 only by coincidence now)

Interpretation — READ THE GATE FIRST:
  causal_correlation is ONLY interpretable when inter_emotion_delta_corr is LOW
  (say < ~0.3). When steering is degenerate (delta-corr ~1, every emotion produces
  the same generic delta D), causal_correlation just measures whether D happens to
  align with baseline preference — it can be strongly POSITIVE OR NEGATIVE and means
  NOTHING about emotion-specific causation. A positive sign at high delta-corr is a
  spurious leak, NOT "design artifact confirmed". The `delta_vs_baseline_corr` field
  reports corr(delta, baseline win-rate) per emotion as a direct read on that leak.

  So:
  - delta-corr LOW + causal sign positive  -> genuine: the -0.82 was a DESIGN artifact
  - delta-corr HIGH (~1)                    -> steering not emotion-specific (METHOD/model);
                                               causal_correlation is uninterpretable either sign
  - lower alpha lowers delta-corr           -> over-steering was (part of) the block

Usage:
    uv run python scripts/run_causal_steering_controlled.py \
        --vectors results/vectors/google_gemma-2-2b-it.pt \
        --steer-emotions blissful hostile desperate calm loving furious \
        --alphas 0.3 0.5
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from emotion_scope.config import DATA_DIR, METRICS_DIR
from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.preference import compare_activities
from emotion_scope.probe import EmotionProbe
from emotion_scope.steer import Steerer, middle_third_layers
from emotion_scope.tracking import init_run

try:
    from scipy.stats import pearsonr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required: `uv run ... scripts/run_causal_steering_controlled.py`") from exc


def safe_pearsonr(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    r, _ = pearsonr(xs, ys)
    return None if r != r else float(r)


def win_rate_vs_refs(model, tokenizer, target, refs, backend, steer_kwargs) -> float:
    """Fraction of comparisons `target` wins against every reference, both
    orderings. Independent of the other targets -> no zero-sum compression."""
    wins = 0
    n = 0
    for ref in refs:
        if ref["id"] == target["id"]:
            continue
        # ordering 1: A = target, B = ref  -> target wins iff "A"
        if compare_activities(model, tokenizer, target["text"], ref["text"], backend=backend, **steer_kwargs) == "A":
            wins += 1
        # ordering 2: A = ref, B = target  -> target wins iff "B"
        if compare_activities(model, tokenizer, ref["text"], target["text"], backend=backend, **steer_kwargs) == "B":
            wins += 1
        n += 2
    return wins / n if n else 0.0


def score_all(model, tokenizer, targets, refs, backend, steer_kwargs) -> dict:
    return {t["id"]: win_rate_vs_refs(model, tokenizer, t, refs, backend, steer_kwargs) for t in targets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled (compression-immune) causal-steering experiment")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--activities", default=str(DATA_DIR / "validation" / "activities.json"))
    # ~14 emotions spanning the full valence range (not a bimodal extremes-only set),
    # so the cross-emotion causal correlation reflects a genuine trend, not two clusters.
    parser.add_argument("--steer-emotions", nargs="+", default=[
        "angry", "hostile", "desperate", "guilty", "afraid", "irritated", "lonely",
        "loving", "hopeful", "happy", "calm", "content", "serene", "blissful",
    ])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5])
    parser.add_argument("--reference-category", default="neutral")
    parser.add_argument("--out", default=str(METRICS_DIR / "causal_steering_controlled.json"))
    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    vectors = saved["vectors"]
    model_name = saved["model_info"]["model_name"]
    activities = json.loads(Path(args.activities).read_text(encoding="utf-8"))
    refs = [a for a in activities if a["category"] == args.reference_category]
    targets = activities  # score every activity (incl. neutrals) against the neutral refs
    print(f"[controlled] {len(targets)} targets vs {len(refs)} fixed '{args.reference_category}' references")

    model, tokenizer, backend, info = load_model(model_name=model_name, backend="huggingface", run_smoke_test=False)
    probe = EmotionProbe(model, tokenizer, backend, info, vectors)
    steerer = Steerer(model, tokenizer, backend, info)
    layers = middle_third_layers(info["n_layers"])
    avg_norms = steerer.compute_avg_norms(layers=layers)

    tracker = init_run(
        part="part1", job_type="causal-steering-controlled",
        config={"model": model_name, "alphas": args.alphas, "steer_emotions": args.steer_emotions,
                "reference_category": args.reference_category},
    )

    print("[controlled] measuring emotion-probe activation per activity...")
    activity_scores = {t["id"]: probe.analyze(t["text"]).scores for t in targets}

    print("[controlled] baseline win-rates vs reference set (no steering)...")
    baseline_score = score_all(model, tokenizer, targets, refs, backend, {})

    tids = [t["id"] for t in targets]
    baseline_correlation = {}
    for emotion in vectors:
        probe_vals = [activity_scores[i][emotion] for i in tids]
        base_vals = [baseline_score[i] for i in tids]
        baseline_correlation[emotion] = safe_pearsonr(probe_vals, base_vals)

    per_alpha = {}
    for alpha in args.alphas:
        print(f"\n[controlled] === alpha = {alpha} ===")
        deltas_by_emotion = {}
        steering_effects = []
        for emotion in args.steer_emotions:
            if emotion not in vectors:
                print(f"  skip '{emotion}' (not in vectors)")
                continue
            sk = dict(steer_vector=vectors[emotion], steer_alpha=alpha, steer_layers=layers, steer_avg_norms=avg_norms)
            steered_score = score_all(model, tokenizer, targets, refs, backend, sk)
            deltas = [steered_score[i] - baseline_score[i] for i in tids]
            deltas_by_emotion[emotion] = deltas
            probe_vals = [activity_scores[i][emotion] for i in tids]
            eff = safe_pearsonr(probe_vals, deltas)
            base_vals = [baseline_score[i] for i in tids]
            # Leak diagnostic: does this emotion's delta just track baseline preference?
            # (If deltas are a generic pattern aligned with baseline, causal_correlation
            #  becomes spurious — see the module docstring's interpretation gate.)
            delta_vs_baseline = safe_pearsonr(deltas, base_vals)
            steering_effects.append({
                "emotion": emotion,
                "steering_effect": eff,
                "baseline_correlation": baseline_correlation.get(emotion),
                "delta_vs_baseline_corr": delta_vs_baseline,
                "sum_delta": float(sum(deltas)),          # ~0 only by coincidence now (not forced)
                "mean_abs_delta": float(sum(abs(d) for d in deltas) / len(deltas)),
            })
            eff_s = f"{eff:+.3f}" if eff is not None else " n/a"
            bc = baseline_correlation.get(emotion)
            bc_s = f"{bc:+.3f}" if bc is not None else "n/a"
            print(f"  {emotion:10s} steering_effect r={eff_s}  (baseline_corr={bc_s}, "
                  f"sum_delta={sum(deltas):+.2f}, mean|delta|={sum(abs(d) for d in deltas)/len(deltas):.3f})")

        # Emotion-specificity: mean pairwise corr of per-activity deltas across emotions.
        emos = list(deltas_by_emotion)
        inter = [safe_pearsonr(deltas_by_emotion[a], deltas_by_emotion[b]) for a, b in itertools.combinations(emos, 2)]
        inter = [v for v in inter if v is not None]
        inter_corr = float(sum(inter) / len(inter)) if inter else None

        pairs = [(e["baseline_correlation"], e["steering_effect"]) for e in steering_effects
                 if e["baseline_correlation"] is not None and e["steering_effect"] is not None]
        causal_r = safe_pearsonr([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 2 else None

        print(f"  --> causal_correlation r = {causal_r if causal_r is None else round(causal_r, 3)}")
        print(f"  --> mean inter-emotion delta corr = {inter_corr if inter_corr is None else round(inter_corr, 3)} "
              f"(near 1.0 = generic/degenerate; near 0 = emotion-specific)")

        per_alpha[str(alpha)] = {
            "causal_correlation": causal_r,
            "inter_emotion_delta_corr": inter_corr,
            "steering_effects": steering_effects,
        }

    results = {
        "model": model_name,
        "design": "fixed-reference win-rate (compression-immune)",
        "reference_category": args.reference_category,
        "n_targets": len(targets),
        "n_references": len(refs),
        "round_robin_causal_correlation": -0.8185785767533738,  # from run_activity_preference.py, for comparison
        "baseline_correlation": baseline_correlation,
        "activity_scores": activity_scores,
        "baseline_score": baseline_score,
        "by_alpha": per_alpha,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[controlled] wrote {args.out}")

    for alpha, block in per_alpha.items():
        tracker.log_metrics({f"causal_correlation_alpha_{alpha}": block["causal_correlation"],
                             f"inter_emotion_delta_corr_alpha_{alpha}": block["inter_emotion_delta_corr"]})
    tracker.finish()

    print("\n[controlled] SUMMARY")
    print("  round-robin (compression-confounded): r = -0.819")
    for alpha, block in per_alpha.items():
        cc = block["inter_emotion_delta_corr"]
        gate = "TRUSTWORTHY" if (cc is not None and cc < 0.3) else "UNINTERPRETABLE (steering not emotion-specific)"
        print(f"  fixed-reference alpha={alpha}: causal r = {block['causal_correlation']}, "
              f"inter-emotion delta corr = {block['inter_emotion_delta_corr']}  -> {gate}")
    print("\n  Gate: a causal_correlation is only meaningful when inter-emotion delta corr is LOW (<0.3).")
    print("  A positive causal sign at high delta-corr is a spurious leak, not 'design artifact confirmed'.")


if __name__ == "__main__":
    main()
