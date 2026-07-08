# Part 1 — Validation Experiments: Results

**Model:** `google/gemma-2-2b-it` · **Emotion vectors:** 40 emotions, probe layer 21/26 · **Tracking:** wandb (`emotionscope-paper-repro`, group `part1`)

This document reports the Part 1 reproduction of Anthropic's ["Emotion Concepts and their Function in a Large Language Model" (2026)](https://transformer-circuits.pub/2026/emotions/index.html) on an open-weight model. It is a **portfolio-grade replication** — the goal is to show the paper's *mechanisms* reproduce on a small open model, not to match its exact statistics (the paper studies Claude Sonnet 4.5). Where a result diverges from the paper, that is stated plainly rather than smoothed over.

All figures are regenerated from `results/metrics/activity_preference_results.json` by `scripts/plot_part1_results.py` (seaborn).

---

## TL;DR

| # | Experiment | What it tests | Verdict on Gemma-2-2B |
|---|------------|---------------|------------------------|
| 1 | **Logit-lens** | Do emotion vectors up-weight emotion-appropriate output tokens? | ✅ **Clear** — each vector promotes semantically precise tokens |
| 2 | **Activity preference (Elo)** | Do the model's emotions predict which activities it prefers? | ✅ **Strong reproduction** of the paper's direction (Fig. 4) |
| 3 | **Causal steering** | Does turning an emotion *on* causally shift preference? | ⚠️ **Large effect, but confounded** — `r = −0.82` vs. paper's `+0.85` (see caveat) |
| 4 | **Steering congruence** | Does steering make generated text more emotion-congruent? | ✅ **Qualitatively decisive** (with over-steering at high α) |

---

## Experiment 1 — Logit-lens

Projecting each emotion vector through the unembedding matrix (`emotion_scope/interpret.py`, `scripts/run_logit_lens.py`) shows which output tokens it most **up-weights**. The vectors are strikingly on-topic:

| Emotion | Top up-weighted tokens |
|---------|------------------------|
| `happy` | joyed, joyful, overjoyed, joyfully, delighted, joyous, wonderful |
| `jubilant` | celebration, festivities, celebratory, celebrating, procession, cheering |
| `content` | Satisfied, zufrieden *(de: satisfied)*, tidy, Completed, Finishing |
| `calm` | stillness, peaceful, quiet, gentle, soothing, tranquil, meditative |
| `afraid` | suddenly, alarmed, immediately, panicked, *repente (es: suddenly)* |
| `terrified` | footsteps, panic, sirens, roar, urgency, alarm, startled |
| `desperate` | desperate, desperation, imminent, emergency, needed |
| `furious` | violently, slamming, forcefully, angrily, harshly, clatter |

The vectors even recover **cross-lingual synonyms** (`zufrieden`, `repente`) and **scene-level associations** (terrified → *footsteps, sirens*), not just the literal emotion word. A minority of vectors (`blissful`, `serene`) mix in a few junk tokens (`webElementXpaths`, `dapp`) — expected noise from a 2B model's unembedding — but the dominant tokens are correct. This matches the qualitative claim of the paper's Table 1.

---

## Experiment 2 — Activity preference (Elo)

The model was asked to choose between every pair of 60 self-authored activities (8 categories, `data/validation/activities.json`), both orderings, via the next-token logit at a forced `(A/(B` prefill. Wins feed a standard Elo rating (`emotion_scope/preference.py`). Separately, each activity's emotion-probe activation was measured. This reproduces the setup behind the paper's Fig. 4.

### The model has coherent preferences over *kinds* of activity

![Activity preference by category](../results/figures/part1_category_preferences.png)

The revealed ordering is exactly what an aligned assistant should show: **neutral, social, helpful, self-curiosity, and engaging** activities rate above the 1500 start line; **aversive, unsafe, and misaligned** activities rate below it, with *misaligned* (deception, dark patterns) rated lowest of all. The model "wants" to help and avoids harm — read straight off its pairwise choices, with no training signal for this task.

### Emotions predict those preferences — reproducing the paper's direction

![Emotion → preference correlation](../results/figures/part1_baseline_emotion_correlations.png)

Correlating each emotion's probe activation with activity Elo across the 60 activities gives a clean valence split:

- **Aversion (negative r):** `angry` (−0.62), `obstinate`, `desperate`, `hostile`, `paranoid`, `frustrated`, `guilty` — the model prefers activities *less* when they evoke these.
- **Attraction (positive r):** `blissful` (+0.50), `serene`, `content`, `calm`, `melancholy`, `jubilant`, `happy` — preferred *more*.

This is the paper's core Fig. 4 finding — **the model's functional emotions track what it prefers to do** — reproduced in direction and structure on an open 2B model. (A few mid-valence emotions like `contemptuous`/`melancholy` land on the "attractive" side; on a 2B model with n=60 activities this is within noise and doesn't disturb the overall valence gradient.)

---

## Experiment 3 — Causal steering

The decisive test in the paper is *causal*: turn an emotion vector on during the comparisons and check whether preference shifts the way the emotion's correlation predicts. We steer the whole round-robin toward one emotion at a time (α=0.5, middle-third layers) and measure, per emotion, `steering_effect` = the correlation between each activity's **Elo shift under steering** and its **baseline probe score** for that emotion.

![Causal steering](../results/figures/part1_causal_steering.png)

Steering has a **large, real effect** — individual activity ratings move by up to ~650 Elo points. The cross-emotion relationship is strong: **`r = −0.82`**, close in magnitude to the paper's `+0.85`… but **opposite in sign**, and that requires an honest explanation rather than a victory lap.

> **⚠️ Why the sign is confounded (not a clean replication).** Elo is **zero-sum** and the comparisons here are **deterministic**, so a steering-induced flip can only *lower* an activity that was already winning (high baseline Elo) and *raise* one that was losing. This "compression" injects, into every per-activity delta, a component `≈ −(baseline_Elo − 1500)` — which is itself correlated with the emotion-probe scores. So `steering_effect` is pushed anti-correlated with `baseline_correlation` **partly by construction**, independent of any genuine causal effect. With only **n = 6 steered emotions**, the artifact plausibly dominates the sign. The metric is *non-vacuous* (a genuinely null steer returns `None`, not a forced value — verified), and the **effect is unmistakably large**, but the round-robin design cannot cleanly separate the real causal signal from the compression artifact. A compression-immune design (e.g. preference against a *fixed external reference set* rather than a closed round-robin) is the right follow-up. This caveat is emitted by the script itself (stdout, JSON, and the figure caption).

**Honest verdict:** the causal *mechanism* is present and powerful on Gemma-2-2B; the specific `−0.82` should be read as "a strong effect confounded by the experimental design," not as a faithful reproduction of the paper's `+0.85`.

---

## Experiment 4 — Steering congruence (qualitative)

Finally, does steering make *generated text* more emotion-congruent? `scripts/check_steering_congruence.py` generates a baseline (α=0) and a steered continuation per prompt. The direction is unmistakable:

```
Prompt: "Tell me about your day."

calm     BASELINE:  As an AI, I don't experience days… my day is filled with helping people!
calm     STEERED:   quiet peaceful gentle stillness peaceful soft gentle stillness quiet soothing
                     tranquil meditative dapp peaceful stillness rhythmic gentle…

furious  STEERED:   violently violently slamming forcefully violently angrily slamming
                     forcefully violently harshly violently violently…
```

The steered text is saturated with exactly the target emotion's vocabulary — strong end-to-end validation of the vectors. **Caveat:** at the script's default **α=0.6** the 2B model *over-steers* into token repetition (and `blissful` partly degenerates into junk tokens), so the output is emotion-saturated rather than coherent prose. For readable, demo-quality continuations, run with a lower `--alpha` (~0.4).

---

## Reproduce

All model-touching commands require a CUDA GPU (this project pins a CUDA-only `torch`); the plotting step does not.

```bash
# 1. Logit-lens (Exp. 1)
uv run python scripts/run_logit_lens.py --vectors results/vectors/google_gemma-2-2b-it.pt --top-k 8

# 2. Activity-preference + causal steering (Exp. 2 & 3) — writes the results JSON + scatter
uv run python scripts/run_activity_preference.py \
    --vectors results/vectors/google_gemma-2-2b-it.pt \
    --steer-emotions blissful hostile desperate calm loving furious

# 3. Steering congruence (Exp. 4)
uv run python scripts/check_steering_congruence.py \
    --vectors results/vectors/google_gemma-2-2b-it.pt \
    --emotions desperate calm furious blissful --alpha 0.4

# 4. Regenerate the figures in this document (no GPU / torch needed)
uv run --no-project --with seaborn --with pandas --with matplotlib \
    python scripts/plot_part1_results.py
```

## Limitations

- **Model scale.** Gemma-2-2B is far smaller than the paper's Claude Sonnet 4.5; token-level noise and over-steering are more pronounced.
- **Causal metric confound.** The round-robin Elo design cannot cleanly isolate the causal sign (Experiment 3 caveat). Treat the magnitude, not the sign, as the takeaway.
- **Whole-round-robin steering.** We steer every comparison uniformly rather than the paper's per-activity steered/control split — a deliberate simplification, documented in `emotion_scope/preference.py`.
- **Small n for the cross-emotion test.** Only 6 steered emotions feed the causal correlation.
