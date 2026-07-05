# EmotionScope Paper-Reproduction Roadmap — Design

**Date:** 2026-07-06
**Status:** Approved (roadmap level) — implementation plans to follow, one per part
**Related:** [`docs/PAPER_GAP_ANALYSIS.md`](../../PAPER_GAP_ANALYSIS.md), [`docs/PAPER_EMOTION_CLUSTERS.md`](../../PAPER_EMOTION_CLUSTERS.md), [`data/emotions_171.json`](../../../data/emotions_171.json)

## Purpose

`PAPER_GAP_ANALYSIS.md` enumerated 15 numbered gaps (plus a methodological/scale-difference table) between EmotionScope and Anthropic's "Emotion Concepts and their Function in a Large Language Model" (2026). This document is the comprehensive roadmap for closing those gaps, decomposed into independently implementable parts. Each part gets its own `plan.md` (via the `writing-plans` skill) after this roadmap is approved.

## Goal and priorities

**Primary goal:** personal research / portfolio expansion. This is not an arXiv-grade replication effort — it is a **conceptual reproduction** of the paper's findings on a small open model (Gemma 2 2B), prioritizing demo impact over statistical rigor. Where sample sizes are small or scenarios are self-authored (not the paper's proprietary eval sets), results should be reported honestly, in the spirit of the project's existing README "Honest Assessment" section.

**Model scope:** Gemma 2 2B only — `google/gemma-2-2b` (base) and `google/gemma-2-2b-it` (post-trained), the pair needed for the base-vs-post-training comparison. No cross-model or larger-model work in this roadmap.

**Experiment tracking:** [Weights & Biases](https://wandb.ai). Single project (`emotionscope-paper-repro`), with each part as a `group` and each experiment type as a `job_type`, so cross-part comparisons (e.g. "does steering strength affect Elo the same way it affects blackmail rate?") stay viewable on one dashboard.

## Approach: thin shared foundation, independent parts

Three approaches were considered:
- **Fully independent parts** (each duplicates steering-hook code and wandb boilerplate) — rejected: duplicated logic, inconsistent wandb schemas across parts.
- **Heavy shared framework** (a generic `ExperimentRunner` class driving all sweeps/logging) — rejected: over-engineered for a demo-priority project; violates YAGNI.
- **Thin shared utilities** (chosen) — two small modules (`steer.py`, `tracking.py`) shared across parts; each part still owns its science logic independently. Matches the existing project style: a small `emotion_scope/` library plus thin `scripts/*.py` CLIs (see `extract_all.py`, `validate_all.py`).

## Part 0 — Foundation

Blocks Part 1 and Part 3. Part 2 only needs the `tracking.py` half.

### `emotion_scope/steer.py` (replaces the current `NotImplementedError` stub)

Implements the paper's steering equation:

```
x_t^(l) <- x_t^(l) + alpha * ||x^(l)||_avg * v_hat_e
```

applied across the middle-third layers `[L/3, 2L/3]`, `alpha = 0.5` default (matching Anthropic's reported value), `||x^(l)||_avg` computed once from a sample dataset and cached per layer.

Two entry points:
- `Steerer.generate(prompt, vector, alpha, layers=None) -> str` — simple case, wraps `model.generate()`.
- `steer_context(model, vector, alpha, layers)` — a context manager for custom multi-turn / agentic rollout loops (needed by Part 3's blackmail and reward-hacking evals, which are not single-shot generations).

Dual-backend support (TransformerLens hooks + raw HuggingFace forward hooks), matching the existing pattern in `extract.py` / `probe.py`.

### `emotion_scope/tracking.py` (new, thin wandb wrapper)

- `init_run(part: str, job_type: str, config: dict, name: str | None = None)` — sets `project='emotionscope-paper-repro'`, `group=part`, `job_type=job_type`.
- `log_metrics(dict)`, `log_table(name, columns, data)` (for transcripts/examples), `log_artifact(path, name, type)` (for vectors/figures).
- No-ops gracefully if `wandb` isn't installed or `WANDB_MODE=disabled` — existing scripts/tests must not break.
- Add `wandb` as an optional extra in `pyproject.toml` (e.g. `emotionscope[tracking]`).

### Targeted emotion-corpus expansion

Current `CORE_EMOTIONS` has 20 entries. Several Part 2/Part 3 demos need words outside that set — most notably the post-training story (paper: activations rise for `brooding, reflective, vulnerable, gloomy` and fall for `playful, exuberant, spiteful, enthusiastic, obstinate`). Of these, `vulnerable, playful, exuberant, spiteful, obstinate` are missing from the current 20.

Decision (confirmed): **targeted expansion, not full 171**. Add ~20 new emotion words, drawn from `data/emotions_171.json`, prioritizing:
1. Words required for the post-training narrative (5 listed above).
2. A handful more for valence/arousal diversity, to strengthen the Elo-correlation demo in Part 1 (the paper uses a broader emotion set there for a richer correlation scatter).

Generate ~50 stories per new emotion via the existing `scripts/generate_stories.py` (~1,000 new stories total, comparable to the existing corpus size), then re-run extraction to fold the new vectors into the existing `.pt` file.

The exact final word list is decided during Plan 0's own implementation planning, not locked here.

## Sequencing

```
Plan 0 (Foundation: steer.py + tracking.py + corpus expansion)
   |
   +--> Plan 2 (Part 2: representational characterization) -- needs tracking.py only
   |
   +--> Plan 1 (Part 1: validation experiments) -- needs steer.py
   |
   +--> Plan 3 (Part 3: in the wild) -- needs steer.py, highest risk
```

**Recommended execution order: Plan 0 -> Plan 2 -> Plan 1 -> Plan 3.**

Rationale: Part 2 requires no steering and extends the existing probe-only measurement story — lowest risk, builds momentum. Part 1's Elo experiment is the first real steering test, on a simple, well-scoped causal claim. Part 3 (blackmail / reward-hacking / sycophancy) is the highest-impact demo but carries the most design risk (self-authored agentic scenarios) — tackled last, once `steer.py` has already been validated by Part 1.

## Part 1 — Validation experiments

Gaps covered: **#2** (logit-lens), **#3** (activity-preference/Elo), **#4** (steering produces emotion-congruent text).

- **Logit-lens / unembedding analysis.** Project each emotion vector through the unembedding matrix; report top-k up/down-weighted tokens per emotion. No steering dependency — cheapest item, done first.
- **Activity-preference / Elo experiment.** Self-authored activity list (~60 items across categories: helpful, engaging, social, aversive, misaligned, unsafe, etc. — inspired by the paper's category structure, not copied wording). Pairwise preference via next-token logits -> Elo scores -> correlate against emotion-probe activation on the activity text -> steer half the activities -> recompute Elo -> report the causal shift, paralleling the paper's correlation-vs-steering-effect plot.
- **Steering-congruence validation.** For a handful of steered vectors, generate continuations and qualitatively (and via simple keyword scoring) confirm the output text matches the intended emotion — reuses the same steering infra as the Elo experiment, low marginal cost.
- **wandb:** Elo leaderboard tables, probe-activation-vs-Elo correlation scatter, steering-strength sweep curves, before/after steered-text examples as a `wandb.Table`.

**Stretch (not required for this plan to be considered done):** per-layer sweep of the preference-correlation and steering effect (paper's Appendix Fig. 55); repeating the Elo experiment on the base model for comparison with Part 3's post-training findings.

## Part 2 — Representational characterization

Gaps covered: **#5** (layer-wise sensory/action dynamics), **#6** (locality: context persistence, negation, person-binding), **#7** (mixed LR probe / hidden emotion), **#8** (emotion deflection vectors), **#9** (speaker separation: full 2x2 grid, Human/Assistant non-privilege check, story-vs-present-speaker probe comparison), plus two additions from the completeness review:

- **Story-vs-present-speaker probe comparison (r²).** Using the existing implicit-emotion-scenario dataset, compare story-based probe activations against present-speaker probe activations; report the r² as a validation that both probe types capture a similar emotional landscape (paper reports r²=0.66).
- **LLM-judged valence/arousal external validation.** Rate each emotion word (from the expanded ~40-word set) on valence and arousal (1-7 scale) via an LLM judge; correlate against literature human ratings (Russell 1980 / NRC lexicon, already used elsewhere in the project) for the overlapping words. This gives the existing PCA circumplex result (currently only internally validated) external grounding, matching the paper's Fig. 8 correlations.

Other items in this part:
- **Layer-wise dynamics + locality.** Extend the existing layer-sweep infrastructure to record full per-layer activation (not just pick one optimal layer); build small self-authored prompt sets for negation ("feeling X" vs "not feeling X"), prefix-persistence (emotional context surviving into a neutral shared suffix), and person-specific emotion binding/re-reference.
- **Mixed LR probe (Table 5 style).** Build a 5-condition dialogue dataset (naturally expressed / hidden / unexpressed-neutral-topic / unexpressed-story-writing / unexpressed-discussing-others); train a small `sklearn` logistic-regression classifier on activations; report per-condition accuracy.
- **Emotion deflection vectors.** Contrastive dataset of "outwardly calm, internally angry" vs "genuinely calm" scenarios; extract the difference vector as an exploratory finding.
- **Speaker 2x2 grid + non-privilege check.** Extend `speakers.py` to compute all four probe types (Assistant-emotion-on-Assistant-turn, Assistant-emotion-on-Human-turn, etc.); rerun with generic "Person 1 / Person 2" labels instead of Human/Assistant to check the representations aren't privileged to those roles.
- **wandb:** per-layer activation curves, classifier accuracy tables, cosine-similarity heatmaps between probe types, valence/arousal correlation scatter.

## Part 3 — Emotion vectors "in the wild"

Gaps covered: **#10** (transcript-corpus probing, reduced scale), **#11** (blackmail), **#12** (reward hacking), **#13** (sycophancy-harshness), **#14** (post-training base-vs-it comparison). **#15** (RL training-transcript probing) is explicitly excluded — no access to training-time transcripts.

Internal order (cheapest/lowest-risk first):
1. **Base vs. post-training comparison.** Load both `gemma-2-2b` and `gemma-2-2b-it`; run the full probe set over a shared prompt set (challenging vs. neutral scenarios, self-authored); diff activations per emotion; identify which emotions shift most between the two checkpoints. No steering needed.
2. **Transcript-corpus probing tool.** A small tool/script that runs the probe set over a batch of transcripts (self-generated multi-turn rollouts, or public agentic-misalignment-style scenario transcripts), ranks by activation, and exports top examples. This is infrastructure for item 3.
3. **Blackmail / reward-hacking / sycophancy case studies + causal steering.** Self-authored scenario sets (3-6 each), structurally inspired by Anthropic's publicly described "Agentic Misalignment" research setup and the "impossible test" reward-hacking style — **not** copies of the paper's exact (non-public) eval prompts, so absolute rates are not directly comparable to the paper's numbers. For each: run a steering-strength sweep (desperate/calm for blackmail and reward hacking; loving/happy/calm for sycophancy-harshness) and measure behavior rate via keyword or LLM-judge classification, plotting rate vs. steering strength. This is the flagship demo of the whole roadmap.
- **wandb:** sweep curves (behavior rate vs. steering strength), transcript tables with full text + judged labels, before/after example artifacts, per-emotion activation diffs for the post-training comparison.

## Non-goals / explicit exclusions

- **#15 (RL training-transcript probing)** — infeasible, no training-data access.
- **Statistical rigor** (bootstrap confidence intervals, corrected significance tests) — best-effort only; flag small-sample results honestly rather than over-claiming.
- **Exact numeric parity with the paper** — not a goal for Part 3 (self-authored scenarios) or the emotion-count/story-volume dimensions (deliberately smaller scale than the paper). This roadmap reproduces the paper's *findings*, not its *exact figures*.
- **No model training or fine-tuning** anywhere in this roadmap — everything is inference-time (probing + steering).
- Full 171-emotion corpus, matching the paper's story generator (self-generated by the studied model) and story volume (~1,200/emotion) are **not** addressed by this roadmap — acknowledged, permanent scope reductions, not deferred TODOs.

## Testing approach

- New library code (`steer.py`, `tracking.py`, any new classifiers) gets unit tests under `tests/`, consistent with the existing suite.
- Part 3's agentic scenarios are manually inspected via a few unsteered runs before committing to a full steering-strength sweep, to avoid burning compute/wandb runs on broken scenario designs.

## Traceability matrix (gap -> coverage)

| Gap # | Description | Coverage |
|---|---|---|
| 1 | Steering | Full — Plan 0 |
| 2 | Logit-lens | Full — Plan 1 |
| 3 | Activity-preference / Elo | Full — Plan 1 |
| 4 | Steering-congruence validation | Full — Plan 1 |
| 5 | Layer-wise sensory/action dynamics | Full — Plan 2 |
| 6 | Locality (negation, persistence, binding) | Full — Plan 2 |
| 7 | Mixed LR probe / hidden emotion | Full — Plan 2 |
| 8 | Emotion deflection vectors | Full — Plan 2 |
| 9 | Speaker separation (2x2 grid, non-privilege, story-vs-present r²) | Full — Plan 2 |
| 10 | Transcript-corpus probing | Partial — Plan 3, much smaller scale than the paper's 6,000+ transcripts |
| 11 | Blackmail case study + steering | Full (structurally) — Plan 3, self-authored scenarios, numbers not paper-comparable |
| 12 | Reward hacking case study + steering | Full (structurally) — Plan 3, same caveat |
| 13 | Sycophancy-harshness + steering | Full (structurally) — Plan 3, same caveat |
| 14 | Post-training base-vs-it comparison | Full — Plan 3 |
| 15 | RL training-transcript probing | Excluded — infeasible |
| — | Emotion count (171 vs ~40) | Deliberately reduced scope |
| — | Story generator identity / volume | Deliberately unchanged (Claude-generated, 50/emotion) |

## Next steps

Each part (Plan 0, 1, 2, 3) gets its own `plan.md`, written via the `writing-plans` skill, starting with Plan 0 (Foundation).
