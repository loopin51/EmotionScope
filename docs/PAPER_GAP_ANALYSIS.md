# Paper Gap Analysis — EmotionScope vs. Anthropic (2026)

**Reference paper:** Sofroniew, N., Kauvar, I., Saunders, W., et al. (2026). *Emotion Concepts and their Function in a Large Language Model.* Transformer Circuits Thread. <https://transformer-circuits.pub/2026/emotions/index.html>

**Local copy:** [`anthropic_emotion_concepts_2026.html`](anthropic_emotion_concepts_2026.html) (full HTML, downloaded 2026-07-05)

**Purpose:** Map what the paper does against what EmotionScope currently implements, and enumerate the gaps. This is a checklist for prioritizing future work, written for researchers evaluating or extending the toolkit.

---

## TL;DR

> EmotionScope faithfully reproduces the paper's **Part 1 (emotion-vector extraction + validation)** and the **geometry** portion of Part 2 on a small open model (Gemma 2 2B IT). But the paper's central contribution — **causal steering** — and the entire **Part 3 ("emotion vectors in the wild": blackmail / reward hacking / sycophancy + post-training analysis)** are **not implemented**.

The single highest-leverage next step is **steering** ([`emotion_scope/steer.py`](../emotion_scope/steer.py) is currently a `NotImplementedError` stub). Almost everything in Part 3 depends on it.

---

## What EmotionScope already covers ✅

| Paper section | EmotionScope implementation |
|---|---|
| Finding emotion vectors (contrastive mean diff − grand mean) | [`extract.py`](../emotion_scope/extract.py) `_compute_contrastive_vectors` |
| Neutral-PCA denoising (top components at 50% variance, projected out) | [`extract.py`](../emotion_scope/extract.py) `_compute_neutral_pca` / `_denoise_vectors` |
| Probing at the "Assistant colon" / response-preparation position | [`probe.py`](../emotion_scope/probe.py) `_select_probe_index` |
| Activation on implicit-emotion scenarios (Fig. 2 confusion matrix) | [`validate.py`](../emotion_scope/validate.py) `test_confusion_matrix` |
| Numerical-intensity test (Fig. 3 Tylenol dosage) | [`validate.py`](../emotion_scope/validate.py) `test_tylenol` |
| Geometry: pairwise cosine similarity (Fig. 5) | `results/figures/similarity_matrix.png` |
| Geometry: PCA → valence/arousal circumplex (Figs. 7–8) | `results/figures/circumplex_pca.png` |
| Layer selection (related to Fig. 9 cross-layer stability) | [`extract.py`](../emotion_scope/extract.py) `find_best_probe_layer` |
| Present/other-speaker vectors — *partial* (see gap #9) | [`speakers.py`](../emotion_scope/speakers.py) |

---

## Gaps — what the paper has that EmotionScope does not

Legend: 🔴 core contribution · 🟠 significant · 🟡 methodological/scale

### 🔴 The central gap

**1. Steering (causal intervention) — the paper's key finding.**
The paper's headline claim is that emotion vectors *causally* drive behavior, via injection into the residual stream at middle-third layers:
`x_t^(l) ← x_t^(l) + α · ‖x^(l)‖_avg · v̂_e`, with α = 0.5.
EmotionScope's [`steer.py`](../emotion_scope/steer.py) is an empty stub (`raise NotImplementedError`); the README roadmap lists steering as unstarted Phase 2. **The toolkit currently only *measures* (probe), it cannot *manipulate* (steer).** Gaps 2–13 below all depend on this.

---

### 🟠 Part 1 — validation experiments not implemented

**2. Logit-lens / unembedding analysis (Table 1).** Project each emotion vector through the unembedding matrix to see which output tokens it up/down-weights (e.g. desperate → "urgent", "bankrupt"). Not implemented. *(Note: grep hits for `unembed` in `models.py` are TransformerLens config comments, not this analysis.)*

**3. Activity-preference experiment (Fig. 4).** 64 activities × 4032 pairs → model preference via logits → **Elo** scores → correlation with emotion-probe activation (blissful r=0.71, hostile r=−0.74) → steering causally shifts preference (r=0.85 between correlation and steering effect). Entirely absent.

**4. Steering-produces-congruent-text validation** (Appendix "Causal effects on the emotional content of model continuations"). Absent (depends on #1).

---

### 🟠 Part 2 — representational characterization not implemented

**5. Layer-wise dynamics: "sensory" (early-mid) vs "action" (mid-late) representations.** The paper shows emotion representations evolve across layers — early layers encode local emotional connotation, late layers encode the "planned" emotion for upcoming tokens. EmotionScope selects a **single** probe layer (layer 22); the layer sweep exists only to pick that one optimum, not to analyze cross-layer meaning shifts.

**6. "Locality" token/layer experiments (Figs. 12–15).**
- Emotional-context persistence (marriage "hard"/"good" → identical suffix still differs in late layers)
- **Negation** ("feeling X" vs "not feeling X" separating in mid-late layers)
- **Person-specific emotion binding & retrieval on re-reference** ("her" reactivating a specific character's emotion)
- Tylenol dosage integration across layers

None implemented.

**7. Mixed logistic-regression probe + "chronically represented" emotional state (Table 5).** Five dialogue conditions — naturally expressed / **hidden** / unexpressed-neutral-topic / story-writing / discussing-others — with an LR classifier that detects *internal, unexpressed* emotion. EmotionScope uses only contrastive-mean + cosine; no classifier, no hidden-emotion datasets.

**8. "Emotion deflection" vectors.** Representation of outwardly-calm-while-internally-angry situations. Not implemented.

**9. Speaker separation — partial only.** EmotionScope extracts present/other-speaker vectors and tests orthogonality + "thermostat" (found absent). Missing vs. paper:
- The full **2×2 probe grid** ("Assistant emotion on Human turn", etc.)
- **Human/Assistant non-privilege check** (replacing with generic "Person 1/Person 2" yields identical probes — Fig. 19)
- "Emotional regulation" circuit exploration

---

### 🔴 Part 3 — "Emotion vectors in the wild" (essentially entirely missing)

**10. Large-scale probe sweep over 6,000+ real evaluation transcripts.** Rank transcripts by emotion-vector activation, examine top examples, build a transcript visualization tool. EmotionScope analyzes only a **single-prompt, single-turn** demo chat. (The frontend per-token heatmap exists but does not scale to long agentic transcripts.)

**11. Blackmail case study + causal steering (Figs. 26–29).** desperate↑ / calm↓ raises blackmail rate (22% → 72% under steering). Not implemented.

**12. Reward-hacking case study + causal steering (Figs. 30–31).** On the "impossible code" eval, desperate steering raises reward hacking ~5% → ~70% (14×). Not implemented.

**13. Sycophancy–harshness tradeoff + steering (Figs. 32–35).** loving/happy/calm steering increases sycophancy; suppressing them increases harshness. Not implemented.

**14. Post-training analysis: base vs. post-trained model (Figs. 36–39).** Post-training shifts activations toward brooding/gloomy/vulnerable and away from playful/exuberant/spiteful. **This is one of the most achievable gaps** — it needs no steering and no extra data, just comparing `gemma-2-2b` (base) vs `gemma-2-2b-it` (post-trained) with the existing probe.

**15. Emotion probes over RL training transcripts** (frustrated → GUI stuck, panicked → UI error, etc.). Not feasible without access to training transcripts.

---

### 🟡 Methodological / scale differences

| Dimension | Paper | EmotionScope |
|---|---|---|
| Emotion count | **171** emotion words | **20** core emotions |
| Story generator | the **studied model itself** (Sonnet 4.5) | Claude, cross-model |
| Story scale | 100 topics × 12 stories/emotion (~1,200/emotion) | 50/emotion |
| Extraction token start | from 50th token, averaged | content-range averaging ✅ (equivalent) |
| Real-time monitoring | proposed as a safety monitor | demo visualization only (no deployment safeguard) |

---

## Suggested priority order

1. **Implement steering** ([`steer.py`](../emotion_scope/steer.py)) — unblocks gaps #3, #4, #11, #12, #13.
2. **Base vs. post-training comparison** (#14) — high insight, low cost, no steering required.
3. **Activity-preference / Elo experiment** (#3) — self-contained causal test once steering exists.
4. **Layer-wise + locality analyses** (#5, #6) — deepen the Part 2 story with the existing probe.
5. **Transcript-corpus probing tool** (#10) — infrastructure for any "in the wild" study.

---

*Generated 2026-07-05 by reading the full paper text against the current codebase (`master` @ 8e8a2ae).*
