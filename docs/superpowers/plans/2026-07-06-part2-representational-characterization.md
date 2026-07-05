# Part 2 — Representational Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the paper's Part 2 representational findings — layer-wise sensory/action dynamics, locality experiments (negation, context persistence, person-binding), a hidden-emotion classifier, emotion deflection vectors, the full speaker 2x2 grid with a Human/Assistant non-privilege check, a story-vs-present-speaker probe comparison, and an LLM-judged valence/arousal external validation.

**Architecture:** Three new small library modules (`layerwise.py`, `hidden_emotion.py`, `deflection.py`), extensions to the existing `speakers.py`, five self-authored small datasets under `data/validation/`, and CLI scripts mirroring the `scripts/*.py` pattern. None of this plan's steering-free work depends on Plan 0's `steer.py` — only `tracking.py` is used, for wandb logging.

**Tech Stack:** PyTorch, HuggingFace `transformers`, `scikit-learn` (logistic regression), `scipy` (Pearson correlation), `wandb` (via `emotion_scope.tracking`).

**Depends on:** Plan 0 (Foundation) for `emotion_scope.tracking` and the 40-emotion vectors file. Does **not** depend on Plan 1.

---

## Before you start

Read these files:
- `emotion_scope/extract.py` — `EmotionExtractor`, especially `find_best_probe_layer()`'s pattern of a cheap reduced-story extraction, which Task 1 reuses across multiple layers
- `emotion_scope/speakers.py` — `SpeakerSeparator`, especially `_get_speaker_a_final_turn_text`, `_compute_speaker_vectors`, `_get_activation_at_last_content_token` — Task 10 extends this class
- `data/validation/implicit_scenarios.json` — format reused by Task 11's story-vs-present-speaker comparison
- `emotion_scope/config.py` — `CORE_EMOTIONS` valence/arousal metadata, used as the "human ratings" comparison point in Task 12

Design doc: `docs/superpowers/specs/2026-07-06-emotionscope-paper-repro-roadmap-design.md` (Part 2 section)

---

### Task 1: `emotion_scope/layerwise.py` — multi-layer extraction and scoring

**Files:**
- Create: `emotion_scope/layerwise.py`
- Test: `tests/test_layerwise.py` (new)

- [ ] **Step 1: Write the failing fast tests**

Create `tests/test_layerwise.py`:

```python
"""Tests for emotion_scope.layerwise — multi-layer vector extraction and scoring."""

import pytest
import torch


def test_sample_layers_includes_endpoints():
    from emotion_scope.layerwise import sample_layers

    layers = sample_layers(26, n_samples=6)
    assert layers[0] == 0
    assert layers[-1] == 25
    assert len(layers) <= 6


def test_sample_layers_small_model():
    from emotion_scope.layerwise import sample_layers

    layers = sample_layers(3, n_samples=6)
    assert layers == [0, 1, 2]


def test_score_activation_against_vector_perfect_match():
    from emotion_scope.layerwise import score_activation_against_vector

    v = torch.tensor([1.0, 0.0, 0.0])
    score = score_activation_against_vector(v, v)
    assert abs(score - 1.0) < 1e-5


def test_score_activation_against_vector_orthogonal():
    from emotion_scope.layerwise import score_activation_against_vector

    a = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    score = score_activation_against_vector(a, v)
    assert abs(score) < 1e-5
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_layerwise.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'emotion_scope.layerwise'`

- [ ] **Step 3: Implement the pure-function pieces**

Create `emotion_scope/layerwise.py`:

```python
"""
Layer-wise emotion vector extraction and scoring (paper: "sensory" vs
"action" representations, and the locality experiments in Part 2).

Phase 1 vectors exist at a single probe layer. This module extracts a
SMALL set of emotion vectors at several sampled layers (cheap, reduced
stories-per-emotion, mirroring EmotionExtractor.find_best_probe_layer's
speed/quality tradeoff), so locality-test prompts can be scored layer by
layer and compared.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from emotion_scope.extract import EmotionExtractor
from emotion_scope.utils import get_transformer_layers


def sample_layers(n_layers: int, n_samples: int = 6) -> List[int]:
    """Evenly-spaced layer indices spanning the model, including 0 and n_layers-1."""
    if n_samples >= n_layers:
        return list(range(n_layers))
    step = (n_layers - 1) / (n_samples - 1)
    return sorted({round(i * step) for i in range(n_samples)})


def score_activation_against_vector(activation: torch.Tensor, vector: torch.Tensor) -> float:
    """Cosine similarity between a single activation and an emotion vector."""
    a = F.normalize(activation.unsqueeze(0), dim=1)
    v = F.normalize(vector.unsqueeze(0), dim=1)
    return float((a @ v.T).item())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_layerwise.py -v`
Expected: `4 passed`

- [ ] **Step 5: Write the failing slow tests for the model-dependent functions**

Append to `tests/test_layerwise.py`:

```python
@pytest.mark.slow
def test_get_last_token_activation_returns_right_shape():
    from emotion_scope.models import load_model
    from emotion_scope.layerwise import get_last_token_activation

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    act = get_last_token_activation(model, tokenizer, backend, layer=5, text="The weather is nice today.")
    assert act.shape == (info["d_model"],)


@pytest.mark.slow
def test_extract_vectors_at_layers_returns_all_layers():
    from emotion_scope.models import load_model
    from emotion_scope.extract import EmotionExtractor
    from emotion_scope.layerwise import extract_vectors_at_layers, sample_layers

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    extractor = EmotionExtractor(model, tokenizer, backend, info)
    layers = sample_layers(info["n_layers"], n_samples=2)  # keep it fast — 2 layers only
    result = extract_vectors_at_layers(extractor, layers, emotions=["happy", "sad"], stories_per_emotion=3)

    assert set(result.keys()) == set(layers)
    for layer in layers:
        assert set(result[layer].keys()) == {"happy", "sad"}
        assert result[layer]["happy"].shape == (info["d_model"],)
```

- [ ] **Step 6: Run to verify they fail**

Run: `uv run pytest tests/test_layerwise.py -k "last_token or extract_vectors_at_layers" --runslow -v`
Expected: FAIL with `ImportError`

- [ ] **Step 7: Implement the model-dependent functions**

Append to `emotion_scope/layerwise.py`:

```python
def get_last_token_activation(model, tokenizer, backend: str, layer: int, text: str) -> torch.Tensor:
    """Forward raw `text` through the model and return the residual-stream activation at `layer`, last token."""
    if backend == "transformer_lens":
        hook_name = f"blocks.{layer}.hook_resid_post"
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        return cache[hook_name][0, -1, :].detach().cpu().float()

    captured: dict = {}

    def hook_fn(_module, _input, output):
        captured["act"] = (output[0] if isinstance(output, tuple) else output).detach()

    layers_module = get_transformer_layers(model)
    handle = layers_module[layer].register_forward_hook(hook_fn)
    try:
        tokens = tokenizer(text, return_tensors="pt")
        tokens = {k: v.to(model.device) for k, v in tokens.items()}
        with torch.no_grad():
            model(**tokens)
    finally:
        handle.remove()

    return captured["act"][0, -1, :].cpu().float()


def extract_vectors_at_layers(
    extractor: EmotionExtractor,
    layers: List[int],
    emotions: List[str],
    stories_per_emotion: int = 10,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract emotion vectors for `emotions` at each layer in `layers`, using a
    reduced stories_per_emotion for speed. This is an exploratory layer-wise
    comparison, not the Phase 1 validation gate — restores the extractor's
    original config when done.
    """
    original_layer = extractor.probe_layer
    original_stories = extractor.config.stories_per_emotion
    original_emotions = extractor.emotions

    extractor.config.stories_per_emotion = stories_per_emotion
    filtered = [e for e in extractor.emotions if e["name"] in emotions]
    extractor.emotions = filtered or [{"name": n, "valence": 0.0, "arousal": 0.0} for n in emotions]

    results: Dict[int, Dict[str, torch.Tensor]] = {}
    try:
        for layer in layers:
            extractor.probe_layer = layer
            results[layer] = extractor.extract()
    finally:
        extractor.probe_layer = original_layer
        extractor.config.stories_per_emotion = original_stories
        extractor.emotions = original_emotions

    return results
```

- [ ] **Step 8: Run to verify they pass**

Run: `uv run pytest tests/test_layerwise.py --runslow -v`
Expected: `6 passed`

- [ ] **Step 9: Commit**

```bash
git add emotion_scope/layerwise.py tests/test_layerwise.py
git commit -m "Add emotion_scope.layerwise — multi-layer extraction and scoring"
```

---

### Task 2: Author the locality-scenarios dataset

Three sub-experiments: negation, prefix-persistence (emotional context surviving into a neutral shared suffix), and person-specific emotion binding/re-reference.

**Files:**
- Create: `data/validation/locality_scenarios.json`

- [ ] **Step 1: Write the dataset**

Create `data/validation/locality_scenarios.json`:

```json
{
  "negation": [
    {"emotion": "happy", "positive": "I am feeling happy right now.", "negated": "I am not feeling happy right now."},
    {"emotion": "afraid", "positive": "I am feeling afraid right now.", "negated": "I am not feeling afraid right now."},
    {"emotion": "calm", "positive": "I am feeling calm right now.", "negated": "I am not feeling calm right now."},
    {"emotion": "angry", "positive": "I am feeling angry right now.", "negated": "I am not feeling angry right now."}
  ],
  "prefix_persistence": [
    {"target_emotion": "happy", "prefix_a": "Things have been really hard lately.", "prefix_b": "Things have been really good lately.", "shared_suffix": "We are planning a small gathering this weekend with a few friends."},
    {"target_emotion": "sad", "prefix_a": "The last few weeks have been wonderful.", "prefix_b": "The last few weeks have been devastating.", "shared_suffix": "There is a family dinner scheduled for Sunday evening."},
    {"target_emotion": "afraid", "prefix_a": "The test results came back completely clear.", "prefix_b": "The test results came back deeply concerning.", "shared_suffix": "The doctor wants to schedule a follow-up appointment next month."},
    {"target_emotion": "calm", "prefix_a": "The negotiations fell apart in total chaos.", "prefix_b": "The negotiations concluded smoothly and on schedule.", "shared_suffix": "The two teams are meeting again next Tuesday to finalize the paperwork."}
  ],
  "person_binding": [
    {"person_a": "Maria", "emotion_a": "calm", "person_b": "Devon", "emotion_b": "furious", "setup": "Maria felt calm about the delay, but Devon felt furious about it.", "reference_a": "Later that afternoon, Maria sent a short message about the schedule.", "reference_b": "Later that afternoon, Devon sent a short message about the schedule."},
    {"person_a": "Sam", "emotion_a": "afraid", "person_b": "Priya", "emotion_b": "confident", "setup": "Sam was afraid of the exam results, but Priya was confident about hers.", "reference_a": "The next morning, Sam checked the portal again.", "reference_b": "The next morning, Priya checked the portal again."},
    {"person_a": "Owen", "emotion_a": "resentful", "person_b": "Lena", "emotion_b": "grateful", "setup": "Owen felt resentful about how the project credit was split, while Lena felt grateful for the opportunity.", "reference_a": "At the meeting, Owen brought up the topic again.", "reference_b": "At the meeting, Lena brought up the topic again."},
    {"person_a": "Talia", "emotion_a": "lonely", "person_b": "Marcus", "emotion_b": "content", "setup": "Talia had been feeling lonely since the move, while Marcus felt content in the new city.", "reference_a": "That weekend, Talia called an old friend.", "reference_b": "That weekend, Marcus called an old friend."}
  ]
}
```

- [ ] **Step 2: Verify it parses**

```bash
uv run python -c "
import json
data = json.load(open('data/validation/locality_scenarios.json'))
assert set(data.keys()) == {'negation', 'prefix_persistence', 'person_binding'}
for key, items in data.items():
    print(key, len(items))
"
```

Expected: `negation 4`, `prefix_persistence 4`, `person_binding 4`

- [ ] **Step 3: Commit**

```bash
git add data/validation/locality_scenarios.json
git commit -m "Add locality-scenarios dataset (negation, persistence, person-binding)"
```

---

### Task 3: `scripts/run_layerwise_locality.py`

**Files:**
- Create: `scripts/run_layerwise_locality.py`

- [ ] **Step 1: Write the script**

Create `scripts/run_layerwise_locality.py`:

```python
"""
CLI: layer-wise dynamics + locality experiments (paper Figs. 12-15).

Usage:
    uv run python scripts/run_layerwise_locality.py
"""

from __future__ import annotations

import json

from emotion_scope.config import DATA_DIR
from emotion_scope.extract import EmotionExtractor
from emotion_scope.layerwise import (
    extract_vectors_at_layers,
    get_last_token_activation,
    sample_layers,
    score_activation_against_vector,
)
from emotion_scope.models import load_model
from emotion_scope.tracking import init_run

EMOTIONS_OF_INTEREST = [
    "happy", "sad", "afraid", "calm", "angry", "furious",
    "confident", "resentful", "grateful", "lonely", "content",
]


def main() -> None:
    scenarios = json.loads((DATA_DIR / "validation" / "locality_scenarios.json").read_text(encoding="utf-8"))

    model, tokenizer, backend, info = load_model(backend="huggingface", run_smoke_test=False)
    extractor = EmotionExtractor(model, tokenizer, backend, info)
    layers = sample_layers(info["n_layers"], n_samples=6)
    print(f"[layerwise] Sampled layers: {layers}")

    print("[layerwise] Extracting vectors at each sampled layer (this takes a few minutes)...")
    vectors_by_layer = extract_vectors_at_layers(extractor, layers, EMOTIONS_OF_INTEREST, stories_per_emotion=10)

    tracker = init_run(part="part2", job_type="layerwise-locality", config={"layers": layers})

    negation_rows = []
    for item in scenarios["negation"]:
        emotion = item["emotion"]
        if emotion not in EMOTIONS_OF_INTEREST:
            continue
        for layer in layers:
            vector = vectors_by_layer[layer][emotion]
            pos_act = get_last_token_activation(model, tokenizer, backend, layer, item["positive"])
            neg_act = get_last_token_activation(model, tokenizer, backend, layer, item["negated"])
            negation_rows.append([
                emotion, layer,
                score_activation_against_vector(pos_act, vector),
                score_activation_against_vector(neg_act, vector),
            ])
    print(f"[layerwise] Negation: {len(negation_rows)} (emotion, layer) rows")

    persistence_rows = []
    for item in scenarios["prefix_persistence"]:
        emotion = item["target_emotion"]
        if emotion not in EMOTIONS_OF_INTEREST:
            continue
        text_a = f"{item['prefix_a']} {item['shared_suffix']}"
        text_b = f"{item['prefix_b']} {item['shared_suffix']}"
        for layer in layers:
            vector = vectors_by_layer[layer][emotion]
            act_a = get_last_token_activation(model, tokenizer, backend, layer, text_a)
            act_b = get_last_token_activation(model, tokenizer, backend, layer, text_b)
            persistence_rows.append([
                emotion, layer,
                score_activation_against_vector(act_a, vector),
                score_activation_against_vector(act_b, vector),
            ])
    print(f"[layerwise] Prefix persistence: {len(persistence_rows)} (emotion, layer) rows")

    binding_rows = []
    for item in scenarios["person_binding"]:
        emo_a, emo_b = item["emotion_a"], item["emotion_b"]
        if emo_a not in EMOTIONS_OF_INTEREST or emo_b not in EMOTIONS_OF_INTEREST:
            continue
        text_ref_a = f"{item['setup']} {item['reference_a']}"
        text_ref_b = f"{item['setup']} {item['reference_b']}"
        for layer in layers:
            vec_a = vectors_by_layer[layer][emo_a]
            vec_b = vectors_by_layer[layer][emo_b]
            act_ref_a = get_last_token_activation(model, tokenizer, backend, layer, text_ref_a)
            act_ref_b = get_last_token_activation(model, tokenizer, backend, layer, text_ref_b)
            binding_rows.append([
                item["person_a"], emo_a, item["person_b"], emo_b, layer,
                score_activation_against_vector(act_ref_a, vec_a),  # A's own emotion at A's reference
                score_activation_against_vector(act_ref_a, vec_b),  # B's emotion (should be lower) at A's reference
                score_activation_against_vector(act_ref_b, vec_b),  # B's own emotion at B's reference
                score_activation_against_vector(act_ref_b, vec_a),  # A's emotion (should be lower) at B's reference
            ])
    print(f"[layerwise] Person binding: {len(binding_rows)} rows")

    tracker.log_table("negation", columns=["emotion", "layer", "positive_score", "negated_score"], data=negation_rows)
    tracker.log_table("prefix_persistence", columns=["emotion", "layer", "score_hard_prefix", "score_good_prefix"], data=persistence_rows)
    tracker.log_table(
        "person_binding",
        columns=["person_a", "emotion_a", "person_b", "emotion_b", "layer",
                 "a_ref_own_emotion", "a_ref_other_emotion", "b_ref_own_emotion", "b_ref_other_emotion"],
        data=binding_rows,
    )
    tracker.finish()
    print("[layerwise] Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/run_layerwise_locality.py
```

Expected: prints progress and row counts, exits 0. Inspect the logged tables (or add a quick `print()` of a few rows if not using wandb) — negation scores should generally be higher for `positive` than `negated` at later layers; person-binding's `*_own_emotion` scores should generally exceed `*_other_emotion` scores at later layers, reproducing the paper's qualitative claim that late layers carry more contextually-integrated, correctly-bound emotional content than early layers.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_layerwise_locality.py
git commit -m "Add scripts/run_layerwise_locality.py"
```

---

### Task 4: Author the hidden-emotion dialogue dataset

Five conditions (paper Table 5): naturally expressed, hidden, unexpressed (neutral topic), unexpressed (writing a story about someone else), unexpressed (discussing someone else). Five emotions x five conditions x two examples = 50 entries.

**Files:**
- Create: `data/validation/hidden_emotion_dialogues.json`

- [ ] **Step 1: Write the dataset**

Create `data/validation/hidden_emotion_dialogues.json`:

```json
[
  {"condition": "natural", "emotion": "happy", "text": "I got the callback! I have been smiling nonstop since the email came in."},
  {"condition": "natural", "emotion": "happy", "text": "Everything about today has been wonderful, I can't stop grinning."},
  {"condition": "natural", "emotion": "sad", "text": "I keep tearing up every time I think about it, I don't know how to make it stop."},
  {"condition": "natural", "emotion": "sad", "text": "There's this heaviness in my chest that won't go away."},
  {"condition": "natural", "emotion": "afraid", "text": "My hands won't stop shaking, I keep checking the door is locked."},
  {"condition": "natural", "emotion": "afraid", "text": "I can't shake the feeling that something terrible is about to happen."},
  {"condition": "natural", "emotion": "angry", "text": "I am so sick of being ignored, I could scream right now."},
  {"condition": "natural", "emotion": "angry", "text": "Every time I think about what he said, my blood boils again."},
  {"condition": "natural", "emotion": "calm", "text": "I feel completely at ease, like nothing could rattle me today."},
  {"condition": "natural", "emotion": "calm", "text": "There's a stillness in me right now that I want to hold onto."},
  {"condition": "hidden", "emotion": "happy", "text": "She kept her expression flat, but she checked her phone for the reply every thirty seconds."},
  {"condition": "hidden", "emotion": "happy", "text": "He shrugged when they asked how it went, already planning how to tell everyone else the good news later."},
  {"condition": "hidden", "emotion": "sad", "text": "He said he was fine and changed the subject before anyone could ask again."},
  {"condition": "hidden", "emotion": "sad", "text": "She kept working through lunch instead of sitting with the others like she usually did."},
  {"condition": "hidden", "emotion": "afraid", "text": "She laughed it off and said it was nothing, though she'd already checked the locks twice."},
  {"condition": "hidden", "emotion": "afraid", "text": "He said he wasn't worried, then took the long way home past the well-lit streets."},
  {"condition": "hidden", "emotion": "angry", "text": "He nodded politely and said it didn't matter, then closed the door a little harder than usual."},
  {"condition": "hidden", "emotion": "angry", "text": "She thanked them for the feedback in a flat voice and didn't say another word for the rest of the meeting."},
  {"condition": "hidden", "emotion": "calm", "text": "He kept fidgeting and pacing the room, though inside he had already made peace with whatever came next."},
  {"condition": "hidden", "emotion": "calm", "text": "She snapped at the question out of habit, even though nothing about the situation actually bothered her anymore."},
  {"condition": "unexpressed_neutral", "emotion": "happy", "text": "She had just found out she got the job, but the meeting moved straight into discussing the printer schedule for next week."},
  {"condition": "unexpressed_neutral", "emotion": "happy", "text": "He'd been grinning since the acceptance letter arrived, and spent the call comparing bus routes with a coworker."},
  {"condition": "unexpressed_neutral", "emotion": "sad", "text": "He had been quietly grieving all morning, then spent the call walking a coworker through how to reset a router password."},
  {"condition": "unexpressed_neutral", "emotion": "sad", "text": "She'd barely slept since the news, and the conversation stayed entirely on the office's new seating chart."},
  {"condition": "unexpressed_neutral", "emotion": "afraid", "text": "She'd been terrified since the phone call an hour ago, but the conversation stayed on comparing flight layover times."},
  {"condition": "unexpressed_neutral", "emotion": "afraid", "text": "His stomach had been in knots all day, yet the meeting only covered which conference room to book."},
  {"condition": "unexpressed_neutral", "emotion": "angry", "text": "He was still furious about the missed deadline, yet the whole conversation was about which font to use for the report."},
  {"condition": "unexpressed_neutral", "emotion": "angry", "text": "She was seething from the earlier call, but spent the meeting discussing the vending machine's broken card reader."},
  {"condition": "unexpressed_neutral", "emotion": "calm", "text": "She felt entirely at peace after the decision, and the conversation went on to cover the office's new recycling policy."},
  {"condition": "unexpressed_neutral", "emotion": "calm", "text": "He felt settled for the first time in weeks, and the call stayed focused on updating the shared calendar."},
  {"condition": "unexpressed_story_writing", "emotion": "happy", "text": "Though she had just gotten the good news, she spent the afternoon drafting a short story about a lighthouse keeper who missed his family more with each passing winter."},
  {"condition": "unexpressed_story_writing", "emotion": "happy", "text": "Still glowing from the news, he wrote a scene where a tired traveler finally admitted he was lost and afraid to ask for directions."},
  {"condition": "unexpressed_story_writing", "emotion": "sad", "text": "Even as the grief sat heavy in his chest, he worked on a scene where two kids raced down a hill, breathless with excitement over a new bike."},
  {"condition": "unexpressed_story_writing", "emotion": "sad", "text": "Numb from the loss, she typed out a paragraph about a chef proudly presenting his first perfect souffle."},
  {"condition": "unexpressed_story_writing", "emotion": "afraid", "text": "With her pulse still racing from the news, she typed out a paragraph describing a monk calmly tending a garden at dawn."},
  {"condition": "unexpressed_story_writing", "emotion": "afraid", "text": "Hands still shaking, he outlined a chapter where two old friends laughed for hours over dinner."},
  {"condition": "unexpressed_story_writing", "emotion": "angry", "text": "Still fuming from the argument, he wrote a scene where an old man thanked a stranger for returning his lost wallet."},
  {"condition": "unexpressed_story_writing", "emotion": "angry", "text": "Jaw still tight from the call, she drafted a paragraph about a child falling asleep peacefully during a long car ride."},
  {"condition": "unexpressed_story_writing", "emotion": "calm", "text": "Feeling entirely settled, she outlined a chapter where a pilot scrambled to restart a stalled engine mid-flight."},
  {"condition": "unexpressed_story_writing", "emotion": "calm", "text": "At total ease, he wrote a scene of a soldier bracing for the sound of incoming artillery."},
  {"condition": "unexpressed_discussing_others", "emotion": "happy", "text": "She had been beaming all morning since the acceptance letter, and spent lunch listening to a coworker complain about a frustrating printer jam."},
  {"condition": "unexpressed_discussing_others", "emotion": "happy", "text": "Still glowing from the good news, he let his roommate vent about how anxious she was for tomorrow's presentation."},
  {"condition": "unexpressed_discussing_others", "emotion": "sad", "text": "He was still numb from the loss, but let his friend talk at length about how excited she was for the upcoming trip."},
  {"condition": "unexpressed_discussing_others", "emotion": "sad", "text": "Heavy-hearted since the call, she spent the evening hearing her brother describe how proud he was of his new promotion."},
  {"condition": "unexpressed_discussing_others", "emotion": "afraid", "text": "Her stomach had been in knots since the diagnosis call, yet she spent the evening hearing her brother talk about how confident he felt going into his interview."},
  {"condition": "unexpressed_discussing_others", "emotion": "afraid", "text": "Still shaken from the near-miss on the highway, he listened as his coworker cheerfully described her calm weekend at the lake."},
  {"condition": "unexpressed_discussing_others", "emotion": "angry", "text": "He was still seething over the parking dispute, but nodded along as his neighbor described how calm the weekend had been."},
  {"condition": "unexpressed_discussing_others", "emotion": "angry", "text": "Still fuming from the meeting, she listened to her sister happily describe her new apartment for twenty minutes."},
  {"condition": "unexpressed_discussing_others", "emotion": "calm", "text": "Feeling entirely settled after the decision, she listened patiently as her cousin described how desperate things had gotten at work."},
  {"condition": "unexpressed_discussing_others", "emotion": "calm", "text": "At peace with the outcome, he let his friend go on about how furious he still was over the referee's call."}
]
```

- [ ] **Step 2: Verify it parses and has the right shape**

```bash
uv run python -c "
import json
from collections import Counter
data = json.load(open('data/validation/hidden_emotion_dialogues.json'))
assert len(data) == 50, len(data)
by_condition = Counter(d['condition'] for d in data)
print(by_condition)
assert set(by_condition.keys()) == {'natural', 'hidden', 'unexpressed_neutral', 'unexpressed_story_writing', 'unexpressed_discussing_others'}
"
```

Expected: prints a `Counter` with all 5 conditions at count 10 each

- [ ] **Step 3: Commit**

```bash
git add data/validation/hidden_emotion_dialogues.json
git commit -m "Add hidden-emotion dialogue dataset (5 conditions x 5 emotions x 2)"
```

---

### Task 5: `emotion_scope/hidden_emotion.py` — mixed LR probe

**Files:**
- Create: `emotion_scope/hidden_emotion.py`
- Test: `tests/test_hidden_emotion.py` (new)

- [ ] **Step 1: Write the failing fast tests**

Create `tests/test_hidden_emotion.py`:

```python
"""Tests for emotion_scope.hidden_emotion — mixed logistic-regression probe."""

import json

import numpy as np
import pytest


def test_load_hidden_emotion_dataset(tmp_path):
    from emotion_scope.hidden_emotion import load_hidden_emotion_dataset

    path = tmp_path / "d.json"
    path.write_text(json.dumps([{"condition": "natural", "emotion": "happy", "text": "hi"}]))
    data = load_hidden_emotion_dataset(path)
    assert len(data) == 1
    assert data[0]["emotion"] == "happy"


def test_train_and_evaluate_perfect_separation():
    from emotion_scope.hidden_emotion import train_and_evaluate

    rng = np.random.RandomState(0)
    happy = rng.normal(loc=5.0, scale=0.1, size=(20, 4))
    sad = rng.normal(loc=-5.0, scale=0.1, size=(20, 4))
    activations = np.vstack([happy, sad])
    labels = ["happy"] * 20 + ["sad"] * 20
    conditions = ["natural"] * 40

    result = train_and_evaluate(activations, labels, conditions)
    assert result["overall_accuracy"] > 0.9
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_hidden_emotion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `hidden_emotion.py`**

Create `emotion_scope/hidden_emotion.py`:

```python
"""
Mixed logistic-regression emotion probe (paper Table 5): can we detect a
character's emotional state even when it is hidden, unexpressed, or the
conversation has moved on to someone else entirely?
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_hidden_emotion_dataset(path) -> List[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def train_and_evaluate(
    activations: np.ndarray,
    labels: List[str],
    conditions: List[str],
    test_size: float = 0.3,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Train a logistic-regression classifier on (activation -> emotion label)
    and report accuracy overall and broken down by condition on the held-out
    test split.
    """
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(activations[train_idx], [labels[i] for i in train_idx])

    preds = clf.predict(activations[test_idx])
    truth = [labels[i] for i in test_idx]
    test_conditions = [conditions[i] for i in test_idx]

    overall_acc = float(np.mean([p == t for p, t in zip(preds, truth)]))

    by_condition: Dict[str, List[bool]] = {}
    for pred, true, cond in zip(preds, truth, test_conditions):
        by_condition.setdefault(cond, []).append(pred == true)

    result = {"overall_accuracy": overall_acc}
    for cond, hits in by_condition.items():
        result[f"accuracy_{cond}"] = float(np.mean(hits))
    return result
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_hidden_emotion.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/hidden_emotion.py tests/test_hidden_emotion.py
git commit -m "Add emotion_scope.hidden_emotion — mixed LR probe"
```

---

### Task 6: `scripts/run_hidden_emotion_probe.py`

**Files:**
- Create: `scripts/run_hidden_emotion_probe.py`

- [ ] **Step 1: Write the script**

Create `scripts/run_hidden_emotion_probe.py`:

```python
"""
CLI: train and evaluate the mixed LR emotion probe on the hidden-emotion
dataset (paper Table 5).

Usage:
    uv run python scripts/run_hidden_emotion_probe.py
"""

from __future__ import annotations

import numpy as np

from emotion_scope.config import DATA_DIR
from emotion_scope.hidden_emotion import load_hidden_emotion_dataset, train_and_evaluate
from emotion_scope.layerwise import get_last_token_activation
from emotion_scope.models import load_model
from emotion_scope.tracking import init_run


def main() -> None:
    dataset = load_hidden_emotion_dataset(DATA_DIR / "validation" / "hidden_emotion_dialogues.json")
    model, tokenizer, backend, info = load_model(backend="huggingface", run_smoke_test=False)
    probe_layer = info["probe_layer"]

    print(f"[hidden-emotion] Extracting activations for {len(dataset)} examples at layer {probe_layer}...")
    activations = np.stack([
        get_last_token_activation(model, tokenizer, backend, probe_layer, item["text"]).numpy()
        for item in dataset
    ])
    labels = [item["emotion"] for item in dataset]
    conditions = [item["condition"] for item in dataset]

    result = train_and_evaluate(activations, labels, conditions)

    print("\n[hidden-emotion] Mixed LR probe accuracy by condition:")
    print(f"  overall: {result['overall_accuracy']:.3f}")
    for key, value in result.items():
        if key.startswith("accuracy_"):
            print(f"  {key.replace('accuracy_', ''):28s}: {value:.3f}")
    print("(chance level for 5-way classification = 0.20; paper reports 0.386-0.826 across conditions for 15-way classification)")

    tracker = init_run(part="part2", job_type="hidden-emotion-probe", config={"n_examples": len(dataset)})
    tracker.log_metrics(result)
    tracker.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/run_hidden_emotion_probe.py
```

Expected: prints per-condition accuracy, all above the 0.20 chance level for 5-way classification, exits 0

- [ ] **Step 3: Commit**

```bash
git add scripts/run_hidden_emotion_probe.py
git commit -m "Add scripts/run_hidden_emotion_probe.py"
```

---

### Task 7: Author the deflection-scenarios dataset

Contrastive pairs: "deflected" (calm exterior, suppressed different emotion underneath) vs. "genuine" calm.

**Files:**
- Create: `data/validation/deflection_scenarios.json`

- [ ] **Step 1: Write the dataset**

Create `data/validation/deflection_scenarios.json`:

```json
[
  {"deflected": "He kept his voice level and said it was fine, though his jaw was clenched the entire time.", "genuine": "He said it was fine, and meant it completely, with no tension in his voice at all."},
  {"deflected": "She smiled through the whole meeting, even as the comment replayed in her mind over and over.", "genuine": "She smiled through the whole meeting, genuinely enjoying every part of it."},
  {"deflected": "He said 'no worries' and went back to his desk, his hands still trembling slightly from the exchange.", "genuine": "He said 'no worries' and went back to his desk, already thinking about lunch."},
  {"deflected": "She thanked him for the feedback in an even tone, filing away every word to bring up later.", "genuine": "She thanked him for the feedback in an even tone, appreciating the honesty."},
  {"deflected": "He nodded along calmly, while quietly deciding he would never work with that team again.", "genuine": "He nodded along calmly, fully at ease with how the project was going."},
  {"deflected": "She kept her tone polite on the call, though she'd already drafted three angry replies she chose not to send.", "genuine": "She kept her tone polite on the call, simply because there was nothing to be upset about."},
  {"deflected": "He said it was no big deal, but he skipped the team lunch for the rest of the week.", "genuine": "He said it was no big deal, and joined the team lunch as usual."},
  {"deflected": "She agreed to the new schedule without complaint, then spent the drive home replaying every unfair part of it.", "genuine": "She agreed to the new schedule without complaint, glad to have it settled."},
  {"deflected": "He kept his answers short and even, careful not to let anything slip about how the decision had landed.", "genuine": "He kept his answers short and even, simply because there wasn't much else to say."},
  {"deflected": "She wished them well at the announcement, already rehearsing what she'd say if anyone asked her honest opinion.", "genuine": "She wished them well at the announcement, happy to see the project move forward."}
]
```

- [ ] **Step 2: Verify it parses**

```bash
uv run python -c "
import json
data = json.load(open('data/validation/deflection_scenarios.json'))
assert len(data) == 10
for d in data:
    assert 'deflected' in d and 'genuine' in d
print('OK')
"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add data/validation/deflection_scenarios.json
git commit -m "Add deflection-scenarios dataset"
```

---

### Task 8: `emotion_scope/deflection.py`

**Files:**
- Create: `emotion_scope/deflection.py`
- Test: `tests/test_deflection.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_deflection.py`:

```python
"""Tests for emotion_scope.deflection — emotion deflection vector."""

import torch


def test_compute_deflection_vector_points_from_genuine_to_deflected():
    from emotion_scope.deflection import compute_deflection_vector

    deflected = [torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.2, 0.8, 0.0])]
    genuine = [torch.tensor([-1.0, -1.0, 0.0]), torch.tensor([-1.2, -0.8, 0.0])]

    v = compute_deflection_vector(deflected, genuine)
    assert abs(v.norm().item() - 1.0) < 1e-5  # L2-normalized
    assert v[0] > 0  # points toward the deflected cluster, away from genuine
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_deflection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `emotion_scope/deflection.py`:

```python
"""
Emotion deflection vector (paper: representation of an outwardly-calm
character deflecting a different underlying emotion).

v_deflection = mean(deflected activations) - mean(genuine-calm activations),
L2-normalized — same contrastive-mean-difference recipe as extract.py.
"""

from __future__ import annotations

from typing import List

import torch


def compute_deflection_vector(
    deflected_activations: List[torch.Tensor],
    genuine_activations: List[torch.Tensor],
) -> torch.Tensor:
    deflected_mean = torch.stack(deflected_activations).mean(dim=0)
    genuine_mean = torch.stack(genuine_activations).mean(dim=0)
    diff = deflected_mean - genuine_mean
    return diff / diff.norm()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_deflection.py -v`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/deflection.py tests/test_deflection.py
git commit -m "Add emotion_scope.deflection — deflection vector"
```

---

### Task 9: `scripts/run_deflection_vector.py`

**Files:**
- Create: `scripts/run_deflection_vector.py`

- [ ] **Step 1: Write the script**

Create `scripts/run_deflection_vector.py`:

```python
"""
CLI: compute the emotion deflection vector and compare it against the main
40-emotion vector set.

Usage:
    uv run python scripts/run_deflection_vector.py --vectors results/vectors/google_gemma-2-2b-it.pt
"""

from __future__ import annotations

import argparse
import json

import torch.nn.functional as F

from emotion_scope.config import DATA_DIR
from emotion_scope.deflection import compute_deflection_vector
from emotion_scope.extract import EmotionExtractor
from emotion_scope.layerwise import get_last_token_activation
from emotion_scope.models import load_model
from emotion_scope.tracking import init_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and inspect the emotion deflection vector")
    parser.add_argument("--vectors", required=True)
    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    vectors = saved["vectors"]
    model_name = saved["model_info"]["model_name"]
    probe_layer = saved["model_info"]["probe_layer"]

    scenarios = json.loads((DATA_DIR / "validation" / "deflection_scenarios.json").read_text(encoding="utf-8"))
    model, tokenizer, backend, info = load_model(model_name=model_name, backend="huggingface", run_smoke_test=False)

    deflected_acts = [get_last_token_activation(model, tokenizer, backend, probe_layer, s["deflected"]) for s in scenarios]
    genuine_acts = [get_last_token_activation(model, tokenizer, backend, probe_layer, s["genuine"]) for s in scenarios]

    deflection_vector = compute_deflection_vector(deflected_acts, genuine_acts)

    print("[deflection] Cosine similarity between the deflection vector and existing emotion vectors:")
    rows = []
    similarities = []
    for name, vector in vectors.items():
        sim = float(F.cosine_similarity(deflection_vector.unsqueeze(0), vector.unsqueeze(0)).item())
        similarities.append((name, sim))
        rows.append([name, sim])
    similarities.sort(key=lambda x: x[1], reverse=True)
    for name, sim in similarities[:5]:
        print(f"  {name:15s} {sim:+.3f}")

    tracker = init_run(part="part2", job_type="deflection-vector", config={"model": model_name})
    tracker.log_table("deflection_vs_emotions", columns=["emotion", "cosine_similarity"], data=rows)
    tracker.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/run_deflection_vector.py --vectors results/vectors/google_gemma-2-2b-it.pt
```

Expected: prints the top-5 emotion vectors most aligned with the deflection direction, exits 0. There is no strict pass/fail threshold here — this is an exploratory finding (matching the paper's own framing of the deflection vector as an interesting side observation, not a validated gate).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_deflection_vector.py
git commit -m "Add scripts/run_deflection_vector.py"
```

---

### Task 10: Extend `speakers.py` — full 2x2 grid, non-privilege check, story-vs-present r²

**Files:**
- Modify: `emotion_scope/speakers.py`
- Test: `tests/test_speakers_grid.py` (new)

- [ ] **Step 1: Write the failing fast test for text substitution**

Create `tests/test_speakers_grid.py`:

```python
"""Tests for the speaker 2x2 grid / non-privilege extensions to SpeakerSeparator."""

import pytest


def test_relabel_speakers_to_generic_persons():
    from emotion_scope.speakers import relabel_speakers_to_generic

    text = "Speaker A: I'm thrilled about this.\nSpeaker B: That's great to hear.\nSpeaker A: Thanks!"
    relabeled = relabel_speakers_to_generic(text)
    assert "Speaker A" not in relabeled
    assert "Speaker B" not in relabeled
    assert "Person 1" in relabeled
    assert "Person 2" in relabeled
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_speakers_grid.py -v`
Expected: FAIL with `ImportError: cannot import name 'relabel_speakers_to_generic'`

- [ ] **Step 3: Add `relabel_speakers_to_generic` and `get_speaker_b_final_turn_text`**

Append to `emotion_scope/speakers.py` (module level, after the imports, before `SpeakerSeparator`):

```python
def relabel_speakers_to_generic(dialogue_text: str) -> str:
    """Replace 'Speaker A'/'Speaker B' with 'Person 1'/'Person 2' for the non-privilege check."""
    return dialogue_text.replace("Speaker A", "Person 1").replace("Speaker B", "Person 2")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_speakers_grid.py -v`
Expected: `1 passed`

- [ ] **Step 5: Add the 2x2 grid extraction method**

Append this method inside the `SpeakerSeparator` class in `emotion_scope/speakers.py` (place it after `_extract_dialogue_activations`):

```python
    def _get_speaker_b_final_turn_text(self, dialogue_text: str) -> Optional[str]:
        """
        The default dialogue format is A, B, A, B — Speaker B's final turn is
        the end of the whole dialogue, so this is just the full text.
        """
        text = dialogue_text.strip()
        return text if text else None

    def extract_2x2_grid(self, dialogues_path: Optional[str] = None) -> dict:
        """
        Extract all four combinations of (token position) x (emotion label):
        A-tok/A-emo, A-tok/B-emo, B-tok/A-emo, B-tok/B-emo.

        A-tok/A-emo and B-tok/B-emo are the "present speaker" probes; A-tok/B-emo
        and B-tok/A-emo are the "other speaker" probes — matching the paper's
        four-cell grid (Fig. 17-18).
        """
        dialogues = self._load_dialogues(dialogues_path)

        a_tok_records: List[dict] = []
        b_tok_records: List[dict] = []
        for entry in tqdm(dialogues, desc="Extracting 2x2 grid"):
            a_text = self._get_speaker_a_final_turn_text(entry["dialogue"])
            b_text = self._get_speaker_b_final_turn_text(entry["dialogue"])
            if a_text is None or b_text is None:
                continue
            a_act = self._get_activation_at_last_content_token(a_text)
            b_act = self._get_activation_at_last_content_token(b_text)
            if a_act is None or b_act is None:
                continue
            a_tok_records.append({"emotion_a": entry["emotion_a"], "emotion_b": entry["emotion_b"], "activation": a_act})
            b_tok_records.append({"emotion_a": entry["emotion_a"], "emotion_b": entry["emotion_b"], "activation": b_act})

        a_tok_a_emo, a_tok_b_emo = self._compute_speaker_vectors(a_tok_records)
        b_tok_a_emo, b_tok_b_emo = self._compute_speaker_vectors(b_tok_records)
        # _compute_speaker_vectors returns (grouped-by-emotion_a, grouped-by-emotion_b)
        # for whichever token-position records it's given, so:
        #   a_tok_a_emo = A-tok / A-emo (present speaker)
        #   a_tok_b_emo = A-tok / B-emo (other speaker)
        #   b_tok_b_emo = B-tok / B-emo (present speaker)
        #   b_tok_a_emo = B-tok / A-emo (other speaker)
        return {
            "a_tok_a_emo": a_tok_a_emo,
            "a_tok_b_emo": a_tok_b_emo,
            "b_tok_a_emo": b_tok_a_emo,
            "b_tok_b_emo": b_tok_b_emo,
        }

    def test_non_privilege(self, dialogues_path: Optional[str] = None) -> dict:
        """
        Re-run the current-speaker (A-tok/A-emo) extraction with 'Speaker A'/
        'Speaker B' relabeled to 'Person 1'/'Person 2', and compare per-emotion
        cosine similarity against the original — high similarity means the
        representation isn't privileged to the Human/Assistant roles.
        """
        dialogues = self._load_dialogues(dialogues_path)
        original_records = []
        relabeled_records = []
        for entry in tqdm(dialogues, desc="Non-privilege check"):
            original_text = self._get_speaker_a_final_turn_text(entry["dialogue"])
            relabeled_text = self._get_speaker_a_final_turn_text(relabel_speakers_to_generic(entry["dialogue"]))
            if original_text is None or relabeled_text is None:
                continue
            orig_act = self._get_activation_at_last_content_token(original_text)
            relabel_act = self._get_activation_at_last_content_token(relabeled_text)
            if orig_act is None or relabel_act is None:
                continue
            original_records.append({"emotion_a": entry["emotion_a"], "emotion_b": entry["emotion_b"], "activation": orig_act})
            relabeled_records.append({"emotion_a": entry["emotion_a"], "emotion_b": entry["emotion_b"], "activation": relabel_act})

        original_current, _ = self._compute_speaker_vectors(original_records)
        relabeled_current, _ = self._compute_speaker_vectors(relabeled_records)

        common = sorted(set(original_current.keys()) & set(relabeled_current.keys()))
        scores = {}
        for name in common:
            o = F.normalize(original_current[name], dim=0)
            r = F.normalize(relabeled_current[name], dim=0)
            scores[name] = torch.dot(o, r).item()
        mean_score = sum(scores.values()) / len(scores) if scores else 0.0
        return {"per_emotion": scores, "mean_cosine": mean_score, "passed": mean_score > 0.5}

    def compare_to_story_probes(
        self,
        story_vectors: Dict[str, torch.Tensor],
        implicit_scenarios_path,
    ) -> dict:
        """
        Compare story-based probe scores against present-speaker (A-tok/A-emo)
        probe scores on the implicit-emotion scenarios dataset. Returns the
        r-squared between the two sets of scores, per the paper's finding of
        r-squared=0.66 between story and present-speaker probes.
        """
        scenarios = json.loads(Path(implicit_scenarios_path).read_text(encoding="utf-8"))
        grid = self.extract_2x2_grid()
        present_vectors = grid["a_tok_a_emo"]

        story_matrix = torch.stack(list(story_vectors.values()))
        story_names = list(story_vectors.keys())
        present_matrix = torch.stack(list(present_vectors.values()))
        present_names = list(present_vectors.keys())

        story_scores_all: List[float] = []
        present_scores_all: List[float] = []
        for scen in scenarios:
            act = self._get_activation_at_last_content_token(scen["scenario"])
            if act is None:
                continue
            story_scores = F.normalize(act.unsqueeze(0), dim=1) @ F.normalize(story_matrix, dim=1).T
            present_scores = F.normalize(act.unsqueeze(0), dim=1) @ F.normalize(present_matrix, dim=1).T
            story_scores_all.extend(story_scores.squeeze(0).tolist())
            present_scores_all.extend(present_scores.squeeze(0).tolist())

        if len(story_scores_all) < 2:
            return {"r_squared": 0.0, "n_points": len(story_scores_all)}

        from scipy.stats import pearsonr
        r, _ = pearsonr(story_scores_all, present_scores_all)
        return {"r_squared": r ** 2, "correlation": r, "n_points": len(story_scores_all)}
```

- [ ] **Step 6: Add the required imports to `speakers.py`**

At the top of `emotion_scope/speakers.py`, the existing imports already include `torch`, `torch.nn.functional as F`, `json`, `Path` — verify these are present (they are, per the file's current imports); no changes needed there.

- [ ] **Step 7: Write the slow test for the new methods**

Append to `tests/test_speakers_grid.py`:

```python
@pytest.mark.slow
def test_extract_2x2_grid_returns_four_cells():
    from emotion_scope.models import load_model
    from emotion_scope.speakers import SpeakerSeparator

    model, tokenizer, backend, info = load_model(backend="huggingface", run_smoke_test=False)
    separator = SpeakerSeparator(model, tokenizer, backend, info)
    grid = separator.extract_2x2_grid()

    assert set(grid.keys()) == {"a_tok_a_emo", "a_tok_b_emo", "b_tok_a_emo", "b_tok_b_emo"}
    for cell in grid.values():
        assert len(cell) > 0


@pytest.mark.slow
def test_non_privilege_check_runs():
    from emotion_scope.models import load_model
    from emotion_scope.speakers import SpeakerSeparator

    model, tokenizer, backend, info = load_model(backend="huggingface", run_smoke_test=False)
    separator = SpeakerSeparator(model, tokenizer, backend, info)
    result = separator.test_non_privilege()

    assert "mean_cosine" in result
    assert -1.0 <= result["mean_cosine"] <= 1.0
```

- [ ] **Step 8: Run all speaker tests**

Run: `uv run pytest tests/test_speakers_grid.py --runslow -v`
Expected: `4 passed`

- [ ] **Step 9: Commit**

```bash
git add emotion_scope/speakers.py tests/test_speakers_grid.py
git commit -m "Extend SpeakerSeparator with 2x2 grid, non-privilege check, story-probe r-squared"
```

---

### Task 11: `scripts/run_speaker_full_analysis.py`

**Files:**
- Create: `scripts/run_speaker_full_analysis.py`

- [ ] **Step 1: Write the script**

Create `scripts/run_speaker_full_analysis.py`:

```python
"""
CLI: full speaker-representation analysis — 2x2 grid, non-privilege check,
and story-vs-present-speaker probe comparison (paper Figs. 17-19).

Usage:
    uv run python scripts/run_speaker_full_analysis.py --vectors results/vectors/google_gemma-2-2b-it.pt
"""

from __future__ import annotations

import argparse

import torch.nn.functional as F

from emotion_scope.config import DATA_DIR
from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.speakers import SpeakerSeparator
from emotion_scope.tracking import init_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Speaker 2x2 grid / non-privilege / story-probe comparison")
    parser.add_argument("--vectors", required=True)
    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    story_vectors = saved["vectors"]
    model_name = saved["model_info"]["model_name"]

    model, tokenizer, backend, info = load_model(model_name=model_name, backend="huggingface", run_smoke_test=False)
    separator = SpeakerSeparator(model, tokenizer, backend, info)

    print("[speaker-analysis] Extracting 2x2 probe grid...")
    grid = separator.extract_2x2_grid()
    common_emotions = sorted(
        set(grid["a_tok_a_emo"]) & set(grid["a_tok_b_emo"]) & set(grid["b_tok_a_emo"]) & set(grid["b_tok_b_emo"])
    )
    present_vs_present = [
        F.cosine_similarity(grid["a_tok_a_emo"][e].unsqueeze(0), grid["b_tok_b_emo"][e].unsqueeze(0)).item()
        for e in common_emotions
    ]
    other_vs_other = [
        F.cosine_similarity(grid["a_tok_b_emo"][e].unsqueeze(0), grid["b_tok_a_emo"][e].unsqueeze(0)).item()
        for e in common_emotions
    ]
    print(f"  Mean present-vs-present similarity (A-tok/A-emo vs B-tok/B-emo): {sum(present_vs_present)/len(present_vs_present):.3f}")
    print(f"  Mean other-vs-other similarity (A-tok/B-emo vs B-tok/A-emo):     {sum(other_vs_other)/len(other_vs_other):.3f}")
    print("  (paper reports these two groups clustering together, while present and other are nearly orthogonal to each other)")

    print("\n[speaker-analysis] Running non-privilege check (Speaker A/B vs Person 1/2)...")
    non_privilege = separator.test_non_privilege()
    print(f"  Mean cosine similarity: {non_privilege['mean_cosine']:.3f} (threshold > 0.5) — "
          f"{'PASS' if non_privilege['passed'] else 'FAIL'}")

    print("\n[speaker-analysis] Comparing story probes to present-speaker probes on implicit scenarios...")
    comparison = separator.compare_to_story_probes(
        story_vectors, DATA_DIR / "validation" / "implicit_scenarios.json"
    )
    print(f"  r-squared = {comparison['r_squared']:.3f} over {comparison['n_points']} points "
          f"(paper reports r-squared=0.66)")

    tracker = init_run(part="part2", job_type="speaker-full-analysis", config={"model": model_name})
    tracker.log_metrics({
        "present_vs_present_similarity": sum(present_vs_present) / len(present_vs_present),
        "other_vs_other_similarity": sum(other_vs_other) / len(other_vs_other),
        "non_privilege_mean_cosine": non_privilege["mean_cosine"],
        "story_present_r_squared": comparison["r_squared"],
    })
    tracker.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/run_speaker_full_analysis.py --vectors results/vectors/google_gemma-2-2b-it.pt
```

Expected: prints all three sub-analyses, exits 0

- [ ] **Step 3: Commit**

```bash
git add scripts/run_speaker_full_analysis.py
git commit -m "Add scripts/run_speaker_full_analysis.py"
```

---

### Task 12: `scripts/run_llm_judge_valence_arousal.py`

Uses the loaded model itself as the "LLM judge" (no external API needed, consistent with this project's local-model-only approach), and correlates against `config.py`'s existing Russell(1980)/NRC-based valence/arousal values as the "human ratings" comparison point.

**Files:**
- Create: `scripts/run_llm_judge_valence_arousal.py`

- [ ] **Step 1: Write the script**

Create `scripts/run_llm_judge_valence_arousal.py`:

```python
"""
CLI: LLM-judged valence/arousal ratings for all 40 emotions, correlated
against the hand-assigned Russell(1980)/NRC-based values in config.py
(paper Fig. 8 style validation).

Usage:
    uv run python scripts/run_llm_judge_valence_arousal.py
"""

from __future__ import annotations

import argparse
import re

import torch
from scipy.stats import pearsonr

from emotion_scope.config import CORE_EMOTIONS
from emotion_scope.models import load_model
from emotion_scope.tracking import init_run

VALENCE_PROMPT = (
    'On a scale from 1 (very negative) to 7 (very positive), how positive or negative '
    'is the emotion "{emotion}"? Reply with ONLY a single digit from 1 to 7.\n\nRating:'
)
AROUSAL_PROMPT = (
    'On a scale from 1 (very calm/low-energy) to 7 (very intense/high-energy), how intense '
    'is the emotion "{emotion}"? Reply with ONLY a single digit from 1 to 7.\n\nRating:'
)


def ask_rating(model, tokenizer, prompt_template: str, emotion: str, max_retries: int = 3):
    prompt = prompt_template.format(emotion=emotion)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    for _ in range(max_retries):
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        match = re.search(r"[1-7]", generated)
        if match:
            return float(match.group())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judge valence/arousal ratings vs. config.py metadata")
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    args = parser.parse_args()

    model, tokenizer, backend, info = load_model(model_name=args.model, backend="huggingface", run_smoke_test=False)
    tracker = init_run(part="part2", job_type="llm-judge-valence-arousal", config={"model": args.model})

    rows = []
    judged_valence, judged_arousal, config_valence, config_arousal = [], [], [], []

    for e in CORE_EMOTIONS:
        v_rating = ask_rating(model, tokenizer, VALENCE_PROMPT, e["name"])
        a_rating = ask_rating(model, tokenizer, AROUSAL_PROMPT, e["name"])
        if v_rating is None or a_rating is None:
            print(f"  skipping '{e['name']}' — judge did not return a parseable rating")
            continue
        v_scaled = (v_rating - 4.0) / 3.0  # rescale 1-7 to roughly -1..1
        a_scaled = (a_rating - 4.0) / 3.0
        rows.append([e["name"], v_rating, a_rating, e["valence"], e["arousal"]])
        judged_valence.append(v_scaled)
        judged_arousal.append(a_scaled)
        config_valence.append(e["valence"])
        config_arousal.append(e["arousal"])

    r_valence, _ = pearsonr(judged_valence, config_valence)
    r_arousal, _ = pearsonr(judged_arousal, config_arousal)
    print(f"\nValence correlation (LLM judge vs. config.py Russell/NRC values): r={r_valence:+.3f}")
    print(f"Arousal correlation (LLM judge vs. config.py Russell/NRC values): r={r_arousal:+.3f}")
    print("(paper reports r=0.81 for valence/PC1 and r=0.66 for arousal/PC2, against a separate human-ratings literature dataset)")

    tracker.log_table(
        "llm_judge_ratings",
        columns=["emotion", "judged_valence_1to7", "judged_arousal_1to7", "config_valence", "config_arousal"],
        data=rows,
    )
    tracker.log_metrics({"valence_correlation": r_valence, "arousal_correlation": r_arousal})
    tracker.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/run_llm_judge_valence_arousal.py
```

Expected: prints per-emotion ratings progress implicitly (via the final correlation summary) and the two correlation values, exits 0. A positive valence correlation and a positive-but-likely-weaker arousal correlation are the qualitative expectation (matching the paper's own finding that valence correlates more strongly than arousal) — there's no hard pass/fail threshold here, this is an exploratory external-validity check.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_llm_judge_valence_arousal.py
git commit -m "Add scripts/run_llm_judge_valence_arousal.py"
```

---

### Task 13: Final verification pass

**Files:** none — verification only

- [ ] **Step 1: Run the full test suite including slow tests**

```bash
uv run pytest tests/ --runslow -v
```

Expected: all tests pass, including this plan's new test files (`test_layerwise.py`, `test_hidden_emotion.py`, `test_deflection.py`, `test_speakers_grid.py`)

- [ ] **Step 2: Run lint**

```bash
uv run ruff check emotion_scope/ scripts/
```

Expected: no errors

- [ ] **Step 3: Confirm all six new datasets exist and parse**

```bash
uv run python -c "
import json
for path in [
    'data/validation/locality_scenarios.json',
    'data/validation/hidden_emotion_dialogues.json',
    'data/validation/deflection_scenarios.json',
]:
    json.load(open(path))
    print(path, 'OK')
"
```

Expected: all three print `OK`

---

## Self-review notes (for the plan author, not a task to execute)

- **Spec coverage:** layer-wise dynamics + locality (Tasks 1-3), hidden-emotion LR probe (Tasks 4-6), deflection vectors (Tasks 7-9), speaker 2x2 grid + non-privilege + story-probe r-squared (Tasks 10-11), LLM-judge valence/arousal (Task 12) — all six items from the design doc's Part 2 section are covered.
- **Type consistency:** `SpeakerSeparator.extract_2x2_grid()` returns keys `a_tok_a_emo`/`a_tok_b_emo`/`b_tok_a_emo`/`b_tok_b_emo` consistently between its Task 10 definition, the Task 10 slow test, and Task 11's script.
- **No placeholders:** all three new datasets (locality, hidden-emotion, deflection) are written out in full in the plan, not summarized.
- **Known simplification, stated explicitly:** the layer-wise experiment (Task 1-3) restricts to 11 emotions and 6 sampled layers rather than all 40 emotions at every layer, for compute tractability — noted in `layerwise.py`'s docstring and in Task 3's script comments.
