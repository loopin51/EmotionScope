# Part 0 — Foundation (steer.py + tracking.py + emotion corpus expansion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `emotion_scope/steer.py` stub with a real dual-backend steering implementation, add a thin `wandb` tracking wrapper, and expand the emotion vocabulary from 20 to 40 words so that Plans 1–3 have the shared infrastructure and data they need.

**Architecture:** Two new/rewritten library modules (`steer.py`, `tracking.py`) follow the existing `emotion_scope/` pattern of dual TransformerLens/HuggingFace backend support (matching `extract.py`/`probe.py`). The corpus expansion reuses the existing `data/story_contributions/` → `ingest_stories.py` → `emotion_scope/config.py` pipeline already used for the current 20 emotions — no new pipeline is built.

**Tech Stack:** PyTorch, TransformerLens, HuggingFace `transformers`, `wandb` (optional extra), `pytest`.

---

## Before you start

Read these files to understand the patterns you'll be following:
- `emotion_scope/extract.py` — dual-backend activation extraction (the hook patterns you'll mirror in `steer.py`)
- `emotion_scope/probe.py` — dual-backend probing, chat-template formatting
- `emotion_scope/config.py` — `CORE_EMOTIONS` list you'll extend
- `scripts/ingest_stories.py` — the story-corpus validation/merge pipeline you'll extend
- `tests/conftest.py` — the `@pytest.mark.slow` / `--runslow` convention for model-loading tests

Design doc: `docs/superpowers/specs/2026-07-06-emotionscope-paper-repro-roadmap-design.md`

---

### Task 1: Shared `get_transformer_layers()` helper in `utils.py`

`extract.py`, `probe.py`, and `speakers.py` each have their own private copy of the "find the `nn.ModuleList` of transformer blocks" logic. `steer.py` needs the same logic. Rather than writing a 4th copy, add one shared, public version to `utils.py` and have `steer.py` use it (existing files are left untouched — they already work and are out of scope for this plan).

**Files:**
- Modify: `emotion_scope/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_utils.py`:

```python
import torch.nn as nn


def test_get_transformer_layers_finds_model_layers():
    from emotion_scope.utils import get_transformer_layers

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    layers = get_transformer_layers(FakeModel())
    assert len(layers) == 3


def test_get_transformer_layers_finds_transformer_h():
    from emotion_scope.utils import get_transformer_layers

    class FakeGPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    layers = get_transformer_layers(FakeGPT2())
    assert len(layers) == 2


def test_get_transformer_layers_raises_for_unknown_structure():
    from emotion_scope.utils import get_transformer_layers

    class FakeUnknown(nn.Module):
        pass

    with pytest.raises(ValueError):
        get_transformer_layers(FakeUnknown())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -k get_transformer_layers -v`
Expected: FAIL with `ImportError: cannot import name 'get_transformer_layers'`

- [ ] **Step 3: Implement the helper**

Add to `emotion_scope/utils.py`, after the `get_device` function:

```python
# ---------------------------------------------------------------------------
# Transformer layer lookup — shared by extract.py-style hook code and steer.py
# ---------------------------------------------------------------------------

def get_transformer_layers(model):
    """
    Locate the nn.ModuleList of transformer blocks inside a HuggingFace model.

    Tries the common attribute paths used by Gemma/Llama-style models
    (model.layers), GPT-2-style models (transformer.h), and GPT-NeoX-style
    models (gpt_neox.layers).
    """
    for attr_path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in attr_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -k get_transformer_layers -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/utils.py tests/test_utils.py
git commit -m "Add shared get_transformer_layers() helper for steer.py"
```

---

### Task 2: `middle_third_layers()` pure function

**Files:**
- Create: `emotion_scope/steer.py` (replaces the existing stub entirely)
- Test: `tests/test_steer.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_steer.py`:

```python
"""Tests for emotion_scope.steer — steering-vector injection."""

import pytest
import torch


def test_middle_third_layers_gemma_2_2b():
    from emotion_scope.steer import middle_third_layers

    layers = middle_third_layers(26)
    assert layers == list(range(9, 18))
    assert len(layers) == 9


def test_middle_third_layers_small_model():
    from emotion_scope.steer import middle_third_layers

    layers = middle_third_layers(3)
    assert len(layers) >= 1
    assert all(0 <= l < 3 for l in layers)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_steer.py -k middle_third -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'emotion_scope.steer'` (or the old stub raising `NotImplementedError` if it still exists)

- [ ] **Step 3: Replace the stub with the new module (first piece)**

Overwrite `emotion_scope/steer.py` completely:

```python
"""
Emotion steering — causal intervention on the residual stream.

Implements Anthropic's steering equation from "Emotion Concepts and their
Function in a Large Language Model" (2026):

    x_t^(l) <- x_t^(l) + alpha * ||x^(l)||_avg * v_hat_e

applied across the middle-third layers of the model, where alpha=0.5 is
Anthropic's reported default and ||x^(l)||_avg is the average residual
stream L2 norm at layer l, computed once over a reference dataset.

Two entry points:
    - `Steerer.generate(...)` — simple case, wraps model.generate().
      HuggingFace backend only (TransformerLens does not support .generate(),
      matching the existing constraint in scripts/generate_stories.py).
    - `steer_context(...)` — a context manager for custom generation loops
      (used by Part 3's multi-turn agentic rollouts). Works with both
      TransformerLens and HuggingFace backends.

Usage:
    from emotion_scope.steer import Steerer

    steerer = Steerer(model, tokenizer, backend, model_info)
    text = steerer.generate("Tell me about your day.", vector=vectors["desperate"], alpha=0.5)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from emotion_scope.config import DATA_DIR
from emotion_scope.utils import get_transformer_layers


def middle_third_layers(n_layers: int) -> List[int]:
    """
    Return the middle-third layer indices [L/3, 2L/3] (inclusive), matching
    the layer range Anthropic steers at in the paper.
    """
    start = round(n_layers / 3)
    end = round(2 * n_layers / 3)
    if end <= start:
        end = start + 1
    return list(range(start, end + 1))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_steer.py -k middle_third -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/steer.py tests/test_steer.py
git commit -m "Replace steer.py stub with middle_third_layers()"
```

---

### Task 3: `steer_context()` — dual-backend activation injection

**Files:**
- Modify: `emotion_scope/steer.py`
- Test: `tests/test_steer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_steer.py`:

```python
@pytest.mark.slow
def test_steer_context_modifies_huggingface_activation():
    """Injecting a steering vector should measurably shift the residual stream."""
    from emotion_scope.models import load_model
    from emotion_scope.steer import steer_context, middle_third_layers
    from emotion_scope.utils import get_transformer_layers

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    layers = middle_third_layers(info["n_layers"])
    d_model = info["d_model"]

    torch.manual_seed(0)
    vector = torch.randn(d_model)
    avg_norms = {l: 10.0 for l in layers}  # fixed norm, easy to reason about

    tokens = tokenizer("Hello, how are you?", return_tensors="pt")
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    layers_module = get_transformer_layers(model)
    captured = {}

    def make_hook(l):
        def hook_fn(_module, _input, output):
            captured[l] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return hook_fn

    handles = [layers_module[l].register_forward_hook(make_hook(l)) for l in layers]
    with torch.no_grad():
        model(**tokens)
    for h in handles:
        h.remove()
    unsteered = {l: captured[l].clone() for l in layers}

    captured.clear()
    handles = [layers_module[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        with steer_context(model, backend, vector, alpha=0.5, layers=layers, avg_norms=avg_norms):
            with torch.no_grad():
                model(**tokens)
    finally:
        for h in handles:
            h.remove()
    steered = {l: captured[l].clone() for l in layers}

    for l in layers:
        assert not torch.allclose(unsteered[l], steered[l]), f"layer {l} activation did not change"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_steer.py -k steer_context --runslow -v`
Expected: FAIL with `ImportError: cannot import name 'steer_context'`

- [ ] **Step 3: Implement `steer_context()`**

Append to `emotion_scope/steer.py`:

```python
@contextmanager
def steer_context(
    model,
    backend: str,
    vector: torch.Tensor,
    alpha: float,
    layers: List[int],
    avg_norms: Dict[int, float],
):
    """
    Temporarily inject `alpha * avg_norms[layer] * v_hat` into the residual
    stream at each layer in `layers`, for the duration of the `with` block.

    Works for both TransformerLens and HuggingFace backends. Use this
    directly (instead of Steerer.generate) inside a custom multi-turn or
    agentic rollout loop.
    """
    v_hat = F.normalize(vector, dim=0)

    if backend == "transformer_lens":
        fwd_hooks = []
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            scale = alpha * avg_norms[layer]

            def make_hook(scale=scale):
                def hook_fn(resid, hook):
                    return resid + scale * v_hat.to(resid.dtype).to(resid.device)
                return hook_fn

            fwd_hooks.append((hook_name, make_hook()))
        with model.hooks(fwd_hooks=fwd_hooks):
            yield
    else:
        layers_module = get_transformer_layers(model)
        handles = []
        try:
            for layer in layers:
                scale = alpha * avg_norms[layer]

                def make_hook(scale=scale):
                    def hook_fn(_module, _input, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        steered = hidden + scale * v_hat.to(hidden.dtype).to(hidden.device)
                        if isinstance(output, tuple):
                            return (steered,) + output[1:]
                        return steered
                    return hook_fn

                handles.append(layers_module[layer].register_forward_hook(make_hook()))
            yield
        finally:
            for h in handles:
                h.remove()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_steer.py -k steer_context --runslow -v`
Expected: `1 passed` (this downloads `google/gemma-2-2b-it` on first run — several GB, requires either `huggingface-cli login` acceptance of the Gemma license or a cached copy)

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/steer.py tests/test_steer.py
git commit -m "Implement steer_context() dual-backend activation injection"
```

---

### Task 4: `Steerer` class — `compute_avg_norms()` and `generate()`

**Files:**
- Modify: `emotion_scope/steer.py`
- Test: `tests/test_steer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_steer.py`:

```python
@pytest.mark.slow
def test_compute_avg_norms_returns_positive_floats():
    from emotion_scope.models import load_model
    from emotion_scope.steer import Steerer, middle_third_layers

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    steerer = Steerer(model, tokenizer, backend, info)
    layers = middle_third_layers(info["n_layers"])[:2]  # keep it fast — 2 layers only
    norms = steerer.compute_avg_norms(texts=["The weather today is mild.", "Please open the door."], layers=layers)

    assert set(norms.keys()) == set(layers)
    for l in layers:
        assert norms[l] > 0.0


@pytest.mark.slow
def test_generate_raises_for_transformer_lens_backend():
    from emotion_scope.steer import Steerer

    class FakeInfo(dict):
        pass

    steerer = Steerer(model=None, tokenizer=None, backend="transformer_lens", model_info={"n_layers": 26})
    with pytest.raises(ValueError, match="HuggingFace backend"):
        steerer.generate("hello", vector=torch.randn(4))


@pytest.mark.slow
def test_generate_produces_text():
    from emotion_scope.models import load_model
    from emotion_scope.steer import Steerer

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    steerer = Steerer(model, tokenizer, backend, info)
    torch.manual_seed(0)
    vector = torch.randn(info["d_model"])

    text = steerer.generate("Tell me about your day.", vector=vector, alpha=0.5, max_new_tokens=20)
    assert isinstance(text, str)
    assert len(text) > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_steer.py -k "compute_avg_norms or generate" --runslow -v`
Expected: FAIL with `ImportError: cannot import name 'Steerer'`

- [ ] **Step 3: Implement `Steerer`**

Append to `emotion_scope/steer.py`:

```python
def _load_default_norm_texts() -> List[str]:
    """Load the existing neutral-prompts corpus as the default reference dataset for avg-norm computation."""
    path = DATA_DIR / "neutral" / "neutral_prompts.jsonl"
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def _format_chat_prompt(tokenizer, user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is not None:
        try:
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return f"User: {user_message}\n\nAssistant:"


class Steerer:
    """Steers a loaded model's generation using an emotion vector."""

    def __init__(self, model, tokenizer, backend: str, model_info: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        self.model_info = model_info
        self.n_layers: int = model_info["n_layers"]
        self._avg_norms: Optional[Dict[int, float]] = None

    def compute_avg_norms(
        self,
        texts: Optional[List[str]] = None,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Compute the average residual-stream L2 norm at each layer, over a
        reference dataset. Defaults to the existing neutral-prompts corpus.
        """
        layers = layers or middle_third_layers(self.n_layers)
        texts = texts or _load_default_norm_texts()

        sums = {l: 0.0 for l in layers}
        counts = {l: 0 for l in layers}

        if self.backend == "transformer_lens":
            hook_names = [f"blocks.{l}.hook_resid_post" for l in layers]
            for text in texts:
                tokens = self.model.to_tokens(text)
                _, cache = self.model.run_with_cache(tokens, names_filter=lambda n: n in hook_names)
                for l, hook_name in zip(layers, hook_names):
                    resid = cache[hook_name][0]  # (seq_len, d_model)
                    norms = resid.norm(dim=-1)
                    sums[l] += norms.sum().item()
                    counts[l] += norms.numel()
        else:
            layers_module = get_transformer_layers(self.model)
            captured: Dict[int, torch.Tensor] = {}

            def make_hook(l):
                def hook_fn(_module, _input, output):
                    captured[l] = (output[0] if isinstance(output, tuple) else output).detach()
                return hook_fn

            handles = [layers_module[l].register_forward_hook(make_hook(l)) for l in layers]
            try:
                for text in texts:
                    tokens = self.tokenizer(text, return_tensors="pt")
                    tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
                    with torch.no_grad():
                        self.model(**tokens)
                    for l in layers:
                        resid = captured[l][0]
                        norms = resid.norm(dim=-1)
                        sums[l] += norms.sum().item()
                        counts[l] += norms.numel()
            finally:
                for h in handles:
                    h.remove()

        avg_norms = {l: (sums[l] / counts[l] if counts[l] else 0.0) for l in layers}
        self._avg_norms = avg_norms
        return avg_norms

    def generate(
        self,
        prompt: str,
        vector: torch.Tensor,
        alpha: float = 0.5,
        layers: Optional[List[int]] = None,
        avg_norms: Optional[Dict[int, float]] = None,
        max_new_tokens: int = 150,
        use_chat_template: bool = True,
    ) -> str:
        """Generate a steered continuation for `prompt`. HuggingFace backend only."""
        if self.backend != "huggingface":
            raise ValueError(
                "Steerer.generate() requires the HuggingFace backend — "
                "TransformerLens does not support .generate(). "
                "Load the model with backend='huggingface', or use "
                "steer_context() directly for a custom generation loop."
            )
        layers = layers or middle_third_layers(self.n_layers)
        avg_norms = avg_norms or self._avg_norms or self.compute_avg_norms(layers=layers)

        text = _format_chat_prompt(self.tokenizer, prompt) if use_chat_template else prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with steer_context(self.model, self.backend, vector, alpha, layers, avg_norms):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        generated_ids = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_steer.py --runslow -v`
Expected: `7 passed` (2 from Task 2, 1 from Task 3, 4 from this task — note `test_generate_raises_for_transformer_lens_backend` doesn't actually need a loaded model since it fails before touching `self.model`)

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/steer.py tests/test_steer.py
git commit -m "Implement Steerer.compute_avg_norms() and Steerer.generate()"
```

---

### Task 5: Retire the obsolete stub test and export `Steerer`

**Files:**
- Modify: `tests/test_imports.py`
- Modify: `emotion_scope/__init__.py`

- [ ] **Step 1: Update the obsolete test**

In `tests/test_imports.py`, replace:

```python
def test_steer_stub_raises():
    import pytest
    from emotion_scope.steer import steer
    with pytest.raises(NotImplementedError):
        steer()
```

with:

```python
def test_steer_module_loads():
    """steer.py is now a real implementation, not a stub."""
    from emotion_scope.steer import Steerer, steer_context, middle_third_layers
    assert callable(Steerer)
    assert callable(steer_context)
    assert callable(middle_third_layers)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest tests/test_imports.py -v`
Expected: `test_steer_module_loads PASSED` (and the other existing tests in the file still pass)

- [ ] **Step 3: Export `Steerer` from the package root**

In `emotion_scope/__init__.py`, add the import and `__all__` entry:

```python
from emotion_scope.steer import Steerer
```

Add `"Steerer",` to the `__all__` list.

- [ ] **Step 4: Run the full fast test suite**

Run: `uv run pytest tests/ -v`
Expected: all non-slow tests pass (slow tests remain skipped without `--runslow`)

- [ ] **Step 5: Commit**

```bash
git add tests/test_imports.py emotion_scope/__init__.py
git commit -m "Retire steer.py stub test; export Steerer from package root"
```

---

### Task 6: `emotion_scope/tracking.py` — thin wandb wrapper

**Files:**
- Create: `emotion_scope/tracking.py`
- Test: `tests/test_tracking.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tracking.py`:

```python
"""Tests for emotion_scope.tracking — thin wandb wrapper. No network access required."""

import os

import pytest


def test_tracker_noop_direct():
    from emotion_scope.tracking import Tracker

    tracker = Tracker(None)
    assert tracker.enabled is False
    # None of these should raise, even though there's no underlying run.
    tracker.log_metrics({"a": 1})
    tracker.log_table("t", ["x"], [[1]])
    tracker.log_artifact("nonexistent/path.txt", "name", "type")
    tracker.finish()


def test_init_run_disabled_via_env(monkeypatch):
    from emotion_scope.tracking import init_run

    monkeypatch.setenv("WANDB_MODE", "disabled")
    tracker = init_run(part="part0", job_type="smoke-test", config={"alpha": 0.5})
    assert tracker.enabled is False
    tracker.log_metrics({"loss": 0.1})  # must not raise


def test_is_wandb_available_returns_bool():
    from emotion_scope.tracking import is_wandb_available

    assert isinstance(is_wandb_available(), bool)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'emotion_scope.tracking'`

- [ ] **Step 3: Implement `tracking.py`**

Create `emotion_scope/tracking.py`:

```python
"""
Thin Weights & Biases wrapper for the paper-reproduction experiments
(see docs/superpowers/specs/2026-07-06-emotionscope-paper-repro-roadmap-design.md).

Every run uses a single project ('emotionscope-paper-repro'), with the
roadmap part ('part1', 'part2', 'part3', ...) as the wandb group and the
experiment type as the job_type, so cross-part comparisons stay on one
dashboard.

Gracefully no-ops if `wandb` isn't installed, or if WANDB_MODE=disabled is
set — existing scripts and tests never require the `tracking` extra.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def is_wandb_available() -> bool:
    """Whether the `wandb` package can be imported."""
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


class Tracker:
    """Wraps a wandb run. Becomes a no-op if `run` is None."""

    def __init__(self, run: Optional[Any]):
        self._run = run

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_table(self, name: str, columns: List[str], data: List[list]) -> None:
        if self._run is not None:
            import wandb
            self._run.log({name: wandb.Table(columns=columns, data=data)})

    def log_artifact(self, path: str, name: str, type: str) -> None:
        if self._run is not None:
            import wandb
            artifact = wandb.Artifact(name=name, type=type)
            artifact.add_file(path)
            self._run.log_artifact(artifact)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def init_run(
    part: str,
    job_type: str,
    config: Dict[str, Any],
    name: Optional[str] = None,
    project: str = "emotionscope-paper-repro",
) -> Tracker:
    """
    Start (or no-op) a wandb run for one part of the paper-reproduction roadmap.

    Set WANDB_MODE=disabled, or leave `wandb` uninstalled, to get a Tracker
    that silently no-ops on every call.
    """
    if not is_wandb_available():
        return Tracker(None)
    if os.environ.get("WANDB_MODE") == "disabled":
        return Tracker(None)

    import wandb
    run = wandb.init(project=project, group=part, job_type=job_type, config=config, name=name)
    return Tracker(run)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/tracking.py tests/test_tracking.py
git commit -m "Add emotion_scope.tracking — thin wandb wrapper"
```

---

### Task 7: Add `wandb` as an optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the `tracking` extra**

In `pyproject.toml`, change:

```toml
[project.optional-dependencies]
demo = ["gradio>=4.0"]
viz = ["plotly>=5.0", "matplotlib>=3.8"]
notebooks = ["jupyter>=1.0", "ipywidgets>=8.0"]
cloud = ["google-cloud-compute>=1.0", "runpod>=1.0"]
dev = ["pytest>=8.0", "ruff>=0.3"]
all = ["emotion-scope[demo,viz,notebooks,dev]"]
```

to:

```toml
[project.optional-dependencies]
demo = ["gradio>=4.0"]
viz = ["plotly>=5.0", "matplotlib>=3.8"]
notebooks = ["jupyter>=1.0", "ipywidgets>=8.0"]
cloud = ["google-cloud-compute>=1.0", "runpod>=1.0"]
tracking = ["wandb>=0.16"]
dev = ["pytest>=8.0", "ruff>=0.3"]
all = ["emotion-scope[demo,viz,notebooks,tracking,dev]"]
```

- [ ] **Step 2: Verify the project still resolves**

Run: `uv sync`
Expected: exits 0, no dependency resolution errors

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Add wandb as an optional 'tracking' extra"
```

---

### Task 8: Open the ingest pipeline to the 20 new emotion words

Before any new-emotion stories can be merged, `scripts/ingest_stories.py`'s hardcoded `APPROVED_EMOTIONS`/`EMOTION_SYNONYMS` must accept them — otherwise every new-emotion line is silently rejected during ingest.

The 20 new emotion words (name, valence, arousal — sourced from `data/emotions_171.json`):

| name | valence | arousal |
|---|---|---|
| vulnerable | -0.4 | 0.2 |
| playful | 0.7 | 0.6 |
| exuberant | 0.85 | 0.85 |
| spiteful | -0.6 | 0.6 |
| obstinate | -0.3 | 0.3 |
| blissful | 0.9 | 0.5 |
| jubilant | 0.9 | 0.8 |
| ecstatic | 0.95 | 0.9 |
| content | 0.6 | -0.3 |
| serene | 0.7 | -0.6 |
| melancholy | -0.5 | -0.2 |
| weary | -0.4 | -0.5 |
| lonely | -0.7 | -0.1 |
| furious | -0.8 | 0.9 |
| irritated | -0.4 | 0.5 |
| contemptuous | -0.5 | 0.4 |
| resentful | -0.6 | 0.4 |
| terrified | -0.9 | 0.9 |
| paranoid | -0.6 | 0.6 |
| astonished | 0.3 | 0.8 |

**Files:**
- Modify: `scripts/ingest_stories.py`

- [ ] **Step 1: Extend `APPROVED_EMOTIONS`**

In `scripts/ingest_stories.py`, change:

```python
APPROVED_EMOTIONS = {
    "happy", "sad", "afraid", "angry", "calm", "desperate", "hopeful",
    "frustrated", "curious", "proud", "guilty", "surprised", "loving",
    "hostile", "nervous", "confident", "brooding", "enthusiastic",
    "reflective", "gloomy",
}
```

to:

```python
APPROVED_EMOTIONS = {
    "happy", "sad", "afraid", "angry", "calm", "desperate", "hopeful",
    "frustrated", "curious", "proud", "guilty", "surprised", "loving",
    "hostile", "nervous", "confident", "brooding", "enthusiastic",
    "reflective", "gloomy",
    # --- expansion batch (2026-07-06), see data/emotions_171.json ---
    "vulnerable", "playful", "exuberant", "spiteful", "obstinate",
    "blissful", "jubilant", "ecstatic", "content", "serene",
    "melancholy", "weary", "lonely", "furious", "irritated",
    "contemptuous", "resentful", "terrified", "paranoid", "astonished",
}
```

- [ ] **Step 2: Extend `EMOTION_SYNONYMS`**

In `scripts/ingest_stories.py`, add these entries to the `EMOTION_SYNONYMS` dict (after the existing `"gloomy"` entry):

```python
    "vulnerable": {"vulnerable", "vulnerably", "vulnerability"},
    "playful": {"playful", "playfully", "playfulness"},
    "exuberant": {"exuberant", "exuberantly", "exuberance"},
    "spiteful": {"spiteful", "spitefully", "spite"},
    "obstinate": {"obstinate", "obstinately", "obstinacy"},
    "blissful": {"blissful", "blissfully", "bliss"},
    "jubilant": {"jubilant", "jubilantly", "jubilation"},
    "ecstatic": {"ecstatic", "ecstatically", "ecstasy"},
    "content": {"content", "contentment", "contentedly"},
    "serene": {"serene", "serenely", "serenity"},
    "melancholy": {"melancholy", "melancholic", "melancholically"},
    "weary": {"weary", "wearily", "weariness"},
    "lonely": {"lonely", "loneliness", "lonelier"},
    "furious": {"furious", "furiously", "fury"},
    "irritated": {"irritated", "irritating", "irritation", "irritably"},
    "contemptuous": {"contemptuous", "contemptuously", "contempt"},
    "resentful": {"resentful", "resentfully", "resentment"},
    "terrified": {"terrified", "terrifying", "terror"},
    "paranoid": {"paranoid", "paranoia", "paranoically"},
    "astonished": {"astonished", "astonishing", "astonishment"},
```

- [ ] **Step 3: Fix the hardcoded target-count math**

In `scripts/ingest_stories.py`, `main()` currently has:

```python
    total_target = 50 * 20
```

Change to:

```python
    total_target = 50 * len(APPROVED_EMOTIONS)
```

- [ ] **Step 4: Verify the dry run still works with zero new stories**

Run: `uv run python scripts/ingest_stories.py --dry-run`
Expected: exits 0; per-emotion breakdown now lists all 40 emotions, with the 20 new ones showing `NEED 50 MORE` (0 stories yet — expected, Task 9 adds them) and the original 20 still showing `OK`. Total shown as `X/2000`.

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest_stories.py
git commit -m "Open ingest_stories.py to 20 new emotion words"
```

---

### Task 9: Author and ingest stories for the 20 new emotions

Write ~50 short story vignettes per new emotion, matching the existing corpus's style exactly: third-person, past tense, one sentence-to-short-paragraph, concrete sensory/situational detail, never naming the emotion or an obvious synonym (the `ingest_stories.py` leakage check will flag — but not reject — violations, so double-check manually too).

**Files:**
- Create: `data/story_contributions/expansion_batch1.jsonl`

- [ ] **Step 1: Write one example story per new emotion — the quality bar**

Create `data/story_contributions/expansion_batch1.jsonl` and start it with exactly these 20 lines (one worked example per emotion, in the corpus's established style):

```jsonl
{"emotion": "vulnerable", "text": "She had never shown anyone the notebook before, and her hands trembled slightly as she slid it across the table, watching his face for the first flicker of a reaction"}
{"emotion": "playful", "text": "He snuck up behind her with the water balloon held high, grinning so hard he could barely keep quiet, and she pretended not to notice until the very last second"}
{"emotion": "exuberant", "text": "The whole team burst out of the conference room at once, high-fiving everyone in the hallway, someone already shouting for pizza to celebrate the launch"}
{"emotion": "spiteful", "text": "She left the group chat without a word, then quietly told two of their mutual friends exactly what he'd said about them, timing it so it would reach him by morning"}
{"emotion": "obstinate", "text": "Three people had already explained the safer route, but he folded his arms and said he'd take the shortcut through the canyon anyway, the same as always"}
{"emotion": "blissful", "text": "She lay in the hammock with the hum of cicadas and the smell of woodsmoke drifting over from someone's grill, not thinking about anything at all"}
{"emotion": "jubilant", "text": "The whistle blew and the entire stadium erupted at once, strangers hugging strangers, the home team's flag passed hand over hand through the stands"}
{"emotion": "ecstatic", "text": "She screamed into the phone the second she saw the acceptance email, jumping up and down so hard the neighbors later asked if something had fallen"}
{"emotion": "content", "text": "He finished the last dish, dried his hands, and sat down on the porch step with nothing left to do for the rest of the evening"}
{"emotion": "serene", "text": "The lake was glass-still at dawn, just the occasional ripple from a fish, and she let the canoe drift wherever it wanted to go"}
{"emotion": "melancholy", "text": "The old photographs were still in the box exactly where his mother had left them, and he found himself looking at them longer than he meant to"}
{"emotion": "weary", "text": "It was the fourth double shift that week, and she sat in the parking lot for a full minute before finding the energy to turn the key"}
{"emotion": "lonely", "text": "He set two plates out of habit before remembering, and ate standing at the counter instead of sitting down at the empty table"}
{"emotion": "furious", "text": "He slammed the laptop shut hard enough that the neighbors probably heard it, then stood there breathing through his teeth for a full minute"}
{"emotion": "irritated", "text": "The printer jammed for the third time in ten minutes, and she yanked the paper tray out harder than necessary, muttering under her breath"}
{"emotion": "contemptuous", "text": "He glanced at the proposal for exactly two seconds before sliding it back across the table without comment, already turning to his phone"}
{"emotion": "resentful", "text": "She smiled and congratulated her coworker on the promotion, then went home and reread her own performance reviews for the third time that month"}
{"emotion": "terrified", "text": "The floor creaked again, closer this time, and she stood frozen in the dark hallway with her hand still on the light switch that wouldn't turn on"}
{"emotion": "paranoid", "text": "He checked the rearview mirror for the fifth time in two blocks, certain now that the same gray sedan had been behind him since the gas station"}
{"emotion": "astonished", "text": "She read the letter twice, then a third time, because the number at the bottom couldn't possibly be right, could it"}
```

- [ ] **Step 2: Validate the seed batch**

Run: `uv run python scripts/ingest_stories.py --dry-run`
Expected: exits 0, 20 new stories counted (1 per new emotion), no hard errors. Soft warnings for emotion-word leakage are acceptable if any appear — review them, but they don't block ingestion.

- [ ] **Step 3: Write the remaining ~49 stories per emotion**

Following the exact style demonstrated in Step 1 (third person, past tense, concrete situational/sensory detail, no naming the emotion or its synonyms), append 49 more stories per emotion to `data/story_contributions/expansion_batch1.jsonl`, for a total of 50 per emotion (1,000 lines total for the 20 new emotions). Vary the scenarios — avoid repeating the same situation type (e.g. don't write 10 "waiting for test results" stories for `terrified`).

- [ ] **Step 4: Validate the full batch**

Run: `uv run python scripts/ingest_stories.py --dry-run`
Expected: exits 0; per-emotion breakdown shows all 20 new emotions at `count >= 50` (`OK`), 0 hard errors.

- [ ] **Step 5: Merge for real**

Run: `uv run python scripts/ingest_stories.py`
Expected: exits 0; `data/templates/emotion_stories.jsonl` now contains ~2,000 lines (1,000 original + 1,000 new); prints `Wrote 2000 stories to data/templates/emotion_stories.jsonl` (exact count may vary slightly if any lines were flagged as duplicates)

- [ ] **Step 6: Verify no existing tests broke**

Run: `uv run pytest tests/ -v`
Expected: all tests still pass — `test_emotion_stories_jsonl_exists_and_covers_all_emotions` only checks the *current* `CORE_EMOTION_NAMES` (still 20 at this point in the plan), which are unaffected by the new lines being present in the file.

- [ ] **Step 7: Commit**

```bash
git add data/story_contributions/expansion_batch1.jsonl data/templates/emotion_stories.jsonl
git commit -m "Add 1,000 stories for 20 new emotions, merge into corpus"
```

---

### Task 10: Add the 20 new emotions to `CORE_EMOTIONS`

**Files:**
- Modify: `emotion_scope/config.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_config.py`, change:

```python
def test_core_emotions_count():
    assert len(CORE_EMOTIONS) == 20
    assert len(CORE_EMOTION_NAMES) == 20
```

to:

```python
def test_core_emotions_count():
    assert len(CORE_EMOTIONS) == 40
    assert len(CORE_EMOTION_NAMES) == 40
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -k test_core_emotions_count -v`
Expected: FAIL — `assert 20 == 40`

- [ ] **Step 3: Extend `CORE_EMOTIONS`**

In `emotion_scope/config.py`, change the comment above `CORE_EMOTIONS` from `# Core emotions (20)` to `# Core emotions (40)`, and append these 20 entries to the `CORE_EMOTIONS` list (after the existing `"gloomy"` entry, before the closing `]`):

```python
    # --- expansion batch (2026-07-06) — see data/emotions_171.json ---
    {"name": "vulnerable",   "valence": -0.4,  "arousal":  0.2},
    {"name": "playful",      "valence":  0.7,  "arousal":  0.6},
    {"name": "exuberant",    "valence":  0.85, "arousal":  0.85},
    {"name": "spiteful",     "valence": -0.6,  "arousal":  0.6},
    {"name": "obstinate",    "valence": -0.3,  "arousal":  0.3},
    {"name": "blissful",     "valence":  0.9,  "arousal":  0.5},
    {"name": "jubilant",     "valence":  0.9,  "arousal":  0.8},
    {"name": "ecstatic",     "valence":  0.95, "arousal":  0.9},
    {"name": "content",      "valence":  0.6,  "arousal": -0.3},
    {"name": "serene",       "valence":  0.7,  "arousal": -0.6},
    {"name": "melancholy",   "valence": -0.5,  "arousal": -0.2},
    {"name": "weary",        "valence": -0.4,  "arousal": -0.5},
    {"name": "lonely",       "valence": -0.7,  "arousal": -0.1},
    {"name": "furious",      "valence": -0.8,  "arousal":  0.9},
    {"name": "irritated",    "valence": -0.4,  "arousal":  0.5},
    {"name": "contemptuous", "valence": -0.5,  "arousal":  0.4},
    {"name": "resentful",    "valence": -0.6,  "arousal":  0.4},
    {"name": "terrified",    "valence": -0.9,  "arousal":  0.9},
    {"name": "paranoid",     "valence": -0.6,  "arousal":  0.6},
    {"name": "astonished",   "valence":  0.3,  "arousal":  0.8},
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_data_files.py -v`
Expected: all pass — `test_core_emotions_count` now sees 40; `test_emotion_stories_jsonl_exists_and_covers_all_emotions` now checks all 40 names against the merged corpus from Task 9 and finds >=50 (well above the required >=10) for each

- [ ] **Step 5: Commit**

```bash
git add emotion_scope/config.py tests/test_config.py
git commit -m "Add 20 new emotions to CORE_EMOTIONS (20 -> 40)"
```

---

### Task 11: Regenerate the emotion metadata snapshot file

`data/emotions_core20.json` is a static snapshot checked by `tests/test_data_files.py::test_emotions_core20_json`. It needs to become a 40-entry file with an accurate name.

**Files:**
- Modify: `data/emotions_core20.json` -> rename to `data/emotions_core40.json`
- Modify: `tests/test_data_files.py`

- [ ] **Step 1: Generate the new snapshot file**

Run this one-off command to regenerate the file from the now-updated `CORE_EMOTIONS`:

```bash
uv run python -c "
import json
from emotion_scope.config import CORE_EMOTIONS

def category(v, a):
    if v > 0.3 and a > 0.3: return 'positive_high'
    if v > 0.3 and a < -0.3: return 'positive_low'
    if v > 0.3: return 'positive_moderate'
    if v < -0.3 and a > 0.3: return 'negative_high'
    if v < -0.3 and a < -0.3: return 'negative_low'
    if v < -0.3: return 'negative_moderate'
    if a > 0.3: return 'neutral_high'
    if a < -0.3: return 'neutral_low'
    return 'neutral_moderate'

data = [
    {'name': e['name'], 'valence': e['valence'], 'arousal': e['arousal'], 'category': category(e['valence'], e['arousal'])}
    for e in CORE_EMOTIONS
]
with open('data/emotions_core40.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
print(f'Wrote {len(data)} entries to data/emotions_core40.json')
"
```

Expected output: `Wrote 40 entries to data/emotions_core40.json`

- [ ] **Step 2: Remove the old file**

```bash
git rm data/emotions_core20.json
```

- [ ] **Step 3: Update the test**

In `tests/test_data_files.py`, change:

```python
def test_emotions_core20_json():
    path = DATA_DIR / "emotions_core20.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 20
    names = {e["name"] for e in data}
    assert names == set(CORE_EMOTION_NAMES)
```

to:

```python
def test_emotions_core40_json():
    path = DATA_DIR / "emotions_core40.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 40
    names = {e["name"] for e in data}
    assert names == set(CORE_EMOTION_NAMES)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data_files.py -v`
Expected: all pass, including `test_emotions_core40_json`

- [ ] **Step 5: Commit**

```bash
git add data/emotions_core40.json tests/test_data_files.py
git commit -m "Rename emotions_core20.json -> emotions_core40.json"
```

---

### Task 12: Re-extract emotion vectors covering all 40 emotions

This regenerates `results/vectors/google_gemma-2-2b-it.pt` so it contains vectors for all 40 emotions (Plans 1–3 depend on this). This is a live-model run, not a unit test.

**Files:**
- No new files — regenerates `results/vectors/google_gemma-2-2b-it.pt` in place

- [ ] **Step 1: Back up the current vectors file**

```bash
cp results/vectors/google_gemma-2-2b-it.pt results/vectors/google_gemma-2-2b-it.pt.bak-20emotions
```

- [ ] **Step 2: Re-run extraction with a layer sweep**

```bash
uv run python scripts/extract_all.py --model google/gemma-2-2b-it --sweep-layers
```

Expected output ends with something like:

```
n_vectors: 40
valence_separation: <some value < -0.2>
avg_pairwise_cosine: <some value < 0.5>
[extract] Saved emotion vectors to results/vectors/google_gemma-2-2b-it.pt
```

If `valence_separation` or `avg_pairwise_cosine` fail their target thresholds, do not treat this as a blocker for Plan 0 — note the values in the commit message and flag it as a known risk for Plan 1/2/3 (which consume this vectors file); the original 20-emotion vectors already passed these thresholds per the README, and adding 20 more should not systematically break the geometry, but this is the first real checkpoint that confirms it.

- [ ] **Step 3: Validate against the existing gates**

```bash
uv run python scripts/validate_all.py --vectors results/vectors/google_gemma-2-2b-it.pt
```

Expected: prints a `ValidationResult` summary; the four Phase 1 gates (Tylenol, top-3 recall, valence separation, richness) should still show PASS — these gates only depend on the original 20 emotions plus the corpus's baseline structure, not on the new 20 being individually validated (that's Plan 2/3's job).

- [ ] **Step 4: Commit the updated vectors file**

```bash
git add results/vectors/google_gemma-2-2b-it.pt
git commit -m "Re-extract emotion vectors covering all 40 emotions"
```

- [ ] **Step 5: Remove the backup once satisfied**

```bash
rm results/vectors/google_gemma-2-2b-it.pt.bak-20emotions
```

---

### Task 13: Final verification pass

**Files:** none — verification only

- [ ] **Step 1: Run the full test suite including slow tests**

```bash
uv run pytest tests/ --runslow -v
```

Expected: all tests pass (this is the first time `--runslow` tests exist in this repo — `tests/test_steer.py`'s four `@pytest.mark.slow` tests are new)

- [ ] **Step 2: Run lint**

```bash
uv run ruff check emotion_scope/ scripts/
```

Expected: no errors (fix any that appear — most likely unused imports if any step above was adapted)

- [ ] **Step 3: Confirm wandb no-op path works without the extra installed**

```bash
uv run python -c "
from emotion_scope.tracking import init_run
t = init_run('part0', 'smoke-test', {})
print('enabled:', t.enabled)
t.log_metrics({'ok': 1})
print('no-op path works')
"
```

Expected: `enabled: False` followed by `no-op path works` (assuming `wandb` isn't installed in the base environment — if it is, this will actually start a live run; use `WANDB_MODE=disabled` as a prefix to force the no-op path either way: `WANDB_MODE=disabled uv run python -c "..."`)

- [ ] **Step 4: Review git log for this plan**

```bash
git log --oneline -15
```

Expected: one commit per task above (13 commits), all with clear messages

---

## Self-review notes (for the plan author, not a task to execute)

- **Spec coverage:** Foundation section of the design doc covers `steer.py` (Tasks 2-5), `tracking.py` (Tasks 6-7), and targeted corpus expansion (Tasks 8-12). All three are represented.
- **Type consistency:** `Steerer(model, tokenizer, backend, model_info)` constructor signature is used identically in Tasks 3-5's tests. `steer_context(model, backend, vector, alpha, layers, avg_norms)` parameter order is consistent between its Task 3 definition and its Task 4 call site inside `Steerer.generate()`.
- **No placeholders:** the story-authoring task (Task 9) cannot embed all 1,000 lines in the plan document itself (impractical), so it gives an exact style specification, one fully-written example per emotion (20 total), and an executable acceptance check (`ingest_stories.py --dry-run` counts) — this is a concrete, verifiable task, not a vague "write more stories" placeholder.
