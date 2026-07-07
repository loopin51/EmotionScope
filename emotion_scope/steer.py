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
