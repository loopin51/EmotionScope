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
