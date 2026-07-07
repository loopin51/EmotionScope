"""
Logit-lens analysis for emotion vectors.

Projects each emotion vector through the unembedding matrix to find which
output tokens it up-weights and down-weights — reproducing the paper's
Table 1 ("Emotion Vector Top Tokens").
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def top_k_tokens_for_direction(
    direction: torch.Tensor,
    unembed: torch.Tensor,
    tokenizer,
    k: int = 5,
) -> Tuple[List[str], List[str]]:
    """
    Project `direction` (d_model,) through `unembed` (vocab_size, d_model)
    and return the top-k up-weighted and top-k down-weighted token strings.
    """
    logits_delta = unembed @ direction  # (vocab_size,)
    top_up = torch.topk(logits_delta, k).indices.tolist()
    top_down = torch.topk(-logits_delta, k).indices.tolist()
    up_tokens = [tokenizer.decode([i]) for i in top_up]
    down_tokens = [tokenizer.decode([i]) for i in top_down]
    return up_tokens, down_tokens


def get_unembed_matrix(model, backend: str) -> torch.Tensor:
    """
    Return the unembedding matrix as (vocab_size, d_model), regardless of
    backend convention (TransformerLens stores it transposed).
    """
    if backend == "transformer_lens":
        return model.W_U.T.detach().cpu().float()  # (d_model, vocab) -> (vocab, d_model)
    output_embeddings = model.get_output_embeddings()
    return output_embeddings.weight.detach().cpu().float()  # (vocab_size, d_model)
