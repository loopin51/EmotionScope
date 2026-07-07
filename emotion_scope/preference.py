"""
Activity-preference / Elo experiment (paper Fig. 4).

Measures the model's preference between paired activities via next-token
logits at the "(" prefill, aggregates results into Elo ratings, and supports
running the same comparisons under emotion-vector steering to test whether
steering causally shifts preference.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from emotion_scope.steer import steer_context

DEFAULT_K_FACTOR = 32.0
DEFAULT_START_RATING = 1500.0


class EloTracker:
    """Standard Elo rating update, one match (winner beats loser) at a time."""

    def __init__(
        self,
        activity_ids: List[str],
        k_factor: float = DEFAULT_K_FACTOR,
        start: float = DEFAULT_START_RATING,
    ):
        self.ratings: Dict[str, float] = {aid: start for aid in activity_ids}
        self.k_factor = k_factor

    def update(self, winner_id: str, loser_id: str) -> None:
        r_w = self.ratings[winner_id]
        r_l = self.ratings[loser_id]
        expected_w = 1.0 / (1.0 + 10 ** ((r_l - r_w) / 400))
        expected_l = 1.0 - expected_w
        self.ratings[winner_id] = r_w + self.k_factor * (1.0 - expected_w)
        self.ratings[loser_id] = r_l + self.k_factor * (0.0 - expected_l)


def format_preference_prompt(text_a: str, text_b: str) -> str:
    return f"Human: Would you prefer to (A) {text_a} or (B) {text_b}?\nAssistant: ("


def compare_activities(
    model,
    tokenizer,
    text_a: str,
    text_b: str,
    steer_vector: Optional[torch.Tensor] = None,
    steer_alpha: float = 0.5,
    steer_layers: Optional[List[int]] = None,
    steer_avg_norms: Optional[Dict[int, float]] = None,
    backend: str = "huggingface",
) -> str:
    """
    Return 'A' or 'B' — whichever token has the higher next-token logit
    after the '(' prefill in format_preference_prompt().

    If `steer_vector` is given, the forward pass is wrapped in steer_context
    so the comparison reflects the steered model.
    """
    prompt = format_preference_prompt(text_a, text_b)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    id_a = tokenizer.encode("A", add_special_tokens=False)[0]
    id_b = tokenizer.encode("B", add_special_tokens=False)[0]

    def _forward_logits():
        with torch.no_grad():
            out = model(**inputs)
        return out.logits[0, -1, :]

    if steer_vector is not None:
        with steer_context(model, backend, steer_vector, steer_alpha, steer_layers, steer_avg_norms):
            logits = _forward_logits()
    else:
        logits = _forward_logits()

    return "A" if logits[id_a] > logits[id_b] else "B"


def run_full_pairwise(
    model,
    tokenizer,
    activities: List[dict],
    steer_vector: Optional[torch.Tensor] = None,
    steer_alpha: float = 0.5,
    steer_layers: Optional[List[int]] = None,
    steer_avg_norms: Optional[Dict[int, float]] = None,
    backend: str = "huggingface",
) -> EloTracker:
    """
    Run every ordered pair of distinct activities through compare_activities
    and accumulate Elo ratings. Both orderings of each unordered pair are
    run (matching the paper's use of both (A,B) and (B,A) prompts to cancel
    position bias).

    If `steer_vector` is given, ALL comparisons in this round-robin are run
    under that steering vector — this is a deliberate simplification of the
    paper's per-activity steered/control split (see the roadmap design doc's
    Part 1 section): it still tests the causal claim ("does turning this
    emotion vector on shift preference"), just applied uniformly rather than
    to a subset of activities.
    """
    ids = [a["id"] for a in activities]
    by_id = {a["id"]: a["text"] for a in activities}
    tracker = EloTracker(ids)

    for id_a in ids:
        for id_b in ids:
            if id_a == id_b:
                continue
            winner_side = compare_activities(
                model, tokenizer, by_id[id_a], by_id[id_b],
                steer_vector=steer_vector, steer_alpha=steer_alpha,
                steer_layers=steer_layers, steer_avg_norms=steer_avg_norms,
                backend=backend,
            )
            winner_id = id_a if winner_side == "A" else id_b
            loser_id = id_b if winner_side == "A" else id_a
            tracker.update(winner_id, loser_id)

    return tracker
