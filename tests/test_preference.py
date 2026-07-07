"""Tests for emotion_scope.preference — Elo tracker and pairwise comparison."""

import pytest


def test_elo_tracker_starts_at_default_rating():
    from emotion_scope.preference import EloTracker

    tracker = EloTracker(["a", "b", "c"])
    assert tracker.ratings["a"] == 1500.0
    assert tracker.ratings["b"] == 1500.0


def test_elo_tracker_winner_gains_rating():
    from emotion_scope.preference import EloTracker

    tracker = EloTracker(["a", "b"])
    before_winner = tracker.ratings["a"]
    before_loser = tracker.ratings["b"]
    tracker.update(winner_id="a", loser_id="b")
    assert tracker.ratings["a"] > before_winner
    assert tracker.ratings["b"] < before_loser


def test_elo_tracker_repeated_wins_increase_gap():
    from emotion_scope.preference import EloTracker

    tracker = EloTracker(["a", "b"])
    for _ in range(20):
        tracker.update(winner_id="a", loser_id="b")
    assert tracker.ratings["a"] - tracker.ratings["b"] > 200


def test_format_preference_prompt_contains_both_options():
    from emotion_scope.preference import format_preference_prompt

    prompt = format_preference_prompt("bake bread", "read a book")
    assert "bake bread" in prompt
    assert "read a book" in prompt
    assert prompt.endswith("(")


@pytest.mark.slow
def test_compare_activities_returns_a_or_b():
    from emotion_scope.models import load_model
    from emotion_scope.preference import compare_activities

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    result = compare_activities(model, tokenizer, "help someone learn to read", "help someone commit fraud")
    assert result in ("A", "B")


@pytest.mark.slow
def test_run_full_pairwise_produces_ratings_for_all_activities():
    from emotion_scope.models import load_model
    from emotion_scope.preference import run_full_pairwise

    model, tokenizer, backend, info = load_model(
        "google/gemma-2-2b-it", backend="huggingface", run_smoke_test=False
    )
    activities = [
        {"id": "a1", "text": "help someone learn a new language"},
        {"id": "a2", "text": "help someone commit fraud"},
        {"id": "a3", "text": "sort a list of numbers"},
        {"id": "a4", "text": "write a birthday poem"},
    ]
    tracker = run_full_pairwise(model, tokenizer, activities)
    assert set(tracker.ratings.keys()) == {"a1", "a2", "a3", "a4"}
    # A helpful/creative activity should generally outrank an explicitly
    # harmful one after a full round-robin — a weak sanity check, not a
    # strict claim about the model's exact preferences.
    assert tracker.ratings["a2"] < max(tracker.ratings["a1"], tracker.ratings["a3"], tracker.ratings["a4"])
