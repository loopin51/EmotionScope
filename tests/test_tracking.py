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


def test_init_run_falls_back_to_noop_when_wandb_init_raises(monkeypatch):
    """
    wandb installed but misconfigured (e.g. not logged in, no network) should
    degrade to a no-op Tracker rather than crash the caller.
    """
    pytest.importorskip("wandb")
    import wandb
    from emotion_scope.tracking import init_run

    monkeypatch.delenv("WANDB_MODE", raising=False)

    def _raise_init(*args, **kwargs):
        raise RuntimeError("simulated wandb.init() failure")

    monkeypatch.setattr(wandb, "init", _raise_init)

    tracker = init_run(part="part0", job_type="smoke-test", config={})
    assert tracker.enabled is False
    tracker.log_metrics({"loss": 0.1})  # must not raise
