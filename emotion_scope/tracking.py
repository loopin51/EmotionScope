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
