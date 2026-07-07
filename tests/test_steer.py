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
