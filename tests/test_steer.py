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


@pytest.mark.slow
def test_steer_context_modifies_huggingface_activation():
    """
    Injecting a steering vector should measurably shift the model's output
    logits (a downstream consequence of the residual stream actually being
    modified in the middle layers).

    This does NOT compare `model(output_hidden_states=True)` tuples directly:
    that readout was found (via manual investigation while implementing this
    test) to not reliably reflect forward-hook modifications for this
    Gemma2 HF implementation/version — a hook registered on the same layer
    module, immediately after steer_context's own hook, DOES see the
    steered value (confirmed manually), but the model's own
    `hidden_states` bookkeeping does not consistently match it. Comparing
    final logits sidesteps that internal-bookkeeping ambiguity and directly
    tests the thing that actually matters: steering causally changes the
    model's output.
    """
    from emotion_scope.models import load_model
    from emotion_scope.steer import steer_context, middle_third_layers

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

    with torch.no_grad():
        unsteered_logits = model(**tokens).logits[0, -1, :].clone()

    with steer_context(model, backend, vector, alpha=0.5, layers=layers, avg_norms=avg_norms):
        with torch.no_grad():
            steered_logits = model(**tokens).logits[0, -1, :].clone()

    max_diff = (unsteered_logits - steered_logits).abs().max().item()
    assert max_diff > 0.1, f"steering barely changed output logits (max abs diff = {max_diff})"
