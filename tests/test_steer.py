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


def test_generate_raises_for_transformer_lens_backend():
    from emotion_scope.steer import Steerer

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
