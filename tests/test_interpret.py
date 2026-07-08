"""Tests for emotion_scope.interpret — logit-lens token analysis."""

import torch


class _FakeTokenizer:
    """Minimal tokenizer double: decode(ids) -> f'tok{id}'."""
    def decode(self, ids):
        return f"tok{ids[0]}"


def test_top_k_tokens_for_direction_finds_aligned_row():
    from emotion_scope.interpret import top_k_tokens_for_direction

    # unembed: 6 vocab rows, d_model=3. Row 0 points strongly in +x, row 5 in -x.
    unembed = torch.zeros(6, 3)
    unembed[0] = torch.tensor([5.0, 0.0, 0.0])
    unembed[5] = torch.tensor([-5.0, 0.0, 0.0])
    for i in (1, 2, 3, 4):
        unembed[i] = torch.tensor([0.0, 0.1 * i, 0.0])

    direction = torch.tensor([1.0, 0.0, 0.0])
    tokenizer = _FakeTokenizer()

    up, down = top_k_tokens_for_direction(direction, unembed, tokenizer, k=1)
    assert up == ["tok0"]
    assert down == ["tok5"]


def test_top_k_tokens_for_direction_respects_k():
    from emotion_scope.interpret import top_k_tokens_for_direction

    unembed = torch.eye(5, 5)
    direction = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    tokenizer = _FakeTokenizer()

    up, down = top_k_tokens_for_direction(direction, unembed, tokenizer, k=2)
    assert len(up) == 2
    assert len(down) == 2
