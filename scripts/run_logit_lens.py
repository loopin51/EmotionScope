"""
CLI: logit-lens analysis for all emotion vectors — reproduces Table 1.

Usage:
    uv run python scripts/run_logit_lens.py --vectors results/vectors/google_gemma-2-2b-it.pt
    uv run python scripts/run_logit_lens.py --vectors results/vectors/google_gemma-2-2b-it.pt --top-k 5
"""

from __future__ import annotations

import argparse

from emotion_scope.extract import EmotionExtractor
from emotion_scope.interpret import get_unembed_matrix, top_k_tokens_for_direction
from emotion_scope.models import load_model
from emotion_scope.tracking import init_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Logit-lens analysis for emotion vectors")
    parser.add_argument("--vectors", required=True, help="Path to saved emotion vectors .pt")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    vectors = saved["vectors"]
    model_name = saved["model_info"]["model_name"]

    model, tokenizer, backend, info = load_model(model_name=model_name, run_smoke_test=False)
    unembed = get_unembed_matrix(model, backend)

    tracker = init_run(part="part1", job_type="logit-lens", config={"model": model_name, "top_k": args.top_k})

    rows = []
    for name, vector in vectors.items():
        up, down = top_k_tokens_for_direction(vector, unembed, tokenizer, k=args.top_k)
        print(f"\n{name}")
        print(f"  UP:   {up}")
        print(f"  DOWN: {down}")
        rows.append([name, ", ".join(up), ", ".join(down)])

    tracker.log_table("logit_lens_table", columns=["emotion", "top_up_tokens", "top_down_tokens"], data=rows)
    tracker.finish()


if __name__ == "__main__":
    main()
