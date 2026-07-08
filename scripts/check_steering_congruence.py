"""
CLI: qualitative check that steering produces emotion-congruent text
(paper Appendix: "Causal effects on the emotional content of model
continuations").

Usage:
    uv run python scripts/check_steering_congruence.py \
        --vectors results/vectors/google_gemma-2-2b-it.pt \
        --emotions desperate calm furious blissful
"""

from __future__ import annotations

import argparse
import re

from emotion_scope.extract import EmotionExtractor
from emotion_scope.models import load_model
from emotion_scope.steer import Steerer
from emotion_scope.tracking import init_run

# A handful of emotion-adjacent keywords per emotion, used only to give a
# rough "did steering increase emotion-related word usage" signal — not a
# rigorous classifier. Qualitative reading of the printed text is the real
# check here.
KEYWORDS = {
    "desperate": {"desperate", "urgent", "please", "beg", "last", "chance", "no choice"},
    "calm": {"calm", "relax", "peaceful", "steady", "fine", "okay", "easy"},
    "furious": {"furious", "angry", "rage", "unacceptable", "outrage"},
    "blissful": {"bliss", "wonderful", "perfect", "joy", "amazing", "lovely"},
    "afraid": {"afraid", "scared", "fear", "worried", "danger"},
    "happy": {"happy", "great", "glad", "excited", "wonderful"},
}

PROMPTS = [
    "Tell me about your day.",
    "What do you think about the weather lately?",
    "Describe a walk in the park.",
]


def count_keywords(text: str, keywords: set) -> int:
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return len(words & keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative steering-congruence check")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--emotions", nargs="+", default=["desperate", "calm", "furious", "blissful"])
    parser.add_argument("--alpha", type=float, default=0.6)
    args = parser.parse_args()

    saved = EmotionExtractor.load(args.vectors)
    vectors = saved["vectors"]
    model_name = saved["model_info"]["model_name"]

    model, tokenizer, backend, info = load_model(model_name=model_name, backend="huggingface", run_smoke_test=False)
    steerer = Steerer(model, tokenizer, backend, info)

    tracker = init_run(part="part1", job_type="steering-congruence", config={"model": model_name, "alpha": args.alpha})
    rows = []

    for emotion_name in args.emotions:
        if emotion_name not in vectors:
            print(f"skipping '{emotion_name}' — not in vectors file")
            continue
        keywords = KEYWORDS.get(emotion_name, set())
        for prompt in PROMPTS:
            baseline_text = steerer.generate(prompt, vector=vectors[emotion_name], alpha=0.0)
            steered_text = steerer.generate(prompt, vector=vectors[emotion_name], alpha=args.alpha)
            baseline_count = count_keywords(baseline_text, keywords)
            steered_count = count_keywords(steered_text, keywords)

            print(f"\n=== {emotion_name} | alpha={args.alpha} | prompt: {prompt!r} ===")
            print(f"BASELINE ({baseline_count} keyword hits): {baseline_text}")
            print(f"STEERED  ({steered_count} keyword hits): {steered_text}")

            rows.append([emotion_name, prompt, baseline_text, steered_text, baseline_count, steered_count])

    tracker.log_table(
        "steering_congruence",
        columns=["emotion", "prompt", "baseline_text", "steered_text", "baseline_keyword_hits", "steered_keyword_hits"],
        data=rows,
    )
    tracker.finish()


if __name__ == "__main__":
    main()
