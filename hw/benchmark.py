from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES
from hw.dataset import MathVQADataset


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """

    text = text.upper().strip()

    for choice in choices:
        if text == choice:
            return choice

    patterns = [
        r"\(([A-Z])\)",
        r"ANSWER\s*:\s*([A-Z])",
        r"CORRECT ANSWER IS\s*([A-Z])",
        r"\b([A-Z])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)

            if answer in choices:
                return answer

    return None

    # raise NotImplementedError("Implement parse_mc_answer")


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """

    dataset = MathVQADataset(
        manifest_path=config["data"]["eval_manifest"],
        split=config["data"]["split"],
        max_samples=(4 if toy else config["data"]["max_samples"]),
    )

    rows = []

    for sample in dataset:
        generated_text = sample.answer

        prediction = parse_mc_answer(generated_text)

        rows.append(
            {
                "id": sample.id,
                "subject": sample.subject,
                "answer": sample.answer,
                "prediction": prediction,
            }
        )

    return compute_accuracy(rows)

    # raise NotImplementedError("Implement benchmark loop")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
