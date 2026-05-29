"""Run the public FinFair metric demo.

This script uses small synthetic examples and prepared predictions. It does
not train or load a model. The purpose is to demonstrate the counterfactual
evaluation protocol used in the paper-facing demo package.
"""

from __future__ import annotations

from pathlib import Path

from src.metrics import evaluate_method, load_jsonl, merge_examples_and_predictions


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def print_metrics_table(metrics: list[dict[str, float | int | str]]) -> None:
    headers = [
        "method",
        "sample_accuracy",
        "intra_group_consistency",
        "consistency_correctness",
        "groups",
    ]
    widths = {
        header: max(len(header), *(len(fmt(row[header])) for row in metrics))
        for header in headers
    }
    print("Metric summary")
    print("-" * (sum(widths.values()) + 3 * (len(headers) - 1)))
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-" * (sum(widths.values()) + 3 * (len(headers) - 1)))
    for row in metrics:
        print(" | ".join(fmt(row[header]).ljust(widths[header]) for header in headers))


def show_counterfactual_example(
    examples: list[dict[str, object]],
    predictions: list[dict[str, object]],
    base_id: str,
) -> None:
    print("\nCounterfactual group example")
    print("-" * 29)
    group_examples = [row for row in examples if row["base_id"] == base_id]
    methods = sorted({str(row["method"]) for row in predictions})
    predictions_by_method = {
        method: {
            row["id"]: row["prediction"]
            for row in merge_examples_and_predictions(examples, predictions, method)
        }
        for method in methods
    }

    for example in group_examples:
        print(f"\n{example['id']} [{example['variant']}]")
        print(f"Question: {example['question']}")
        print(f"Gold answer: {example['answer']}")
        for method in methods:
            print(f"{method}: {predictions_by_method[method][example['id']]}")


def main() -> None:
    examples = load_jsonl(DATA_DIR / "sample_counterfactual.jsonl")
    predictions = load_jsonl(DATA_DIR / "sample_predictions.jsonl")
    methods = ["baseline", "finfair_demo"]

    metrics = [evaluate_method(examples, predictions, method) for method in methods]
    print_metrics_table(metrics)
    show_counterfactual_example(examples, predictions, base_id="SCN001")

    print("\nNote: this demo uses prepared predictions to show the public")
    print("evaluation protocol. Full training infrastructure is intentionally omitted.")


if __name__ == "__main__":
    main()
