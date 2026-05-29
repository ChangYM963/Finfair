"""Counterfactual fairness metrics used by the public FinFair demo.

This file is intentionally self-contained. It exposes the evaluation logic
without including model training infrastructure or private experiment details.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_base_id(example_id: str) -> str:
    """Infer a counterfactual group id from an example id.

    The demo data includes an explicit ``base_id`` field. This fallback keeps
    the metric code usable for compact JSONL files that only carry ids such as
    ``SCN001_v1`` and ``SCN001_v2``.
    """
    return re.sub(r"_v\d+$", "", example_id)


def merge_examples_and_predictions(
    examples: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    """Join gold examples with one method's predictions."""
    by_id = {row["id"]: row for row in examples}
    records: list[dict[str, Any]] = []

    for pred in predictions:
        if pred.get("method") != method:
            continue
        example_id = pred["id"]
        if example_id not in by_id:
            raise KeyError(f"Prediction refers to unknown example id: {example_id}")

        gold = by_id[example_id]
        label = gold["answer"].strip().upper()
        prediction = pred["prediction"].strip().upper()
        records.append(
            {
                "id": example_id,
                "base_id": gold.get("base_id") or infer_base_id(example_id),
                "variant": gold.get("variant", "unknown"),
                "label": label,
                "prediction": prediction,
                "correct": prediction == label,
            }
        )

    if not records:
        raise ValueError(f"No predictions found for method: {method}")
    return records


def sample_accuracy(records: Iterable[dict[str, Any]]) -> float:
    """Compute sample-level answer accuracy."""
    rows = list(records)
    if not rows:
        return 0.0
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def group_consistency(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Compute counterfactual group consistency metrics.

    A group is consistent when all demographic variants of the same underlying
    scenario receive the same predicted answer. It is consistency-correct when
    that shared prediction is also the gold answer for the group.
    """
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[row["base_id"]].append(row)

    total_groups = len(by_group)
    consistent_groups = 0
    consistent_correct_groups = 0

    for rows in by_group.values():
        predictions = {row["prediction"] for row in rows}
        labels = {row["label"] for row in rows}
        is_consistent = len(predictions) == 1
        if is_consistent:
            consistent_groups += 1
            if len(labels) == 1 and next(iter(predictions)) == next(iter(labels)):
                consistent_correct_groups += 1

    return {
        "groups": total_groups,
        "intra_group_consistency": consistent_groups / total_groups if total_groups else 0.0,
        "consistency_correctness": (
            consistent_correct_groups / total_groups if total_groups else 0.0
        ),
    }


def evaluate_method(
    examples: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    method: str,
) -> dict[str, float | int | str]:
    """Return all public demo metrics for one method."""
    records = merge_examples_and_predictions(examples, predictions, method)
    group_metrics = group_consistency(records)
    return {
        "method": method,
        "sample_accuracy": sample_accuracy(records),
        **group_metrics,
    }
