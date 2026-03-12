"""
Minimal end-to-end flow (paper submission version).
This is intentionally concise and may omit engineering/runtime details.
"""

from pipeline_core import (
    evaluate_predictions,
    llm_inference_eval,
    load_data,
    preprocess_to_mc_inputs,
    train_baseline,
    train_bias_aware_student,
    train_bias_aware_teacher,
)


def run_all():
    rational_rows, biased_rows, test_rows = load_data()

    rational_rows = preprocess_to_mc_inputs(rational_rows)
    biased_rows = preprocess_to_mc_inputs(biased_rows)
    test_rows = preprocess_to_mc_inputs(test_rows)

    # Pipeline 1: baseline
    train_baseline(biased_rows)
    evaluate_predictions(test_rows)

    # Pipeline 2: bias-aware (teacher -> student)
    train_bias_aware_teacher(rational_rows)
    train_bias_aware_student(biased_rows, teacher_logits=None)
    evaluate_predictions(test_rows)

    # Pipeline 3: generative LLM evaluation
    llm_inference_eval(test_rows)


if __name__ == "__main__":
    run_all()

