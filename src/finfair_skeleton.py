"""Method skeleton for the public FinFair demo.

The goal of this file is to show the method structure, not to reproduce the
private training stack. It deliberately omits tokenizer details, model
selection, optimizer settings, training loops, checkpointing, and full
HMA-BDE data generation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TensorLike(Protocol):
    """Minimal protocol used to keep this skeleton framework-agnostic."""

    def __add__(self, other: "TensorLike") -> "TensorLike": ...

    def __mul__(self, other: float) -> "TensorLike": ...

    def detach(self) -> "TensorLike": ...


class EncoderBackbone:
    """Placeholder for a lightweight encoder.

    In the full experiment this role is filled by compact encoder models.
    The public demo keeps it abstract to avoid exposing engineering details.
    """

    def encode(self, batch: object) -> TensorLike:
        raise NotImplementedError("Demo skeleton only: plug in a lightweight encoder.")


class TaskHead:
    """Financial multiple-choice reasoning head."""

    def logits(self, representation: TensorLike) -> TensorLike:
        raise NotImplementedError

    def loss(self, task_logits: TensorLike, labels: object) -> TensorLike:
        raise NotImplementedError


class AdversarialBiasHead:
    """Sensitive-attribute adversary attached through gradient reversal."""

    def logits_after_gradient_reversal(self, representation: TensorLike) -> TensorLike:
        raise NotImplementedError

    def loss(self, bias_logits: TensorLike, attribute_labels: object) -> TensorLike:
        raise NotImplementedError


class TeacherGuidanceModule:
    """Rational-distribution alignment from a teacher model."""

    def teacher_logits(self, batch: object) -> TensorLike:
        raise NotImplementedError

    def distillation_loss(
        self,
        student_logits: TensorLike,
        teacher_logits: TensorLike,
    ) -> TensorLike:
        raise NotImplementedError


@dataclass(frozen=True)
class FinFairWeights:
    """Public-facing objective weights with non-experimental defaults."""

    alpha_adv: float = 1.0
    alpha_kd: float = 1.0


def finfair_objective(
    task_loss: TensorLike,
    adversarial_loss: TensorLike,
    distillation_loss: TensorLike,
    weights: FinFairWeights = FinFairWeights(),
) -> TensorLike:
    """Combine the three conceptual FinFair losses.

    Full objective:
        task_loss + alpha_adv * adversarial_loss + alpha_kd * distillation_loss
    """
    return (
        task_loss
        + adversarial_loss * weights.alpha_adv
        + distillation_loss * weights.alpha_kd
    )


def training_step_skeleton(
    batch: object,
    encoder: EncoderBackbone,
    task_head: TaskHead,
    bias_head: AdversarialBiasHead,
    teacher: TeacherGuidanceModule,
    weights: FinFairWeights = FinFairWeights(),
) -> TensorLike:
    """Illustrate one conceptual FinFair training step.

    This is not executable by design. It documents the public method flow while
    keeping the full experimental implementation private.
    """
    representation = encoder.encode(batch)
    task_logits = task_head.logits(representation)
    bias_logits = bias_head.logits_after_gradient_reversal(representation)
    rational_logits = teacher.teacher_logits(batch).detach()

    task = task_head.loss(task_logits, labels="answer_labels")
    adversarial = bias_head.loss(bias_logits, attribute_labels="variant_labels")
    distillation = teacher.distillation_loss(task_logits, rational_logits)
    return finfair_objective(task, adversarial, distillation, weights)
