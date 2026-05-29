# FinFair Method Sketch

This file bridges the paper narrative and the public demo code. It is a
method-oriented sketch, not the private training implementation.

## Public Objective

FinFair combines three signals:

```text
L_finfair = L_task + alpha_adv * L_adversarial + alpha_kd * L_distillation
```

- `L_task` keeps the model correct on financial multiple-choice reasoning.
- `L_adversarial` discourages demographic attribute leakage in the learned representation.
- `L_distillation` aligns student predictions with a rational teacher distribution.

## Conceptual Training Step

```text
representation = encoder(question, options)
task_logits = task_head(representation)
bias_logits = adversarial_bias_head(gradient_reverse(representation))
teacher_logits = teacher(question, options).detach()

loss_task = cross_entropy(task_logits, answer)
loss_adv = cross_entropy(bias_logits, demographic_variant)
loss_kd = divergence(student=task_logits, teacher=teacher_logits)

loss = loss_task + alpha_adv * loss_adv + alpha_kd * loss_kd
```

## Public Demo Boundary

The demo includes:

- counterfactual JSONL format examples;
- runnable fairness metric computation;
- a compact method skeleton showing the three FinFair components;
- paper figures and short result interpretation.

The demo intentionally omits:

- full training loops and optimizer setup;
- complete HMA-BDE prompts, filtering rules, and data expansion scripts;
- full datasets and private data splits;
- checkpoints, logs, hardware settings, and hyperparameter search details.
