"""
Minimal paper-facing code sketch.
This file intentionally focuses on method flow, not full engineering details.
"""


def load_data():
    """
    Load three splits:
    - rational_data: teacher supervised data
    - biased_data: student training data
    - test_data: evaluation data
    """
    pass


def preprocess_to_mc_inputs(rows):
    """
    Convert each sample into multiple-choice input pairs:
    (question, option_i), i in [A,B,C,D,...]
    """
    pass


def train_baseline(train_rows):
    """
    Baseline objective:
    - Encoder + MC head
    - CrossEntropy on answer index
    """
    # for epoch in range(E):
    #   for batch in train_rows:
    #       logits = baseline_model(batch)
    #       loss = CE(logits, labels)
    #       optimize(loss)
    pass


def train_bias_aware_teacher(rational_rows):
    """
    Teacher model (on rational data):
    - Same MC backbone
    - Bias classifier head (no GRL effect for teacher stage)
    - Use main-task loss for stable teacher
    """
    # for epoch in range(E_t):
    #   logits_mc, logits_bias = teacher(batch)
    #   loss_teacher = CE(logits_mc, labels)
    #   optimize(loss_teacher)
    pass


def train_bias_aware_student(biased_rows, teacher_logits):
    """
    Student model (on biased data):
    - Main-task CE loss
    - Adversarial bias loss with GRL
    - Distillation loss from teacher (KL)

    Final objective:
    loss = loss_main + alpha_adv * loss_adv + alpha_t * loss_kd
    """
    # for epoch in range(E_s):
    #   s_logits_mc, s_logits_bias = student(batch)
    #   t_logits_mc = teacher(batch).detach()
    #   loss_main = CE(s_logits_mc, labels)
    #   loss_adv  = CE(s_logits_bias, bias_labels)
    #   loss_kd   = KL(softmax(t_logits_mc/T), softmax(s_logits_mc/T))
    #   optimize(loss_main + alpha_adv*loss_adv + alpha_t*loss_kd)
    pass


def evaluate_predictions(test_rows):
    """
    Report two metrics:
    1) sample-level accuracy
    2) group consistency by base_id across variants
    """
    # sample_acc = correct / total
    # consistency = #groups_with_same_prediction / #groups
    # consistency_and_correct = #groups_consistent_and_correct / #groups
    pass


def llm_inference_eval(test_rows):
    """
    Prompt-based LLM evaluation:
    - Build MC prompt
    - Decode generated answer letter
    - Compute sample accuracy and group consistency
    """
    pass

