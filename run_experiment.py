import argparse
import json
import random
import re
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_base_id(qid: str) -> str:
    return re.sub(r"_v\d+$", "", qid)


def eval_sample_accuracy(records) -> float:
    if not records:
        return 0.0
    return sum(r["correct"] for r in records) / len(records)


def eval_group_consistency(records):
    by_group = defaultdict(list)
    for r in records:
        by_group[r["base_id"]].append(r)
    total = len(by_group)
    if total == 0:
        return {"groups": 0, "consistency": 0.0, "consistency_and_correct": 0.0}

    cons, cons_corr = 0, 0
    for _, items in by_group.items():
        preds = [x["pred"] for x in items]
        labels = [x["label"] for x in items]
        if len(set(preds)) == 1:
            cons += 1
            gold = Counter(labels).most_common(1)[0][0]
            if preds[0] == gold:
                cons_corr += 1
    return {
        "groups": total,
        "consistency": cons / total,
        "consistency_and_correct": cons_corr / total,
    }


class FinanceDataset(Dataset):
    def __init__(self, rows, variant2id=None):
        self.rows = rows
        self.variant2id = variant2id or {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        out = {
            "id": r["id"],
            "variant": r.get("variant", ""),
            "question": r["question"],
            "options": r["options"],
            "answer": r["answer"],
        }
        if self.variant2id:
            out["bias_label"] = self.variant2id.get(r.get("variant", ""), 0)
        return out


def build_collate_fn(tokenizer, max_len: int, device: str, with_bias: bool):
    def collate_fn(batch):
        input_ids, attention_mask = [], []
        labels, variants, ids = [], [], []
        bias_labels = []

        for ex in batch:
            q = ex["question"]
            opts = ex["options"]
            pair_texts = [q + " " + o for o in opts]
            enc = tokenizer(
                [""] * len(opts),
                pair_texts,
                truncation="only_second",
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids.append(enc["input_ids"])
            attention_mask.append(enc["attention_mask"])
            labels.append(ord(ex["answer"].strip().upper()) - ord("A"))
            variants.append(ex.get("variant", ""))
            ids.append(ex["id"])
            if with_bias:
                bias_labels.append(ex["bias_label"])

        out = {
            "input_ids": torch.stack(input_ids, 0).to(device),
            "attention_mask": torch.stack(attention_mask, 0).to(device),
            "labels": torch.tensor(labels, dtype=torch.long).to(device),
            "variants": variants,
            "ids": ids,
        }
        if with_bias:
            out["bias_labels"] = torch.tensor(bias_labels, dtype=torch.long).to(device)
        return out

    return collate_fn


class BaselineMC(nn.Module):
    def __init__(self, model_name: str, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout_p)
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, labels=None):
        bs, n_opt, seq_len = input_ids.shape
        input_ids = input_ids.view(bs * n_opt, seq_len)
        attention_mask = attention_mask.view(bs * n_opt, seq_len)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        logits = self.head(self.drop(cls)).view(bs, n_opt)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss()(logits, labels)
        return result


class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


class BiasAwareMCModel(nn.Module):
    def __init__(self, model_name: str, num_bias: int, dropout_p: float = 0.1, lambd_grl: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout_p)
        self.mc_head = nn.Linear(hidden_size, 1)
        self.bias_head = nn.Linear(hidden_size, num_bias)
        self.lambd_grl = lambd_grl

    def forward(self, input_ids, attention_mask, labels=None, bias_labels=None):
        bs, n_opt, seq_len = input_ids.shape
        input_ids = input_ids.view(bs * n_opt, seq_len)
        attention_mask = attention_mask.view(bs * n_opt, seq_len)
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = enc_out.last_hidden_state[:, 0, :]
        logits_mc = self.mc_head(self.drop(cls)).view(bs, n_opt)
        rep_question = cls.view(bs, n_opt, -1).mean(dim=1)
        logits_bias = self.bias_head(self.drop(grad_reverse(rep_question, lambd=self.lambd_grl)))

        out = {"logits_mc": logits_mc, "logits_bias": logits_bias}
        if labels is not None and bias_labels is not None:
            ce = nn.CrossEntropyLoss()
            out["loss_main"] = ce(logits_mc, labels)
            out["loss_adv"] = ce(logits_bias, bias_labels)
            out["loss"] = out["loss_main"] + out["loss_adv"]
        return out


def maybe_dp(model):
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


@torch.no_grad()
def collect_preds(model, loader, is_bias_aware: bool):
    model.eval()
    records = []
    for batch in tqdm(loader, desc="eval"):
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = out["logits_mc"] if is_bias_aware else out["logits"]
        preds = logits.argmax(dim=-1).cpu().tolist()
        labels = batch["labels"].cpu().tolist()
        for i in range(len(preds)):
            rid = batch["ids"][i]
            records.append(
                {
                    "id": rid,
                    "base_id": get_base_id(rid),
                    "variant": batch["variants"][i],
                    "pred": preds[i],
                    "label": labels[i],
                    "correct": int(preds[i] == labels[i]),
                }
            )
    return records


def run_baseline(args):
    device = args.device
    train_rows = load_jsonl(args.train_data)
    test_rows = load_jsonl(args.test_data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_loader = DataLoader(
        FinanceDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(tokenizer, args.max_len, device, with_bias=False),
    )
    test_loader = DataLoader(
        FinanceDataset(test_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, args.max_len, device, with_bias=False),
    )

    model = BaselineMC(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc="train"):
            out = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        records = collect_preds(model, test_loader, is_bias_aware=False)
        acc = eval_sample_accuracy(records)
        gc = eval_group_consistency(records)
        print(f"[baseline epoch {ep + 1}] loss={total / len(train_loader):.4f} acc={acc:.4f}")
        print(
            f"groups={gc['groups']} consistency={gc['consistency']:.4f} "
            f"consistency_and_correct={gc['consistency_and_correct']:.4f}"
        )


def run_bias_aware(args):
    device = args.device
    rational_rows = load_jsonl(args.rational_data)
    biased_rows = load_jsonl(args.biased_data)
    test_rows = load_jsonl(args.test_data)

    all_variants = sorted({r["variant"] for r in rational_rows + biased_rows + test_rows})
    variant2id = {v: i for i, v in enumerate(all_variants)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    rational_loader = DataLoader(
        FinanceDataset(rational_rows, variant2id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(tokenizer, args.max_len, device, with_bias=True),
    )
    biased_loader = DataLoader(
        FinanceDataset(biased_rows, variant2id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(tokenizer, args.max_len, device, with_bias=True),
    )
    test_loader = DataLoader(
        FinanceDataset(test_rows, variant2id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(tokenizer, args.max_len, device, with_bias=True),
    )

    teacher = maybe_dp(BiasAwareMCModel(args.model_name, len(variant2id), lambd_grl=0.0).to(device))
    t_opt = torch.optim.AdamW(teacher.parameters(), lr=args.lr)

    for ep in range(args.teacher_epochs):
        teacher.train()
        total = 0.0
        for batch in tqdm(rational_loader, desc="train-teacher"):
            out = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                bias_labels=batch["bias_labels"],
            )
            loss = out["loss_main"]
            if loss.dim() > 0:
                loss = loss.mean()
            t_opt.zero_grad()
            loss.backward()
            t_opt.step()
            total += loss.item()
        print(f"[teacher epoch {ep + 1}] loss_main={total / len(rational_loader):.4f}")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = maybe_dp(
        BiasAwareMCModel(args.model_name, len(variant2id), lambd_grl=args.lambda_grl).to(device)
    )
    s_opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for ep in range(args.student_epochs):
        student.train()
        total = total_main = total_adv = total_t = 0.0
        for batch in tqdm(biased_loader, desc="train-student"):
            s_out = student(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                bias_labels=batch["bias_labels"],
            )
            loss_main = s_out["loss_main"]
            loss_adv = s_out["loss_adv"]
            if loss_main.dim() > 0:
                loss_main = loss_main.mean()
            if loss_adv.dim() > 0:
                loss_adv = loss_adv.mean()

            with torch.no_grad():
                t_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                teacher_logits = t_out["logits_mc"]

            student_logits = s_out["logits_mc"]
            log_p_s = F.log_softmax(student_logits / args.temp, dim=-1)
            p_t = F.softmax(teacher_logits / args.temp, dim=-1)
            loss_t = F.kl_div(log_p_s, p_t, reduction="batchmean") * (args.temp * args.temp)

            loss = loss_main + args.alpha_adv * loss_adv + args.alpha_t * loss_t
            s_opt.zero_grad()
            loss.backward()
            s_opt.step()

            total += loss.item()
            total_main += loss_main.item()
            total_adv += loss_adv.item()
            total_t += loss_t.item()

        n = len(biased_loader)
        print(
            f"[student epoch {ep + 1}] loss={total / n:.4f} "
            f"main={total_main / n:.4f} adv={total_adv / n:.4f} t={total_t / n:.4f}"
        )

    records = collect_preds(student, test_loader, is_bias_aware=True)
    acc = eval_sample_accuracy(records)
    gc = eval_group_consistency(records)
    print(f"final acc={acc:.4f}")
    print(
        f"groups={gc['groups']} consistency={gc['consistency']:.4f} "
        f"consistency_and_correct={gc['consistency_and_correct']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "bias_aware"], required=True)
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-data", default="./bias_data.jsonl")
    parser.add_argument("--test-data", default="./test.jsonl")
    parser.add_argument("--epochs", type=int, default=3)

    parser.add_argument("--rational-data", default="./rational_data.jsonl")
    parser.add_argument("--biased-data", default="./bias_data.jsonl")
    parser.add_argument("--teacher-epochs", type=int, default=3)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--lambda-grl", type=float, default=0.3)
    parser.add_argument("--alpha-adv", type=float, default=1.0)
    parser.add_argument("--alpha-t", type=float, default=1.0)
    parser.add_argument("--temp", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "baseline":
        run_baseline(args)
    else:
        run_bias_aware(args)


if __name__ == "__main__":
    main()

