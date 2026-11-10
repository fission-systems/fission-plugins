"""LoRA fine-tune a base model on a D0 set (prompt -> edit-list JSON), loss on the completion only.

Plain torch + transformers + peft, no trainer framework, so every choice below is visible. Used for M2a
(docs/PLAN_M2.md section 7): the same data, split and protocol for each candidate model.

  - Training functions are a nested prefix of one fixed shuffled order, so --functions 1000 is a subset of
    --functions 3000. Every build/mask variant of a chosen function is kept (the whole function, never part).
  - Labels are -100 on the prompt, so loss is on `completion` + EOS only. Each optimizer step divides the summed
    token loss by the number of target tokens in the step, not by the number of sequences.
  - Validation loss (same completion-only loss) on a fixed val subset every --eval-every steps; the adapter with
    the lowest validation loss is saved as <out>/best. The final adapter is <out>/final. The test split is
    never read here.
  - Validation here is in-project (see training/README.md): treat the loss as a pipeline check, not as
    generalization. Checkpoint choice by loss alone may differ from the choice by name accuracy.

usage: finetune.py --model Qwen/Qwen3.5-2B-Base --data <d0 dir> --out <new dir> --functions 3000 [--seed 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def function_key(example: dict) -> str:
    return f"{example['project']}/{example['name']}"


def pick_functions(examples: list[dict], n: int | None, seed: int) -> list[dict]:
    """Examples of the first n functions in a seeded shuffle that does not depend on n (nested subsets)."""
    keys = sorted({function_key(e) for e in examples})
    keys.sort(key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    chosen = set(keys if n is None else keys[:n])
    return [e for e in examples if function_key(e) in chosen]


def encode(tokenizer, example: dict) -> dict:
    prompt = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
    completion = tokenizer(example["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return {"input_ids": prompt + completion, "labels": [-100] * len(prompt) + completion}


def make_batches(lengths: list[int], max_tokens: int, rng: random.Random | None) -> list[list[int]]:
    """Batches of similar length whose padded size (longest * count) stays under max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, longest = [], [], 0
    for i in order:
        longest_next = max(longest, lengths[i])
        if cur and longest_next * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, longest_next = [], lengths[i]
        cur.append(i)
        longest = longest_next
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(items: list[dict], pad_id: int, device: str):
    import torch

    width = max(len(x["input_ids"]) for x in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for r, x in enumerate(items):  # right padding
        n = len(x["input_ids"])
        ids[r, :n] = torch.tensor(x["input_ids"])
        labels[r, :n] = torch.tensor(x["labels"])
        mask[r, :n] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def batch_loss_sum(model, batch, pad_id: int, device: str):
    """Summed token cross-entropy over target tokens and the number of target tokens."""
    import torch.nn.functional as F

    ids, labels, mask = batch
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    target = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, int((target != -100).sum())


def evaluate(model, items, pad_id, device, max_tokens) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for idx in make_batches([len(x["input_ids"]) for x in items], max_tokens, None):
            loss, n = batch_loss_sum(model, collate([items[i] for i in idx], pad_id, device), pad_id, device)
            total += float(loss)
            count += n
    model.train()
    return total / max(count, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--functions", type=int, default=None, help="train on the first N functions (default: all)")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch-tokens", type=int, default=8192, help="padded tokens per micro-batch")
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-examples", type=int, default=600, help="fixed val subset (functions' unmasked and masked)")
    p.add_argument("--seed", type=int, default=1, help="training seed (init of LoRA, batch order)")
    p.add_argument("--subset-seed", type=int, default=20261110, help="seed of the function order (keep fixed across runs)")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train_ex = pick_functions(load_jsonl(a.data / "train.jsonl"), a.functions, a.subset_seed)
    val_all = load_jsonl(a.data / "val.jsonl") + load_jsonl(a.data / "val-masked.jsonl")
    val_ex = pick_functions(val_all, a.val_examples, a.subset_seed)  # whole functions, both variants
    train = [encode(tok, e) for e in train_ex]
    val = [encode(tok, e) for e in val_ex]
    train = [x for x in train if len(x["input_ids"]) <= a.max_seq]
    val = [x for x in val if len(x["input_ids"]) <= a.max_seq]
    n_target = sum(sum(t != -100 for t in x["labels"]) for x in train)
    print(f"device={device} dtype={dtype} train: {len(train)} examples, "
          f"{len({function_key(e) for e in train_ex})} functions, {sum(len(x['input_ids']) for x in train)} tokens, "
          f"{n_target} target tokens | val: {len(val)} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))

    rng = random.Random(a.seed)
    lengths = [len(x["input_ids"]) for x in train]
    per_epoch = len(make_batches(lengths, a.batch_tokens, None))
    total_steps = max(1, math.ceil(a.epochs * per_epoch / a.accum))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)

    def lr_at(step: int) -> float:
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / max(1, total_steps - a.warmup)))

    a.out.mkdir(parents=True)
    log = (a.out / "log.jsonl").open("w")

    def record(**kv) -> None:
        log.write(json.dumps(kv) + "\n")
        log.flush()
        print(json.dumps(kv), flush=True)

    meta = dict(args={k: str(v) for k, v in vars(a).items()}, train_examples=len(train), val_examples=len(val),
                train_functions=len({function_key(e) for e in train_ex}), micro_batches_per_epoch=per_epoch,
                total_steps=total_steps,
                data_sha256={f: hashlib.sha256((a.data / f).read_bytes()).hexdigest() for f in ("train.jsonl", "val.jsonl", "val-masked.jsonl")})
    (a.out / "run.json").write_text(json.dumps(meta, indent=1) + "\n")

    record(step=0, val_loss=evaluate(model, val, pad_id, device, a.batch_tokens))
    best = float("inf")
    step, started = 0, time.time()
    model.train()
    stop = False
    while not stop:
        batches = make_batches(lengths, a.batch_tokens, rng)
        for g in range(0, len(batches), a.accum):
            group = batches[g : g + a.accum]
            n_tok = sum(sum(t != -100 for t in train[i]["labels"][1:]) for b in group for i in b)
            for param_group in opt.param_groups:
                param_group["lr"] = lr_at(step)
            running = 0.0
            for b in group:
                loss, _ = batch_loss_sum(model, collate([train[i] for i in b], pad_id, device), pad_id, device)
                (loss / n_tok).backward()
                running += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            record(step=step, train_loss=running / n_tok, lr=lr_at(step - 1), target_tokens=n_tok,
                   elapsed_s=round(time.time() - started, 1))
            if step % a.eval_every == 0 or step >= total_steps:
                v = evaluate(model, val, pad_id, device, a.batch_tokens)
                record(step=step, val_loss=v)
                if v < best:
                    best = v
                    model.save_pretrained(a.out / "best")
                    record(step=step, saved="best", val_loss=v)
            if step >= total_steps:
                stop = True
                break
    model.save_pretrained(a.out / "final")
    record(done=True, steps=step, best_val_loss=best, seconds=round(time.time() - started, 1))


if __name__ == "__main__":
    main()
