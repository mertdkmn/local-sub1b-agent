"""P2/P3: LoRA SFT of Qwen3-0.6B on Dataset v1, with optional action-weighted loss.

Two correctness features beyond the stock `mlx_lm.lora` CLI:

1. ``NoThinkChatDataset`` tokenizes with ``enable_thinking=False`` (matches MLXAgent
   inference). mlx_lm's ChatDataset leaves ``enable_thinking`` unset, which the
   TokenizerWrapper defaults to ``has_thinking`` (True for Qwen3), producing an
   out-of-distribution mask offset.

2. **Action-weighted loss (P3 sweep #2)**. The stock loss masks the prompt (loss on
   assistant tokens only) and weights every action token equally. Here each step's
   tokens are additionally scaled by an action-type weight, so critical-but-rare
   actions (SUBMIT, TYPE) get more gradient than CLICK/SCROLL. This directly targets
   the "CLICK-loop / never-submit" regression diagnosed in P3 sweep #1.

Run:
  .venv/bin/python -m training.train \
      --train-data data/dataset_v1.jsonl \
      --adapter-path training/adapters/p3-lossweight \
      --iters 200 --batch-size 4 --learning-rate 1e-4 --num-layers 16
"""
from __future__ import annotations

import argparse
import json
import random
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, build_parser
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    print_trainable_parameters,
)
from mlx_lm.utils import save_config

# Action-type weights (P3 loss sweep). A SUBMIT token gets 3x the gradient of a
# CLICK token; SCROLL/WAIT get 0.5x. Starting values — not claimed optimal.
ACTION_WEIGHTS = {
    "SUBMIT": 3.0,
    "TYPE": 2.0,
    "SEARCH": 2.0,
    "GOTO": 1.5,
    "CLICK": 1.0,
    "GO_BACK": 1.0,
    "SCROLL": 0.5,
    "WAIT": 0.5,
}


def action_weight(action_str: str) -> float:
    """Map a canonical action string ("TYPE(ref=5, 'x')") to its type weight."""
    tool = action_str.split("(")[0].strip().upper()
    return ACTION_WEIGHTS.get(tool, 1.0)


class NoThinkChatDataset(ChatDataset):
    """ChatDataset tokenized with thinking OFF + a per-step action weight.

    ``process`` returns ``(tokens, offset, weight)``: the weight is the action-type
    weight of this step's single action, applied to all its (non-prompt) tokens.
    """

    def process(self, d):
        messages = d[self.chat_key]
        tools = d.get("tools", None)
        tokens = self.tokenizer.apply_chat_template(
            messages, tools=tools, return_dict=False, enable_thinking=False
        )
        weight = 1.0
        offset = 0
        if self.mask_prompt:
            add_generation_prompt = messages[-1].get("role") == "assistant"
            offset = len(
                self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    return_dict=False,
                    enable_thinking=False,
                )
            )
            if messages[-1].get("role") == "assistant":
                weight = action_weight(messages[-1]["content"])
        return (tokens, offset, weight)


def weighted_loss(model, batch, lengths, weights):
    """Same as mlx_lm's default_loss, but the prompt mask is scaled per-example."""
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model(inputs)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    w = mx.array(weights, dtype=mx.float32).reshape(-1, 1)
    weighted_mask = mask.astype(mx.float32) * w

    ce = nn.losses.cross_entropy(logits, targets) * weighted_mask
    ntoks = weighted_mask.sum()
    ce = ce.astype(mx.float32).sum() / ntoks

    return ce, ntoks


def weighted_iterate_batches(
    dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None
):
    """mlx_lm's iterate_batches, but items are (tokens, offset, weight) triples."""
    if isinstance(dataset, CacheDataset):
        len_fn = lambda idx: dataset.itemlen(idx)
    else:
        len_fn = lambda idx: len(dataset[idx][0])
    idx = sorted(range(len(dataset)), key=len_fn)
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    if comm_group is not None:
        offset, step = comm_group.rank(), comm_group.size()
    else:
        offset, step = 0, 1
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    batch_idx = [
        idx[i + offset : i + offset + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]
    if seed is not None:
        np.random.seed(seed)

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            tokens, offsets, weights = zip(*batch)
            lengths = [len(x) for x in tokens]
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} "
                    "tokens; they will be truncated."
                )

            pad_to = 32
            max_len = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_len = min(max_len, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_len), np.int32)
            for j in range(batch_size // step):
                truncated = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated] = tokens[j][:truncated]
                lengths[j] = truncated

            yield (
                mx.array(batch_arr),
                mx.array(list(zip(offsets, lengths))),
                mx.array(weights, dtype=mx.float32),
            )

        if not loop:
            break


def train_model_weighted(args, model, train_set, valid_set):
    """Replicate mlx_lm.lora.train_model but pass the weighted loss + batcher."""
    mx.random.seed(args.seed)
    model.freeze()
    if args.num_layers > len(model.layers):
        raise ValueError(
            f"Requested to train {args.num_layers} layers "
            f"but the model only has {len(model.layers)} layers."
        )

    if args.fine_tune_type == "full":
        for l in model.layers[-max(args.num_layers, 0) :]:
            l.unfreeze()
        args.lora_parameters = None
    elif args.fine_tune_type in ["lora", "dora"]:
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {args.fine_tune_type}")

    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    optimizer_name = args.optimizer.lower()
    optimizer_config = args.optimizer_config.get(optimizer_name, {})
    opt_class = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "muon": optim.Muon,
        "sgd": optim.SGD,
        "adafactor": optim.Adafactor,
    }.get(optimizer_name)
    if opt_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    opt = opt_class(learning_rate=lr, **optimizer_config)

    train(
        model=model,
        args=training_args,
        optimizer=opt,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(valid_set),
        loss=weighted_loss,
        iterate_batches=weighted_iterate_batches,
    )


def load_examples(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def main(argv=None) -> int:
    parser = build_parser()
    parser.add_argument("--train-data", type=str, default="data/dataset_v1.jsonl")
    parser.add_argument("--valid-frac", type=float, default=0.1)
    args = parser.parse_args(argv)
    args = vars(args)
    for k, v in CONFIG_DEFAULTS.items():
        if args.get(k) is None:
            args[k] = v
    args["mask_prompt"] = True  # SFT = loss on assistant tokens only
    args = types.SimpleNamespace(**args)

    root = Path(__file__).resolve().parent.parent
    data_path = Path(args.train_data)
    if not data_path.is_absolute():
        data_path = root / data_path
    args.adapter_path = str((root / args.adapter_path).resolve())

    examples = load_examples(data_path)
    rng = random.Random(args.seed or 0)
    rng.shuffle(examples)
    n_valid = int(len(examples) * args.valid_frac) if len(examples) >= 10 else 0
    valid = examples[:n_valid]
    train_data = examples[n_valid:] or examples

    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )

    train_set = NoThinkChatDataset(train_data, tokenizer, mask_prompt=args.mask_prompt)
    valid_set = (
        NoThinkChatDataset(valid, tokenizer, mask_prompt=args.mask_prompt)
        if valid
        else None
    )

    print(
        f"train examples: {len(train_data)}, valid: {len(valid)}, "
        f"iters: {args.iters}, lr: {args.learning_rate}, "
        f"num_layers: {args.num_layers}, adapter: {args.adapter_path}, "
        f"action weights: {ACTION_WEIGHTS}"
    )
    train_model_weighted(args, model, train_set, valid_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
