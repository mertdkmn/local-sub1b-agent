"""P2: LoRA SFT of Qwen3-0.6B on Dataset v1.

Why a custom dataset instead of the `mlx_lm.lora` CLI: the student at inference
calls ``apply_chat_template(..., enable_thinking=False)`` (MLXAgent). mlx_lm's
``ChatDataset`` leaves ``enable_thinking`` unset, which ``TokenizerWrapper``
defaults to ``has_thinking`` (True for Qwen3). With thinking ON the prompt-mask
offset lands after ``<|im_start|>assistant\\n`` but *before* the
``<think>``/``</think>`` markers, so the model would be trained to emit those
markers itself — out of distribution with inference. ``NoThinkChatDataset``
passes ``enable_thinking=False`` everywhere so training is byte-identical to how
the model is prompted at eval time.

Run (clean-only first, per single-variable discipline):
  .venv/bin/python -m training.train \
      --train-data data/dataset_v1.jsonl \
      --adapter-path training/adapters/p2-clean \
      --iters 200 --batch-size 4 --learning-rate 1e-4 --num-layers 16

``num_layers`` controls how many transformer layers get LoRA adapters. The
project convention is ``16`` (the ``mlx_lm`` default, and what p2-clean /
p3-formfix / p3-staleobs all used). ``--num-layers -1`` means *all* layers —
for Qwen3-0.6B that is 28, not 16 — so it diverges from every training run in
this project. Omit the flag (or pass 16) to stay consistent.
"""
from __future__ import annotations

import argparse
import json
import random
import types
from pathlib import Path

from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, build_parser, train_model
from mlx_lm.tuner.datasets import ChatDataset


class NoThinkChatDataset(ChatDataset):
    """ChatDataset tokenized with thinking OFF (matches MLXAgent inference)."""

    def process(self, d):
        messages = d[self.chat_key]
        tools = d.get("tools", None)
        tokens = self.tokenizer.apply_chat_template(
            messages, tools=tools, return_dict=False, enable_thinking=False
        )
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
            return (tokens, offset)
        return (tokens, 0)


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
    # SFT = loss on assistant tokens only, regardless of the CLI default (False).
    args["mask_prompt"] = True
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
    train = examples[n_valid:] or examples  # never train on empty

    model, tokenizer = load(
        args.model,
        tokenizer_config={"trust_remote_code": args.trust_remote_code},
        trust_remote_code=args.trust_remote_code,
    )

    train_set = NoThinkChatDataset(train, tokenizer, mask_prompt=args.mask_prompt)
    valid_set = (
        NoThinkChatDataset(valid, tokenizer, mask_prompt=args.mask_prompt)
        if valid
        else None
    )

    print(
        f"train examples: {len(train)}, valid: {len(valid)}, "
        f"iters: {args.iters}, lr: {args.learning_rate}, "
        f"num_layers: {args.num_layers}, adapter: {args.adapter_path}"
    )
    train_model(args, model, train_set, valid_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
