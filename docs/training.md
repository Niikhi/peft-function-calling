# Training: QLoRA fine-tuning with Unsloth

## Why QLoRA + Unsloth

- **QLoRA** loads the base model in 4-bit and trains small low-rank adapters
  (~1% of parameters). This is what lets a 3B model fine-tune inside a free
  16 GB T4.
- **Unsloth** rewrites the attention/MLP kernels and the LoRA path, giving
  ~2× faster training and ~50% less VRAM than vanilla TRL + PEFT, with no
  change in final quality. We still use TRL's `SFTTrainer` underneath — Unsloth
  just patches the model.

## What we train on

Each example is rendered with Qwen's **native tool-calling chat template**:

```
<|im_start|>system
# Tools ... (the available functions, injected by apply_chat_template(tools=...))
<|im_start|>user
<the natural-language request>
<|im_start|>assistant
<tool_call>
{"name": "...", "arguments": {...}}
</tool_call>
```

We compute loss **only on the assistant completion** (the `<tool_call>` blocks)
via `DataCollatorForCompletionOnlyLM`, keyed on the `<|im_start|>assistant\n`
marker. The model never wastes capacity learning to reproduce the prompt.

## Key hyperparameters (`config/config.yaml`)

| Param | Value | Note |
|---|---|---|
| LoRA rank `r` | 16 | capacity; alpha == r is a robust Unsloth default |
| target modules | all attn + MLP proj | q/k/v/o, gate/up/down |
| max_seq_length | 2048 | safe on T4; raise to ~4096 with grad checkpointing |
| batch × grad-accum | 2 × 8 | effective batch 16 |
| LR / scheduler | 2e-4 / cosine | warmup 3% |
| epochs | 1 | 60k examples is plenty; bump if underfitting |
| optimizer | adamw_8bit | memory-efficient |

## Tuning guide

- **OOM on T4** → drop `per_device_train_batch_size` to 1 (raise grad-accum to
  16), or lower `max_seq_length`.
- **Underfitting** (eval loss plateaus high) → 2–3 epochs, or `r=32`.
- **Faster iteration** → set `data.max_samples` to e.g. 5000.

## Outputs

- `outputs/adapter/` — the LoRA adapter + tokenizer (~100–200 MB).
- `results/metrics/train_log_history.json` — losses/LR for plotting.
- `results/figures/loss_curves.png`, `lr_schedule.png`.
