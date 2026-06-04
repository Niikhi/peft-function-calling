# Evaluation: real metrics, not loss

The reference project reported only training/eval loss and an unmeasured
"95% accuracy". We evaluate on a **held-out test split** the model never saw,
parsing each prediction and scoring it against ground truth.

## The metrics (`src/metrics.py`)

| Metric | Question it answers |
|---|---|
| **Validity** | Did the model emit a parseable `<tool_call>` at all? |
| **Function-name accuracy** | Are the called function names correct (as a multiset)? |
| **Exact match** | Names **and** every argument correct? (the strict headline number) |
| **Argument precision/recall/F1** | Partial credit on argument key/value correctness |
| **Hallucination rate** | Did it call a function not in the provided tool list? |
| **Count accuracy** | Right number of calls (matters for multi-call queries)? |
| **Exact match (single vs multi-call)** | Where does it struggle? |
| **Latency / sec-per-example** | Speed, reported per quant level |

### How comparison works

Predictions are parsed with `src/parsing.py` (robust to tags, bare JSON, or
embedded objects). Calls are aligned to gold by function name; arguments are
compared after normalization (`"5"` == `5`, `"true"` == `true`, whitespace
trimmed) so trivial formatting differences don't count as errors.

### Error taxonomy (for the breakdown chart)

Each example gets exactly one label, in priority order:

1. `invalid_output` — nothing parseable
2. `hallucinated_function` — called a tool that wasn't offered
3. `wrong_function` — wrong name/count
4. `wrong_arguments` — right function, wrong args
5. `correct`

## What we compare

`scripts/run_eval.py` scores the **base** model and the **fine-tuned** model on
identical prompts, writing `results/metrics/eval_base.json` and
`eval_finetuned.json`, then renders:

- `metric_comparison.png` — headline metrics, base vs fine-tuned
- `error_breakdown.png` — stacked error taxonomy per model

Notebook 04 additionally scores each **GGUF quant** (via Ollama) to show how
much quality each quantization level costs.
