# Conversion: adapter → multiple GGUF quantizations

GGUF is the format `llama.cpp` / Ollama use to run models efficiently on CPU.
Where the reference project shipped a single `Q8_0`, we export several quants
and **measure** the quality/size trade-off.

## Pipeline (`src/gguf_convert.py`)

```
adapter + base ──merge──▶ fp16 HF model
fp16 model ──convert_hf_to_gguf.py──▶ model-f16.gguf
model-f16.gguf ──llama-imatrix (calibration)──▶ imatrix.dat
model-f16.gguf ──llama-quantize --imatrix──▶ Q4_K_M, Q5_K_M
model-f16.gguf ──llama-quantize──▶ Q8_0        (imatrix unneeded)
```

### Importance matrix (imatrix)

An imatrix records which weights matter most, computed by running the f16 model
over a small **calibration corpus** (we use real prompts from the train split).
`llama-quantize --imatrix` then preserves those weights more carefully. It
noticeably improves low-bit quants (≤ Q4); for `Q8_0` quantization is already
near-lossless, so we skip the imatrix there.

### Why `convert_hf_to_gguf.py` (not `convert.py`)

llama.cpp renamed/removed the old `convert.py`. The current entry point is
`convert_hf_to_gguf.py`, which supports the Qwen2 architecture directly.

## Quant types we emit (`config.yaml: gguf.quant_types`)

| Quant | ~Size (3B) | Use when |
|---|---|---|
| `Q4_K_M` | ~2.0 GB | default for laptops / lowest RAM |
| `Q5_K_M` | ~2.3 GB | a bit more quality |
| `Q8_0`   | ~3.4 GB | near-lossless, best fidelity |
| `f16`    | ~6.2 GB | reference / re-quantization source |

## Fallback

If building llama.cpp from source isn't possible, `save_gguf_unsloth()` uses
Unsloth's one-call `model.save_pretrained_gguf(..., quantization_method=[...])`.
Known issue: some mid-2025 Unsloth versions produced corrupt GGUFs for certain
4-bit conversions — if output looks garbled, use the manual llama.cpp path.

## Output

`outputs/gguf/model-<quant>.gguf` plus `results/metrics/gguf_manifest.json`
(file sizes), consumed by the quality-vs-size chart.
