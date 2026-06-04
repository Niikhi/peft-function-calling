# Qwen2.5-3B Function-Calling — QLoRA Fine-Tuning + GGUF Quantization

Fine-tune **Qwen2.5-3B-Instruct** to turn natural-language requests into
structured **function/tool calls**, evaluate it with **real metrics**, export
**multiple GGUF quantizations**, and deploy locally with **Ollama + Streamlit**.

Trained on the **real** [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
dataset — **no synthetic data generation**. Runs end-to-end on a **free Kaggle/Colab T4 (16 GB)**.

```
"What's the weather in Paris in celsius, and convert 100 USD to EUR?"
        │
        ▼   (fine-tuned Qwen2.5-3B)
<tool_call>{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}</tool_call>
<tool_call>{"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}</tool_call>
```

---

## What makes this build solid

This project was built as an improved take on a common pattern (QLoRA fine-tune
→ GGUF → Ollama). The emphasis is on the parts that are usually skipped:

| Area | This project |
|---|---|
| **Data** | Real, human-curated xLAM-60k (gated → automatic ungated-mirror fallback). No synthetic generation. |
| **Model** | Qwen2.5-3B-Instruct — strong native tool-calling chat template. |
| **Training** | **Unsloth** QLoRA (~2× faster, ~50% less VRAM than vanilla TRL); loss masked to the completion only. |
| **Evaluation** | **Measured** on a held-out test split: exact match, function-name accuracy, argument F1, hallucination rate, single- vs multi-call breakdown — base vs fine-tuned. |
| **Visualization** | Loss curves, dataset EDA, metric comparison, error-type breakdown, quant quality-vs-size, latency. |
| **Quantization** | Multiple GGUF quants (Q4_K_M / Q5_K_M / Q8_0) with an **importance matrix**, plus measured quality cost per quant. |
| **Engineering** | Fully config-driven, modular `src/`, Kaggle/local path auto-detection, no hardcoded paths. |

---

## Project structure

```
peft-finetuning/
├── config/config.yaml      # every setting: model, LoRA, data, training, eval, gguf, paths
├── src/                    # modular library
│   ├── config.py           # YAML + Kaggle/local path resolution
│   ├── data.py             # load/normalize xLAM, JSON-schema tools, splits
│   ├── prompts.py          # Qwen tool-calling chat-template formatting
│   ├── model.py            # Unsloth load + LoRA
│   ├── train.py            # TRL SFTTrainer (completion-only masking)
│   ├── parsing.py          # robust <tool_call> extraction
│   ├── metrics.py          # all evaluation metrics (pure, tested)
│   ├── evaluate.py         # generate + score (HF model or Ollama GGUF)
│   ├── visualize.py        # all charts
│   ├── gguf_convert.py     # merge → f16 → imatrix → multi-quant
│   └── inference.py        # HFToolCaller / OllamaToolCaller
├── scripts/                # thin CLIs: prepare_data, run_train, run_eval, convert_gguf
├── notebooks/              # Kaggle-ready: 01 EDA · 02 train · 03 eval · 04 gguf
├── deploy/                 # Modelfile (Ollama) + app.py (Streamlit)
├── docs/                   # training / evaluation / conversion / deployment deep-dives
└── results/                # figures/ + metrics/ (generated)
```

---

## Quickstart (Kaggle / Colab T4)

1. **Enable GPU + Internet** in the notebook settings.
2. Open `notebooks/02_train_kaggle.ipynb` and run it. Each notebook's first cell
   clones this repo (set `REPO_URL`) and `pip install`s `unsloth`.
3. For the **gated** dataset, log in once:
   ```python
   from huggingface_hub import notebook_login; notebook_login()
   ```
   (Accept the dataset terms on its HF page first.) If you skip this, the code
   automatically falls back to an ungated mirror.

Run order: **01 → 02 → 03 → 04**.

### Or via CLI (on a GPU box)

```bash
pip install -r requirements.txt
python scripts/prepare_data.py     # → data/{train,val,test}.jsonl
python scripts/run_train.py        # → outputs/adapter + loss curves
python scripts/run_eval.py         # → metrics + comparison charts
python scripts/convert_gguf.py     # → outputs/gguf/*.gguf
```

---

## Results & visualizations

After running the pipeline, charts land in `results/figures/`:

- `loss_curves.png` — training vs validation loss
- `dataset_eda.png` — tools/query, calls/answer, query-length distributions
- `metric_comparison.png` — **base vs fine-tuned** across all headline metrics
- `error_breakdown.png` — error taxonomy (wrong-args / wrong-fn / hallucination / invalid)
- `quant_quality_size.png` — exact-match vs file size across quants
- `latency.png` — speed per quant

Raw numbers are in `results/metrics/*.json`. The Streamlit **Metrics** tab
renders them live — every number shown is measured, none hardcoded.

---

## Deployment

```bash
ollama create qwen-tools -f deploy/Modelfile     # uses outputs/gguf/model-Q4_K_M.gguf
streamlit run deploy/app.py                       # playground + metrics dashboard
```

See [`docs/deployment.md`](docs/deployment.md).

---

## Configuration & model swapping

Everything is in [`config/config.yaml`](config/config.yaml). To change the base
model, edit `model.base_id` + `model.hf_id`:

| Model | License | Fits T4? |
|---|---|---|
| **Qwen2.5-3B-Instruct** (default) | `qwen-research` (research only) | ✅ |
| Qwen2.5-1.5B-Instruct | Apache-2.0 (fully open) | ✅ (smaller/faster) |
| Qwen2.5-7B-Instruct | Apache-2.0 | needs ~24 GB VRAM |

> ⚠️ **License note:** Qwen2.5-**3B** is released under the research-only
> `qwen-research` license — fine for learning/portfolio/research, but switch to
> 1.5B or 7B (both Apache-2.0) for permissive/commercial use.

---

## Documentation

- [Training](docs/training.md) — QLoRA + Unsloth choices and tuning
- [Evaluation](docs/evaluation.md) — what each metric means and how it's computed
- [Conversion](docs/conversion.md) — GGUF, imatrix, and quant trade-offs
- [Deployment](docs/deployment.md) — Ollama + Streamlit

## Credits / licenses

- Base model: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (`qwen-research`)
- Dataset: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC-BY-4.0)
- Tooling: [Unsloth](https://github.com/unslothai/unsloth) · [TRL](https://github.com/huggingface/trl) · [llama.cpp](https://github.com/ggml-org/llama.cpp) · [Ollama](https://ollama.ai)
