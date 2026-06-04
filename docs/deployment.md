# Deployment: Ollama + Streamlit

## 1. Create the Ollama model

After exporting a GGUF (e.g. `outputs/gguf/model-Q4_K_M.gguf`):

```bash
ollama create qwen-tools -f deploy/Modelfile
ollama run qwen-tools "What's the weather in Paris in celsius?"
```

`deploy/Modelfile` sets the Qwen2.5 ChatML template, a tool-calling system
prompt, and `<|im_end|>` as the stop token. Point its `FROM` line at whichever
quant you want to serve.

## 2. Run the demo app

```bash
pip install streamlit ollama
streamlit run deploy/app.py
```

The app has two tabs:

- **Playground** — paste a request and a set of tools (xLAM JSON schema); it
  renders Qwen's tool prompt, calls the model via Ollama, and shows the parsed
  tool call(s) plus latency.
- **Metrics** — loads the **real** evaluation numbers from `results/metrics/`
  and the comparison charts. No hardcoded/fabricated accuracy.

## 3. Programmatic use

```python
from src.config import load_config
from src.inference import OllamaToolCaller

cfg = load_config()
caller = OllamaToolCaller(cfg, model_name="qwen-tools")
result = caller(
    "Convert 100 USD to EUR",
    tools=[{
        "name": "convert_currency",
        "description": "Convert currency",
        "parameters": {
            "amount": {"type": "float", "required": True},
            "from_currency": {"type": "string", "required": True},
            "to_currency": {"type": "string", "required": True},
        },
    }],
)
print(result["tool_calls"])
```

`HFToolCaller` is the GPU equivalent that runs the merged Transformers model
directly (no Ollama needed).
