from __future__ import annotations

import json
import pathlib
import sys
import time

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

st.set_page_config(page_title="Qwen2.5 Tool-Calling", page_icon="🛠️", layout="wide")

cfg = load_config()

DEFAULT_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "city": {"type": "string", "description": "City name", "required": True},
            "unit": {"type": "string", "description": "celsius or fahrenheit", "required": False},
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "amount": {"type": "float", "description": "Amount to convert", "required": True},
            "from_currency": {"type": "string", "description": "ISO code", "required": True},
            "to_currency": {"type": "string", "description": "ISO code", "required": True},
        },
    },
]


@st.cache_resource
def get_caller(model_name: str):
    from src.inference import OllamaToolCaller

    return OllamaToolCaller(cfg, model_name=model_name)


st.title("🛠️ Qwen2.5-3B Function-Calling Assistant")
st.caption("Fine-tuned with Unsloth QLoRA on the real xLAM-60k dataset · served as GGUF via Ollama")

tab_play, tab_metrics = st.tabs(["🚀 Playground", "📊 Metrics"])


with tab_play:
    with st.sidebar:
        st.header("Settings")
        model_name = st.text_input("Ollama model", value="qwen-tools")
        st.markdown("Define the tools the model is allowed to call (xLAM JSON schema):")

    col_in, col_out = st.columns(2)
    with col_in:
        query = st.text_area(
            "User request",
            value="What's the weather in Paris in celsius, and convert 100 USD to EUR?",
            height=120,
        )
        tools_text = st.text_area(
            "Available tools (JSON)", value=json.dumps(DEFAULT_TOOLS, indent=2), height=320
        )
        go = st.button("🚀 Generate tool call", type="primary", use_container_width=True)

    with col_out:
        if go:
            try:
                tools = json.loads(tools_text)
            except json.JSONDecodeError as e:
                st.error(f"Tools JSON is invalid: {e}")
                st.stop()
            try:
                caller = get_caller(model_name)
                start = time.time()
                result = caller(query, tools)
                elapsed = time.time() - start
            except Exception as e:  # noqa: BLE001
                st.error(f"Inference failed: {e}")
                st.info("Is Ollama running and the model created?\n\n"
                        "`ollama create qwen-tools -f deploy/Modelfile`")
                st.stop()

            st.metric("Latency", f"{elapsed:.2f}s")
            if result["tool_calls"]:
                st.success(f"Parsed {len(result['tool_calls'])} tool call(s)")
                st.json(result["tool_calls"])
            else:
                st.warning("No valid tool call parsed from the output.")
            with st.expander("Raw model output"):
                st.code(result["raw"])
        else:
            st.info("Enter a request and click generate.")

with tab_metrics:
    metrics_dir = cfg.path("metrics_dir")
    figures_dir = cfg.path("figures_dir")

    summaries = {}
    for label in ("base", "finetuned"):
        p = metrics_dir / f"eval_{label}.json"
        if p.exists():
            summaries[label] = json.loads(p.read_text())["summary"]

    if not summaries:
        st.info("No evaluation results yet. Run `scripts/run_eval.py` "
                "(or notebook 03) to populate this tab.")
    else:
        st.subheader("Measured test-set metrics")
        cols = st.columns(len(summaries))
        for col, (label, s) in zip(cols, summaries.items()):
            with col:
                st.markdown(f"**{label}**")
                st.metric("Exact match", f"{s['exact_match']:.1%}")
                st.metric("Function-name accuracy", f"{s['name_accuracy']:.1%}")
                st.metric("Argument F1", f"{s['argument_f1']:.1%}")
                st.metric("Hallucination rate", f"{s['hallucination_rate']:.1%}")

        for name in ("metric_comparison.png", "error_breakdown.png",
                     "quant_quality_size.png", "loss_curves.png"):
            fig = figures_dir / name
            if fig.exists():
                st.image(str(fig), caption=name)
