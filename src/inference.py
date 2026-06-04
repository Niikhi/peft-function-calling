from __future__ import annotations

from typing import Any

from .config import Config
from .data import to_openai_tool
from .parsing import parse_tool_calls
from .prompts import build_prompt_text


class HFToolCaller:

    def __init__(self, cfg: Config, model_dir: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = model_dir or str(cfg.path("merged_dir"))
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        self.model.eval()

    def __call__(self, query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        oa_tools = [to_openai_tool(t) for t in tools]
        prompt = build_prompt_text(self.tokenizer, query, oa_tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.evaluation["max_new_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            gen[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return {"raw": text, "tool_calls": parse_tool_calls(text)}


class OllamaToolCaller:
    def __init__(self, cfg: Config, model_name: str = "qwen-tools"):
        from ollama import Client
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.model_name = model_name
        self.client = Client()
        # Tokenizer only needed to render the chat template / tool block.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model["hf_id"], trust_remote_code=True)

    def __call__(self, query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        oa_tools = [to_openai_tool(t) for t in tools]
        prompt = build_prompt_text(self.tokenizer, query, oa_tools)
        resp = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            raw=True,
            options={"temperature": 0.0, "num_predict": self.cfg.evaluation["max_new_tokens"]},
        )
        text = resp.get("response", "")
        return {"raw": text, "tool_calls": parse_tool_calls(text)}
