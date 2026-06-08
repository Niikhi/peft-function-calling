from __future__ import annotations

import json
from typing import Any

from .config import Config
from .data import to_openai_tool
from .parsing import parse_tool_calls


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
        from .prompts import build_prompt_text

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


def _build_prompt_no_tokenizer(query: str, tools: list[dict[str, Any]]) -> str:
    tools_json = json.dumps(tools, indent=2)
    return (
        f"<|im_start|>system\nYou are a function-calling assistant. "
        f"Given a user request and a set of available tools, respond ONLY with the "
        f"appropriate tool call(s) in the format:\n"
        f"<tool_call>\n{{\"name\": \"<function_name>\", \"arguments\": {{<args>}}}}\n</tool_call>\n\n"
        f"Available tools:\n{tools_json}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class OllamaToolCaller:
    def __init__(self, cfg: Config, model_name: str = "qwentools"):
        from ollama import Client

        self.cfg = cfg
        self.model_name = model_name
        self.client = Client()

    def __call__(self, query: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
        oa_tools = [to_openai_tool(t) for t in tools]
        prompt = _build_prompt_no_tokenizer(query, oa_tools)
        resp = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            raw=True,
            options={"temperature": 0.0, "num_predict": self.cfg.evaluation["max_new_tokens"]},
        )
        text = resp.get("response", "")
        return {"raw": text, "tool_calls": parse_tool_calls(text)}
