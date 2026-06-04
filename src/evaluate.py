from __future__ import annotations

import json
import time
from typing import Callable

from .config import Config
from .data import available_tool_names, to_openai_tool
from .metrics import aggregate, score_example
from .parsing import parse_tool_calls
from .prompts import build_prompt_text

GenerateFn = Callable[[list[str]], list[str]]

def make_hf_generate_fn(cfg: Config, model, tokenizer) -> GenerateFn:
    import torch

    ev = cfg.evaluation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        bs = ev["batch_size"]
        for i in range(0, len(prompts), bs):
            batch = prompts[i : i + bs]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=cfg.model["max_seq_length"],
            ).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=ev["max_new_tokens"],
                    do_sample=ev["temperature"] > 0,
                    temperature=max(ev["temperature"], 1e-5),
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = gen[:, inputs["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    return generate


def make_ollama_generate_fn(cfg: Config, model_name: str) -> GenerateFn:
    from ollama import Client

    client = Client()
    ev = cfg.evaluation

    def generate(prompts: list[str]) -> list[str]:
        outs = []
        for p in prompts:
            resp = client.generate(
                model=model_name,
                prompt=p,
                raw=True,
                options={
                    "temperature": ev["temperature"],
                    "num_predict": ev["max_new_tokens"],
                },
            )
            outs_text = resp.get("response", "")
            outs.append(outs_text)
        return outs

    return generate

def run_eval(
    cfg: Config,
    records: list[dict],
    tokenizer,
    generate_fn: GenerateFn,
    label: str,
) -> dict:
    limit = cfg.evaluation.get("num_test_samples")
    if limit:
        records = records[: int(limit)]

    prompts = [
        build_prompt_text(tokenizer, r["query"], [to_openai_tool(t) for t in r["tools"]])
        for r in records
    ]

    start = time.time()
    completions = generate_fn(prompts)
    elapsed = time.time() - start

    per_example = []
    for rec, comp in zip(records, completions):
        pred = parse_tool_calls(comp)
        gold = rec["answers"]
        avail = available_tool_names(rec)
        score = score_example(pred, gold, avail)
        per_example.append({**score, "raw_output": comp[:2000]})

    metrics = aggregate(per_example)
    metrics["label"] = label
    metrics["seconds_total"] = elapsed
    metrics["seconds_per_example"] = elapsed / max(len(records), 1)

    metrics_path = cfg.file_in("metrics_dir", f"eval_{label}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"summary": metrics, "examples": per_example}, f, indent=2)
    print(f"[eval:{label}] exact_match={metrics['exact_match']:.3f} "
          f"name_acc={metrics['name_accuracy']:.3f} "
          f"arg_f1={metrics['argument_f1']:.3f}  -> {metrics_path}")
    return metrics
