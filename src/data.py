from __future__ import annotations

import json
import random
from typing import Any

from .config import Config
from .prompts import build_full_text

_TYPE_MAP = {
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "double": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "str": "string",
    "string": "string",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


def _json_type(raw_type: str) -> str:
    base = (raw_type or "string").split("[")[0].split(",")[0].strip().lower()
    return _TYPE_MAP.get(base, "string")


def to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, spec in (tool.get("parameters") or {}).items():
        if not isinstance(spec, dict):
            properties[pname] = {"type": "string"}
            continue
        prop = {"type": _json_type(spec.get("type", "string"))}
        if spec.get("description"):
            prop["description"] = spec["description"]
        if "default" in spec and spec["default"] not in (None, ""):
            prop["default"] = spec["default"]
        properties[pname] = prop
        if spec.get("required"):
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _normalize_primary(row: dict[str, Any]) -> dict[str, Any] | None:
    tools = _maybe_json(row.get("tools"))
    answers = _maybe_json(row.get("answers"))
    query = row.get("query")
    if not (query and isinstance(tools, list) and isinstance(answers, list)):
        return None
    return {"query": query, "tools": tools, "answers": answers}


def _normalize_mirror(row: dict[str, Any]) -> dict[str, Any] | None:
    tools = _maybe_json(row.get("tools"))
    messages = _maybe_json(row.get("messages")) or []
    query = next((m.get("content") for m in messages if m.get("role") == "user"), None)
    answer_raw = next(
        (m.get("content") for m in messages if m.get("role") == "assistant"), None
    )
    answers = _maybe_json(answer_raw)
    if not (query and isinstance(tools, list) and isinstance(answers, list)):
        return None
    return {"query": query, "tools": tools, "answers": answers}


def load_records(cfg: Config) -> list[dict[str, Any]]:
    from datasets import load_dataset

    primary = cfg.data["dataset_id"]
    mirror = cfg.data["fallback_dataset_id"]

    try:
        ds = load_dataset(primary, split="train")
        normalizer = _normalize_primary
        source = primary
    except Exception as err:  # gated / auth / network -> fall back
        print(f"[data] Could not load gated '{primary}' ({type(err).__name__}: {err}).")
        print(f"[data] Falling back to ungated mirror '{mirror}'.")
        ds = load_dataset(mirror, split="train")
        # The mirror may still carry the canonical columns; pick by presence.
        normalizer = _normalize_mirror if "messages" in ds.column_names else _normalize_primary
        source = mirror

    records: list[dict[str, Any]] = []
    for row in ds:
        rec = normalizer(row)
        if rec:
            records.append(rec)

    max_samples = cfg.data.get("max_samples")
    if max_samples:
        records = records[: int(max_samples)]

    print(f"[data] Loaded {len(records)} records from '{source}'.")
    return records

def split_records(
    cfg: Config, records: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(cfg.data["split_seed"])
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * cfg.data["train_frac"])
    n_val = int(n * cfg.data["val_frac"])

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    print(
        f"[data] Split -> train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )
    return splits


def save_jsonl(records: list[dict[str, Any]], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_text_dataset(records: list[dict[str, Any]], tokenizer):
    from datasets import Dataset

    texts = []
    for rec in records:
        tools = [to_openai_tool(t) for t in rec["tools"]]
        texts.append(build_full_text(tokenizer, rec["query"], tools, rec["answers"]))
    return Dataset.from_dict({"text": texts})


def available_tool_names(record: dict[str, Any]) -> set[str]:
    return {t["name"] for t in record["tools"]}
