from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


def _coerce_call(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if not isinstance(args, dict):
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        else:
            args = {}
    return {"name": str(obj["name"]), "arguments": args}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    calls: list[dict[str, Any]] = []

    matches = _TOOL_CALL_RE.findall(text)
    if matches:
        for raw in matches:
            try:
                call = _coerce_call(json.loads(raw))
            except json.JSONDecodeError:
                continue
            if call:
                calls.append(call)
        if calls:
            return calls

    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, list):
            calls = [c for c in (_coerce_call(o) for o in parsed) if c]
        else:
            one = _coerce_call(parsed)
            calls = [one] if one else []
        if calls:
            return calls
    except json.JSONDecodeError:
        pass

    for raw in _JSON_OBJ_RE.findall(text):
        try:
            call = _coerce_call(json.loads(raw))
        except json.JSONDecodeError:
            continue
        if call:
            calls.append(call)

    return calls


def is_valid_tool_output(text: str) -> bool:
    return len(parse_tool_calls(text)) > 0
