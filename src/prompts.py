from __future__ import annotations

import json
from typing import Any
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def format_tool_calls(answers: list[dict[str, Any]]) -> str:
    blocks = []
    for call in answers:
        payload = {"name": call["name"], "arguments": call.get("arguments", {})}
        blocks.append(
            "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
        )
    return "\n".join(blocks)


def build_prompt_text(tokenizer, query: str, tools: list[dict[str, Any]]) -> str:
    messages = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_full_text(
    tokenizer,
    query: str,
    tools: list[dict[str, Any]],
    answers: list[dict[str, Any]],
) -> str:
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": format_tool_calls(answers)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
