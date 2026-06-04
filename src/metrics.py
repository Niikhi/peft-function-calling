from __future__ import annotations

import json
from collections import Counter
from typing import Any

Call = dict[str, Any]

def normalize_value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in v.items()}
    return v


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    return {k: normalize_value(v) for k, v in (args or {}).items()}


def _canonical(call: Call) -> str:
    """A stable, comparable signature for a full call (name + args)."""
    return json.dumps(
        {"name": call["name"], "arguments": normalize_args(call.get("arguments", {}))},
        sort_keys=True,
        ensure_ascii=False,
    )

def names_match(pred: list[Call], gold: list[Call]) -> bool:
    return Counter(c["name"] for c in pred) == Counter(c["name"] for c in gold)


def exact_match(pred: list[Call], gold: list[Call]) -> bool:
    return Counter(_canonical(c) for c in pred) == Counter(_canonical(c) for c in gold)


def _arg_counts_for_pair(pred: Call, gold: Call) -> tuple[int, int, int]:
    p = normalize_args(pred.get("arguments", {}))
    g = normalize_args(gold.get("arguments", {}))
    tp = sum(1 for k, gv in g.items() if k in p and p[k] == gv)
    fn = len(g) - tp
    fp = sum(1 for k in p if k not in g or p[k] != g[k])
    return tp, fp, fn


def argument_counts(pred: list[Call], gold: list[Call]) -> tuple[int, int, int]:
    tp = fp = fn = 0
    remaining = list(pred)
    for g in gold:
        match_idx = next((i for i, p in enumerate(remaining) if p["name"] == g["name"]), None)
        if match_idx is None:
            fn += len(normalize_args(g.get("arguments", {})))  # whole call missed
            continue
        p = remaining.pop(match_idx)
        dtp, dfp, dfn = _arg_counts_for_pair(p, g)
        tp += dtp
        fp += dfp
        fn += dfn
    for p in remaining:  # leftover predicted calls = pure false positives
        fp += len(normalize_args(p.get("arguments", {})))
    return tp, fp, fn


def classify_error(
    pred: list[Call], gold: list[Call], available_names: set[str]
) -> str:
    if not pred:
        return "invalid_output"
    if any(c["name"] not in available_names for c in pred):
        return "hallucinated_function"
    if not names_match(pred, gold):
        return "wrong_function"
    if not exact_match(pred, gold):
        return "wrong_arguments"
    return "correct"


def score_example(
    pred: list[Call], gold: list[Call], available_names: set[str]
) -> dict[str, Any]:
    tp, fp, fn = argument_counts(pred, gold)
    return {
        "valid": bool(pred),
        "name_correct": names_match(pred, gold),
        "exact": exact_match(pred, gold),
        "count_correct": len(pred) == len(gold),
        "hallucinated": any(c["name"] not in available_names for c in pred),
        "arg_tp": tp,
        "arg_fp": fp,
        "arg_fn": fn,
        "n_gold": len(gold),
        "error_type": classify_error(pred, gold, available_names),
    }

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scores)
    if n == 0:
        return {}

    tp = sum(s["arg_tp"] for s in scores)
    fp = sum(s["arg_fp"] for s in scores)
    fn = sum(s["arg_fn"] for s in scores)
    arg_p = _safe_div(tp, tp + fp)
    arg_r = _safe_div(tp, tp + fn)

    error_dist = Counter(s["error_type"] for s in scores)

    single = [s for s in scores if s["n_gold"] <= 1]
    multi = [s for s in scores if s["n_gold"] > 1]

    return {
        "n_examples": n,
        "validity": _safe_div(sum(s["valid"] for s in scores), n),
        "name_accuracy": _safe_div(sum(s["name_correct"] for s in scores), n),
        "exact_match": _safe_div(sum(s["exact"] for s in scores), n),
        "count_accuracy": _safe_div(sum(s["count_correct"] for s in scores), n),
        "hallucination_rate": _safe_div(sum(s["hallucinated"] for s in scores), n),
        "argument_precision": arg_p,
        "argument_recall": arg_r,
        "argument_f1": _safe_div(2 * arg_p * arg_r, arg_p + arg_r),
        "exact_match_single_call": _safe_div(sum(s["exact"] for s in single), len(single)),
        "exact_match_multi_call": _safe_div(sum(s["exact"] for s in multi), len(multi)),
        "n_single_call": len(single),
        "n_multi_call": len(multi),
        "error_distribution": dict(error_dist),
    }


HEADLINE_METRICS = [
    "validity",
    "name_accuracy",
    "exact_match",
    "argument_f1",
    "count_accuracy",
]


if __name__ == "__main__":
    gold = [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}]
    avail = {"get_weather", "get_time"}

    perfect = [{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}]
    wrong_arg = [{"name": "get_weather", "arguments": {"city": "London", "unit": "c"}}]
    wrong_fn = [{"name": "get_time", "arguments": {"city": "Paris"}}]
    halluc = [{"name": "make_coffee", "arguments": {}}]
    empty: list = []

    for label, pred in [
        ("perfect", perfect),
        ("wrong_arg", wrong_arg),
        ("wrong_fn", wrong_fn),
        ("hallucination", halluc),
        ("empty", empty),
    ]:
        s = score_example(pred, gold, avail)
        print(f"{label:14s} -> error_type={s['error_type']:22s} exact={s['exact']}")

    agg = aggregate([score_example(p, gold, avail) for p in [perfect, wrong_arg, wrong_fn]])
    print("\nAggregate:", json.dumps(agg, indent=2))
