from __future__ import annotations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import HEADLINE_METRICS

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, out_path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")
    return str(out_path)

def plot_loss_curves(log_history_path, out_path) -> str:
    with open(log_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train = [(h["step"], h["loss"]) for h in history if "loss" in h]
    evals = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

    fig, ax = plt.subplots(figsize=(10, 6))
    if train:
        xs, ys = zip(*train)
        ax.plot(xs, ys, label="train loss", color="#3b528b", linewidth=2)
    if evals:
        xs, ys = zip(*evals)
        ax.plot(xs, ys, label="eval loss", color="#5ec962", marker="o", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training & validation loss")
    ax.legend()
    return _save(fig, out_path)


def plot_lr_schedule(log_history_path, out_path) -> str:
    with open(log_history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    lrs = [(h["step"], h["learning_rate"]) for h in history if "learning_rate" in h]
    fig, ax = plt.subplots(figsize=(10, 5))
    if lrs:
        xs, ys = zip(*lrs)
        ax.plot(xs, ys, color="#fde725", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning-rate schedule")
    return _save(fig, out_path)

def plot_dataset_eda(records: list[dict], out_path) -> str:
    tools_per_query = [len(r["tools"]) for r in records]
    calls_per_answer = [len(r["answers"]) for r in records]
    query_lengths = [len(r["query"].split()) for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.histplot(tools_per_query, bins=20, ax=axes[0], color="#3b528b")
    axes[0].set_title("Tools offered per query")
    axes[0].set_xlabel("# tools")

    sns.histplot(calls_per_answer, bins=range(1, max(calls_per_answer) + 2),
                 ax=axes[1], color="#21918c")
    axes[1].set_title("Function calls per answer")
    axes[1].set_xlabel("# calls")

    sns.histplot(query_lengths, bins=30, ax=axes[2], color="#5ec962")
    axes[2].set_title("Query length")
    axes[2].set_xlabel("# words")
    return _save(fig, out_path)

def plot_metric_comparison(metrics_by_model: dict[str, dict], out_path) -> str:
    models = list(metrics_by_model.keys())
    n_groups = len(HEADLINE_METRICS)
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, model in enumerate(models):
        vals = [metrics_by_model[model].get(m, 0.0) for m in HEADLINE_METRICS]
        xs = [j + i * width for j in range(n_groups)]
        bars = ax.bar(xs, vals, width=width, label=model)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)

    ax.set_xticks([j + width * (len(models) - 1) / 2 for j in range(n_groups)])
    ax.set_xticklabels([m.replace("_", "\n") for m in HEADLINE_METRICS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Tool-calling metrics: model comparison")
    ax.legend()
    return _save(fig, out_path)


def plot_error_breakdown(metrics_by_model: dict[str, dict], out_path) -> str:
    categories = ["correct", "wrong_arguments", "wrong_function",
                  "hallucinated_function", "invalid_output"]
    colors = ["#5ec962", "#fde725", "#f98e09", "#bc3754", "#57106e"]
    models = list(metrics_by_model.keys())

    fig, ax = plt.subplots(figsize=(11, 7))
    bottoms = [0.0] * len(models)
    for cat, color in zip(categories, colors):
        vals = []
        for m in models:
            dist = metrics_by_model[m].get("error_distribution", {})
            total = sum(dist.values()) or 1
            vals.append(dist.get(cat, 0) / total)
        ax.bar(models, vals, bottom=bottoms, label=cat, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_ylabel("fraction of examples")
    ax.set_title("Error-type breakdown")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, out_path)

def plot_quant_quality_size(quant_rows: list[dict], out_path) -> str:
    labels = [r["label"] for r in quant_rows]
    sizes = [r.get("size_mb", 0) for r in quant_rows]
    quality = [r.get("exact_match", 0) for r in quant_rows]

    fig, ax1 = plt.subplots(figsize=(11, 7))
    bars = ax1.bar(labels, sizes, color="#cbe11e", alpha=0.7, label="size (MB)")
    ax1.set_ylabel("file size (MB)")
    for b, s in zip(bars, sizes):
        ax1.text(b.get_x() + b.get_width() / 2, s, f"{s:.0f}", ha="center", va="bottom")

    ax2 = ax1.twinx()
    ax2.plot(labels, quality, color="#3b528b", marker="o", linewidth=3, label="exact match")
    ax2.set_ylabel("exact match")
    ax2.set_ylim(0, 1.05)
    for x, q in zip(labels, quality):
        ax2.text(x, q + 0.02, f"{q:.2f}", ha="center", color="#3b528b")

    ax1.set_title("Quantization: quality vs size")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
    return _save(fig, out_path)


def plot_latency(quant_rows: list[dict], out_path) -> str:
    labels = [r["label"] for r in quant_rows]
    latency = [r.get("seconds_per_example", 0) for r in quant_rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, latency, color="#21918c")
    for b, v in zip(bars, latency):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}s", ha="center", va="bottom")
    ax.set_ylabel("seconds / example")
    ax.set_title("Inference latency by quantization")
    return _save(fig, out_path)
